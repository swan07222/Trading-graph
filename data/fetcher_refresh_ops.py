# data/fetcher_refresh_ops.py
import math
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pandas as pd

from config.settings import CONFIG
from data.fetcher_sources import BARS_PER_DAY, INTERVAL_MAX_DAYS, _is_offline
from data.session_cache import get_session_bar_cache
from utils.logger import get_logger

log = get_logger(__name__)

def refresh_trained_stock_history(
    self: Any,
    codes: list[str],
    *,
    interval: str = "1m",
    window_days: int = 29,
    allow_online: bool = True,
    sync_session_cache: bool = True,
    replace_realtime_after_close: bool = True,
    get_session_bar_cache_fn: Callable[[], Any] | None = None,
) -> dict[str, object]:
    """Refresh recent history for trained stocks from online providers and persist to DB.

    Incremental behavior:
    - Default anchor is the last saved official-history cache timestamp.
    - If no official-history cache anchor exists, use DB/cache latest timestamp.
    - After market close, if realtime rows exist and replacement is enabled,
      fetch from the first realtime timestamp and replace those realtime rows
      with official-history rows in session cache.
    """
    iv = self._normalize_interval_token(interval)
    wd = max(1, int(window_days or 29))
    intraday = iv not in {"1d", "1wk", "1mo"}
    bpd = float(BARS_PER_DAY.get(iv, 1.0) or 1.0)
    if bpd <= 0:
        bpd = 1.0

    target_bars = int(max(1, math.ceil(float(wd) * bpd)))
    db_limit = int(max(target_bars * 2, target_bars + 800))
    max_api_days = int(max(1, INTERVAL_MAX_DAYS.get(iv, wd)))
    now = self._now_shanghai_naive()
    window_start = pd.Timestamp(now - timedelta(days=wd))
    market_open = bool(CONFIG.is_market_open())
    do_sync_cache = bool(sync_session_cache)
    session_cache_getter = (
        get_session_bar_cache
        if get_session_bar_cache_fn is None
        else get_session_bar_cache_fn
    )

    session_cache = None
    if do_sync_cache:
        try:
            session_cache = session_cache_getter()
        except Exception as exc:
            log.debug("Session cache unavailable for refresh: %s", exc)
            session_cache = None

    codes6 = list(
        dict.fromkeys(
            c for c in (self.clean_code(x) for x in list(codes or [])) if c
        )
    )
    report: dict[str, object] = {
        "interval": iv,
        "window_days": int(wd),
        "window_bars": int(target_bars),
        "total": int(len(codes6)),
        "completed": 0,
        "updated": 0,
        "cached": 0,
        "rows": {},
        "fetched_days": {},
        "purged_realtime_rows": {},
        "cache_markers": {},
        "replacement_anchor_used": {},
        "cache_sync_errors": {},
        "quorum_blocked": {},
        "status": {},
        "errors": {},
    }
    reconcile_queue = self._load_refresh_reconcile_queue()
    reconcile_dirty = False
    report["pending_reconcile_before"] = int(len(reconcile_queue))
    report["pending_reconcile_after"] = int(len(reconcile_queue))
    report["pending_reconcile_codes"] = sorted(list(reconcile_queue.keys()))

    for idx, code6 in enumerate(codes6, start=1):
        fetched = pd.DataFrame()
        fetched_meta: dict[str, object] = {}
        quorum_blocked = False
        fetched_days = int(wd)
        purged_rows = 0
        cache_sync_attempted = False
        cache_sync_errors: list[str] = []
        pending_key = self._refresh_reconcile_key(code6, iv)
        had_pending = bool(pending_key and pending_key in reconcile_queue)
        try:
            if intraday:
                db_df = self._clean_dataframe(
                    self._db.get_intraday_bars(
                        code6, interval=iv, limit=db_limit
                    ),
                    interval=iv,
                )
                db_df = self._filter_cn_intraday_session(db_df, iv)
            else:
                db_df = self._clean_dataframe(
                    self._db.get_bars(code6, limit=db_limit),
                    interval="1d",
                )
                db_df = self._resample_daily_to_interval(db_df, iv)

            if (
                not db_df.empty
                and isinstance(db_df.index, pd.DatetimeIndex)
            ):
                db_recent = db_df.loc[db_df.index >= window_start]
            else:
                db_recent = db_df

            first_rt_ts = None
            first_rt_after_ak_ts = None
            last_ak_ts = None
            last_cache_ts = None
            purge_required = False
            purge_attempted = False
            if intraday and session_cache is not None:
                try:
                    markers = session_cache.describe_symbol_interval(code6, iv)
                except Exception as exc:
                    log.debug(
                        "Session cache marker read failed for %s (%s): %s",
                        code6, iv, exc,
                    )
                    markers = {}
                first_rt_ts = markers.get("first_realtime_ts")
                first_rt_after_ak_ts = markers.get("first_realtime_after_akshare_ts")
                last_ak_ts = markers.get("last_akshare_ts")
                last_cache_ts = markers.get("last_ts")
                report_markers = dict(report.get("cache_markers") or {})
                report_markers[code6] = {
                    "first_realtime_ts": (
                        pd.Timestamp(first_rt_ts).isoformat()
                        if first_rt_ts is not None
                        else None
                    ),
                    "last_akshare_ts": (
                        pd.Timestamp(last_ak_ts).isoformat()
                        if last_ak_ts is not None
                        else None
                    ),
                    "last_cache_ts": (
                        pd.Timestamp(last_cache_ts).isoformat()
                        if last_cache_ts is not None
                        else None
                    ),
                    "first_realtime_after_akshare_ts": (
                        pd.Timestamp(first_rt_after_ak_ts).isoformat()
                        if first_rt_after_ak_ts is not None
                        else None
                    ),
                }
                report["cache_markers"] = report_markers

            purge_required = bool(
                intraday
                and bool(replace_realtime_after_close)
                and (first_rt_after_ak_ts is not None)
            )

            replace_realtime = bool(
                intraday
                and (session_cache is not None)
                and (not market_open)
                and bool(replace_realtime_after_close)
                and (first_rt_after_ak_ts is not None)
            )

            anchor_ts = None
            if replace_realtime:
                anchor_ts = pd.Timestamp(first_rt_after_ak_ts)
            elif last_ak_ts is not None:
                anchor_ts = pd.Timestamp(last_ak_ts)
            elif last_cache_ts is not None:
                anchor_ts = pd.Timestamp(last_cache_ts)
            elif (
                not db_recent.empty
                and isinstance(db_recent.index, pd.DatetimeIndex)
            ):
                anchor_ts = pd.Timestamp(db_recent.index.max())
            else:
                anchor_ts = pd.Timestamp(window_start)

            if anchor_ts is not None:
                try:
                    if anchor_ts.tzinfo is not None:
                        anchor_ts = anchor_ts.tz_localize(None)
                except Exception as exc:
                    log.debug("Suppressed exception in data/fetcher.py", exc_info=exc)

            if (
                anchor_ts is not None
                and (not replace_realtime)
                and (anchor_ts < window_start)
            ):
                anchor_ts = pd.Timestamp(window_start)

            fetched_days = int(wd)
            if anchor_ts is not None:
                try:
                    gap_seconds = float(
                        max(
                            0.0,
                            (
                                now - pd.Timestamp(anchor_ts).to_pydatetime()
                            ).total_seconds(),
                        )
                    )
                    step_seconds = float(max(60, self._interval_seconds(iv)))
                    if gap_seconds <= (step_seconds * 1.1):
                        fetched_days = 1
                    else:
                        fetched_days = int(
                            max(1, math.ceil(gap_seconds / 86400.0) + 1)
                        )
                except Exception:
                    fetched_days = int(wd)
            fetched_days = int(min(max(1, fetched_days), max_api_days))

            report_days = dict(report.get("fetched_days") or {})
            report_days[code6] = int(fetched_days)
            report["fetched_days"] = report_days

            if bool(allow_online) and (not _is_offline()) and fetched_days > 0:
                inst = {"market": "CN", "asset": "EQUITY", "symbol": code6}
                try:
                    fetched_out = self._fetch_from_sources_instrument(
                        inst=inst,
                        days=fetched_days,
                        interval=iv,
                        include_localdb=False,
                        return_meta=True,
                    )
                except TypeError:
                    fetched_out = self._fetch_from_sources_instrument(
                        inst=inst,
                        days=fetched_days,
                        interval=iv,
                        include_localdb=False,
                    )
                if (
                    isinstance(fetched_out, tuple)
                    and len(fetched_out) == 2
                    and isinstance(fetched_out[0], pd.DataFrame)
                ):
                    fetched = fetched_out[0]
                    if isinstance(fetched_out[1], dict):
                        fetched_meta = dict(fetched_out[1])
                else:
                    fetched = (
                        fetched_out
                        if isinstance(fetched_out, pd.DataFrame)
                        else pd.DataFrame()
                    )
                fetched = self._clean_dataframe(fetched, interval=iv)
                if intraday:
                    fetched = self._filter_cn_intraday_session(fetched, iv)

                if (
                    not fetched.empty
                    and isinstance(fetched.index, pd.DatetimeIndex)
                ):
                    lower_bound = window_start - pd.Timedelta(days=2)
                    if anchor_ts is not None:
                        lower_bound = min(
                            lower_bound,
                            pd.Timestamp(anchor_ts) - pd.Timedelta(days=1),
                        )
                    fetched = fetched.loc[fetched.index >= lower_bound]
                    fetched_meta["selected_rows"] = int(len(fetched))

                if not fetched.empty:
                    if intraday:
                        self._db.upsert_intraday_bars(code6, iv, fetched)
                    else:
                        if self._history_quorum_allows_persist(
                            interval=iv,
                            symbol=code6,
                            meta=fetched_meta,
                        ):
                            self._db.upsert_bars(code6, fetched)
                        else:
                            quorum_blocked = True
                            fetched = pd.DataFrame()

            cache_sync_frame = fetched
            if (
                intraday
                and cache_sync_frame.empty
                and had_pending
                and isinstance(db_recent, pd.DataFrame)
                and (not db_recent.empty)
                and isinstance(db_recent.index, pd.DatetimeIndex)
            ):
                cache_sync_frame = db_recent.copy()

            if bool(do_sync_cache):
                if session_cache is None:
                    if intraday and (not cache_sync_frame.empty):
                        cache_sync_errors.append("session_cache_unavailable")
                else:
                    if not cache_sync_frame.empty:
                        cache_sync_attempted = True
                        try:
                            session_cache.upsert_history_frame(
                                code6,
                                iv,
                                cache_sync_frame,
                                source="official_history",
                                is_final=True,
                            )
                        except Exception as exc:
                            msg = str(exc)
                            cache_sync_errors.append(f"upsert_failed:{msg}")
                            log.warning(
                                "Session cache upsert failed for %s (%s): %s",
                                code6,
                                iv,
                                msg,
                            )

                    if (
                        replace_realtime
                        and (anchor_ts is not None)
                        and (not cache_sync_frame.empty)
                    ):
                        cache_sync_attempted = True
                        purge_attempted = True
                        purge_anchor = pd.Timestamp(anchor_ts)
                        if isinstance(cache_sync_frame.index, pd.DatetimeIndex):
                            try:
                                fetched_min_ts = pd.Timestamp(cache_sync_frame.index.min())
                                if fetched_min_ts.tzinfo is not None:
                                    fetched_min_ts = fetched_min_ts.tz_localize(None)
                                if fetched_min_ts > purge_anchor:
                                    log.warning(
                                        (
                                            "Realtime replacement window limited for %s (%s): "
                                            "requested_anchor=%s, fetched_start=%s; "
                                            "preserving older realtime cache rows."
                                        ),
                                        code6,
                                        iv,
                                        purge_anchor.isoformat(),
                                        fetched_min_ts.isoformat(),
                                    )
                                    purge_anchor = fetched_min_ts
                            except Exception as exc:
                                log.debug("Suppressed exception in data/fetcher.py", exc_info=exc)

                        report_anchor = dict(report.get("replacement_anchor_used") or {})
                        report_anchor[code6] = str(purge_anchor.isoformat())
                        report["replacement_anchor_used"] = report_anchor

                        try:
                            purged_rows = int(
                                session_cache.purge_realtime_rows(
                                    code6,
                                    iv,
                                    since_ts=purge_anchor,
                                )
                            )
                        except Exception as exc:
                            msg = str(exc)
                            cache_sync_errors.append(f"purge_failed:{msg}")
                            log.debug(
                                "Session realtime purge failed for %s (%s): %s",
                                code6,
                                iv,
                                msg,
                            )
                            purged_rows = 0

            if cache_sync_errors:
                sync_msg = "; ".join(cache_sync_errors)
                report_sync = dict(report.get("cache_sync_errors") or {})
                report_sync[code6] = sync_msg
                report["cache_sync_errors"] = report_sync
                if intraday and bool(do_sync_cache) and (not cache_sync_frame.empty):
                    if self._mark_refresh_reconcile_pending(
                        reconcile_queue,
                        code6,
                        iv,
                        error_text=sync_msg,
                    ):
                        reconcile_dirty = True
            elif had_pending and cache_sync_attempted and (
                (not purge_required) or purge_attempted
            ):
                if self._clear_refresh_reconcile_pending(
                    reconcile_queue,
                    code6,
                    iv,
                ):
                    reconcile_dirty = True

            report_purged = dict(report.get("purged_realtime_rows") or {})
            report_purged[code6] = int(max(0, purged_rows))
            report["purged_realtime_rows"] = report_purged
            report_quorum = dict(report.get("quorum_blocked") or {})
            report_quorum[code6] = bool(quorum_blocked)
            report["quorum_blocked"] = report_quorum

            if intraday:
                db_after = self._clean_dataframe(
                    self._db.get_intraday_bars(
                        code6, interval=iv, limit=db_limit
                    ),
                    interval=iv,
                )
                db_after = self._filter_cn_intraday_session(db_after, iv)
            else:
                db_after = self._clean_dataframe(
                    self._db.get_bars(code6, limit=db_limit),
                    interval="1d",
                )
                db_after = self._resample_daily_to_interval(db_after, iv)

            if (
                not db_after.empty
                and isinstance(db_after.index, pd.DatetimeIndex)
            ):
                db_after = db_after.loc[db_after.index >= window_start]

            if (
                not db_after.empty
                and isinstance(db_after, pd.DataFrame)
            ):
                try:
                    from core.instruments import instrument_key

                    hist_key = instrument_key(
                        {
                            "market": "CN",
                            "asset": "EQUITY",
                            "symbol": code6,
                        }
                    )
                    cache_key = f"history:{hist_key}:{iv}"
                    keep_rows = min(
                        len(db_after),
                        self._history_cache_store_rows(iv, target_bars),
                    )
                    self._cache.set(
                        cache_key,
                        db_after.tail(max(1, int(keep_rows))).copy(),
                    )
                except Exception as exc:
                    log.debug(
                        "History cache refresh sync failed for %s (%s): %s",
                        code6,
                        iv,
                        exc,
                    )

            rows = int(len(db_after.tail(target_bars)))
            report_rows = dict(report.get("rows") or {})
            report_rows[code6] = rows
            report["rows"] = report_rows

            report_status = dict(report.get("status") or {})
            if not fetched.empty:
                report_status[code6] = "updated"
                report["updated"] = int(report.get("updated", 0)) + 1
            elif quorum_blocked:
                report_status[code6] = "quorum_blocked"
            elif rows > 0:
                report_status[code6] = "cached"
                report["cached"] = int(report.get("cached", 0)) + 1
            else:
                report_status[code6] = "empty"
            report["status"] = report_status

        except Exception as exc:
            report_status = dict(report.get("status") or {})
            report_status[code6] = "error"
            report["status"] = report_status
            report_errors = dict(report.get("errors") or {})
            report_errors[code6] = str(exc)
            report["errors"] = report_errors
            log.exception(
                "Trained-stock history refresh failed for %s (%s)",
                code6,
                iv,
            )
            if self._strict_errors:
                raise
        finally:
            report["completed"] = int(idx)

    if reconcile_dirty:
        self._save_refresh_reconcile_queue(reconcile_queue)
    report["pending_reconcile_after"] = int(len(reconcile_queue))
    report["pending_reconcile_codes"] = sorted(list(reconcile_queue.keys()))

    return report
