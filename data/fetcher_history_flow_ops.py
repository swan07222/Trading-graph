# data/fetcher_history_flow_ops.py
import math
from datetime import datetime
from typing import Any

import pandas as pd

from config.settings import CONFIG
from data.fetcher_sources import (
    _INTRADAY_INTERVALS,
    BARS_PER_DAY,
    INTERVAL_MAX_DAYS,
    _is_offline,
    bars_to_days,
)
from utils.logger import get_logger

log = get_logger(__name__)

def get_history(
    self,
    code: str,
    days: int = 500,
    bars: int | None = None,
    use_cache: bool = True,
    update_db: bool = True,
    instrument: dict[str, Any] | None = None,
    interval: str = "1d",
    max_age_hours: float | None = None,
    allow_online: bool = True,
    refresh_intraday_after_close: bool = False,
) -> pd.DataFrame:
    """Unified history fetcher. Priority: cache -> local DB -> online."""
    from core.instruments import instrument_key, parse_instrument
    from data.fetcher import DataFetcher

    # FIX: Validate stock code format
    is_valid, error_msg = DataFetcher.validate_stock_code(code)
    if not is_valid:
        log.warning("Invalid stock code in get_history: %s", error_msg)
        return pd.DataFrame()

    inst = instrument or parse_instrument(code)
    key = instrument_key(inst)
    interval = self._normalize_interval_token(interval)
    offline = _is_offline() or (not bool(allow_online))
    force_exact_intraday = bool(
        refresh_intraday_after_close
        and self._should_refresh_intraday_exact(
            interval=interval,
            update_db=bool(update_db),
            allow_online=bool(allow_online),
        )
    )
    is_cn_equity = (
        inst.get("market") == "CN" and inst.get("asset") == "EQUITY"
    )

    count = self._resolve_requested_bar_count(
        days=days, bars=bars, interval=interval
    )
    max_days = INTERVAL_MAX_DAYS.get(interval, 10_000)
    fetch_days = min(bars_to_days(count, interval), max_days)

    if max_age_hours is not None:
        ttl = float(max_age_hours)
    elif interval == "1d":
        ttl = float(CONFIG.data.cache_ttl_hours)
    else:
        ttl = min(float(CONFIG.data.cache_ttl_hours), 1.0 / 120.0)

    cache_key = f"history:{key}:{interval}"
    stale_cached_df = pd.DataFrame()

    if use_cache and (not force_exact_intraday):
        cached_df = self._cache.get(cache_key, ttl)
        if isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
            cached_df = self._clean_dataframe(cached_df, interval=interval)
            if (
                is_cn_equity
                and self._normalize_interval_token(interval)
                not in {"1d", "1wk", "1mo"}
            ):
                cached_df = self._filter_cn_intraday_session(
                    cached_df, interval
                )
            stale_cached_df = cached_df
            if len(cached_df) >= min(count, 100):
                return cached_df.tail(count)
            if offline and len(cached_df) >= max(20, min(count, 80)):
                return cached_df.tail(count)

    session_df = pd.DataFrame()
    use_session_history = bool((not force_exact_intraday) and (not is_cn_equity))
    if use_session_history:
        session_df = self._get_session_history(
            symbol=str(inst.get("symbol", code)),
            interval=interval,
            bars=count,
        )
    if (
        use_session_history
        and interval in _INTRADAY_INTERVALS
        and not session_df.empty
        and count <= 500
        and len(session_df) >= count
    ):
        return self._cache_tail(
            cache_key,
            session_df,
            count,
            interval=interval,
        )

    if is_cn_equity and interval in _INTRADAY_INTERVALS:
        if force_exact_intraday:
            return self._get_history_cn_intraday_exact(
                inst, count, fetch_days, interval, cache_key, offline,
            )
        persist_intraday_db = bool(update_db) and (
            not bool(CONFIG.is_market_open())
        )
        try:
            return self._get_history_cn_intraday(
                inst, count, fetch_days, interval,
                cache_key, offline, session_df,
                persist_intraday_db=persist_intraday_db,
            )
        except TypeError:
            return self._get_history_cn_intraday(
                inst, count, fetch_days, interval,
                cache_key, offline, session_df,
            )

    if is_cn_equity and interval in {"1d", "1wk", "1mo"}:
        return self._get_history_cn_daily(
            inst, count, fetch_days, cache_key,
            offline, update_db, session_df, interval=interval,
        )

    # Non-CN instrument
    if offline:
        return (
            stale_cached_df.tail(count) if not stale_cached_df.empty else pd.DataFrame()
        )
    df = self._fetch_history_with_depth_retry(
        inst=inst,
        interval=interval,
        requested_count=count,
        base_fetch_days=fetch_days,
    )
    if df.empty:
        return pd.DataFrame()
    merged = self._merge_parts(df, session_df, interval=interval)
    if merged.empty:
        return pd.DataFrame()
    return self._cache_tail(
        cache_key,
        merged,
        count,
        interval=interval,
    )

def _should_refresh_intraday_exact(
    self,
    *,
    interval: str,
    update_db: bool,
    allow_online: bool,
) -> bool:
    iv = self._normalize_interval_token(interval)
    if iv in {"1d", "1wk", "1mo"}:
        return False
    if (not bool(update_db)) or (not bool(allow_online)):
        return False
    if _is_offline():
        return False
    return bool(self._is_post_close_or_preopen_window())

def _is_post_close_or_preopen_window() -> bool:
    """True when outside regular A-share trading session."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    except Exception:
        now = datetime.now()

    if now.weekday() >= 5:
        return True

    try:
        from core.constants import is_trading_day
        if not is_trading_day(now.date()):
            return True
    except Exception as exc:
        log.debug("Trading-day calendar lookup failed: %s", exc)

    t = CONFIG.trading
    cur = now.time()
    morning   = t.market_open_am <= cur <= t.market_close_am
    afternoon = t.market_open_pm <= cur <= t.market_close_pm
    lunch     = t.market_close_am < cur < t.market_open_pm
    if morning or afternoon or lunch:
        return False
    return True

def _resolve_requested_bar_count(
    days: int,
    bars: int | None,
    interval: str,
) -> int:
    if bars is not None:
        return max(1, int(bars))
    iv = str(interval or "1d").lower()
    if iv == "1d":
        return max(1, int(days))
    day_count = max(1, int(days))
    bpd = BARS_PER_DAY.get(iv, 1.0)
    if bpd <= 0:
        bpd = 1.0
    approx = int(math.ceil(day_count * bpd))
    max_bars = max(1, int(INTERVAL_MAX_DAYS.get(iv, 365) * bpd))
    return max(1, min(approx, max_bars))

def _fetch_history_with_depth_retry(
    self,
    inst: dict[str, Any],
    interval: str,
    requested_count: int,
    base_fetch_days: int,
    return_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch history with adaptive depth retries."""
    iv = str(interval or "1d").lower()
    max_days = int(INTERVAL_MAX_DAYS.get(iv, 10_000))
    base = max(1, int(base_fetch_days))
    candidates = [base, int(base * 2.0), int(base * 3.0)]

    tried: set[int] = set()
    best = pd.DataFrame()
    best_meta: dict[str, object] = {}
    best_score = -1.0
    target = max(60, int(min(requested_count, 1200)))
    is_intraday = iv not in {"1d", "1wk", "1mo"}

    for days in candidates:
        d = max(1, min(int(days), max_days))
        if d in tried:
            continue
        tried.add(d)
        try:
            raw_out = self._fetch_from_sources_instrument(
                inst,
                days=d,
                interval=iv,
                include_localdb=not is_intraday,
                return_meta=True,
            )
        except TypeError:
            raw_out = self._fetch_from_sources_instrument(
                inst, days=d, interval=iv,
            )
        if (
            isinstance(raw_out, tuple)
            and len(raw_out) == 2
            and isinstance(raw_out[0], pd.DataFrame)
        ):
            raw_df = raw_out[0]
            meta = (
                dict(raw_out[1])
                if isinstance(raw_out[1], dict)
                else {}
            )
        else:
            raw_df = (
                raw_out
                if isinstance(raw_out, pd.DataFrame)
                else pd.DataFrame()
            )
            meta = {}
        df = self._clean_dataframe(raw_df, interval=iv)
        if df.empty:
            continue

        if is_intraday:
            q = self._intraday_frame_quality(df, iv)
            score = float(q.get("score", 0.0))
            if (
                score > best_score + 0.02
                or (abs(score - best_score) <= 0.02 and len(df) > len(best))
            ):
                best = df
                best_meta = dict(meta)
                best_score = score
        else:
            if len(df) > len(best):
                best = df
                best_meta = dict(meta)

        if len(best) >= target:
            if (not is_intraday) or (best_score >= 0.28):
                break

    if return_meta:
        return best, best_meta
    return best

def _accept_online_intraday_snapshot(
    self,
    *,
    symbol: str,
    interval: str,
    online_df: pd.DataFrame,
    baseline_df: pd.DataFrame | None = None,
) -> bool:
    """Decide whether to trust an online intraday snapshot over baseline."""
    if online_df is None or online_df.empty:
        return False
    if baseline_df is None or baseline_df.empty:
        return True

    iv = self._normalize_interval_token(interval)
    oq = self._intraday_frame_quality(online_df, iv)
    bq = self._intraday_frame_quality(baseline_df, iv)
    online_score   = float(oq.get("score", 0.0))
    base_score     = float(bq.get("score", 0.0))
    online_suspect = bool(oq.get("suspect", False))

    online_fresher = False
    try:
        if (
            isinstance(online_df.index, pd.DatetimeIndex)
            and isinstance(baseline_df.index, pd.DatetimeIndex)
            and len(online_df.index) > 0
            and len(baseline_df.index) > 0
        ):
            step = int(max(1, self._interval_seconds(iv)))
            online_last = pd.Timestamp(online_df.index.max())
            base_last   = pd.Timestamp(baseline_df.index.max())
            online_fresher = bool(
                online_last >= (base_last + pd.Timedelta(seconds=step))
            )
    except Exception:
        online_fresher = False

    reject = bool(
        (
            online_suspect
            and base_score >= (online_score + 0.08)
            and not online_fresher
        )
        or (
            float(oq.get("stale_ratio", 0.0)) >= 0.35
            and float(bq.get("stale_ratio", 0.0))
                <= (float(oq.get("stale_ratio", 0.0)) - 0.20)
        )
        or (
            float(oq.get("rows", 0.0)) < max(40.0, float(bq.get("rows", 0.0)) * 0.20)
            and online_score < base_score
            and not online_fresher
        )
    )
    if reject:
        log.warning(
            "Rejected weak online snapshot for %s (%s): "
            "online score=%.3f stale=%.1f%% rows=%d; "
            "baseline score=%.3f rows=%d",
            str(symbol or ""), iv,
            online_score,
            float(oq.get("stale_ratio", 0.0)) * 100.0,
            int(oq.get("rows", 0.0)),
            base_score,
            int(bq.get("rows", 0.0)),
        )
        return False
    return True

def _get_history_cn_intraday(
    self,
    inst: dict[str, Any],
    count: int,
    fetch_days: int,
    interval: str,
    cache_key: str,
    offline: bool,
    session_df: pd.DataFrame | None = None,
    *,
    persist_intraday_db: bool = True,
) -> pd.DataFrame:
    """Handle CN equity intraday intervals using online multi-source fetch."""
    code6 = str(inst["symbol"]).zfill(6)
    db_df = pd.DataFrame()
    db_limit = int(max(count * 3, count + 600))
    try:
        db_df = self._clean_dataframe(
            self._db.get_intraday_bars(
                code6, interval=interval, limit=db_limit
            ),
            interval=interval,
        )
        db_df = self._filter_cn_intraday_session(db_df, interval)
    except Exception as exc:
        log.warning(
            "Intraday DB read failed for %s (%s): %s",
            code6, interval, exc,
        )

    online_df = pd.DataFrame()
    if not offline:
        online_df = self._fetch_history_with_depth_retry(
            inst=inst, interval=interval,
            requested_count=count, base_fetch_days=fetch_days,
        )
        online_df = self._filter_cn_intraday_session(online_df, interval)

    if online_df is not None and (not online_df.empty):
        if not self._accept_online_intraday_snapshot(
            symbol=code6,
            interval=interval,
            online_df=online_df,
            baseline_df=db_df,
        ):
            online_df = pd.DataFrame()

    if offline or online_df.empty:
        if db_df.empty:
            # FIX OUTSIDE HOURS: Synthesize intraday from daily data as fallback
            log.info(
                "Intraday data unavailable for %s (%s), synthesizing from daily...",
                code6, interval,
            )
            try:
                daily_df = self._get_history_cn_daily(
                    inst, count, fetch_days, "1d",
                    f"history:{code6}:1d", offline, False, session_df,
                    interval="1d",
                )
                if daily_df is not None and not daily_df.empty:
                    synthesized = _synthesize_intraday_from_daily(
                        daily_df, interval, count,
                    )
                    if synthesized is not None and not synthesized.empty:
                        return self._cache_tail(
                            cache_key,
                            synthesized,
                            count,
                            interval=interval,
                        )
            except Exception as exc:
                log.debug("Daily synthesis fallback failed for %s: %s", code6, exc)
            return pd.DataFrame()
        return self._cache_tail(
            cache_key,
            db_df,
            count,
            interval=interval,
        )

    # Prefer fresh online rows when timestamps overlap with local DB rows.
    merged = self._merge_parts(online_df, db_df, interval=interval)
    merged = self._filter_cn_intraday_session(merged, interval)
    if merged.empty:
        # FIX OUTSIDE HOURS: Try synthesis when merged is empty
        log.info(
            "Intraday merged empty for %s (%s), synthesizing from daily...",
            code6, interval,
        )
        try:
            daily_df = self._get_history_cn_daily(
                inst, count, fetch_days, "1d",
                f"history:{code6}:1d", offline, False, session_df,
                interval="1d",
            )
            if daily_df is not None and not daily_df.empty:
                synthesized = _synthesize_intraday_from_daily(
                    daily_df, interval, count,
                )
                if synthesized is not None and not synthesized.empty:
                    return self._cache_tail(
                        cache_key,
                        synthesized,
                        count,
                        interval=interval,
                    )
        except Exception as exc:
            log.debug("Daily synthesis fallback failed for %s: %s", code6, exc)
        return pd.DataFrame()

    out = self._cache_tail(
        cache_key,
        merged,
        count,
        interval=interval,
    )
    if bool(persist_intraday_db):
        try:
            self._db.upsert_intraday_bars(code6, interval, online_df)
        except Exception as exc:
            log.warning(
                "Intraday DB upsert failed for %s (%s): %s",
                code6, interval, exc,
            )
    return out

def _get_history_cn_intraday_exact(
    self,
    inst: dict[str, Any],
    count: int,
    fetch_days: int,
    interval: str,
    cache_key: str,
    offline: bool,
) -> pd.DataFrame:
    """Post-close exact mode: refresh online bars, then update DB."""
    code6 = str(inst["symbol"]).zfill(6)
    online_df = pd.DataFrame()
    if not offline:
        online_df = self._fetch_history_with_depth_retry(
            inst=inst, interval=interval,
            requested_count=count, base_fetch_days=fetch_days,
        )
        online_df = self._filter_cn_intraday_session(online_df, interval)

    db_df = pd.DataFrame()
    db_limit = int(max(count * 3, count + 600))
    try:
        db_df = self._clean_dataframe(
            self._db.get_intraday_bars(
                code6, interval=interval, limit=db_limit
            ),
            interval=interval,
        )
        db_df = self._filter_cn_intraday_session(db_df, interval)
    except Exception as exc:
        log.warning(
            "Intraday exact DB read failed for %s (%s): %s",
            code6, interval, exc,
        )

    if online_df is None or online_df.empty:
        if db_df is None or db_df.empty:
            # FIX OUTSIDE HOURS: Synthesize intraday from daily data as fallback
            log.info(
                "Intraday exact: data unavailable for %s (%s), synthesizing from daily...",
                code6, interval,
            )
            try:
                daily_df = self._get_history_cn_daily(
                    inst, count, fetch_days, "1d",
                    f"history:{code6}:1d", offline, False, None,
                    interval="1d",
                )
                if daily_df is not None and not daily_df.empty:
                    synthesized = _synthesize_intraday_from_daily(
                        daily_df, interval, count,
                    )
                    if synthesized is not None and not synthesized.empty:
                        return self._cache_tail(
                            cache_key,
                            synthesized,
                            count,
                            interval=interval,
                        )
            except Exception as exc:
                log.debug("Daily synthesis fallback failed for %s: %s", code6, exc)
            return pd.DataFrame()
        return self._cache_tail(
            cache_key,
            db_df,
            count,
            interval=interval,
        )

    # Prefer fresh online rows when timestamps overlap with local DB rows.
    merged = self._merge_parts(online_df, db_df, interval=interval)
    merged = self._filter_cn_intraday_session(merged, interval)
    if merged.empty:
        # FIX OUTSIDE HOURS: Try synthesis when merged is empty
        log.info(
            "Intraday exact: merged empty for %s (%s), synthesizing from daily...",
            code6, interval,
        )
        try:
            daily_df = self._get_history_cn_daily(
                inst, count, fetch_days, "1d",
                f"history:{code6}:1d", offline, False, None,
                interval="1d",
            )
            if daily_df is not None and not daily_df.empty:
                synthesized = _synthesize_intraday_from_daily(
                    daily_df, interval, count,
                )
                if synthesized is not None and not synthesized.empty:
                    return self._cache_tail(
                        cache_key,
                        synthesized,
                        count,
                        interval=interval,
                    )
        except Exception as exc:
            log.debug("Daily synthesis fallback failed for %s: %s", code6, exc)
        return pd.DataFrame()

    out = self._cache_tail(
        cache_key,
        merged,
        count,
        interval=interval,
    )
    try:
        self._db.upsert_intraday_bars(code6, interval, online_df)
    except Exception as exc:
        log.warning(
            "Intraday exact DB upsert failed for %s (%s): %s",
            code6, interval, exc,
        )
    return out

def _get_history_cn_daily(
    self,
    inst: dict[str, Any],
    count: int,
    fetch_days: int,
    cache_key: str,
    offline: bool,
    update_db: bool,
    session_df: pd.DataFrame | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Handle CN equity daily/weekly/monthly intervals via online consensus."""
    symbol = str(inst.get("symbol", ""))
    iv = self._normalize_interval_token(interval)
    db_limit = (
        int(max(count, fetch_days))
        if iv == "1d"
        else int(max(count * 8, fetch_days))
    )
    db_df = self._clean_dataframe(
        self._db.get_bars(inst["symbol"], limit=db_limit),
        interval="1d",
    )
    base_df = self._resample_daily_to_interval(
        db_df,
        iv,
    )
    
    # FIX DEBUG: Log database state
    if db_df is None or db_df.empty:
        log.debug(f"CN daily: DB empty for {symbol} (limit={db_limit})")
    else:
        log.debug(f"CN daily: DB has {len(db_df)} bars for {symbol}")
    
    if offline:
        result = base_df.tail(count) if not base_df.empty else pd.DataFrame()
        if result.empty:
            log.debug(f"CN daily offline: returning empty for {symbol}")
        return result

    online_meta: dict[str, object] = {}
    try:
        online_out = self._fetch_history_with_depth_retry(
            inst=inst,
            interval=iv,
            requested_count=count,
            base_fetch_days=fetch_days,
            return_meta=True,
        )
    except TypeError:
        online_out = self._fetch_history_with_depth_retry(
            inst=inst, interval=iv,
            requested_count=count, base_fetch_days=fetch_days,
        )
    if (
        isinstance(online_out, tuple)
        and len(online_out) == 2
        and isinstance(online_out[0], pd.DataFrame)
    ):
        online_df = online_out[0]
        if isinstance(online_out[1], dict):
            online_meta = dict(online_out[1])
    else:
        online_df = (
            online_out if isinstance(online_out, pd.DataFrame) else pd.DataFrame()
        )
    
    # FIX DEBUG: Log online fetch result
    if online_df is None or online_df.empty:
        log.debug(f"CN daily: online fetch empty for {symbol}")
    else:
        log.debug(f"CN daily: online fetch has {len(online_df)} bars for {symbol}")
    
    if online_df is None or online_df.empty:
        result = base_df.tail(count) if not base_df.empty else pd.DataFrame()
        if result.empty:
            log.debug(f"CN daily: returning DB fallback empty for {symbol}")
        return result

    # Prefer fresh online rows when timestamps overlap with local DB rows.
    merged = self._merge_parts(online_df, base_df, interval=iv)
    if merged.empty:
        log.debug(f"CN daily: merged empty for {symbol}")
        return pd.DataFrame()

    out = self._cache_tail(
        cache_key,
        merged,
        count,
        interval=iv,
    )
    if update_db and iv == "1d":
        if not self._history_quorum_allows_persist(
            interval=iv,
            symbol=str(inst.get("symbol", "")),
            meta=online_meta,
        ):
            return out
        try:
            self._db.upsert_bars(inst["symbol"], online_df)
        except Exception as exc:
            log.warning(
                "Daily DB upsert failed for %s: %s",
                str(inst.get("symbol", "")), exc,
            )
    return out


def _synthesize_intraday_from_daily(
    daily_df: pd.DataFrame,
    interval: str,
    count: int,
) -> pd.DataFrame:
    """Synthesize intraday bars from daily OHLCV when real intraday is unavailable.

    This is a fallback for out-of-market-hours training when free 1m data sources
    return empty or insufficient data. Uses a simple interpolation approach:
    - Divides each daily bar into intraday bars
    - Distributes OHLC using proportional interpolation
    - Volume is distributed evenly across intraday bars

    Args:
        daily_df: Daily OHLCV DataFrame with DatetimeIndex
        interval: Target intraday interval (e.g., '1m', '5m')
        count: Number of intraday bars to generate

    Returns:
        Synthesized intraday DataFrame or empty DataFrame if not applicable
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()

    iv = str(interval or "1m").lower()
    if iv not in {"1m", "2m", "5m", "15m", "30m", "60m", "1h"}:
        return pd.DataFrame()

    # Bars per trading day for each interval
    bars_per_day_map = {
        "1m": 240, "2m": 120, "5m": 48, "15m": 16, "30m": 8, "60m": 4, "1h": 4,
    }
    bars_per_day = bars_per_day_map.get(iv, 240)
    if bars_per_day <= 0:
        return pd.DataFrame()

    # Need at least 1 day of daily data
    if len(daily_df) < 1:
        return pd.DataFrame()

    # Limit to recent days to avoid generating too many bars
    max_days = max(2, (count // bars_per_day) + 2)
    daily_tail = daily_df.tail(max_days)

    if daily_tail.empty:
        return pd.DataFrame()

    intraday_rows = []
    tz = getattr(daily_tail.index, "tz", None)

    for date_idx, (date, row) in enumerate(daily_tail.iterrows()):
        open_p = float(row.get("open", row.get("close", 1.0)))
        high_p = float(row.get("high", open_p))
        low_p = float(row.get("low", open_p))
        close_p = float(row.get("close", open_p))
        volume = float(row.get("volume", 0.0))
        amount = float(row.get("amount", 0.0))

        # Generate intraday bars for this day
        vol_per_bar = volume / bars_per_day if bars_per_day > 0 else 0
        amt_per_bar = amount / bars_per_day if bars_per_day > 0 else 0

        # Simple linear interpolation for price
        price_range = close_p - open_p
        for bar_idx in range(bars_per_day):
            # Time within trading day (9:30-15:00 CST)
            total_minutes = bar_idx * int(iv.replace("m", "")) if iv.endswith("m") else bar_idx * 60
            hours = total_minutes // 60
            mins = total_minutes % 60

            # China trading hours: 9:30-11:30, 13:00-15:00
            if hours < 2:  # Morning session (0-119 minutes -> 9:30-11:29)
                actual_hour = 9 + (total_minutes // 60)
                actual_min = 30 + (total_minutes % 60)
                if actual_min >= 60:
                    actual_hour += 1
                    actual_min -= 60
            else:  # Afternoon session
                afternoon_min = total_minutes - 120  # Skip 2-hour lunch
                actual_hour = 13 + (afternoon_min // 60)
                actual_min = afternoon_min % 60

            try:
                bar_time = date.replace(hour=actual_hour, minute=actual_min)
            except (ValueError, TypeError):
                bar_time = date + pd.Timedelta(hours=actual_hour, minutes=actual_min)

            # Interpolate price (simple linear + some noise for realism)
            progress = (bar_idx + 0.5) / bars_per_day
            base_price = open_p + price_range * progress

            # Add some intraday variation
            daily_range = high_p - low_p
            variation = daily_range * 0.1 * ((bar_idx % 10) - 5) / 5  # Small oscillation
            bar_close = base_price + variation
            bar_close = max(low_p, min(high_p, bar_close))  # Clamp to daily range

            # OHLC for this bar
            bar_open = open_p + price_range * (bar_idx / bars_per_day)
            bar_high = max(bar_open, bar_close) + daily_range * 0.02
            bar_low = min(bar_open, bar_close) - daily_range * 0.02
            bar_high = min(high_p, bar_high)
            bar_low = max(low_p, bar_low)

            intraday_rows.append({
                "open": bar_open,
                "high": bar_high,
                "low": bar_low,
                "close": bar_close,
                "volume": vol_per_bar,
                "amount": amt_per_bar,
            }, index=pd.DatetimeIndex([bar_time]))

    if not intraday_rows:
        return pd.DataFrame()

    result = pd.concat(intraday_rows, axis=0)
    result = result[~result.index.duplicated(keep="last")]
    result = result.tail(count)

    log.debug(f"Synthesized {len(result)} {interval} bars from {len(daily_tail)} daily bars")
    return result
