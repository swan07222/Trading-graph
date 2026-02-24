# data/fetcher_history_flow_ops.py
import math
from datetime import datetime
from typing import Any

import numpy as np
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


def _median_tail_close(frame: pd.DataFrame, tail_rows: int = 240) -> float:
    """Median close of recent rows; 0.0 when unavailable."""
    if frame is None or frame.empty:
        return 0.0
    if "close" not in frame.columns:
        return 0.0
    closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
    closes = closes[closes > 0]
    if closes.empty:
        return 0.0
    return float(closes.tail(max(1, int(tail_rows))).median())


def _validate_ohlcv_frame(frame: pd.DataFrame, require_positive_volume: bool = True) -> bool:
    """Validate OHLCV DataFrame has valid data for caching.
    
    FIX #7 & #8: Ensures data quality before caching.
    
    Args:
        frame: DataFrame to validate
        require_positive_volume: Whether to require positive volume
        
    Returns:
        True if valid, False otherwise
    """
    if frame is None or frame.empty:
        return False
    
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(frame.columns):
        return False
    
    # Check for NaN values in required columns
    for col in required_cols:
        if frame[col].isna().all():
            return False
    
    # Check for Inf values
    for col in required_cols:
        if np.isinf(frame[col]).any():
            return False
    
    # Check OHLC relationships: high >= low, high >= open, high >= close, low <= open, low <= close
    if not (frame["high"] >= frame["low"]).all():
        return False
    if not (frame["high"] >= frame["open"]).all():
        return False
    if not (frame["high"] >= frame["close"]).all():
        return False
    if not (frame["low"] <= frame["open"]).all():
        return False
    if not (frame["low"] <= frame["close"]).all():
        return False
    
    # Check for positive prices
    for col in ["open", "high", "low", "close"]:
        if (frame[col] <= 0).any():
            return False
    
    # Check volume if required
    if require_positive_volume and "volume" in frame.columns:
        if (frame["volume"] < 0).any():
            return False
    
    return True

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
        # FIX #11: Use reasonable TTL for intraday data (5 minutes instead of 30 seconds)
        ttl = min(float(CONFIG.data.cache_ttl_hours), 5.0 / 60.0)

    cache_key = f"history:{key}:{interval}"
    stale_cached_df = pd.DataFrame()
    cache_is_stale_or_partial = False

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
            # FIX #1 & #5: Track if cache is partial/stale for later use
            if len(cached_df) >= min(count, 100):
                return cached_df.tail(count)
            if len(cached_df) < count:
                cache_is_stale_or_partial = True
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
    # FIX #2: Cache session history even if partial (useful for future requests)
    if (
        use_session_history
        and interval in _INTRADAY_INTERVALS
        and not session_df.empty
        and count <= 500
    ):
        # FIX #10: Wrap cache write in try-except
        try:
            return self._cache_tail(
                cache_key,
                session_df,
                count,
                interval=interval,
            )
        except Exception as exc:
            log.warning("Cache write failed for session history: %s", exc)
            return session_df.tail(count) if len(session_df) >= count else session_df

    if is_cn_equity and interval in _INTRADAY_INTERVALS:
        if force_exact_intraday:
            return self._get_history_cn_intraday_exact(
                inst, count, fetch_days, interval, cache_key, offline,
            )
        # FIX #12: Remove redundant TypeError retry - use consistent signature
        persist_intraday_db = bool(update_db) and (
            not bool(CONFIG.is_market_open())
        )
        return self._get_history_cn_intraday(
            inst, count, fetch_days, interval,
            cache_key, offline, session_df,
            persist_intraday_db=persist_intraday_db,
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

    iv = self._normalize_interval_token(interval)
    baseline = (
        baseline_df
        if isinstance(baseline_df, pd.DataFrame)
        else pd.DataFrame()
    )
    oq = self._intraday_frame_quality(online_df, iv)
    bq = self._intraday_frame_quality(baseline, iv)
    online_score   = float(oq.get("score", 0.0))
    base_score     = float(bq.get("score", 0.0))
    online_suspect = bool(oq.get("suspect", False))

    online_med = _median_tail_close(online_df, tail_rows=240)
    ref_close = 0.0
    try:
        daily = self._db.get_bars(str(symbol or ""), limit=1)
        if isinstance(daily, pd.DataFrame) and not daily.empty:
            ref_close = _median_tail_close(daily, tail_rows=1)
    except Exception:
        ref_close = 0.0
    if ref_close <= 0:
        ref_close = _median_tail_close(baseline, tail_rows=120)
    if online_med > 0 and ref_close > 0:
        ratio = online_med / max(ref_close, 1e-8)
        min_ratio, max_ratio = 0.45, 2.2
        if ratio < min_ratio or ratio > max_ratio:
            log.warning(
                "Rejected scale-mismatched online snapshot for %s (%s): "
                "online_med=%.6f ref=%.6f ratio=%.3f (allowed %.3f..%.3f)",
                str(symbol or ""),
                iv,
                online_med,
                ref_close,
                ratio,
                min_ratio,
                max_ratio,
            )
            return False

    online_fresher = False
    try:
        if (
            isinstance(online_df.index, pd.DatetimeIndex)
            and isinstance(baseline.index, pd.DatetimeIndex)
            and len(online_df.index) > 0
            and len(baseline.index) > 0
        ):
            step = int(max(1, self._interval_seconds(iv)))
            online_last = pd.Timestamp(online_df.index.max())
            base_last   = pd.Timestamp(baseline.index.max())
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
            # Fallback: synthesize intraday from daily bars
            log.info(
                "Intraday data unavailable for %s (%s); synthesizing from daily",
                code6, interval,
            )
            # FIX #4: Use correct daily cache key
            daily_cache_key = f"history:{code6}:1d"
            daily_df = self._get_history_cn_daily(
                inst, count, fetch_days, daily_cache_key,
                offline, update_db=False, session_df=None, interval="1d",
            )
            if daily_df is not None and not daily_df.empty:
                # FIX #8: Validate synthesized data before caching
                # FIX #2026-02-24: Pass symbol for better logging
                synthesized = _synthesize_intraday_from_daily(
                    daily_df=daily_df,
                    interval=interval,
                    count=count,
                    symbol=code6,
                )
                if not synthesized.empty and self._validate_ohlcv_frame(synthesized):
                    # FIX #10: Wrap cache write in try-except
                    try:
                        return self._cache_tail(
                            cache_key,
                            synthesized,
                            count,
                            interval=interval,
                        )
                    except Exception as exc:
                        log.warning("Cache write failed for synthesized intraday: %s", exc)
                        return synthesized.tail(count)
            log.warning(
                "Intraday synthesis failed for %s (%s); returning empty",
                code6, interval,
            )
            return pd.DataFrame()
        # FIX #5: Mark partial cache with metadata if db_df has fewer rows than count
        if len(db_df) < count:
            log.info(
                "Intraday returning partial DB data for %s (%s): %d/%d bars",
                code6, interval, len(db_df), count,
            )
        # FIX #10: Wrap cache write in try-except
        try:
            return self._cache_tail(
                cache_key,
                db_df,
                count,
                interval=interval,
            )
        except Exception as exc:
            log.warning("Cache write failed for intraday DB data: %s", exc)
            return db_df.tail(count) if len(db_df) >= count else db_df

    # Prefer fresh online rows when timestamps overlap with local DB rows.
    merged = self._merge_parts(online_df, db_df, interval=interval)
    merged = self._filter_cn_intraday_session(merged, interval)
    if merged.empty:
        # Fallback: synthesize intraday from daily bars
        log.info(
            "Intraday merged empty for %s (%s); synthesizing from daily",
            code6, interval,
        )
        # FIX #4: Use correct daily cache key
        daily_cache_key = f"history:{code6}:1d"
        daily_df = self._get_history_cn_daily(
            inst, count, fetch_days, daily_cache_key,
            offline, update_db=False, session_df=None, interval="1d",
        )
        if daily_df is not None and not daily_df.empty:
            # FIX #8: Validate synthesized data before caching
            # FIX #2026-02-24: Pass symbol for better logging
            synthesized = _synthesize_intraday_from_daily(
                daily_df=daily_df,
                interval=interval,
                count=count,
                symbol=code6,
            )
            if not synthesized.empty and self._validate_ohlcv_frame(synthesized):
                # FIX #10: Wrap cache write in try-except
                try:
                    return self._cache_tail(
                        cache_key,
                        synthesized,
                        count,
                        interval=interval,
                    )
                except Exception as exc:
                    log.warning("Cache write failed for synthesized intraday: %s", exc)
                    return synthesized.tail(count)
        log.warning(
            "Intraday synthesis failed for %s (%s); returning empty",
            code6, interval,
        )
        return pd.DataFrame()

    # FIX #10: Wrap cache write in try-except
    try:
        out = self._cache_tail(
            cache_key,
            merged,
            count,
            interval=interval,
        )
    except Exception as exc:
        log.warning("Cache write failed for merged intraday: %s", exc)
        out = merged.tail(count)

    # FIX #9: Invalidate cache after DB update to prevent inconsistency
    if bool(persist_intraday_db):
        try:
            self._db.upsert_intraday_bars(code6, interval, online_df)
            # Re-cache with fresh DB data to ensure consistency
            try:
                db_fresh = self._clean_dataframe(
                    self._db.get_intraday_bars(
                        code6, interval=interval, limit=db_limit
                    ),
                    interval=interval,
                )
                db_fresh = self._filter_cn_intraday_session(db_fresh, interval)
                if not db_fresh.empty:
                    self._cache.set(cache_key, db_fresh)
            except Exception as cache_exc:
                log.debug("Cache refresh after DB update skipped: %s", cache_exc)
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
    if online_df is not None and (not online_df.empty):
        if not self._accept_online_intraday_snapshot(
            symbol=code6,
            interval=interval,
            online_df=online_df,
            baseline_df=db_df,
        ):
            online_df = pd.DataFrame()

    if online_df is None or online_df.empty:
        if db_df is None or db_df.empty:
            # Fallback: synthesize intraday from daily bars
            log.info(
                "Intraday exact: data unavailable for %s (%s); synthesizing from daily",
                code6, interval,
            )
            # FIX #4: Use correct daily cache key
            daily_cache_key = f"history:{code6}:1d"
            daily_df = self._get_history_cn_daily(
                inst, count, fetch_days, daily_cache_key,
                offline=True, update_db=False, session_df=None, interval="1d",
            )
            if daily_df is not None and not daily_df.empty:
                # FIX #8: Validate synthesized data before caching
                # FIX #2026-02-24: Pass symbol for better logging
                synthesized = _synthesize_intraday_from_daily(
                    daily_df=daily_df,
                    interval=interval,
                    count=count,
                    symbol=code6,
                )
                if not synthesized.empty and self._validate_ohlcv_frame(synthesized):
                    # FIX #10: Wrap cache write in try-except
                    try:
                        return self._cache_tail(
                            cache_key,
                            synthesized,
                            count,
                            interval=interval,
                        )
                    except Exception as exc:
                        log.warning("Cache write failed for synthesized intraday: %s", exc)
                        return synthesized.tail(count)
            log.warning(
                "Intraday exact synthesis failed for %s (%s); returning empty",
                code6, interval,
            )
            return pd.DataFrame()
        # FIX #10: Wrap cache write in try-except
        try:
            return self._cache_tail(
                cache_key,
                db_df,
                count,
                interval=interval,
            )
        except Exception as exc:
            log.warning("Cache write failed for intraday DB data: %s", exc)
            return db_df.tail(count) if len(db_df) >= count else db_df

    # Prefer fresh online rows when timestamps overlap with local DB rows.
    merged = self._merge_parts(online_df, db_df, interval=interval)
    merged = self._filter_cn_intraday_session(merged, interval)
    if merged.empty:
        # Fallback: synthesize intraday from daily bars
        log.info(
            "Intraday exact: merged empty for %s (%s); synthesizing from daily",
            code6, interval,
        )
        # FIX #4: Use correct daily cache key
        daily_cache_key = f"history:{code6}:1d"
        daily_df = self._get_history_cn_daily(
            inst, count, fetch_days, daily_cache_key,
            offline=True, update_db=False, session_df=None, interval="1d",
        )
        if daily_df is not None and not daily_df.empty:
            # FIX #8: Validate synthesized data before caching
            # FIX #2026-02-24: Pass symbol for better logging
            synthesized = _synthesize_intraday_from_daily(
                daily_df=daily_df,
                interval=interval,
                count=count,
                symbol=code6,
            )
            if not synthesized.empty and self._validate_ohlcv_frame(synthesized):
                # FIX #10: Wrap cache write in try-except
                try:
                    return self._cache_tail(
                        cache_key,
                        synthesized,
                        count,
                        interval=interval,
                    )
                except Exception as exc:
                    log.warning("Cache write failed for synthesized intraday: %s", exc)
                    return synthesized.tail(count)
        log.warning(
            "Intraday exact synthesis failed for %s (%s); returning empty",
            code6, interval,
        )
        return pd.DataFrame()

    # FIX #10: Wrap cache write in try-except
    try:
        out = self._cache_tail(
            cache_key,
            merged,
            count,
            interval=interval,
        )
    except Exception as exc:
        log.warning("Cache write failed for merged intraday: %s", exc)
        out = merged.tail(count)

    # FIX #9: Invalidate cache after DB update to prevent inconsistency
    try:
        self._db.upsert_intraday_bars(code6, interval, online_df)
        # Re-cache with fresh DB data to ensure consistency
        try:
            db_fresh = self._clean_dataframe(
                self._db.get_intraday_bars(
                    code6, interval=interval, limit=db_limit
                ),
                interval=interval,
            )
            db_fresh = self._filter_cn_intraday_session(db_fresh, interval)
            if not db_fresh.empty:
                self._cache.set(cache_key, db_fresh)
        except Exception as cache_exc:
            log.debug("Cache refresh after DB update skipped: %s", cache_exc)
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

    # FIX #10: Wrap cache write in try-except
    try:
        out = self._cache_tail(
            cache_key,
            merged,
            count,
            interval=iv,
        )
    except Exception as exc:
        log.warning("Cache write failed for CN daily: %s", exc)
        out = merged.tail(count)

    # FIX #9: Invalidate cache after DB update to prevent inconsistency
    if update_db and iv == "1d":
        if not self._history_quorum_allows_persist(
            interval=iv,
            symbol=str(inst.get("symbol", "")),
            meta=online_meta,
        ):
            return out
        try:
            self._db.upsert_bars(inst["symbol"], online_df)
            # Re-cache with fresh DB data to ensure consistency
            try:
                db_fresh = self._clean_dataframe(
                    self._db.get_bars(inst["symbol"], limit=db_limit),
                    interval="1d",
                )
                db_fresh_resampled = self._resample_daily_to_interval(db_fresh, iv)
                if not db_fresh_resampled.empty:
                    self._cache.set(cache_key, db_fresh_resampled)
            except Exception as cache_exc:
                log.debug("Cache refresh after DB update skipped: %s", cache_exc)
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
    symbol: str = "",
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
        symbol: Optional symbol for logging purposes

    Returns:
        Synthesized intraday DataFrame or empty DataFrame if not applicable
    
    FIX #2026-02-24:
    - Added symbol parameter for better logging
    - Timestamp column now uses consistent ISO format string for serialization
    - Datetime column removed to avoid pickle serialization issues
    - Added interval column for cache context verification
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
    daily_tail = daily_tail = daily_df.tail(max_days)

    if daily_tail.empty:
        return pd.DataFrame()

    intraday_rows: list[dict[str, Any]] = []
    intraday_index: list[pd.Timestamp] = []
    tz = getattr(daily_tail.index, "tz", None)
    step_minutes = int(iv.replace("m", "")) if iv.endswith("m") else 60

    # China A-share trading hours: 9:30-11:30 (morning), 13:00-15:00 (afternoon)
    # Morning session: 120 minutes (9:30-11:30)
    # Afternoon session: 120 minutes (13:00-15:00)
    # Total: 240 minutes of trading per day
    MORNING_SESSION_MINUTES = 120  # 9:30-11:30
    MORNING_START_HOUR = 9
    MORNING_START_MIN = 30
    AFTERNOON_START_HOUR = 13
    AFTERNOON_START_MIN = 0

    for date, row in daily_tail.iterrows():
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
            total_minutes = bar_idx * step_minutes

            # China trading hours: 9:30-11:30, 13:00-15:00
            if total_minutes < MORNING_SESSION_MINUTES:
                # Morning session (9:30-11:30)
                actual_hour = MORNING_START_HOUR
                actual_min = MORNING_START_MIN + total_minutes
                # Handle minute overflow past 60
                while actual_min >= 60:
                    actual_hour += 1
                    actual_min -= 60
            else:
                # Afternoon session (13:00-15:00)
                # Subtract morning session length to get afternoon offset
                afternoon_offset = total_minutes - MORNING_SESSION_MINUTES
                actual_hour = AFTERNOON_START_HOUR
                actual_min = AFTERNOON_START_MIN + afternoon_offset
                # Handle minute overflow past 60
                while actual_min >= 60:
                    actual_hour += 1
                    actual_min -= 60

            try:
                bar_time = date.replace(hour=actual_hour, minute=actual_min)
            except (ValueError, TypeError):
                bar_time = date + pd.Timedelta(hours=actual_hour, minutes=actual_min)
            ts = pd.Timestamp(bar_time)
            if tz is not None and ts.tzinfo is None:
                try:
                    ts = ts.tz_localize(tz)
                except (TypeError, ValueError):
                    pass

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

            # FIX #2026-02-24: Use consistent ISO format string for timestamp
            # This ensures reliable serialization across cache operations
            ts_iso = ts.isoformat()
            
            intraday_rows.append(
                {
                    "open": float(bar_open),
                    "high": float(bar_high),
                    "low": float(bar_low),
                    "close": float(bar_close),
                    "volume": float(vol_per_bar),
                    "amount": float(amt_per_bar),
                    "interval": iv,
                    "timestamp": ts_iso,
                    # FIX #2026-02-24: Store datetime as ISO string, not Timestamp object
                    # This prevents pickle serialization issues with timezone-aware timestamps
                    "datetime": ts_iso,
                }
            )
            intraday_index.append(ts)

    if not intraday_rows:
        return pd.DataFrame()

    result = pd.DataFrame(intraday_rows, index=pd.DatetimeIndex(intraday_index))
    result = result[~result.index.duplicated(keep="last")]
    result = result.tail(count)

    # FIX #8: Validate synthesized data before returning
    if not _validate_ohlcv_frame(result):
        log.warning(
            "Synthesized intraday data failed validation for %s bars from %d daily rows",
            interval, len(daily_tail),
        )
        return pd.DataFrame()

    log.debug(f"Synthesized {len(result)} {interval} bars from {len(daily_tail)} daily bars")
    return result
