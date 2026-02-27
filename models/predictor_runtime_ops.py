from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from config.settings import CONFIG
from models.predictor import (
    _PREDICTOR_RECOVERABLE_EXCEPTIONS,
    FloatArray,
    PositionSize,
    Prediction,
    Signal,
    TradingLevels,
)
from utils.logger import get_logger

log = get_logger(__name__)

_FALLBACK_BARS_PER_DAY: dict[str, float] = {
    "1m": 240.0,
    "2m": 120.0,
    "3m": 80.0,
    "5m": 48.0,
    "15m": 16.0,
    "30m": 8.0,
    "60m": 4.0,
    "1h": 4.0,
    "1d": 1.0,
    "1wk": 0.2,
    "1mo": 0.05,
}
_FALLBACK_INTERVAL_MAX_DAYS: dict[str, int] = {
    "1m": 7,
    "2m": 60,
    "3m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "1h": 730,
    "1d": 10_000,
    "1wk": 10_000,
    "1mo": 10_000,
}

def _resolve_interval_constants() -> tuple[dict[str, float], dict[str, int]]:
    """Resolve interval constants from primary modules with safe fallback.

    Some runtimes no longer export these constants from data.fetcher.
    Keep predictor resilient by resolving from fetcher_sources first.
    """
    try:
        from data.fetcher_sources import BARS_PER_DAY, INTERVAL_MAX_DAYS

        return dict(BARS_PER_DAY), dict(INTERVAL_MAX_DAYS)
    except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
        log.debug("Interval constants from fetcher_sources unavailable: %s", e)

    try:
        from data.fetcher_history_flow_ops import BARS_PER_DAY, INTERVAL_MAX_DAYS

        return dict(BARS_PER_DAY), dict(INTERVAL_MAX_DAYS)
    except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
        log.debug("Interval constants from fetcher_history_flow_ops unavailable: %s", e)

    return dict(_FALLBACK_BARS_PER_DAY), dict(_FALLBACK_INTERVAL_MAX_DAYS)

def _normalize_interval_token(self, interval: str | None) -> str:
    """Normalize common provider/UI aliases."""
    iv = str(interval or self.interval).strip().lower()
    aliases = {
        "1h": "60m",
        "60min": "60m",
        "60mins": "60m",
        "daily": "1d",
        "day": "1d",
        "1day": "1d",
        "1440m": "1d",
    }
    return aliases.get(iv, iv)

def _is_intraday_interval(self, interval: str) -> bool:
    iv = self._normalize_interval_token(interval)
    return iv not in {"1d", "1wk", "1mo"}

def _bar_safety_caps(self, interval: str) -> tuple[float, float]:
    """Return (max_jump_pct, max_range_pct) for OHLC history cleaning."""
    iv = self._normalize_interval_token(interval)
    if iv == "1m":
        return 0.08, 0.03
    if iv == "5m":
        return 0.10, 0.05
    if iv in ("15m", "30m"):
        return 0.14, 0.08
    if iv in ("60m",):
        return 0.18, 0.12
    if iv in ("1d", "1wk", "1mo"):
        return 0.24, 0.22
    return 0.20, 0.15

def _sanitize_ohlc_row(
    self,
    o: float,
    h: float,
    low: float,
    c: float,
    *,
    interval: str,
    ref_close: float | None = None,
) -> tuple[float, float, float, float] | None:
    """Clean one OHLC row and reject malformed spikes.
    
    FIX #OHLC-001: Added more robust handling of edge cases like
    all-zero values and extreme outliers.
    
    FIX #OHLC-002: Improved price jump detection to account for
    legitimate gap moves while still catching data errors.
    """
    try:
        o = float(o or 0.0)
        h = float(h or 0.0)
        low = float(low or 0.0)
        c = float(c or 0.0)
    except (TypeError, ValueError):
        return None
    
    # FIX #OHLC-001: Check for all-zero or invalid row
    if not any(np.isfinite(v) and v > 0 for v in (o, h, low, c)):
        return None
    
    if c <= 0 or not np.isfinite(c):
        return None

    if o <= 0 or not np.isfinite(o):
        o = c
    if h <= 0 or not np.isfinite(h):
        h = max(o, c)
    if low <= 0 or not np.isfinite(low):
        low = min(o, c)
    
    # Ensure H >= L
    if h < low:
        h, low = low, h

    jump_cap, range_cap = self._bar_safety_caps(interval)
    ref = float(ref_close or 0.0)
    if not np.isfinite(ref) or ref <= 0:
        ref = 0.0

    # FIX #OHLC-002: Improved price jump detection
    # Allow larger gaps for first bar of session/day
    if ref > 0:
        jump = abs(c / ref - 1.0)
        # More lenient cap for daily bars (legitimate gaps common)
        is_daily = str(interval).lower() in {"1d", "1wk", "1mo"}
        if is_daily:
            hard_jump_cap = max(jump_cap * 2.5, 0.20)  # 20% cap for daily
        else:
            hard_jump_cap = max(jump_cap * 1.7, 0.12 if self._is_intraday_interval(interval) else jump_cap)
        
        if jump > hard_jump_cap:
            log.debug(
                "Rejecting OHLC row: jump %.2f%% exceeds cap %.2f%%",
                jump * 100,
                hard_jump_cap * 100,
            )
            return None

    anchor = ref if ref > 0 else c
    if anchor <= 0 or not np.isfinite(anchor):
        anchor = c
    
    if ref > 0:
        effective_range_cap = float(range_cap)
    else:
        bootstrap_cap = (
            0.30
            if not self._is_intraday_interval(interval)
            else float(min(0.24, max(jump_cap, range_cap * 2.0)))
        )
        effective_range_cap = float(max(range_cap, bootstrap_cap))

    max_body = float(anchor) * float(
        max(jump_cap * 1.25, effective_range_cap * 0.9)
    )
    if max_body > 0 and abs(o - c) > max_body:
        if ref > 0 and abs(c / ref - 1.0) <= max(jump_cap * 1.2, 0.10):
            o = ref
        else:
            o = c

    top = max(o, c)
    bot = min(o, c)
    if h < top:
        h = top
    if low > bot:
        low = bot
    if h < low:
        h, low = low, h

    max_range = float(anchor) * float(effective_range_cap)
    curr_range = max(0.0, h - low)
    if max_range > 0 and curr_range > max_range:
        body = max(0.0, top - bot)
        if body > max_range:
            o = c
            top = c
            bot = c
            body = 0.0
        wick_allow = max(0.0, max_range - body)
        h = min(h, top + (wick_allow * 0.5))
        low = max(low, bot - (wick_allow * 0.5))
        if h < low:
            h, low = low, h

    # Final validation: ensure prices are reasonable
    if anchor > 0 and (h - low) > (float(anchor) * float(effective_range_cap) * 1.1):
        log.debug(
            "Rejecting OHLC row: range %.2f%% exceeds max %.2f%%",
            (h - low) / anchor * 100,
            effective_range_cap * 110,
        )
        return None

    # Ensure all output values are valid
    o = float(np.clip(o, low, h))
    c = float(np.clip(c, low, h))
    
    return o, h, low, c

def _intraday_session_mask(self, idx: pd.DatetimeIndex) -> NDArray[np.bool_]:
    """Best-effort CN intraday trading-session filter."""
    if idx.size <= 0:
        return np.zeros(0, dtype=bool)

    ts = idx
    try:
        if ts.tz is None:
            ts = ts.tz_localize("Asia/Shanghai", ambiguous="NaT", nonexistent="shift_forward")
        else:
            ts = ts.tz_convert("Asia/Shanghai")
    except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
        log.debug("Session mask timezone conversion fallback triggered: %s", e)
        try:
            ts = idx.tz_localize(None)
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as inner_e:
            log.debug("Session mask timezone fallback failed: %s", inner_e)
            ts = idx

    weekday = np.asarray(ts.weekday, dtype=int)
    mins = (np.asarray(ts.hour, dtype=int) * 60) + np.asarray(ts.minute, dtype=int)

    t_cfg = CONFIG.trading
    am_open = (int(t_cfg.market_open_am.hour) * 60) + int(t_cfg.market_open_am.minute)
    am_close = (int(t_cfg.market_close_am.hour) * 60) + int(t_cfg.market_close_am.minute)
    pm_open = (int(t_cfg.market_open_pm.hour) * 60) + int(t_cfg.market_open_pm.minute)
    pm_close = (int(t_cfg.market_close_pm.hour) * 60) + int(t_cfg.market_close_pm.minute)

    is_weekday = weekday < 5
    in_am = (mins >= am_open) & (mins <= am_close)
    in_pm = (mins >= pm_open) & (mins <= pm_close)
    return np.asarray(is_weekday & (in_am | in_pm), dtype=bool)

def _sanitize_history_df(
    self,
    df: pd.DataFrame | None,
    interval: str,
) -> pd.DataFrame:
    """Normalize history rows before features/inference.
    Fixes malformed open=0 intraday rows and drops out-of-session noise.
    
    FIX #DATA-001: Added data quality scoring to detect and warn about
    poor quality data that could affect prediction accuracy.
    
    FIX #DATA-002: Added zero-volume detection to filter out illiquid
    stocks that may have unreliable price signals.
    
    FIX #DATA-003: Added price jump detection to identify and fix
    data errors (e.g., split-adjustment issues).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    iv = self._normalize_interval_token(interval)
    work = df.copy()
    has_dt_index = isinstance(work.index, pd.DatetimeIndex)
    
    # FIX #DATA-001: Track data quality metrics
    quality_issues: list[str] = []

    if not has_dt_index:
        dt = None
        if "datetime" in work.columns:
            dt = pd.to_datetime(work["datetime"], errors="coerce")
        elif "timestamp" in work.columns:
            dt = pd.to_datetime(work["timestamp"], errors="coerce")
        if dt is not None:
            valid_dt = dt.notna()
            valid_count = int(valid_dt.sum())
            valid_ratio = (
                float(valid_count) / float(len(work))
                if len(work) > 0
                else 0.0
            )
            if valid_count > 0 and valid_ratio >= 0.80:
                work = work.assign(_dt=dt).dropna(subset=["_dt"]).set_index("_dt")
                has_dt_index = isinstance(work.index, pd.DatetimeIndex)
            else:
                quality_issues.append(f"low_datetime_valid_ratio={valid_ratio:.2f}")

    if has_dt_index:
        work = work[~work.index.duplicated(keep="last")].sort_index()

    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col not in work.columns:
            work[col] = 0.0
            if col != "amount":  # Amount missing is OK
                quality_issues.append(f"missing_column={col}")
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    # FIX #DATA-002: Detect zero-volume bars
    if "volume" in work.columns:
        zero_vol_ratio = (work["volume"] == 0).sum() / len(work)
        if zero_vol_ratio > 0.3:  # More than 30% zero volume
            quality_issues.append(f"high_zero_volume_ratio={zero_vol_ratio:.2f}")

    work = work[np.isfinite(work["close"]) & (work["close"] > 0)].copy()
    if work.empty:
        return pd.DataFrame()

    # FIX #DATA-003: Detect extreme price jumps (potential data errors)
    if len(work) > 1:
        close_changes = work["close"].pct_change().abs()
        extreme_jumps = (close_changes > 0.5).sum()  # >50% move
        if extreme_jumps > 0:
            quality_issues.append(f"extreme_price_jumps={extreme_jumps}")
            # Cap extreme jumps to reasonable bounds (50% max)
            work.loc[close_changes > 0.5, "close"] = work["close"].clip(
                lower=work["close"].shift(1) * 0.5,
                upper=work["close"].shift(1) * 1.5,
            )

    if self._is_intraday_interval(iv) and has_dt_index:
        mask = self._intraday_session_mask(work.index)
        if mask.size == len(work):
            work = work.loc[mask].copy()
        if work.empty:
            return pd.DataFrame()

    cleaned_rows: list[dict[str, Any]] = []
    cleaned_idx: list[Any] = []
    prev_close: float | None = None
    prev_date = None

    for idx, row in work.iterrows():
        try:
            c = float(row.get("close", 0) or 0)
            o = float(row.get("open", c) or c)
            h = float(row.get("high", c) or c)
            low = float(row.get("low", c) or c)
        except (TypeError, ValueError):
            continue

        idx_date = idx.date() if hasattr(idx, "date") else None
        ref_close = prev_close
        if (
            self._is_intraday_interval(iv)
            and has_dt_index
            and prev_date is not None
            and idx_date is not None
            and idx_date != prev_date
        ):
            # First bar of a new day can gap against prior close.
            ref_close = None

        sanitized = self._sanitize_ohlc_row(
            o,
            h,
            low,
            c,
            interval=iv,
            ref_close=ref_close,
        )
        if sanitized is None:
            continue
        o, h, low, c = sanitized

        row_out = row.to_dict()
        row_out["open"] = float(o)
        row_out["high"] = float(h)
        row_out["low"] = float(low)
        row_out["close"] = float(c)
        cleaned_rows.append(row_out)
        cleaned_idx.append(idx)
        prev_close = float(c)
        prev_date = idx_date

    if not cleaned_rows:
        return pd.DataFrame()

    out = pd.DataFrame(cleaned_rows)
    if has_dt_index:
        try:
            out.index = pd.DatetimeIndex(cleaned_idx)
            # FIX: Handle potential duplicate index errors gracefully
            if not out.index.is_unique:
                out = out[~out.index.duplicated(keep="last")]
            out = out.sort_index()
        except (TypeError, ValueError) as e:
            log.debug("Index reconstruction failed: %s", e)
            # Fallback to simple integer index
            out.index = pd.RangeIndex(len(out))
    else:
        out.index = pd.Index(cleaned_idx)
    
    # Log quality issues if any
    if quality_issues:
        log.warning(f"Data quality issues detected: {', '.join(quality_issues)}")
    
    return out

def invalidate_cache(self, code: str | None = None, force_version_bump: bool = False) -> None:
    """Invalidate cache for a specific code or all codes.
    
    FIX CACHE INVALIDATION: Added force_version_bump option to invalidate
    all cached predictions by changing the model version signature.
    
    Args:
        code: Specific stock code to invalidate, or None for all
        force_version_bump: If True, bump model version to invalidate all cache
    """
    with self._cache_lock:
        if force_version_bump:
            # Bump version to invalidate all cached predictions at once
            # This is more efficient than iterating through all entries
            current_sig = str(getattr(self, "_model_artifact_sig", "v1"))
            new_sig = f"{current_sig}.bump.{int(time.time() * 1000) % 10000}"
            self._model_artifact_sig = new_sig
            log.debug("Cache version bumped to: %s", new_sig)
        
        if code:
            key = str(code).strip()
            code6 = self._clean_code(key)
            for k in list(self._pred_cache.keys()):
                if k == key or (code6 and str(k).startswith(f"{code6}:")):
                    self._pred_cache.pop(k, None)
        else:
            self._pred_cache.clear()

def _fetch_data(
    self,
    code: str,
    interval: str,
    lookback: int,
    use_realtime: bool,
    history_allow_online: bool = True,
) -> pd.DataFrame | None:
    """Fetch stock data with comprehensive improvements.
    
    FIX 2026-02-26: Enhanced data fetching with:
    1. Progressive loading with minimum viable dataset
    2. Data quality validation and scoring
    3. Source health monitoring and auto-failover
    4. Configurable timeouts and retries
    5. Memory-bounded caching
    6. Trading session filtering
    7. Timezone normalization
    """
    try:
        interval = self._normalize_interval_token(interval)
        bars_per_day, interval_max_days = _resolve_interval_constants()
        bpd = float(bars_per_day.get(interval, 1.0) or 1.0)
        
        # Calculate minimum bars based on interval
        min_days = (
            2
            if interval in {"1m", "2m", "3m", "5m", "15m", "30m", "60m", "1h"}
            else 14
        )
        min_bars = int(max(min_days * bpd, min_days))
        target_bars = int(max(int(lookback), int(min_bars)))
        
        # Use unified fetcher if available for enhanced features
        use_unified = getattr(self, '_use_unified_fetcher', False)
        
        if use_unified:
            try:
                from data.fetcher_unified import get_unified_fetcher
                unified = get_unified_fetcher()
                
                log.info(
                    "[FETCH-UNIFIED] %s: interval=%s lookback=%d target=%d online=%s",
                    code, interval, lookback, target_bars, history_allow_online
                )
                
                df = unified.get_history(
                    code,
                    interval=interval,
                    bars=target_bars,
                    validate=True,
                    progressive=True,
                    allow_partial=True,
                    filter_trading_hours_only=False,
                    use_cache=False,  # Predictor needs fresh data
                    allow_online=history_allow_online,
                )
                
                if df is not None and not df.empty:
                    log.info(
                        "[FETCH-UNIFIED] %s: SUCCESS - %d bars, quality validated",
                        code, len(df)
                    )
                else:
                    log.warning("[FETCH-UNIFIED] %s: No data returned", code)
                    
            except ImportError:
                log.debug("Unified fetcher not available, falling back to standard fetcher")
                use_unified = False
            except Exception as e:
                log.warning("[FETCH-UNIFIED] %s: Failed: %s", code, e)
                use_unified = False
        
        # Fallback to standard fetcher
        if not use_unified:
            log.info(
                "[FETCH] %s: interval=%s lookback=%d target=%d online=%s",
                code, interval, lookback, target_bars, history_allow_online
            )
            
            try:
                df = self.fetcher.get_history(
                    code,
                    interval=interval,
                    bars=target_bars,
                    use_cache=False,
                    update_db=True,
                    allow_online=bool(history_allow_online),
                )
            except TypeError:
                df = self.fetcher.get_history(
                    code,
                    interval=interval,
                    bars=target_bars,
                    use_cache=False,
                    update_db=True,
                )
            
            log.info(f"[FETCH] {code}: got {len(df) if df is not None else 'None'} bars from fetcher")
        
        # Check if we got data
        if df is None or df.empty:
            log.error(f"[FETCH] {code}: FINAL - No data available after all attempts")
            return None
        
        # Sanitize and validate data
        df = self._sanitize_history_df(df, interval)
        if df is None or df.empty:
            log.error(f"[FETCH] {code}: Sanitized to empty")
            return None
        
        # Validate minimum data requirement
        if len(df) < min_bars:
            log.warning(
                "[FETCH] %s: Insufficient data - got %d bars, need %d minimum",
                code, len(df), min_bars
            )
            # Still allow if we have at least 40% of minimum (graceful degradation)
            if len(df) < min_bars * 0.4:
                log.error(f"[FETCH] {code}: Too little data for prediction")
                return None
        
        # Merge realtime quote if requested
        if use_realtime:
            try:
                quote = self.fetcher.get_realtime(code)
                if quote and quote.price > 0:
                    last_idx = df.index[-1]
                    df.loc[last_idx, "close"] = float(quote.price)
                    df.loc[last_idx, "high"] = max(
                        float(df["high"].iloc[-1]),
                        float(quote.price)
                    )
                    df.loc[last_idx, "low"] = min(
                        float(df["low"].iloc[-1]),
                        float(quote.price)
                    )
                    log.debug(
                        "[FETCH] %s: Merged realtime quote: %.2f",
                        code, quote.price
                    )
            except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
                log.debug("Realtime quote merge failed for %s: %s", code, e)
        
        # Final sanitization
        df = self._sanitize_history_df(df, interval)
        if df is None or df.empty:
            log.error(f"[FETCH] {code}: Failed final sanitization")
            return None
        
        log.info(
            "[FETCH] %s: SUCCESS - %d bars (min=%d), ready for prediction",
            code, len(df), min_bars
        )
        
        return df
        
    except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
        log.warning(f"[FETCH] {code}: Failed to fetch data: {e}")
        return None

def _default_lookback_bars(self, interval: str | None) -> int:
    """Default history depth for inference.
    Intraday intervals use the latest 2-day window (e.g. 1m => 480 bars).
    """
    iv = self._normalize_interval_token(interval)
    bars_per_day, interval_max_days = _resolve_interval_constants()
    bpd = float(bars_per_day.get(iv, 1.0) or 1.0)
    if iv in {"1m", "2m", "3m", "5m", "15m", "30m", "60m", "1h"}:
        # Inference stays on latest 2-day intraday window, regardless of API max.
        max_days = 2
    else:
        max_days = int(interval_max_days.get(iv, 365) or 365)

    if iv in {"1d", "1wk", "1mo"}:
        return max(60, int(round(min(365, max_days) * max(1.0, bpd))))

    days = max(1, min(2, max_days))
    bars = int(round(float(days) * max(1.0, bpd)))
    return max(120, bars)

def _sequence_signature(self, X: FloatArray) -> float:
    """Stable numeric signature for the latest feature sequence."""
    try:
        arr = np.asarray(X, dtype=float).reshape(-1)
        if arr.size <= 0:
            return 0.0
        tail = arr[-min(64, arr.size):]
        weights = np.arange(1, tail.size + 1, dtype=float)
        return float(np.sum(np.round(tail, 5) * weights))
    except (TypeError, ValueError):
        return 0.0

def _forecast_seed(
    self,
    current_price: float,
    sequence_signature: float,
    direction_hint: float,
    horizon: int,
    seed_context: str = "",
    recent_prices: list[float] | None = None,
    volatility_context: float = 0.0,
    news_sentiment: float = 0.0,
    market_regime: str = "",
) -> int:
    """Deterministic seed for forecast noise.

    Includes symbol/interval context to avoid repeated template curves
    when feature signatures are similar across symbols.

    IMPROVEMENTS:
    - Enhanced with news sentiment for event-driven differentiation
    - Market regime awareness for regime-specific trajectories
    - Time-based decorrelation to prevent stale patterns
    - Multi-scale price pattern capture for better symbol uniqueness
    """
    # Context hash from symbol/interval
    ctx_hash = 0
    context_str = str(seed_context or "")
    for ch in context_str:
        ctx_hash = ((ctx_hash * 131) + ord(ch)) & 0x7FFFFFFF

    # Add price-level invariant hash (avoids similar seeds for similar-priced stocks)
    price_log_hash = 0
    if current_price > 0:
        # Use log price to make seed invariant to price scale
        price_normalized = int(abs(np.log(max(current_price, 1.0)) * 1000)) & 0x7FFFFFFF
        price_log_hash = price_normalized

    # Recent price pattern hash (captures recent volatility pattern)
    recent_hash = 0
    recent_skew_hash = 0
    if recent_prices is not None:
        try:
            rp = np.array(
                [float(p) for p in recent_prices if float(p) > 0],
                dtype=float,
            )
            if rp.size > 0:
                # Use returns instead of absolute prices for better invariance
                if rp.size > 1:
                    returns = np.diff(np.log(rp))
                    # Quantize returns to reduce sensitivity to small changes
                    returns_quant = np.round(returns * 100).astype(np.int64)
                    # Hash the return pattern (recent 20)
                    for _i, val in enumerate(returns_quant[-min(20, len(returns_quant)):]):
                        recent_hash = ((recent_hash * 137) + int(val)) & 0x7FFFFFFF
                    # Add skewness capture for asymmetry detection
                    if returns.size >= 10:
                        recent_skew = float(np.mean(((returns - np.mean(returns)) / (np.std(returns) + 1e-9)) ** 3))
                        recent_skew_hash = int(abs(recent_skew) * 10000) & 0x7FFFFFFF
                else:
                    # Fallback for single price
                    tail = rp[-min(12, rp.size):]
                    weights = np.arange(1, tail.size + 1, dtype=float)
                    recent_hash = int(
                        abs(np.sum(np.round(tail, 4) * weights) * 10.0)
                    ) & 0x7FFFFFFF
        except (TypeError, ValueError):
            recent_hash = 0
            recent_skew_hash = 0

    # Volatility context adds regime-specific variation
    vol_hash = 0
    if volatility_context > 0:
        vol_hash = int(abs(volatility_context) * 10000) & 0x7FFFFFFF

    # Direction hint contribution (ensures different seeds for different biases)
    dir_hash = int((float(np.clip(direction_hint, -1.0, 1.0)) + 1.0) * 500000) & 0x7FFFFFFF

    # News sentiment hash - adds event-driven differentiation
    news_hash = 0
    if abs(news_sentiment) > 1e-6:
        news_hash = int((float(np.clip(news_sentiment, -1.0, 1.0)) + 1.0) * 250000) & 0x7FFFFFFF

    # Market regime hash - regime-specific trajectory shaping
    regime_hash = 0
    if market_regime:
        regime_str = str(market_regime).strip().upper()
        for ch in regime_str:
            regime_hash = ((regime_hash * 149) + ord(ch)) & 0x7FFFFFFF

    # Time-based decorrelation - prevents stale patterns across days
    # Changes every 6 hours to provide fresh randomness while maintaining intraday consistency
    time_slot = int(time.time() // (6 * 3600)) & 0xFFFF
    time_hash = (time_slot * 271) & 0x7FFFFFFF

    # Combine all components with different prime multipliers for mixing
    seed = (
        (int(abs(float(sequence_signature)) * 1000) * 17)
        ^ (int(max(1, int(horizon)) * 131) * 23)
        ^ (ctx_hash * 31)
        ^ (price_log_hash * 37)
        ^ (recent_hash * 41)
        ^ (recent_skew_hash * 47)
        ^ (vol_hash * 43)
        ^ (dir_hash * 53)
        ^ (news_hash * 59)
        ^ (regime_hash * 61)
        ^ time_hash
    ) % (2**31 - 1)

    # Ensure non-zero seed with better distribution
    if seed == 0:
        seed = (ctx_hash + price_log_hash + recent_hash + vol_hash) % (2**31 - 1)
        if seed == 0:
            seed = 42  # Fallback constant

    return int(seed)

def _generate_forecast(
    self,
    X: FloatArray,
    current_price: float,
    horizon: int,
    atr_pct: float = 0.02,
    sequence_signature: float = 0.0,
    seed_context: str = "",
    recent_prices: list[float] | None = None,
    news_bias: float = 0.0,
) -> list[float]:
    """Generate price forecast using forecaster or ensemble."""
    if current_price <= 0:
        return []
    horizon = max(1, int(horizon))
    atr_pct = float(np.nan_to_num(atr_pct, nan=0.02, posinf=0.02, neginf=0.02))
    if atr_pct <= 0:
        atr_pct = 0.02
    news_bias = float(
        np.clip(
            np.nan_to_num(news_bias, nan=0.0, posinf=0.0, neginf=0.0),
            -0.50,
            0.50,
        )
    )

    direction_hint = 0.0
    hint_confidence = 0.5
    hint_entropy = 0.5
    if self.ensemble is not None:
        try:
            hint_pred = self.ensemble.predict(X)
            probs_hint = getattr(hint_pred, "probabilities", None)
            if probs_hint is not None and len(probs_hint) >= 3:
                direction_hint = (
                    float(probs_hint[2]) - float(probs_hint[0])
                )
            hint_confidence = float(
                np.clip(getattr(hint_pred, "confidence", 0.5), 0.0, 1.0)
            )
            hint_entropy = float(
                np.clip(getattr(hint_pred, "entropy", 0.5), 0.0, 1.0)
            )
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Ensemble direction hint unavailable: %s", e)
            direction_hint = 0.0
            hint_confidence = 0.5
            hint_entropy = 0.5
    direction_hint = float(
        np.nan_to_num(direction_hint, nan=0.0, posinf=0.0, neginf=0.0)
    )
    if abs(news_bias) > 1e-8:
        direction_hint = float(
            np.clip(direction_hint + (news_bias * 0.65), -1.0, 1.0)
        )
    hint_confidence = float(
        np.clip(
            np.nan_to_num(hint_confidence, nan=0.5, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
    )
    hint_entropy = float(
        np.clip(
            np.nan_to_num(hint_entropy, nan=0.5, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
    )

    quality_scale = float(
        np.clip(
            (0.35 + (0.65 * hint_confidence)) * (1.0 - (0.55 * hint_entropy)),
            0.35,
            1.0,
        )
    )

    if self.forecaster is not None:
        try:
            import torch

            self.forecaster.eval()
            with torch.inference_mode():
                X_tensor = torch.FloatTensor(X)
                returns, _ = self.forecaster(X_tensor)
                returns_arr = np.asarray(
                    returns[0].detach().cpu().numpy(), dtype=float
                ).reshape(-1)

            if returns_arr.size <= 0:
                raise ValueError("Forecaster produced empty output")

            returns_arr = np.nan_to_num(
                returns_arr, nan=0.0, posinf=0.0, neginf=0.0
            )

            neutral_mode = abs(direction_hint) < 0.10
            neutral_bias = 0.0
            if neutral_mode and returns_arr.size > 0:
                neutral_bias = float(
                    np.mean(returns_arr[:min(8, returns_arr.size)])
                )

            prices_arr = np.array(
                [float(p) for p in (recent_prices or []) if float(p) > 0],
                dtype=float,
            )
            recent_mu_pct = 0.0
            if prices_arr.size >= 6:
                try:
                    rets = np.diff(np.log(prices_arr[-min(90, prices_arr.size):]))
                    if rets.size > 0:
                        # FIX: Robust return estimation - detect and handle outlier candles
                        # A single large green/red candle can skew the mean significantly
                        rets_clean = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Detect outliers using median absolute deviation (MAD)
                        median_ret = float(np.median(rets_clean))
                        mad = float(np.median(np.abs(rets_clean - median_ret)))
                        mad_scale = max(mad * 1.4826, 1e-6)  # Scale MAD to be comparable to std
                        
                        # Flag outliers beyond 3 MAD-scaled deviations
                        outlier_threshold = 3.0 * mad_scale
                        outlier_mask = np.abs(rets_clean - median_ret) > outlier_threshold
                        
                        # If outliers detected, use robust estimation
                        if np.any(outlier_mask):
                            # Clip outliers to the threshold boundary
                            rets_clipped = np.clip(
                                rets_clean,
                                median_ret - outlier_threshold,
                                median_ret + outlier_threshold,
                            )
                            # Use trimmed mean (exclude extreme outliers entirely)
                            rets_trimmed = rets_clean[~outlier_mask]
                            if rets_trimmed.size >= 3:
                                recent_mu_pct = float(
                                    np.clip(np.mean(rets_trimmed) * 100.0, -0.25, 0.25)
                                )
                            else:
                                # Fall back to median if too many outliers
                                recent_mu_pct = float(
                                    np.clip(median_ret * 100.0, -0.25, 0.25)
                                )
                            log.debug(
                                "Outlier candles detected in recent history: %d/%d rows clipped, "
                                "using robust return estimate (%.4f%% vs raw %.4f%%)",
                                int(np.sum(outlier_mask)),
                                rets_clean.size,
                                recent_mu_pct,
                                np.clip(np.mean(rets_clean) * 100.0, -0.25, 0.25),
                            )
                        else:
                            # No outliers - use standard mean
                            recent_mu_pct = float(
                                np.clip(np.mean(rets_clean) * 100.0, -0.25, 0.25)
                            )
                except (ValueError, FloatingPointError):
                    recent_mu_pct = 0.0

            step_cap_pct = float(
                np.clip(max(float(atr_pct), 0.0035) * 140.0, 0.18, 3.0)
            )
            if neutral_mode:
                step_cap_pct = min(
                    step_cap_pct,
                    float(max(float(atr_pct) * 70.0, 0.35)),
                )
            step_cap_pct = max(
                step_cap_pct * quality_scale,
                0.12 if neutral_mode else 0.20,
            )
            news_drift_pct = float(
                news_bias
                * step_cap_pct
                * (0.14 if neutral_mode else 0.28)
            )

            # Determine market regime for regime-aware forecasting
            market_regime = getattr(self, '_current_market_regime', '')
            
            # Deterministic symbol-specific residual to avoid template-like tails.
            # IMPROVEMENT: Pass news sentiment and market regime for better differentiation
            seed = self._forecast_seed(
                current_price=current_price,
                sequence_signature=sequence_signature,
                direction_hint=direction_hint,
                horizon=horizon,
                seed_context=seed_context,
                recent_prices=recent_prices,
                volatility_context=float(atr_pct),
                news_sentiment=float(news_bias),
                market_regime=market_regime,
            )
            rng = np.random.RandomState(seed)

            tail_window = returns_arr[-min(10, returns_arr.size):]
            tail_mu = 0.0
            tail_sigma = 0.0
            if tail_window.size > 0:
                # FIX: Robust tail estimation - prevent single outlier returns from distorting forecast
                tail_clean = np.nan_to_num(tail_window, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Use median-based estimators for robustness
                tail_mu = float(np.median(tail_clean))
                
                # MAD-based volatility estimate (more robust than std)
                mad = float(np.median(np.abs(tail_clean - tail_mu)))
                tail_sigma = mad * 1.4826  # Scale MAD to be comparable to std
                
                # Fallback to std if MAD is zero (all values identical)
                if tail_sigma < 1e-8:
                    tail_sigma = float(np.std(tail_clean))
            
            tail_sigma_floor = step_cap_pct * (0.05 if neutral_mode else 0.10)
            tail_sigma = float(
                np.clip(
                    tail_sigma,
                    max(0.01, tail_sigma_floor),
                    max(0.06, step_cap_pct * 0.45),
                )
            )

            # IMPROVEMENT: Volatility-adaptive news impact scaling
            # Higher volatility = news has less immediate impact (market noise drowns signal)
            # Lower volatility = news has more sustained impact
            vol_regime = min(1.0, float(atr_pct) / 0.05)  # Normalize to 0-1 range
            news_impact_scale = 1.0 - (0.4 * vol_regime)  # 0.6 to 1.0 range
            
            # IMPROVEMENT: Asymmetric news impact (bad news hits harder than good news)
            news_asymmetry = 1.0 if news_bias >= 0 else 1.25  # Bad news has 25% more impact
            
            prev_eps = 0.0
            prev_ret = 0.0
            prev_model_ret = 0.0

            prices = [current_price]
            for i in range(horizon):
                if i < returns_arr.size:
                    raw_ret = float(returns_arr[i])
                else:
                    extra_i = i - returns_arr.size + 1
                    decay = float(
                        np.exp(
                            -extra_i / max(4.0, float(horizon) * 0.22)
                        )
                    )
                    tail_target = (
                        (tail_mu * (0.55 + (0.45 * decay)))
                        + (recent_mu_pct * (0.45 * (1.0 - decay)))
                    )
                    tail_noise = float(
                        rng.normal(
                            0.0,
                            tail_sigma * (0.35 + (0.65 * decay)),
                        )
                    )
                    raw_ret = (
                        (0.74 * prev_model_ret)
                        + (0.26 * tail_target)
                        + tail_noise
                    )
                prev_model_ret = raw_ret
                r_val = raw_ret

                if neutral_mode:
                    r_val = ((r_val - neutral_bias) * 0.45) + (recent_mu_pct * 0.35)
                    mean_pull_pct = (-(prices[-1] / current_price - 1.0)) * 8.0
                    r_val += mean_pull_pct
                else:
                    r_val = (r_val * 0.84) + (recent_mu_pct * 0.16)

                # IMPROVEMENT: Enhanced news bias integration
                # - Front-loaded decay for event-driven moves
                # - Volatility-adaptive scaling
                # - Asymmetric impact for positive/negative news
                if abs(news_drift_pct) > 1e-9:
                    # IMPROVEMENT: Multi-phase news decay
                    # Phase 1 (early bars): Strong immediate impact
                    # Phase 2 (later bars): Gradual fade to background
                    early_phase = max(0.0, 1.0 - (float(i) / max(1.0, float(horizon) * 0.35)))
                    late_phase = max(0.0, 1.0 - (float(i) / max(3.0, float(horizon) * 1.25)))
                    news_decay = (0.65 * early_phase) + (0.35 * late_phase)
                    
                    # Apply volatility-adaptive and asymmetric scaling
                    effective_news = (
                        float(news_drift_pct) 
                        * news_decay 
                        * news_impact_scale 
                        * news_asymmetry
                    )
                    r_val += effective_news

                # IMPROVEMENT: Volatility-adaptive noise scaling
                # Add heteroskedasticity: noise scales with both quality and volatility regime
                noise_scale = 0.55 + (0.45 * quality_scale)
                vol_adjusted_noise = noise_scale * (0.8 + 0.4 * vol_regime)  # Higher vol = more noise
                eps_scale = step_cap_pct * (0.06 if neutral_mode else 0.10) * vol_adjusted_noise
                eps = (0.62 * prev_eps) + float(rng.normal(0.0, eps_scale))
                prev_eps = eps
                r_val += eps
                
                # IMPROVEMENT: Adaptive momentum with volatility damping
                # In high vol regimes, reduce momentum to avoid runaway predictions
                momentum_retain = 0.78 - (0.15 * vol_regime)  # 0.63 to 0.78 range
                r_val = (momentum_retain * r_val) + ((1.0 - momentum_retain) * prev_ret)
                r_val = float(np.clip(r_val, -step_cap_pct, step_cap_pct))
                prev_ret = r_val
                next_price = prices[-1] * (1 + r_val / 100)
                next_price = max(
                    next_price, current_price * 0.5
                )
                next_price = min(
                    next_price, current_price * 2.0
                )
                prices.append(float(next_price))

            forecast_prices = prices[1:]
            if neutral_mode:
                neutral_cap = max(float(atr_pct) * 1.8, 0.015)
                lo = current_price * (1.0 - neutral_cap)
                hi = current_price * (1.0 + neutral_cap)
                forecast_prices = [float(np.clip(p, lo, hi)) for p in forecast_prices]
                if len(forecast_prices) >= 2:
                    for i in range(1, len(forecast_prices)):
                        forecast_prices[i] = float(
                            (0.82 * forecast_prices[i]) + (0.18 * forecast_prices[i - 1])
                        )
                    forecast_prices = [
                        float(np.clip(p, lo, hi)) for p in forecast_prices
                    ]

            return forecast_prices

        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug(f"Forecaster failed: {e}")

    # Fallback: ensemble-guided path shaped by recent symbol behavior.
    if self.ensemble:
        try:
            pred = self.ensemble.predict(X)

            probs = np.asarray(
                getattr(pred, "probabilities", [0.33, 0.34, 0.33]),
                dtype=float,
            ).reshape(-1)
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            if probs.size < 3:
                probs = np.pad(probs, (0, 3 - probs.size), constant_values=0.0)
            prob_sum = float(np.sum(probs[:3]))
            if prob_sum <= 0 or not np.isfinite(prob_sum):
                probs = np.array([0.33, 0.34, 0.33], dtype=float)
            else:
                probs = probs[:3] / prob_sum
            direction = (
                (float(probs[2]) if len(probs) > 2 else 0.33)
                - (float(probs[0]) if len(probs) > 0 else 0.33)
            )
            if abs(news_bias) > 1e-8:
                direction = float(
                    np.clip(direction + (news_bias * 0.70), -1.0, 1.0)
                )
            confidence = float(np.clip(getattr(pred, "confidence", 0.5), 0.0, 1.0))
            entropy = float(np.clip(getattr(pred, "entropy", 0.5), 0.0, 1.0))
            quality_scale = float(
                np.clip(
                    (0.35 + (0.65 * confidence)) * (1.0 - (0.55 * entropy)),
                    0.35,
                    1.0,
                )
            )
            neutral_mode = abs(direction) < 0.10

            # IMPROVEMENT: Volatility regime detection for adaptive forecasting
            vol_regime = min(1.0, float(atr_pct) / 0.05)  # Normalize to 0-1 range
            volatility = max(float(atr_pct), 0.005) * quality_scale
            if neutral_mode:
                volatility = min(volatility, 0.012)

            prices_arr = np.array(
                [float(p) for p in (recent_prices or []) if float(p) > 0],
                dtype=float,
            )
            if prices_arr.size >= 4:
                rets = np.diff(np.log(prices_arr[-min(80, prices_arr.size):]))
                ret_mu = float(np.clip(np.mean(rets), -0.02, 0.02))
                ret_sigma = float(
                    np.clip(
                        np.std(rets),
                        0.0006,
                        max(volatility * 0.8, 0.03),
                    )
                )
            else:
                ret_mu = 0.0
                ret_sigma = float(max(volatility * 0.45, 0.001))
            if neutral_mode:
                ret_mu *= 0.35
                ret_sigma = max(ret_sigma * 0.55, 0.0004)
            else:
                ret_sigma *= (0.70 + (0.30 * quality_scale))

            prices = []
            price = current_price

            # IMPROVEMENT: Volatility-adaptive news impact for ensemble path
            news_impact_scale = 1.0 - (0.4 * vol_regime)
            news_asymmetry = 1.0 if news_bias >= 0 else 1.25

            # Determine market regime for regime-aware forecasting
            market_regime = getattr(self, '_current_market_regime', '')

            # Deterministic seed using sequence signature to keep each symbol distinct.
            # IMPROVEMENT: Pass news sentiment and market regime for better differentiation
            seed = self._forecast_seed(
                current_price=current_price,
                sequence_signature=sequence_signature,
                direction_hint=direction,
                horizon=horizon,
                seed_context=seed_context,
                recent_prices=recent_prices,
                volatility_context=float(atr_pct),
                news_sentiment=float(news_bias),
                market_regime=market_regime,
            )
            rng = np.random.RandomState(seed)

            for i in range(horizon):
                decay = 1.0 - (i / (horizon * 1.8))
                drift_scale = 0.10 if neutral_mode else 0.20
                mu_scale = 0.20 if neutral_mode else 0.35
                drift = (direction * volatility * drift_scale) + (ret_mu * mu_scale)
                
                # IMPROVEMENT: Enhanced news integration with multi-phase decay
                if abs(news_bias) > 1e-8:
                    early_phase = max(0.0, 1.0 - (float(i) / max(1.0, float(horizon) * 0.35)))
                    late_phase = max(0.0, 1.0 - (float(i) / max(3.0, float(horizon) * 1.25)))
                    news_decay = (0.65 * early_phase) + (0.35 * late_phase)
                    
                    effective_news = (
                        news_bias
                        * volatility
                        * ((0.10 if neutral_mode else 0.22) * news_decay)
                        * news_impact_scale
                        * news_asymmetry
                    )
                    drift += effective_news
                    
                # IMPROVEMENT: Volatility-adaptive noise scaling
                vol_adjusted_noise = ((0.35 if neutral_mode else 0.55) + (0.45 * decay)) * (0.8 + 0.4 * vol_regime)
                noise = float(rng.normal(0.0, ret_sigma * vol_adjusted_noise))
                
                mean_revert = (-(0.18 if neutral_mode else 0.10)) * (
                    (price / current_price) - 1.0
                )
                change = drift + noise + mean_revert
                if neutral_mode:
                    max_step = max(volatility * 1.3, 0.007)
                else:
                    max_step = max(volatility * 2.2, 0.02)
                change = float(np.clip(change, -max_step, max_step))
                price = price * (1 + change)

                price = max(price, current_price * 0.5)
                price = min(price, current_price * 2.0)
                if neutral_mode:
                    neutral_cap = max(volatility * 1.5, 0.012)
                    price = float(
                        np.clip(
                            price,
                            current_price * (1.0 - neutral_cap),
                            current_price * (1.0 + neutral_cap),
                        )
                    )

                prices.append(float(price))

            return prices

        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug(f"Ensemble forecast failed: {e}")

    # Last resort: deterministic micro-trajectory from recent volatility
    # instead of a flat line when model artifacts are unavailable.
    prices_arr = np.array(
        [float(p) for p in (recent_prices or []) if float(p) > 0],
        dtype=float,
    )
    if prices_arr.size >= 4:
        try:
            rets = np.diff(np.log(prices_arr[-min(120, prices_arr.size):]))
            if rets.size > 0 and np.all(np.isfinite(rets)):
                ret_mu = float(np.clip(np.mean(rets), -0.01, 0.01))
                ret_sigma = float(
                    np.clip(
                        np.std(rets),
                        max(float(atr_pct) * 0.02, 0.0003),
                        max(float(atr_pct) * 0.18, 0.0060),
                    )
                )
            else:
                ret_mu = 0.0
                ret_sigma = float(max(float(atr_pct) * 0.05, 0.0006))
        except (ValueError, FloatingPointError):
            ret_mu = 0.0
            ret_sigma = float(max(float(atr_pct) * 0.05, 0.0006))
    else:
        ret_mu = 0.0
        ret_sigma = float(max(float(atr_pct) * 0.05, 0.0006))

    max_step = float(max(float(atr_pct) * 0.22, 0.0012))
    
    # IMPROVEMENT: Volatility-adaptive news impact for fallback path
    vol_regime = min(1.0, float(atr_pct) / 0.05)
    news_impact_scale = 1.0 - (0.4 * vol_regime)
    news_asymmetry = 1.0 if news_bias >= 0 else 1.25
    effective_news_drift = news_bias * 0.0012 * news_impact_scale * news_asymmetry
    
    drift = float(np.clip(ret_mu + effective_news_drift, -max_step * 0.45, max_step * 0.45))

    # Determine market regime for regime-aware forecasting
    market_regime = getattr(self, '_current_market_regime', '')

    # IMPROVEMENT: Pass news sentiment and market regime for better differentiation
    seed = self._forecast_seed(
        current_price=current_price,
        sequence_signature=sequence_signature,
        direction_hint=direction_hint,
        horizon=horizon,
        seed_context=seed_context,
        recent_prices=recent_prices,
        volatility_context=float(atr_pct),
        news_sentiment=float(news_bias),
        market_regime=market_regime,
    )
    rng = np.random.RandomState(seed)

    fallback_prices: list[float] = []
    price = float(current_price)
    prev_eps = 0.0
    
    # IMPROVEMENT: Volatility-adaptive decay and noise scaling
    vol_adjusted_noise_scale = 0.45 + (0.55 * (0.8 + 0.4 * vol_regime))
    
    for i in range(horizon):
        decay = max(0.25, 1.0 - (float(i) / float(max(horizon, 1))) * 0.65)
        eps = (0.55 * prev_eps) + float(
            rng.normal(0.0, ret_sigma * (decay * vol_adjusted_noise_scale))
        )
        prev_eps = eps
        mean_revert = -0.08 * ((price / max(float(current_price), 1e-8)) - 1.0)
        change = float(np.clip(drift + eps + mean_revert, -max_step, max_step))
        if abs(change) < 1e-7:
            change = float(((-1.0) ** i) * max_step * 0.03)
        price = float(np.clip(price * (1.0 + change), current_price * 0.5, current_price * 2.0))
        fallback_prices.append(price)
    return fallback_prices

def _determine_signal(self, ensemble_pred: Any, pred: Prediction) -> Signal:
    """Determine trading signal from prediction.

    FIX #SIG-001: Added adaptive threshold adjustment based on market regime
    and volatility - higher thresholds in uncertain conditions.

    FIX #SIG-002: Improved edge calculation to use probability distribution
    shape, not just prob_up - prob_down difference.

    FIX #SIG-003: Added confidence-entropy interaction - high entropy
    requires higher confidence to generate actionable signals.
    
    FIX #SIG-004: Fixed signal consistency - prevent oscillation between
    HOLD and directional signals when metrics are near thresholds.
    """
    confidence = float(np.clip(ensemble_pred.confidence, 0.0, 1.0))
    predicted_class = int(np.clip(ensemble_pred.predicted_class, 0, 2))
    edge = float(np.clip(pred.prob_up - pred.prob_down, -1.0, 1.0))

    # FIX #SIG-002: Enhanced edge with probability distribution shape.
    neutral_prob = float(np.clip(getattr(pred, "prob_neutral", 0.34), 0.0, 1.0))
    if neutral_prob > 0.5:
        edge = edge * (1.0 - neutral_prob)

    trend = str(getattr(pred, "trend", "") or "").upper()
    is_sideways = trend == "SIDEWAYS"

    # FIX #SIG-001: Adaptive edge floor based on volatility and regime.
    base_edge_floor = 0.04
    vol_adjustment = 0.0
    atr_pct = float(np.clip(getattr(pred, "atr_pct_value", 0.02), 0.0, 1.0))
    if atr_pct > 0.04:
        vol_adjustment = 0.02
    elif atr_pct > 0.03:
        vol_adjustment = 0.01
    if is_sideways:
        vol_adjustment += 0.02
    edge_floor = float(np.clip(base_edge_floor + vol_adjustment, 0.02, 0.15))
    strong_edge_floor = float(np.clip(max(0.12, edge_floor * 2.0), 0.10, 0.25))

    # FIX #SIG-003: Entropy adjustment to confidence.
    entropy = float(np.clip(getattr(pred, "entropy", 0.0), 0.0, 1.0))
    effective_confidence = confidence
    if entropy > 0.6:
        effective_confidence = confidence * (1.0 - (entropy - 0.6) * 0.3)
        effective_confidence = float(np.clip(effective_confidence, 0.0, 1.0))
    elif entropy < 0.3:
        effective_confidence = min(0.99, confidence * 1.02)

    raw_code = str(getattr(pred, "stock_code", "") or "").strip()
    try:
        code = str(self._clean_code(raw_code)).strip()
    except _PREDICTOR_RECOVERABLE_EXCEPTIONS:
        digits = "".join(ch for ch in raw_code if ch.isdigit())
        code = digits.zfill(6) if digits else raw_code
    interval = str(
        getattr(pred, "interval", getattr(self, "interval", "1m")) or "1m"
    )
    try:
        interval = str(self._normalize_interval_token(interval))
    except _PREDICTOR_RECOVERABLE_EXCEPTIONS:
        interval = str(interval).lower()

    signal_key = f"{code}:{interval}" if code else ""
    if (
        not hasattr(self, "_signal_memory_lock")
        or getattr(self, "_signal_memory_lock", None) is None
    ):
        self._signal_memory_lock = threading.Lock()
    if (
        not hasattr(self, "_signal_memory")
        or not isinstance(getattr(self, "_signal_memory", None), dict)
    ):
        self._signal_memory = {}

    prev_signal: Signal | None = None
    prev_direction = "hold"
    last_flip_ts = 0.0
    last_signal_ts = 0.0
    state: dict[str, Any] = {}
    if signal_key:
        with self._signal_memory_lock:
            state = dict(self._signal_memory.get(signal_key, {}) or {})
        prev_raw = str(state.get("signal", "") or "").upper()
        if prev_raw:
            try:
                prev_signal = Signal(prev_raw)
            except ValueError:
                prev_signal = None
        prev_direction = str(state.get("direction", "hold") or "hold").lower()
        try:
            last_flip_ts = float(state.get("last_flip_ts", 0.0) or 0.0)
        except (TypeError, ValueError):
            last_flip_ts = 0.0
        try:
            last_signal_ts = float(state.get("ts", 0.0) or 0.0)
        except (TypeError, ValueError):
            last_signal_ts = 0.0
    else:
        prev_signal = getattr(pred, "_prev_signal", None)

    signal_threshold_buffer = 0.02
    candidate = Signal.HOLD

    # Apply signal logic with effective confidence.
    if predicted_class == 2:  # UP
        if edge >= edge_floor:
            if effective_confidence >= float(CONFIG.STRONG_BUY_THRESHOLD):
                candidate = (
                    Signal.STRONG_BUY if edge >= strong_edge_floor else Signal.BUY
                )
            elif effective_confidence >= float(CONFIG.BUY_THRESHOLD):
                candidate = Signal.BUY
    elif predicted_class == 0:  # DOWN
        if edge <= -edge_floor:
            if effective_confidence >= float(CONFIG.STRONG_SELL_THRESHOLD):
                candidate = (
                    Signal.STRONG_SELL if edge <= -strong_edge_floor else Signal.SELL
                )
            elif effective_confidence >= float(CONFIG.SELL_THRESHOLD):
                candidate = Signal.SELL
    else:  # Neutral class: allow edge override when probabilities disagree.
        edge_override_floor = edge_floor + 0.04
        if effective_confidence >= max(float(CONFIG.BUY_THRESHOLD), 0.60):
            if edge >= edge_override_floor:
                candidate = Signal.BUY
            elif edge <= -edge_override_floor:
                candidate = Signal.SELL

    # FIX #SIG-004: Hysteresis using persisted previous signal state.
    if (
        candidate == Signal.HOLD
        and prev_signal
        in (Signal.BUY, Signal.STRONG_BUY, Signal.SELL, Signal.STRONG_SELL)
    ):
        was_bullish = prev_signal in (Signal.BUY, Signal.STRONG_BUY)
        if (
            was_bullish
            and edge >= (edge_floor - signal_threshold_buffer)
            and effective_confidence >= 0.45
        ):
            candidate = Signal.BUY if edge >= 0 else Signal.HOLD
        elif (
            (not was_bullish)
            and edge <= (-edge_floor + signal_threshold_buffer)
            and effective_confidence >= 0.45
        ):
            candidate = Signal.SELL if edge < 0 else Signal.HOLD

    # Behavioral guard: avoid rapid side flips unless edge and confidence are
    # clearly strong.
    now = time.monotonic()
    candidate_direction = "hold"
    if candidate in (Signal.BUY, Signal.STRONG_BUY):
        candidate_direction = "buy"
    elif candidate in (Signal.SELL, Signal.STRONG_SELL):
        candidate_direction = "sell"

    try:
        intraday = bool(self._is_intraday_interval(interval))
    except _PREDICTOR_RECOVERABLE_EXCEPTIONS:
        intraday = True

    flip_cooldown_s = 45.0 if intraday else 1800.0
    if is_sideways or atr_pct >= 0.04:
        flip_cooldown_s *= 2.0

    if (
        candidate_direction in {"buy", "sell"}
        and prev_direction in {"buy", "sell"}
        and candidate_direction != prev_direction
    ):
        ref_ts = float(last_flip_ts if last_flip_ts > 0 else last_signal_ts)
        elapsed = max(0.0, now - ref_ts)
        edge_strength = abs(float(edge))
        strong_reversal = (
            effective_confidence >= 0.93
            and edge_strength >= (strong_edge_floor * 2.0)
        )
        if elapsed < flip_cooldown_s and not strong_reversal:
            candidate = Signal.HOLD
            remaining = int(max(1.0, round(flip_cooldown_s - elapsed)))
            append_once = getattr(self, "_append_warning_once", None)
            if callable(append_once):
                append_once(
                    pred,
                    "Signal flip cooldown blocked rapid reversal "
                    f"({remaining}s remaining)",
                )

    if signal_key:
        new_direction = "hold"
        if candidate in (Signal.BUY, Signal.STRONG_BUY):
            new_direction = "buy"
        elif candidate in (Signal.SELL, Signal.STRONG_SELL):
            new_direction = "sell"

        if (
            new_direction in {"buy", "sell"}
            and prev_direction in {"buy", "sell"}
            and new_direction != prev_direction
        ):
            state["last_flip_ts"] = float(now)
        state["signal"] = candidate.value
        state["direction"] = new_direction
        state["ts"] = float(now)
        with self._signal_memory_lock:
            self._signal_memory[signal_key] = state
            if len(self._signal_memory) > 4096:
                stale_keys = sorted(
                    self._signal_memory.items(),
                    key=lambda kv: float((kv[1] or {}).get("ts", 0.0)),
                )[:1024]
                for k, _ in stale_keys:
                    self._signal_memory.pop(k, None)

    return candidate

def _calculate_signal_strength(self, ensemble_pred: Any, pred: Prediction) -> float:
    """Calculate signal strength 0-1."""
    confidence = float(ensemble_pred.confidence)
    agreement = float(getattr(ensemble_pred, "agreement", 1.0))
    entropy_inv = 1.0 - float(
        getattr(ensemble_pred, "entropy", 0.0)
    )

    return float(
        np.clip(
            (confidence + agreement + entropy_inv) / 3.0,
            0.0, 1.0
        )
    )

def _calculate_levels(self, pred: Prediction) -> TradingLevels:
    """Calculate trading levels using actual ATR from features."""
    price = pred.current_price

    if price <= 0:
        return TradingLevels()

    # Use actual ATR percentage from features, with floor
    atr_pct = max(pred.atr_pct_value, 0.005)

    levels = TradingLevels(entry=price)

    if pred.signal in [Signal.STRONG_BUY, Signal.BUY]:
        levels.stop_loss = price * (1 - atr_pct * 1.5)
        levels.target_1 = price * (1 + atr_pct * 1.5)
        levels.target_2 = price * (1 + atr_pct * 3.0)
        levels.target_3 = price * (1 + atr_pct * 5.0)
    elif pred.signal in [Signal.STRONG_SELL, Signal.SELL]:
        levels.stop_loss = price * (1 + atr_pct * 1.5)
        levels.target_1 = price * (1 - atr_pct * 1.5)
        levels.target_2 = price * (1 - atr_pct * 3.0)
        levels.target_3 = price * (1 - atr_pct * 5.0)
    else:
        levels.stop_loss = price * (1 - atr_pct)
        levels.target_1 = price * (1 + atr_pct)
        levels.target_2 = price * (1 + atr_pct * 2.0)
        levels.target_3 = price * (1 + atr_pct * 3.5)

    if price > 0:
        levels.stop_loss_pct = (levels.stop_loss / price - 1) * 100
        levels.target_1_pct = (levels.target_1 / price - 1) * 100
        levels.target_2_pct = (levels.target_2 / price - 1) * 100
        levels.target_3_pct = (levels.target_3 / price - 1) * 100

    return levels

def _calculate_position(self, pred: Prediction) -> PositionSize:
    """Calculate position size using risk, quality and expected-edge gating."""
    price = pred.current_price

    if price <= 0:
        return PositionSize()

    stop_distance, reward_distance = self._resolve_trade_distances(pred)
    if stop_distance <= 0 or reward_distance <= 0:
        return PositionSize()

    risk_pct = float(CONFIG.RISK_PER_TRADE) / 100.0
    if risk_pct <= 0:
        return PositionSize()

    quality_scale = self._quality_scale(pred)
    edge = self._expected_edge(pred, price, stop_distance, reward_distance)
    rr_ratio = reward_distance / max(stop_distance, 1e-9)
    min_edge = max(0.0, float(CONFIG.risk.min_expected_edge_pct) / 100.0)
    min_rr = max(0.1, float(CONFIG.risk.min_risk_reward_ratio))

    if rr_ratio < min_rr or edge <= 0.0:
        return PositionSize(
            expected_edge_pct=edge * 100.0,
            risk_reward_ratio=rr_ratio,
        )
    if quality_scale < 0.35 and float(np.clip(pred.confidence, 0.0, 1.0)) < 0.75:
        return PositionSize(
            expected_edge_pct=edge * 100.0,
            risk_reward_ratio=rr_ratio,
        )

    edge_scale = 1.0
    if min_edge > 0:
        edge_scale = float(np.clip(edge / min_edge, 0.0, CONFIG.risk.max_position_scale))

    tail_risk = float(np.clip(getattr(pred, "tail_risk_score", 0.5), 0.0, 1.0))
    uncertainty = float(np.clip(getattr(pred, "uncertainty_score", 0.5), 0.0, 1.0))
    if tail_risk >= 0.78 or uncertainty >= 0.86:
        return PositionSize(
            expected_edge_pct=edge * 100.0,
            risk_reward_ratio=rr_ratio,
        )

    risk_dampen = (1.0 - (0.50 * tail_risk)) * (1.0 - (0.35 * uncertainty))
    risk_dampen = float(np.clip(risk_dampen, 0.25, 1.0))
    risk_amount = self.capital * risk_pct * quality_scale * edge_scale * risk_dampen

    lot_size = max(1, CONFIG.LOT_SIZE)

    shares = int(risk_amount / stop_distance)
    shares = (shares // lot_size) * lot_size

    if shares < lot_size:
        return PositionSize(
            expected_edge_pct=edge * 100.0,
            risk_reward_ratio=rr_ratio,
        )

    max_value = self.capital * (CONFIG.MAX_POSITION_PCT / 100)
    if shares * price > max_value:
        shares = int(max_value / price)
        shares = (shares // lot_size) * lot_size

    if shares < lot_size:
        return PositionSize(
            expected_edge_pct=edge * 100.0,
            risk_reward_ratio=rr_ratio,
        )

    max_risk_amount = self.capital * risk_pct
    realized_risk = float(shares * stop_distance)
    if realized_risk > (max_risk_amount * 1.05):
        capped_shares = int(max_risk_amount / max(stop_distance, 1e-9))
        capped_shares = (capped_shares // lot_size) * lot_size
        if capped_shares < lot_size:
            return PositionSize(
                expected_edge_pct=edge * 100.0,
                risk_reward_ratio=rr_ratio,
            )
        shares = capped_shares

    if shares <= 0:
        return PositionSize(
            expected_edge_pct=edge * 100.0,
            risk_reward_ratio=rr_ratio,
        )

    if shares * price > self.capital:
        # Can't afford even one lot
        return PositionSize()

    return PositionSize(
        shares=int(shares),
        value=float(shares * price),
        risk_amount=float(shares * stop_distance),
        risk_pct=float(
            (shares * stop_distance / self.capital) * 100
        ),
        expected_edge_pct=float(edge * 100.0),
        risk_reward_ratio=float(rr_ratio),
    )

def _resolve_trade_distances(self, pred: Prediction) -> tuple[float, float]:
    """Resolve stop and reward distances from level plan."""
    price = float(pred.current_price)
    if price <= 0:
        return 0.0, 0.0

    stop_distance = abs(price - float(pred.levels.stop_loss))
    if stop_distance <= 0:
        stop_distance = price * 0.02

    d1 = abs(float(pred.levels.target_1) - price)
    d2 = abs(float(pred.levels.target_2) - price)
    # Weighted reward estimate: partial at target_1 plus runner to target_2.
    reward_distance = (0.7 * d1) + (0.3 * d2)
    if reward_distance <= 0:
        reward_distance = d1 if d1 > 0 else stop_distance

    return stop_distance, reward_distance

def _quality_scale(self, pred: Prediction) -> float:
    """Scale risk by signal quality (confidence and strength)."""
    conf = float(np.clip(pred.confidence, 0.0, 1.0))
    strength = float(np.clip(pred.signal_strength, 0.0, 1.0))
    agreement = float(np.clip(pred.model_agreement, 0.0, 1.0))
    quality = (0.5 * conf) + (0.35 * strength) + (0.15 * agreement)
    return float(np.clip(quality, 0.25, CONFIG.risk.max_position_scale))

def _expected_edge(
    self,
    pred: Prediction,
    price: float,
    stop_distance: float,
    reward_distance: float,
) -> float:
    """Estimate expected edge after costs.

    Returns decimal edge (e.g. 0.003 means +0.3% expected value).
    """
    if price <= 0:
        return 0.0

    if pred.signal in (Signal.BUY, Signal.STRONG_BUY):
        p_win = float(np.clip(pred.prob_up, 0.0, 1.0))
        p_loss = float(np.clip(pred.prob_down, 0.0, 1.0))
        side = "buy"
    elif pred.signal in (Signal.SELL, Signal.STRONG_SELL):
        p_win = float(np.clip(pred.prob_down, 0.0, 1.0))
        p_loss = float(np.clip(pred.prob_up, 0.0, 1.0))
        side = "sell"
    else:
        return 0.0

    reward_pct = reward_distance / price
    risk_pct = stop_distance / price
    gross_edge = (p_win * reward_pct) - (p_loss * risk_pct)
    cost_pct = self._round_trip_cost_pct(side)
    return float(gross_edge - cost_pct)

def _round_trip_cost_pct(self, side: str) -> float:
    """Estimate round-trip friction cost as decimal percentage."""
    commission = max(float(CONFIG.COMMISSION), 0.0)
    slippage = max(float(CONFIG.SLIPPAGE), 0.0)
    stamp_tax = max(float(CONFIG.STAMP_TAX), 0.0)

    if side == "sell":
        # Short-cover path includes one sell leg with stamp tax.
        return (2.0 * commission) + (2.0 * slippage) + stamp_tax
    # Long round trip: buy then sell, stamp tax on sell leg.
    return (2.0 * commission) + (2.0 * slippage) + stamp_tax

def _extract_technicals(self, df: pd.DataFrame, pred: Prediction) -> None:
    """Extract technical indicators from dataframe.

    IMPORTANT: FeatureEngine normalizes indicators:
    - rsi_14 = raw_rsi/100 - 0.5  (range: -0.5 to 0.5)
    - macd_hist = hist/close*100   (scale-invariant percentage)
    - ma_ratio_5_20 = (ma5/ma20 - 1) * 100
    """
    try:
        # RSI: reverse FeatureEngine normalization
        if "rsi_14" in df.columns:
            normalized_rsi = float(df["rsi_14"].iloc[-1])
            # FeatureEngine does: rsi_14 = raw_rsi/100 - 0.5
            # So: raw_rsi = (normalized_rsi + 0.5) * 100
            raw_rsi = (normalized_rsi + 0.5) * 100.0
            pred.rsi = float(np.clip(raw_rsi, 0.0, 100.0))

        if "macd_hist" in df.columns:
            macd_hist = float(df["macd_hist"].iloc[-1])
            if macd_hist > 0.001:
                pred.macd_signal = "BULLISH"
            elif macd_hist < -0.001:
                pred.macd_signal = "BEARISH"
            else:
                pred.macd_signal = "NEUTRAL"

        if "ma_ratio_5_20" in df.columns:
            ma_ratio = float(df["ma_ratio_5_20"].iloc[-1])
            if ma_ratio > 1.0:
                pred.trend = "UPTREND"
            elif ma_ratio < -1.0:
                pred.trend = "DOWNTREND"
            else:
                pred.trend = "SIDEWAYS"

        pred.atr_pct_value = self._get_atr_pct(df)

    except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
        log.debug(f"Technical extraction error: {e}")

def _get_atr_pct(self, df: pd.DataFrame) -> float:
    """Get ATR as a decimal fraction (e.g., 0.02 for 2%)."""
    try:
        if "atr_pct" in df.columns:
            atr = float(df["atr_pct"].iloc[-1])
            # atr_pct from FeatureEngine is: atr_14 / close * 100
            return max(atr / 100.0, 0.005)
    except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
        log.debug("ATR extraction failed; using default: %s", e)
    return 0.02  # default 2%

def _generate_reasons(self, pred: Prediction) -> None:
    """Generate analysis reasons and warnings."""
    existing_reasons = list(pred.reasons or [])
    existing_warnings = list(pred.warnings or [])
    reasons = []
    warnings = []

    if pred.confidence >= 0.7:
        reasons.append(
            f"High AI confidence: {pred.confidence:.0%}"
        )
    elif pred.confidence >= 0.6:
        reasons.append(
            f"Moderate AI confidence: {pred.confidence:.0%}"
        )
    else:
        warnings.append(
            f"Low AI confidence: {pred.confidence:.0%}"
        )

    if pred.model_agreement < 0.6:
        warnings.append(
            f"Low model agreement: {pred.model_agreement:.0%}"
        )

    if pred.prob_up > 0.5:
        reasons.append(
            f"AI predicts UP with {pred.prob_up:.0%} probability"
        )
    elif pred.prob_down > 0.5:
        reasons.append(
            f"AI predicts DOWN with {pred.prob_down:.0%} probability"
        )

    if pred.rsi > 70:
        warnings.append(f"RSI overbought: {pred.rsi:.0f}")
    elif pred.rsi < 30:
        warnings.append(f"RSI oversold: {pred.rsi:.0f}")
    else:
        reasons.append(f"RSI neutral: {pred.rsi:.0f}")

    # Signal-trend alignment
    if (
        pred.signal in [Signal.STRONG_BUY, Signal.BUY]
        and pred.trend == "UPTREND"
    ):
        reasons.append("Signal aligned with uptrend")
    elif (
        pred.signal in [Signal.STRONG_SELL, Signal.SELL]
        and pred.trend == "DOWNTREND"
    ):
        reasons.append("Signal aligned with downtrend")
    elif (
        pred.trend != "SIDEWAYS"
        and pred.signal != Signal.HOLD
    ):
        warnings.append(f"Signal against trend ({pred.trend})")

    if pred.macd_signal != "NEUTRAL":
        reasons.append(f"MACD: {pred.macd_signal}")

    if pred.entropy > 0.8:
        warnings.append(
            f"High prediction uncertainty "
            f"(entropy: {pred.entropy:.2f})"
        )

    uncertainty = float(np.clip(getattr(pred, "uncertainty_score", 0.5), 0.0, 1.0))
    tail_risk = float(np.clip(getattr(pred, "tail_risk_score", 0.5), 0.0, 1.0))
    if uncertainty >= 0.70:
        warnings.append(f"Wide uncertainty regime (score: {uncertainty:.2f})")
    else:
        reasons.append(f"Uncertainty score: {uncertainty:.2f}")

    if tail_risk >= 0.60:
        warnings.append(f"Elevated tail-event risk ({tail_risk:.2f})")
    else:
        reasons.append(f"Tail-event risk: {tail_risk:.2f}")

    low_band = list(getattr(pred, "predicted_prices_low", []) or [])
    high_band = list(getattr(pred, "predicted_prices_high", []) or [])
    if low_band and high_band and len(low_band) == len(high_band):
        try:
            lo_last = float(low_band[-1])
            hi_last = float(high_band[-1])
            ref = max(float(pred.current_price), 1e-8)
            spread_pct = float((hi_last - lo_last) / ref * 100.0)
            if spread_pct >= 6.0:
                warnings.append(
                    f"Forecast interval is wide ({spread_pct:.1f}% at horizon)"
                )
            else:
                reasons.append(
                    f"Forecast interval width: {spread_pct:.1f}% at horizon"
                )
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Failed computing forecast interval reason for %s: %s", pred.stock_code, e)

    pred.reasons = existing_reasons + [
        msg for msg in reasons if msg not in existing_reasons
    ]
    pred.warnings = existing_warnings + [
        msg for msg in warnings if msg not in existing_warnings
    ]

def _get_stock_name(self, code: str, df: pd.DataFrame) -> str:
    """Get stock name from fetcher."""
    del df
    try:
        quote = self.fetcher.get_realtime(code)
        if quote and quote.name:
            return str(quote.name)
    except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
        log.debug("Stock name lookup failed for %s: %s", code, e)
    return ""

def _clean_code(self, code: str) -> str:
    """Clean and normalize stock code.
    Delegates to DataFetcher when available.
    """
    if self.fetcher is not None:
        try:
            return self.fetcher.clean_code(code)
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Fetcher clean_code failed for %r: %s", code, e)

    if not code:
        return ""
    code = str(code).strip()
    code = "".join(c for c in code if c.isdigit())
    return code.zfill(6) if code else ""
