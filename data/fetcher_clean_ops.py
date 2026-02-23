# data/fetcher_clean_ops.py
from typing import Any

import numpy as np
import pandas as pd

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)

def _to_shanghai_naive_ts(cls: Any, value: object) -> pd.Timestamp:
    """Parse one timestamp-like value -> Asia/Shanghai naive time.
    Returns NaT on failure.
    """
    if value is None:
        return pd.NaT

    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            v = float(value)
            if not np.isfinite(v) or abs(v) < 1e9:
                return pd.NaT
            if abs(v) >= 1e11:
                v /= 1000.0
            ts = pd.to_datetime(v, unit="s", errors="coerce", utc=True)
        else:
            text = str(value).strip()
            if not text:
                return pd.NaT
            if text.isdigit():
                num = float(text)
                if abs(num) < 1e9:
                    return pd.NaT
                if abs(num) >= 1e11:
                    num /= 1000.0
                ts = pd.to_datetime(num, unit="s", errors="coerce", utc=True)
            else:
                ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return pd.NaT

    if pd.isna(ts):
        return pd.NaT

    try:
        ts_obj = pd.Timestamp(ts)
    except Exception:
        return pd.NaT

    try:
        if ts_obj.tzinfo is not None:
            ts_obj = ts_obj.tz_convert("Asia/Shanghai").tz_localize(None)
    except Exception:
        try:
            ts_obj = ts_obj.tz_localize(None)
        except Exception:
            return pd.NaT
    return ts_obj

def _normalize_datetime_index(
    cls: Any,
    idx: object,
) -> pd.DatetimeIndex | None:
    """Convert an index-like object to DatetimeIndex in Asia/Shanghai naive time.
    Returns None when conversion is unreliable.
    """
    if isinstance(idx, pd.DatetimeIndex):
        out = idx
        try:
            if out.tz is not None:
                out = out.tz_convert("Asia/Shanghai").tz_localize(None)
        except Exception:
            try:
                out = out.tz_localize(None)
            except Exception as exc:
                log.debug("DatetimeIndex tz normalization failed: %s", exc)
        return pd.DatetimeIndex(out)

    values = list(idx) if idx is not None else []
    if not values:
        return None

    parsed = [cls._to_shanghai_naive_ts(v) for v in values]
    dt = pd.DatetimeIndex(parsed)
    valid_ratio = float(dt.notna().sum()) / float(max(1, len(dt)))
    if valid_ratio < 0.80:
        return None
    return dt


def _clean_dataframe(
    cls: Any,
    df: pd.DataFrame,
    interval: str | None = None,
    *,
    preserve_truth: bool | None = None,
    aggressive_repairs: bool | None = None,
    allow_synthetic_index: bool | None = None,
) -> pd.DataFrame:
    """Standardize and validate an OHLCV dataframe.

    Defaults are truth-preserving:
    - no synthetic intraday timestamps unless explicitly enabled
    - no aggressive intraday mutation unless explicitly enabled
    - duplicate timestamps keep first occurrence by default
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if preserve_truth is None:
        preserve_truth = bool(
            getattr(CONFIG.data, "truth_preserving_cleaning", True)
        )
    if aggressive_repairs is None:
        aggressive_repairs = bool(
            getattr(CONFIG.data, "aggressive_intraday_repair", False)
        )
    if allow_synthetic_index is None:
        allow_synthetic_index = bool(
            getattr(CONFIG.data, "synthesize_intraday_index", False)
        )

    preserve_truth = bool(preserve_truth)
    aggressive_repairs = bool(aggressive_repairs)
    allow_synthetic_index = bool(allow_synthetic_index)

    out = df.copy()
    iv = cls._normalize_interval_token(interval)
    is_intraday = iv not in {"1d", "1wk", "1mo"}

    # 1) Normalize index to DatetimeIndex when reliable.
    norm_idx = cls._normalize_datetime_index(out.index)
    has_dt_index = norm_idx is not None
    if norm_idx is not None:
        out.index = norm_idx

    if not has_dt_index:
        parsed_dt = None
        for col in ("datetime", "timestamp", "date", "time"):
            if col not in out.columns:
                continue
            dt = cls._normalize_datetime_index(out[col])
            if dt is None or len(dt) == 0:
                continue
            if float(dt.notna().sum()) / float(len(dt)) >= 0.80:
                parsed_dt = dt
                break

        if parsed_dt is None:
            try:
                idx_num = pd.to_numeric(
                    pd.Series(out.index, dtype=object), errors="coerce"
                )
                numeric_ratio = (
                    float(idx_num.notna().sum()) / float(len(idx_num))
                    if len(idx_num) > 0 else 0.0
                )
            except Exception:
                numeric_ratio = 0.0

            if numeric_ratio < 0.60:
                dt = cls._normalize_datetime_index(out.index)
                if dt is not None and len(dt) > 0:
                    if float(dt.notna().sum()) / float(len(dt)) >= 0.80:
                        parsed_dt = dt

        if parsed_dt is not None:
            out.index = parsed_dt
            has_dt_index = isinstance(out.index, pd.DatetimeIndex)
        else:
            if is_intraday and preserve_truth:
                return pd.DataFrame()
            out = out.reset_index(drop=True)

    # 2) Deduplicate and order.
    if has_dt_index:
        out = out[~out.index.isna()]
        keep_mode = "first" if preserve_truth else "last"
        out = out[~out.index.duplicated(keep=keep_mode)].sort_index()

    # 3) Numeric coercion.
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # 4) Basic close validity.
    if "close" not in out.columns:
        return pd.DataFrame()
    out = out.dropna(subset=["close"])
    out = out[out["close"] > 0]
    if out.empty:
        return pd.DataFrame()

    # 5) Minimal open repair.
    if "open" not in out.columns:
        out["open"] = out["close"]
    out["open"] = pd.to_numeric(out["open"], errors="coerce").fillna(0.0)
    out["open"] = out["open"].where(out["open"] > 0, out["close"])

    # 6) Minimal high/low repair.
    if "high" not in out.columns:
        out["high"] = out[["open", "close"]].max(axis=1)
    else:
        out["high"] = pd.to_numeric(out["high"], errors="coerce")

    if "low" not in out.columns:
        out["low"] = out[["open", "close"]].min(axis=1)
    else:
        out["low"] = pd.to_numeric(out["low"], errors="coerce")

    out["high"] = pd.concat(
        [out["high"], out["open"], out["close"]], axis=1
    ).max(axis=1)
    out["low"] = pd.concat(
        [out["low"], out["open"], out["close"]], axis=1
    ).min(axis=1)

    # 7) Aggressive intraday mutation is disabled to preserve raw source truth.
    _ = aggressive_repairs

    # 8) Volume >= 0.
    if "volume" in out.columns:
        out = out[out["volume"].fillna(0) >= 0]
    else:
        out["volume"] = 0.0

    # 9) high >= low.
    if "high" in out.columns and "low" in out.columns:
        out = out[out["high"].fillna(0) >= out["low"].fillna(0)]

    # 10) Derive amount if missing.
    if (
        "amount" not in out.columns
        and "close" in out.columns
        and "volume" in out.columns
    ):
        out["amount"] = out["close"] * out["volume"]

    # 11) Final cleanup.
    out = out.replace([np.inf, -np.inf], np.nan)
    ohlc_cols = [c for c in ("open", "high", "low", "close") if c in out.columns]
    if not ohlc_cols:
        return pd.DataFrame()

    if preserve_truth:
        out = out.dropna(subset=ohlc_cols)
        if out.empty:
            return pd.DataFrame()
        if "volume" in out.columns:
            out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
            out = out[out["volume"] >= 0]
        if "amount" in out.columns:
            out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    else:
        out[ohlc_cols] = out[ohlc_cols].ffill().bfill()
        out = out.fillna(0)

    if has_dt_index:
        keep_mode = "first" if preserve_truth else "last"
        out = out[~out.index.duplicated(keep=keep_mode)].sort_index()

    return out
