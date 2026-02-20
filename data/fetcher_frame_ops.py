import os
from collections.abc import Callable

import pandas as pd

from data.fetcher_sources import _INTRADAY_INTERVALS


def resample_daily_to_interval(
    df: pd.DataFrame,
    interval: str,
    *,
    normalize_interval_token: Callable[[str | None], str],
    clean_dataframe: Callable[[pd.DataFrame, str | None], pd.DataFrame],
) -> pd.DataFrame:
    """Resample daily OHLCV bars to weekly/monthly bars when requested."""
    iv = normalize_interval_token(interval)
    if iv == "1d":
        return clean_dataframe(df, "1d")
    if iv not in {"1wk", "1mo"}:
        return clean_dataframe(df, iv)

    daily = clean_dataframe(df, "1d")
    if daily.empty or not isinstance(daily.index, pd.DatetimeIndex):
        return pd.DataFrame()

    rule = "W-FRI" if iv == "1wk" else "ME"
    agg: dict[str, str] = {}
    if "open" in daily.columns:
        agg["open"] = "first"
    if "high" in daily.columns:
        agg["high"] = "max"
    if "low" in daily.columns:
        agg["low"] = "min"
    if "close" in daily.columns:
        agg["close"] = "last"
    if "volume" in daily.columns:
        agg["volume"] = "sum"
    if "amount" in daily.columns:
        agg["amount"] = "sum"
    if not agg:
        return pd.DataFrame()

    resampled = daily.resample(rule).agg(agg)
    return clean_dataframe(resampled, iv)


def merge_parts(
    *dfs: pd.DataFrame,
    interval: str | None = None,
    clean_dataframe: Callable[[pd.DataFrame, str | None], pd.DataFrame],
) -> pd.DataFrame:
    """Merge and deduplicate non-empty dataframes."""
    parts = [p for p in dfs if isinstance(p, pd.DataFrame) and not p.empty]
    if not parts:
        return pd.DataFrame()
    if len(parts) == 1:
        return clean_dataframe(parts[0], interval)
    return clean_dataframe(pd.concat(parts, axis=0), interval)


def filter_cn_intraday_session(
    df: pd.DataFrame,
    interval: str,
    *,
    normalize_interval_token: Callable[[str | None], str],
    clean_dataframe: Callable[[pd.DataFrame, str | None], pd.DataFrame],
) -> pd.DataFrame:
    """Keep only regular CN A-share intraday session rows."""
    iv = normalize_interval_token(interval)
    if iv in {"1d", "1wk", "1mo"}:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    out = clean_dataframe(df, iv)
    if out.empty or not isinstance(out.index, pd.DatetimeIndex):
        return out

    session_policy = str(
        os.environ.get("TRADING_INTRADAY_SESSION_POLICY", "cn")
    ).strip().lower()
    if session_policy in {"off", "none", "all"}:
        return out

    idx = out.index
    hhmm = (idx.hour * 100) + idx.minute
    in_morning = (hhmm >= 930) & (hhmm <= 1130)
    in_afternoon = (hhmm >= 1300) & (hhmm <= 1500)
    weekday = idx.dayofweek < 5
    mask = weekday & (in_morning | in_afternoon)
    return out.loc[mask]


def history_cache_store_rows(
    interval: str | None,
    requested_rows: int,
    *,
    normalize_interval_token: Callable[[str | None], str],
) -> int:
    """
    Compute how many rows to keep in the shared history cache key.

    A larger shared window prevents cache-key fragmentation while still
    bounding memory and disk usage.
    """
    iv = normalize_interval_token(interval)
    req = max(1, int(requested_rows or 1))
    if iv in _INTRADAY_INTERVALS:
        floor = max(200, min(2400, req * 3))
    else:
        floor = max(400, min(5000, req * 2))
    return int(max(req, floor))
