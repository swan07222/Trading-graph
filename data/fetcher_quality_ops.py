# data/fetcher_quality_ops.py
from typing import Any

import numpy as np
import pandas as pd

from config.settings import CONFIG
from data.fetcher_sources import _INTRADAY_CAPS
from utils.logger import get_logger

log = get_logger(__name__)

def _intraday_quality_caps(
    cls: Any,
    interval: str | None,
) -> tuple[float, float, float, float]:
    """Return (body_cap, span_cap, wick_cap, jump_cap) for intraday cleanup.

    Values are deliberately generous to avoid corrupting legitimate price
    moves (China A-shares can move +/-10% intraday; ST stocks +/-5%).
    Only truly malformed bars are removed.
    """
    iv = cls._normalize_interval_token(interval)
    caps = _INTRADAY_CAPS.get(iv)
    if caps:
        return caps
    # Default: conservative daily-like caps
    return 0.15, 0.22, 0.16, 0.22

def _intraday_frame_quality(
    cls,
    df: pd.DataFrame,
    interval: str,
) -> dict[str, float | bool]:
    """Score intraday frame quality.
    Higher score = cleaner and more usable bars.
    """
    if df is None or df.empty:
        return {
            "score": 0.0, "rows": 0.0,
            "stale_ratio": 1.0, "doji_ratio": 1.0,
            "zero_vol_ratio": 1.0, "extreme_ratio": 1.0,
            "suspect": True,
        }

    out = cls._clean_dataframe(
        df,
        interval=interval,
        preserve_truth=True,
        aggressive_repairs=False,
        allow_synthetic_index=False,
    )
    if out.empty:
        return {
            "score": 0.0, "rows": 0.0,
            "stale_ratio": 1.0, "doji_ratio": 1.0,
            "zero_vol_ratio": 1.0, "extreme_ratio": 1.0,
            "suspect": True,
        }

    body_cap, span_cap, wick_cap, _ = cls._intraday_quality_caps(interval)
    close_safe = out["close"].clip(lower=1e-8)
    rows_n = float(len(out))

    body = (out["open"] - out["close"]).abs() / close_safe
    span = (out["high"] - out["low"]).abs() / close_safe
    oc_top = out[["open", "close"]].max(axis=1)
    oc_bot = out[["open", "close"]].min(axis=1)
    upper_wick = (out["high"] - oc_top).clip(lower=0.0) / close_safe
    lower_wick = (oc_bot - out["low"]).clip(lower=0.0) / close_safe

    vol = (
        out["volume"] if "volume" in out.columns
        else pd.Series(0.0, index=out.index)
    ).fillna(0)

    zero_vol = vol <= 0

    # Stale detection: same close, flat OHLC, zero volume
    same_close = out["close"].diff().abs() <= (close_safe * 1e-6)
    flat_body  = body <= 1e-6
    flat_span  = span <= 2e-6
    stale_flat = same_close & flat_body & flat_span & zero_vol

    # Doji: near-zero body relative to span
    doji_ratio      = float((body <= (span.clip(lower=1e-8) * 0.05)).mean())
    stale_ratio     = float(stale_flat.mean())
    zero_vol_ratio  = float(zero_vol.mean())

    # Extreme: bars with body/span/wick far above expected cap
    extreme_mask = (
        (body > float(body_cap) * 2.0)
        | (span > float(span_cap) * 2.0)
        | (upper_wick > float(wick_cap) * 2.0)
        | (lower_wick > float(wick_cap) * 2.0)
    )
    extreme_ratio = float(extreme_mask.mean())

    # Score: penalize stale/flat bars more aggressively so frames
    # with many O=H=L=C zero-volume bars rank below cleaner sources.
    depth_score = min(1.0, rows_n / 600.0)
    stale_penalty = min(1.0, stale_ratio * 3.0)
    score = (
        (0.40 * depth_score)
        + (0.35 * (1.0 - stale_penalty))
        + (0.15 * (1.0 - min(1.0, zero_vol_ratio)))
        + (0.10 * (1.0 - min(1.0, extreme_ratio * 3.0)))
    )
    if doji_ratio > 0.95:
        score -= float((doji_ratio - 0.95) * 1.5)
    score = float(max(0.0, min(1.0, score)))

    suspect = bool(
        (rows_n < 40)
        or (stale_ratio >= 0.40)
        or (extreme_ratio >= 0.15)
        or (doji_ratio >= 0.98 and zero_vol_ratio >= 0.85)
    )
    return {
        "score":          score,
        "rows":           rows_n,
        "stale_ratio":    float(stale_ratio),
        "doji_ratio":     float(doji_ratio),
        "zero_vol_ratio": float(zero_vol_ratio),
        "extreme_ratio":  float(extreme_ratio),
        "suspect":        suspect,
    }

def _daily_frame_quality(cls, df: pd.DataFrame) -> dict[str, float | bool]:
    """Score daily/weekly/monthly history quality for source comparison."""
    if df is None or df.empty:
        return {
            "score": 0.0,
            "rows": 0.0,
            "stale_ratio": 1.0,
            "doji_ratio": 1.0,
            "zero_vol_ratio": 1.0,
            "extreme_ratio": 1.0,
            "suspect": True,
        }

    out = cls._clean_dataframe(
        df,
        interval="1d",
        preserve_truth=True,
        aggressive_repairs=False,
        allow_synthetic_index=False,
    )
    if out.empty:
        return {
            "score": 0.0,
            "rows": 0.0,
            "stale_ratio": 1.0,
            "doji_ratio": 1.0,
            "zero_vol_ratio": 1.0,
            "extreme_ratio": 1.0,
            "suspect": True,
        }

    rows_n = float(len(out))
    close = pd.to_numeric(out.get("close"), errors="coerce")
    close = close[close > 0] if isinstance(close, pd.Series) else pd.Series(dtype=float)
    if close.empty:
        return {
            "score": 0.0,
            "rows": 0.0,
            "stale_ratio": 1.0,
            "doji_ratio": 1.0,
            "zero_vol_ratio": 1.0,
            "extreme_ratio": 1.0,
            "suspect": True,
        }

    ret = close.pct_change().abs().fillna(0.0)
    extreme_ratio = float((ret > 0.22).mean())
    stale_ratio = float((close.diff().abs() <= 1e-8).mean())
    vol = (
        pd.to_numeric(out.get("volume"), errors="coerce").fillna(0.0)
        if "volume" in out.columns
        else pd.Series(0.0, index=out.index)
    )
    zero_vol_ratio = float((vol <= 0).mean())
    doji_ratio = 0.0

    depth_score = min(1.0, rows_n / 700.0)
    score = (
        (0.50 * depth_score)
        + (0.25 * (1.0 - min(1.0, extreme_ratio * 3.0)))
        + (0.15 * (1.0 - min(1.0, zero_vol_ratio)))
        + (0.10 * (1.0 - min(1.0, stale_ratio * 2.0)))
    )
    score = float(max(0.0, min(1.0, score)))

    suspect = bool(
        (rows_n < 25)
        or (extreme_ratio >= 0.10)
        or (zero_vol_ratio >= 0.65)
    )
    return {
        "score": score,
        "rows": rows_n,
        "stale_ratio": stale_ratio,
        "doji_ratio": doji_ratio,
        "zero_vol_ratio": zero_vol_ratio,
        "extreme_ratio": extreme_ratio,
        "suspect": suspect,
    }

def _max_close_cluster_size(
    closes: list[float],
    tolerance_ratio: float,
) -> int:
    """Largest in-tolerance close-price cluster size."""
    vals: list[float] = []
    for v in closes:
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv) and fv > 0:
            vals.append(fv)
    if not vals:
        return 0
    tol = float(max(0.0, tolerance_ratio))
    if tol <= 0:
        return 1
    best = 1
    for base in vals:
        denom = max(abs(float(base)), 1e-8)
        support = sum(
            1
            for px in vals
            if abs(float(px) - float(base)) / denom <= tol
        )
        if support > best:
            best = int(support)
    return int(best)

def _daily_consensus_quorum_meta(
    cls,
    collected: list[dict[str, Any]],
) -> dict[str, object]:
    """Compute daily provider quorum metadata.

    Quorum passes when at least ``required_sources`` providers align on a
    bar for a sufficient fraction of overlapping bars.
    """
    required_sources = int(
        max(
            2,
            int(
                getattr(
                    getattr(CONFIG, "data", None),
                    "history_quorum_required_sources",
                    2,
                )
                or 2
            ),
        )
    )
    tolerance_bps = float(
        getattr(
            getattr(CONFIG, "data", None),
            "history_quorum_tolerance_bps",
            80.0,
        )
        or 80.0
    )
    tolerance_ratio = max(0.0, tolerance_bps / 10000.0)
    min_ratio = float(
        getattr(
            getattr(CONFIG, "data", None),
            "history_quorum_min_ratio",
            0.55,
        )
        or 0.55
    )
    min_ratio = float(min(1.0, max(0.0, min_ratio)))

    valid = [
        c
        for c in list(collected or [])
        if isinstance(c.get("df"), pd.DataFrame)
        and (not c["df"].empty)
        and str(c.get("source", "")).strip().lower() != "localdb"
    ]
    source_names = sorted(
        {
            str(c.get("source", "")).strip().lower()
            for c in valid
            if str(c.get("source", "")).strip()
        }
    )
    meta: dict[str, object] = {
        "required_sources": int(required_sources),
        "sources": list(source_names),
        "source_count": int(len(source_names)),
        "tolerance_bps": float(tolerance_bps),
        "min_ratio": float(min_ratio),
        "compared_points": 0,
        "agreeing_points": 0,
        "agreeing_ratio": 0.0,
        "quorum_passed": False,
        "reason": "",
    }
    if len(valid) < required_sources:
        meta["reason"] = "insufficient_sources"
        return meta

    all_idx = pd.Index([])
    for item in valid:
        all_idx = all_idx.union(item["df"].index)
    if all_idx.empty:
        meta["reason"] = "empty_index"
        return meta

    compared_points = 0
    agreeing_points = 0
    for ts in all_idx.sort_values():
        closes: list[float] = []
        for item in valid:
            frame = item["df"]
            if ts not in frame.index:
                continue
            row_obj = frame.loc[ts]
            row = (
                row_obj.iloc[-1]
                if isinstance(row_obj, pd.DataFrame)
                else row_obj
            )
            if not isinstance(row, pd.Series):
                continue
            try:
                close_px = float(row.get("close", 0.0) or 0.0)
            except Exception:
                close_px = 0.0
            if close_px > 0 and np.isfinite(close_px):
                closes.append(close_px)
        if len(closes) < required_sources:
            continue
        compared_points += 1
        cluster_size = cls._max_close_cluster_size(
            closes,
            tolerance_ratio=tolerance_ratio,
        )
        if int(cluster_size) >= required_sources:
            agreeing_points += 1

    agreeing_ratio = (
        float(agreeing_points) / float(compared_points)
        if compared_points > 0
        else 0.0
    )
    quorum_passed = bool(
        compared_points > 0
        and agreeing_ratio >= min_ratio
    )

    meta.update(
        {
            "compared_points": int(compared_points),
            "agreeing_points": int(agreeing_points),
            "agreeing_ratio": float(agreeing_ratio),
            "quorum_passed": bool(quorum_passed),
            "reason": "" if quorum_passed else "insufficient_consensus",
        }
    )
    return meta

def _history_quorum_allows_persist(
    self,
    *,
    interval: str,
    symbol: str,
    meta: dict[str, object] | None,
) -> bool:
    """Strict daily quorum gate before DB persistence."""
    iv = self._normalize_interval_token(interval)
    if iv != "1d":
        return True

    if not isinstance(meta, dict) or not meta:
        log.debug(
            "History quorum metadata unavailable for %s (%s); allowing persist",
            str(symbol or ""),
            iv,
        )
        return True

    if bool(meta.get("quorum_passed", False)):
        return True

    log.warning(
        "Skipped DB persist for %s (%s): quorum failed "
        "(agree=%s/%s, ratio=%.2f, required=%s, sources=%s, reason=%s)",
        str(symbol or ""),
        iv,
        int(meta.get("agreeing_points", 0) or 0),
        int(meta.get("compared_points", 0) or 0),
        float(meta.get("agreeing_ratio", 0.0) or 0.0),
        int(meta.get("required_sources", 2) or 2),
        ",".join(str(x) for x in list(meta.get("sources", []) or [])),
        str(meta.get("reason", "quorum_failed")),
    )
    return False

def _merge_daily_by_consensus(
    cls,
    collected: list[dict[str, Any]],
    *,
    interval: str = "1d",
) -> pd.DataFrame:
    """Compare overlapping daily bars across sources and keep the row closest
    to per-timestamp consensus close price.
    """
    valid = [c for c in collected if isinstance(c.get("df"), pd.DataFrame) and not c["df"].empty]
    if not valid:
        return pd.DataFrame()
    if len(valid) == 1:
        return cls._clean_dataframe(valid[0]["df"], interval=interval)

    all_idx = pd.Index([])
    for item in valid:
        all_idx = all_idx.union(item["df"].index)
    if all_idx.empty:
        return pd.DataFrame()

    def _to_float(row: pd.Series, col: str, default: float = 0.0) -> float:
        try:
            val = row.get(col, default)
        except Exception:
            val = default
        try:
            return float(val)
        except Exception:
            return float(default)

    out_rows: list[dict[str, float]] = []
    out_index: list[pd.Timestamp] = []

    for ts in all_idx.sort_values():
        candidates: list[tuple[pd.Series, float, float, int]] = []
        for item in valid:
            frame = item["df"]
            if ts not in frame.index:
                continue
            row_obj = frame.loc[ts]
            row = row_obj.iloc[-1] if isinstance(row_obj, pd.DataFrame) else row_obj
            if not isinstance(row, pd.Series):
                continue
            close_px = _to_float(row, "close", 0.0)
            if close_px <= 0:
                continue
            quality_score = float(dict(item.get("quality") or {}).get("score", 0.0))
            rank = int(item.get("rank", 0))
            candidates.append((row, close_px, quality_score, rank))

        if not candidates:
            continue

        chosen: pd.Series
        if len(candidates) == 1:
            chosen = candidates[0][0]
        else:
            med = float(np.median(np.array([c[1] for c in candidates], dtype=float)))

            def _cost(
                candidate: tuple[pd.Series, float, float, int],
                median: float = med,
            ) -> tuple[float, float, int]:
                row, close_px, quality_score, rank = candidate
                open_px = _to_float(row, "open", close_px)
                high_px = _to_float(row, "high", max(open_px, close_px))
                low_px = _to_float(row, "low", min(open_px, close_px))
                vol = _to_float(row, "volume", 0.0)

                dev = abs(close_px - median) / max(abs(median), 1e-8)
                ohlc_penalty = 0.0
                if high_px < max(open_px, close_px):
                    ohlc_penalty += 0.03
                if low_px > min(open_px, close_px):
                    ohlc_penalty += 0.03
                vol_penalty = 0.02 if vol < 0 else 0.0
                return (float(dev + ohlc_penalty + vol_penalty), -quality_score, rank)

            chosen = min(candidates, key=_cost)[0]

        close_px = _to_float(chosen, "close", 0.0)
        if close_px <= 0:
            continue
        open_px = _to_float(chosen, "open", close_px)
        high_px = max(_to_float(chosen, "high", close_px), open_px, close_px)
        low_px = min(_to_float(chosen, "low", close_px), open_px, close_px)
        vol = max(0.0, _to_float(chosen, "volume", 0.0))
        amount = _to_float(chosen, "amount", 0.0)
        if amount <= 0:
            amount = close_px * vol

        out_rows.append(
            {
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "volume": vol,
                "amount": amount,
            }
        )
        out_index.append(pd.Timestamp(ts))

    if not out_rows:
        return pd.DataFrame()

    out = pd.DataFrame(out_rows, index=pd.DatetimeIndex(out_index, name="date"))
    return cls._clean_dataframe(out, interval=interval)

def _drop_stale_flat_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Remove completely flat bars (O=H=L=C, volume=0) before DB upsert.

    These bars carry no meaningful price information and contaminate
    the local database baseline, making future quality comparisons
    unreliable.
    """
    if df is None or df.empty:
        return df
    try:
        close_safe = df["close"].clip(lower=1e-8)
        body = (df["open"] - df["close"]).abs() / close_safe
        span = (df["high"] - df["low"]).abs() / close_safe
        vol = (
            df["volume"].fillna(0)
            if "volume" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        stale = (body <= 1e-6) & (span <= 2e-6) & (vol <= 0)
        n_stale = int(stale.sum())
        if n_stale > 0:
            log.debug(
                "Dropping %d flat stale bars before DB upsert (%d remaining)",
                n_stale, len(df) - n_stale,
            )
            return df[~stale]
    except Exception as exc:
        log.debug("_drop_stale_flat_bars skipped: %s", exc)
    return df

def _cross_validate_bars(
    cls,
    best_df: pd.DataFrame,
    alternatives: list[pd.DataFrame],
    interval: str,
) -> pd.DataFrame:
    """Replace stale bars in *best_df* with non-stale bars from *alternatives*.

    A bar is considered stale when its close is unchanged from the
    prior bar, body and span are near-zero, and volume is zero.
    For each such bar we look for a matching timestamp in the
    alternative DataFrames and substitute the first non-stale hit.
    """
    if best_df.empty or not alternatives:
        return best_df

    close_safe = best_df["close"].clip(lower=1e-8)
    body = (best_df["open"] - best_df["close"]).abs() / close_safe
    span = (best_df["high"] - best_df["low"]).abs() / close_safe
    vol = (
        best_df["volume"].fillna(0)
        if "volume" in best_df.columns
        else pd.Series(0.0, index=best_df.index)
    )
    same_close = best_df["close"].diff().abs() <= (close_safe * 1e-6)
    stale_mask = same_close & (body <= 1e-6) & (span <= 2e-6) & (vol <= 0)

    stale_indices = best_df.index[stale_mask]
    if len(stale_indices) == 0:
        return best_df

    out = best_df.copy()
    remaining = set(stale_indices)
    for alt_df in alternatives:
        if alt_df.empty or not remaining:
            break
        overlap = alt_df.index.intersection(pd.Index(list(remaining)))
        if overlap.empty:
            continue
        for idx in overlap:
            try:
                alt_row = alt_df.loc[idx]
                alt_body = abs(
                    float(alt_row.get("open", 0) if isinstance(alt_row, dict) else alt_row["open"])
                    - float(alt_row.get("close", 0) if isinstance(alt_row, dict) else alt_row["close"])
                )
                alt_vol = float(
                    (alt_row.get("volume", 0) if isinstance(alt_row, dict) else alt_row["volume"]) or 0
                )
            except Exception:
                continue
            if alt_body > 1e-6 or alt_vol > 0:
                for col in ("open", "high", "low", "close", "volume", "amount"):
                    if col in out.columns and col in alt_df.columns:
                        out.at[idx, col] = alt_df.at[idx, col]
                remaining.discard(idx)
        if not remaining:
            break

    replaced = len(stale_indices) - len(remaining)
    if replaced > 0:
        log.debug(
            "Cross-validated %d/%d stale bars from alternative sources",
            replaced, len(stale_indices),
        )
    return out

