from __future__ import annotations

import math
from datetime import timezone
from typing import Any

import pandas as pd

from config.settings import CONFIG
from utils.logger import get_logger
from utils.type_utils import safe_float

log = get_logger(__name__)

def _norm_symbol(symbol: str) -> str:
    s = "".join(ch for ch in str(symbol or "").strip() if ch.isdigit())
    return s.zfill(6) if s else ""

def _shanghai_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Asia/Shanghai")
    except Exception:
        return timezone.utc

def _interval_safety_caps(interval: str) -> tuple[float, float]:
    """Return (max_jump_pct, max_range_pct) used for cached bar scrubbing."""
    iv = str(interval or "1m").strip().lower()
    if iv == "1m":
        return 0.08, 0.006
    if iv == "5m":
        return 0.10, 0.012
    if iv in ("15m", "30m"):
        return 0.14, 0.020
    if iv in ("60m", "1h"):
        return 0.18, 0.040
    if iv in ("1d", "1wk", "1mo"):
        return 0.24, 0.22
    return 0.20, 0.15


def _reference_scale_bounds(interval: str) -> tuple[float, float]:
    """Return allowed (min_ratio, max_ratio) against DB reference close."""
    iv = str(interval or "1m").strip().lower()
    if iv in ("1d", "1wk", "1mo"):
        return 0.25, 4.0
    return 0.45, 2.2

def read_history(
    self: Any,
    symbol: str,
    interval: str,
    bars: int = 500,
    final_only: bool = True,
) -> pd.DataFrame:
    sym = _norm_symbol(symbol)
    iv = str(interval or "1m").lower()
    if not sym:
        return pd.DataFrame()
    path = self._path(sym, iv)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame()
        if "timestamp" in df.columns:
            ts = df["timestamp"]
            dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
            sh_tz = _shanghai_tz()

            numeric_ts = pd.to_numeric(ts, errors="coerce")
            numeric_mask = numeric_ts.notna()
            if bool(numeric_mask.any()):
                # Treat large epoch values as milliseconds, otherwise seconds.
                numeric_vals = numeric_ts[numeric_mask]
                ms_mask = numeric_vals.abs() >= 1e11
                normalized_ms = numeric_vals.where(ms_mask, numeric_vals * 1000.0)
                parsed_num = pd.to_datetime(
                    normalized_ms,
                    unit="ms",
                    errors="coerce",
                    utc=True,
                ).dt.tz_convert(sh_tz).dt.tz_localize(None)
                dt.loc[numeric_mask] = parsed_num

            text_mask = dt.isna()
            if bool(text_mask.any()):
                text_vals = ts[text_mask].astype(str).str.strip()
                has_tz = text_vals.str.contains(
                    r"(?:Z|[+-]\d{2}:?\d{2})$",
                    regex=True,
                    na=False,
                )

                if bool(has_tz.any()):
                    aware = pd.to_datetime(
                        text_vals[has_tz],
                        format="ISO8601",
                        errors="coerce",
                        utc=True,
                    ).dt.tz_convert(sh_tz).dt.tz_localize(None)
                    dt.loc[text_vals[has_tz].index] = aware

                if bool((~has_tz).any()):
                    naive = pd.to_datetime(
                        text_vals[~has_tz],
                        format="ISO8601",
                        errors="coerce",
                    )
                    if getattr(naive.dt, "tz", None) is None:
                        naive = naive.dt.tz_localize(
                            sh_tz, nonexistent="NaT", ambiguous="NaT"
                        ).dt.tz_localize(None)
                    else:
                        naive = naive.dt.tz_convert(sh_tz).dt.tz_localize(None)
                    dt.loc[text_vals[~has_tz].index] = naive

            df["datetime"] = dt
            df = df.dropna(subset=["datetime"]).sort_values("datetime")
            df = df.drop_duplicates(subset=["datetime"], keep="last")
            df = df.set_index("datetime")
        if final_only and "is_final" in df.columns:
            final_mask = df["is_final"].astype(str).str.lower().isin(("true", "1"))
            if bool(final_mask.any()):
                df = df[final_mask]
            else:
                # Legacy recovery: some sessions wrote rolling partial rows
                # only. Keep stable historical rows and drop the newest row.
                if iv not in ("1d", "1wk", "1mo") and len(df) > 1:
                    df = df.sort_index().iloc[:-1]
        if df.empty:
            return pd.DataFrame()
        if "source" not in df.columns:
            df["source"] = ""
        df["source"] = (
            df["source"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
        df = df[df["close"] > 0].copy()
        if df.empty:
            return pd.DataFrame()

        # Scrub malformed cached bars to avoid rendering spikes after restart.
        jump_cap, range_cap = _interval_safety_caps(iv)
        segments: list[list[tuple[object, tuple[float, float, float, float]]]] = []
        current_seg: list[tuple[object, tuple[float, float, float, float]]] = []
        prev_close: float | None = None
        prev_date = None
        is_intraday = iv not in ("1d", "1wk", "1mo")
        preserve_truth = bool(
            getattr(getattr(CONFIG, "data", None), "truth_preserving_cleaning", True)
        )
        aggressive_repairs = bool(
            getattr(
                getattr(CONFIG, "data", None),
                "aggressive_intraday_repair",
                False,
            )
        )
        for idx, row in df.sort_index().iterrows():
            c = safe_float(row.get("close", 0), 0.0)
            if c <= 0:
                continue
            o = safe_float(row.get("open", c), c)
            h = safe_float(row.get("high", c), c)
            low = safe_float(row.get("low", c), c)
            idx_date = idx.date() if hasattr(idx, "date") else None
            ref_close = prev_close
            if (
                is_intraday
                and prev_date is not None
                and idx_date is not None
                and idx_date != prev_date
            ):
                # First bar of a new day can gap against prior close.
                ref_close = None

            if o <= 0:
                o = c
            if h <= 0:
                h = max(o, c)
            if low <= 0:
                low = min(o, c)
            if h < low:
                h, low = low, h

            if ref_close and ref_close > 0:
                jump = abs(c / ref_close - 1.0)
                if jump > jump_cap:
                    ratio = max(c, ref_close) / max(min(c, ref_close), 1e-8)
                    if ratio >= 20.0:
                        # Cached files can accumulate corrupted regimes over
                        # time (e.g. 1.x scale mixed with 70+ scale). Start
                        # a new segment instead of discarding newer valid bars.
                        if current_seg:
                            segments.append(current_seg)
                        current_seg = []
                        prev_close = None
                        prev_date = idx_date
                    else:
                        # Treat moderate jump as a local outlier row.
                        # Keep previous regime continuity.
                        continue

            anchor = float(ref_close if ref_close and ref_close > 0 else c)
            if ref_close and ref_close > 0:
                effective_range_cap = float(range_cap)
            else:
                bootstrap_cap = (
                    0.60
                    if iv in ("1d", "1wk", "1mo")
                    else float(max(0.008, min(0.020, range_cap * 2.0)))
                )
                effective_range_cap = float(max(range_cap, bootstrap_cap))

            max_body = float(anchor) * float(max(jump_cap * 1.25, effective_range_cap * 0.9))
            if max_body > 0 and abs(o - c) > max_body:
                if ref_close and ref_close > 0 and abs(c / ref_close - 1.0) <= jump_cap:
                    o = float(ref_close)
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
                if aggressive_repairs and not preserve_truth:
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
                else:
                    # Truth-preserving default: remove inflated wicks instead
                    # of synthesizing replacement ranges.
                    if body > max_range:
                        if (
                            ref_close
                            and ref_close > 0
                            and abs(c / ref_close - 1.0) <= jump_cap
                        ):
                            o = float(ref_close)
                        else:
                            o = c
                        top = max(o, c)
                        bot = min(o, c)
                    h = top
                    low = bot

            if anchor > 0 and (h - low) > (float(anchor) * float(effective_range_cap) * 1.05):
                continue

            current_seg.append((idx, (o, h, low, c)))
            prev_close = c
            prev_date = idx_date

        if current_seg:
            segments.append(current_seg)
        if not segments:
            return pd.DataFrame()

        # FIX #2026-02-24: Prefer the MOST RECENT segment by default, only
        # switch to an older segment if it has SIGNIFICANTLY better scale match.
        # This prevents showing outdated price data when scale is ambiguous.
        selected = segments[-1]  # Start with most recent segment
        ref_close = 0.0
        guard_enabled = True
        try:
            guard_fn = getattr(self, "_reference_scale_guard_enabled", None)
            if callable(guard_fn):
                guard_enabled = bool(guard_fn())
        except Exception:
            guard_enabled = True
        if guard_enabled and iv not in ("1d", "1wk", "1mo"):
            ref_close = self._reference_close_from_db(sym)
        if ref_close > 0:
            best = selected
            best_err = float("inf")
            # FIX #2026-02-24: Track index to prefer newer segments on ties
            best_idx = len(segments) - 1
            for seg_idx, seg in enumerate(segments):
                closes = [
                    float(vals[3]) for _idx, vals in seg
                    if float(vals[3]) > 0
                ]
                if not closes:
                    continue
                med = float(pd.Series(closes, dtype=float).median())
                if med <= 0:
                    continue
                ratio = med / float(ref_close)
                if ratio <= 0:
                    continue
                err = abs(math.log(ratio))
                # FIX #2026-02-24: Only switch to older segment if error is
                # SIGNIFICANTLY better (5% threshold), prefer newer on ties
                if err < best_err * 0.95:  # 5% improvement threshold
                    best = seg
                    best_err = err
                    best_idx = seg_idx
                # On tie, prefer newer segment (higher index)
                elif abs(err - best_err) <= best_err * 0.01 and seg_idx > best_idx:
                    best = seg
                    best_err = err
                    best_idx = seg_idx
            selected = best
            closes_sel = [
                float(vals[3]) for _idx, vals in selected
                if float(vals[3]) > 0
            ]
            if closes_sel:
                med_sel = float(pd.Series(closes_sel, dtype=float).median())
                if med_sel > 0:
                    ratio_sel = med_sel / float(ref_close)
                    min_ratio, max_ratio = _reference_scale_bounds(iv)
                    if ratio_sel < min_ratio or ratio_sel > max_ratio:
                        log.warning(
                            "Session cache dropped mismatched segment %s_%s: "
                            "ref=%.6f med=%.6f ratio=%.3f (allowed %.3f..%.3f)",
                            sym,
                            iv,
                            float(ref_close),
                            med_sel,
                            ratio_sel,
                            min_ratio,
                            max_ratio,
                        )
                        return pd.DataFrame()

        keep_idx = [idx for idx, _vals in selected]
        fixed = dict(selected)
        df = df.loc[keep_idx].copy()
        for idx, (o, h, low, c) in fixed.items():
            df.at[idx, "open"] = float(o)
            df.at[idx, "high"] = float(h)
            df.at[idx, "low"] = float(low)
            df.at[idx, "close"] = float(c)

        return df.tail(max(1, int(bars)))
    except Exception as e:
        log.debug("Session cache read failed (%s): %s", path.name, e)
        return pd.DataFrame()
