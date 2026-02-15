from __future__ import annotations

import csv
import math
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


def _norm_symbol(symbol: str) -> str:
    s = "".join(ch for ch in str(symbol or "").strip() if ch.isdigit())
    return s.zfill(6) if s else ""


def _parse_epoch_timestamp(value: float) -> datetime:
    """
    Parse epoch numeric values in seconds or milliseconds.
    """
    v = float(value)
    if abs(v) >= 1e11:
        v = v / 1000.0
    return datetime.fromtimestamp(v, tz=timezone.utc)


def _interval_safety_caps(interval: str) -> tuple[float, float]:
    """Return (max_jump_pct, max_range_pct) used for cached bar scrubbing."""
    iv = str(interval or "1m").strip().lower()
    if iv == "1m":
        return 0.12, 0.045
    if iv == "5m":
        return 0.14, 0.075
    if iv in ("15m", "30m"):
        return 0.18, 0.12
    if iv in ("60m", "1h"):
        return 0.24, 0.18
    if iv in ("1d", "1wk", "1mo"):
        return 0.45, 0.35
    return 0.20, 0.15


class SessionBarCache:
    """
    Persists bars captured during a UI session so auto-learning can reuse
    recently seen market data without refetching.
    """

    def __init__(self, root: Path | None = None) -> None:
        base = Path(root) if root else (CONFIG.data_dir / "session_bars")
        self._root = base
        self._root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._last_row_fingerprint: dict[str, tuple[str, float, bool]] = {}
        self._global_lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def _lock_for(self, key: str) -> threading.Lock:
        with self._global_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def _path(self, symbol: str, interval: str) -> Path:
        return self._root / f"{_norm_symbol(symbol)}_{str(interval).lower()}.csv"

    def _extract_timestamp(self, bar: dict) -> str:
        if not isinstance(bar, dict):
            return datetime.now(timezone.utc).isoformat()
        for key in ("timestamp", "datetime", "time", "ts"):
            raw = bar.get(key)
            if raw is None:
                continue
            if isinstance(raw, datetime):
                dt = raw
            else:
                text = str(raw).strip()
                if not text:
                    continue
                try:
                    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
                except Exception:
                    try:
                        dt = _parse_epoch_timestamp(float(text))
                    except Exception:
                        continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        if not math.isfinite(out):
            return float(default)
        return float(out)

    def append_bar(self, symbol: str, interval: str, bar: dict) -> bool:
        sym = _norm_symbol(symbol)
        iv = str(interval or "1m").lower()
        if not sym or not isinstance(bar, dict):
            return False

        close = self._safe_float(bar.get("close", 0), 0.0)
        if close <= 0:
            return False

        open_px = self._safe_float(bar.get("open", close), close)
        high_px = self._safe_float(bar.get("high", close), close)
        low_px = self._safe_float(bar.get("low", close), close)
        volume = self._safe_float(bar.get("volume", 0), 0.0)
        amount = self._safe_float(bar.get("amount", 0), 0.0)
        high_px = max(high_px, open_px, close)
        low_px = min(low_px, open_px, close)

        timestamp = self._extract_timestamp(bar)
        is_final = bool(bar.get("final", True))
        row = {
            "timestamp": timestamp,
            "open": open_px,
            "high": high_px,
            "low": low_px,
            "close": close,
            "volume": volume,
            "amount": amount,
            "is_final": is_final,
        }
        path = self._path(sym, iv)
        lock = self._lock_for(path.name)
        with lock:
            fingerprint = (
                timestamp,
                close,
                is_final,
            )
            if self._last_row_fingerprint.get(path.name) == fingerprint:
                return False
            write_header = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp", "open", "high", "low", "close",
                        "volume", "amount", "is_final",
                    ],
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            self._last_row_fingerprint[path.name] = fingerprint
        return True

    def read_history(
        self,
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
            if final_only and "is_final" in df.columns:
                df = df[df["is_final"].astype(str).str.lower().isin(("true", "1"))]
            if df.empty:
                return pd.DataFrame()
            if "timestamp" in df.columns:
                ts = df["timestamp"]
                dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

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
                    ).dt.tz_localize(None)
                    dt.loc[numeric_mask] = parsed_num

                text_mask = dt.isna()
                if bool(text_mask.any()):
                    parsed_text = pd.to_datetime(
                        ts[text_mask].astype(str),
                        format="ISO8601",
                        errors="coerce",
                        utc=True,
                    ).dt.tz_localize(None)
                    dt.loc[text_mask] = parsed_text

                df["datetime"] = dt
                df = df.dropna(subset=["datetime"]).sort_values("datetime")
                df = df.drop_duplicates(subset=["datetime"], keep="last")
                df = df.set_index("datetime")
            for col in ("open", "high", "low", "close", "volume", "amount"):
                if col not in df.columns:
                    df[col] = 0.0
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df = df[df["close"] > 0].copy()
            if df.empty:
                return pd.DataFrame()

            # Scrub malformed cached bars to avoid rendering spikes after restart.
            jump_cap, range_cap = _interval_safety_caps(iv)
            keep_idx = []
            fixed: dict = {}
            prev_close: float | None = None
            for idx, row in df.sort_index().iterrows():
                c = self._safe_float(row.get("close", 0), 0.0)
                if c <= 0:
                    continue
                o = self._safe_float(row.get("open", c), c)
                h = self._safe_float(row.get("high", c), c)
                low = self._safe_float(row.get("low", c), c)

                if o <= 0:
                    o = c
                if h <= 0:
                    h = max(o, c)
                if low <= 0:
                    low = min(o, c)
                if h < low:
                    h, low = low, h

                if prev_close and prev_close > 0:
                    jump = abs(c / prev_close - 1.0)
                    if jump > jump_cap:
                        continue

                anchor = float(prev_close if prev_close and prev_close > 0 else c)
                if prev_close and prev_close > 0:
                    effective_range_cap = float(range_cap)
                else:
                    bootstrap_cap = 0.60 if iv in ("1d", "1wk", "1mo") else 0.25
                    effective_range_cap = float(max(range_cap, bootstrap_cap))

                max_body = float(anchor) * float(max(jump_cap * 1.25, effective_range_cap * 0.9))
                if max_body > 0 and abs(o - c) > max_body:
                    if prev_close and prev_close > 0 and abs(c / prev_close - 1.0) <= jump_cap:
                        o = float(prev_close)
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

                if anchor > 0 and (h - low) > (float(anchor) * float(effective_range_cap) * 1.05):
                    continue

                keep_idx.append(idx)
                fixed[idx] = (o, h, low, c)
                prev_close = c

            if not keep_idx:
                return pd.DataFrame()

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

    def get_recent_symbols(
        self,
        interval: str | None = None,
        min_rows: int = 10,
    ) -> list[str]:
        iv = str(interval).lower() if interval else None
        out: list[str] = []
        for path in sorted(self._root.glob("*.csv")):
            stem = path.stem
            if "_" not in stem:
                continue
            sym, file_iv = stem.split("_", 1)
            if iv and file_iv != iv:
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    rows = sum(1 for _ in handle) - 1
            except Exception:
                rows = 0
            if rows >= min_rows:
                out.append(_norm_symbol(sym))
        return sorted(set(c for c in out if c))


_session_cache: SessionBarCache | None = None
_session_cache_lock = threading.Lock()


def get_session_bar_cache() -> SessionBarCache:
    global _session_cache
    if _session_cache is None:
        with _session_cache_lock:
            if _session_cache is None:
                _session_cache = SessionBarCache()
    return _session_cache
