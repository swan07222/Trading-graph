from __future__ import annotations

import csv
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

    def append_bar(self, symbol: str, interval: str, bar: dict) -> bool:
        sym = _norm_symbol(symbol)
        iv = str(interval or "1m").lower()
        if not sym or not isinstance(bar, dict):
            return False

        try:
            close = float(bar.get("close", 0) or 0)
        except Exception:
            close = 0.0
        if close <= 0:
            return False

        row = {
            "timestamp": self._extract_timestamp(bar),
            "open": float(bar.get("open", close) or close),
            "high": float(bar.get("high", close) or close),
            "low": float(bar.get("low", close) or close),
            "close": close,
            "volume": float(bar.get("volume", 0) or 0),
            "amount": float(bar.get("amount", 0) or 0),
            "is_final": bool(bar.get("final", True)),
        }
        path = self._path(sym, iv)
        lock = self._lock_for(path.name)
        with lock:
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
