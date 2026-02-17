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


def _is_cn_session_datetime(dt: datetime) -> bool:
    if not isinstance(dt, datetime):
        return False
    if dt.weekday() >= 5:
        return False
    hhmm = (int(dt.hour) * 100) + int(dt.minute)
    return bool((930 <= hhmm <= 1130) or (1300 <= hhmm <= 1500))


def _is_session_timestamp(ts_raw: str, interval: str) -> bool:
    iv = str(interval or "1m").lower()
    if iv in ("1d", "1wk", "1mo"):
        return True
    try:
        dt = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
    except Exception:
        return False

    sh_tz = _shanghai_tz()
    if dt.tzinfo is None:
        return _is_cn_session_datetime(dt.replace(tzinfo=sh_tz))

    try:
        dt_sh = dt.astimezone(sh_tz)
        if _is_cn_session_datetime(dt_sh):
            return True
    except Exception:
        pass

    # Compatibility fallback: some inputs carry +00:00 while clock time is
    # already market-local. Treat clock fields as local before rejecting.
    try:
        return _is_cn_session_datetime(dt.replace(tzinfo=sh_tz))
    except Exception:
        return False


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
        self._last_close_hint: dict[str, float] = {}
        self._global_lock = threading.Lock()
        self._cleanup_corrupt_files()

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
        sh_tz = _shanghai_tz()
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
                # Treat naive provider timestamps as China market local time.
                dt = dt.replace(tzinfo=sh_tz)
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

    def _load_last_close_hint(self, path: Path) -> float | None:
        """Read last cached close for outlier-write guard."""
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, usecols=["close"])
            if df is None or df.empty:
                return None
            v = self._safe_float(df["close"].iloc[-1], 0.0)
            return float(v) if v > 0 else None
        except Exception:
            return None

    def _reference_close_from_db(self, symbol: str) -> float:
        """
        Best-effort local reference close for cache segment selection.
        Uses DB only (no network) to keep this path deterministic/offline-safe.
        """
        sym = _norm_symbol(symbol)
        if not sym:
            return 0.0
        try:
            from data.database import get_database

            db = get_database()
            intraday = db.get_intraday_bars(sym, interval="1m", limit=1)
            if isinstance(intraday, pd.DataFrame) and not intraday.empty:
                px = self._safe_float(intraday["close"].iloc[-1], 0.0)
                if px > 0:
                    return float(px)

            daily = db.get_bars(sym, limit=1)
            if isinstance(daily, pd.DataFrame) and not daily.empty:
                px = self._safe_float(daily["close"].iloc[-1], 0.0)
                if px > 0:
                    return float(px)
        except Exception:
            return 0.0
        return 0.0

    def _cleanup_corrupt_files(self) -> None:
        """
        Startup cleanup: quarantine obviously wrong-scale intraday cache files.
        """
        try:
            files = sorted(self._root.glob("*.csv"))
        except Exception:
            files = []
        if not files:
            return

        checked = 0
        quarantined = 0
        for path in files:
            stem = str(path.stem or "")
            if "_" not in stem:
                continue
            sym, iv = stem.split("_", 1)
            iv_norm = str(iv or "").lower()
            if iv_norm in ("1d", "1wk", "1mo"):
                continue
            ref = self._reference_close_from_db(sym)
            if ref <= 0:
                continue
            try:
                df = pd.read_csv(path, usecols=["close"])
            except Exception:
                continue
            if df is None or df.empty:
                continue
            closes = pd.to_numeric(df["close"], errors="coerce").dropna()
            closes = closes[closes > 0]
            if closes.empty:
                continue
            med = float(closes.tail(min(240, len(closes))).median())
            if med <= 0:
                continue
            ratio = med / float(ref)
            checked += 1
            if 0.2 <= ratio <= 5.0:
                continue
            try:
                qpath = path.with_suffix(
                    path.suffix + f".bad_{int(datetime.now().timestamp())}"
                )
                path.rename(qpath)
                quarantined += 1
            except Exception:
                continue
        if quarantined > 0:
            log.warning(
                "Session cache cleanup quarantined %s/%s suspicious files",
                quarantined,
                checked,
            )

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
        raw_ts = None
        for key in ("timestamp", "datetime", "time", "ts"):
            if key in bar:
                raw_ts = bar.get(key)
                break
        # Preserve session guard for naive textual timestamps, but do not
        # reject timezone-aware strings because providers vary between UTC and
        # market-local encodings.
        enforce_session = False
        if not isinstance(raw_ts, (int, float)):
            raw_text = str(raw_ts).strip() if raw_ts is not None else ""
            has_explicit_tz = bool(raw_text) and (
                raw_text.endswith("Z")
                or raw_text.endswith("z")
                or ("+" in raw_text[10:])
                or ("-" in raw_text[10:])
            )
            enforce_session = bool(raw_text) and not has_explicit_tz

        if enforce_session and not _is_session_timestamp(timestamp, iv):
            return False
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
            prev_close = self._last_close_hint.get(path.name)
            if prev_close is None:
                prev_close = self._load_last_close_hint(path)
                if prev_close and prev_close > 0:
                    self._last_close_hint[path.name] = float(prev_close)
            if prev_close and prev_close > 0:
                ratio = max(close, float(prev_close)) / max(
                    min(close, float(prev_close)),
                    1e-8,
                )
                if ratio >= 20.0:
                    log.debug(
                        "Session cache dropped extreme-scale write %s_%s: "
                        "prev=%.6f new=%.6f ratio=%.2fx",
                        sym,
                        iv,
                        float(prev_close),
                        float(close),
                        float(ratio),
                    )
                    return False
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
            self._last_close_hint[path.name] = float(close)
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
            for idx, row in df.sort_index().iterrows():
                c = self._safe_float(row.get("close", 0), 0.0)
                if c <= 0:
                    continue
                o = self._safe_float(row.get("open", c), c)
                h = self._safe_float(row.get("high", c), c)
                low = self._safe_float(row.get("low", c), c)
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

                current_seg.append((idx, (o, h, low, c)))
                prev_close = c
                prev_date = idx_date

            if current_seg:
                segments.append(current_seg)
            if not segments:
                return pd.DataFrame()

            # Prefer the segment closest to DB reference close when possible,
            # otherwise use the most recent segment.
            selected = segments[-1]
            ref_close = 0.0
            if iv not in ("1d", "1wk", "1mo"):
                ref_close = self._reference_close_from_db(sym)
            if ref_close > 0:
                best = selected
                best_err = float("inf")
                for seg in segments:
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
                    # Prefer closer scale; break ties with newer segment.
                    if err < best_err:
                        best = seg
                        best_err = err
                selected = best

            keep_idx = [idx for idx, _vals in selected]
            fixed = {idx: vals for idx, vals in selected}
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
