from __future__ import annotations

import csv
import math
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)

_OFFICIAL_HISTORY_SOURCES = frozenset({"akshare", "itick"})


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


def _bars_per_day_for_interval(interval: str) -> float:
    iv = str(interval or "1m").strip().lower()
    mapping = {
        "1m": 240.0,
        "2m": 120.0,
        "5m": 48.0,
        "15m": 16.0,
        "30m": 8.0,
        "60m": 4.0,
        "1h": 4.0,
        "1d": 1.0,
        "1wk": 0.2,
        "1mo": 0.05,
    }
    return float(mapping.get(iv, 1.0))


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
        self._last_row_fingerprint: dict[
            str,
            tuple[str, float, float, float, float, bool, str],
        ] = {}
        self._last_close_hint: dict[str, float] = {}
        self._writes_since_compact: dict[str, int] = {}
        self._global_lock = threading.Lock()
        self._cleanup_corrupt_files()
        self._compact_existing_files()

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

    def _compact_existing_files(self) -> None:
        """Apply retention policy to existing cache files at startup."""
        try:
            files = sorted(self._root.glob("*.csv"))
        except Exception:
            files = []
        if not files:
            return

        compacted = 0
        for path in files:
            stem = str(path.stem or "")
            if "_" not in stem:
                continue
            _sym, iv = stem.split("_", 1)
            lock = self._lock_for(path.name)
            with lock:
                if self._maybe_compact_file_locked(path, iv, force=True):
                    compacted += 1
        if compacted > 0:
            log.info("Session cache startup compaction updated %s file(s)", compacted)

    @staticmethod
    def _compact_every_writes() -> int:
        cfg = getattr(CONFIG, "data", None)
        try:
            n = int(getattr(cfg, "session_cache_compact_every_writes", 240))
        except Exception:
            n = 240
        return max(1, n)

    @staticmethod
    def _max_file_megabytes() -> float:
        cfg = getattr(CONFIG, "data", None)
        try:
            mb = float(getattr(cfg, "session_cache_max_file_mb", 8.0))
        except Exception:
            mb = 8.0
        return max(0.5, mb)

    @staticmethod
    def _retention_days() -> int:
        cfg = getattr(CONFIG, "data", None)
        try:
            days = int(getattr(cfg, "session_cache_retention_days", 45))
        except Exception:
            days = 45
        return max(1, days)

    @staticmethod
    def _max_rows_for_interval(interval: str) -> int:
        cfg = getattr(CONFIG, "data", None)
        try:
            hard_cap = int(getattr(cfg, "session_cache_max_rows_per_symbol", 12000))
        except Exception:
            hard_cap = 12000
        hard_cap = max(1, hard_cap)

        retention_days = SessionBarCache._retention_days()
        target = int(
            max(
                10,
                math.ceil(_bars_per_day_for_interval(interval) * float(retention_days) * 1.35),
            )
        )
        return int(min(hard_cap, target))

    @classmethod
    def _apply_retention_policy(
        cls,
        frame: pd.DataFrame,
        *,
        interval: str,
    ) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        out = frame.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()]
        if out.empty:
            return pd.DataFrame()
        out = out[~out.index.duplicated(keep="last")].sort_index()

        now_sh = datetime.now(tz=_shanghai_tz()).replace(tzinfo=None)
        cutoff = now_sh - timedelta(days=cls._retention_days())
        out = out.loc[out.index >= pd.Timestamp(cutoff)]
        if out.empty:
            return pd.DataFrame()

        row_limit = cls._max_rows_for_interval(interval)
        if len(out) > row_limit:
            out = out.tail(row_limit)
        return out

    @staticmethod
    def _interval_from_path(path: Path) -> str:
        stem = str(path.stem or "")
        if "_" not in stem:
            return "1m"
        _sym, iv = stem.split("_", 1)
        return str(iv or "1m").strip().lower() or "1m"

    def _maybe_compact_file_locked(
        self,
        path: Path,
        interval: str,
        *,
        force: bool = False,
    ) -> bool:
        """
        Compact one cache file if retention thresholds are exceeded.

        Caller must hold the per-file lock.
        """
        if not path.exists():
            return False

        if not force:
            writes = int(self._writes_since_compact.get(path.name, 0))
            over_write_budget = writes >= self._compact_every_writes()
            over_size_budget = False
            try:
                size_mb = float(path.stat().st_size) / (1024.0 * 1024.0)
                over_size_budget = size_mb >= self._max_file_megabytes()
            except Exception:
                over_size_budget = False
            if not (over_write_budget or over_size_budget):
                return False
            self._writes_since_compact[path.name] = 0

        frame = self._read_raw_frame_locked(path)
        if frame.empty:
            return False
        trimmed = self._apply_retention_policy(frame, interval=interval)
        if len(trimmed) == len(frame):
            return False
        self._write_raw_frame_locked(path, trimmed)
        return True

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
        source = str(bar.get("source", "") or "").strip().lower()
        row = {
            "timestamp": timestamp,
            "open": open_px,
            "high": high_px,
            "low": low_px,
            "close": close,
            "volume": volume,
            "amount": amount,
            "is_final": is_final,
            "source": source,
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
            # Include OHLC envelope in dedupe key so same-close updates with
            # refined high/low are not silently dropped.
            fingerprint = (
                timestamp,
                round(float(open_px), 8),
                round(float(high_px), 8),
                round(float(low_px), 8),
                round(float(close), 8),
                is_final,
                source,
            )
            if self._last_row_fingerprint.get(path.name) == fingerprint:
                return False
            write_header = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp", "open", "high", "low", "close",
                        "volume", "amount", "is_final", "source",
                    ],
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            self._last_row_fingerprint[path.name] = fingerprint
            self._last_close_hint[path.name] = float(close)
            self._writes_since_compact[path.name] = (
                int(self._writes_since_compact.get(path.name, 0)) + 1
            )
            self._maybe_compact_file_locked(path, iv)
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

    @staticmethod
    def _to_shanghai_naive_datetime(value) -> datetime | None:
        """Parse mixed timestamp representations to naive Asia/Shanghai datetime."""
        if value is None:
            return None
        try:
            if isinstance(value, datetime):
                dt = value
            else:
                text = str(value).strip()
                if not text:
                    return None
                try:
                    num = float(text)
                except Exception:
                    num = None
                if num is not None:
                    dt = _parse_epoch_timestamp(num)
                else:
                    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            sh_tz = _shanghai_tz()
            if dt.tzinfo is None:
                return dt
            return dt.astimezone(sh_tz).replace(tzinfo=None)
        except Exception:
            return None

    @classmethod
    def _empty_raw_frame(cls) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "open", "high", "low", "close",
                "volume", "amount", "is_final", "source",
            ]
        )

    def _read_raw_frame_locked(self, path: Path) -> pd.DataFrame:
        """
        Read cache CSV with minimal normalization.

        Caller must hold the per-file lock.
        """
        if not path.exists():
            return self._empty_raw_frame()
        try:
            df = pd.read_csv(path)
        except Exception:
            return self._empty_raw_frame()
        if df is None or df.empty or "timestamp" not in df.columns:
            return self._empty_raw_frame()

        parsed = [self._to_shanghai_naive_datetime(v) for v in list(df["timestamp"])]
        dt = pd.to_datetime(parsed, errors="coerce")
        df["datetime"] = dt
        df = df.dropna(subset=["datetime"])
        if df.empty:
            return self._empty_raw_frame()

        df = (
            df.sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
            .set_index("datetime")
        )

        if "source" not in df.columns:
            df["source"] = ""
        df["source"] = (
            df["source"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if "is_final" not in df.columns:
            df["is_final"] = True
        else:
            df["is_final"] = df["is_final"].astype(str).str.lower().isin(("true", "1"))

        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

        return df[
            ["open", "high", "low", "close", "volume", "amount", "is_final", "source"]
        ].copy()

    def _write_raw_frame_locked(self, path: Path, frame: pd.DataFrame) -> None:
        """
        Write normalized cache frame back to disk.

        Caller must hold the per-file lock.
        """
        if frame is None or frame.empty:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
            self._last_row_fingerprint.pop(path.name, None)
            self._last_close_hint.pop(path.name, None)
            return

        work = frame.copy()
        if not isinstance(work.index, pd.DatetimeIndex):
            work.index = pd.to_datetime(work.index, errors="coerce")
        work = work[~work.index.isna()]
        if work.empty:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
            self._last_row_fingerprint.pop(path.name, None)
            self._last_close_hint.pop(path.name, None)
            return

        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col not in work.columns:
                work[col] = 0.0
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0).astype(float)

        if "is_final" not in work.columns:
            work["is_final"] = True
        work["is_final"] = work["is_final"].astype(bool)

        if "source" not in work.columns:
            work["source"] = ""
        work["source"] = (
            work["source"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        work = work[work["close"] > 0]
        if work.empty:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
            self._last_row_fingerprint.pop(path.name, None)
            self._last_close_hint.pop(path.name, None)
            return

        work = work[~work.index.duplicated(keep="last")].sort_index()
        iv = self._interval_from_path(path)
        work = self._apply_retention_policy(work, interval=iv)
        if work.empty:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
            self._last_row_fingerprint.pop(path.name, None)
            self._last_close_hint.pop(path.name, None)
            self._writes_since_compact.pop(path.name, None)
            return

        out = pd.DataFrame(
            {
                "timestamp": [ts.isoformat() for ts in work.index],
                "open": work["open"].astype(float).values,
                "high": work["high"].astype(float).values,
                "low": work["low"].astype(float).values,
                "close": work["close"].astype(float).values,
                "volume": work["volume"].astype(float).values,
                "amount": work["amount"].astype(float).values,
                "is_final": work["is_final"].astype(bool).values,
                "source": work["source"].astype(str).values,
            }
        )
        out.to_csv(path, index=False)

        try:
            last = out.iloc[-1]
            self._last_row_fingerprint[path.name] = (
                str(last.get("timestamp", "")),
                round(float(last.get("open", 0.0)), 8),
                round(float(last.get("high", 0.0)), 8),
                round(float(last.get("low", 0.0)), 8),
                round(float(last.get("close", 0.0)), 8),
                bool(last.get("is_final", True)),
                str(last.get("source", "")),
            )
            self._last_close_hint[path.name] = float(last.get("close", 0.0) or 0.0)
        except Exception:
            pass
        self._writes_since_compact[path.name] = 0

    def describe_symbol_interval(
        self,
        symbol: str,
        interval: str,
    ) -> dict[str, object]:
        """
        Describe cache window markers for one symbol/interval.

        Returns timestamps as naive Asia/Shanghai ``datetime`` objects.
        """
        sym = _norm_symbol(symbol)
        iv = str(interval or "1m").lower()
        if not sym:
            return {
                "rows": 0,
                "first_ts": None,
                "last_ts": None,
                "first_realtime_ts": None,
                "first_realtime_after_akshare_ts": None,
                "last_akshare_ts": None,
            }

        path = self._path(sym, iv)
        lock = self._lock_for(path.name)
        with lock:
            df = self._read_raw_frame_locked(path)

        if df.empty:
            return {
                "rows": 0,
                "first_ts": None,
                "last_ts": None,
                "first_realtime_ts": None,
                "first_realtime_after_akshare_ts": None,
                "last_akshare_ts": None,
            }

        src = df["source"].astype(str).str.strip().str.lower()
        ak_mask = src.isin(_OFFICIAL_HISTORY_SOURCES)
        rt_mask = ~ak_mask

        first_rt = None
        if bool(rt_mask.any()):
            try:
                first_rt = pd.Timestamp(df.loc[rt_mask].index.min()).to_pydatetime()
            except Exception:
                first_rt = None

        last_ak = None
        if bool(ak_mask.any()):
            try:
                last_ak = pd.Timestamp(df.loc[ak_mask].index.max()).to_pydatetime()
            except Exception:
                last_ak = None

        first_rt_after_ak = None
        try:
            if last_ak is not None:
                threshold = pd.Timestamp(last_ak)
                rt_after_mask = rt_mask & (df.index >= threshold)
                if bool(rt_after_mask.any()):
                    first_rt_after_ak = pd.Timestamp(
                        df.loc[rt_after_mask].index.min()
                    ).to_pydatetime()
            else:
                first_rt_after_ak = first_rt
        except Exception:
            first_rt_after_ak = first_rt

        try:
            first_ts = pd.Timestamp(df.index.min()).to_pydatetime()
        except Exception:
            first_ts = None
        try:
            last_ts = pd.Timestamp(df.index.max()).to_pydatetime()
        except Exception:
            last_ts = None

        return {
            "rows": int(len(df)),
            "first_ts": first_ts,
            "last_ts": last_ts,
            "first_realtime_ts": first_rt,
            "first_realtime_after_akshare_ts": first_rt_after_ak,
            "last_akshare_ts": last_ak,
        }

    def purge_realtime_rows(
        self,
        symbol: str,
        interval: str,
        *,
        since_ts: datetime | str | float | int | None = None,
    ) -> int:
        """
        Remove non-official-history rows for one symbol/interval from cache.

        Returns number of removed rows.
        """
        sym = _norm_symbol(symbol)
        iv = str(interval or "1m").lower()
        if not sym:
            return 0
        path = self._path(sym, iv)
        lock = self._lock_for(path.name)
        with lock:
            df = self._read_raw_frame_locked(path)
            if df.empty:
                return 0
            src = df["source"].astype(str).str.strip().str.lower()
            drop_mask = ~src.isin(_OFFICIAL_HISTORY_SOURCES)
            anchor = self._to_shanghai_naive_datetime(since_ts)
            if anchor is not None:
                try:
                    drop_mask = drop_mask & (df.index >= pd.Timestamp(anchor))
                except Exception:
                    pass
            keep_mask = ~drop_mask
            removed = int(drop_mask.sum())
            if removed <= 0:
                return 0
            keep_df = df.loc[keep_mask].copy()
            self._write_raw_frame_locked(path, keep_df)
            return removed

    def upsert_history_frame(
        self,
        symbol: str,
        interval: str,
        frame: pd.DataFrame,
        *,
        source: str = "akshare",
        is_final: bool = True,
    ) -> int:
        """
        Upsert OHLCV rows into session cache in one batch.

        Intended for writing official bars fetched from iTick/AKShare.
        """
        sym = _norm_symbol(symbol)
        iv = str(interval or "1m").lower()
        if not sym or frame is None or frame.empty:
            return 0

        work = frame.copy()
        if not isinstance(work.index, pd.DatetimeIndex):
            work.index = pd.to_datetime(work.index, errors="coerce")
        if getattr(work.index, "tz", None) is not None:
            try:
                work.index = work.index.tz_convert(_shanghai_tz()).tz_localize(None)
            except Exception:
                work.index = work.index.tz_localize(None)
        work = work[~work.index.isna()]
        if work.empty:
            return 0

        for col in ("open", "high", "low", "close"):
            if col not in work.columns:
                close_series = (
                    work["close"]
                    if "close" in work.columns
                    else pd.Series(0.0, index=work.index, dtype=float)
                )
                open_series = (
                    work["open"]
                    if "open" in work.columns
                    else close_series
                )
                if col == "open":
                    work[col] = close_series
                elif col == "high":
                    work[col] = pd.concat(
                        [open_series, close_series],
                        axis=1,
                    ).max(axis=1)
                elif col == "low":
                    work[col] = pd.concat(
                        [open_series, close_series],
                        axis=1,
                    ).min(axis=1)
            work[col] = pd.to_numeric(work[col], errors="coerce")

        for col in ("volume", "amount"):
            if col not in work.columns:
                work[col] = 0.0
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

        work = work.dropna(subset=["close"])
        work = work[work["close"] > 0]
        if work.empty:
            return 0

        work["open"] = work["open"].fillna(work["close"])
        work["high"] = pd.concat(
            [work["high"], work["open"], work["close"]],
            axis=1,
        ).max(axis=1)
        work["low"] = pd.concat(
            [work["low"], work["open"], work["close"]],
            axis=1,
        ).min(axis=1)

        is_intraday = iv not in ("1d", "1wk", "1mo")
        if is_intraday:
            mask = [
                bool(_is_cn_session_datetime(ts))
                for ts in work.index.to_pydatetime()
            ]
            work = work.loc[mask]
            if work.empty:
                return 0

        src = str(source or "").strip().lower()
        work["source"] = src
        work["is_final"] = bool(is_final)

        keep_cols = [
            "open", "high", "low", "close",
            "volume", "amount", "is_final", "source",
        ]
        work = work[keep_cols]
        work = work[~work.index.duplicated(keep="last")].sort_index()

        path = self._path(sym, iv)
        lock = self._lock_for(path.name)
        with lock:
            existing = self._read_raw_frame_locked(path)
            if existing.empty:
                merged = work
            else:
                merged = pd.concat([existing, work], axis=0)
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            self._write_raw_frame_locked(path, merged)
        return int(len(work))

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
