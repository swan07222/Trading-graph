from __future__ import annotations

import math
import time
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Any

from PyQt6.QtCore import QTimer

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)
_APP_SOFT_EXCEPTIONS = (
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    OSError,
    OverflowError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    ZeroDivisionError,
    ConnectionError,
    ConnectionResetError,
    ConnectionAbortedError,
    ConnectionRefusedError,
)


def _seven_day_lookback(self: Any, interval: str) -> int:
    """Return lookback bars representing ~2 trading days for interval."""
    iv = self._normalize_interval_token(interval)
    try:
        from data.fetcher import BARS_PER_DAY
        bpd = float(BARS_PER_DAY.get(iv, 1.0))
    except _APP_SOFT_EXCEPTIONS:
        fallback = {"1m": 240.0, "5m": 48.0, "15m": 16.0, "30m": 8.0, "60m": 4.0, "1h": 4.0, "1d": 1.0}
        bpd = float(fallback.get(iv, 1.0))
    bars = int(max(2, round(2.0 * bpd)))
    return max(50, bars) if iv != "1d" else 2

def _trained_stock_window_bars(
    self: Any, interval: str, window_days: int = 2
) -> int:
    """Return lookback bars representing the trained-stock refresh window."""
    iv = self._normalize_interval_token(interval)
    wd = max(1, int(window_days or 2))
    try:
        from data.fetcher import BARS_PER_DAY
        bpd = float(BARS_PER_DAY.get(iv, 1.0))
    except _APP_SOFT_EXCEPTIONS:
        fallback = {
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
        bpd = float(fallback.get(iv, 1.0))
    return int(max(1, round(float(wd) * max(0.01, bpd))))

def _recommended_lookback(self: Any, interval: str) -> int:
    """Recommended lookback for analysis/forecast per interval.
    Startup 1m uses a true 2-day 1m window; higher intervals keep a
    minimum depth for feature generation stability.
    """
    iv = self._normalize_interval_token(interval)
    base = int(self._seven_day_lookback(iv))
    if iv in ("1d", "1wk", "1mo"):
        return max(60, base)
    return max(120, base)

def _queue_history_refresh(self: Any, symbol: str, interval: str) -> None:
    """Force next history load to bypass memory/session cache once."""
    iv = self._normalize_interval_token(interval)
    sym = self._ui_norm(symbol)
    key = (sym if sym else "*", iv)
    self._history_refresh_once.add(key)
    
    # [DBG] History refresh queued diagnostic
    log.info(f"[DBG] History refresh queued: symbol={sym or '*'} interval={iv}")

def _consume_history_refresh(self: Any, symbol: str, interval: str) -> bool:
    """Consume one queued history refresh request for symbol/interval."""
    iv = self._normalize_interval_token(interval)
    sym = self._ui_norm(symbol)
    direct = (sym, iv)
    wildcard = ("*", iv)
    if direct in self._history_refresh_once:
        self._history_refresh_once.discard(direct)
        # [DBG] History refresh consumed diagnostic
        log.info(f"[DBG] History refresh consumed: symbol={sym} interval={iv} (direct)")
        return True
    if wildcard in self._history_refresh_once:
        self._history_refresh_once.discard(wildcard)
        # [DBG] History refresh consumed diagnostic (wildcard)
        log.info(f"[DBG] History refresh consumed: symbol={sym} interval={iv} (wildcard)")
        return True
    return False

def _schedule_analysis_recovery(
    self: Any,
    symbol: str,
    interval: str,
    warnings: list[str] | None = None,
) -> None:
    """Retry analysis once with a forced history refresh when output is partial.
    Throttled per symbol/interval to avoid retry loops.
    """
    sym = self._ui_norm(symbol)
    if not sym:
        return
    iv = self._normalize_interval_token(interval)
    key = f"{sym}:{iv}"
    now_ts = time.monotonic()
    last_ts = float(self._analysis_recovery_attempt_ts.get(key, 0.0) or 0.0)
    if (now_ts - last_ts) < 25.0:
        return
    self._analysis_recovery_attempt_ts[key] = now_ts

    self._queue_history_refresh(sym, iv)

    reason = ""
    warn_list = list(warnings or [])
    if warn_list:
        for item in warn_list:
            txt = str(item).strip()
            if txt:
                reason = txt
                break
    if reason:
        self.log(
            f"Data warm-up retry for {sym}: {reason}",
            "info",
        )
    else:
        self.log(
            f"Data warm-up retry for {sym}: refreshing history",
            "info",
        )

    def _retry_once() -> None:
        selected = self._ui_norm(self.stock_input.text())
        if selected != sym:
            return
        self._analyze_stock()

    QTimer.singleShot(1800, _retry_once)

def _history_window_bars(self: Any, interval: str) -> int:
    """Rolling chart/session window size (2-day equivalent)."""
    iv = self._normalize_interval_token(interval)
    bars = int(self._seven_day_lookback(iv))
    if iv == "1d":
        return max(7, bars)
    return max(120, bars)

def _ts_to_epoch(self: Any, ts_raw: Any) -> float:
    """Normalize timestamp-like values to epoch seconds."""
    if ts_raw is None:
        return float(time.time())

    try:
        if isinstance(ts_raw, (int, float)):
            v = float(ts_raw)
            # Treat large numeric timestamps as milliseconds.
            if abs(v) >= 1e11:
                v = v / 1000.0
            return v
    except _APP_SOFT_EXCEPTIONS as exc:
        log.debug("Suppressed exception in app_bar_ops", exc_info=exc)

    try:
        if isinstance(ts_raw, datetime):
            dt = ts_raw
        else:
            txt = str(ts_raw).strip()
            if not txt:
                return float(time.time())
            dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            # Most provider timestamps without tz are China local time.
            try:
                from zoneinfo import ZoneInfo
                dt = dt.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
            except _APP_SOFT_EXCEPTIONS:
                dt = dt.replace(tzinfo=UTC)
        return float(dt.timestamp())
    except _APP_SOFT_EXCEPTIONS:
        return float(time.time())

def _epoch_to_iso(self: Any, epoch: float) -> str:
    """Canonical ISO timestamp for chart bars (Asia/Shanghai)."""
    try:
        from zoneinfo import ZoneInfo
        sh_tz = ZoneInfo("Asia/Shanghai")
        return datetime.fromtimestamp(
            float(epoch), tz=sh_tz
        ).isoformat(timespec="seconds")
    except _APP_SOFT_EXCEPTIONS:
        try:
            return datetime.fromtimestamp(
                float(epoch), tz=UTC
            ).isoformat(timespec="seconds")
        except _APP_SOFT_EXCEPTIONS:
            return datetime.now(UTC).isoformat(timespec="seconds")

def _now_iso(self: Any) -> str:
    """Consistent sortable timestamp for live bars (Asia/Shanghai)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    except _APP_SOFT_EXCEPTIONS:
        return datetime.now(UTC).isoformat(timespec="seconds")

def _merge_bars(
    self: Any,
    base: list[dict[str, Any]],
    extra: list[dict[str, Any]],
    interval: str,
) -> list[dict[str, Any]]:
    """Merge+deduplicate bars by timestamp and keep rolling chart window."""
    merged: dict[int, dict[str, Any]] = {}
    iv = self._normalize_interval_token(interval)

    def _upsert(row_in: dict[str, Any]) -> None:
        epoch = self._bar_bucket_epoch(
            row_in.get("_ts_epoch", row_in.get("timestamp", "")),
            iv,
        )
        row = dict(row_in)
        row["_ts_epoch"] = float(epoch)
        row["timestamp"] = self._epoch_to_iso(epoch)

        # Never merge out-of-session intraday rows.
        if not self._is_market_session_timestamp(row["_ts_epoch"], iv):
            return

        key = int(epoch)
        existing = merged.get(key)
        if existing is None:
            merged[key] = row
            return

        existing_final = bool(existing.get("final", True))
        row_final = bool(row.get("final", True))
        if existing_final and not row_final:
            return
        if row_final and not existing_final:
            merged[key] = row
            return

        # Same finality: keep richer bar by volume, otherwise prefer newer row.
        try:
            e_vol = float(existing.get("volume", 0) or 0)
        except _APP_SOFT_EXCEPTIONS:
            e_vol = 0.0
        try:
            r_vol = float(row.get("volume", 0) or 0)
        except _APP_SOFT_EXCEPTIONS:
            r_vol = 0.0
        # Prefer later candidate on equal volume so fresh feed/history rows
        # can replace stale duplicates for the same bucket.
        if r_vol >= e_vol:
            merged[key] = row

    for b in (base or []):
        _upsert(b)
    for b in (extra or []):
        _upsert(b)
    out = list(merged.values())
    out.sort(
        key=lambda x: float(
            x.get(
                "_ts_epoch",
                self._ts_to_epoch(x.get("timestamp", "")),
            )
        )
    )

    # Final pass: sanitize OHLC and drop abrupt jumps.
    cleaned: list[dict[str, Any]] = []
    prev_close: float | None = None
    prev_epoch: float | None = None
    for row in out:
        try:
            c = float(row.get("close", 0) or 0)
            o = float(row.get("open", c) or c)
            h = float(row.get("high", c) or c)
            low = float(row.get("low", c) or c)
        except _APP_SOFT_EXCEPTIONS:
            continue
        row_epoch = float(
            self._bar_bucket_epoch(
                row.get("_ts_epoch", row.get("timestamp")),
                iv,
            )
        )
        ref_close = prev_close
        if (
            prev_epoch is not None
            and self._is_intraday_day_boundary(prev_epoch, row_epoch, iv)
        ):
            ref_close = None

        sanitized = self._sanitize_ohlc(
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
        if ref_close and ref_close > 0 and self._is_outlier_tick(
            ref_close, c, interval=iv
        ):
            continue

        row_out = dict(row)
        row_out["open"] = o
        row_out["high"] = h
        row_out["low"] = low
        row_out["close"] = c
        cleaned.append(row_out)
        prev_close = c
        prev_epoch = row_epoch

    keep = self._history_window_bars(interval)
    return cleaned[-keep:]

def _interval_seconds(self: Any, interval: str) -> int:
    """Map UI interval token to candle duration in seconds."""
    iv = self._normalize_interval_token(interval)
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "60m": 3600,
        "1h": 3600,
        "1d": 86400,
        "1wk": 604800,
        "1mo": 2592000,
    }
    if iv in mapping:
        return int(mapping[iv])
    # Generic support for provider labels like "90m" / "30s".
    try:
        if iv.endswith("m"):
            return max(1, int(float(iv[:-1])) * 60)
        if iv.endswith("s"):
            return max(1, int(float(iv[:-1])))
    except _APP_SOFT_EXCEPTIONS as exc:
        log.debug("Suppressed exception in app_bar_ops", exc_info=exc)
    return 60

def _interval_token_from_seconds(self: Any, seconds: Any) -> str | None:
    """Best-effort inverse mapping from seconds to interval token."""
    try:
        sec = max(1, int(float(seconds)))
    except _APP_SOFT_EXCEPTIONS:
        return None
    known = {
        60: "1m",
        300: "5m",
        900: "15m",
        1800: "30m",
        3600: "60m",
        86400: "1d",
        604800: "1wk",
        2592000: "1mo",
    }
    if sec in known:
        return known[sec]
    if sec % 60 == 0:
        return f"{int(sec // 60)}m"
    return f"{sec}s"

def _bars_needed_from_base_interval(
    self: Any,
    target_interval: str,
    target_bars: int,
    base_interval: str = "1m",
) -> int:
    """Estimate how many base-interval bars are needed to render
    `target_bars` in `target_interval`.
    """
    tgt = self._normalize_interval_token(target_interval)
    base = self._normalize_interval_token(base_interval)
    tgt_n = max(1, int(target_bars))

    try:
        src_sec = float(max(1, self._interval_seconds(base)))
        tgt_sec = float(max(1, self._interval_seconds(tgt)))
        factor = int(max(1, math.ceil(tgt_sec / src_sec)))
    except _APP_SOFT_EXCEPTIONS:
        factor = 1

    # CN market has about 240 one-minute bars per full session day.
    if tgt == "1d":
        factor = max(factor, 240)
    elif tgt == "1wk":
        factor = max(factor, 240 * 5)
    elif tgt == "1mo":
        factor = max(factor, 240 * 20)

    return int(max(tgt_n, (tgt_n * factor) + factor))

def _resample_chart_bars(
    self: Any,
    bars: list[dict[str, Any]],
    source_interval: str,
    target_interval: str,
) -> list[dict[str, Any]]:
    """Aggregate OHLC bars from source interval to target interval.
    Keeps candle integrity (open/close ordering, high/low envelope).
    """
    src = self._normalize_interval_token(source_interval)
    tgt = self._normalize_interval_token(target_interval)
    if src == tgt:
        return list(bars or [])
    if not bars:
        return []

    src_sec = int(max(1, self._interval_seconds(src)))
    tgt_sec = int(max(1, self._interval_seconds(tgt)))
    if tgt_sec <= src_sec:
        return list(bars or [])

    ranked = sorted(
        list(bars or []),
        key=lambda row: float(
            self._ts_to_epoch(
                row.get("_ts_epoch", row.get("timestamp", row.get("time")))
            )
        ),
    )

    buckets: dict[str, dict[str, Any]] = {}
    for row in ranked:
        try:
            ep = float(
                self._ts_to_epoch(
                    row.get("_ts_epoch", row.get("timestamp", row.get("time")))
                )
            )
        except _APP_SOFT_EXCEPTIONS:
            continue
        if not math.isfinite(ep):
            continue

        day_key = self._bar_trading_date(ep)
        if tgt == "1d":
            key = str(day_key) if day_key is not None else str(int(ep // 86400))
        elif tgt == "1wk":
            if day_key is None:
                key = f"week:{int(ep // (86400 * 7))}"
            else:
                iso = day_key.isocalendar()
                key = f"week:{int(iso.year)}-{int(iso.week):02d}"
        elif tgt == "1mo":
            if day_key is None:
                dt = datetime.fromtimestamp(ep)
                key = f"month:{dt.year}-{dt.month:02d}"
            else:
                key = f"month:{day_key.year}-{day_key.month:02d}"
        else:
            key = f"slot:{int(self._bar_bucket_epoch(ep, tgt))}"

        try:
            o = float(row.get("open", 0) or 0)
            h = float(row.get("high", 0) or 0)
            low = float(row.get("low", 0) or 0)
            c = float(row.get("close", 0) or 0)
        except _APP_SOFT_EXCEPTIONS:
            continue
        if c <= 0 or not all(math.isfinite(v) for v in (o, h, low, c)):
            continue

        if key not in buckets:
            try:
                vol = float(row.get("volume", 0) or 0.0)
            except _APP_SOFT_EXCEPTIONS:
                vol = 0.0
            buckets[key] = {
                "open": o if o > 0 else c,
                "high": max(h, o, c),
                "low": min(low, o, c),
                "close": c,
                "volume": max(0.0, vol),
                "_ts_epoch": float(ep),
                "final": bool(row.get("final", True)),
                "interval": tgt,
            }
            continue

        cur = buckets[key]
        cur["high"] = float(max(float(cur["high"]), h, c))
        cur["low"] = float(min(float(cur["low"]), low, c))
        cur["close"] = float(c)
        cur["_ts_epoch"] = float(max(float(cur["_ts_epoch"]), ep))
        cur["final"] = bool(cur.get("final", True) and bool(row.get("final", True)))
        try:
            cur["volume"] = float(cur.get("volume", 0.0)) + max(
                0.0,
                float(row.get("volume", 0) or 0.0),
            )
        except _APP_SOFT_EXCEPTIONS as exc:
            log.debug("Suppressed exception in app_bar_ops", exc_info=exc)

    out: list[dict[str, Any]] = []
    for val in buckets.values():
        row_out = dict(val)
        row_out["timestamp"] = self._epoch_to_iso(float(row_out["_ts_epoch"]))
        out.append(row_out)

    out.sort(key=lambda row: float(row.get("_ts_epoch", 0.0)))
    # Skip _merge_bars here: resampled buckets are already deduplicated
    # with correctly summed volume. Passing through _merge_bars would
    # use volume as a dedup quality signal and potentially replace
    # entries instead of summing, corrupting volume data.
    return out

def _dominant_bar_interval(
    self: Any,
    bars: list[dict[str, Any]] | None,
    fallback: str = "1m",
) -> str:
    """Most frequent interval token in bar list (best effort)."""
    counts: dict[str, int] = {}
    for row in (bars or []):
        if not isinstance(row, dict):
            continue
        iv = self._normalize_interval_token(
            row.get("interval"),
            fallback="",
        )
        if not iv:
            continue
        counts[iv] = int(counts.get(iv, 0)) + 1
    if not counts:
        return self._normalize_interval_token(fallback)
    best_iv = max(counts.items(), key=lambda kv: kv[1])[0]
    return self._normalize_interval_token(best_iv, fallback=fallback)

def _effective_anchor_price(
    self: Any,
    symbol: str,
    candidate: float | None = None,
) -> float:
    """Resolve a robust anchor price for chart scale repair.
    Prefers live/watchlist quote when candidate is obviously off-scale.
    """
    sym = self._ui_norm(symbol)
    try:
        base = float(candidate or 0.0)
    except _APP_SOFT_EXCEPTIONS:
        base = 0.0
    if not math.isfinite(base) or base <= 0:
        base = 0.0

    alt = 0.0
    try:
        rec = self._last_watchlist_price_ui.get(sym)
        if rec is not None:
            alt = float(rec[1] or 0.0)
    except _APP_SOFT_EXCEPTIONS:
        alt = 0.0
    if not math.isfinite(alt) or alt <= 0:
        alt = 0.0

    # Try live quote only when needed to avoid excess calls.
    if alt <= 0 and (base <= 0 or base < 5.0):
        fetcher = None
        try:
            if self.predictor is not None:
                fetcher = getattr(self.predictor, "fetcher", None)
        except _APP_SOFT_EXCEPTIONS:
            fetcher = None
        if fetcher is not None and sym:
            try:
                q = fetcher.get_realtime(sym)
                alt = float(getattr(q, "price", 0) or 0.0) if q is not None else 0.0
            except _APP_SOFT_EXCEPTIONS:
                alt = 0.0
            if not math.isfinite(alt) or alt <= 0:
                alt = 0.0

    if base > 0 and alt > 0:
        ratio = max(base, alt) / max(min(base, alt), 1e-8)
        # If candidate differs by 30x+, trust live/watchlist anchor.
        if ratio >= 30.0:
            return float(alt)
        return float(base)
    if alt > 0:
        return float(alt)
    return float(base)

def _stabilize_chart_depth(
    self: Any,
    symbol: str,
    interval: str,
    candidate: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Avoid replacing a healthy deep window with a transient tiny window."""
    cand = list(candidate or [])
    if not cand:
        return cand

    sym = self._ui_norm(symbol)
    iv = self._normalize_interval_token(interval)
    existing_all = list(self._bars_by_symbol.get(sym) or [])
    if not existing_all:
        return cand
    existing = [
        b for b in existing_all
        if self._normalize_interval_token(
            b.get("interval", iv),
            fallback=iv,
        ) == iv
    ]
    if not existing:
        return cand

    old_len = len(existing)
    new_len = len(cand)
    # Protect even medium-depth windows (for example 20-40 bars) from
    # being replaced by transient 1-5 bar snapshots.
    if old_len < 12 or new_len >= max(6, int(old_len * 0.45)):
        return cand

    merged = self._merge_bars(existing, cand, iv)
    if len(merged) >= max(new_len, int(old_len * 0.62)):
        out = merged
    else:
        out = existing

    self._debug_console(
        f"chart_depth_stabilize:{sym}:{iv}",
        (
            f"depth stabilization for {sym} {iv}: "
            f"new={new_len} old={old_len} final={len(out)}"
        ),
        min_gap_seconds=1.0,
        level="info",
    )
    return out

def _bar_bucket_epoch(self: Any, ts_raw: Any, interval: str) -> float:
    """Floor any timestamp to the interval bucket start (epoch seconds)."""
    epoch = self._ts_to_epoch(ts_raw)
    iv = self._normalize_interval_token(interval)
    if iv in ("1d", "1wk", "1mo"):
        # Use Shanghai trading date as bucket key to avoid UTC date aliasing.
        # Providers may timestamp daily bars at midnight Shanghai time which
        # is 16:00 UTC on the previous day -- naive UTC floor would map them
        # to the wrong date.
        try:
            from zoneinfo import ZoneInfo
            dt_val = datetime.fromtimestamp(float(epoch), tz=ZoneInfo("Asia/Shanghai"))
        except _APP_SOFT_EXCEPTIONS:
            dt_val = datetime.fromtimestamp(float(epoch))
        sh_midnight = dt_val.replace(hour=0, minute=0, second=0, microsecond=0)
        return float(sh_midnight.timestamp())
    step = float(max(1, self._interval_seconds(interval)))
    return float(int(epoch // step) * int(step))

def _bar_trading_date(self: Any, ts_raw: Any) -> object | None:
    """Best-effort Shanghai trading date for a timestamp-like value."""
    try:
        epoch = float(self._ts_to_epoch(ts_raw))
    except _APP_SOFT_EXCEPTIONS:
        return None
    try:
        from zoneinfo import ZoneInfo
        dt_val = datetime.fromtimestamp(epoch, tz=ZoneInfo("Asia/Shanghai"))
    except _APP_SOFT_EXCEPTIONS:
        try:
            dt_val = datetime.fromtimestamp(epoch)
        except _APP_SOFT_EXCEPTIONS:
            return None
    try:
        return dt_val.date()
    except _APP_SOFT_EXCEPTIONS:
        return None

def _is_intraday_day_boundary(
    self: Any,
    prev_ts_raw: Any,
    cur_ts_raw: Any,
    interval: str,
) -> bool:
    """True when two intraday bars fall on different Shanghai trading dates."""
    iv = self._normalize_interval_token(interval)
    if iv in ("1d", "1wk", "1mo"):
        return False
    prev_day = self._bar_trading_date(prev_ts_raw)
    cur_day = self._bar_trading_date(cur_ts_raw)
    if prev_day is None or cur_day is None:
        return False
    return bool(cur_day != prev_day)

def _shanghai_now(self: Any) -> datetime:
    """Current time in Asia/Shanghai when zoneinfo is available."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    except _APP_SOFT_EXCEPTIONS:
        return datetime.now()

def _is_cn_trading_day(self: Any, day_obj: Any) -> bool:
    """Best-effort CN trading-day check (weekday + optional holiday calendar)."""
    try:
        if day_obj.weekday() >= 5:
            return False
    except _APP_SOFT_EXCEPTIONS:
        return False

    try:
        from core.constants import is_trading_day
        return bool(is_trading_day(day_obj))
    except _APP_SOFT_EXCEPTIONS:
        return True

def _market_hours_text(self: Any) -> str:
    """Human-readable CN session hours."""
    t = CONFIG.trading
    return (
        f"{t.market_open_am.strftime('%H:%M')}-{t.market_close_am.strftime('%H:%M')}, "
        f"{t.market_open_pm.strftime('%H:%M')}-{t.market_close_pm.strftime('%H:%M')} CST"
    )

def _next_market_open(self: Any, now_sh: datetime | None = None) -> datetime | None:
    """Next CN market open timestamp in Shanghai time."""
    now_val = now_sh or self._shanghai_now()
    t = CONFIG.trading

    for days_ahead in range(0, 15):
        day_val = (now_val + timedelta(days=days_ahead)).date()
        if not self._is_cn_trading_day(day_val):
            continue

        open_am = now_val.replace(
            year=day_val.year,
            month=day_val.month,
            day=day_val.day,
            hour=t.market_open_am.hour,
            minute=t.market_open_am.minute,
            second=0,
            microsecond=0,
        )
        open_pm = now_val.replace(
            year=day_val.year,
            month=day_val.month,
            day=day_val.day,
            hour=t.market_open_pm.hour,
            minute=t.market_open_pm.minute,
            second=0,
            microsecond=0,
        )

        if days_ahead > 0:
            return open_am

        cur_time = now_val.time()
        if cur_time < t.market_open_am:
            return open_am
        if t.market_close_am < cur_time < t.market_open_pm:
            return open_pm
        if cur_time > t.market_close_pm:
            continue

    return None

def _is_market_session_timestamp(self: Any, ts_raw: Any, interval: str) -> bool:
    """True when timestamp falls inside CN trading session for intraday intervals."""
    iv = self._normalize_interval_token(interval)
    if iv in ("1d", "1wk", "1mo"):
        return True

    epoch = self._ts_to_epoch(ts_raw)
    try:
        from zoneinfo import ZoneInfo
        dt_val = datetime.fromtimestamp(float(epoch), tz=ZoneInfo("Asia/Shanghai"))
    except _APP_SOFT_EXCEPTIONS:
        dt_val = datetime.fromtimestamp(float(epoch))

    if not self._is_cn_trading_day(dt_val.date()):
        return False

    cur_time = dt_val.time()
    t = CONFIG.trading
    morning = t.market_open_am <= cur_time <= t.market_close_am
    # Include up to official close (15:00) so the live candle is visible
    # during the closing session. Auction spikes are handled by sanitize_ohlc.
    afternoon = t.market_open_pm <= cur_time <= t.market_close_pm
    return bool(morning or afternoon)

def _filter_bars_to_market_session(
    self: Any,
    bars: list[dict[str, Any]],
    interval: str,
) -> list[dict[str, Any]]:
    """Drop out-of-session intraday bars before chart rendering."""
    iv = self._normalize_interval_token(interval)
    if iv in ("1d", "1wk", "1mo"):
        return list(bars or [])

    out: list[dict[str, Any]] = []
    for b in (bars or []):
        ts_raw = b.get("_ts_epoch", b.get("timestamp", b.get("time")))
        if self._is_market_session_timestamp(ts_raw, iv):
            out.append(b)
    return out

def _bar_safety_caps(self: Any, interval: str) -> tuple[float, float]:
    """Return (max_jump_pct, max_range_pct) for bar sanitization.
    Values are intentionally conservative for intraday feeds.
    """
    iv = self._normalize_interval_token(interval)
    if iv == "1m":
        return 0.08, 0.006
    if iv == "5m":
        return 0.10, 0.012
    if iv in ("15m", "30m"):
        return 0.14, 0.020
    if iv in ("60m", "1h"):
        return 0.18, 0.040
    if iv == "1d":
        return 0.24, 0.24
    if iv in ("1wk", "1mo"):
        return 0.35, 0.40
    return 0.20, 0.15

def _synthetic_tick_jump_cap(self: Any, interval: str) -> float:
    """Stricter jump cap for tick-driven synthetic bar updates.
    Prevents stale or spiky quotes from creating giant intraday bodies.
    """
    iv = self._normalize_interval_token(interval)
    if iv == "1m":
        return 0.012
    if iv == "5m":
        return 0.018
    if iv in ("15m", "30m"):
        return 0.028
    if iv in ("60m", "1h"):
        return 0.045
    if iv in ("1d", "1wk", "1mo"):
        return 0.12
    return 0.03

def _sanitize_ohlc(
    self: Any,
    o: float,
    h: float,
    low: float,
    c: float,
    interval: str,
    ref_close: float | None = None,
) -> tuple[float, float, float, float] | None:
    """Normalize and clamp OHLC values to avoid malformed long candles
    from bad ticks/partial bars.
    """
    try:
        o = float(o or 0.0)
        h = float(h or 0.0)
        low = float(low or 0.0)
        c = float(c or 0.0)
    except _APP_SOFT_EXCEPTIONS:
        return None
    if not all(math.isfinite(v) for v in (o, h, low, c)):
        return None
    if c <= 0:
        return None

    ref = float(ref_close or 0.0)
    if not math.isfinite(ref) or ref <= 0:
        ref = 0.0

    if o <= 0:
        o = c
    if h <= 0:
        h = max(o, c)
    if low <= 0:
        low = min(o, c)
    if h < low:
        h, low = low, h

    jump_cap, range_cap = self._bar_safety_caps(interval)
    iv = self._normalize_interval_token(interval)
    if ref > 0:
        effective_range_cap = float(range_cap)
    else:
        bootstrap_cap = (
            0.30
            if iv in ("1d", "1wk", "1mo")
            else float(max(0.008, min(0.020, range_cap * 2.0)))
        )
        effective_range_cap = float(max(range_cap, bootstrap_cap))
    if ref > 0:
        jump = abs(c / ref - 1.0)
        if jump > jump_cap:
            return None

    # Keep malformed opens from inflating body/range caps.
    anchor = ref if ref > 0 else c
    if anchor <= 0:
        anchor = c
    max_body = float(anchor) * float(max(jump_cap * 1.25, effective_range_cap * 0.9))
    if max_body > 0 and abs(o - c) > max_body:
        if iv in ("1d", "1wk", "1mo"):
            return None
        if ref > 0 and abs(c / ref - 1.0) <= jump_cap:
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
        if iv in ("1d", "1wk", "1mo"):
            return None
        body = max(0.0, top - bot)
        if body > max_range:
            # Body this large is likely a corrupt open/close pair.
            o = c
            top = c
            bot = c
            body = 0.0
        # Intraday feed glitch guard: if provider injects day-range highs/lows
        # into minute bars, keep only compact wicks around the candle body.
        intraday = iv not in ("1d", "1wk", "1mo")
        if intraday:
            wick_each_cap = max(body * 0.85, float(anchor) * 0.0007)
            wick_each_cap = min(wick_each_cap, max_range * 0.18)
            # Keep total span within the allowed envelope so valid bars
            # are compacted instead of being dropped as malformed.
            wick_each_cap = min(
                wick_each_cap,
                max(0.0, (max_range - body) * 0.5),
            )
            h = min(h, top + wick_each_cap)
            low = max(low, bot - wick_each_cap)
        else:
            wick_allow = max(0.0, max_range - body)
            h = min(h, top + (wick_allow * 0.5))
            low = max(low, bot - (wick_allow * 0.5))
        if h < low:
            h, low = low, h

    # Intraday soft-span guard: keep wick span near candle body so
    # occasional provider spikes do not render as tall barcode lines.
    if iv not in ("1d", "1wk", "1mo") and anchor > 0:
        try:
            body_pct = abs(o - c) / float(anchor)
            span_pct = abs(h - low) / float(anchor)
        except _APP_SOFT_EXCEPTIONS:
            body_pct = 0.0
            span_pct = 0.0
        span_buffer_map = {
            "1m": 0.0022,
            "5m": 0.0032,
            "15m": 0.0048,
            "30m": 0.0048,
            "60m": 0.0075,
            "1h": 0.0075,
        }
        span_buffer = float(span_buffer_map.get(iv, 0.0045))
        soft_cap_pct = float(
            max(
                body_pct + 0.0012,
                min(float(effective_range_cap) * 0.85, body_pct + span_buffer),
            )
        )
        if span_pct > soft_cap_pct:
            top = max(o, c)
            bot = min(o, c)
            body = max(0.0, top - bot)
            target_span = float(anchor) * soft_cap_pct
            wick_each = max(0.0, (target_span - body) * 0.5)
            h = min(h, top + wick_each)
            low = max(low, bot - wick_each)
            if h < low:
                h, low = low, h

    o = min(max(o, low), h)
    c = min(max(c, low), h)

    # Final hard-stop: drop anything still outside allowed envelope.
    if anchor > 0 and (h - low) > (float(anchor) * float(effective_range_cap) * 1.05):
        return None

    return o, h, low, c

def _is_outlier_tick(
    self: Any, prev_price: float, new_price: float, interval: str = "1m"
) -> bool:
    """Guard against bad ticks creating abnormal long candles.
    Uses interval-aware thresholds to avoid rejecting valid fast moves.
    """
    prev = float(prev_price or 0.0)
    new = float(new_price or 0.0)
    if prev <= 0 or new <= 0:
        return False
    jump_cap, _ = self._bar_safety_caps(interval)
    jump_pct = abs(new / prev - 1.0)
    return jump_pct > float(jump_cap)

def _get_levels_dict(self: Any) -> dict[str, float] | None:
    """Get trading levels as dict."""
    if (
        not self.current_prediction
        or not hasattr(self.current_prediction, 'levels')
    ):
        return None

    levels = self.current_prediction.levels
    return {
        "stop_loss": getattr(levels, 'stop_loss', 0),
        "target_1": getattr(levels, 'target_1', 0),
        "target_2": getattr(levels, 'target_2', 0),
        "target_3": getattr(levels, 'target_3', 0),
    }

def _scrub_chart_bars(
    self: Any,
    bars: list[dict[str, Any]] | None,
    interval: str,
    *,
    symbol: str = "",
    anchor_price: float | None = None,
) -> list[dict[str, Any]]:
    """Prepare bars for charting and never fall back to unsanitized rows."""
    arr_in = list(bars or [])
    iv = self._normalize_interval_token(interval)
    arr_out = self._prepare_chart_bars_for_interval(
        arr_in,
        iv,
        symbol=symbol,
    )
    if arr_out:
        arr_out = self._rescale_chart_bars_to_anchor(
            arr_out,
            anchor_price=anchor_price,
            interval=iv,
            symbol=symbol,
        )
        sample = list(arr_out[-min(320, len(arr_out)):])
        if len(sample) >= 20:
            jump_cap, range_cap = self._bar_safety_caps(iv)
            intraday = iv not in ("1d", "1wk", "1mo")
            body_cap = float(max(range_cap * 3.5, 0.08 if intraday else 0.45))
            span_cap = float(max(range_cap * 5.0, 0.12 if intraday else 0.65))
            jump_guard = float(max(jump_cap * 2.8, 0.20 if intraday else 0.55))

            extreme = 0
            parsed = 0
            prev_close: float | None = None
            for row in sample:
                try:
                    o = float(row.get("open", 0) or 0)
                    h = float(row.get("high", 0) or 0)
                    low = float(row.get("low", 0) or 0)
                    c = float(row.get("close", 0) or 0)
                except _APP_SOFT_EXCEPTIONS:
                    continue
                if (
                    c <= 0
                    or not all(math.isfinite(v) for v in (o, h, low, c))
                ):
                    continue
                parsed += 1
                ref = float(prev_close if prev_close and prev_close > 0 else c)
                body = abs(o - c) / max(ref, 1e-8)
                span = abs(h - low) / max(ref, 1e-8)
                jump = (
                    abs(c / max(float(prev_close), 1e-8) - 1.0)
                    if prev_close and prev_close > 0
                    else 0.0
                )
                if body > body_cap or span > span_cap or jump > jump_guard:
                    extreme += 1
                prev_close = float(c)

            extreme_ratio = (
                float(extreme) / float(max(1, parsed))
                if parsed > 0
                else 0.0
            )
            if parsed >= 20 and extreme_ratio >= 0.08:
                recovered = self._recover_chart_bars_from_close(
                    arr_in,
                    interval=iv,
                    symbol=symbol,
                    anchor_price=anchor_price,
                )
                if recovered:
                    sym = self._ui_norm(symbol)
                    self._debug_console(
                        f"chart_scrub_recover:{sym or 'active'}:{iv}",
                        (
                            f"switched to close-based recovery for {sym or '--'} {iv}: "
                            f"extreme={extreme}/{parsed} ratio={extreme_ratio:.1%}"
                        ),
                        min_gap_seconds=0.8,
                        level="warning",
                    )
                    arr_out = recovered
    if arr_in and not arr_out:
        log.warning(
            f"Chart scrub rejected all {len(arr_in)} bars for {self._ui_norm(symbol) or '--'} {iv}"
        )
        recovered = self._recover_chart_bars_from_close(
            arr_in,
            interval=iv,
            symbol=symbol,
            anchor_price=anchor_price,
        )
        if recovered:
            sym = self._ui_norm(symbol)
            self._debug_console(
                f"chart_scrub_recover_empty:{sym or 'active'}:{iv}",
                (
                    f"chart scrub empty -> recovered bars: symbol={sym or '--'} "
                    f"iv={iv} recovered={len(recovered)} raw={len(arr_in)}"
                ),
                min_gap_seconds=0.8,
                level="warning",
            )
            return recovered
        sym = self._ui_norm(symbol)
        self._debug_console(
            f"chart_scrub_empty:{sym or 'active'}:{iv}",
            (
                f"chart scrub rejected bars: symbol={sym or '--'} "
                f"iv={iv} raw={len(arr_in)} kept=0"
            ),
            min_gap_seconds=1.0,
            level="warning",
        )
        return []
    return arr_out

def _rescale_chart_bars_to_anchor(
    self: Any,
    bars: list[dict[str, Any]],
    *,
    anchor_price: float | None,
    interval: str,
    symbol: str = "",
) -> list[dict[str, Any]]:
    """Repair obvious price-scale mismatches (e.g., 1.5 vs 1500) so bars
    are not fully dropped by jump filters.
    """
    arr = list(bars or [])
    if not arr:
        return []
    try:
        anchor = float(anchor_price or 0.0)
    except _APP_SOFT_EXCEPTIONS:
        anchor = 0.0
    if not math.isfinite(anchor) or anchor <= 0:
        return arr

    closes: list[float] = []
    for row in arr:
        try:
            c = float(row.get("close", 0) or 0)
        except _APP_SOFT_EXCEPTIONS:
            c = 0.0
        if c > 0 and math.isfinite(c):
            closes.append(c)
    if len(closes) < 5:
        return arr

    med = float(median(closes[-min(80, len(closes)):]))
    if med <= 0 or not math.isfinite(med):
        return arr

    raw_ratio = anchor / med
    if 0.2 <= raw_ratio <= 5.0:
        return arr

    candidates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best_scale = 1.0
    best_err = float("inf")
    for s in candidates:
        try:
            ratio = (med * float(s)) / anchor
            if ratio <= 0 or not math.isfinite(ratio):
                continue
            err = abs(math.log(ratio))
            if err < best_err:
                best_err = err
                best_scale = float(s)
        except _APP_SOFT_EXCEPTIONS:
            continue

    scaled_ratio = (med * best_scale) / anchor if anchor > 0 else 1.0
    if not (0.2 <= scaled_ratio <= 5.0):
        return arr
    if abs(best_scale - 1.0) < 1e-9:
        return arr

    out: list[dict[str, Any]] = []
    for row in arr:
        item = dict(row)
        for key in ("open", "high", "low", "close"):
            try:
                v = float(item.get(key, 0) or 0)
            except _APP_SOFT_EXCEPTIONS:
                v = 0.0
            if v > 0 and math.isfinite(v):
                item[key] = float(v * best_scale)
        out.append(item)

    iv = self._normalize_interval_token(interval)
    sym = self._ui_norm(symbol)
    self._debug_console(
        f"chart_scale_fix:{sym or 'active'}:{iv}",
        (
            f"applied scale fix x{best_scale:g} for {sym or '--'} {iv}: "
            f"median={med:.6f} anchor={anchor:.6f}"
        ),
        min_gap_seconds=1.0,
        level="info",
    )
    return out

def _recover_chart_bars_from_close(
    self: Any,
    bars: list[dict[str, Any]],
    *,
    interval: str,
    symbol: str = "",
    anchor_price: float | None = None,
) -> list[dict[str, Any]]:
    """Minimal recovery path when strict scrub drops all bars.
    Builds stable OHLC from close/prev-close so chart remains usable.
    """
    iv = self._normalize_interval_token(interval)
    def _build(enforce_session: bool) -> list[dict[str, Any]]:
        merged: dict[int, dict[str, Any]] = {}
        prev_close: float | None = None
        prev_epoch: float | None = None
        for row in list(bars or []):
            if not isinstance(row, dict):
                continue
            row_iv = self._normalize_interval_token(
                row.get("interval", iv),
                fallback=iv,
            )
            if row_iv != iv:
                continue
            epoch = self._bar_bucket_epoch(
                row.get("_ts_epoch", row.get("timestamp")),
                iv,
            )
            if enforce_session and (not self._is_market_session_timestamp(epoch, iv)):
                continue
            ref_close = prev_close
            if (
                prev_epoch is not None
                and self._is_intraday_day_boundary(prev_epoch, epoch, iv)
            ):
                ref_close = None
            try:
                c = float(
                    row.get("close", row.get("price", 0)) or 0
                )
            except _APP_SOFT_EXCEPTIONS:
                c = 0.0
            if c <= 0 or not math.isfinite(c):
                continue

            try:
                o = float(row.get("open", 0) or 0)
            except _APP_SOFT_EXCEPTIONS:
                o = 0.0
            if o <= 0:
                if iv in ("1d", "1wk", "1mo"):
                    o = c
                elif ref_close and ref_close > 0:
                    o = float(ref_close)
                else:
                    o = c

            try:
                h = float(row.get("high", max(o, c)) or max(o, c))
            except _APP_SOFT_EXCEPTIONS:
                h = max(o, c)
            try:
                low = float(row.get("low", min(o, c)) or min(o, c))
            except _APP_SOFT_EXCEPTIONS:
                low = min(o, c)
            if iv not in ("1d", "1wk", "1mo"):
                # Recovery mode: clamp wicks to prevent vendor day-range
                # highs/lows from inflating minute bars, but preserve
                # partial wick data for natural-looking candles.
                top = max(o, c)
                bot = min(o, c)
                body = top - bot
                max_wick = max(body * 1.5, top * 0.003)
                h = min(h, top + max_wick)
                low = max(low, bot - max_wick)
                h = max(h, top)
                low = min(low, bot)

            s = self._sanitize_ohlc(
                o,
                h,
                low,
                c,
                interval=iv,
                ref_close=ref_close,
            )
            if s is None:
                continue
            o, h, low, c = s

            key = int(epoch)
            item = {
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "_ts_epoch": float(epoch),
                "timestamp": self._epoch_to_iso(epoch),
                "final": bool(row.get("final", True)),
                "interval": iv,
            }
            existing = merged.get(key)
            if existing is None:
                merged[key] = item
            else:
                if bool(item.get("final", True)) and not bool(existing.get("final", True)):
                    merged[key] = item
            prev_close = c
            prev_epoch = float(epoch)

        out_local = list(merged.values())
        out_local.sort(key=lambda x: float(x.get("_ts_epoch", 0.0)))
        return out_local[-self._history_window_bars(iv):]

    out = _build(enforce_session=True)
    if not out:
        out = _build(enforce_session=False)
        if out:
            sym = self._ui_norm(symbol)
            self._debug_console(
                f"chart_recover_lenient:{sym or 'active'}:{iv}",
                (
                    f"lenient timestamp recovery enabled for {sym or '--'} {iv}: "
                    f"bars={len(out)}"
                ),
                min_gap_seconds=1.0,
                level="warning",
            )

    out = self._rescale_chart_bars_to_anchor(
        out,
        anchor_price=anchor_price,
        interval=iv,
        symbol=symbol,
    )
    if out:
        sym = self._ui_norm(symbol)
        self._debug_console(
            f"chart_recover:{sym or 'active'}:{iv}",
            (
                f"recovered chart bars from close-only path: "
                f"symbol={sym or '--'} iv={iv} bars={len(out)}"
            ),
            min_gap_seconds=1.0,
            level="warning",
        )
    return out
