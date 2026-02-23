# data/bar_aggregator.py
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from enum import Enum

from config.settings import CONFIG
from core.events import EVENT_BUS, BarEvent
from utils.logger import get_logger

log = get_logger(__name__)
_BAR_SOFT_EXCEPTIONS = (
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
class VolumeMode(Enum):
    """Volume interpretation mode."""
    CUMULATIVE = "cumulative"
    DELTA = "delta"

# Bar aggregator - FIXED to emit partial bars

class BarAggregator:
    """
    Aggregates ticks into OHLCV bars with configurable interval.

    FIXED: Now emits PARTIAL bars on every tick so the chart
    updates in real-time, not just on bar boundaries.
    """

    def __init__(
        self,
        interval_seconds: int = 60,
        volume_mode: VolumeMode = VolumeMode.CUMULATIVE
    ):
        self._interval = max(1, int(interval_seconds))
        self._volume_mode = volume_mode
        self._current_bars: dict[str, dict] = {}
        self._callbacks: list[Callable] = []
        self._lock = threading.RLock()
        self._last_partial_emit_ts: dict[str, float] = {}
        self._min_partial_emit_interval_s: float = 0.20

    @staticmethod
    def _to_shanghai_naive(ts_raw) -> datetime:
        """
        Normalize quote timestamps to naive Asia/Shanghai time.
        """
        try:
            from zoneinfo import ZoneInfo

            sh_tz = ZoneInfo("Asia/Shanghai")
        except _BAR_SOFT_EXCEPTIONS:
            sh_tz = timezone.utc

        if ts_raw is None:
            return datetime.now(tz=sh_tz).replace(tzinfo=None)

        if isinstance(ts_raw, datetime):
            dt = ts_raw
        else:
            text = str(ts_raw).strip()
            if not text:
                return datetime.now(tz=sh_tz).replace(tzinfo=None)
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except _BAR_SOFT_EXCEPTIONS:
                return datetime.now(tz=sh_tz).replace(tzinfo=None)

        try:
            if dt.tzinfo is None:
                # Treat naive provider timestamps as market-local time.
                dt = dt.replace(tzinfo=sh_tz)
            dt = dt.astimezone(sh_tz)
            return dt.replace(tzinfo=None)
        except _BAR_SOFT_EXCEPTIONS:
            try:
                return dt.replace(tzinfo=None)
            except _BAR_SOFT_EXCEPTIONS:
                return datetime.now(tz=sh_tz).replace(tzinfo=None)

    @staticmethod
    def _is_cn_session_time(ts_val: datetime) -> bool:
        """
        Check whether a timestamp is within CN A-share regular trading session.
        """
        if not isinstance(ts_val, datetime):
            return False
        # Monday..Friday
        if ts_val.weekday() >= 5:
            return False
        hhmm = (ts_val.hour * 100) + ts_val.minute
        morning = 930 <= hhmm <= 1130
        # Exclude closing call auction (14:57-15:00) to match chart filter.
        afternoon = 1300 <= hhmm < 1457
        return bool(morning or afternoon)

    def add_callback(self, callback: Callable):
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def set_volume_mode(self, mode: VolumeMode):
        """Change volume interpretation mode."""
        with self._lock:
            self._volume_mode = mode

    def on_tick(self, quote):
        """
        Process incoming tick/quote.

        FIXED: Now emits partial bar on EVERY tick for real-time updates.
        """
        symbol = getattr(quote, "code", None)
        if not symbol:
            return

        ts = self._to_shanghai_naive(getattr(quote, "timestamp", None))
        try:
            quote.timestamp = ts
        except _BAR_SOFT_EXCEPTIONS as exc:
            log.debug("Suppressed exception in data/feeds.py", exc_info=exc)
        px = float(getattr(quote, "price", 0) or 0)
        if px <= 0:
            return

        with self._lock:
            # Ignore off-session ticks, but flush any pending live bar first.
            if not self._is_cn_session_time(ts):
                stale = self._current_bars.pop(symbol, None)
                if stale is not None:
                    try:
                        self._emit_bar(symbol, stale, final=True)
                    except _BAR_SOFT_EXCEPTIONS as exc:
                        log.debug("Suppressed exception in data/feeds.py", exc_info=exc)
                self._last_partial_emit_ts.pop(symbol, None)
                return

            if symbol not in self._current_bars:
                self._current_bars[symbol] = self._new_bar(quote)

            bar = self._current_bars[symbol]

            # Day boundary: finalize old day bar before touching current tick.
            if (
                bar.get("session_date")
                and bar["session_date"] != ts.date()
            ):
                self._emit_bar(symbol, bar, final=True)
                self._current_bars[symbol] = self._new_bar(quote)
                bar = self._current_bars[symbol]

            bar_end = bar["timestamp"] + timedelta(seconds=self._interval)
            if ts >= bar_end:
                # CRITICAL: roll first, then apply tick to new bucket.
                self._emit_bar(symbol, bar, final=True)
                self._current_bars[symbol] = self._new_bar(quote)
                bar = self._current_bars[symbol]
                self._last_partial_emit_ts.pop(symbol, None)

            bar["high"] = max(float(bar["high"]), px)
            bar["low"] = min(float(bar["low"]), px)
            bar["close"] = px

            self._update_volume(bar, quote)

            # Emit partial bar for real-time chart updates.
            now_ts = time.monotonic()
            last_emit = float(self._last_partial_emit_ts.get(symbol, 0.0))
            if (now_ts - last_emit) >= self._min_partial_emit_interval_s:
                self._emit_bar(symbol, bar, final=False)
                self._last_partial_emit_ts[symbol] = now_ts

    def _update_volume(self, bar: dict, quote):
        """Update bar volume based on configured mode."""
        raw_vol = getattr(quote, "volume", None)
        if raw_vol is None:
            return

        try:
            vol_value = int(float(raw_vol))
        except (ValueError, TypeError):
            return

        if self._volume_mode == VolumeMode.CUMULATIVE:
            last_cum = int(bar.get("last_cum_vol", 0) or 0)

            if vol_value < last_cum:
                delta = 0
            else:
                delta = vol_value - last_cum

            bar["volume"] += max(delta, 0)
            bar["last_cum_vol"] = vol_value

        elif self._volume_mode == VolumeMode.DELTA:
            bar["volume"] += max(vol_value, 0)

    def _new_bar(self, quote) -> dict:
        """Create a new bar with proper time alignment."""
        ts = self._to_shanghai_naive(getattr(quote, "timestamp", None))

        total_seconds = ts.hour * 3600 + ts.minute * 60 + ts.second
        remainder = total_seconds % max(self._interval, 1)
        bar_start = ts - timedelta(
            seconds=remainder, microseconds=ts.microsecond
        )

        initial_vol = 0
        if self._volume_mode == VolumeMode.CUMULATIVE:
            raw_vol = getattr(quote, "volume", None)
            if raw_vol is not None:
                try:
                    initial_vol = int(float(raw_vol))
                except (ValueError, TypeError):
                    initial_vol = 0

        return {
            "timestamp": bar_start,
            "open": float(getattr(quote, "price", 0)),
            "high": float(getattr(quote, "price", 0)),
            "low": float(getattr(quote, "price", 0)),
            "close": float(getattr(quote, "price", 0)),
            "volume": 0,
            "last_cum_vol": initial_vol,
            "session_date": bar_start.date(),
        }

    @staticmethod
    def _interval_label(interval_seconds: int) -> str:
        """Canonical interval token used by UI/cache layers."""
        sec = max(1, int(interval_seconds))
        known = {
            60: "1m",
            300: "5m",
            900: "15m",
            1800: "30m",
            3600: "60m",
            86400: "1d",
        }
        if sec in known:
            return known[sec]
        if sec % 60 == 0:
            mins = int(sec // 60)
            return f"{mins}m"
        return f"{sec}s"

    def set_interval(self, interval_seconds: int):
        """Change bar interval; clears partial bars."""
        with self._lock:
            self._interval = max(1, int(interval_seconds))
            self._current_bars.clear()
            self._last_partial_emit_ts.clear()

    def _emit_bar(self, symbol: str, bar: dict, final: bool = True):
        """
        Emit bar to callbacks.

        Args:
            symbol: Stock symbol
            bar: Bar data dict
            final: If True, this is a completed bar. If False, partial/live bar.
        """
        bar_copy = dict(bar)
        interval_label = self._interval_label(self._interval)
        bar_copy["interval"] = interval_label
        bar_copy["interval_seconds"] = int(self._interval)
        bar_copy["final"] = final

        # Publish event only for final bars (to avoid spamming event bus)
        if final:
            EVENT_BUS.publish(
                BarEvent(
                    symbol=symbol,
                    open=bar["open"],
                    high=bar["high"],
                    low=bar["low"],
                    close=bar["close"],
                    volume=bar["volume"],
                    timestamp=bar["timestamp"],
                )
            )

            try:
                from data.session_cache import get_session_bar_cache

                cache_bar = dict(bar_copy)
                cache_bar["source"] = str(
                    cache_bar.get("source", "") or "tencent_rt"
                )
                get_session_bar_cache().append_bar(
                    symbol,
                    interval_label,
                    cache_bar,
                )
            except _BAR_SOFT_EXCEPTIONS as e:
                log.debug(f"Bar session persist failed for {symbol}: {e}")

            should_persist_db = True
            if should_persist_db:
                try:
                    import pandas as pd

                    from data.database import get_database

                    db = get_database()
                    df = pd.DataFrame(
                        [{
                            "open": float(bar["open"]),
                            "high": float(bar["high"]),
                            "low": float(bar["low"]),
                            "close": float(bar["close"]),
                            "volume": int(bar["volume"]),
                            "amount": 0.0,
                        }],
                        index=pd.DatetimeIndex([bar["timestamp"]]),
                    )

                    db.upsert_intraday_bars(symbol, interval_label, df)
                except ImportError:
                    pass
                except _BAR_SOFT_EXCEPTIONS as e:
                    log.debug(f"Bar DB persist failed for {symbol}: {e}")

        with self._lock:
            callbacks = self._callbacks.copy()

        for cb in callbacks:
            try:
                cb(symbol, bar_copy)
            except _BAR_SOFT_EXCEPTIONS as e:
                log.warning(f"Bar callback error: {e}")


