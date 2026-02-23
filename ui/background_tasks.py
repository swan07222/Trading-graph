from __future__ import annotations

import threading
import time
from importlib import import_module
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)
_MONITOR_RECOVERABLE_EXCEPTIONS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _lazy_get(module: str, name: str) -> Any:
    return getattr(import_module(module), name)


def validate_stock_code(code: str) -> bool:
    """Validate that a stock code is a valid 6-digit Chinese stock code."""
    if not code:
        return False
    digits = "".join(c for c in str(code).strip() if c.isdigit())
    if len(digits) != 6:
        return False
    valid_prefixes = (
        "000",
        "001",
        "002",
        "003",
        "300",
        "301",
        "600",
        "601",
        "603",
        "605",
        "688",
        "83",
        "87",
        "43",
    )
    return digits.startswith(valid_prefixes)


def normalize_stock_code(text: str) -> str:
    """Normalize stock code: strip prefixes/suffixes, keep digits, zero-pad."""
    if not text:
        return ""
    normalized = str(text).strip()
    for prefix in ("sh", "sz", "SH", "SZ", "bj", "BJ"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    for suffix in (".SS", ".SZ", ".BJ"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    normalized = "".join(c for c in normalized if c.isdigit())
    return normalized.zfill(6) if normalized else ""


def sanitize_watch_list(
    codes: list[str] | tuple[str, ...] | str | None,
    max_size: int = 50,
) -> list[str]:
    """Normalize, validate, de-duplicate, and cap watchlist symbols."""
    if not codes:
        return []
    if isinstance(codes, str):
        iterable = [codes]
    else:
        iterable = list(codes)

    cap = max(1, int(max_size or 50))
    out: list[str] = []
    seen: set[str] = set()

    for raw in iterable:
        code = normalize_stock_code(str(raw or ""))
        if not code or not validate_stock_code(code):
            continue
        if code in seen:
            continue
        seen.add(code)
        out.append(code)
        if len(out) >= cap:
            break

    return out


def collect_live_readiness_failures() -> list[str]:
    """Return failed required institutional controls for LIVE mode.
    Empty list means readiness checks passed or could not be evaluated.
    """
    try:
        from utils.institutional import collect_institutional_readiness

        report = collect_institutional_readiness()
    except _MONITOR_RECOVERABLE_EXCEPTIONS:
        return []

    if bool(report.get("pass", False)):
        return []

    failed = report.get("failed_required_controls", [])
    if not isinstance(failed, list):
        return ["institutional_readiness_unknown"]
    out = [str(x).strip() for x in failed if str(x).strip()]
    return out


class RealTimeMonitor(QThread):
    """Real-time market monitoring thread.
    Continuously checks for trading signals using the predictor.
    """

    MAX_WATCHLIST_SIZE = 50

    signal_detected = pyqtSignal(object)
    price_updated = pyqtSignal(str, float)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(
        self,
        predictor: Any,
        watch_list: list[str],
        interval: str = "1m",
        forecast_minutes: int = 30,
        lookback_bars: int = 1680,
        history_allow_online: bool = True,
    ):
        super().__init__()
        self.predictor = predictor
        raw_watch = list(watch_list or [])
        self.watch_list = sanitize_watch_list(
            raw_watch,
            max_size=self.MAX_WATCHLIST_SIZE,
        )
        self.running = False
        self._stop_event = threading.Event()

        if len(raw_watch) > len(self.watch_list):
            log.warning(
                "Watchlist sanitized from %d to %d symbols",
                len(raw_watch),
                len(self.watch_list),
            )

        try:
            cfg_interval = int(
                getattr(getattr(CONFIG, "auto_trade", None), "scan_interval_seconds", 30)
                or 30
            )
        except (AttributeError, TypeError, ValueError):
            cfg_interval = 30
        self.scan_interval = max(8, min(30, int(cfg_interval)))
        self.data_interval = str(interval).lower()
        self.forecast_minutes = int(forecast_minutes)
        self.lookback_bars = int(lookback_bars)
        self.history_allow_online = bool(history_allow_online)
        intraday_tokens = {"1m", "2m", "3m", "5m", "15m", "30m", "60m", "1h"}
        if self.data_interval in intraday_tokens:
            self._quick_lookback_bars = max(180, min(self.lookback_bars, 720))
        else:
            self._quick_lookback_bars = max(120, min(self.lookback_bars, 520))

        self._backoff = 1
        self._max_backoff = 60
        self._last_full_emit: dict[str, tuple[str, int, int]] = {}
        self._full_emit_cooldown_seconds: float = 6.0

    def run(self) -> None:
        self.running = True
        self._stop_event.clear()
        self._backoff = 1

        self.status_changed.emit("Monitoring started")

        while self.running and not self._stop_event.is_set():
            loop_start = time.time()

            if not CONFIG.is_market_open():
                self.status_changed.emit("Market closed - waiting for open")
                idle_wait = max(5, min(60, int(self.scan_interval)))
                for _ in range(idle_wait):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)
                continue

            strong: list[Any] = []
            try:
                preds = self.predictor.predict_quick_batch(
                    self.watch_list,
                    use_realtime_price=True,
                    interval=self.data_interval,
                    lookback_bars=self._quick_lookback_bars,
                    history_allow_online=self.history_allow_online,
                )

                for pred in preds:
                    if hasattr(pred, "current_price") and pred.current_price > 0:
                        self.price_updated.emit(pred.stock_code, pred.current_price)

                Signal = _lazy_get("models.predictor", "Signal")

                strong = [
                    pred
                    for pred in preds
                    if hasattr(pred, "signal")
                    and pred.signal
                    in [
                        Signal.STRONG_BUY,
                        Signal.STRONG_SELL,
                        Signal.BUY,
                        Signal.SELL,
                    ]
                    and hasattr(pred, "confidence")
                    and pred.confidence >= CONFIG.MIN_CONFIDENCE
                ]

                strong.sort(key=lambda item: item.confidence, reverse=True)
                strong = strong[:3]

                for pred in strong:
                    if self._stop_event.is_set():
                        break

                    try:
                        now_key_ts = int(time.time())
                        symbol = str(getattr(pred, "stock_code", "") or "").strip()
                        sig_name = str(
                            getattr(
                                getattr(pred, "signal", None),
                                "value",
                                getattr(pred, "signal", ""),
                            )
                        )
                        conf_bp = int(float(getattr(pred, "confidence", 0.0) or 0.0) * 10000.0)
                        quick_key = (
                            sig_name,
                            conf_bp,
                            now_key_ts // int(self._full_emit_cooldown_seconds),
                        )
                        prev_key = self._last_full_emit.get(symbol)
                        if prev_key == quick_key:
                            continue

                        full = self.predictor.predict(
                            pred.stock_code,
                            use_realtime_price=True,
                            interval=self.data_interval,
                            forecast_minutes=self.forecast_minutes,
                            lookback_bars=self.lookback_bars,
                            skip_cache=True,
                            history_allow_online=self.history_allow_online,
                        )
                        full_signal = getattr(full, "signal", None)
                        if full_signal == Signal.HOLD:
                            continue
                        self._last_full_emit[symbol] = quick_key
                        self.signal_detected.emit(full)
                    except _MONITOR_RECOVERABLE_EXCEPTIONS as e:
                        log.warning("Full prediction failed for %s: %s", pred.stock_code, e)

                self._backoff = 1
                self.status_changed.emit(f"Scanned {len(preds)} stocks, {len(strong)} signals")

            except _MONITOR_RECOVERABLE_EXCEPTIONS as e:
                error_msg = str(e)
                self.error_occurred.emit(error_msg)
                log.warning("Monitor error: %s", error_msg)

                sleep_s = min(self._max_backoff, self._backoff)
                self._backoff = min(self._max_backoff, self._backoff * 2)

                self.status_changed.emit(f"Error, retrying in {sleep_s}s")

                for _ in range(sleep_s):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)
                continue

            elapsed = time.time() - loop_start
            target_interval = self.scan_interval
            if strong:
                target_interval = max(6, int(self.scan_interval * 0.66))
            remaining = max(0.0, float(target_interval) - float(elapsed))

            for _ in range(int(remaining)):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

        self.status_changed.emit("Monitoring stopped")

    def stop(self) -> None:
        """Stop monitoring gracefully."""
        self.running = False
        self._stop_event.set()

    def update_config(
        self,
        interval: str | None = None,
        forecast_minutes: int | None = None,
        lookback_bars: int | None = None,
    ) -> None:
        """Update monitoring configuration."""
        if interval:
            self.data_interval = str(interval).lower()
        if forecast_minutes:
            self.forecast_minutes = int(forecast_minutes)
        if lookback_bars:
            self.lookback_bars = int(lookback_bars)


class WorkerThread(QThread):
    """Generic worker thread for background tasks with timeout support."""

    result = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, func: Any, *args: Any, timeout_seconds: float = 300, **kwargs: Any):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._cancelled = False
        self._timeout = timeout_seconds

    def run(self) -> None:
        try:
            if self._cancelled or self.isInterruptionRequested():
                return

            out = self.func(*self.args, **self.kwargs)

            if self._cancelled or self.isInterruptionRequested():
                return
            self.result.emit(out)

        except _MONITOR_RECOVERABLE_EXCEPTIONS as e:
            if not self._cancelled:
                self.error.emit(str(e))

    def cancel(self) -> None:
        """Cancel the worker."""
        self._cancelled = True
        try:
            self.requestInterruption()
        except RuntimeError:
            pass

