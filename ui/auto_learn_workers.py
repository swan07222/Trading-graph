from __future__ import annotations

import os
import threading
import traceback
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from utils.logger import get_logger

log = get_logger(__name__)

_WORKER_RECOVERABLE_EXCEPTIONS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_SUPPORTED_TRAIN_INTERVALS = frozenset(
    {"1m", "2m", "3m", "5m", "15m", "30m", "60m", "1h", "1d"}
)
_LOCKED_TRAIN_INTERVAL = "1m"


def _force_china_direct_network_mode() -> None:
    """Force discovery/training workers to run in China-direct mode."""
    os.environ["TRADING_CHINA_DIRECT"] = "1"
    os.environ["TRADING_VPN"] = "0"
    try:
        from core.network import invalidate_network_cache

        invalidate_network_cache()
    except Exception:
        # Keep worker startup resilient even if network module is unavailable.
        pass


def _get_cancellation_token() -> Any:
    """Lazy import CancellationToken."""
    from utils.cancellation import CancellationToken

    return CancellationToken()


def _get_auto_learner() -> Any | None:
    """Lazy import AutoLearner/ContinuousLearner.
    Returns the class, not an instance.
    """
    try:
        from models.auto_learner import AutoLearner

        return AutoLearner
    except ImportError:
        pass

    try:
        from models.auto_learner import ContinuousLearner

        return ContinuousLearner
    except ImportError:
        pass

    return None


def _safe_int(value: Any, default: int, minimum: int = 1, maximum: int = 1_000_000) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    return max(int(minimum), min(int(maximum), int(out)))


def normalize_training_interval(raw: Any, default: str = "1m") -> str:
    iv = str(raw or default).strip().lower()
    aliases = {"h1": "1h", "d1": "1d"}
    iv = aliases.get(iv, iv)
    if iv in _SUPPORTED_TRAIN_INTERVALS:
        return iv
    return str(default).strip().lower() or "1m"


class _BaseLearnWorker(QThread):
    progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str, str)
    finished_result = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, config: dict[str, Any] | None) -> None:
        super().__init__()
        self.config = dict(config or {})
        self.running = False
        self.token = _get_cancellation_token()
        self._learner: Any | None = None
        self._error_flag = False
        self._error_message = ""

    def _attach_progress_callback(self, results: dict[str, Any], default_error: str) -> None:
        if self._learner is None:
            return

        def on_progress(p: Any) -> None:
            if not self.running:
                return
            try:
                percent = int(max(0, min(100, getattr(p, "progress", 0))))
                message = str(getattr(p, "message", "") or getattr(p, "stage", ""))
                stage = str(getattr(p, "stage", "") or "")
                if stage == "error":
                    self._error_flag = True
                    self._error_message = message or default_error
                
                # FIX: Emit progress and detailed log messages
                self.progress.emit(percent, message)
                
                # Emit detailed log with stage prefix
                log_msg = f"{stage}: {message}" if stage else message
                if log_msg.strip():
                    self.log_message.emit(log_msg, "info")
                
                results["discovered"] = int(getattr(p, "stocks_found", 0) or 0)
                processed_direct = int(getattr(p, "stocks_processed", 0) or 0)
                processed_alt = int(getattr(p, "processed_count", 0) or 0)
                results["processed"] = max(processed_direct, processed_alt)
                results["accuracy"] = float(getattr(p, "validation_accuracy", 0.0) or 0.0)
            except _WORKER_RECOVERABLE_EXCEPTIONS as exc:
                log.debug("Progress callback parsing failed: %s", exc)

        if hasattr(self._learner, "add_callback"):
            self._learner.add_callback(on_progress)

    def _start_and_wait(
        self,
        start_fn: Callable[..., Any],
        kwargs: dict[str, Any],
    ) -> None:
        start_fn(**kwargs)

        if self._learner is None:
            return

        worker_thread = getattr(self._learner, "_thread", None)
        if isinstance(worker_thread, threading.Thread):
            while self.running and not self.token.is_cancelled and worker_thread.is_alive():
                worker_thread.join(timeout=0.2)
            return

        while self.running and not self.token.is_cancelled:
            progress = getattr(self._learner, "progress", None)
            is_running = bool(getattr(progress, "is_running", False))
            if not is_running:
                break
            self.msleep(200)

    def _stop_learner(self) -> None:
        if self._learner is None:
            return
        try:
            stop_fn = getattr(self._learner, "stop", None)
            if callable(stop_fn):
                try:
                    stop_fn(join_timeout=6.0)
                except TypeError:
                    stop_fn()
        except _WORKER_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Learner stop failed: %s", exc)
        finally:
            self._learner = None

    def stop(self) -> None:
        self.running = False
        self.token.cancel()


class AutoLearnWorker(_BaseLearnWorker):
    """Worker thread for auto-learning random stock rotation."""

    def run(self) -> None:
        self.running = True
        results: dict[str, Any] = {
            "discovered": 0,
            "processed": 0,
            "samples": 0,
            "accuracy": 0.0,
        }

        try:
            _force_china_direct_network_mode()
            self.log_message.emit(
                "China-direct network mode enabled for Auto Train GM discovery.",
                "info",
            )
            learner_class = _get_auto_learner()
            if learner_class is None:
                self.error_occurred.emit(
                    "AutoLearner/ContinuousLearner not found. "
                    "Ensure models/auto_learner.py exists."
                )
                return

            self._learner = learner_class()
            self._attach_progress_callback(results, default_error="Auto-learning failed")

            mode = str(self.config.get("mode", "full"))
            requested_interval = normalize_training_interval(
                self.config.get("interval", _LOCKED_TRAIN_INTERVAL)
            )
            interval = _LOCKED_TRAIN_INTERVAL
            if requested_interval != interval:
                self.log_message.emit(
                    f"Interval {requested_interval} overridden to {interval}",
                    "warning",
                )
            horizon = _safe_int(self.config.get("horizon", 30), default=30, minimum=1, maximum=500)
            lookback_raw = self.config.get("lookback_bars")
            lookback_bars = (
                _safe_int(lookback_raw, default=10080, minimum=120, maximum=500_000)
                if lookback_raw is not None
                else None
            )

            learner_kwargs: dict[str, Any] = {
                "mode": mode,
                "max_stocks": _safe_int(self.config.get("max_stocks", 200), default=200, minimum=10),
                "epochs_per_cycle": _safe_int(self.config.get("epochs", 10), default=10, minimum=1),
                "min_market_cap": 10,
                "include_all_markets": True,
                "continuous": True,
                "learning_while_trading": True,
                "interval": interval,
                "prediction_horizon": horizon,
                "cycle_interval_seconds": _safe_int(
                    self.config.get("cycle_interval_seconds", 60),
                    default=60,
                    minimum=30,
                    maximum=86_400,
                ),
                "incremental": bool(self.config.get("incremental", True)),
                "priority_stock_codes": list(self.config.get("priority_stock_codes", []) or []),
            }
            if lookback_bars is not None:
                learner_kwargs["lookback_bars"] = lookback_bars

            self.log_message.emit(
                (
                    f"Auto-learning started ({interval}, horizon={horizon}, "
                    f"max_stocks={learner_kwargs['max_stocks']})"
                ),
                "success",
            )

            self._start_and_wait(self._learner.start, learner_kwargs)
            self._stop_learner()

            if self.token.is_cancelled or not self.running:
                results["status"] = "stopped"
            elif self._error_flag:
                results["status"] = "error"
                results["error"] = self._error_message or "Auto-learning failed"
            else:
                results["status"] = "ok"
            self.finished_result.emit(results)

        except _WORKER_RECOVERABLE_EXCEPTIONS as exc:
            error_msg = str(exc)
            log.error("AutoLearnWorker error: %s", error_msg)
            log.debug(traceback.format_exc())
            if self.running:
                self.error_occurred.emit(error_msg)


class TargetedLearnWorker(_BaseLearnWorker):
    """Worker thread for targeted training on user-selected stocks."""

    def run(self) -> None:
        self.running = True
        results: dict[str, Any] = {
            "discovered": 0,
            "processed": 0,
            "samples": 0,
            "accuracy": 0.0,
            "stocks_trained": [],
        }

        try:
            _force_china_direct_network_mode()
            self.log_message.emit(
                "China-direct network mode enabled for targeted GM learning.",
                "info",
            )
            learner_class = _get_auto_learner()
            if learner_class is None:
                self.error_occurred.emit("AutoLearner/ContinuousLearner not found.")
                return

            stock_codes = [str(c).strip() for c in list(self.config.get("stock_codes", []) or []) if str(c).strip()]
            if not stock_codes:
                self.error_occurred.emit("No stock codes provided.")
                return

            self._learner = learner_class()
            self._attach_progress_callback(results, default_error="Targeted training failed")

            requested_interval = normalize_training_interval(
                self.config.get("interval", _LOCKED_TRAIN_INTERVAL)
            )
            interval = _LOCKED_TRAIN_INTERVAL
            if requested_interval != interval:
                self.log_message.emit(
                    f"Interval {requested_interval} overridden to {interval}",
                    "warning",
                )
            horizon = _safe_int(self.config.get("horizon", 30), default=30, minimum=1, maximum=500)
            lookback_raw = self.config.get("lookback_bars")
            lookback_bars = (
                _safe_int(lookback_raw, default=10080, minimum=120, maximum=500_000)
                if lookback_raw is not None
                else None
            )

            learner_kwargs: dict[str, Any] = {
                "stock_codes": stock_codes,
                "epochs_per_cycle": _safe_int(self.config.get("epochs", 10), default=10, minimum=1),
                "interval": interval,
                "prediction_horizon": horizon,
                "incremental": bool(self.config.get("incremental", True)),
                "continuous": bool(self.config.get("continuous", False)),
                "cycle_interval_seconds": _safe_int(
                    self.config.get("cycle_interval_seconds", 60),
                    default=60,
                    minimum=30,
                    maximum=86_400,
                ),
            }
            if lookback_bars is not None:
                learner_kwargs["lookback_bars"] = lookback_bars

            self.log_message.emit(
                (
                    f"Targeted training started on {len(stock_codes)} stocks "
                    f"(interval={interval}, horizon={horizon})"
                ),
                "success",
            )

            self._start_and_wait(self._learner.start_targeted, learner_kwargs)
            self._stop_learner()

            if self.token.is_cancelled or not self.running:
                results["status"] = "stopped"
            elif self._error_flag:
                results["status"] = "error"
                results["error"] = self._error_message or "Targeted training failed"
            else:
                results["status"] = "ok"
            results["stocks_trained"] = stock_codes
            self.finished_result.emit(results)

        except _WORKER_RECOVERABLE_EXCEPTIONS as exc:
            error_msg = str(exc)
            log.error("TargetedLearnWorker error: %s", error_msg)
            log.debug(traceback.format_exc())
            if self.running:
                self.error_occurred.emit(error_msg)


class StockValidatorWorker(QThread):
    """Validates a stock code in background thread.
    Calls learner.validate_stock_code() which checks:
    - code exists in data sources
    - enough bars for training
    - stock name from spot cache when available.
    """

    validation_result = pyqtSignal(dict)

    def __init__(self, code: str, interval: str = "1m", request_id: int = 0) -> None:
        super().__init__()
        self.code = str(code or "")
        self.interval = normalize_training_interval(interval)
        self.request_id = int(request_id)

    def run(self) -> None:
        try:
            learner_class = _get_auto_learner()
            if learner_class is None:
                self.validation_result.emit(
                    {
                        "valid": False,
                        "code": self.code,
                        "name": "",
                        "bars": 0,
                        "request_id": self.request_id,
                        "message": "Learner module not available",
                    }
                )
                return

            learner = learner_class()
            result = learner.validate_stock_code(self.code, self.interval)
            result["request_id"] = self.request_id
            self.validation_result.emit(result)

        except _WORKER_RECOVERABLE_EXCEPTIONS as exc:
            self.validation_result.emit(
                {
                    "valid": False,
                    "code": self.code,
                    "name": "",
                    "bars": 0,
                    "request_id": self.request_id,
                    "message": f"Validation error: {str(exc)[:200]}",
                }
            )
