# ui/auto_learn_workers.py

import threading
import time
import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from utils.logger import get_logger

log = get_logger(__name__)

def _get_cancellation_token():
    """Lazy import CancellationToken."""
    from utils.cancellation import CancellationToken
    return CancellationToken()

def _get_auto_learner():
    """
    Lazy import AutoLearner/ContinuousLearner.
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

# WORKER: Auto Learn (random rotation)

class AutoLearnWorker(QThread):
    """
    Worker thread for auto-learning (random stock rotation).
    Runs the learner in a separate daemon thread so that
    learner.start() (which may block) doesn't prevent the
    QThread from responding to stop requests.
    """
    progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str, str)
    finished_result = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config) if config else {}
        self.running = False
        self.token = _get_cancellation_token()
        self._learner = None
        self._learner_thread: threading.Thread | None = None
        self._error_flag = False
        self._error_message = ""

    def run(self):
        self.running = True
        results = {
            "discovered": 0,
            "processed": 0,
            "samples": 0,
            "accuracy": 0.0,
        }

        try:
            LearnerClass = _get_auto_learner()
            if LearnerClass is None:
                self.error_occurred.emit(
                    "AutoLearner/ContinuousLearner not found. "
                    "Ensure models/auto_learner.py exists."
                )
                return

            self._learner = LearnerClass()

            max_stocks = int(self.config.get("max_stocks", 200))
            epochs = int(self.config.get("epochs", 10))
            incremental = bool(self.config.get("incremental", True))
            mode = str(self.config.get("mode", "full"))
            interval = "1m"
            horizon = 30
            try:
                lookback_bars = 10080
            except Exception:
                lookback_bars = 10080
            cycle_interval_seconds = 900

            def on_progress(p):
                if not self.running:
                    return
                try:
                    percent = int(max(0, min(100, getattr(p, 'progress', 0))))
                    message = str(
                        getattr(p, 'message', '') or getattr(p, 'stage', '')
                    )
                    stage = str(getattr(p, 'stage', ''))
                    if stage == "error":
                        self._error_flag = True
                        self._error_message = message or "Auto-learning failed"
                    self.progress.emit(percent, message)
                    self.log_message.emit(
                        f"{stage}: {message}" if stage else message,
                        "info",
                    )
                    results["discovered"] = int(
                        getattr(p, "stocks_found", 0) or 0
                    )
                    processed_direct = int(
                        getattr(p, "stocks_processed", 0) or 0
                    )
                    processed_alt = int(
                        getattr(p, "processed_count", 0) or 0
                    )
                    results["processed"] = max(processed_direct, processed_alt)
                    results["accuracy"] = float(
                        getattr(p, "validation_accuracy", 0.0) or 0.0
                    )
                except Exception as e:
                    log.debug(f"Progress callback error: {e}")

            if hasattr(self._learner, 'add_callback'):
                self._learner.add_callback(on_progress)

            learner_kwargs = {
                "mode": mode,
                "max_stocks": max_stocks,
                "epochs_per_cycle": epochs,
                "min_market_cap": 10,
                "include_all_markets": True,
                "continuous": True,
                "learning_while_trading": True,
                "interval": interval,
                "prediction_horizon": horizon,
                "lookback_bars": lookback_bars,
                "cycle_interval_seconds": cycle_interval_seconds,
                "incremental": incremental,
                "priority_stock_codes": list(
                    self.config.get("priority_stock_codes", []) or []
                ),
            }

            self._learner_thread = threading.Thread(
                target=self._run_learner,
                args=(learner_kwargs,),
                daemon=True,
                name="auto_learner_inner",
            )
            self._learner_thread.start()
            self.log_message.emit("Auto-learning started", "success")

            while self.running and not self.token.is_cancelled:
                if (
                    self._learner_thread is not None
                    and not self._learner_thread.is_alive()
                ):
                    break
                self.msleep(200)

            self._stop_learner()
            if self.token.is_cancelled or not self.running:
                results["status"] = "stopped"
            elif self._error_flag:
                results["status"] = "error"
                results["error"] = self._error_message or "Auto-learning failed"
            else:
                results["status"] = "ok"
            self.finished_result.emit(results)

        except Exception as e:
            error_msg = str(e)
            log.error(f"AutoLearnWorker error: {error_msg}")
            log.debug(traceback.format_exc())
            if self.running:
                self.error_occurred.emit(error_msg)

    def _run_learner(self, kwargs: dict):
        try:
            if self._learner is not None:
                self._learner.start(**kwargs)
                # AutoLearner.start() spawns its own thread and returns immediately.
                # Wait on that internal thread with short joins so cancellation
                # can interrupt quickly.
                t = getattr(self._learner, "_thread", None)
                if t is not None:
                    while (
                        self.running
                        and not self.token.is_cancelled
                        and t.is_alive()
                    ):
                        t.join(timeout=0.2)
                    return

                # Fallback: wait on progress flag if internal thread not exposed.
                while self.running and not self.token.is_cancelled:
                    if not getattr(self._learner.progress, "is_running", False):
                        break
                    time.sleep(0.2)
        except Exception as e:
            log.error(f"Learner thread error: {e}")
            log.debug(traceback.format_exc())
            if self.running:
                self._error_flag = True
                self._error_message = str(e)

    def _stop_learner(self):
        if self._learner is not None:
            try:
                if hasattr(self._learner, 'stop'):
                    try:
                        self._learner.stop(join_timeout=6.0)
                    except TypeError:
                        self._learner.stop()
            except Exception as e:
                log.debug(f"Learner stop error: {e}")

        if self._learner_thread is not None:
            try:
                self._learner_thread.join(timeout=8)
                if self._learner_thread.is_alive():
                    log.info("Learner thread still finalizing after stop request")
            except Exception:
                pass

        self._learner = None
        self._learner_thread = None

    def stop(self):
        self.running = False
        self.token.cancel()

# WORKER: Targeted Training (NEW)

class TargetedLearnWorker(QThread):
    """
    Worker thread for targeted training on user-selected stocks.
    Calls learner.start_targeted() which uses the fixed stock list
    instead of rotation/discovery.
    """
    progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str, str)
    finished_result = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config) if config else {}
        self.running = False
        self.token = _get_cancellation_token()
        self._learner = None
        self._learner_thread: threading.Thread | None = None
        self._error_flag = False
        self._error_message = ""

    def run(self):
        self.running = True
        results = {
            "discovered": 0,
            "processed": 0,
            "samples": 0,
            "accuracy": 0.0,
            "stocks_trained": [],
        }

        try:
            LearnerClass = _get_auto_learner()
            if LearnerClass is None:
                self.error_occurred.emit(
                    "AutoLearner/ContinuousLearner not found."
                )
                return

            stock_codes = list(self.config.get("stock_codes", []))
            if not stock_codes:
                self.error_occurred.emit("No stock codes provided.")
                return

            self._learner = LearnerClass()

            epochs = int(self.config.get("epochs", 10))
            incremental = bool(self.config.get("incremental", True))
            requested_interval = str(self.config.get("interval", "1m")).strip().lower()
            interval = "1m"
            horizon = int(self.config.get("horizon", 30))
            lookback = self.config.get("lookback_bars", None)
            continuous = bool(self.config.get("continuous", False))
            if requested_interval != "1m":
                self.log_message.emit(
                    f"Training interval forced to 1m (requested {requested_interval})",
                    "info",
                )

            def on_progress(p):
                if not self.running:
                    return
                try:
                    percent = int(max(0, min(100, getattr(p, 'progress', 0))))
                    message = str(
                        getattr(p, 'message', '') or getattr(p, 'stage', '')
                    )
                    stage = str(getattr(p, 'stage', ''))
                    if stage == "error":
                        self._error_flag = True
                        self._error_message = message or "Targeted training failed"
                    self.progress.emit(percent, message)
                    self.log_message.emit(
                        f"{stage}: {message}" if stage else message,
                        "info",
                    )
                    processed_direct = int(
                        getattr(p, "stocks_processed", 0) or 0
                    )
                    processed_alt = int(
                        getattr(p, "processed_count", 0) or 0
                    )
                    results["processed"] = max(processed_direct, processed_alt)
                    results["accuracy"] = float(
                        getattr(p, "validation_accuracy", 0.0) or 0.0
                    )
                    results["discovered"] = int(
                        getattr(p, "stocks_found", 0) or 0
                    )
                except Exception as e:
                    log.debug(f"Targeted progress callback error: {e}")

            if hasattr(self._learner, 'add_callback'):
                self._learner.add_callback(on_progress)

            learner_kwargs = {
                "stock_codes": stock_codes,
                "epochs_per_cycle": epochs,
                "interval": interval,
                "prediction_horizon": horizon,
                "incremental": incremental,
                "continuous": continuous,
                "cycle_interval_seconds": 900,
            }
            if lookback is not None:
                learner_kwargs["lookback_bars"] = int(lookback)

            self._learner_thread = threading.Thread(
                target=self._run_targeted,
                args=(learner_kwargs,),
                daemon=True,
                name="targeted_learner_inner",
            )
            self._learner_thread.start()

            self.log_message.emit(
                f"Targeted training started on {len(stock_codes)} stocks",
                "success",
            )

            while self.running and not self.token.is_cancelled:
                if (
                    self._learner_thread is not None
                    and not self._learner_thread.is_alive()
                ):
                    break
                self.msleep(200)

            self._stop_learner()
            if self.token.is_cancelled or not self.running:
                results["status"] = "stopped"
            elif self._error_flag:
                results["status"] = "error"
                results["error"] = (
                    self._error_message or "Targeted training failed"
                )
            else:
                results["status"] = "ok"
            results["stocks_trained"] = stock_codes
            self.finished_result.emit(results)

        except Exception as e:
            error_msg = str(e)
            log.error(f"TargetedLearnWorker error: {error_msg}")
            log.debug(traceback.format_exc())
            if self.running:
                self.error_occurred.emit(error_msg)

    def _run_targeted(self, kwargs: dict):
        try:
            if self._learner is not None:
                self._learner.start_targeted(**kwargs)
                # start_targeted() also spawns an internal learner thread.
                t = getattr(self._learner, "_thread", None)
                if t is not None:
                    while (
                        self.running
                        and not self.token.is_cancelled
                        and t.is_alive()
                    ):
                        t.join(timeout=0.2)
                    return

                while self.running and not self.token.is_cancelled:
                    if not getattr(self._learner.progress, "is_running", False):
                        break
                    time.sleep(0.2)
        except Exception as e:
            log.error(f"Targeted learner thread error: {e}")
            log.debug(traceback.format_exc())
            if self.running:
                self._error_flag = True
                self._error_message = str(e)

    def _stop_learner(self):
        if self._learner is not None:
            try:
                if hasattr(self._learner, 'stop'):
                    try:
                        self._learner.stop(join_timeout=6.0)
                    except TypeError:
                        self._learner.stop()
            except Exception as e:
                log.debug(f"Targeted learner stop error: {e}")

        if self._learner_thread is not None:
            try:
                self._learner_thread.join(timeout=8)
                if self._learner_thread.is_alive():
                    log.info("Targeted learner thread still finalizing after stop request")
            except Exception:
                pass

        self._learner = None
        self._learner_thread = None

    def stop(self):
        self.running = False
        self.token.cancel()

# STOCK VALIDATOR WORKER (for search)

class StockValidatorWorker(QThread):
    """
    Validates a stock code in background thread.
    Calls learner.validate_stock_code() which checks:
    - Code exists in the fetcher's data sources
    - Has enough bars for training (SEQUENCE_LENGTH + 20)
    - Returns stock name from spot cache if available
    """
    validation_result = pyqtSignal(dict)

    def __init__(self, code: str, interval: str = "1m", request_id: int = 0):
        super().__init__()
        self.code = code
        self.interval = interval
        self.request_id = int(request_id)

    def run(self):
        try:
            LearnerClass = _get_auto_learner()
            if LearnerClass is None:
                self.validation_result.emit({
                    'valid': False,
                    'code': self.code,
                    'name': '',
                    'bars': 0,
                    'request_id': self.request_id,
                    'message': 'Learner module not available',
                })
                return

            learner = LearnerClass()
            result = learner.validate_stock_code(self.code, self.interval)
            result["request_id"] = self.request_id
            self.validation_result.emit(result)

        except Exception as e:
            self.validation_result.emit({
                'valid': False,
                'code': self.code,
                'name': '',
                'bars': 0,
                'request_id': self.request_id,
                'message': f'Validation error: {str(e)[:200]}',
            })
