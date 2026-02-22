# models/auto_learner.py
import json
import random
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

from config.settings import CONFIG
from data.fetcher import get_fetcher
from models.auto_learner_components import (
    ExperienceReplayBuffer,
    LearningProgress,
    LRScheduler,
    MetricTracker,
    ModelGuardian,
    ParallelFetcher,
    StockRotator,
)
from models.auto_learner_cycle_ops import _main_loop as _main_loop_impl
from models.auto_learner_cycle_ops import _run_cycle as _run_cycle_impl
from models.auto_learner_cycle_ops import (
    _run_targeted_cycle as _run_targeted_cycle_impl,
)
from models.auto_learner_flow_ops import _compute_lookback_bars as _compute_lookback_bars_impl
from models.auto_learner_flow_ops import (
    _filter_priority_session_codes as _filter_priority_session_codes_impl,
)
from models.auto_learner_flow_ops import _interval_seconds as _interval_seconds_impl
from models.auto_learner_flow_ops import _norm_code as _norm_code_impl
from models.auto_learner_flow_ops import _prioritize_codes_by_news as _prioritize_codes_by_news_impl
from models.auto_learner_flow_ops import (
    _session_continuous_window_seconds as _session_continuous_window_seconds_impl,
)
from models.auto_learner_flow_ops import pause as _pause_impl
from models.auto_learner_flow_ops import resume as _resume_impl
from models.auto_learner_flow_ops import run as _run_impl
from models.auto_learner_flow_ops import stop as _stop_impl
from models.auto_learner_flow_ops import validate_stock_code as _validate_stock_code_impl
from models.auto_learner_lifecycle_ops import _interruptible_sleep as _interruptible_sleep_impl
from models.auto_learner_lifecycle_ops import _load_state as _load_state_impl
from models.auto_learner_lifecycle_ops import _save_state as _save_state_impl
from models.auto_learner_lifecycle_ops import _targeted_loop as _targeted_loop_impl
from models.auto_learner_lifecycle_ops import run_targeted as _run_targeted_impl
from models.auto_learner_lifecycle_ops import start as _start_impl
from models.auto_learner_lifecycle_ops import start_targeted as _start_targeted_impl
from utils.cancellation import CancellationToken, CancelledException
from utils.logger import get_logger
from utils.recoverable import JSON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)
_AUTO_LEARNER_RECOVERABLE_EXCEPTIONS = JSON_RECOVERABLE_EXCEPTIONS

# THREAD-LOCAL LR OVERRIDE (FIX C1)

_thread_local = threading.local()

def get_effective_learning_rate() -> float:
    """
    Get thread-local LR override or global default.

    This allows ContinuousLearner to set a per-cycle LR without
    mutating the global CONFIG, avoiding race conditions when
    multiple threads read CONFIG.model.learning_rate.
    """
    return getattr(_thread_local, 'learning_rate', CONFIG.model.learning_rate)

def set_thread_local_lr(lr: float):
    """Set thread-local learning rate override."""
    _thread_local.learning_rate = lr

def clear_thread_local_lr():
    """Clear thread-local learning rate override."""
    if hasattr(_thread_local, 'learning_rate'):
        delattr(_thread_local, 'learning_rate')


class ContinuousLearner:
    """
    Production continuous learning system.

    Supports two modes:
    - Auto learning: discovers and rotates through stocks automatically
    - Targeted learning: trains on specific user-selected stocks
    """

    # FIX M2: Complete BARS_PER_DAY fallback dictionary
    _BARS_PER_DAY_FALLBACK: dict[str, int] = {
        "1m": 240,
        "2m": 120,
        "5m": 48,
        "15m": 16,
        "30m": 8,
        "60m": 4,
        "1h": 4,
        "1d": 1,
        "1wk": 1,
        "1mo": 1,
    }

    _INTERVAL_MAX_DAYS_FALLBACK: dict[str, int] = {
        "1m": 7,
        "2m": 60,
        "5m": 60,
        "15m": 60,
        "30m": 60,
        "60m": 730,
        "1h": 730,
        "1d": 10000,
        "1wk": 10000,
        "1mo": 10000,
    }

    # FIX VAL: Minimum holdout predictions for reliable comparison
    _MIN_HOLDOUT_PREDICTIONS = 3
    _MIN_TUNED_TRADES = 3
    _MIN_1M_LOOKBACK_BARS = 10080
    _FULL_RETRAIN_EVERY_CYCLES = 6
    _FORCE_FULL_RETRAIN_CYCLES = 2
    _FORCE_FULL_RETRAIN_AFTER_REJECTIONS = 2
    _REJECTION_COOLDOWN_AFTER_STREAK = 4
    _MIN_REJECTION_COOLDOWN_SECONDS = 600
    _MAX_REJECTION_COOLDOWN_SECONDS = 3600

    def __init__(self):
        self.progress = LearningProgress()
        self._cancel_token = CancellationToken()
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.RLock()
        self._callbacks: list[Callable[[LearningProgress], None]] = []
        self._lock = threading.RLock()

        self._rotator = StockRotator()
        self._replay = ExperienceReplayBuffer(max_size=2000)
        self._guardian = ModelGuardian()
        self._metrics = MetricTracker(window=10)
        self._lr_scheduler = LRScheduler()
        self._fetcher = ParallelFetcher(max_workers=5)

        self._holdout_codes: list[str] = []
        self._holdout_size: int = 10
        self._holdout_refresh_interval: int = 50

        self.state_path = CONFIG.DATA_DIR / "learner_state.json"
        self._load_state()

    # =========================================================================
    # =========================================================================

    def add_callback(self, callback: Callable[[LearningProgress], None]):
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[LearningProgress], None]):
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def _notify(self):
        self.progress.last_update = datetime.now()
        with self._lock:
            callbacks = self._callbacks.copy()
        for cb in callbacks:
            try:
                cb(self.progress)
            except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
                log.warning("Progress callback failed: %s", e)

    def _update(self, stage=None, message=None, progress=None, **kw):
        if stage:
            self.progress.stage = stage
        if message:
            self.progress.message = message
        if progress is not None:
            self.progress.progress = progress
        for k, v in kw.items():
            if hasattr(self.progress, k):
                setattr(self.progress, k, v)
        self._notify()

    def _should_stop(self) -> bool:
        return self._cancel_token.is_cancelled

    def _get_holdout_set(self) -> set[str]:
        with self._lock:
            return set(self._holdout_codes)

    def _set_holdout_codes(self, codes: list[str]):
        with self._lock:
            self._holdout_codes = list(codes)
        self._guardian.set_holdout(codes)

    # FIX PAUSE: Extracted to reusable method
    def _wait_if_paused(self) -> bool:
        """
        Block while paused. Returns True if should stop.
        """
        while self.progress.is_paused and not self._should_stop():
            if self._cancel_token.wait(timeout=1.0):
                break
        return self._should_stop()

    # =========================================================================
    # LIFECYCLE - AUTO MODE
    # =========================================================================

    def start(
        self,
        mode="full",
        max_stocks=None,
        epochs_per_cycle=10,
        min_market_cap=10,
        include_all_markets=True,
        continuous=True,
        learning_while_trading=True,
        interval="1m",
        prediction_horizon=30,
        lookback_bars=None,
        cycle_interval_seconds=900,
        incremental=True,
        priority_stock_codes: list[str] | None = None,
    ):
        return _start_impl(
            self,
            mode=mode,
            max_stocks=max_stocks,
            epochs_per_cycle=epochs_per_cycle,
            min_market_cap=min_market_cap,
            include_all_markets=include_all_markets,
            continuous=continuous,
            learning_while_trading=learning_while_trading,
            interval=interval,
            prediction_horizon=prediction_horizon,
            lookback_bars=lookback_bars,
            cycle_interval_seconds=cycle_interval_seconds,
            incremental=incremental,
            priority_stock_codes=priority_stock_codes,
        )

    def _compute_lookback_bars(self, interval: str) -> int:
        return _compute_lookback_bars_impl(self, interval)

    @staticmethod
    def _interval_seconds(interval: str) -> int:
        return _interval_seconds_impl(interval)

    def _normalize_interval_token(self, interval: str) -> str:
        req = str(interval or "1m").strip().lower()
        req = {"h1": "1h", "d1": "1d"}.get(req, req)
        try:
            from data.fetcher import BARS_PER_DAY
        except ImportError:
            BARS_PER_DAY = self._BARS_PER_DAY_FALLBACK
        return req if req in BARS_PER_DAY else "1m"

    def _session_continuous_window_seconds(
        self,
        code: str,
        interval: str,
        max_bars: int = 5000,
    ) -> float:
        return _session_continuous_window_seconds_impl(
            self,
            code,
            interval,
            max_bars=max_bars,
        )

    def _filter_priority_session_codes(
        self,
        codes: list[str],
        interval: str,
        min_seconds: float = 3600.0,
    ) -> list[str]:
        return _filter_priority_session_codes_impl(
            self,
            codes,
            interval,
            min_seconds=min_seconds,
        )

    @staticmethod
    def _norm_code(raw: str) -> str:
        return _norm_code_impl(raw)

    def _prioritize_codes_by_news(
        self,
        codes: list[str],
        interval: str,
        max_probe: int = 16,
    ) -> list[str]:
        return _prioritize_codes_by_news_impl(
            self,
            codes,
            interval,
            max_probe=max_probe,
        )

    def run(self, **kwargs):
        return _run_impl(self, **kwargs)

    def stop(self, join_timeout: float = 30.0):
        _stop_impl(self, join_timeout=join_timeout)

    def pause(self):
        _pause_impl(self)

    def resume(self):
        _resume_impl(self)

    # =========================================================================
    # LIFECYCLE - TARGETED MODE
    # =========================================================================

    def start_targeted(
        self,
        stock_codes: list[str],
        epochs_per_cycle: int = 10,
        interval: str = "1m",
        prediction_horizon: int = 30,
        lookback_bars: int | None = None,
        incremental: bool = True,
        continuous: bool = False,
        cycle_interval_seconds: int = 900,
    ):
        return _start_targeted_impl(
            self,
            stock_codes=stock_codes,
            epochs_per_cycle=epochs_per_cycle,
            interval=interval,
            prediction_horizon=prediction_horizon,
            lookback_bars=lookback_bars,
            incremental=incremental,
            continuous=continuous,
            cycle_interval_seconds=cycle_interval_seconds,
        )

    def run_targeted(self, **kwargs):
        return _run_targeted_impl(self, **kwargs)

    def validate_stock_code(
        self, code: str, interval: str = "1m"
    ) -> dict[str, Any]:
        return _validate_stock_code_impl(
            self,
            code,
            interval=interval,
            get_fetcher_fn=get_fetcher,
        )

    def _targeted_loop(
        self,
        stock_codes: list[str],
        epochs: int,
        interval: str,
        horizon: int,
        lookback: int,
        incremental: bool,
        continuous: bool,
        cycle_seconds: int,
    ):
        return _targeted_loop_impl(
            self,
            stock_codes=stock_codes,
            epochs=epochs,
            interval=interval,
            horizon=horizon,
            lookback=lookback,
            incremental=incremental,
            continuous=continuous,
            cycle_seconds=cycle_seconds,
        )

    def _interruptible_sleep(self, seconds: int):
        return _interruptible_sleep_impl(self, seconds)

    def _handle_plateau(
        self, plateau: dict, current_epochs: int, incremental: bool,
    ) -> tuple[int, bool]:
        """Graduated plateau response - uses public rotator methods."""
        action = plateau['action']
        log.info(f"Plateau response: {plateau['message']}")
        self._update(message=plateau['message'])
        self.progress.plateau_count = self._metrics.plateau_count

        if action == 'increase_epochs':
            new_epochs = min(int(current_epochs * plateau.get('factor', 1.5)), 200)
            return new_epochs, incremental
        elif action == 'reset_rotation':
            self._rotator.reset_processed()
            return current_epochs, incremental
        elif action == 'increase_diversity':
            self._rotator.reset_processed()
            self._rotator.reset_discovery()
            return current_epochs, incremental
        elif action == 'full_reset':
            self._rotator.reset_processed()
            self._rotator.reset_discovery()
            self._lr_scheduler.apply_boost(plateau.get('lr_boost', 2.0))
            return current_epochs, incremental

        return current_epochs, incremental

    # =========================================================================
    # SINGLE CYCLE - AUTO MODE
    def _finalize_cycle(
        self, accepted: bool,
        ok_codes: list[str], new_batch: list[str], replay_batch: list[str],
        interval: str, horizon: int, lookback: int,
        acc: float, cycle_number: int, start_time: datetime,
    ):
        """Shared logic for finalizing a training cycle (auto or targeted)."""
        mode = self.progress.training_mode

        if accepted:
            self.progress.model_was_rejected = False
            self.progress.consecutive_rejections = 0
            self._replay.add(ok_codes, confidence=acc)
            if mode == "auto":
                self._rotator.mark_processed(new_batch)
            self._cache_training_sequences(
                ok_codes, interval, horizon, lookback,
            )
            self.progress.total_stocks_learned += len(ok_codes)
            duration = (datetime.now() - start_time).total_seconds() / 3600
            self.progress.total_training_hours += duration

            if acc > self.progress.best_accuracy_ever:
                self.progress.best_accuracy_ever = acc
                extra_meta = {}
                if mode == "targeted":
                    extra_meta['targeted_stocks'] = ok_codes[:50]
                self._guardian.save_as_best(
                    interval, horizon,
                    {
                        'accuracy': acc, 'cycle': cycle_number,
                        'total_learned': len(self._replay),
                        'timestamp': datetime.now().isoformat(),
                        **extra_meta,
                    },
                )

            label = "Targeted " if mode == "targeted" else ""
            self._update(
                stage="complete", progress=100.0,
                message=(
                    f"[OK] {label}Cycle {cycle_number}: acc={acc:.1%}, "
                    f"{len(ok_codes)} trained, "
                    f"total={len(self._replay)} | ACCEPTED"
                ),
            )
        else:
            self.progress.model_was_rejected = True
            self.progress.consecutive_rejections = (
                int(max(0, self.progress.consecutive_rejections)) + 1
            )
            label = "Targeted " if mode == "targeted" else ""
            self._update(
                stage="complete", progress=100.0,
                message=(
                    f"REJECTED {label}cycle {cycle_number}: acc={acc:.1%} | "
                    f"streak={self.progress.consecutive_rejections} | "
                    f"previous model restored"
                ),
            )

        self._log_cycle(
            cycle_number, new_batch, replay_batch, ok_codes, acc, accepted,
        )

    # =========================================================================
    # =========================================================================

    def _resolve_interval(self, interval, horizon, lookback):
        """FIX M2: Use complete BARS_PER_DAY fallback."""
        try:
            from data.fetcher import BARS_PER_DAY, INTERVAL_MAX_DAYS
        except ImportError:
            BARS_PER_DAY = self._BARS_PER_DAY_FALLBACK
            INTERVAL_MAX_DAYS = self._INTERVAL_MAX_DAYS_FALLBACK

        req_interval = str(interval or "1m").strip().lower()
        eff_interval = self._normalize_interval_token(req_interval)
        if eff_interval != req_interval:
            log.info(
                "Resolved training interval %s is unsupported; using %s",
                req_interval,
                eff_interval,
            )
        eff_horizon = max(1, int(horizon))

        bpd = BARS_PER_DAY.get(eff_interval, 1)
        max_avail = int(INTERVAL_MAX_DAYS.get(eff_interval, 500) * bpd)
        if eff_interval == "1m":
            max_avail = max(max_avail, int(self._MIN_1M_LOOKBACK_BARS))
        eff_lookback = min(max(1, int(lookback)), max_avail)
        if eff_interval == "1m":
            eff_lookback = max(int(eff_lookback), int(self._MIN_1M_LOOKBACK_BARS))

        if eff_interval in ("1m", "2m", "5m"):
            min_bars = max(CONFIG.SEQUENCE_LENGTH + 20, 80)
        elif eff_interval in ("15m", "30m", "60m", "1h"):
            min_bars = max(CONFIG.SEQUENCE_LENGTH + 30, 90)
        else:
            min_bars = CONFIG.SEQUENCE_LENGTH + 50

        return eff_interval, eff_horizon, eff_lookback, min_bars

    def _ensure_holdout(self, interval, lookback, min_bars, cycle_number):
        """FIX: Adaptive holdout size based on pool size."""
        min_required = max(1, int(self._MIN_HOLDOUT_PREDICTIONS))
        with self._lock:
            current_holdout_size = len(self._holdout_codes)
            should_refresh = (
                not self._holdout_codes
                or current_holdout_size < min_required
                or (cycle_number > 1 and cycle_number % self._holdout_refresh_interval == 0)
            )
            if not should_refresh:
                return
            old_holdout_set = set(self._holdout_codes)

        candidates = self._rotator.get_pool_snapshot()
        if not candidates:
            candidates = list(CONFIG.STOCK_POOL)
        replay_all = set(self._replay.get_all())
        extra = [c for c in replay_all if c not in candidates]
        random.shuffle(extra)
        candidates.extend(extra[:20])
        random.shuffle(candidates)

        # FIX: Adaptive holdout size - never more than 30% of pool
        pool_size = len(candidates)
        max_holdout = max(3, int(pool_size * 0.30))  # 30% max
        target_holdout = min(self._holdout_size, max_holdout)

        log.debug(f"Holdout: pool={pool_size}, target={target_holdout}")

        new_holdout = []
        fetcher = get_fetcher()

        for code in candidates:
            if len(new_holdout) >= target_holdout:  # Use target, not self._holdout_size
                break
            if self._should_stop():
                raise CancelledException()
            try:
                try:
                    df = fetcher.get_history(
                        code,
                        interval=interval,
                        bars=lookback,
                        use_cache=True,
                        update_db=True,
                        allow_online=True,
                        refresh_intraday_after_close=True,
                    )
                except TypeError:
                    df = fetcher.get_history(
                        code, interval=interval, bars=lookback, use_cache=True,
                    )
                if df is not None and len(df) >= min_bars:
                    new_holdout.append(code)
            except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
                log.debug("Holdout candidate fetch failed for %s: %s", code, e)
                continue

        if self._should_stop():
            raise CancelledException()

        if len(new_holdout) < min_required:
            log.warning("Failed to build new holdout set - keeping existing")
            return

        # Atomic check-and-swap
        with self._lock:
            current_holdout_set = set(self._holdout_codes)
            if current_holdout_set != old_holdout_set and self._holdout_codes:
                log.debug("Holdout already updated by another thread - skipping")
                return
            self._holdout_codes = new_holdout

        self._guardian.set_holdout(new_holdout)
        log.info(f"Holdout set: {len(new_holdout)} stocks (30% of {pool_size})")

    @staticmethod
    def _trainer_result_is_deployable(
        result: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        """Require trainer quality gate and artifact deployment success."""
        payload = result if isinstance(result, dict) else {}

        quality_gate = payload.get("quality_gate", {})
        if isinstance(quality_gate, dict) and quality_gate:
            if not bool(quality_gate.get("passed", False)):
                action = str(
                    quality_gate.get(
                        "recommended_action",
                        "shadow_mode_recommended",
                    )
                )
                reasons = ",".join(
                    str(x)
                    for x in list(quality_gate.get("failed_reasons", []) or [])
                    if str(x).strip()
                )
                if reasons:
                    return False, f"trainer_quality_gate_failed:{action}:{reasons}"
                return False, f"trainer_quality_gate_failed:{action}"

        deployment = payload.get("deployment", {})
        if isinstance(deployment, dict) and deployment:
            if not bool(deployment.get("deployed", False)):
                reason = str(deployment.get("reason", "not_deployed"))
                return False, f"trainer_artifact_not_deployed:{reason}"

        return True, "ok"

    def _emit_model_drift_alarm_if_needed(
        self,
        result: dict[str, Any] | None,
        *,
        context: str = "",
    ) -> bool:
        """Escalate trainer drift-guard failures to runtime auto-trade controls."""
        payload = result if isinstance(result, dict) else {}
        drift_guard = payload.get("drift_guard", {})
        quality_gate = payload.get("quality_gate", {})
        if not isinstance(drift_guard, dict):
            drift_guard = {}
        if not isinstance(quality_gate, dict):
            quality_gate = {}

        action = str(drift_guard.get("action", "") or "").strip().lower()
        failed_reasons = {
            str(x).strip().lower()
            for x in list(quality_gate.get("failed_reasons", []) or [])
            if str(x).strip()
        }
        drift_blocked = (
            action == "rollback_recommended"
            or "drift_guard_block" in failed_reasons
        )
        if not drift_blocked:
            return False

        try:
            score_drop = float(drift_guard.get("score_drop", 0.0) or 0.0)
        except (TypeError, ValueError):
            score_drop = 0.0
        try:
            accuracy_drop = float(drift_guard.get("accuracy_drop", 0.0) or 0.0)
        except (TypeError, ValueError):
            accuracy_drop = 0.0

        ctx = str(context).strip()
        prefix = f"{ctx}: " if ctx else ""
        reason = (
            f"{prefix}model drift guard triggered "
            f"(action={action or 'unknown'}, "
            f"score_drop={score_drop:.3f}, accuracy_drop={accuracy_drop:.3f})"
        )
        self.progress.add_warning(reason)
        log.warning("Trainer drift alarm: %s", reason)

        try:
            from trading.executor import ExecutionEngine

            handled = int(
                ExecutionEngine.trigger_model_drift_alarm(
                    reason=reason,
                    severity="critical",
                    metadata={
                        "context": str(ctx),
                        "action": str(action),
                        "score_drop": float(score_drop),
                        "accuracy_drop": float(accuracy_drop),
                    },
                )
            )
            if handled <= 0:
                log.warning(
                    "Model drift alarm raised but no active execution engine handled it"
                )
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.warning("Failed to raise model drift runtime alarm: %s", e)
        return True

    def _train(
        self, ok_codes, epochs, interval, horizon, lookback, incremental, lr,
    ) -> dict:
        """
        Train model.

        FIX LR: Passes learning_rate explicitly to trainer.train() instead
        of mutating global CONFIG.model.learning_rate.
        """
        from models.trainer import Trainer

        trainer = Trainer()

        # Set scaler-freeze flag via the documented attribute
        if incremental:
            scaler_path = CONFIG.MODEL_DIR / f"scaler_{interval}_{horizon}.pkl"
            if scaler_path.exists():
                loaded = trainer.processor.load_scaler(str(scaler_path))
                if loaded:
                    trainer._skip_scaler_fit = True
                    log.info("Existing scaler injected (no refit)")

        model_epoch_map: dict[str, int] = {}
        max_progress_seen = 50.0

        def cb(model_name, epoch_idx, val_acc):
            nonlocal max_progress_seen
            if self._should_stop():
                raise CancelledException()

            key = str(model_name or "model").strip().lower() or "model"
            prev_epoch = int(model_epoch_map.get(key, 0) or 0)
            model_epoch_map[key] = max(prev_epoch, int(epoch_idx + 1))

            observed_models = max(1, len(model_epoch_map))
            completed_epochs = sum(
                min(max(1, int(epochs)), int(v))
                for v in model_epoch_map.values()
            )
            aggregate_ratio = completed_epochs / float(
                max(1, observed_models * max(1, int(epochs)))
            )
            progress_value = 50.0 + (35.0 * float(aggregate_ratio))
            if progress_value < max_progress_seen:
                progress_value = max_progress_seen
            else:
                max_progress_seen = progress_value

            self.progress.training_epoch = int(max(model_epoch_map.values()))
            self.progress.validation_accuracy = float(val_acc)
            self._update(
                message=(
                    f"Training {model_name}: {epoch_idx + 1}/{epochs} "
                    f"({observed_models} model(s))"
                ),
                progress=float(min(99.0, max(50.0, progress_value))),
            )

        set_thread_local_lr(lr)

        try:
            result = trainer.train(
                stock_codes=ok_codes,
                epochs=epochs,
                callback=cb,
                stop_flag=self._cancel_token,
                save_model=True,
                incremental=incremental,
                interval=interval,
                prediction_horizon=horizon,
                lookback_bars=lookback,
                learning_rate=lr,
            )
        except CancelledException:
            return {"status": "cancelled"}
        finally:
            clear_thread_local_lr()

        return result

    def _validate_and_decide(
        self, interval, horizon, lookback, pre_val, new_acc
    ) -> bool:
        """
        Decide whether to accept or reject the new model based on
        holdout validation.

        FIX VAL: Requires minimum number of holdout predictions before
        making rejection decisions.
        """
        MAX_DEGRADATION = 0.15
        MIN_PREDS = self._MIN_HOLDOUT_PREDICTIONS
        holdout_snapshot = list(self._get_holdout_set())

        if not holdout_snapshot:
            log.info("No holdout validation - accepting")
            return True

        post_val = self._guardian.validate_model(
            interval, horizon, holdout_snapshot, lookback, collect_samples=True
        )
        post_acc = post_val.get('accuracy', 0)
        post_conf = post_val.get('avg_confidence', 0)
        post_preds = post_val.get('predictions_made', 0)

        self.progress.old_stock_accuracy = post_acc
        self.progress.old_stock_confidence = post_conf

        # Safety gate: insufficient holdout predictions cannot validate quality.
        if post_preds < MIN_PREDS:
            log.warning(
                f"REJECTED: holdout produced only {post_preds} predictions "
                f"(need {MIN_PREDS}); restoring previous model"
            )
            self.progress.add_warning(
                f"Rejected: holdout insufficient ({post_preds}/{MIN_PREDS} predictions)"
            )
            self._guardian.restore_backup(interval, horizon)
            return False

        if not pre_val or pre_val.get('predictions_made', 0) < MIN_PREDS:
            log.info(
                f"No reliable pre-validation baseline "
                f"(preds={pre_val.get('predictions_made', 0) if pre_val else 0}). "
                f"Holdout acc={post_acc:.1%}"
            )
            accepted = post_acc >= 0.30
            if accepted:
                self._maybe_tune_precision_thresholds(
                    interval, horizon, post_val.get("samples", [])
                )
            return accepted

        pre_acc = pre_val.get('accuracy', 0)
        pre_conf = pre_val.get('avg_confidence', 0)

        log.info(
            f"Validation: holdout acc {pre_acc:.1%}->{post_acc:.1%}, "
            f"conf {pre_conf:.3f}->{post_conf:.3f}, train acc={new_acc:.1%}"
        )

        if pre_acc > 0.1:
            degradation = (pre_acc - post_acc) / pre_acc
            if degradation > MAX_DEGRADATION:
                log.warning(f"REJECTED: holdout acc degraded {degradation:.1%}")
                self._guardian.restore_backup(interval, horizon)
                self.progress.add_warning(f"Rejected: holdout acc {pre_acc:.1%}->{post_acc:.1%}")
                return False

        if pre_conf > 0.1:
            conf_deg = (pre_conf - post_conf) / pre_conf
            if conf_deg > MAX_DEGRADATION:
                log.warning(f"REJECTED: holdout conf degraded {conf_deg:.1%}")
                self._guardian.restore_backup(interval, horizon)
                return False

        log.info(f"ACCEPTED: holdout acc={post_acc:.1%}")
        self._maybe_tune_precision_thresholds(
            interval, horizon, post_val.get("samples", [])
        )
        return True

    def _maybe_tune_precision_thresholds(
        self,
        interval: str,
        horizon: int,
        samples: list[dict[str, Any]],
    ) -> None:
        cfg = getattr(CONFIG, "precision", None)
        if not cfg or not bool(getattr(cfg, "enable_threshold_tuning", False)):
            return
        min_samples = int(getattr(cfg, "min_tuning_samples", 12))
        if not samples or len(samples) < min_samples:
            return
        tuned = self._tune_precision_thresholds(samples)
        if not tuned:
            return
        self._save_precision_profile(interval, horizon, tuned, samples)

    def _tune_precision_thresholds(
        self, samples: list[dict[str, Any]]
    ) -> dict[str, float] | None:
        """
        Grid-search confidence/agreement/entropy/edge thresholds that maximize
        a profit-quality proxy on holdout samples.
        """
        conf_grid = [0.60, 0.65, 0.70, 0.75, 0.80]
        agree_grid = [0.55, 0.60, 0.65, 0.70, 0.75]
        entropy_grid = [0.30, 0.40, 0.50, 0.60]
        edge_grid = [0.06, 0.10, 0.14, 0.18]

        best_score = -1e18
        best: dict[str, float] | None = None

        for c in conf_grid:
            for a in agree_grid:
                for e in entropy_grid:
                    for edge in edge_grid:
                        metrics = self._score_thresholds(samples, c, a, e, edge)
                        if metrics["trades"] < self._MIN_TUNED_TRADES:
                            continue
                        # Weighted objective: profit factor first, then precision.
                        score = (
                            metrics["profit_factor"] * 2.0
                            + metrics["precision"] * 1.2
                            + metrics["expectancy"] * 0.2
                            - metrics["trade_rate"] * 0.05
                        )
                        if score > best_score:
                            best_score = score
                            best = {
                                "min_confidence": float(c),
                                "min_agreement": float(a),
                                "max_entropy": float(e),
                                "min_edge": float(edge),
                                "precision": float(metrics["precision"]),
                                "profit_factor": float(metrics["profit_factor"]),
                                "expectancy": float(metrics["expectancy"]),
                                "trades": float(metrics["trades"]),
                                "trade_rate": float(metrics["trade_rate"]),
                            }
        return best

    @staticmethod
    def _score_thresholds(
        samples: list[dict[str, Any]],
        min_conf: float,
        min_agree: float,
        max_entropy: float,
        min_edge: float,
    ) -> dict[str, float]:
        wins = 0
        losses = 0
        pnl_win = 0.0
        pnl_loss = 0.0
        trades = 0

        for s in samples:
            pred_cls = int(s.get("predicted", 1))
            if pred_cls not in (0, 2):
                continue
            conf = float(s.get("confidence", 0.0))
            agree = float(s.get("agreement", 0.0))
            entropy = float(s.get("entropy", 1.0))
            edge = abs(float(s.get("prob_up", 0.33)) - float(s.get("prob_down", 0.33)))
            if conf < min_conf or agree < min_agree or entropy > max_entropy or edge < min_edge:
                continue

            trades += 1
            actual = int(s.get("actual", 1))
            ret = float(s.get("future_return", 0.0))
            # Proxy net return after costs.
            cost_pct = (
                float(getattr(CONFIG.trading, "commission", 0.0))
                + float(getattr(CONFIG.trading, "slippage", 0.0))
                + float(getattr(CONFIG.trading, "stamp_tax", 0.0))
            ) * 100.0
            signed_ret = ret if pred_cls == 2 else -ret
            net = signed_ret - cost_pct

            if actual == pred_cls and net > 0:
                wins += 1
                pnl_win += net
            else:
                losses += 1
                pnl_loss += abs(net) if net < 0 else cost_pct

        precision = wins / max(trades, 1)
        expectancy = (pnl_win - pnl_loss) / max(trades, 1)
        if pnl_loss <= 1e-9:
            profit_factor = float(wins) if wins > 0 else 0.0
        else:
            profit_factor = pnl_win / pnl_loss
        return {
            "trades": float(trades),
            "precision": float(precision),
            "profit_factor": float(profit_factor),
            "expectancy": float(expectancy),
            "trade_rate": float(trades / max(len(samples), 1)),
        }

    def _save_precision_profile(
        self,
        interval: str,
        horizon: int,
        tuned: dict[str, float],
        samples: list[dict[str, Any]],
    ) -> None:
        try:
            filename = str(getattr(CONFIG.precision, "profile_filename", "precision_thresholds.json"))
            path = CONFIG.data_dir / filename
            payload = {
                "updated_at": datetime.now().isoformat(),
                "interval": str(interval),
                "horizon": int(horizon),
                "sample_count": int(len(samples)),
                "thresholds": {
                    "min_confidence": float(tuned["min_confidence"]),
                    "min_agreement": float(tuned["min_agreement"]),
                    "max_entropy": float(tuned["max_entropy"]),
                    "min_edge": float(tuned["min_edge"]),
                },
                "metrics": {
                    "precision": float(tuned.get("precision", 0.0)),
                    "profit_factor": float(tuned.get("profit_factor", 0.0)),
                    "expectancy": float(tuned.get("expectancy", 0.0)),
                    "trades": int(tuned.get("trades", 0.0)),
                    "trade_rate": float(tuned.get("trade_rate", 0.0)),
                },
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from utils.atomic_io import atomic_write_json
                atomic_write_json(path, payload, indent=2, use_lock=True)
            except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
                log.debug("Atomic precision-profile save failed; using plain write: %s", e)
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log.info(
                "Precision profile saved: conf>=%.2f agree>=%.2f ent<=%.2f edge>=%.2f "
                "(PF=%.2f, precision=%.2f, trades=%d)",
                payload["thresholds"]["min_confidence"],
                payload["thresholds"]["min_agreement"],
                payload["thresholds"]["max_entropy"],
                payload["thresholds"]["min_edge"],
                payload["metrics"]["profit_factor"],
                payload["metrics"]["precision"],
                payload["metrics"]["trades"],
            )
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Failed saving precision profile: %s", e)

    def _cache_training_sequences(self, codes, interval, horizon, lookback):
        try:
            from data.features import FeatureEngine
            from data.processor import DataProcessor

            feature_engine = FeatureEngine()
            processor = DataProcessor()
            feature_cols = feature_engine.get_feature_columns()

            scaler_path = CONFIG.MODEL_DIR / f"scaler_{interval}_{horizon}.pkl"
            if scaler_path.exists():
                processor.load_scaler(str(scaler_path))

            fetcher = get_fetcher()
            cached = 0

            for code in codes[:30]:
                try:
                    try:
                        df = fetcher.get_history(
                            code,
                            interval=interval,
                            bars=lookback,
                            use_cache=True,
                            update_db=True,
                            allow_online=True,
                            refresh_intraday_after_close=True,
                        )
                    except TypeError:
                        df = fetcher.get_history(
                            code, interval=interval, bars=lookback, use_cache=True,
                        )
                    if df is None or len(df) < CONFIG.SEQUENCE_LENGTH + 20:
                        continue
                    df = feature_engine.create_features(df)
                    df = processor.create_labels(df, horizon=horizon)
                    X, y, _ = processor.prepare_sequences(df, feature_cols, fit_scaler=False)
                    if len(X) > 0:
                        self._replay.cache_sequences(code, X, y)
                        cached += 1
                except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
                    log.debug("Sequence cache build failed for %s: %s", code, e)
                    continue

            log.debug(f"Cached sequences for {cached}/{min(len(codes), 30)} stocks")
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.debug(f"Sequence caching failed: {e}")

    def _log_cycle(self, cycle, new_batch, replay_batch, ok_codes, acc, accepted):
        try:
            history_dir = CONFIG.DATA_DIR / "cycle_history"
            history_dir.mkdir(parents=True, exist_ok=True)
            record = {
                'cycle': cycle,
                'timestamp': datetime.now().isoformat(),
                'training_mode': self.progress.training_mode,
                'new_stocks': new_batch[:50],
                'replay_stocks': replay_batch[:50],
                'ok_stocks': ok_codes[:50],
                'accuracy': acc,
                'accepted': accepted,
                'total_learned': len(self._replay),
                'trend': self._metrics.trend,
                'ema': self._metrics.ema,
            }
            try:
                from utils.atomic_io import atomic_write_json
                path = history_dir / f"cycle_{cycle:04d}.json"
                atomic_write_json(path, record, use_lock=True)
            except ImportError:
                path = history_dir / f"cycle_{cycle:04d}.json"
                with open(path, 'w') as f:
                    json.dump(record, f, indent=2)

            records = sorted(history_dir.glob("cycle_*.json"))
            for old in records[:-100]:
                old.unlink(missing_ok=True)
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.debug(f"Cycle logging failed: {e}")

    # =========================================================================
    # =========================================================================

    def _save_state(self):
        return _save_state_impl(self)

    def _load_state(self):
        return _load_state_impl(self)

    def reset_rotation(self):
        self._rotator.reset_processed()
        self._rotator.clear_old_failures()
        self._rotator.clear_pool()
        self._rotator.reset_discovery()
        self._save_state()
        log.info("Rotation reset")

    def reset_all(self):
        self._rotator = StockRotator()
        self._replay = ExperienceReplayBuffer()
        self._metrics = MetricTracker()
        self._set_holdout_codes([])
        self.progress = LearningProgress()
        self._save_state()
        log.info("Full reset")

    def get_stats(self) -> dict:
        return {
            'is_running': self.progress.is_running,
            'is_paused': self.progress.is_paused,
            'stage': self.progress.stage,
            'progress': self.progress.progress,
            'message': self.progress.message,
            'total_sessions': self.progress.total_training_sessions,
            'total_stocks': self.progress.total_stocks_learned,
            'total_hours': self.progress.total_training_hours,
            'best_accuracy': self.progress.best_accuracy_ever,
            'current_accuracy': self.progress.training_accuracy,
            'validation_accuracy': self.progress.validation_accuracy,
            'interval': self.progress.current_interval,
            'horizon': self.progress.current_horizon,
            'processed': self._rotator.processed_count,
            'pool_size': self._rotator.pool_size,
            'replay_size': len(self._replay),
            'holdout_size': len(self._holdout_codes),
            'accuracy_trend': self._metrics.trend,
            'accuracy_ema': self._metrics.ema,
            'plateau_count': self._metrics.plateau_count,
            'old_accuracy': self.progress.old_stock_accuracy,
            'old_confidence': self.progress.old_stock_confidence,
            'rejected': self.progress.model_was_rejected,
            'consecutive_rejections': self.progress.consecutive_rejections,
            'full_retrain_cycles_remaining': (
                self.progress.full_retrain_cycles_remaining
            ),
            'training_mode': self.progress.training_mode,
            'targeted_stocks': self.progress.targeted_stocks,
            'errors': self.progress.errors[-10:],
            'warnings': self.progress.warnings[-10:],
        }


ContinuousLearner._main_loop = _main_loop_impl
ContinuousLearner._run_cycle = _run_cycle_impl
ContinuousLearner._run_targeted_cycle = _run_targeted_cycle_impl

AutoLearner = ContinuousLearner
