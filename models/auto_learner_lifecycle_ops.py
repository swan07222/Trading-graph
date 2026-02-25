from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime

from utils.cancellation import CancellationToken, CancelledException
from utils.logger import get_logger
from utils.recoverable import JSON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)
_AUTO_LEARNER_RECOVERABLE_EXCEPTIONS = JSON_RECOVERABLE_EXCEPTIONS


def start(
    self, mode="full", max_stocks=None, epochs_per_cycle=10,
    min_market_cap=10, include_all_markets=True, continuous=True,
    learning_while_trading=True, interval="1m", prediction_horizon=30,
    lookback_bars=None, cycle_interval_seconds=60, incremental=True,
    priority_stock_codes: list[str] | None = None,
) -> None:
    requested_interval = str(interval or "1m").strip().lower()
    interval = self._normalize_interval_token(requested_interval)
    if interval != requested_interval:
        log.info(
            "Training interval %s is unsupported; falling back to %s",
            requested_interval,
            interval,
        )
    with self._thread_lock:
        existing_thread = self._thread
        if existing_thread and existing_thread.is_alive():
            if self.progress.is_paused:
                self.resume()
                return
            log.warning("Learning already in progress")
            return

    self._cancel_token = CancellationToken()
    self.progress.reset()
    self.progress.is_running = True
    self.progress.training_mode = "auto"
    self.progress.current_interval = str(interval)
    self.progress.current_horizon = int(prediction_horizon)

    if lookback_bars is None:
        lookback_bars = self._compute_lookback_bars(interval)

    try:
        from core.network import get_network_env, invalidate_network_cache
        invalidate_network_cache()
        get_network_env(force_refresh=True)
    except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
        log.debug("Network cache refresh skipped before learning start: %s", e)

    thread = threading.Thread(
        target=self._main_loop,
        args=(
            mode, max_stocks or 200, max(1, int(epochs_per_cycle)),
            float(min_market_cap), bool(include_all_markets),
            bool(continuous), str(interval).lower(),
            int(prediction_horizon), int(lookback_bars),
            int(cycle_interval_seconds), bool(incremental),
            list(priority_stock_codes or []),
        ),
        daemon=False,
        name="auto_learner_main",
    )
    with self._thread_lock:
        self._thread = thread
    thread.start()


def start_targeted(
    self,
    stock_codes: list[str],
    epochs_per_cycle: int = 10,
    interval: str = "1m",
    prediction_horizon: int = 30,
    lookback_bars: int | None = None,
    incremental: bool = True,
    continuous: bool = False,
    cycle_interval_seconds: int = 60,
) -> None:
    """Train on specific user-selected stocks instead of random rotation."""
    requested_interval = str(interval or "1m").strip().lower()
    interval = self._normalize_interval_token(requested_interval)
    if interval != requested_interval:
        log.info(
            "Targeted interval %s is unsupported; falling back to %s",
            requested_interval,
            interval,
        )
    if not stock_codes:
        log.warning("No stock codes provided for targeted training")
        return

    with self._thread_lock:
        existing_thread = self._thread
        if existing_thread and existing_thread.is_alive():
            if self.progress.is_paused:
                self.resume()
                return
            log.warning("Learning already in progress")
            return

    self._cancel_token = CancellationToken()
    self.progress.reset()
    self.progress.is_running = True
    self.progress.training_mode = "targeted"
    self.progress.current_interval = str(interval)
    self.progress.current_horizon = int(prediction_horizon)

    if lookback_bars is None:
        lookback_bars = self._compute_lookback_bars(interval)

    clean_codes = []
    seen = set()
    for code in stock_codes:
        code = str(code).strip()
        if code and code not in seen:
            seen.add(code)
            clean_codes.append(code)

    self.progress.targeted_stocks = clean_codes[:50]

    log.info(
        f"Starting targeted training on {len(clean_codes)} stocks: "
        f"{clean_codes[:10]}{'...' if len(clean_codes) > 10 else ''}"
    )

    thread = threading.Thread(
        target=self._targeted_loop,
        args=(
            clean_codes,
            max(1, int(epochs_per_cycle)),
            str(interval).lower(),
            int(prediction_horizon),
            int(lookback_bars),
            bool(incremental),
            bool(continuous),
            int(cycle_interval_seconds),
        ),
        daemon=False,
        name="auto_learner_targeted",
    )
    with self._thread_lock:
        self._thread = thread
    thread.start()

def run_targeted(self, **kwargs):
    """Run targeted training synchronously (blocking)."""
    kwargs.setdefault('continuous', False)
    self.start_targeted(**kwargs)
    # Join the currently active targeted thread while tolerating thread
    # replacement races from concurrent start calls.
    while True:
        with self._thread_lock:
            thread = self._thread
        if (
            thread is None
            or not thread.is_alive()
            or str(getattr(thread, "name", "")) != "auto_learner_targeted"
        ):
            break
        thread.join(timeout=0.5)
    return self.progress

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
) -> None:
    """Main loop for targeted training."""
    cycle = 0

    try:
        while not self._should_stop():
            cycle += 1
            self._update(
                stage="cycle_start",
                message=(
                    f"=== Targeted Cycle {cycle} | "
                    f"Stocks: {len(stock_codes)} | "
                    f"Best: {self.progress.best_accuracy_ever:.1%} ==="
                ),
                progress=0.0,
                stocks_processed=0,
                training_epoch=0,
                training_total_epochs=max(1, int(epochs)),
                validation_accuracy=0.0,
            )

            if self._wait_if_paused():
                break

            success = self._run_targeted_cycle(
                stock_codes=stock_codes,
                epochs=epochs,
                interval=interval,
                horizon=horizon,
                lookback=lookback,
                incremental=incremental,
                cycle_number=cycle,
            )

            if success:
                self.progress.total_training_sessions += 1
            self._save_state()

            if not continuous:
                break

            self._update(
                stage="waiting",
                message=f"Targeted cycle {cycle} done. Next in {cycle_seconds}s...",
                progress=100.0,
            )
            self._interruptible_sleep(cycle_seconds)

    except CancelledException:
        log.info("Targeted learning cancelled")
    except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
        log.error(f"Targeted learning error: {e}")
        import traceback
        traceback.print_exc()
        self.progress.add_error(str(e))
    except BaseException as e:
        log.error("Targeted learning fatal error: %s", e)
        self.progress.add_error(f"fatal:{type(e).__name__}:{e}")
        raise
    finally:
        self.progress.is_running = False
        self._save_state()
        self._notify()

def _interruptible_sleep(self, seconds: int) -> None:
    """Sleep for up to `seconds`, checking cancellation frequently."""
    deadline = time.monotonic() + max(0.0, float(seconds))
    while not self._should_stop():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if self._cancel_token.wait(timeout=min(0.2, remaining)):
            break

def _save_state(self) -> None:
    """Persist learner state atomically."""
    state = {
        'version': 4,
        'total_sessions': self.progress.total_training_sessions,
        'total_stocks': self.progress.total_stocks_learned,
        'total_hours': self.progress.total_training_hours,
        'best_accuracy': self.progress.best_accuracy_ever,
        'consecutive_rejections': self.progress.consecutive_rejections,
        'full_retrain_cycles_remaining': (
            self.progress.full_retrain_cycles_remaining
        ),
        'rotator': self._rotator.to_dict(),
        'replay': self._replay.to_dict(),
        'metrics': self._metrics.to_dict(),
        'holdout_codes': list(self._get_holdout_set()),
        'last_interval': self.progress.current_interval,
        'last_horizon': self.progress.current_horizon,
        'last_save': datetime.now().isoformat(),
    }

    try:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        data_str = json.dumps(state, indent=2, sort_keys=True)
        checksum = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        envelope = {'_checksum': checksum, '_data': state}

        try:
            from utils.atomic_io import atomic_write_json
            atomic_write_json(
                self.state_path, envelope, indent=2, use_lock=True
            )
        except ImportError:
            envelope_str = json.dumps(envelope, indent=2, sort_keys=True)
            tmp = self.state_path.with_suffix('.json.tmp')
            with open(tmp, 'w') as f:
                f.write(envelope_str)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, self.state_path)

    except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
        log.warning(f"State save failed: {e}")

def _load_state(self) -> None:
    """Load learner state from disk.

    FIX PRIV: Uses public StockRotator methods for state migration
    instead of directly accessing private attributes.
    """
    if not self.state_path.exists():
        return
    try:
        try:
            from utils.atomic_io import read_json
            raw = read_json(self.state_path)
        except ImportError:
            with open(self.state_path) as f:
                raw = json.load(f)

        if '_data' in raw and '_checksum' in raw:
            state = raw['_data']
            saved_checksum = raw['_checksum']
            data_str = json.dumps(state, indent=2, sort_keys=True)
            expected = hashlib.sha256(data_str.encode()).hexdigest()[:16]
            if saved_checksum != expected:
                log.warning("State file checksum mismatch - may be corrupted")
        elif '_checksum' in raw:
            raw_copy = dict(raw)
            raw_copy.pop('_checksum', None)
            state = raw_copy
        else:
            state = raw

        self.progress.total_training_sessions = state.get('total_sessions', 0)
        self.progress.total_stocks_learned = state.get('total_stocks', 0)
        self.progress.total_training_hours = state.get('total_hours', 0.0)
        self.progress.best_accuracy_ever = state.get('best_accuracy', 0.0)
        self.progress.consecutive_rejections = int(
            max(0, state.get('consecutive_rejections', 0) or 0)
        )
        self.progress.full_retrain_cycles_remaining = int(
            max(0, state.get('full_retrain_cycles_remaining', 0) or 0)
        )
        raw_last_interval = str(state.get('last_interval', '1m')).strip().lower()
        last_interval = self._normalize_interval_token(raw_last_interval)
        if last_interval != raw_last_interval:
            log.info(
                "Learner state interval %s unsupported; using %s",
                raw_last_interval,
                last_interval,
            )
        self.progress.current_interval = last_interval
        try:
            last_h = int(state.get('last_horizon', 30))
        except (TypeError, ValueError):
            last_h = 30
        self.progress.current_horizon = max(1, last_h)

        rotator_data = state.get('rotator', {})
        if rotator_data:
            self._rotator.from_dict(rotator_data)
        replay_data = state.get('replay', {})
        if replay_data:
            self._replay.from_dict(replay_data)
        metrics_data = state.get('metrics', {})
        if metrics_data:
            self._metrics.from_dict(metrics_data)

        self._set_holdout_codes(state.get('holdout_codes', []))

        # FIX PRIV: Migrate old format using public methods
        if state.get('version', 1) < 3:
            old_processed = state.get('processed_stocks', [])
            old_failed = state.get('failed_stocks', {})
            if old_processed or old_failed:
                self._rotator.set_processed(set(old_processed))
                if isinstance(old_failed, list):
                    self._rotator.set_failed({c: 1 for c in old_failed})
                else:
                    self._rotator.set_failed(
                        {k: int(v) for k, v in old_failed.items()}
                    )
            old_replay = state.get('replay_buffer', {})
            if old_replay and not replay_data:
                self._replay.from_dict(old_replay)

        log.info(
            f"State loaded (v{state.get('version', 1)}): "
            f"{self.progress.total_training_sessions} sessions, "
            f"best={self.progress.best_accuracy_ever:.1%}, "
            f"replay={len(self._replay)}, "
            f"processed={self._rotator.processed_count}, "
            f"holdout={len(self._holdout_codes)}, "
            f"reject_streak={self.progress.consecutive_rejections}"
        )
    except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
        log.warning(f"State load failed: {e}")
