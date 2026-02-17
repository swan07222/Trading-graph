# models/auto_learner.py

import hashlib
import json
import os
import random
import shutil
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import CONFIG
from data.fetcher import get_fetcher
from utils.cancellation import CancellationToken, CancelledException
from utils.logger import get_logger

log = get_logger(__name__)

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

_MAX_MESSAGES = 100  # Bound for error/warning lists (Issue 11)

@dataclass
class LearningProgress:
    """Track learning progress across all cycles"""
    stage: str = "idle"
    progress: float = 0.0
    message: str = ""
    stocks_found: int = 0
    stocks_processed: int = 0
    stocks_total: int = 0
    training_epoch: int = 0
    training_total_epochs: int = 0
    training_accuracy: float = 0.0
    validation_accuracy: float = 0.0
    is_running: bool = False
    is_paused: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)

    total_training_sessions: int = 0
    total_stocks_learned: int = 0
    total_training_hours: float = 0.0
    best_accuracy_ever: float = 0.0
    current_interval: str = "1m"
    current_horizon: int = 30

    processed_count: int = 0
    pool_size: int = 0

    old_stock_accuracy: float = 0.0
    old_stock_confidence: float = 0.0
    model_was_rejected: bool = False

    accuracy_trend: str = "stable"  # improving, stable, degrading
    plateau_count: int = 0

    training_mode: str = "auto"  # "auto" or "targeted"
    targeted_stocks: list[str] = field(default_factory=list)

    # --- bounded message helpers (Issue 11) ---

    def add_error(self, msg: str):
        self.errors.append(msg)
        if len(self.errors) > _MAX_MESSAGES:
            self.errors = self.errors[-_MAX_MESSAGES:]

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        if len(self.warnings) > _MAX_MESSAGES:
            self.warnings = self.warnings[-_MAX_MESSAGES:]

    def reset(self):
        self.stage = "idle"
        self.progress = 0.0
        self.message = ""
        self.stocks_processed = 0
        self.training_epoch = 0
        self.is_running = False
        self.is_paused = False
        self.errors = []
        self.warnings = []
        self.model_was_rejected = False
        self.training_mode = "auto"
        self.targeted_stocks = []

    def to_dict(self) -> dict:
        """FIX M5: Include all fields in serialization."""
        return {
            'stage': self.stage,
            'progress': self.progress,
            'message': self.message,
            'stocks_found': self.stocks_found,
            'stocks_processed': self.stocks_processed,
            'stocks_total': self.stocks_total,
            'training_epoch': self.training_epoch,
            'training_total_epochs': self.training_total_epochs,
            'training_accuracy': self.training_accuracy,
            'validation_accuracy': self.validation_accuracy,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'total_sessions': self.total_training_sessions,
            'total_stocks_learned': self.total_stocks_learned,
            'total_training_hours': self.total_training_hours,
            'best_accuracy': self.best_accuracy_ever,
            'interval': self.current_interval,
            'horizon': self.current_horizon,
            'processed_count': self.processed_count,
            'pool_size': self.pool_size,
            'old_stock_accuracy': self.old_stock_accuracy,
            'old_stock_confidence': self.old_stock_confidence,
            'accuracy_trend': self.accuracy_trend,
            'plateau_count': self.plateau_count,
            'model_was_rejected': self.model_was_rejected,
            'training_mode': self.training_mode,
            'targeted_stocks': self.targeted_stocks[:20],
            'errors': self.errors[-10:],  # Last 10 for UI
            'warnings': self.warnings[-10:],
        }

class MetricTracker:
    """
    Tracks accuracy trend with exponential moving average.
    Detects improvement, plateau, and degradation.

    FIX M4: Thread-safe access to _plateau_count.
    """

    def __init__(self, window: int = 10, plateau_threshold: float = 0.005):
        self._history: deque = deque(maxlen=window)
        self._window = window
        self._plateau_threshold = plateau_threshold
        self._plateau_count = 0
        self._best_ema: float = 0.0
        self._lock = threading.Lock()

    def record(self, accuracy: float):
        """Record a new accuracy measurement"""
        with self._lock:
            self._history.append(accuracy)

    @property
    def trend(self) -> str:
        """Current trend: improving, stable, degrading"""
        with self._lock:
            if len(self._history) < 3:
                return "stable"

            recent = list(self._history)
            mid = len(recent) // 2
            first_half = np.mean(recent[:mid])
            second_half = np.mean(recent[mid:])
            diff = second_half - first_half

            if diff > self._plateau_threshold:
                return "improving"
            elif diff < -self._plateau_threshold:
                return "degrading"
            return "stable"

    @property
    def ema(self) -> float:
        """Exponential moving average of accuracy"""
        with self._lock:
            if not self._history:
                return 0.0
            alpha = 2.0 / (len(self._history) + 1)
            ema = self._history[0]
            for val in list(self._history)[1:]:
                ema = alpha * val + (1 - alpha) * ema
            return float(ema)

    @property
    def is_plateau(self) -> bool:
        """Check if accuracy has plateaued"""
        with self._lock:
            return self._is_plateau_unlocked()

    @property
    def plateau_count(self) -> int:
        """FIX M4: Thread-safe access to plateau count."""
        with self._lock:
            return self._plateau_count

    def get_plateau_response(self) -> dict[str, Any]:
        """Graduated response to plateau."""
        with self._lock:
            if not self._is_plateau_unlocked():
                self._plateau_count = 0
                return {'action': 'none'}

            self._plateau_count += 1
            count = self._plateau_count

        if count <= 2:
            return {
                'action': 'increase_epochs',
                'factor': 1.5,
                'message': (
                    f"Plateau level 1 (count={count}): "
                    f"increasing epochs"
                ),
            }
        elif count <= 4:
            return {
                'action': 'reset_rotation',
                'message': (
                    f"Plateau level 2 (count={count}): "
                    f"resetting rotation"
                ),
            }
        elif count <= 6:
            return {
                'action': 'increase_diversity',
                'message': (
                    f"Plateau level 3 (count={count}): "
                    f"more diverse stocks"
                ),
            }
        else:
            with self._lock:
                self._plateau_count = 0
            return {
                'action': 'full_reset',
                'lr_boost': 2.0,
                'message': "Plateau level 4: full reset with LR boost",
            }

    def _is_plateau_unlocked(self) -> bool:
        """Check plateau without acquiring lock (caller must hold lock)."""
        if len(self._history) < self._window:
            return False
        spread = max(self._history) - min(self._history)
        return spread < self._plateau_threshold

    def to_dict(self) -> dict:
        with self._lock:
            return {
                'history': list(self._history),
                'plateau_count': self._plateau_count,
                'best_ema': self._best_ema,
            }

    def from_dict(self, data: dict):
        with self._lock:
            self._history = deque(
                data.get('history', []), maxlen=self._window
            )
            self._plateau_count = data.get('plateau_count', 0)
            self._best_ema = data.get('best_ema', 0.0)

class ExperienceReplayBuffer:
    """
    Stores trained stock codes with cached sequences.
    Bounded size, cache TTL, stratified sampling.

    FIX M3: sample() logs warning when returning fewer items than requested.
    FIX SAMP: sample() handles edge cases (n=0, empty strata).
    """

    def __init__(
        self,
        max_size: int = 2000,
        cache_dir: Path = None,
        cache_ttl_hours: float = 72.0,
    ):
        self.max_size = max_size
        self._buffer: list[str] = []
        self._performance: dict[str, float] = {}
        self._cache_times: dict[str, float] = {}
        self._cache_ttl = cache_ttl_hours * 3600
        self._lock = threading.Lock()

        self._cache_dir = cache_dir or CONFIG.DATA_DIR / "replay_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def add(self, codes: list[str], confidence: float = 0.5):
        """Add successfully trained codes"""
        with self._lock:
            for code in codes:
                if code not in self._buffer:
                    self._buffer.append(code)
                self._performance[code] = confidence

            if len(self._buffer) > self.max_size:
                removed = self._buffer[: len(self._buffer) - self.max_size]
                self._buffer = self._buffer[-self.max_size:]
                for code in removed:
                    self._performance.pop(code, None)
                    self._remove_cache(code)

    def sample(self, n: int) -> list[str]:
        """
        Stratified sampling.

        FIX M3: Log warning if returning fewer than n items.
        FIX SAMP: Handle edge cases properly.
        """
        with self._lock:
            if not self._buffer or n <= 0:
                return []

            n = min(n, len(self._buffer))
            if n >= len(self._buffer):
                return list(self._buffer)

            top = [
                c for c in self._buffer
                if self._performance.get(c, 0.5) >= 0.6
            ]
            mid = [
                c for c in self._buffer
                if 0.4 <= self._performance.get(c, 0.5) < 0.6
            ]
            low = [
                c for c in self._buffer
                if self._performance.get(c, 0.5) < 0.4
            ]

            n_top = max(1, int(n * 0.3))
            n_mid = max(1, int(n * 0.4))
            n_low = max(0, n - n_top - n_mid)

            def safe_sample(pool, count):
                count = min(count, len(pool))
                return random.sample(pool, count) if count > 0 else []

            selected = (
                safe_sample(top, n_top)
                + safe_sample(mid, n_mid)
                + safe_sample(low, n_low)
            )

            remaining = n - len(selected)
            if remaining > 0:
                selected_set = set(selected)
                available = [
                    c for c in self._buffer if c not in selected_set
                ]
                selected.extend(safe_sample(available, remaining))

            # FIX M3: Log warning if returning fewer than requested
            if len(selected) < n:
                log.warning(
                    f"ExperienceReplayBuffer.sample(): requested {n}, "
                    f"returning {len(selected)} (buffer has {len(self._buffer)})"
                )

            return selected

    def cache_sequences(self, code: str, X: np.ndarray, y: np.ndarray):
        """Cache training sequences with timestamp"""
        try:
            path = self._cache_dir / f"{code}.npz"
            np.savez_compressed(path, X=X, y=y)
            with self._lock:
                self._cache_times[code] = time.time()
        except Exception as e:
            log.debug(f"Cache save failed for {code}: {e}")

    def get_cached_sequences(
        self, code: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Load cached sequences if not stale"""
        try:
            path = self._cache_dir / f"{code}.npz"
            if not path.exists():
                return None
            with self._lock:
                cached_at = self._cache_times.get(code, 0)
            if time.time() - cached_at > self._cache_ttl:
                path.unlink(missing_ok=True)
                return None
            data = np.load(path)
            return data['X'], data['y']
        except Exception:
            return None

    def get_cached_codes(self) -> list[str]:
        """Get codes with valid cache"""
        now = time.time()
        with self._lock:
            return [
                c for c in self._buffer
                if (self._cache_dir / f"{c}.npz").exists()
                and (now - self._cache_times.get(c, 0)) < self._cache_ttl
            ]

    def update_performance(self, code: str, confidence: float):
        with self._lock:
            if code in self._performance:
                old = self._performance[code]
                self._performance[code] = 0.7 * old + 0.3 * confidence

    def _remove_cache(self, code: str):
        try:
            path = self._cache_dir / f"{code}.npz"
            path.unlink(missing_ok=True)
            self._cache_times.pop(code, None)
        except Exception:
            pass

    def get_all(self) -> list[str]:
        with self._lock:
            return list(self._buffer)

    def __len__(self):
        with self._lock:
            return len(self._buffer)

    def to_dict(self) -> dict:
        with self._lock:
            return {
                'buffer': list(self._buffer[-self.max_size:]),
                'performance': dict(self._performance),
                'cache_times': {
                    k: v for k, v in self._cache_times.items()
                    if k in self._buffer
                },
            }

    def from_dict(self, data: dict):
        with self._lock:
            self._buffer = list(data.get('buffer', []))[-self.max_size:]
            self._performance = dict(data.get('performance', {}))
            self._cache_times = {
                k: float(v)
                for k, v in data.get('cache_times', {}).items()
            }

    def cleanup_stale_cache(self):
        now = time.time()
        try:
            for path in self._cache_dir.glob("*.npz"):
                code = path.stem
                cached_at = self._cache_times.get(code, 0)
                if now - cached_at > self._cache_ttl:
                    path.unlink(missing_ok=True)
        except Exception:
            pass

class ModelGuardian:
    """
    Protects best model from degradation.

    FIX C2: validate_model() loads exact model path instead of using discovery.
    FIX GUARD: validate_model() handles individual stock errors without
    aborting the entire validation.
    """

    def __init__(self, model_dir: Path = None, max_backups: int = 5):
        self.model_dir = model_dir or CONFIG.MODEL_DIR
        self._best_metrics: dict[str, float] = {}
        self._max_backups = max_backups
        self._lock = threading.Lock()
        self._holdout_codes: list[str] = []

    def set_holdout(self, codes: list[str]):
        with self._lock:
            self._holdout_codes = list(codes)

    def get_holdout(self) -> list[str]:
        with self._lock:
            return list(self._holdout_codes)

    def backup_current(self, interval: str, horizon: int) -> bool:
        with self._lock:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = self.model_dir / "backups" / timestamp
                backup_dir.mkdir(parents=True, exist_ok=True)
                files = self._model_files(interval, horizon)
                copied_any = False
                for filename in files:
                    src = self.model_dir / filename
                    if src.exists():
                        shutil.copy2(src, backup_dir / filename)
                        shutil.copy2(src, self.model_dir / f"{filename}.backup")
                        copied_any = True
                self._prune_backups()
                return copied_any
            except Exception as e:
                log.warning(f"Backup failed: {e}")
                return False

    def restore_backup(self, interval: str, horizon: int) -> bool:
        with self._lock:
            try:
                files = self._model_files(interval, horizon)
                restored = False
                for filename in files:
                    src = self.model_dir / f"{filename}.backup"
                    dst = self.model_dir / filename
                    if src.exists():
                        shutil.copy2(src, dst)
                        restored = True
                if restored:
                    log.info("Model restored from backup")
                return restored
            except Exception as e:
                log.error(f"Restore failed: {e}")
                return False

    def save_as_best(self, interval: str, horizon: int, metrics: dict) -> bool:
        with self._lock:
            try:
                for filename in self._model_files(interval, horizon):
                    src = self.model_dir / filename
                    dst = self.model_dir / f"{filename}.best"
                    if src.exists():
                        shutil.copy2(src, dst)

                metrics_path = (
                    self.model_dir / f"best_metrics_{interval}_{horizon}.json"
                )

                try:
                    from utils.atomic_io import atomic_write_json
                    atomic_write_json(metrics_path, metrics, use_lock=True)
                except ImportError:
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)

                self._best_metrics = dict(metrics)
                log.info(f"Saved as all-time best: acc={metrics.get('accuracy', 0):.1%}")
                return True
            except Exception as e:
                log.warning(f"Save best failed: {e}")
                return False

    def get_best_metrics(self, interval: str, horizon: int) -> dict:
        if self._best_metrics:
            return self._best_metrics
        try:
            path = self.model_dir / f"best_metrics_{interval}_{horizon}.json"
            if path.exists():
                with open(path) as f:
                    self._best_metrics = json.load(f)
                return self._best_metrics
        except Exception:
            pass
        return {}

    def validate_model(
        self, interval: str, horizon: int,
        validation_codes: list[str], lookback_bars: int,
        collect_samples: bool = False,
    ) -> dict[str, float]:
        """
        Validate model on holdout stocks.

        FIX C2: Loads the EXACT model file instead of using Predictor's
        discovery fallback which may load a different model.
        FIX GUARD: Individual stock errors don't abort validation.
        """
        _empty_result = {'accuracy': 0, 'avg_confidence': 0, 'predictions_made': 0}

        if not validation_codes:
            return _empty_result

        try:
            from data.features import FeatureEngine
            from data.processor import DataProcessor
            from models.ensemble import EnsembleModel

            model_path = self.model_dir / f"ensemble_{interval}_{horizon}.pt"
            scaler_path = self.model_dir / f"scaler_{interval}_{horizon}.pkl"

            if not model_path.exists():
                log.warning(f"Model file not found for validation: {model_path}")
                return _empty_result

            feature_engine = FeatureEngine()
            processor = DataProcessor()
            feature_cols = feature_engine.get_feature_columns()

            if scaler_path.exists():
                if not processor.load_scaler(str(scaler_path)):
                    log.warning(f"Failed to load scaler: {scaler_path}")
                    return _empty_result
            else:
                log.warning(f"Scaler not found: {scaler_path}")
                return _empty_result

            input_size = processor.n_features or len(feature_cols)
            ensemble = EnsembleModel(input_size=input_size)

            if not ensemble.load(str(model_path)):
                log.warning(f"Failed to load ensemble for validation: {model_path}")
                return _empty_result

            fetcher = get_fetcher()
            correct = 0
            total = 0
            confidences = []
            errors = 0
            samples: list[dict[str, Any]] = []

            for code in validation_codes:
                try:
                    df = fetcher.get_history(
                        code, interval=interval, bars=lookback_bars, use_cache=True,
                    )
                    if df is None or len(df) < CONFIG.SEQUENCE_LENGTH + horizon + 10:
                        continue

                    df = feature_engine.create_features(df)

                    missing = set(feature_cols) - set(df.columns)
                    if missing:
                        log.debug(f"Validation: {code} missing features: {missing}")
                        continue

                    cutoff = len(df) - horizon
                    if cutoff < CONFIG.SEQUENCE_LENGTH:
                        continue

                    pred_df = df.iloc[:cutoff].copy()
                    future_df = df.iloc[cutoff:].copy()

                    if len(future_df) == 0 or 'close' not in future_df.columns:
                        continue

                    X = processor.prepare_inference_sequence(pred_df, feature_cols)
                    ensemble_pred = ensemble.predict(X)

                    price_at = float(pred_df['close'].iloc[-1])
                    price_after = float(future_df['close'].iloc[-1])

                    if price_at <= 0:
                        continue

                    ret_pct = (price_after / price_at - 1) * 100

                    if ret_pct >= CONFIG.UP_THRESHOLD:
                        actual = 2
                    elif ret_pct <= CONFIG.DOWN_THRESHOLD:
                        actual = 0
                    else:
                        actual = 1

                    if ensemble_pred.predicted_class == actual:
                        correct += 1
                    total += 1
                    confidences.append(float(ensemble_pred.confidence))
                    if collect_samples:
                        probs = getattr(ensemble_pred, "probabilities", np.array([0.33, 0.34, 0.33]))
                        prob_down = float(probs[0]) if len(probs) > 0 else 0.33
                        prob_up = float(probs[2]) if len(probs) > 2 else 0.33
                        samples.append(
                            {
                                "code": str(code),
                                "actual": int(actual),
                                "predicted": int(ensemble_pred.predicted_class),
                                "confidence": float(ensemble_pred.confidence),
                                "agreement": float(getattr(ensemble_pred, "agreement", 1.0)),
                                "entropy": float(getattr(ensemble_pred, "entropy", 0.0)),
                                "prob_up": prob_up,
                                "prob_down": prob_down,
                                "future_return": float(ret_pct),
                            }
                        )

                except Exception as e:
                    errors += 1
                    log.debug(f"Validation error for {code}: {e}")
                    continue

            if errors > 0:
                log.debug(f"Validation: {errors} stocks had errors")

            if total == 0:
                return _empty_result

            result = {
                'accuracy': float(correct / total),
                'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
                'predictions_made': total,
                'coverage': total / max(len(validation_codes), 1),
                'errors': errors,
            }
            if collect_samples:
                result["samples"] = samples
            return result
        except Exception as e:
            log.warning(f"Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return _empty_result

    def _model_files(self, interval, horizon) -> list[str]:
        return [
            f"ensemble_{interval}_{horizon}.pt",
            f"forecast_{interval}_{horizon}.pt",
            f"scaler_{interval}_{horizon}.pkl",
        ]

    def _prune_backups(self):
        try:
            backup_root = self.model_dir / "backups"
            if not backup_root.exists():
                return
            backups = sorted(
                [d for d in backup_root.iterdir() if d.is_dir()], reverse=True,
            )
            for old in backups[self._max_backups:]:
                shutil.rmtree(old, ignore_errors=True)
        except Exception:
            pass

class StockRotator:
    """
    Manages stock discovery and rotation.

    FIX PRIV: Provides public methods for state migration in _load_state
    instead of requiring direct private attribute access.
    """

    def __init__(self):
        self._processed: set[str] = set()
        self._failed: dict[str, int] = {}
        self._fail_max = 3
        self._pool: list[str] = []
        self._last_discovery: float = 0.0
        self._discovery_ttl: float = 90.0
        self._new_listing_probe_ttl: float = 5.0
        self._last_listing_probe: float = 0.0
        self._last_network_state: bool | None = None

    def discover_new(
        self, max_stocks: int, min_market_cap: float,
        stop_check: Callable, progress_cb: Callable,
    ) -> list[str]:
        self._maybe_refresh_pool(max_stocks, min_market_cap, stop_check, progress_cb)
        self._maybe_probe_new_listings(stop_check, progress_cb)
        if not self._pool:
            self._pool = list(CONFIG.STOCK_POOL)

        available = [c for c in self._pool if c not in self._processed]
        available = [c for c in available if self._failed.get(c, 0) < self._fail_max]

        if not available:
            log.info(f"All {len(self._processed)} stocks processed. Resetting rotation.")
            self._processed.clear()
            self._failed.clear()
            available = list(self._pool)

        never_tried = [c for c in available if c not in self._failed]
        retries = [c for c in available if c in self._failed]
        ordered = never_tried + retries
        return ordered[:max_stocks]

    @staticmethod
    def _norm_code(raw: str) -> str:
        code = "".join(ch for ch in str(raw or "").strip() if ch.isdigit())
        return code.zfill(6) if code else ""

    def mark_processed(self, codes: list[str]):
        for code in codes:
            self._processed.add(code)

    def mark_failed(self, code: str):
        self._failed[code] = self._failed.get(code, 0) + 1

    def clear_old_failures(self):
        self._failed.clear()

    def reset_processed(self):
        """Clear processed set 鈥?used by plateau handler."""
        self._processed.clear()

    def reset_discovery(self):
        """Force pool re-discovery on next call."""
        self._last_discovery = 0

    def clear_pool(self):
        """Clear the stock pool 鈥?used by reset_rotation()."""
        self._pool.clear()

    # FIX PRIV: Public methods for state migration from old format
    def set_processed(self, codes: set[str]):
        """Set processed codes from loaded state."""
        self._processed = set(codes)

    def set_failed(self, failed: dict[str, int]):
        """Set failed codes from loaded state."""
        self._failed = dict(failed)

    def _maybe_refresh_pool(self, max_stocks, min_market_cap, stop_check, progress_cb):
        del max_stocks, min_market_cap  # Pool refresh is universe-wide.
        now = time.time()
        expired = (now - self._last_discovery) > self._discovery_ttl

        network_changed = False
        try:
            from core.network import get_network_env
            env = get_network_env()
            current_state = env.is_china_direct
            if self._last_network_state is not None and current_state != self._last_network_state:
                network_changed = True
            self._last_network_state = current_state
        except Exception:
            pass

        if self._pool and not expired and not network_changed:
            return

        if callable(stop_check) and stop_check():
            return

        try:
            from data.universe import get_new_listings, get_universe_codes
            universe = get_universe_codes(
                force_refresh=bool(expired or network_changed),
                max_age_hours=max(5.0 / 3600.0, self._discovery_ttl / 3600.0),
            )
            new_listed = get_new_listings(
                days=120,
                force_refresh=bool(expired or network_changed),
            )
        except Exception:
            universe, new_listed = [], []

        if len(list(universe or [])) < 1200:
            try:
                from data.discovery import UniversalStockDiscovery

                discovered = UniversalStockDiscovery().discover_all(
                    callback=None,
                    max_stocks=None,
                    min_market_cap=0,
                    include_st=True,
                )
                if discovered:
                    discovered_codes = [
                        self._norm_code(getattr(row, "code", ""))
                        for row in discovered
                    ]
                    discovered_codes = [c for c in discovered_codes if c]
                    universe = list(universe or []) + discovered_codes
            except Exception:
                pass

        pool = []
        seen = set()
        for c in (new_listed or []):
            code = self._norm_code(c)
            if code and code not in seen:
                seen.add(code)
                pool.append(code)
        for c in (universe or []):
            code = self._norm_code(c)
            if code and code not in seen:
                seen.add(code)
                pool.append(code)
        if not pool:
            pool = list(CONFIG.STOCK_POOL)

        head = pool[:256]
        tail = pool[256:]
        random.shuffle(tail)
        self._pool = head + tail
        self._last_discovery = now
        if callable(progress_cb):
            try:
                progress_cb("Universe refreshed", len(self._pool))
            except Exception:
                pass
        log.info(f"Pool refreshed (universe-first): {len(self._pool)} stocks")

    def _maybe_probe_new_listings(self, stop_check: Callable, progress_cb: Callable) -> None:
        """Fast probe to inject newly listed symbols between full refreshes."""
        now = time.time()
        if (now - self._last_listing_probe) < self._new_listing_probe_ttl:
            return
        self._last_listing_probe = now

        if callable(stop_check) and stop_check():
            return

        try:
            from data.universe import get_new_listings
            new_listed = get_new_listings(days=14, force_refresh=True)
        except Exception:
            return

        if not new_listed:
            return

        seen = set(self._pool)
        injected: list[str] = []
        for c in new_listed:
            code = self._norm_code(c)
            if not code or code in seen:
                continue
            seen.add(code)
            injected.append(code)
        if not injected:
            return

        self._pool = injected + self._pool
        if callable(progress_cb):
            try:
                progress_cb("New listings added to pool", len(self._pool))
            except Exception:
                pass
        log.info("Injected %d newly listed stocks into discovery pool", len(injected))

    @property
    def processed_count(self) -> int:
        return len(self._processed)

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    def get_pool_snapshot(self) -> list[str]:
        return list(self._pool)

    def to_dict(self) -> dict:
        return {
            'processed': list(self._processed),
            'failed': dict(self._failed),
            'pool': list(self._pool),
            'last_discovery': self._last_discovery,
            'last_listing_probe': self._last_listing_probe,
        }

    def from_dict(self, data: dict):
        self._processed = {
            self._norm_code(c)
            for c in data.get('processed', [])
            if self._norm_code(c)
        }
        failed = data.get('failed', {})
        if isinstance(failed, list):
            self._failed = {
                self._norm_code(c): 1
                for c in failed
                if self._norm_code(c)
            }
        else:
            self._failed = {}
            for k, v in failed.items():
                code = self._norm_code(k)
                if not code:
                    continue
                try:
                    self._failed[code] = int(v)
                except Exception:
                    self._failed[code] = 1
        raw_pool = list(data.get('pool', []) or [])
        seen: set[str] = set()
        pool: list[str] = []
        for c in raw_pool:
            code = self._norm_code(c)
            if not code or code in seen:
                continue
            seen.add(code)
            pool.append(code)
        self._pool = pool
        self._last_discovery = data.get('last_discovery', 0.0)
        self._last_listing_probe = float(data.get('last_listing_probe', 0.0) or 0.0)

class LRScheduler:
    """Learning rate with warmup + decay + plateau boost."""

    def __init__(
        self, base_lr: float = None, decay_rate: float = 0.05,
        warmup_cycles: int = 2, min_lr_ratio: float = 0.05,
    ):
        self._base_lr = base_lr or CONFIG.model.learning_rate
        self._decay_rate = decay_rate
        self._warmup_cycles = warmup_cycles
        self._min_lr = self._base_lr * min_lr_ratio
        self._boost: float = 1.0

    def get_lr(self, cycle: int, incremental: bool) -> float:
        if not incremental or cycle <= 0:
            return self._base_lr

        if cycle <= self._warmup_cycles:
            warmup = 0.5 + 0.5 * (cycle / self._warmup_cycles)
            lr = self._base_lr * warmup
        else:
            effective_cycle = cycle - self._warmup_cycles
            decay = max(
                self._min_lr / self._base_lr,
                1.0 / (1.0 + self._decay_rate * effective_cycle),
            )
            lr = self._base_lr * decay

        lr *= self._boost
        self._boost = max(1.0, self._boost * 0.9)
        return max(lr, self._min_lr)

    def apply_boost(self, factor: float):
        self._boost = float(factor)
        log.info(f"LR boost applied: {factor}x")

class ParallelFetcher:
    """
    Fetch stock data with thread pool and proper rate limiting.

    FIX FETCH: Handles empty codes list without error.
    """

    def __init__(self, max_workers: int = 5):
        self._max_workers = max_workers

    def fetch_batch(
        self,
        codes: list[str],
        interval: str,
        lookback: int,
        min_bars: int,
        stop_check: Callable,
        progress_cb: Callable,
    ) -> tuple[list[str], list[str]]:
        """
        Fetch data for multiple stocks in parallel.
        Returns (ok_codes, failed_codes).
        """
        # FIX FETCH: Handle empty codes list
        if not codes:
            return [], []

        fetcher = get_fetcher()
        ok_codes: list[str] = []
        failed_codes: list[str] = []
        completed = 0
        lock = threading.Lock()

        # Rate-limit parameters based on interval
        if interval in ("1m", "5m", "15m", "30m"):
            delay = 0.8
            max_concurrent = 2
        elif interval in ("60m", "1h"):
            delay = 0.4
            max_concurrent = 3
        else:
            delay = 0.2
            max_concurrent = 5

        # VPN + Yahoo mode is sensitive to burst traffic; use gentler fetch pacing.
        try:
            from core.network import get_network_env
            env = get_network_env()
            if interval not in ("1m", "2m", "5m", "15m", "30m", "60m", "1h") and env.is_vpn_active:
                delay = max(delay, 0.9)
                max_concurrent = min(max_concurrent, 2)
        except Exception:
            pass

        semaphore = threading.Semaphore(max_concurrent)

        def fetch_one(code: str) -> tuple[str, bool]:
            if stop_check():
                return code, False

            try:
                from data.fetcher import BARS_PER_DAY
                bpd = float(BARS_PER_DAY.get(str(interval).lower(), 1))
            except ImportError:
                bpd = 1.0

            # Keep warm cache aligned to the 7-day intraday training window.
            min_cache_bars = int(max(7 * bpd, 7))

            with semaphore:
                time.sleep(delay)
                try:
                    df = fetcher.get_history(
                        code,
                        interval=interval,
                        bars=max(int(lookback), int(min_cache_bars)),
                        use_cache=True,
                        update_db=True,
                    )
                    if df is not None and not df.empty and len(df) >= min_bars:
                        return code, True
                    return code, False
                except Exception as e:
                    log.debug(f"Fetch failed for {code}: {e}")
                    return code, False

        workers = min(self._max_workers, max_concurrent, len(codes))
        workers = max(1, workers)

        executor = ThreadPoolExecutor(max_workers=workers)
        futures: dict = {}
        cancelled_early = False
        try:
            for code in codes:
                if stop_check():
                    cancelled_early = True
                    break
                futures[executor.submit(fetch_one, code)] = code

            for future in as_completed(list(futures.keys())):
                if stop_check():
                    cancelled_early = True
                    break

                try:
                    code, success = future.result(timeout=120)
                except Exception:
                    code = futures[future]
                    success = False

                with lock:
                    if success:
                        ok_codes.append(code)
                    else:
                        failed_codes.append(code)
                    completed += 1

                progress_cb(
                    f"Fetched {completed}/{len(codes)} stocks",
                    completed,
                )
        finally:
            if cancelled_early:
                for f in list(futures.keys()):
                    try:
                        f.cancel()
                    except Exception:
                        pass
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=False)
            else:
                executor.shutdown(wait=True)

        return ok_codes, failed_codes

# CONTINUOUS LEARNER (Main Class)

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

    def __init__(self):
        self.progress = LearningProgress()
        self._cancel_token = CancellationToken()
        self._thread: threading.Thread | None = None
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
            except Exception:
                pass

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
            time.sleep(1)
        return self._should_stop()

    # =========================================================================
    # LIFECYCLE 鈥?AUTO MODE
    # =========================================================================

    def start(
        self, mode="full", max_stocks=None, epochs_per_cycle=10,
        min_market_cap=10, include_all_markets=True, continuous=True,
        learning_while_trading=True, interval="1m", prediction_horizon=30,
        lookback_bars=None, cycle_interval_seconds=900, incremental=True,
        priority_stock_codes: list[str] | None = None,
    ):
        requested_interval = str(interval or "1m").strip().lower()
        interval = "1m"
        if requested_interval != "1m":
            log.info(
                "Training interval locked to 1m (requested=%s)",
                requested_interval,
            )
        if self._thread and self._thread.is_alive():
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
        except Exception:
            pass

        self._thread = threading.Thread(
            target=self._main_loop,
            args=(
                mode, max_stocks or 200, max(1, int(epochs_per_cycle)),
                float(min_market_cap), bool(include_all_markets),
                bool(continuous), str(interval).lower(),
                int(prediction_horizon), int(lookback_bars),
                int(cycle_interval_seconds), bool(incremental),
                list(priority_stock_codes or []),
            ),
            daemon=True,
        )
        self._thread.start()

    def _compute_lookback_bars(self, interval: str) -> int:
        """Compute default lookback bars for an interval."""
        try:
            from data.fetcher import BARS_PER_DAY, INTERVAL_MAX_DAYS
            bpd = BARS_PER_DAY.get(str(interval).lower(), 1)
            max_d = INTERVAL_MAX_DAYS.get(str(interval).lower(), 500)
        except ImportError:
            bpd = self._BARS_PER_DAY_FALLBACK.get(str(interval).lower(), 1)
            max_d = self._INTERVAL_MAX_DAYS_FALLBACK.get(str(interval).lower(), 500)
        iv = str(interval).lower()
        is_intraday = iv in ("1m", "2m", "5m", "15m", "30m", "60m", "1h")
        target_days = min(int(max_d), 7) if is_intraday else min(int(max_d), 365)
        bars = int(max(1, round(float(bpd) * float(target_days))))
        if is_intraday:
            return max(120, bars)
        return min(max(200, bars), 3000)

    @staticmethod
    def _interval_seconds(interval: str) -> int:
        """Map interval token to seconds for continuity checks."""
        iv = str(interval or "1m").strip().lower()
        mapping = {
            "1m": 60,
            "2m": 120,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "60m": 3600,
            "1h": 3600,
            "1d": 86400,
            "1wk": 604800,
            "1mo": 2592000,
        }
        return int(mapping.get(iv, 60))

    def _session_continuous_window_seconds(
        self,
        code: str,
        interval: str,
        max_bars: int = 5000,
    ) -> float:
        """
        Longest continuous cached window for a symbol/interval in seconds.
        Uses session bars captured during trading, including partial bars.
        """
        try:
            from data.session_cache import get_session_bar_cache
            cache = get_session_bar_cache()
            df = cache.read_history(
                symbol=code,
                interval=interval,
                bars=max(10, int(max_bars)),
                final_only=False,
            )
        except Exception:
            return 0.0

        if df is None or df.empty:
            return 0.0

        step = float(max(1, self._interval_seconds(interval)))
        buckets: list[int] = []
        try:
            for ts in df.index.tolist():
                try:
                    ep = float(ts.timestamp())
                except Exception:
                    continue
                if not np.isfinite(ep):
                    continue
                buckets.append(int(ep // step))
        except Exception:
            return 0.0

        if not buckets:
            return 0.0

        uniq = sorted(set(buckets))
        run = 1
        longest = 1
        for i in range(1, len(uniq)):
            if (uniq[i] - uniq[i - 1]) <= 1:
                run += 1
            else:
                longest = max(longest, run)
                run = 1
        longest = max(longest, run)

        return float(max(0, longest - 1) * step)

    def _filter_priority_session_codes(
        self,
        codes: list[str],
        interval: str,
        min_seconds: float = 3600.0,
    ) -> list[str]:
        """
        Keep only session-priority symbols with enough continuous captured data.
        For intraday training this enforces >=1 hour of contiguous session bars.
        """
        iv = str(interval or "").strip().lower()
        intraday = {"1m", "2m", "3m", "5m", "15m", "30m", "60m", "1h"}

        dedup: list[str] = []
        seen: set[str] = set()
        for raw in (codes or []):
            c = str(raw).strip()
            if not c or c in seen:
                continue
            seen.add(c)
            dedup.append(c)

        if iv not in intraday:
            return dedup

        filtered: list[str] = []
        dropped = 0
        for c in dedup:
            span_s = self._session_continuous_window_seconds(c, iv)
            if span_s >= float(min_seconds):
                filtered.append(c)
            else:
                dropped += 1

        if dropped > 0:
            self.progress.add_warning(
                f"Skipped {dropped} session-priority stocks without >=1h continuous {iv} bars"
            )
        return filtered

    @staticmethod
    def _norm_code(raw: str) -> str:
        code = "".join(c for c in str(raw or "").strip() if c.isdigit())
        return code.zfill(6) if code else ""

    def _prioritize_codes_by_news(
        self,
        codes: list[str],
        interval: str,
        max_probe: int = 16,
    ) -> list[str]:
        """
        Reorder candidate symbols by fresh market/stock news relevance.
        Keeps original order for ties and when news is unavailable.
        """
        ordered = [self._norm_code(c) for c in list(codes or [])]
        ordered = [c for c in ordered if c]
        if len(ordered) <= 1:
            return ordered

        try:
            from data.news import get_news_aggregator
            agg = get_news_aggregator()
        except Exception:
            return ordered

        candidate_set = set(ordered)
        scores: dict[str, float] = {c: 0.0 for c in ordered}
        now = datetime.now()

        try:
            market_news = agg.get_market_news(count=80, force_refresh=False)
        except Exception:
            market_news = []

        for item in list(market_news or []):
            linked = {
                self._norm_code(x)
                for x in list(getattr(item, "stock_codes", []) or [])
            }
            linked = {c for c in linked if c and c in candidate_set}
            if not linked:
                continue
            try:
                age_h = max(
                    0.0,
                    (now - getattr(item, "publish_time", now)).total_seconds() / 3600.0,
                )
            except Exception:
                age_h = 24.0
            recency = 1.0 / (1.0 + (age_h / 10.0))
            sentiment_mag = abs(float(getattr(item, "sentiment_score", 0.0) or 0.0))
            importance = float(getattr(item, "importance", 0.5) or 0.5)
            weight = recency * max(0.2, min(1.6, importance)) * (0.40 + sentiment_mag)
            for code in linked:
                scores[code] = float(scores.get(code, 0.0) + weight)

        # Optional light probe for top unseen candidates to capture stock-specific headlines.
        probed = 0
        for code in ordered:
            if self._should_stop():
                break
            if probed >= int(max(0, max_probe)):
                break
            if scores.get(code, 0.0) > 0.0:
                continue
            try:
                summary = agg.get_sentiment_summary(code)
            except Exception:
                continue
            count = int(summary.get("total", 0) or 0)
            if count <= 0:
                continue
            conf = float(summary.get("confidence", 0.0) or 0.0)
            sent = abs(float(summary.get("overall_sentiment", 0.0) or 0.0))
            momentum = abs(float(summary.get("sentiment_momentum_6h", 0.0) or 0.0))
            scores[code] = float((0.45 * sent + 0.25 * momentum + 0.30 * conf) * min(1.0, count / 12.0))
            probed += 1

        ranked = sorted(
            enumerate(ordered),
            key=lambda it: (-float(scores.get(it[1], 0.0)), it[0]),
        )
        out = [code for _, code in ranked]

        moved = sum(1 for i, code in enumerate(out) if i < len(ordered) and code != ordered[i])
        if moved > 0:
            self._update(
                message=(
                    f"News-prioritized candidates: {sum(1 for v in scores.values() if v > 0):d} "
                    f"stocks with signal"
                ),
                progress=max(3.0, float(self.progress.progress)),
            )
        return out

    def run(self, **kwargs):
        kwargs.setdefault('continuous', False)
        self.start(**kwargs)
        if self._thread:
            self._thread.join()
        return self.progress

    def stop(self, join_timeout: float = 30.0):
        log.info("Stopping learning...")
        self._cancel_token.cancel()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            timeout = max(0.5, float(join_timeout))
            thread.join(timeout=timeout)
            if thread.is_alive():
                log.info("Learning thread still finalizing after stop request")
        self._save_state()
        self.progress.is_running = False
        self._notify()

    def pause(self):
        self.progress.is_paused = True
        self._notify()

    def resume(self):
        self.progress.is_paused = False
        self._notify()

    # =========================================================================
    # LIFECYCLE 鈥?TARGETED MODE
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
        """Train on specific user-selected stocks instead of random rotation."""
        requested_interval = str(interval or "1m").strip().lower()
        interval = "1m"
        if requested_interval != "1m":
            log.info(
                "Targeted training interval locked to 1m (requested=%s)",
                requested_interval,
            )
        if not stock_codes:
            log.warning("No stock codes provided for targeted training")
            return

        if self._thread and self._thread.is_alive():
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

        self._thread = threading.Thread(
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
            daemon=True,
        )
        self._thread.start()

    def run_targeted(self, **kwargs):
        """Run targeted training synchronously (blocking)."""
        kwargs.setdefault('continuous', False)
        self.start_targeted(**kwargs)
        if self._thread:
            self._thread.join()
        return self.progress

    # =========================================================================
    # STOCK VALIDATION (used by UI search)
    # =========================================================================

    def validate_stock_code(
        self, code: str, interval: str = "1m"
    ) -> dict[str, Any]:
        """Validate that a stock code exists and has sufficient data."""
        code = str(code).strip()
        if not code:
            return {
                'valid': False, 'code': code, 'name': '', 'bars': 0,
                'message': 'Empty stock code',
            }

        try:
            fetcher = get_fetcher()
            bars_for_interval = max(
                300,
                int(self._compute_lookback_bars(interval)),
            )
            df = fetcher.get_history(
                code, interval=interval, bars=bars_for_interval, use_cache=True,
            )

            if df is None or df.empty:
                return {
                    'valid': False, 'code': code, 'name': '', 'bars': 0,
                    'message': f'No data found for {code}',
                }

            bars = len(df)
            min_bars = CONFIG.SEQUENCE_LENGTH + 20

            if bars < min_bars:
                return {
                    'valid': False, 'code': code, 'name': '', 'bars': bars,
                    'message': (
                        f'Insufficient data: {bars} bars '
                        f'(need at least {min_bars})'
                    ),
                }

            name = ''
            try:
                from data.fetcher import get_spot_cache
                spot = get_spot_cache()
                quote = spot.get_quote(code)
                if quote and quote.get('name'):
                    name = str(quote['name'])
            except Exception:
                pass

            return {
                'valid': True, 'code': code, 'name': name, 'bars': bars,
                'message': f'OK 鈥?{bars} bars available',
            }

        except Exception as e:
            return {
                'valid': False, 'code': code, 'name': '', 'bars': 0,
                'message': f'Validation error: {str(e)[:200]}',
            }

    # =========================================================================
    # MAIN LOOP 鈥?AUTO MODE
    # =========================================================================

    def _main_loop(
        self, mode, max_stocks, epochs, min_market_cap, include_all,
        continuous, interval, horizon, lookback, cycle_seconds, incremental,
        priority_stock_codes,
    ):
        cycle = 0
        current_epochs = epochs

        try:
            while not self._should_stop():
                cycle += 1
                self._update(
                    stage="cycle_start",
                    message=(
                        f"=== Cycle {cycle} | Learned: {len(self._replay)} | "
                        f"Best: {self.progress.best_accuracy_ever:.1%} | "
                        f"Trend: {self._metrics.trend} ==="
                    ),
                    progress=0.0,
                    stocks_processed=0,
                    training_epoch=0,
                    training_total_epochs=max(1, int(current_epochs)),
                    validation_accuracy=0.0,
                )

                if self._wait_if_paused():
                    break

                plateau = self._metrics.get_plateau_response()
                if plateau['action'] != 'none':
                    current_epochs, incremental = self._handle_plateau(
                        plateau, current_epochs, incremental
                    )

                success = self._run_cycle(
                    max_stocks=max_stocks, epochs=current_epochs,
                    min_market_cap=min_market_cap, interval=interval,
                    horizon=horizon, lookback=lookback,
                    incremental=incremental, cycle_number=cycle,
                    priority_stock_codes=priority_stock_codes,
                )

                if success:
                    self.progress.total_training_sessions += 1
                self._save_state()

                if not continuous:
                    break

                if cycle % 5 == 0:
                    self._rotator.clear_old_failures()
                    self._replay.cleanup_stale_cache()

                self._update(
                    stage="waiting",
                    message=f"Cycle {cycle} done. Next in {cycle_seconds}s...",
                    progress=100.0,
                )
                self._interruptible_sleep(cycle_seconds)

        except CancelledException:
            log.info("Learning cancelled")
        except Exception as e:
            log.error(f"Learning error: {e}")
            import traceback
            traceback.print_exc()
            self.progress.add_error(str(e))
        finally:
            self.progress.is_running = False
            self._save_state()
            self._notify()

    # =========================================================================
    # MAIN LOOP 鈥?TARGETED MODE
    # =========================================================================

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
        except Exception as e:
            log.error(f"Targeted learning error: {e}")
            import traceback
            traceback.print_exc()
            self.progress.add_error(str(e))
        finally:
            self.progress.is_running = False
            self._save_state()
            self._notify()

    def _interruptible_sleep(self, seconds: int):
        """Sleep for up to `seconds`, checking cancellation frequently."""
        deadline = time.monotonic() + max(0.0, float(seconds))
        while not self._should_stop():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.2, remaining))

    def _handle_plateau(
        self, plateau: dict, current_epochs: int, incremental: bool,
    ) -> tuple[int, bool]:
        """Graduated plateau response 鈥?uses public rotator methods."""
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
    # SINGLE CYCLE 鈥?AUTO MODE
    # =========================================================================

    def _run_cycle(
        self, max_stocks, epochs, min_market_cap, interval,
        horizon, lookback, incremental, cycle_number,
        priority_stock_codes: list[str] | None = None,
    ) -> bool:
        start_time = datetime.now()

        try:
            # === 1. Resolve interval ===
            eff_interval, eff_horizon, eff_lookback, min_bars = (
                self._resolve_interval(interval, horizon, lookback)
            )

            # === 2. Setup holdout ===
            self._ensure_holdout(eff_interval, eff_lookback, min_bars, cycle_number)
            if self._should_stop():
                raise CancelledException()

            # === 3. Discover new stocks ===
            self._update(stage="discovering", progress=2.0, message="Discovering stocks...")
            new_codes = self._rotator.discover_new(
                max_stocks=max_stocks, min_market_cap=min_market_cap,
                stop_check=self._should_stop,
                progress_cb=lambda msg, cnt: self._update(message=msg, stocks_found=cnt),
            )
            if self._should_stop():
                raise CancelledException()

            holdout_set = self._get_holdout_set()
            new_codes = [c for c in new_codes if c not in holdout_set]

            if priority_stock_codes:
                usable_priority = self._filter_priority_session_codes(
                    list(priority_stock_codes),
                    eff_interval,
                    min_seconds=3600.0,
                )
                prioritized = []
                seen = set(new_codes)
                for code in usable_priority:
                    c = str(code).strip()
                    if not c or c in holdout_set or c in seen:
                        continue
                    prioritized.append(c)
                    seen.add(c)
                if prioritized:
                    self._update(
                        message=f"Injecting {len(prioritized)} priority session stocks",
                        progress=4.0,
                    )
                    new_codes = prioritized + new_codes

            if new_codes:
                try:
                    new_codes = self._prioritize_codes_by_news(
                        new_codes,
                        eff_interval,
                        max_probe=min(16, int(max_stocks)),
                    )
                except Exception as e:
                    log.debug("News prioritization skipped: %s", e)

            # Recovery: if holdout filters everything and replay is empty,
            # reset rotation/holdout once and re-discover.
            if not new_codes and len(self._replay) == 0:
                self._update(
                    message="No candidates after holdout; resetting rotation/holdout",
                    progress=3.0,
                )
                self._rotator.reset_processed()
                self._rotator.reset_discovery()
                self._set_holdout_codes([])
                new_codes = self._rotator.discover_new(
                    max_stocks=max_stocks, min_market_cap=min_market_cap,
                    stop_check=self._should_stop,
                    progress_cb=lambda msg, cnt: self._update(
                        message=msg, stocks_found=cnt,
                    ),
                )
                if self._should_stop():
                    raise CancelledException()
                holdout_set = self._get_holdout_set()
                new_codes = [c for c in new_codes if c not in holdout_set]

            # === 4. Mix with replay ===
            total_learned = len(self._replay)
            if total_learned < 20:
                new_ratio = 0.9
            elif total_learned < 100:
                new_ratio = 0.7
            elif total_learned < 500:
                new_ratio = 0.5
            else:
                new_ratio = 0.3

            num_new = max(3, int(max_stocks * new_ratio))
            num_replay = max_stocks - num_new

            new_batch = new_codes[:num_new]
            replay_batch = self._replay.sample(num_replay)
            replay_batch = [
                c for c in replay_batch
                if c not in new_batch and c not in holdout_set
            ]
            codes = new_batch + replay_batch

            # In VPN mode, large batches can overwhelm upstream providers.
            try:
                from core.network import get_network_env
                env = get_network_env()
                if env.is_vpn_active and len(codes) > 30:
                    codes = codes[:30]
                    self.progress.add_warning(
                        "VPN mode: batch capped to 30 stocks for fetch stability"
                    )
            except Exception:
                pass

            if not codes:
                self._update(stage="error", message="No stocks available")
                return False

            self.progress.stocks_found = len(codes)
            self.progress.stocks_total = len(codes)
            self.progress.processed_count = self._rotator.processed_count
            self.progress.pool_size = self._rotator.pool_size

            self._update(
                message=f"Batch: {len(new_batch)} new + {len(replay_batch)} replay",
                progress=5.0,
            )

            # === 5. Fetch data ===
            self._update(
                stage="downloading", progress=10.0,
                message=f"Fetching {eff_interval} data...",
                stocks_processed=0,
            )

            ok_codes, failed_codes = self._fetcher.fetch_batch(
                codes, eff_interval, eff_lookback, min_bars,
                stop_check=self._should_stop,
                progress_cb=lambda msg, cnt: self._update(
                    message=msg, stocks_processed=cnt,
                    progress=10.0 + 30.0 * (cnt / max(len(codes), 1)),
                ),
            )
            if self._should_stop():
                raise CancelledException()

            min_ok = max(3, int(len(codes) * 0.05))
            try:
                from core.network import get_network_env
                if get_network_env().is_vpn_active:
                    min_ok = max(2, int(len(codes) * 0.03))
            except Exception:
                pass
            if len(ok_codes) < min_ok and failed_codes and not self._should_stop():
                relaxed_min_bars = max(CONFIG.SEQUENCE_LENGTH + 20, int(min_bars * 0.7))
                retry_codes = failed_codes[: min(len(failed_codes), max(8, min_ok * 2))]
                retry_base_processed = int(max(0, len(ok_codes)))
                self._update(
                    message=(
                        f"Retrying {len(retry_codes)} failed stocks "
                        f"(relaxed min bars {relaxed_min_bars})"
                    ),
                    progress=36.0,
                )
                retry_ok, retry_failed = self._fetcher.fetch_batch(
                    retry_codes, eff_interval, eff_lookback, relaxed_min_bars,
                    stop_check=self._should_stop,
                    progress_cb=lambda msg, cnt: self._update(
                        message=f"Retry pass: {msg}",
                        stocks_processed=min(
                            len(codes),
                            retry_base_processed + int(cnt),
                        ),
                        progress=36.0 + 4.0 * (cnt / max(len(retry_codes), 1)),
                    ),
                )
                if self._should_stop():
                    raise CancelledException()
                ok_set = set(ok_codes)
                for code in retry_ok:
                    if code not in ok_set:
                        ok_codes.append(code)
                        ok_set.add(code)
                failed_codes = [
                    c for c in failed_codes
                    if c not in set(retry_ok) and c in set(retry_failed)
                ] + [c for c in failed_codes if c not in retry_codes]

            for code in failed_codes:
                self._rotator.mark_failed(code)

            if len(ok_codes) < min_ok:
                for code in new_batch:
                    self._rotator.mark_processed([code])
                self._update(
                    stage="error",
                    message=f"Too few stocks: {len(ok_codes)}/{len(codes)}",
                )
                return False

            # === 6. Backup model ===
            self._update(stage="backup", progress=42.0, message="Backing up current model...")
            self._guardian.backup_current(eff_interval, eff_horizon)
            if self._should_stop():
                raise CancelledException()

            # === 7. Pre-training validation ===
            pre_val = None
            holdout_snapshot = list(self._get_holdout_set())
            if holdout_snapshot and len(self._replay) > 10:
                self._update(message="Pre-training validation...", progress=45.0)
                pre_val = self._guardian.validate_model(
                    eff_interval, eff_horizon, holdout_snapshot, eff_lookback,
                )
                log.info(f"Pre-validation: {pre_val}")
                if self._should_stop():
                    raise CancelledException()

            # === 8. Train ===
            lr = self._lr_scheduler.get_lr(cycle_number, incremental)
            self._update(
                stage="training", progress=50.0,
                message=f"Training {len(ok_codes)} stocks (lr={lr:.6f}, e={epochs})...",
                training_total_epochs=epochs,
            )

            result = self._train(
                ok_codes, epochs, eff_interval, eff_horizon,
                eff_lookback, incremental, lr,
            )

            if result.get("status") == "cancelled":
                raise CancelledException()

            acc = float(result.get("best_accuracy", 0.0))
            self.progress.training_accuracy = acc
            self._metrics.record(acc)
            self.progress.accuracy_trend = self._metrics.trend

            # === 9. Post-training validation ===
            self._update(
                stage="validating", progress=90.0,
                message="Validating on holdout stocks...",
            )

            accepted = self._validate_and_decide(
                eff_interval, eff_horizon, eff_lookback, pre_val, acc,
            )

            # === 10. Update state ===
            self._finalize_cycle(
                accepted, ok_codes, new_batch, replay_batch,
                eff_interval, eff_horizon, eff_lookback,
                acc, cycle_number, start_time,
            )

            return accepted

        except CancelledException:
            raise
        except Exception as e:
            log.error(f"Cycle error: {e}")
            import traceback
            traceback.print_exc()
            self._update(stage="error", message=str(e))
            self.progress.add_error(str(e))
            return False

    # =========================================================================
    # SINGLE CYCLE 鈥?TARGETED MODE
    # =========================================================================

    def _run_targeted_cycle(
        self,
        stock_codes: list[str],
        epochs: int,
        interval: str,
        horizon: int,
        lookback: int,
        incremental: bool,
        cycle_number: int,
    ) -> bool:
        """Single training cycle on user-specified stocks."""
        start_time = datetime.now()

        try:
            # === 1. Resolve interval ===
            eff_interval, eff_horizon, eff_lookback, min_bars = (
                self._resolve_interval(interval, horizon, lookback)
            )

            # === 2. Setup holdout ===
            self._ensure_holdout(eff_interval, eff_lookback, min_bars, cycle_number)
            if self._should_stop():
                raise CancelledException()

            holdout_set = self._get_holdout_set()
            train_codes = [c for c in stock_codes if c not in holdout_set]

            if not train_codes:
                self.progress.add_warning(
                    "All selected stocks overlap with holdout set 鈥?"
                    "training on them anyway"
                )
                train_codes = list(stock_codes)

            self.progress.stocks_found = len(train_codes)
            self.progress.stocks_total = len(train_codes)

            self._update(
                stage="targeted_training",
                message=f"Targeted batch: {len(train_codes)} stocks",
                progress=5.0,
            )

            # === 3. Fetch data ===
            self._update(
                stage="downloading",
                progress=10.0,
                message=f"Fetching {eff_interval} data for {len(train_codes)} stocks...",
                stocks_processed=0,
            )

            ok_codes, failed_codes = self._fetcher.fetch_batch(
                train_codes, eff_interval, eff_lookback, min_bars,
                stop_check=self._should_stop,
                progress_cb=lambda msg, cnt: self._update(
                    message=msg,
                    stocks_processed=cnt,
                    progress=10.0 + 30.0 * (cnt / max(len(train_codes), 1)),
                ),
            )
            if self._should_stop():
                raise CancelledException()

            if not ok_codes and failed_codes and not self._should_stop():
                relaxed_min_bars = max(CONFIG.SEQUENCE_LENGTH + 20, int(min_bars * 0.7))
                retry_codes = failed_codes[: min(len(failed_codes), 12)]
                retry_base_processed = int(max(0, len(ok_codes)))
                self._update(
                    message=(
                        f"Retrying targeted fetch for {len(retry_codes)} stocks "
                        f"(min bars {relaxed_min_bars})"
                    ),
                    progress=36.0,
                )
                retry_ok, retry_failed = self._fetcher.fetch_batch(
                    retry_codes, eff_interval, eff_lookback, relaxed_min_bars,
                    stop_check=self._should_stop,
                    progress_cb=lambda msg, cnt: self._update(
                        message=f"Retry pass: {msg}",
                        stocks_processed=min(
                            len(train_codes),
                            retry_base_processed + int(cnt),
                        ),
                        progress=36.0 + 4.0 * (cnt / max(len(retry_codes), 1)),
                    ),
                )
                if self._should_stop():
                    raise CancelledException()
                ok_codes = retry_ok or ok_codes
                failed_codes = retry_failed + [c for c in failed_codes if c not in retry_codes]

            if failed_codes:
                failed_display = ', '.join(failed_codes[:10])
                extra = f" (+{len(failed_codes) - 10} more)" if len(failed_codes) > 10 else ""
                self.progress.add_warning(
                    f"Failed to fetch: {failed_display}{extra}"
                )

            if not ok_codes:
                self._update(
                    stage="error",
                    message=(
                        f"No valid data for any of the {len(train_codes)} stocks. "
                        f"Check codes and network connection."
                    ),
                )
                self.progress.add_error(
                    f"All {len(train_codes)} stocks failed data fetch"
                )
                return False

            # === 4. Backup model ===
            self._update(
                stage="backup", progress=42.0,
                message="Backing up current model...",
            )
            self._guardian.backup_current(eff_interval, eff_horizon)
            if self._should_stop():
                raise CancelledException()

            # === 5. Pre-training validation ===
            pre_val = None
            holdout_snapshot = list(self._get_holdout_set())
            if holdout_snapshot and len(self._replay) > 10:
                self._update(message="Pre-training validation...", progress=45.0)
                pre_val = self._guardian.validate_model(
                    eff_interval, eff_horizon, holdout_snapshot, eff_lookback,
                )
                log.info(f"Pre-validation: {pre_val}")
                if self._should_stop():
                    raise CancelledException()

            # === 6. Train ===
            lr = self._lr_scheduler.get_lr(cycle_number, incremental)
            self._update(
                stage="training",
                progress=50.0,
                message=(
                    f"Training on {len(ok_codes)} targeted stocks "
                    f"(lr={lr:.6f}, epochs={epochs})..."
                ),
                training_total_epochs=epochs,
            )

            result = self._train(
                ok_codes, epochs, eff_interval, eff_horizon,
                eff_lookback, incremental, lr,
            )

            if result.get("status") == "cancelled":
                raise CancelledException()

            acc = float(result.get("best_accuracy", 0.0))
            self.progress.training_accuracy = acc
            self._metrics.record(acc)
            self.progress.accuracy_trend = self._metrics.trend

            # === 7. Post-training validation ===
            self._update(
                stage="validating",
                progress=90.0,
                message="Validating on holdout stocks...",
            )

            accepted = self._validate_and_decide(
                eff_interval, eff_horizon, eff_lookback, pre_val, acc,
            )

            # === 8. Update state ===
            self._finalize_cycle(
                accepted, ok_codes, ok_codes, [],
                eff_interval, eff_horizon, eff_lookback,
                acc, cycle_number, start_time,
            )

            return accepted

        except CancelledException:
            raise
        except Exception as e:
            log.error(f"Targeted cycle error: {e}")
            import traceback
            traceback.print_exc()
            self._update(stage="error", message=str(e))
            self.progress.add_error(str(e))
            return False

    # =========================================================================
    # SHARED CYCLE FINALIZATION (DRY)
    # =========================================================================

    def _finalize_cycle(
        self, accepted: bool,
        ok_codes: list[str], new_batch: list[str], replay_batch: list[str],
        interval: str, horizon: int, lookback: int,
        acc: float, cycle_number: int, start_time: datetime,
    ):
        """Shared logic for finalizing a training cycle (auto or targeted)."""
        mode = self.progress.training_mode

        if accepted:
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
                    f"鉁?{label}Cycle {cycle_number}: acc={acc:.1%}, "
                    f"{len(ok_codes)} trained, "
                    f"total={len(self._replay)} | ACCEPTED"
                ),
            )
        else:
            self.progress.model_was_rejected = True
            label = "Targeted " if mode == "targeted" else ""
            self._update(
                stage="complete", progress=100.0,
                message=(
                    f"鈿狅笍 {label}Cycle {cycle_number}: acc={acc:.1%} | "
                    f"REJECTED 鈥?previous model restored"
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
        eff_interval = "1m"
        if req_interval != "1m":
            log.info(
                "Resolved training interval forced to 1m (requested=%s)",
                req_interval,
            )
        eff_horizon = horizon

        bpd = BARS_PER_DAY.get(eff_interval, 1)
        max_avail = int(INTERVAL_MAX_DAYS.get(eff_interval, 500) * bpd)
        eff_lookback = min(lookback, max_avail)

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
                df = fetcher.get_history(
                    code, interval=interval, bars=lookback, use_cache=True,
                )
                if df is not None and len(df) >= min_bars:
                    new_holdout.append(code)
            except Exception:
                continue

        if self._should_stop():
            raise CancelledException()

        if len(new_holdout) < min_required:
            log.warning("Failed to build new holdout set 鈥?keeping existing")
            return

        # Atomic check-and-swap
        with self._lock:
            current_holdout_set = set(self._holdout_codes)
            if current_holdout_set != old_holdout_set and self._holdout_codes:
                log.debug("Holdout already updated by another thread 鈥?skipping")
                return
            self._holdout_codes = new_holdout

        self._guardian.set_holdout(new_holdout)
        log.info(f"Holdout set: {len(new_holdout)} stocks (30% of {pool_size})")

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
            log.info("No holdout validation 鈥?accepting")
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
            except Exception:
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
        except Exception as e:
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
                except Exception:
                    continue

            log.debug(f"Cached sequences for {cached}/{min(len(codes), 30)} stocks")
        except Exception as e:
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
        except Exception as e:
            log.debug(f"Cycle logging failed: {e}")

    # =========================================================================
    # =========================================================================

    def _save_state(self):
        """Persist learner state atomically."""
        state = {
            'version': 3,
            'total_sessions': self.progress.total_training_sessions,
            'total_stocks': self.progress.total_stocks_learned,
            'total_hours': self.progress.total_training_hours,
            'best_accuracy': self.progress.best_accuracy_ever,
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

        except Exception as e:
            log.warning(f"State save failed: {e}")

    def _load_state(self):
        """
        Load learner state from disk.

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
                    log.warning("State file checksum mismatch 鈥?may be corrupted")
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
            last_interval = str(state.get('last_interval', '1m')).strip().lower()
            if last_interval != "1m":
                log.info(
                    "Learner state interval %s ignored; using locked 1m",
                    last_interval,
                )
            self.progress.current_interval = "1m"
            try:
                last_h = int(state.get('last_horizon', 30))
            except Exception:
                last_h = 30
            self.progress.current_horizon = (
                max(1, last_h) if last_interval == "1m" else 30
            )

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
                f"holdout={len(self._holdout_codes)}"
            )
        except Exception as e:
            log.warning(f"State load failed: {e}")

    # =========================================================================
    # =========================================================================

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
            'training_mode': self.progress.training_mode,
            'targeted_stocks': self.progress.targeted_stocks,
            'errors': self.progress.errors[-10:],
            'warnings': self.progress.warnings[-10:],
        }

AutoLearner = ContinuousLearner
