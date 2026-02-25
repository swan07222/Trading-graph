from __future__ import annotations

import json
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
from utils.logger import get_logger
from utils.recoverable import JSON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)
_AUTO_LEARNER_RECOVERABLE_EXCEPTIONS = JSON_RECOVERABLE_EXCEPTIONS

_MAX_MESSAGES = 100  # Bound for error/warning lists (Issue 11)

@dataclass
class LearningProgress:
    """Track learning progress across all cycles."""
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
    consecutive_rejections: int = 0
    full_retrain_cycles_remaining: int = 0

    accuracy_trend: str = "stable"  # improving, stable, degrading
    plateau_count: int = 0

    training_mode: str = "auto"  # "auto" or "targeted"
    targeted_stocks: list[str] = field(default_factory=list)

    # --- bounded message helpers (Issue 11) ---
    # FIX: Use deque for O(1) append instead of O(n) list slicing
    _errors_deque: deque = field(default_factory=lambda: deque(maxlen=_MAX_MESSAGES), repr=False)
    _warnings_deque: deque = field(default_factory=lambda: deque(maxlen=_MAX_MESSAGES), repr=False)

    def add_error(self, msg: str) -> None:
        """Add error message with automatic bounds (O(1) operation)."""
        self._errors_deque.append(msg)
        self.errors = list(self._errors_deque)  # Sync with list for serialization

    def add_warning(self, msg: str) -> None:
        """Add warning message with automatic bounds (O(1) operation)."""
        self._warnings_deque.append(msg)
        self.warnings = list(self._warnings_deque)  # Sync with list for serialization

    def reset(self) -> None:
        self.stage = "idle"
        self.progress = 0.0
        self.message = ""
        self.stocks_processed = 0
        self.training_epoch = 0
        self.is_running = False
        self.is_paused = False
        self.errors = []
        self.warnings = []
        self._errors_deque.clear()
        self._warnings_deque.clear()
        self.model_was_rejected = False
        self.consecutive_rejections = 0
        self.full_retrain_cycles_remaining = 0
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
            'consecutive_rejections': self.consecutive_rejections,
            'full_retrain_cycles_remaining': self.full_retrain_cycles_remaining,
            'training_mode': self.training_mode,
            'targeted_stocks': self.targeted_stocks[:20],
            'errors': self.errors[-10:],  # Last 10 for UI
            'warnings': self.warnings[-10:],
        }

class MetricTracker:
    """Tracks accuracy trend with exponential moving average.
    Detects improvement, plateau, and degradation.

    FIX M4: Thread-safe access to _plateau_count.
    """

    def __init__(self, window: int = 10, plateau_threshold: float = 0.005) -> None:
        self._history: deque = deque(maxlen=window)
        self._window = window
        self._plateau_threshold = plateau_threshold
        self._plateau_count = 0
        self._best_ema: float = 0.0
        self._lock = threading.Lock()

    def record(self, accuracy: float) -> None:
        """Record a new accuracy measurement."""
        with self._lock:
            self._history.append(accuracy)

    @property
    def trend(self) -> str:
        """Current trend: improving, stable, degrading."""
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
        """Exponential moving average of accuracy."""
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
        """Check if accuracy has plateaued."""
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

    def from_dict(self, data: dict) -> None:
        with self._lock:
            self._history = deque(
                data.get('history', []), maxlen=self._window
            )
            self._plateau_count = data.get('plateau_count', 0)
            self._best_ema = data.get('best_ema', 0.0)

class ExperienceReplayBuffer:
    """Stores trained stock codes with cached sequences.
    Bounded size, cache TTL, stratified sampling.

    FIX M3: sample() logs warning when returning fewer items than requested.
    FIX SAMP: sample() handles edge cases (n=0, empty strata).
    """

    def __init__(
        self,
        max_size: int = 2000,
        cache_dir: Path = None,
        cache_ttl_hours: float = 72.0,
    ) -> None:
        self.max_size = max_size
        self._buffer: list[str] = []
        self._performance: dict[str, float] = {}
        self._cache_times: dict[str, float] = {}
        self._cache_ttl = cache_ttl_hours * 3600
        self._lock = threading.Lock()

        self._cache_dir = cache_dir or CONFIG.DATA_DIR / "replay_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def add(self, codes: list[str], confidence: float = 0.5) -> None:
        """Add successfully trained codes."""
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
        """Stratified sampling.

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

    def cache_sequences(self, code: str, X: np.ndarray, y: np.ndarray) -> None:
        """Cache training sequences with timestamp."""
        try:
            path = self._cache_dir / f"{code}.npz"
            np.savez_compressed(path, X=X, y=y)
            with self._lock:
                self._cache_times[code] = time.time()
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.debug(f"Cache save failed for {code}: {e}")

    def get_cached_sequences(
        self, code: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Load cached sequences if not stale."""
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
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Cache load failed for %s: %s", code, e)
            return None

    def get_cached_codes(self) -> list[str]:
        """Get codes with valid cache."""
        now = time.time()
        with self._lock:
            return [
                c for c in self._buffer
                if (self._cache_dir / f"{c}.npz").exists()
                and (now - self._cache_times.get(c, 0)) < self._cache_ttl
            ]

    def update_performance(self, code: str, confidence: float) -> None:
        with self._lock:
            if code in self._performance:
                old = self._performance[code]
                self._performance[code] = 0.7 * old + 0.3 * confidence

    def _remove_cache(self, code: str) -> None:
        try:
            path = self._cache_dir / f"{code}.npz"
            path.unlink(missing_ok=True)
            self._cache_times.pop(code, None)
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Cache cleanup failed for %s: %s", code, e)

    def get_all(self) -> list[str]:
        with self._lock:
            return list(self._buffer)

    def __len__(self) -> int:
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

    def from_dict(self, data: dict) -> None:
        with self._lock:
            self._buffer = list(data.get('buffer', []))[-self.max_size:]
            self._performance = dict(data.get('performance', {}))
            self._cache_times = {
                k: float(v)
                for k, v in data.get('cache_times', {}).items()
            }

    def cleanup_stale_cache(self) -> None:
        now = time.time()
        try:
            for path in self._cache_dir.glob("*.npz"):
                code = path.stem
                cached_at = self._cache_times.get(code, 0)
                if now - cached_at > self._cache_ttl:
                    path.unlink(missing_ok=True)
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Stale cache cleanup failed: %s", e)

class ModelGuardian:
    """Protects best model from degradation.

    FIX C2: validate_model() loads exact model path instead of using discovery.
    FIX GUARD: validate_model() handles individual stock errors without
    aborting the entire validation.
    """

    def __init__(self, model_dir: Path = None, max_backups: int = 5) -> None:
        self.model_dir = model_dir or CONFIG.MODEL_DIR
        self._best_metrics: dict[str, float] = {}
        self._max_backups = max_backups
        self._lock = threading.Lock()
        self._holdout_codes: list[str] = []

    def set_holdout(self, codes: list[str]) -> None:
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
            except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
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
            except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
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
            except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
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
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Failed loading best metrics for %s/%s: %s", interval, horizon, e)
        return {}

    def validate_model(
        self, interval: str, horizon: int,
        validation_codes: list[str], lookback_bars: int,
        collect_samples: bool = False,
    ) -> dict[str, float]:
        """Validate model on holdout stocks.

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
            covered_symbols = 0
            confidences = []
            errors = 0
            samples: list[dict[str, Any]] = []

            for code in validation_codes:
                symbol_samples = 0
                try:
                    try:
                        df = fetcher.get_history(
                            code,
                            interval=interval,
                            bars=lookback_bars,
                            use_cache=True,
                            update_db=True,
                            allow_online=True,
                            refresh_intraday_after_close=True,
                        )
                    except TypeError:
                        df = fetcher.get_history(
                            code, interval=interval, bars=lookback_bars, use_cache=True,
                        )
                    if df is None or len(df) < CONFIG.SEQUENCE_LENGTH + horizon + 10:
                        continue

                    if "close" not in df.columns:
                        continue

                    raw_df = df.copy()
                    raw_df["close"] = pd.to_numeric(raw_df["close"], errors="coerce")
                    raw_df = raw_df.replace([np.inf, -np.inf], np.nan).dropna(
                        subset=["close"]
                    )
                    if len(raw_df) < CONFIG.SEQUENCE_LENGTH + horizon + 10:
                        continue

                    max_anchor = int(len(raw_df) - horizon)
                    min_anchor = int(
                        max(
                            getattr(feature_engine, "MIN_ROWS", CONFIG.SEQUENCE_LENGTH),
                            CONFIG.SEQUENCE_LENGTH,
                        )
                    )
                    if max_anchor <= min_anchor:
                        continue

                    sample_count = int(min(5, max(1, max_anchor - min_anchor)))
                    anchors = sorted(
                        {
                            int(v)
                            for v in np.linspace(
                                min_anchor,
                                max_anchor - 1,
                                num=sample_count,
                                dtype=int,
                            )
                        }
                    )
                    if not anchors:
                        continue

                    for anchor in anchors:
                        hist_raw = raw_df.iloc[:anchor].copy()
                        fut_raw = raw_df.iloc[anchor: anchor + horizon].copy()
                        if (
                            len(hist_raw)
                            < getattr(feature_engine, "MIN_ROWS", CONFIG.SEQUENCE_LENGTH)
                            or len(fut_raw) < horizon
                        ):
                            continue

                        hist_feat = feature_engine.create_features(hist_raw)
                        missing = set(feature_cols) - set(hist_feat.columns)
                        if missing:
                            log.debug(f"Validation: {code} missing features: {missing}")
                            continue

                        X = processor.prepare_inference_sequence(hist_feat, feature_cols)
                        ensemble_pred = ensemble.predict(X)

                        price_at = float(pd.to_numeric(hist_raw["close"], errors="coerce").iloc[-1])
                        price_after = float(pd.to_numeric(fut_raw["close"], errors="coerce").iloc[-1])
                        if not np.isfinite(price_at) or not np.isfinite(price_after) or price_at <= 0:
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
                        symbol_samples += 1
                        confidences.append(float(ensemble_pred.confidence))
                        if collect_samples:
                            probs = getattr(
                                ensemble_pred,
                                "probabilities",
                                np.array([0.33, 0.34, 0.33]),
                            )
                            prob_down = float(probs[0]) if len(probs) > 0 else 0.33
                            prob_up = float(probs[2]) if len(probs) > 2 else 0.33
                            samples.append(
                                {
                                    "code": str(code),
                                    "anchor": int(anchor),
                                    "actual": int(actual),
                                    "predicted": int(ensemble_pred.predicted_class),
                                    "confidence": float(ensemble_pred.confidence),
                                    "agreement": float(
                                        getattr(ensemble_pred, "agreement", 1.0)
                                    ),
                                    "entropy": float(
                                        getattr(ensemble_pred, "entropy", 0.0)
                                    ),
                                    "prob_up": prob_up,
                                    "prob_down": prob_down,
                                    "future_return": float(ret_pct),
                                }
                            )

                    if symbol_samples > 0:
                        covered_symbols += 1

                except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
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
                'coverage': covered_symbols / max(len(validation_codes), 1),
                'errors': errors,
            }
            if collect_samples:
                result["samples"] = samples
            return result
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
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

    def _prune_backups(self) -> None:
        try:
            backup_root = self.model_dir / "backups"
            if not backup_root.exists():
                return
            backups = sorted(
                [d for d in backup_root.iterdir() if d.is_dir()], reverse=True,
            )
            for old in backups[self._max_backups:]:
                shutil.rmtree(old, ignore_errors=True)
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Backup prune failed: %s", e)

class StockRotator:
    """Manages stock discovery and rotation.

    FIX PRIV: Provides public methods for state migration in _load_state
    instead of requiring direct private attribute access.
    """

    def __init__(self) -> None:
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

    def mark_processed(self, codes: list[str]) -> None:
        for code in codes:
            self._processed.add(code)

    def mark_failed(self, code: str) -> None:
        self._failed[code] = self._failed.get(code, 0) + 1

    def clear_old_failures(self) -> None:
        self._failed.clear()

    def reset_processed(self) -> None:
        """Clear processed set - used by plateau handler."""
        self._processed.clear()

    def reset_discovery(self) -> None:
        """Force pool re-discovery on next call."""
        self._last_discovery = 0

    def clear_pool(self) -> None:
        """Clear the stock pool - used by reset_rotation()."""
        self._pool.clear()

    # FIX PRIV: Public methods for state migration from old format
    def set_processed(self, codes: set[str]) -> None:
        """Set processed codes from loaded state."""
        self._processed = set(codes)

    def set_failed(self, failed: dict[str, int]) -> None:
        """Set failed codes from loaded state."""
        self._failed = dict(failed)

    def _maybe_refresh_pool(self, max_stocks, min_market_cap, stop_check, progress_cb) -> None:
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
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Network state refresh failed during pool update: %s", e)

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
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.warning("Universe fetch failed; using fallback pool sources: %s", e)
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
            except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
                log.debug("Universal discovery fallback skipped: %s", e)

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
            except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
                log.debug("Progress callback failed on universe refresh: %s", e)
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
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.debug("New-listing probe failed: %s", e)
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
            except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
                log.debug("Progress callback failed on new-listing injection: %s", e)
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

    def from_dict(self, data: dict) -> None:
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
                except (TypeError, ValueError):
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
    ) -> None:
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

    def apply_boost(self, factor: float) -> None:
        self._boost = float(factor)
        log.info(f"LR boost applied: {factor}x")

class ParallelFetcher:
    """Fetch stock data with thread pool and proper rate limiting.

    FIX FETCH: Handles empty codes list without error.
    """

    def __init__(self, max_workers: int = 5) -> None:
        self._max_workers = max_workers

    def fetch_batch(
        self,
        codes: list[str],
        interval: str,
        lookback: int,
        min_bars: int,
        stop_check: Callable,
        progress_cb: Callable,
        *,
        allow_online: bool = True,
        update_db: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Fetch data for multiple stocks in parallel.
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
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Network-aware fetch pacing unavailable: %s", e)

        semaphore = threading.Semaphore(max_concurrent)

        def _paced_wait(seconds: float) -> bool:
            """Wait with cooperative cancellation checks.

            Returns True if stop was requested during the wait.
            """
            wait_s = max(0.0, float(seconds))
            if wait_s <= 0.0:
                return bool(stop_check())
            deadline = time.monotonic() + wait_s
            while True:
                if stop_check():
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                time.sleep(min(0.05, remaining))

        def fetch_one(code: str) -> tuple[str, bool]:
            if stop_check():
                return code, False

            try:
                from data.fetcher import BARS_PER_DAY
                bpd = float(BARS_PER_DAY.get(str(interval).lower(), 1))
            except ImportError:
                bpd = 1.0

            # Keep warm cache aligned to the 2-day intraday training window.
            min_cache_bars = int(max(2 * bpd, 2))

            with semaphore:
                if _paced_wait(delay):
                    return code, False
                try:
                    try:
                        df = fetcher.get_history(
                            code,
                            interval=interval,
                            bars=max(int(lookback), int(min_cache_bars)),
                            use_cache=True,
                            update_db=bool(update_db),
                            allow_online=bool(allow_online),
                            refresh_intraday_after_close=bool(
                                update_db and allow_online
                            ),
                        )
                    except TypeError:
                        df = fetcher.get_history(
                            code,
                            interval=interval,
                            bars=max(int(lookback), int(min_cache_bars)),
                            use_cache=True,
                            update_db=bool(update_db),
                        )
                    if df is not None and not df.empty and len(df) >= min_bars:
                        return code, True
                    # FIX: Log at WARNING level for systematic failure detection
                    if df is None:
                        log.warning(f"Fetch returned None for {code} (interval={interval})")
                    elif df.empty:
                        log.warning(f"Fetch returned empty DataFrame for {code} (interval={interval}, lookback={lookback}, min_bars={min_bars})")
                    else:
                        log.warning(f"Fetch returned insufficient bars for {code}: got {len(df)}, need {min_bars} (interval={interval})")
                    return code, False
                except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
                    log.warning(f"Fetch failed for {code} (interval={interval}): {type(e).__name__}: {e}")
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
                except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
                    code = futures[future]
                    success = False
                    log.debug("Concurrent fetch future failed for %s: %s", code, e)

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
                    except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
                        log.debug("Future cancellation failed: %s", e)
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=False)
            else:
                executor.shutdown(wait=True)

        return ok_codes, failed_codes

# CONTINUOUS LEARNER (Main Class)
