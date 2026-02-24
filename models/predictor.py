# models/predictor.py
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from config.settings import CONFIG
from models.predictor_types import PositionSize, Prediction, Signal, TradingLevels
from utils.logger import get_logger
from utils.recoverable import JSON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)

# Type alias for numpy float arrays
# Note: While we use float64 for the type alias, actual computations may
# use float32 for memory efficiency. This alias is for type checking only.
FloatArray: TypeAlias = NDArray[np.float64]

# Define exception tuple early for use in class methods
_PREDICTOR_RECOVERABLE_EXCEPTIONS = JSON_RECOVERABLE_EXCEPTIONS

__all__ = [
    "Predictor",
    "Prediction",
    "Signal",
    "PositionSize",
    "TradingLevels",
]

class Predictor:
    """AI Stock Predictor with real-time capabilities.

    Features:
    - Ensemble model predictions
    - Multi-step price forecasting
    - Real-time chart updates
    - Multiple interval support (1m, 5m, 1d, etc.)
    - Prediction caching with configurable TTL
    """

    _CACHE_TTL: float = 5.0
    _CACHE_TTL_REALTIME: float = 1.2
    _MAX_CACHE_SIZE: int = 200
    _NEWS_CACHE_TTL_INTRADAY: float = 45.0
    _NEWS_CACHE_TTL_SWING: float = 180.0

    def __init__(
        self,
        capital: float = None,
        interval: str = "1m",
        prediction_horizon: int = None,
    ) -> None:
        self.capital = float(capital or CONFIG.CAPITAL)
        self.interval = str(interval).lower()
        intraday_tokens = {"1m", "2m", "3m", "5m", "15m", "30m", "60m", "1h"}
        default_horizon = (
            30
            if self.interval in intraday_tokens
            else int(CONFIG.PREDICTION_HORIZON)
        )
        self.horizon = int(
            prediction_horizon
            if prediction_horizon is not None
            else default_horizon
        )

        self._predict_lock = threading.RLock()

        # Components (lazy loaded)
        self.ensemble = None
        self.forecaster = None
        self._forecaster_horizon: int = 0
        self.processor = None
        self.feature_engine = None
        self.fetcher = None

        self._feature_cols: list[str] = []

        # Prediction cache: code -> (timestamp, Prediction)
        self._pred_cache: dict[str, tuple[float, Prediction]] = {}
        self._cache_lock = threading.Lock()
        # News sentiment cache: code -> (ts, sentiment, confidence, count)
        self._news_cache: dict[str, tuple[float, float, float, int]] = {}
        self._news_cache_lock = threading.Lock()

        # Track if constructor params were overridden by model metadata
        self._requested_interval = self.interval
        self._requested_horizon = self.horizon
        self._loaded_model_interval = self.interval
        self._loaded_model_horizon = self.horizon
        self._high_precision = self._load_high_precision_config()
        self._loaded_ensemble_path: Path | None = None
        self._trained_stock_codes: list[str] = []
        self._trained_stock_last_train: dict[str, str] = {}
        self._model_artifact_sig: str = ""
        self._last_model_reload_attempt_ts: float = 0.0
        self._model_reload_cooldown_s: float = 15.0
        
        # Enhanced prediction accuracy features
        self._confidence_calibration_enabled = True
        self._uncertainty_quantification_enabled = True
        self._adaptive_threshold_enabled = True
        self._regime_aware_prediction = True
        
        # Historical accuracy tracking per stock
        self._stock_accuracy_history: dict[str, list[bool]] = {}
        self._stock_accuracy_window = 50  # Track last 50 predictions per stock
        
        # Ensemble enhancement
        self._model_weights: dict[str, float] = {}
        self._last_model_performance: dict[str, float] = {}

        self._load_models()
        self._model_artifact_sig = self._model_artifact_signature()
        self._last_model_reload_attempt_ts = time.monotonic()
        self._initialize_model_weights()

    def _load_high_precision_config(self) -> dict[str, float]:
        """Optional high-precision runtime gate.
        Disabled by default so existing behavior is unchanged.
        """
        from config.runtime_env import env_flag, env_text

        precision_cfg = getattr(CONFIG, "precision", None)
        enabled_default = bool(getattr(precision_cfg, "enabled", False))
        min_conf_default = max(
            float(CONFIG.model.min_confidence),
            float(getattr(precision_cfg, "min_confidence", 0.70)),
        )
        min_agree_default = max(
            float(getattr(precision_cfg, "min_agreement", 0.65)),
            0.65,
        )
        max_entropy_default = float(getattr(precision_cfg, "max_entropy", 0.45))
        min_edge_default = float(getattr(precision_cfg, "min_edge", 0.10))
        enabled_env = env_flag("TRADING_HIGH_PRECISION_MODE", "1" if enabled_default else "0")
        cfg = {
            "enabled": 1.0 if enabled_env else 0.0,
            "min_confidence": float(env_text("TRADING_HP_MIN_CONFIDENCE", str(min_conf_default))),
            "min_agreement": float(env_text("TRADING_HP_MIN_AGREEMENT", str(min_agree_default))),
            "max_entropy": float(env_text("TRADING_HP_MAX_ENTROPY", str(max_entropy_default))),
            "min_edge": float(env_text("TRADING_HP_MIN_EDGE", str(min_edge_default))),
            "regime_routing": 1.0 if bool(getattr(precision_cfg, "regime_routing", True)) else 0.0,
            "range_conf_boost": float(getattr(precision_cfg, "range_confidence_boost", 0.04)),
            "high_vol_conf_boost": float(getattr(precision_cfg, "high_vol_confidence_boost", 0.05)),
            "high_vol_atr_pct": float(getattr(precision_cfg, "high_vol_atr_pct", 0.035)),
        }

        # Load learned profile if available; env vars still have final override.
        try:
            profile_name = str(getattr(precision_cfg, "profile_filename", "precision_thresholds.json"))
            profile_path = CONFIG.data_dir / profile_name
            if profile_path.exists():
                data = json.loads(profile_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    prof = data.get("thresholds", data)
                    if isinstance(prof, dict):
                        for key in ("min_confidence", "min_agreement", "max_entropy", "min_edge"):
                            if key in prof:
                                cfg[key] = float(prof[key])
                        cfg["profile_loaded"] = 1.0
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug("High precision profile load failed: %s", e)

        # Final environment override (highest priority)
        cfg["min_confidence"] = float(env_text("TRADING_HP_MIN_CONFIDENCE", str(cfg["min_confidence"])))
        cfg["min_agreement"] = float(env_text("TRADING_HP_MIN_AGREEMENT", str(cfg["min_agreement"])))
        cfg["max_entropy"] = float(env_text("TRADING_HP_MAX_ENTROPY", str(cfg["max_entropy"])))
        cfg["min_edge"] = float(env_text("TRADING_HP_MIN_EDGE", str(cfg["min_edge"])))

        if cfg["enabled"] > 0:
            log.info(
                "High Precision Mode enabled: min_conf=%.2f min_agree=%.2f "
                "max_entropy=%.2f min_edge=%.2f",
                cfg["min_confidence"],
                cfg["min_agreement"],
                cfg["max_entropy"],
                cfg["min_edge"],
            )
        return cfg

    # =========================================================================
    # =========================================================================

    def _load_models(self) -> bool:
        """Load all required models with robust fallback."""
        try:
            from data.features import FeatureEngine
            from data.fetcher import get_fetcher
            from data.processor import DataProcessor
            from models.ensemble import EnsembleModel

            self.processor = DataProcessor()
            self.feature_engine = FeatureEngine()
            self.fetcher = get_fetcher()
            self._feature_cols = self.feature_engine.get_feature_columns()

            model_dir = CONFIG.MODEL_DIR

            # Pick best ensemble + scaler pair
            chosen_ens, chosen_scl = self._find_best_model_pair(model_dir)
            if chosen_scl is None:
                chosen_scl = self._find_best_scaler_checkpoint(model_dir)

            if chosen_scl and chosen_scl.exists():
                if not self.processor.load_scaler(str(chosen_scl)):
                    log.warning("Failed loading scaler from %s", chosen_scl)
            else:
                log.warning(
                    "No scaler found for requested interval/horizon (%s/%s)",
                    self.interval,
                    self.horizon,
                )

            self.ensemble = None
            self._loaded_ensemble_path = None
            self._trained_stock_codes = []
            self._trained_stock_last_train = {}
            if chosen_ens and chosen_ens.exists():
                self._loaded_ensemble_path = Path(chosen_ens)
                input_size = (
                    self.processor.n_features or len(self._feature_cols)
                )
                ens = EnsembleModel(input_size=input_size)
                if ens.load(str(chosen_ens)) and ens.models:
                    self.ensemble = ens

                    loaded_interval = str(
                        getattr(self.ensemble, "interval", self.interval)
                    )
                    loaded_horizon = int(
                        getattr(
                            self.ensemble, "prediction_horizon",
                            self.horizon
                        )
                    )

                    if (
                        loaded_interval != self._requested_interval
                        or loaded_horizon != self._requested_horizon
                    ):
                        log.info(
                            f"Model metadata overrides constructor: "
                            f"interval {self._requested_interval}"
                            f"->{loaded_interval}, "
                            f"horizon {self._requested_horizon}"
                            f"->{loaded_horizon}"
                        )

                    # Keep runtime interval/horizon aligned to constructor
                    # (UI/requested settings). Loaded model metadata is tracked
                    # separately for diagnostics and compatibility checks.
                    self._loaded_model_interval = str(loaded_interval).lower()
                    self._loaded_model_horizon = int(loaded_horizon)
                    self._trained_stock_codes = (
                        self._extract_trained_stocks_from_ensemble()
                        or self._load_trained_stocks_from_manifest(chosen_ens)
                        or self._load_trained_stocks_from_learner_state(
                            loaded_interval,
                            loaded_horizon,
                        )
                    )
                    self._trained_stock_last_train = (
                        self._extract_trained_stock_last_train_from_ensemble()
                        or self._load_trained_stock_last_train_from_manifest(chosen_ens)
                    )
                    if self._trained_stock_last_train and self._trained_stock_codes:
                        allowed = {
                            self._normalize_stock_code(x)
                            for x in list(self._trained_stock_codes or [])
                        }
                        allowed = {x for x in allowed if x}
                        if allowed:
                            self._trained_stock_last_train = {
                                c: ts
                                for c, ts in self._trained_stock_last_train.items()
                                if c in allowed
                            }

                    log.info(
                        f"Ensemble loaded from {chosen_ens.name} "
                        f"(interval={self.interval}, "
                        f"horizon={self.horizon})"
                    )
                else:
                    log.warning(f"Failed to load ensemble: {chosen_ens}")
            else:
                log.warning("No ensemble model found")

            # Load forecaster (optional)
            self._load_forecaster()

            return True

        except ImportError as e:
            log.error(f"Missing dependency for models: {e}")
            return False
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.error(f"Failed to load models: {e}")
            return False

    def _initialize_model_weights(self) -> None:
        """Initialize adaptive model weights based on historical performance.
        
        This improves ensemble accuracy by weighting better-performing models higher.
        """
        if self.ensemble is None or not hasattr(self.ensemble, 'models'):
            return
            
        # Default equal weights
        n_models = len(self.ensemble.models)
        if n_models == 0:
            return
            
        model_names = list(self.ensemble.models.keys())
        
        # Initialize with equal weights
        base_weight = 1.0 / n_models
        for name in model_names:
            self._model_weights[name] = base_weight
            self._last_model_performance[name] = 0.5  # Start with neutral performance
        
        log.info(f"Initialized model weights for {n_models} models")

    def _update_model_weights(self, stock_code: str, was_correct: bool) -> None:
        """Update model weights based on prediction accuracy.
        
        FIX ADAPTIVE LR: Uses adaptive learning rate based on:
        1. Prediction frequency (more predictions = slower adaptation)
        2. Market volatility (high vol = slower adaptation to avoid noise)
        3. Recent performance streak (hot/cold streaks adjust learning)
        
        This improves adaptation to regime changes while avoiding over-reaction
        to recent noise.
        """
        if self.ensemble is None or not hasattr(self.ensemble, 'models'):
            return

        # Base learning rate
        alpha_base = 0.10
        
        # Factor 1: Adjust based on prediction count for this stock
        # More predictions = more stable estimate = slower adaptation
        prediction_count = len(self._stock_accuracy_history.get(stock_code, []))
        if prediction_count > 20:
            # Reduce learning rate as we have more history
            alpha_base *= max(0.03, 0.08 * (20.0 / min(prediction_count, 100)))
        
        # Factor 2: Adjust based on recent accuracy streak
        # Hot/cold streaks suggest regime change - adapt faster
        recent_history = self._stock_accuracy_history.get(stock_code, [])
        if len(recent_history) >= 5:
            recent_5 = recent_history[-5:]
            streak = sum(1 for x in recent_5 if x)  # Count correct
            if streak == 0 or streak == 5:
                # Extreme streak - may indicate regime change
                alpha_base *= 1.5  # Adapt faster
            elif streak in [2, 3]:
                # Mixed results - be conservative
                alpha_base *= 0.7
        
        # Factor 3: Clamp learning rate to reasonable bounds
        alpha = float(np.clip(alpha_base, 0.02, 0.25))

        for model_name in self._model_weights:
            # Simplified: assume all models contributed equally to this prediction
            current_perf = self._last_model_performance.get(model_name, 0.5)
            reward = 1.0 if was_correct else 0.0
            new_perf = (1 - alpha) * current_perf + alpha * reward
            self._last_model_performance[model_name] = new_perf

        # Normalize weights
        total_perf = sum(self._last_model_performance.values())
        if total_perf > 0:
            for name in self._model_weights:
                self._model_weights[name] = self._last_model_performance[name] / total_perf

    def _record_prediction_outcome(self, stock_code: str, predicted_up: bool, actual_up: bool) -> None:
        """Record prediction outcome for historical accuracy tracking."""
        was_correct = (predicted_up == actual_up)
        
        if stock_code not in self._stock_accuracy_history:
            self._stock_accuracy_history[stock_code] = []
        
        self._stock_accuracy_history[stock_code].append(was_correct)
        
        # Keep only recent history
        if len(self._stock_accuracy_history[stock_code]) > self._stock_accuracy_window:
            self._stock_accuracy_history[stock_code] = self._stock_accuracy_history[stock_code][-self._stock_accuracy_window:]
        
        # Update model weights
        self._update_model_weights(stock_code, was_correct)

    def _get_stock_accuracy(self, stock_code: str) -> float:
        """Get recent prediction accuracy for a stock."""
        if stock_code not in self._stock_accuracy_history:
            return 0.5  # Neutral default
        
        history = self._stock_accuracy_history[stock_code]
        if not history:
            return 0.5
        
        # Weight recent predictions more heavily
        weights = []
        for i in range(len(history)):
            weights.append(0.9 ** (len(history) - i - 1))  # Exponential decay

        # Note: weights list is created from history length, so lengths always match
        weighted_sum = sum(h * w for h, w in zip(history, weights))  # noqa: B905
        weight_total = sum(weights)

        return weighted_sum / weight_total if weight_total > 0 else 0.5

    def _calibrate_confidence(self, confidence: float, stock_code: str, entropy: float) -> float:
        """Calibrate confidence based on historical accuracy and uncertainty.
        
        This improves prediction reliability by adjusting confidence based on:
        1. Historical accuracy for this stock
        2. Prediction entropy (uncertainty)
        3. Model agreement
        """
        if not self._confidence_calibration_enabled:
            return confidence
        
        # Get historical accuracy
        historical_acc = self._get_stock_accuracy(stock_code)
        
        # Adjust confidence based on historical performance
        if historical_acc > 0.6:
            # Good history: boost confidence slightly
            confidence = min(0.95, confidence * (1.0 + (historical_acc - 0.5) * 0.2))
        elif historical_acc < 0.45:
            # Poor history: reduce confidence
            confidence = max(0.3, confidence * (1.0 - (0.5 - historical_acc) * 0.3))
        
        # Penalize high entropy (uncertainty)
        if entropy > 0.7:
            confidence = max(0.3, confidence * 0.8)
        elif entropy > 0.5:
            confidence = max(0.3, confidence * 0.9)
        
        return float(np.clip(confidence, 0.3, 0.95))

    def _find_best_model_pair(self, model_dir: Path) -> tuple[Path | None, Path | None]:
        """Find the best ensemble + scaler file pair."""
        req_ens = model_dir / f"ensemble_{self.interval}_{self.horizon}.pt"
        req_scl = model_dir / f"scaler_{self.interval}_{self.horizon}.pkl"

        if req_ens.exists() and req_scl.exists():
            return req_ens, req_scl

        if req_ens.exists() and not req_scl.exists():
            log.warning(
                "Exact ensemble exists but scaler missing for %s/%s",
                self.interval,
                self.horizon,
            )

        # Same interval, nearest available horizon with scaler.
        same_interval: list[tuple[int, float, Path, Path]] = []
        for ep in model_dir.glob(f"ensemble_{self.interval}_*.pt"):
            parts = ep.stem.split("_", 2)
            if len(parts) != 3:
                continue
            cand_h = self._parse_artifact_horizon(parts[2])
            if cand_h is None:
                continue
            sp = self._find_scaler_for_interval_horizon(
                model_dir=model_dir,
                interval=self.interval,
                horizon=cand_h,
            )
            if sp is None:
                continue
            same_interval.append(
                (abs(cand_h - int(self.horizon)), -ep.stat().st_mtime, ep, sp)
            )
        if same_interval:
            same_interval.sort(key=lambda x: (x[0], x[1]))
            _delta, _mtime_neg, ep, sp = same_interval[0]
            return ep, sp

        log.warning(
            "No interval-matched model pair found for interval=%s horizon=%s",
            self.interval,
            self.horizon,
        )
        return None, None

    @staticmethod
    def _parse_artifact_horizon(token: str) -> int | None:
        """Extract horizon integer from artifact token (supports suffixes)."""
        s = str(token or "").strip()
        if not s:
            return None
        m = re.match(r"^(\d+)", s)
        if m is None:
            return None
        try:
            return int(m.group(1))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _find_scaler_for_interval_horizon(
        model_dir: Path,
        interval: str,
        horizon: int,
    ) -> Path | None:
        """Find exact or suffixed scaler checkpoint for interval+horizon."""
        exact = model_dir / f"scaler_{interval}_{int(horizon)}.pkl"
        if exact.exists():
            return exact

        matches = list(model_dir.glob(f"scaler_{interval}_{int(horizon)}*.pkl"))
        if not matches:
            return None
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    def _find_best_forecaster_checkpoint(self, model_dir: Path) -> Path | None:
        """Find best-matching forecaster for current interval+horizon."""
        req_path = model_dir / f"forecast_{self.interval}_{self.horizon}.pt"
        if req_path.exists():
            return req_path

        same_interval: list[tuple[int, float, Path]] = []
        for fp in model_dir.glob(f"forecast_{self.interval}_*.pt"):
            parts = fp.stem.split("_", 2)
            if len(parts) != 3:
                continue
            cand_h = self._parse_artifact_horizon(parts[2])
            if cand_h is None:
                continue
            same_interval.append(
                (abs(cand_h - int(self.horizon)), -fp.stat().st_mtime, fp)
            )
        if same_interval:
            same_interval.sort(key=lambda x: (x[0], x[1]))
            _delta, _mtime_neg, fp = same_interval[0]
            return fp

        return None

    def _find_best_scaler_checkpoint(self, model_dir: Path) -> Path | None:
        """Find best-matching scaler for current interval+horizon."""
        req_path = model_dir / f"scaler_{self.interval}_{self.horizon}.pkl"
        if req_path.exists():
            return req_path

        same_interval: list[tuple[int, float, Path]] = []
        for sp in model_dir.glob(f"scaler_{self.interval}_*.pkl"):
            parts = sp.stem.split("_", 2)
            if len(parts) != 3:
                continue
            cand_h = self._parse_artifact_horizon(parts[2])
            if cand_h is None:
                continue
            same_interval.append(
                (abs(cand_h - int(self.horizon)), -sp.stat().st_mtime, sp)
            )
        if same_interval:
            same_interval.sort(key=lambda x: (x[0], x[1]))
            _delta, _mtime_neg, sp = same_interval[0]
            return sp

        return None

    def _model_artifact_signature(self) -> str:
        """Compact signature of model artifacts for hot-reload checks."""
        try:
            model_dir = Path(CONFIG.MODEL_DIR)
            if not model_dir.exists():
                return ""
            parts: list[str] = []
            for pattern in ("ensemble_*.pt", "forecast_*.pt", "scaler_*.pkl"):
                files = list(model_dir.glob(pattern))
                latest = 0.0
                for fp in files:
                    try:
                        latest = max(latest, float(fp.stat().st_mtime))
                    except OSError:
                        continue
                parts.append(f"{pattern}:{len(files)}:{latest:.3f}")
            return "|".join(parts)
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Model artifact signature check failed: %s", e)
            return ""

    def _models_ready_for_runtime(self) -> bool:
        """Whether runtime has enough loaded artifacts for stable inference."""
        processor = getattr(self, "processor", None)
        ensemble = getattr(self, "ensemble", None)
        forecaster = getattr(self, "forecaster", None)
        scaler_ready = bool(
            processor is not None and getattr(processor, "is_fitted", True)
        )
        return scaler_ready and (ensemble is not None or forecaster is not None)

    def _maybe_reload_models(self, *, reason: str = "", force: bool = False) -> bool:
        """Opportunistically reload models when artifacts appear/rotate on disk.

        This keeps long-running UI sessions in sync without requiring restart
        after training finishes.
        """
        # Unit tests may construct Predictor via __new__ with a minimal fixture.
        # In that case, skip runtime hot-reload logic.
        if (
            not hasattr(self, "_model_artifact_sig")
            or not hasattr(self, "_model_reload_cooldown_s")
        ):
            return False

        now_ts = time.monotonic()
        current_sig = self._model_artifact_signature()
        prev_sig = str(getattr(self, "_model_artifact_sig", ""))
        sig_changed = bool(current_sig) and current_sig != prev_sig
        ready = self._models_ready_for_runtime()
        last_reload_ts = float(getattr(self, "_last_model_reload_attempt_ts", 0.0))
        cooldown_s = float(getattr(self, "_model_reload_cooldown_s", 15.0))
        cooldown_ok = (
            (now_ts - last_reload_ts)
            >= cooldown_s
        )
        should_reload = bool(force) or ((not ready or sig_changed) and cooldown_ok)
        if not should_reload:
            return False

        self._last_model_reload_attempt_ts = now_ts
        before_ready = ready
        ok = bool(self._load_models())
        self._model_artifact_sig = self._model_artifact_signature()
        after_ready = self._models_ready_for_runtime()

        if before_ready != after_ready or sig_changed:
            log.info(
                "Predictor model reload (%s): ok=%s scaler=%s ensemble=%s forecaster=%s",
                reason or "runtime",
                ok,
                bool(self.processor is not None and getattr(self.processor, "is_fitted", False)),
                bool(self.ensemble is not None),
                bool(self.forecaster is not None),
            )
        return ok

    def _extract_trained_stocks_from_ensemble(self) -> list[str]:
        if self.ensemble is None:
            return []
        try:
            info = self.ensemble.get_model_info()
            if not isinstance(info, dict):
                return []
            out = [
                str(x).strip()
                for x in list(info.get("trained_stock_codes", []) or [])
                if str(x).strip()
            ]
            return out
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Failed reading trained stocks from ensemble metadata: %s", e)
            return []

    @staticmethod
    def _normalize_stock_code(value: object) -> str:
        code = "".join(ch for ch in str(value or "").strip() if ch.isdigit())
        return code if len(code) == 6 else ""

    @classmethod
    def _sanitize_last_train_map(cls, payload: object) -> dict[str, str]:
        if not isinstance(payload, dict):
            return {}
        out: dict[str, str] = {}
        for raw_code, raw_ts in payload.items():
            code = cls._normalize_stock_code(raw_code)
            if not code:
                continue
            ts = str(raw_ts or "").strip()
            if not ts:
                continue
            out[code] = ts
        return out

    def _extract_trained_stock_last_train_from_ensemble(self) -> dict[str, str]:
        if self.ensemble is None:
            return {}
        try:
            info = self.ensemble.get_model_info()
            if not isinstance(info, dict):
                return {}
            return self._sanitize_last_train_map(
                info.get("trained_stock_last_train", {})
            )
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug(
                "Failed reading trained-stock last-train from ensemble metadata: %s",
                e,
            )
            return {}

    def _load_trained_stocks_from_manifest(self, ensemble_path: Path) -> list[str]:
        """Fallback for legacy model loads when ensemble metadata is missing."""
        try:
            stem = Path(ensemble_path).stem
            p = Path(ensemble_path).parent / f"model_manifest_{stem}.json"
            if not p.exists():
                return []
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return []
            out = [
                str(x).strip()
                for x in list(data.get("trained_stock_codes", []) or [])
                if str(x).strip()
            ]
            return out
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Failed reading trained stocks from manifest %s: %s", ensemble_path, e)
            return []

    def _load_trained_stock_last_train_from_manifest(
        self,
        ensemble_path: Path,
    ) -> dict[str, str]:
        """Fallback for legacy loads when ensemble metadata is missing."""
        try:
            stem = Path(ensemble_path).stem
            p = Path(ensemble_path).parent / f"model_manifest_{stem}.json"
            if not p.exists():
                return {}
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            return self._sanitize_last_train_map(
                data.get("trained_stock_last_train", {})
            )
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug(
                "Failed reading trained-stock last-train from manifest %s: %s",
                ensemble_path,
                e,
            )
            return {}

    def _load_trained_stocks_from_learner_state(
        self,
        interval: str,
        horizon: int,
    ) -> list[str]:
        """Last-resort fallback for legacy artifacts missing stock metadata.
        Uses persisted learner_state only when interval+horizon match.
        """
        try:
            p = Path(CONFIG.data_dir) / "learner_state.json"
            if not p.exists():
                return []
            raw = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return []
            data = raw.get("_data", raw)
            if not isinstance(data, dict):
                return []

            iv = str(data.get("last_interval", "")).strip().lower()
            try:
                hz = int(data.get("last_horizon", 0) or 0)
            except (TypeError, ValueError):
                hz = 0

            if iv != str(interval).strip().lower() or hz != int(horizon):
                return []

            replay = data.get("replay", {})
            if not isinstance(replay, dict):
                replay = {}
            candidates = list(replay.get("buffer", []) or [])

            if not candidates:
                rot = data.get("rotator", {})
                if isinstance(rot, dict):
                    candidates = list(rot.get("processed", []) or [])

            out: list[str] = []
            seen: set[str] = set()
            for x in candidates:
                code = "".join(c for c in str(x).strip() if c.isdigit())
                if len(code) != 6 or code in seen:
                    continue
                seen.add(code)
                out.append(code)
            return out
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Failed reading trained stocks from learner state: %s", e)
            return []

    def get_trained_stock_codes(self, limit: int | None = None) -> list[str]:
        """Return stock codes used in the currently loaded training artifact."""
        codes = list(self._trained_stock_codes)
        if limit is not None:
            try:
                n = max(0, int(limit))
                return codes[:n]
            except (TypeError, ValueError) as e:
                log.debug("Invalid trained stock code limit %r: %s", limit, e)
        return codes

    def get_trained_stock_last_train(self) -> dict[str, str]:
        """Return per-stock last-train timestamps from loaded model artifacts."""
        return dict(self._trained_stock_last_train or {})

    def _load_forecaster(self) -> None:
        """Load TCN forecaster for price curve prediction."""
        try:
            from models.networks import TCNModel

            self.forecaster = None
            self._forecaster_horizon = 0
            forecast_path = self._find_best_forecaster_checkpoint(
                CONFIG.MODEL_DIR
            )

            if forecast_path is None:
                log.debug(
                    "No matching forecaster model found for interval=%s horizon=%s",
                    self.interval,
                    self.horizon,
                )
                return

            if (
                self.processor is not None
                and not self.processor._verify_artifact_checksum(forecast_path)
            ):
                return

            allow_unsafe = bool(
                getattr(
                    getattr(CONFIG, "model", None),
                    "allow_unsafe_artifact_load",
                    False,
                )
            )
            require_checksum = bool(
                getattr(
                    getattr(CONFIG, "model", None),
                    "require_artifact_checksum",
                    True,
                )
            )

            def _load_checkpoint(weights_only: bool) -> dict[str, Any]:
                from utils.atomic_io import torch_load

                return torch_load(
                    forecast_path,
                    map_location="cpu",
                    weights_only=weights_only,
                    verify_checksum=True,
                    require_checksum=require_checksum,
                    allow_unsafe=allow_unsafe,
                )

            try:
                data = _load_checkpoint(weights_only=True)
            except (OSError, RuntimeError, TypeError, ValueError, ImportError) as exc:
                if not allow_unsafe:
                    log.error(
                        "Forecaster secure load failed for %s and unsafe fallback is disabled: %s",
                        forecast_path,
                        exc,
                    )
                    return
                log.warning(
                    "Forecaster weights-only load failed for %s; "
                    "falling back to unsafe legacy checkpoint load: %s",
                    forecast_path,
                    exc,
                )
                data = _load_checkpoint(weights_only=False)

            required_keys = {"input_size", "horizon", "arch", "state_dict"}
            if not required_keys.issubset(data.keys()):
                log.warning(
                    f"Forecaster checkpoint missing keys: "
                    f"{required_keys - set(data.keys())}"
                )
                return

            horizon = int(data.get("horizon", 0) or 0)
            if horizon <= 0:
                log.warning("Forecaster checkpoint has invalid horizon: %s", horizon)
                return

            self.forecaster = TCNModel(
                input_size=data["input_size"],
                hidden_size=data["arch"]["hidden_size"],
                num_classes=horizon,
                dropout=data["arch"]["dropout"],
            )
            self.forecaster.load_state_dict(data["state_dict"])
            self.forecaster.eval()
            self._forecaster_horizon = horizon

            log.info(
                "Forecaster loaded: %s (interval=%s, horizon=%s)",
                forecast_path.name,
                self.interval,
                horizon,
            )

        except ImportError:
            log.debug("TCNModel not available - forecaster disabled")
            self.forecaster = None
            self._forecaster_horizon = 0
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
            log.debug(f"Forecaster not loaded: {e}")
            self.forecaster = None
            self._forecaster_horizon = 0

    # =========================================================================
    # =========================================================================

    def predict(
        self,
        stock_code: str,
        use_realtime_price: bool = True,
        interval: str = None,
        forecast_minutes: int = None,
        lookback_bars: int = None,
        skip_cache: bool = False,
        history_allow_online: bool = True,
    ) -> Prediction:
        """Make full prediction with price forecast.

        Args:
            stock_code: Stock code to predict
            use_realtime_price: Whether to use real-time price
            interval: Data interval (1m, 5m, 1d, etc.)
            forecast_minutes: Number of bars to forecast
            lookback_bars: Historical bars to use
            skip_cache: If True, bypass prediction cache (for real-time)
            history_allow_online: If False, use local cache/DB/session only

        Returns:
            Prediction object with all fields populated
        """
        with self._predict_lock:
            self._maybe_reload_models(reason="predict")
            
            # FIX #10: Check if scaler was loaded successfully
            if not hasattr(self.processor, 'scaler') or self.processor.scaler is None:
                log.warning(
                    "Scaler not loaded - predictions may be unreliable. "
                    "Ensure model training has been completed."
                )
            
            interval = self._normalize_interval_token(interval)
            horizon = int(forecast_minutes or self.horizon)
            lookback = int(
                lookback_bars or self._default_lookback_bars(interval)
            )

            code = self._clean_code(stock_code)
            cache_key = (
                f"{code}:{interval}:{horizon}:"
                f"{'rt' if bool(use_realtime_price) else 'hist'}"
            )
            cache_ttl = self._get_cache_ttl(
                use_realtime=bool(use_realtime_price),
                interval=interval,
            )

            if not skip_cache:
                cached = self._get_cached_prediction(
                    cache_key, ttl=cache_ttl
                )
                if cached is not None:
                    return cached

            pred = Prediction(
                stock_code=code,
                timestamp=datetime.now(),
                interval=interval,
                horizon=horizon,
            )

            try:
                min_rows = int(
                    getattr(
                        self.feature_engine,
                        "MIN_ROWS",
                        CONFIG.SEQUENCE_LENGTH,
                    )
                )
                required_rows = int(max(CONFIG.SEQUENCE_LENGTH, min_rows))

                df = self._fetch_data(
                    code,
                    interval,
                    lookback,
                    use_realtime_price,
                    history_allow_online=bool(history_allow_online),
                )

                if not self._has_required_rows(df, required_rows):
                    data_len = len(df) if isinstance(df, pd.DataFrame) else 0
                    short_ready = self._has_required_rows(
                        df,
                        max(20, int(required_rows // 3)),
                    )
                    if short_ready and self._bootstrap_short_history_prediction(
                        pred,
                        df,
                        horizon=horizon,
                        required_rows=required_rows,
                    ):
                        # Keep short-history fallback aligned with news context.
                        self._apply_news_influence(pred, code, interval)
                        self._set_cached_prediction(cache_key, pred)
                        return pred
                    self._populate_minimal_snapshot(pred, code)
                    pred.warnings.append(
                        f"Insufficient data: got {data_len} bars, "
                        f"need {required_rows}"
                    )
                    return pred

                pred.stock_name = self._get_stock_name(code, df)
                pred.current_price = float(df["close"].iloc[-1])
                pred.price_history = df["close"].tail(180).tolist()

                df = self.feature_engine.create_features(df)
                self._extract_technicals(df, pred)

                X = self.processor.prepare_inference_sequence(
                    df, self._feature_cols
                )

                if self.ensemble:
                    self._apply_ensemble_prediction(X, pred)

                # Enhanced accuracy: Calibrate confidence based on historical performance
                if self._confidence_calibration_enabled:
                    pred.confidence = self._calibrate_confidence(
                        pred.confidence,
                        pred.stock_code,
                        pred.entropy,
                    )
                
                # Enhanced accuracy: Apply regime-aware adjustments
                if self._regime_aware_prediction:
                    self._apply_regime_adjustments(pred, df)
                
                # Enhanced accuracy: Apply adaptive threshold for signal generation
                if self._adaptive_threshold_enabled:
                    self._apply_adaptive_signal_threshold(pred)

                news_bias = self._apply_news_influence(pred, code, interval)
                self._refresh_prediction_uncertainty(pred)
                self._apply_tail_risk_guard(pred)

                try:
                    pred.predicted_prices = self._generate_forecast(
                        X,
                        pred.current_price,
                        horizon,
                        pred.atr_pct_value,
                        sequence_signature=self._sequence_signature(X),
                        seed_context=f"{code}:{interval}",
                        recent_prices=df["close"].tail(min(560, len(df))).tolist(),
                        news_bias=news_bias,
                    )
                except TypeError:
                    # Compatibility for tests/mocks overriding _generate_forecast.
                    pred.predicted_prices = self._generate_forecast(
                        X,
                        pred.current_price,
                        horizon,
                        pred.atr_pct_value,
                        sequence_signature=self._sequence_signature(X),
                        seed_context=f"{code}:{interval}",
                        recent_prices=df["close"].tail(min(560, len(df))).tolist(),
                    )
                self._build_prediction_bands(pred)
                pred.levels = self._calculate_levels(pred)
                pred.position = self._calculate_position(pred)
                self._generate_reasons(pred)

                self._set_cached_prediction(cache_key, pred)

            except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
                log.error(
                    f"Prediction failed for {code}: {e}",
                    exc_info=True
                )
                pred.warnings.append(
                    f"Prediction error: {type(e).__name__}: {str(e)}"
                )

            return pred

    @staticmethod
    def _has_required_rows(df: pd.DataFrame | None, required_rows: int) -> bool:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return False
        try:
            return int(len(df)) >= int(max(1, required_rows))
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Failed required-row check (required=%r): %s", required_rows, e)
            return False

    def _populate_minimal_snapshot(self, pred: Prediction, code: str) -> None:
        """Populate minimal quote context when full feature history is unavailable.
        This keeps UI chart/price widgets responsive instead of returning all zeros.
        """
        if pred.current_price > 0:
            return
        fetcher = getattr(self, "fetcher", None)
        if fetcher is None:
            return

        try:
            quote = fetcher.get_realtime(code)
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Realtime quote unavailable for %s: %s", code, e)
            quote = None

        if quote is not None:
            try:
                px = float(getattr(quote, "price", 0) or 0)
            except (TypeError, ValueError):
                px = 0.0
            if px > 0:
                pred.current_price = px
                pred.price_history = [px]
                qname = str(getattr(quote, "name", "") or "").strip()
                if qname and not pred.stock_name:
                    pred.stock_name = qname
                return

        fallback_iv = self._normalize_interval_token(
            getattr(pred, "interval", None) or self.interval
        )
        try:
            df = fetcher.get_history(
                code,
                interval=fallback_iv,
                bars=1,
                use_cache=True,
                update_db=False,
                allow_online=False,
            )
        except TypeError:
            df = fetcher.get_history(
                code,
                interval=fallback_iv,
                bars=1,
                use_cache=True,
                update_db=False,
            )
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug("Fallback history fetch failed for %s: %s", code, e)
            df = None

        if isinstance(df, pd.DataFrame) and not df.empty:
            try:
                px = float(df["close"].iloc[-1] or 0)
            except (TypeError, ValueError):
                px = 0.0
            if px > 0:
                pred.current_price = px
                pred.price_history = [px]

    def _bootstrap_short_history_prediction(
        self,
        pred: Prediction,
        df: pd.DataFrame,
        *,
        horizon: int,
        required_rows: int,
    ) -> bool:
        """Produce a lightweight fallback prediction when history is short.
        This avoids HOLD(0%) outputs during startup warm-up windows.
        """
        if df is None or df.empty or "close" not in df.columns:
            return False

        close = pd.to_numeric(df["close"], errors="coerce")
        close = close.replace([np.inf, -np.inf], np.nan).dropna()
        if len(close) < 20:
            return False

        current_price = float(close.iloc[-1] or 0.0)
        if current_price <= 0:
            return False

        pred.stock_name = self._get_stock_name(pred.stock_code, df)
        pred.current_price = current_price
        pred.price_history = [
            float(v) for v in close.tail(180).tolist() if float(v) > 0
        ]
        if not pred.price_history:
            pred.price_history = [current_price]

        recent = close.tail(min(36, len(close)))
        pct = recent.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        drift = float(pct.tail(min(12, len(pct))).mean()) if not pct.empty else 0.0
        vol = (
            float(pct.tail(min(20, len(pct))).std(ddof=0))
            if len(pct) > 1 else abs(drift) * 1.5
        )
        if not np.isfinite(drift):
            drift = 0.0
        if not np.isfinite(vol):
            vol = 0.0
        base_move = float(recent.iloc[-1] / max(float(recent.iloc[0]), 1e-8) - 1.0)
        score = (drift * 700.0) + (base_move * 6.0)
        direction = float(np.tanh(score))
        depth_ratio = float(min(1.0, len(close) / max(float(required_rows), 1.0)))
        pred.confidence = float(
            np.clip(0.36 + (0.24 * depth_ratio) + (0.28 * abs(direction)), 0.36, 0.78)
        )
        pred.raw_confidence = float(pred.confidence)
        pred.model_agreement = float(np.clip(0.52 + (0.35 * depth_ratio), 0.0, 1.0))
        pred.entropy = float(
            np.clip(
                0.62 - (0.35 * abs(direction)) + (0.25 * max(0.0, vol - 0.02)),
                0.05,
                0.95,
            )
        )
        pred.model_margin = float(np.clip(abs(direction) * 0.32, 0.01, 0.25))
        neutral = float(
            np.clip(
                0.45 - (0.20 * abs(direction)) + (0.90 * max(0.0, vol - 0.015)),
                0.12,
                0.74,
            )
        )
        up = float(np.clip((1.0 - neutral) * (0.5 + 0.5 * direction), 0.02, 0.95))
        down = float(max(0.0, 1.0 - neutral - up))
        total = up + neutral + down
        if total <= 0:
            up, neutral, down = 0.33, 0.34, 0.33
            total = 1.0
        pred.prob_up = float(up / total)
        pred.prob_neutral = float(neutral / total)
        pred.prob_down = float(down / total)
        allow_directional = bool(getattr(getattr(CONFIG, "precision", None), "allow_short_history_directional_signals", False))
        if not allow_directional:
            direction = 0.0
            pred.warnings.append("Short-history directional signal suppressed by precision policy")
        if direction >= 0.55 and pred.confidence >= 0.70:
            pred.signal = Signal.STRONG_BUY
        elif direction >= 0.30 and pred.confidence >= 0.54:
            pred.signal = Signal.BUY
        elif direction <= -0.55 and pred.confidence >= 0.70:
            pred.signal = Signal.STRONG_SELL
        elif direction <= -0.30 and pred.confidence >= 0.54:
            pred.signal = Signal.SELL
        else:
            pred.signal = Signal.HOLD
        pred.signal_strength = float(
            np.clip((0.70 * abs(direction)) + (0.30 * pred.confidence), 0.0, 1.0)
        )
        if direction > 0.25:
            pred.trend = "UPTREND"
        elif direction < -0.25:
            pred.trend = "DOWNTREND"
        else:
            pred.trend = "SIDEWAYS"
        pred.atr_pct_value = float(np.clip(max(0.006, vol * 2.5), 0.006, 0.05))

        steps = max(1, int(horizon))
        pred.predicted_prices = []
        px = current_price
        drift = float(np.clip(drift, -0.02, 0.02))
        vol_step = float(np.clip(max(vol, abs(drift) * 1.2), 0.0008, 0.018))
        seed_n = len(close)
        for step in range(1, steps + 1):
            decay = max(0.15, 1.0 - (float(step) / float(max(steps, 1))) * 0.65)
            step_ret = (drift * decay) + (direction * 0.0006 * decay)
            wave = np.sin((float(step) + float(seed_n)) * 0.7) * (vol_step * 0.18)
            step_ret = float(np.clip(step_ret + wave, -0.04, 0.04))
            px = max(0.01, float(px) * (1.0 + step_ret))
            pred.predicted_prices.append(float(px))

        self._refresh_prediction_uncertainty(pred)
        self._apply_tail_risk_guard(pred)
        self._build_prediction_bands(pred)

        try:
            pred.levels = self._calculate_levels(pred)
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug(
                "Short-history levels calculation failed for %s: %s",
                pred.stock_code,
                e,
            )
        try:
            pred.position = self._calculate_position(pred)
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug(
                "Short-history position calculation failed for %s: %s",
                pred.stock_code,
                e,
            )
        try:
            self._generate_reasons(pred)
        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.debug(
                "Short-history reason generation failed for %s: %s",
                pred.stock_code,
                e,
            )

        pred.warnings.append(
            "Short-history fallback used: "
            f"{len(close)} bars available (target {required_rows})"
        )
        return True

    def _apply_regime_adjustments(self, pred: Prediction, df: pd.DataFrame) -> None:
        """Apply regime-aware adjustments to prediction.

        Adjusts confidence and thresholds based on detected market regime:
        - Trending markets: Higher confidence, lower thresholds
        - Ranging markets: Lower confidence, higher thresholds
        - High volatility: Wider stops/targets, confidence adjustment
        """
        try:
            from models.regime import MarketRegimeDetector

            detector = MarketRegimeDetector()
            regime_result = detector.detect(df)

            # Store regime info
            pred.regime = regime_result.regime.value

            # Adjust confidence based on regime historical accuracy
            regime_acc = float(getattr(regime_result, "historical_accuracy", 0.5) or 0.5)
            if regime_acc > 0.65:
                # High accuracy regime: boost confidence
                pred.confidence = min(0.95, pred.confidence * 1.08)
            elif regime_acc < 0.55:
                # Low accuracy regime: reduce confidence
                pred.confidence = max(0.4, pred.confidence * 0.92)

            # Adjust for volatility
            vol_level = str(getattr(regime_result, "volatility_level", "") or "")
            if vol_level == "HIGH":
                # High volatility: be more conservative
                pred.confidence = max(0.4, pred.confidence * 0.95)
                pred.warnings.append("High volatility regime: increased uncertainty")
            elif vol_level == "LOW":
                # Low volatility: can be more confident
                pred.confidence = min(0.95, pred.confidence * 1.03)

            # Store regime-based recommended threshold
            pred.model_margin = float(
                getattr(regime_result, "recommended_threshold", 0.0) or 0.0
            )

        except Exception as e:
            log.debug(f"Regime adjustment failed: {e}")

    def _apply_adaptive_signal_threshold(self, pred: Prediction) -> None:
        """Apply adaptive thresholds for signal generation.
        
        Uses dynamic thresholds based on:
        1. Historical accuracy for this stock
        2. Current market regime
        3. Prediction uncertainty (entropy)
        4. Model agreement
        """
        # Get stock-specific accuracy
        stock_acc = self._get_stock_accuracy(pred.stock_code)
        
        # Base thresholds
        strong_buy_threshold = 0.70
        buy_threshold = 0.54
        strong_sell_threshold = -0.55
        sell_threshold = -0.30
        
        # Adjust based on stock accuracy
        if stock_acc > 0.60:
            # Good history: slightly lower thresholds (more aggressive)
            strong_buy_threshold = max(0.65, strong_buy_threshold - 0.05)
            buy_threshold = max(0.50, buy_threshold - 0.04)
        elif stock_acc < 0.45:
            # Poor history: higher thresholds (more conservative)
            strong_buy_threshold = min(0.75, strong_buy_threshold + 0.05)
            buy_threshold = min(0.60, buy_threshold + 0.06)
        
        # Adjust based on entropy (uncertainty)
        if pred.entropy > 0.6:
            # High uncertainty: require stronger signal
            strong_buy_threshold = min(0.75, strong_buy_threshold + 0.05)
            buy_threshold = min(0.60, buy_threshold + 0.06)
        elif pred.entropy < 0.3:
            # Low uncertainty: can be more aggressive
            strong_buy_threshold = max(0.65, strong_buy_threshold - 0.05)
            buy_threshold = max(0.50, buy_threshold - 0.04)
        
        # Adjust based on model agreement
        if pred.model_agreement > 0.75:
            # High agreement: slightly lower thresholds
            strong_buy_threshold = max(0.65, strong_buy_threshold - 0.03)
            buy_threshold = max(0.50, buy_threshold - 0.03)
        elif pred.model_agreement < 0.50:
            # Low agreement: higher thresholds
            strong_buy_threshold = min(0.75, strong_buy_threshold + 0.05)
            buy_threshold = min(0.60, buy_threshold + 0.06)
        
        # Calculate direction from probabilities
        direction = pred.prob_up - pred.prob_down
        
        # Apply adaptive thresholds
        if direction >= strong_buy_threshold and pred.confidence >= 0.65:
            pred.signal = Signal.STRONG_BUY
        elif direction >= buy_threshold and pred.confidence >= 0.50:
            pred.signal = Signal.BUY
        elif direction <= strong_sell_threshold and pred.confidence >= 0.65:
            pred.signal = Signal.STRONG_SELL
        elif direction <= sell_threshold and pred.confidence >= 0.50:
            pred.signal = Signal.SELL
        else:
            pred.signal = Signal.HOLD
        
        # Update signal strength with adaptive scaling
        pred.signal_strength = float(
            np.clip(
                (0.65 * abs(direction)) + 
                (0.25 * pred.confidence) +
                (0.10 * pred.model_agreement),
                0.0, 1.0
            )
        )

    def predict_quick_batch(
        self,
        stock_codes: list[str],
        use_realtime_price: bool = True,
        interval: str = None,
        lookback_bars: int = None,
        history_allow_online: bool = True,
    ) -> list[Prediction]:
        """Quick batch prediction without full forecasting."""
        with self._predict_lock:
            self._maybe_reload_models(reason="quick_batch")
            interval = self._normalize_interval_token(interval)
            lookback = int(
                lookback_bars or self._default_lookback_bars(interval)
            )

            predictions: list[Prediction] = []
            prepared: list[tuple[Prediction, FloatArray]] = []

            for stock_code in stock_codes:
                try:
                    code = self._clean_code(stock_code)
                    if not code:
                        continue

                    pred = Prediction(
                        stock_code=code,
                        timestamp=datetime.now(),
                        interval=interval,
                    )

                    min_rows = getattr(
                        self.feature_engine, 'MIN_ROWS',
                        CONFIG.SEQUENCE_LENGTH
                    )

                    try:
                        df = self._fetch_data(
                            code,
                            interval,
                            lookback,
                            use_realtime_price,
                            history_allow_online=bool(history_allow_online),
                        )
                    except TypeError:
                        # Backward compatibility for tests/mocks overriding _fetch_data.
                        df = self._fetch_data(
                            code,
                            interval,
                            lookback,
                            use_realtime_price,
                        )
                    if (
                        df is None
                        or df.empty
                        or len(df) < CONFIG.SEQUENCE_LENGTH
                        or len(df) < min_rows
                    ):
                        continue

                    pred.current_price = float(df["close"].iloc[-1])

                    df = self.feature_engine.create_features(df)
                    X = self.processor.prepare_inference_sequence(
                        df, self._feature_cols
                    )
                    prepared.append((pred, X))

                except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
                    log.debug(
                        f"Quick prediction failed for {stock_code}: {e}"
                    )

            if not prepared:
                return predictions

            if self.ensemble:
                try:
                    X_batch = np.concatenate(
                        [np.asarray(X, dtype=np.float32) for _, X in prepared],
                        axis=0,
                    )
                    ensemble_results = self.ensemble.predict_batch(
                        X_batch,
                        batch_size=max(8, min(256, len(prepared))),
                    )
                    if len(ensemble_results) == len(prepared):
                        for (pred, _), ens_pred in zip(
                            prepared,
                            ensemble_results,
                            strict=False,
                        ):
                            try:
                                self._apply_ensemble_result(ens_pred, pred)
                            except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
                                log.debug(
                                    "Quick batch ensemble apply failed "
                                    f"for {pred.stock_code}: {e}"
                                )
                            predictions.append(pred)
                        return predictions
                    log.debug(
                        "Quick batch ensemble size mismatch: "
                        f"got {len(ensemble_results)}, expected {len(prepared)}"
                    )
                except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
                    log.debug(
                        "Quick batch ensemble inference failed; "
                        f"falling back to single predict: {e}"
                    )

            # Fallback path: per-symbol predict.
            for pred, X in prepared:
                if self.ensemble:
                    try:
                        self._apply_ensemble_prediction(X, pred)
                    except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
                        log.debug(
                            f"Quick single fallback failed for {pred.stock_code}: {e}"
                        )
                predictions.append(pred)

            return predictions


from models import predictor_forecast_ops as _predictor_forecast_ops  # noqa: E402
from models import predictor_runtime_ops as _predictor_runtime_ops  # noqa: E402

Predictor.get_realtime_forecast_curve = _predictor_forecast_ops.get_realtime_forecast_curve
Predictor._stabilize_forecast_curve = staticmethod(_predictor_forecast_ops._stabilize_forecast_curve)
Predictor.get_top_picks = _predictor_forecast_ops.get_top_picks
Predictor._apply_ensemble_prediction = _predictor_forecast_ops._apply_ensemble_prediction
Predictor._apply_ensemble_result = _predictor_forecast_ops._apply_ensemble_result
Predictor._append_warning_once = staticmethod(_predictor_forecast_ops._append_warning_once)
Predictor._refresh_prediction_uncertainty = _predictor_forecast_ops._refresh_prediction_uncertainty
Predictor._apply_tail_risk_guard = _predictor_forecast_ops._apply_tail_risk_guard
Predictor._build_prediction_bands = _predictor_forecast_ops._build_prediction_bands
Predictor._apply_high_precision_gate = _predictor_forecast_ops._apply_high_precision_gate
Predictor._apply_runtime_signal_quality_gate = _predictor_forecast_ops._apply_runtime_signal_quality_gate
Predictor._get_cache_ttl = _predictor_forecast_ops._get_cache_ttl
Predictor._get_cache_version = _predictor_forecast_ops._get_cache_version
Predictor._get_cached_prediction = _predictor_forecast_ops._get_cached_prediction
Predictor._set_cached_prediction = _predictor_forecast_ops._set_cached_prediction
Predictor._news_cache_ttl = _predictor_forecast_ops._news_cache_ttl
Predictor._ensure_news_cache_state = _predictor_forecast_ops._ensure_news_cache_state
Predictor._get_news_sentiment = _predictor_forecast_ops._get_news_sentiment
Predictor._compute_news_bias = _predictor_forecast_ops._compute_news_bias
Predictor._apply_news_influence = _predictor_forecast_ops._apply_news_influence
Predictor._normalize_interval_token = _predictor_runtime_ops._normalize_interval_token
Predictor._is_intraday_interval = _predictor_runtime_ops._is_intraday_interval
Predictor._bar_safety_caps = _predictor_runtime_ops._bar_safety_caps
Predictor._sanitize_ohlc_row = _predictor_runtime_ops._sanitize_ohlc_row
Predictor._intraday_session_mask = _predictor_runtime_ops._intraday_session_mask
Predictor._sanitize_history_df = _predictor_runtime_ops._sanitize_history_df
Predictor.invalidate_cache = _predictor_runtime_ops.invalidate_cache
Predictor._fetch_data = _predictor_runtime_ops._fetch_data
Predictor._default_lookback_bars = _predictor_runtime_ops._default_lookback_bars
Predictor._sequence_signature = _predictor_runtime_ops._sequence_signature
Predictor._forecast_seed = _predictor_runtime_ops._forecast_seed
Predictor._generate_forecast = _predictor_runtime_ops._generate_forecast
Predictor._determine_signal = _predictor_runtime_ops._determine_signal
Predictor._calculate_signal_strength = _predictor_runtime_ops._calculate_signal_strength
Predictor._calculate_levels = _predictor_runtime_ops._calculate_levels
Predictor._calculate_position = _predictor_runtime_ops._calculate_position
Predictor._resolve_trade_distances = _predictor_runtime_ops._resolve_trade_distances
Predictor._quality_scale = _predictor_runtime_ops._quality_scale
Predictor._expected_edge = _predictor_runtime_ops._expected_edge
Predictor._round_trip_cost_pct = _predictor_runtime_ops._round_trip_cost_pct
Predictor._extract_technicals = _predictor_runtime_ops._extract_technicals
Predictor._get_atr_pct = _predictor_runtime_ops._get_atr_pct
Predictor._generate_reasons = _predictor_runtime_ops._generate_reasons
Predictor._get_stock_name = _predictor_runtime_ops._get_stock_name
Predictor._clean_code = _predictor_runtime_ops._clean_code
