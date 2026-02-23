from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import CONFIG
from data.processor import DataProcessor
from utils.logger import get_logger

log = get_logger(__name__)

class RealtimePredictor:
    """High-level helper for real-time AI predictions.

    Combines FeatureEngine, DataProcessor (with scaler), and
    EnsembleModel for end-to-end live prediction.
    """

    def __init__(
        self,
        interval: str = "1m",
        horizon: int = 30,
        auto_load: bool = True,
    ) -> None:
        self.interval = str(interval).lower()
        self.horizon = int(horizon)

        self.processor = DataProcessor()
        self.feature_engine: Any | None = None
        self.ensemble: Any | None = None
        self.forecaster: Any | None = None

        self._feature_cols: list[str] = []
        self._loaded = False
        self._loading = False
        self._lock = threading.RLock()
        self._device: str = "cpu"

        if auto_load:
            self.load_models()

    def load_models(self) -> bool:
        """Load all required models for prediction.

        FIX RACE: Uses double-check locking to prevent redundant loads
        when multiple threads call load_models() simultaneously.
        """
        # Fast path: already loaded
        if self._loaded:
            return True

        with self._lock:
            # Double-check under lock
            if self._loaded:
                return True

            if self._loading:
                log.debug("load_models() already in progress, waiting...")
                return self._loaded

            self._loading = True

        try:
            return self._do_load_models()
        finally:
            with self._lock:
                self._loading = False

    def _do_load_models(self) -> bool:
        """Internal model loading implementation."""
        from data.features import FeatureEngine
        from models.ensemble import EnsembleModel

        try:
            self.feature_engine = FeatureEngine()
            self._feature_cols = (
                self.feature_engine.get_feature_columns()
            )

            scaler_path = (
                CONFIG.MODEL_DIR
                / f"scaler_{self.interval}_{self.horizon}.pkl"
            )
            if not self.processor.load_scaler(str(scaler_path)):
                log.warning(f"Failed to load scaler: {scaler_path}")
                return False

            if (
                self.processor.n_features is not None
                and self.processor.n_features != len(self._feature_cols)
            ):
                log.error(
                    f"Feature count mismatch: scaler expects "
                    f"{self.processor.n_features}, engine produces "
                    f"{len(self._feature_cols)}"
                )
                return False

            ensemble_path = (
                CONFIG.MODEL_DIR
                / f"ensemble_{self.interval}_{self.horizon}.pt"
            )
            if ensemble_path.exists():
                self.ensemble = EnsembleModel(
                    input_size=self.processor.n_features
                    or len(self._feature_cols)
                )
                if not self.ensemble.load(str(ensemble_path)):
                    log.warning(
                        f"Failed to load ensemble: {ensemble_path}"
                    )
                    self.ensemble = None
                else:
                    self._device = self.ensemble.device

            forecast_path = (
                CONFIG.MODEL_DIR
                / f"forecast_{self.interval}_{self.horizon}.pt"
            )
            if forecast_path.exists():
                self._load_forecaster(forecast_path)

            with self._lock:
                self._loaded = True

            log.info(
                f"Models loaded: interval={self.interval}, "
                f"horizon={self.horizon}, device={self._device}"
            )
            return True

        except Exception as e:
            log.error(f"Failed to load models: {e}")
            return False

    def _load_forecaster(self, path: Path) -> None:
        """Load TCN forecaster model.

        The forecaster outputs ``horizon`` regression values (not class
        probabilities), so ``num_classes`` in the TCN is set to the
        horizon value from the saved checkpoint.

        Secure-by-default:
        - requires checksum sidecar when configured
        - blocks legacy weights_only=False fallback unless explicitly enabled
        """
        from models.networks import TCNModel

        try:
            if not self.processor._verify_artifact_checksum(path):
                return

            model_cfg = getattr(CONFIG, "model", None)
            allow_unsafe = bool(
                getattr(model_cfg, "allow_unsafe_artifact_load", False)
            )
            require_checksum = bool(
                getattr(model_cfg, "require_artifact_checksum", True)
            )

            def _load_checkpoint(weights_only: bool):
                from utils.atomic_io import torch_load

                return torch_load(
                    path,
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
                        path,
                        exc,
                    )
                    return
                log.warning(
                    "Forecaster weights-only load failed for %s; "
                    "falling back to unsafe legacy checkpoint load: %s",
                    path,
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

            output_size = int(data["horizon"])

            self.forecaster = TCNModel(
                input_size=int(data["input_size"]),
                hidden_size=int(data["arch"]["hidden_size"]),
                num_classes=output_size,
                dropout=float(data["arch"]["dropout"]),
            )
            self.forecaster.load_state_dict(data["state_dict"])
            self.forecaster.eval()

            # FIX DEVICE: Move forecaster to same device as ensemble
            self.forecaster.to(self._device)

            log.info(
                f"Forecaster loaded: {path} "
                f"(output_size={output_size}, device={self._device})"
            )
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
            log.warning(f"Failed to load forecaster: {e}")
            self.forecaster = None

    def predict(
        self,
        df: pd.DataFrame,
        include_forecast: bool = True,
    ) -> dict[str, Any] | None:
        """Make real-time prediction.

        FIX LOCK: Does NOT hold self._lock during inference, only during
        state checks. This prevents blocking other threads during the
        potentially slow forward pass.

        Args:
            df: DataFrame with OHLCV data (must have >= MIN_ROWS rows
                for feature computation + SEQUENCE_LENGTH for sequence).
            include_forecast: Whether to include multi-step forecast.

        Returns:
            Dict with signal, confidence, probabilities, etc. or None on failure.
        """
        # Check loaded state under lock (fast)
        with self._lock:
            is_loaded = self._loaded

        if not is_loaded:
            if not self.load_models():
                return None

        # FIX LOCK: Perform inference WITHOUT holding lock
        return self._do_predict(df, include_forecast)

    def _do_predict(
        self,
        df: pd.DataFrame,
        include_forecast: bool,
    ) -> dict[str, Any] | None:
        """Internal prediction logic.

        FIX LOCK: No longer requires self._lock to be held.
        The models are only modified during load_models() which is
        protected by its own locking. During prediction, models are
        read-only (eval mode).
        """
        try:
            min_rows = getattr(self.feature_engine, "MIN_ROWS", 60)
            if len(df) < min_rows:
                log.warning(
                    f"Insufficient data for prediction: "
                    f"{len(df)} rows < {min_rows} minimum"
                )
                return None

            df_features = self.feature_engine.create_features(df.copy())
            X = self.processor.prepare_inference_sequence(
                df_features, self._feature_cols
            )

            num_classes = int(CONFIG.NUM_CLASSES)
            result: dict[str, Any] = {
                "timestamp": datetime.now(),
                "signal": "HOLD",
                "confidence": 0.0,
                "probabilities": [1.0 / num_classes] * num_classes,
                "predicted_class": 1,
            }

            if self.ensemble is not None:
                pred = self.ensemble.predict(X)

                result["probabilities"] = pred.probabilities.tolist()
                result["predicted_class"] = pred.predicted_class
                result["confidence"] = pred.confidence
                result["entropy"] = pred.entropy
                result["agreement"] = pred.agreement

                min_conf = float(CONFIG.MIN_CONFIDENCE)
                strong_buy = float(CONFIG.STRONG_BUY_THRESHOLD)
                strong_sell = float(CONFIG.STRONG_SELL_THRESHOLD)

                if pred.predicted_class == 2 and pred.confidence >= min_conf:
                    if pred.confidence >= strong_buy:
                        result["signal"] = "STRONG_BUY"
                    else:
                        result["signal"] = "BUY"
                elif pred.predicted_class == 0 and pred.confidence >= min_conf:
                    if pred.confidence >= strong_sell:
                        result["signal"] = "STRONG_SELL"
                    else:
                        result["signal"] = "SELL"
                else:
                    result["signal"] = "HOLD"

            if include_forecast and self.forecaster is not None:
                import torch

                self.forecaster.eval()
                with torch.inference_mode():
                    X_tensor = torch.FloatTensor(X).to(self._device)
                    forecast, _ = self.forecaster(X_tensor)
                    result["forecast"] = (
                        forecast[0].cpu().numpy().tolist()
                    )

            return result

        except Exception as e:
            log.error(f"Prediction failed: {e}")
            return None

    def predict_batch(
        self,
        dataframes: dict[str, pd.DataFrame],
        include_forecast: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Make predictions for multiple stocks."""
        results: dict[str, dict[str, Any]] = {}

        for code, df in dataframes.items():
            pred = self.predict(
                df, include_forecast=include_forecast
            )
            if pred is not None:
                results[code] = pred

        return results

    def update_and_predict(
        self,
        code: str,
        new_bar: dict[str, float],
    ) -> dict[str, Any] | None:
        """Update buffer with new bar and make prediction."""
        with self._lock:
            is_loaded = self._loaded

        if not is_loaded:
            if not self.load_models():
                return None

        # FIX LOCK: No lock held during prediction
        return self._do_update_and_predict(code, new_bar)

    def _do_update_and_predict(
        self,
        code: str,
        new_bar: dict[str, float],
    ) -> dict[str, Any] | None:
        """Internal update and predict logic."""
        X = self.processor.prepare_realtime_sequence(
            code, new_bar, self._feature_cols, self.feature_engine
        )

        if X is None:
            return None

        result: dict[str, Any] = {
            "code": code,
            "timestamp": datetime.now(),
            "signal": "HOLD",
            "confidence": 0.0,
        }

        if self.ensemble is not None:
            pred = self.ensemble.predict(X)
            result["probabilities"] = pred.probabilities.tolist()
            result["predicted_class"] = pred.predicted_class
            result["confidence"] = pred.confidence

            min_conf = float(CONFIG.MIN_CONFIDENCE)

            if pred.predicted_class == 2 and pred.confidence >= min_conf:
                result["signal"] = "BUY"
            elif pred.predicted_class == 0 and pred.confidence >= min_conf:
                result["signal"] = "SELL"

        return result

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        with self._lock:
            return self._loaded

    @property
    def device(self) -> str:
        """Get the device models are running on."""
        with self._lock:
            return self._device
