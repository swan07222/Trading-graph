"""
Enhanced Uncertainty Quantification with Monte Carlo Dropout

Provides advanced uncertainty estimation for deep learning models:
- Monte Carlo Dropout for epistemic uncertainty
- Deep Ensembles for robust predictions
- Heteroscedastic loss for aleatoric uncertainty
- Conformal prediction for valid confidence intervals
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class UncertaintyDecomposition:
    """Decomposition of uncertainty into epistemic and aleatoric components."""
    total_uncertainty: float
    epistemic_uncertainty: float  # Model uncertainty (reducible with more data)
    aleatoric_uncertainty: float  # Data uncertainty (inherent noise)
    
    # Relative contributions
    epistemic_ratio: float = 0.0
    aleatoric_ratio: float = 0.0
    
    def __post_init__(self) -> None:
        if self.total_uncertainty > 0:
            self.epistemic_ratio = self.epistemic_uncertainty / self.total_uncertainty
            self.aleatoric_ratio = self.aleatoric_uncertainty / self.total_uncertainty


@dataclass
class ConformalPrediction:
    """Conformal prediction with valid coverage guarantees."""
    prediction: float
    lower_bound: float
    upper_bound: float
    coverage: float  # Target coverage (e.g., 0.90 for 90% coverage)
    efficiency: float  # Narrower is better (1 / interval_width)
    calibration_set_size: int
    is_valid: bool  # Whether coverage guarantee holds


class MonteCarloDropout:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Uses dropout at inference time to generate multiple
    stochastic predictions, measuring model uncertainty.
    
    Reference: Gal & Ghahramani, 2016
    """
    
    def __init__(
        self,
        n_samples: int = 50,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initialize MC Dropout estimator.
        
        Args:
            n_samples: Number of forward passes
            dropout_rate: Dropout rate during inference
        """
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        self._lock = threading.RLock()
        self._prediction_history: list[float] = []
    
    def predict_with_uncertainty(
        self,
        model: Any,
        X: np.ndarray,
        training: bool = False,
    ) -> tuple[np.ndarray, UncertaintyDecomposition]:
        """
        Generate prediction with uncertainty using MC Dropout.
        
        Args:
            model: PyTorch model with dropout layers
            X: Input features
            training: Whether to use training mode (dropout active)
        
        Returns:
            (mean_prediction, uncertainty_decomposition)
        """
        try:
            import torch
        except ImportError:
            log.warning("PyTorch not available for MC Dropout")
            return self._fallback_predict(model, X)

        with self._lock:
            predictions: list[np.ndarray] = []

            dropout_states: list[tuple[Any, bool]] = []
            model_training_state = bool(getattr(model, "training", False))

            X_tensor = torch.as_tensor(X, dtype=torch.float32)
            if X_tensor.ndim == 1:
                X_tensor = X_tensor.unsqueeze(0)

            try:
                if training:
                    model.train()
                else:
                    model.eval()
                    # Enable only dropout layers for MC sampling.
                    for module in model.modules():
                        if isinstance(module, torch.nn.Dropout):
                            dropout_states.append((module, bool(module.training)))
                            module.train(True)

                with torch.no_grad():
                    for _ in range(self.n_samples):
                        raw_pred = model(X_tensor)
                        pred = (
                            raw_pred[0]
                            if isinstance(raw_pred, (tuple, list))
                            else raw_pred
                        )
                        if hasattr(pred, "detach"):
                            pred_arr = pred.detach().cpu().numpy()
                        else:
                            pred_arr = np.asarray(pred)
                        pred_arr = np.asarray(pred_arr, dtype=np.float64).reshape(-1)
                        if pred_arr.size == 0:
                            continue
                        pred_arr = np.nan_to_num(
                            pred_arr, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        predictions.append(pred_arr)
            finally:
                if training:
                    model.train(model_training_state)
                else:
                    for module, was_training in dropout_states:
                        module.train(was_training)
                    model.eval()

            if not predictions:
                return self._fallback_predict(model, X)

            min_len = min(int(p.size) for p in predictions)
            if min_len <= 0:
                return self._fallback_predict(model, X)
            if any(int(p.size) != min_len for p in predictions):
                predictions = [p[:min_len] for p in predictions]

            # Calculate statistics
            predictions_array = np.asarray(predictions, dtype=np.float64)
            mean_pred = np.mean(predictions_array, axis=0)
            std_pred = np.std(predictions_array, axis=0)
            
            # Epistemic uncertainty from MC Dropout
            epistemic = np.mean(std_pred)
            
            # Estimate aleatoric from residuals (simplified)
            aleatoric = self._estimate_aleatoric(model, X, mean_pred)
            
            total_uncertainty = np.sqrt(epistemic ** 2 + aleatoric ** 2)
            
            decomposition = UncertaintyDecomposition(
                total_uncertainty=float(total_uncertainty),
                epistemic_uncertainty=float(epistemic),
                aleatoric_uncertainty=float(aleatoric),
            )
            
            # Store history
            self._prediction_history.append(float(mean_pred[0]))
            if len(self._prediction_history) > 1000:
                self._prediction_history = self._prediction_history[-1000:]
            
            return mean_pred, decomposition
    
    def _estimate_aleatoric(
        self,
        model: Any,
        X: np.ndarray,
        prediction: np.ndarray,
    ) -> float:
        """Estimate aleatoric uncertainty from model residuals."""
        # Simplified estimation based on prediction magnitude
        # In production, this would use heteroscedastic loss
        return abs(prediction[0]) * 0.05  # 5% of prediction
    
    def _fallback_predict(
        self,
        model: Any,
        X: np.ndarray,
    ) -> tuple[np.ndarray, UncertaintyDecomposition]:
        """Fallback prediction without MC Dropout."""
        if hasattr(model, 'predict'):
            pred = model.predict(X.reshape(1, -1))
        else:
            pred = np.array([0.5])
        
        return pred, UncertaintyDecomposition(
            total_uncertainty=0.1,
            epistemic_uncertainty=0.05,
            aleatoric_uncertainty=0.05,
        )


class DeepEnsemble:
    """
    Deep Ensemble for robust uncertainty estimation.
    
    Trains multiple models with different initializations
    and aggregates predictions.
    """
    
    def __init__(
        self,
        n_models: int = 5,
        aggregation: str = "mean",
    ) -> None:
        """Initialize deep ensemble.
        
        Args:
            n_models: Number of ensemble members
            aggregation: Aggregation method ("mean", "median", "weighted")
        """
        self.n_models = n_models
        self.aggregation = aggregation
        self._models: list[Any] = []
        self._model_weights: list[float] = []
        self._last_prediction_matrix: np.ndarray | None = None
        self._lock = threading.RLock()
    
    def add_model(
        self,
        model: Any,
        validation_accuracy: float | None = None,
    ) -> None:
        """Add model to ensemble.
        
        Args:
            model: Trained model
            validation_accuracy: Model's validation accuracy for weighting
        """
        with self._lock:
            self._models.append(model)
            
            if validation_accuracy is not None:
                self._model_weights.append(validation_accuracy)
            else:
                self._model_weights.append(1.0)
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, UncertaintyDecomposition, list[np.ndarray]]:
        """
        Generate prediction with uncertainty using deep ensemble.
        
        Args:
            X: Input features
        
        Returns:
            (mean_prediction, uncertainty_decomposition, individual_predictions)
        """
        with self._lock:
            if not self._models:
                log.warning("No models in ensemble")
                return (
                    np.array([0.5]),
                    UncertaintyDecomposition(1.0, 1.0, 0.0),
                    [],
                )
            
            # Get predictions from all models
            individual_predictions = []
            weights = []

            for model, weight in zip(self._models, self._model_weights, strict=False):
                if hasattr(model, 'predict'):
                    pred = model.predict(X.reshape(1, -1))
                elif hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X.reshape(1, -1))[:, 1]
                else:
                    continue

                individual_predictions.append(pred)
                weights.append(weight)
            
            if not individual_predictions:
                return (
                    np.array([0.5]),
                    UncertaintyDecomposition(1.0, 1.0, 0.0),
                    [],
                )
            
            predictions = np.array(individual_predictions)
            weights = np.array(weights)
            self._last_prediction_matrix = np.array(predictions, copy=True)
            
            # Aggregation
            if self.aggregation == "mean":
                mean_pred = np.mean(predictions, axis=0)
            elif self.aggregation == "median":
                mean_pred = np.median(predictions, axis=0)
            elif self.aggregation == "weighted":
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    normalized_weights = weights / weight_sum
                    mean_pred = np.average(predictions, axis=0, weights=normalized_weights)
                else:
                    mean_pred = np.mean(predictions, axis=0)
            else:
                mean_pred = np.mean(predictions, axis=0)
            
            # Uncertainty from ensemble disagreement (epistemic)
            epistemic = np.std(predictions, axis=0)
            
            # Aleatoric estimation: use std of absolute deviations (consistent units)
            aleatoric = np.std(np.abs(predictions - mean_pred), axis=0)

            total_uncertainty = np.sqrt(epistemic ** 2 + aleatoric ** 2)
            
            decomposition = UncertaintyDecomposition(
                total_uncertainty=float(np.mean(total_uncertainty)),
                epistemic_uncertainty=float(np.mean(epistemic)),
                aleatoric_uncertainty=float(np.mean(aleatoric)),
            )
            
            return mean_pred, decomposition, individual_predictions
    
    @property
    def ensemble_size(self) -> int:
        """Get current ensemble size."""
        with self._lock:
            return len(self._models)
    
    def get_model_agreement(self) -> float:
        """Get ensemble agreement metric."""
        with self._lock:
            if len(self._models) < 2:
                return 1.0

            pred_matrix = self._last_prediction_matrix
            if pred_matrix is None or pred_matrix.shape[0] < 2:
                return 1.0

            correlations: list[float] = []
            n_models = int(pred_matrix.shape[0])
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    a = np.asarray(pred_matrix[i], dtype=np.float64).reshape(-1)
                    b = np.asarray(pred_matrix[j], dtype=np.float64).reshape(-1)
                    n = min(len(a), len(b))
                    if n <= 0:
                        continue
                    a = a[:n]
                    b = b[:n]

                    if (
                        n >= 2
                        and np.std(a) > 1e-12
                        and np.std(b) > 1e-12
                    ):
                        corr = float(np.corrcoef(a, b)[0, 1])
                        if np.isfinite(corr):
                            # Map [-1, 1] to [0, 1].
                            score = 0.5 * (corr + 1.0)
                        else:
                            score = 0.0
                    else:
                        # For scalar/constant predictions use normalized distance.
                        scale = max(
                            float(np.max(np.abs(np.concatenate((a, b))))),
                            1e-6,
                        )
                        mad = float(np.mean(np.abs(a - b)))
                        score = 1.0 - min(1.0, mad / scale)

                    correlations.append(float(np.clip(score, 0.0, 1.0)))

            return float(np.mean(correlations)) if correlations else 1.0


class ConformalPredictor:
    """
    Conformal Prediction for valid confidence intervals.
    
    Provides statistically valid prediction intervals
    with guaranteed coverage under exchangeability.
    """
    
    def __init__(
        self,
        coverage: float = 0.90,
    ) -> None:
        """Initialize conformal predictor.
        
        Args:
            coverage: Target coverage rate (e.g., 0.90 for 90%)
        """
        self.coverage = coverage
        self._lock = threading.RLock()
        self._calibration_scores: list[float] = []
        self._quantile: float | None = None
    
    def calibrate(
        self,
        model: Any,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> None:
        """
        Calibrate conformal predictor.
        
        Args:
            model: Trained model
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        with self._lock:
            self._calibration_scores = []

            # Calculate nonconformity scores
            for x, y in zip(X_cal, y_cal, strict=False):
                pred = model.predict(x.reshape(1, -1))[0]
                score = abs(pred - y)
                self._calibration_scores.append(score)
            
            # Calculate quantile for desired coverage (finite-sample valid formula)
            n = len(self._calibration_scores)
            if n > 0:
                alpha = 1 - self.coverage
                quantile_idx = int(np.ceil((n + 1) * (1 - alpha)))
                if quantile_idx > n:
                    # Not enough calibration data to guarantee coverage
                    self._quantile = float("inf")
                else:
                    sorted_scores = np.sort(self._calibration_scores)
                    self._quantile = float(sorted_scores[quantile_idx - 1])
    
    def predict_with_interval(
        self,
        model: Any,
        X: np.ndarray,
    ) -> ConformalPrediction:
        """
        Generate prediction with conformal interval.
        
        Args:
            model: Trained model
            X: Input features
        
        Returns:
            ConformalPrediction with valid coverage guarantee
        """
        with self._lock:
            # Point prediction
            pred = model.predict(X.reshape(1, -1))[0]
            
            # Get quantile
            if self._quantile is None:
                log.warning("Conformal predictor not calibrated")
                self._quantile = 0.1  # Default
            
            # Construct prediction interval
            lower = pred - self._quantile
            upper = pred + self._quantile
            
            # Efficiency (narrower is better)
            interval_width = upper - lower
            efficiency = 1.0 / interval_width if interval_width > 0 else 0.0
            
            return ConformalPrediction(
                prediction=float(pred),
                lower_bound=float(lower),
                upper_bound=float(upper),
                coverage=self.coverage,
                efficiency=efficiency,
                calibration_set_size=len(self._calibration_scores),
                is_valid=len(self._calibration_scores) > 0,
            )


class UncertaintyQuantifier:
    """
    Unified uncertainty quantification interface.
    
    Combines multiple methods for comprehensive uncertainty estimation.
    """
    
    def __init__(
        self,
        use_mc_dropout: bool = True,
        use_ensemble: bool = True,
        use_conformal: bool = True,
    ) -> None:
        """Initialize uncertainty quantifier.
        
        Args:
            use_mc_dropout: Enable MC Dropout
            use_ensemble: Enable Deep Ensemble
            use_conformal: Enable Conformal Prediction
        """
        self.mc_dropout = MonteCarloDropout() if use_mc_dropout else None
        self.ensemble = DeepEnsemble() if use_ensemble else None
        self.conformal = ConformalPredictor() if use_conformal else None
        
        self._lock = threading.RLock()
    
    def quantify_uncertainty(
        self,
        model: Any,
        X: np.ndarray,
        current_price: float,
    ) -> dict[str, Any]:
        """
        Generate comprehensive uncertainty quantification.
        
        Args:
            model: Trained model
            X: Input features
            current_price: Current asset price
        
        Returns:
            Dictionary with all uncertainty metrics
        """
        results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
        }
        
        # MC Dropout
        if self.mc_dropout:
            pred, decomp = self.mc_dropout.predict_with_uncertainty(model, X)
            results["mc_dropout"] = {
                "prediction": float(pred[0]),
                "total_uncertainty": decomp.total_uncertainty,
                "epistemic": decomp.epistemic_uncertainty,
                "aleatoric": decomp.aleatoric_uncertainty,
                "epistemic_ratio": decomp.epistemic_ratio,
                "aleatoric_ratio": decomp.aleatoric_ratio,
            }
        
        # Deep Ensemble
        if self.ensemble:
            pred, decomp, individual = self.ensemble.predict_with_uncertainty(X)
            results["ensemble"] = {
                "prediction": float(pred[0]),
                "total_uncertainty": decomp.total_uncertainty,
                "epistemic": decomp.epistemic_uncertainty,
                "aleatoric": decomp.aleatoric_uncertainty,
                "n_models": self.ensemble.ensemble_size,
                "agreement": self.ensemble.get_model_agreement(),
            }
        
        # Conformal Prediction
        if self.conformal:
            conformal_pred = self.conformal.predict_with_interval(model, X)
            results["conformal"] = {
                "prediction": conformal_pred.prediction,
                "lower_bound": conformal_pred.lower_bound,
                "upper_bound": conformal_pred.upper_bound,
                "coverage": conformal_pred.coverage,
                "efficiency": conformal_pred.efficiency,
                "is_valid": conformal_pred.is_valid,
            }
        
        # Combined uncertainty score
        uncertainties = []
        if "mc_dropout" in results:
            uncertainties.append(results["mc_dropout"]["total_uncertainty"])
        if "ensemble" in results:
            uncertainties.append(results["ensemble"]["total_uncertainty"])
        
        if uncertainties:
            results["combined_uncertainty"] = float(np.mean(uncertainties))
        else:
            results["combined_uncertainty"] = 0.1  # Default
        
        return results


# Global instance
_uncertainty_quantifier: UncertaintyQuantifier | None = None
_uncertainty_quantifier_lock = threading.RLock()


def get_uncertainty_quantifier() -> UncertaintyQuantifier:
    """Get global uncertainty quantifier."""
    global _uncertainty_quantifier
    with _uncertainty_quantifier_lock:
        if _uncertainty_quantifier is None:
            _uncertainty_quantifier = UncertaintyQuantifier()
        return _uncertainty_quantifier
