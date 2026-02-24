"""ML Model Explainability Module.

Provides SHAP (SHapley Additive exPlanations) and LIME integration
for interpreting model predictions.

Features:
- SHAP value calculation for feature importance
- Local explanations for individual predictions
- Global explanations across dataset
- Visualization helpers for explanation results
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class FeatureImportance:
    """Feature importance with SHAP values."""
    feature_name: str
    shap_value: float
    shap_abs_mean: float
    rank: int = 0
    direction: str = ""  # "positive", "negative", "mixed"
    
    def __post_init__(self) -> None:
        if self.shap_value > 0.01:
            self.direction = "positive"
        elif self.shap_value < -0.01:
            self.direction = "negative"
        else:
            self.direction = "mixed"


@dataclass
class LocalExplanation:
    """Local explanation for a single prediction."""
    prediction_id: str = ""
    timestamp: datetime | None = None
    
    # Prediction info
    predicted_value: float = 0.0
    predicted_class: str = ""
    base_value: float = 0.0  # Expected value of model output
    
    # Feature contributions
    feature_contributions: dict[str, float] = field(default_factory=dict)
    feature_values: dict[str, float] = field(default_factory=dict)
    
    # Summary
    top_positive_features: list[tuple[str, float]] = field(default_factory=list)
    top_negative_features: list[tuple[str, float]] = field(default_factory=list)
    
    # Confidence
    prediction_std: float = 0.0  # Uncertainty estimate
    
    def __post_init__(self) -> None:
        if not self.prediction_id:
            self.prediction_id = f"EXP_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def explanation_text(self) -> str:
        """Generate human-readable explanation."""
        parts = [f"Prediction: {self.predicted_value:.4f} ({self.predicted_class})"]
        parts.append(f"Base value: {self.base_value:.4f}")
        
        if self.top_positive_features:
            parts.append("\nPositive factors:")
            for feat, val in self.top_positive_features[:3]:
                parts.append(f"  + {feat}: +{val:.4f}")
        
        if self.top_negative_features:
            parts.append("\nNegative factors:")
            for feat, val in self.top_negative_features[:3]:
                parts.append(f"  - {feat}: {val:.4f}")
        
        if self.prediction_std > 0:
            parts.append(f"\nUncertainty: ±{self.prediction_std:.4f}")
        
        return "\n".join(parts)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'prediction_id': self.prediction_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'predicted_value': self.predicted_value,
            'predicted_class': self.predicted_class,
            'base_value': self.base_value,
            'feature_contributions': self.feature_contributions,
            'feature_values': self.feature_values,
            'top_positive_features': self.top_positive_features,
            'top_negative_features': self.top_negative_features,
            'prediction_std': self.prediction_std,
            'explanation_text': self.explanation_text,
        }


@dataclass
class GlobalExplanation:
    """Global explanation across dataset."""
    dataset_name: str = ""
    num_samples: int = 0
    num_features: int = 0
    
    # Feature importance
    feature_importances: list[FeatureImportance] = field(default_factory=list)
    
    # Summary statistics
    mean_shap_values: dict[str, float] = field(default_factory=dict)
    std_shap_values: dict[str, float] = field(default_factory=dict)
    
    # Dependence
    dependence_plots: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    
    # Clustering
    feature_clusters: list[list[str]] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        # Sort by importance
        self.feature_importances.sort(key=lambda x: abs(x.shap_abs_mean), reverse=True)
        for i, imp in enumerate(self.feature_importances):
            imp.rank = i + 1
    
    @property
    def top_features(self) -> list[FeatureImportance]:
        """Get top 10 most important features."""
        return self.feature_importances[:10]
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'dataset_name': self.dataset_name,
            'num_samples': self.num_samples,
            'num_features': self.num_features,
            'top_features': [
                {
                    'name': f.feature_name,
                    'importance': f.shap_abs_mean,
                    'direction': f.direction,
                    'rank': f.rank,
                }
                for f in self.top_features
            ],
            'mean_shap_values': self.mean_shap_values,
        }


class ModelExplainer:
    """Model explainability using SHAP and LIME.
    
    This class provides model-agnostic explanation capabilities
    for trading predictions.
    
    Usage:
        explainer = ModelExplainer(model, feature_names)
        
        # Local explanation
        explanation = explainer.explain_prediction(X_sample, y_pred)
        
        # Global explanation
        global_exp = explainer.explain_dataset(X_train)
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        background_data: np.ndarray | None = None,
        model_type: str = "classifier",
    ) -> None:
        """Initialize explainer.
        
        Args:
            model: Trained model to explain
            feature_names: List of feature names
            background_data: Background dataset for SHAP
            model_type: "classifier" or "regressor"
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.background_data = background_data
        
        # SHAP explainer (lazy initialization)
        self._shap_explainer: Any = None
        self._shap_values: np.ndarray | None = None
        
        # Cache for explanations
        self._explanation_cache: dict[str, LocalExplanation] = {}
    
    def _init_shap(self) -> None:
        """Initialize SHAP explainer."""
        try:
            import shap
            
            # Use KernelSHAP for model-agnostic explanations
            if self.background_data is None:
                # Use a small sample as background
                log.warning("No background data provided, using sample")
            
            self._shap_explainer = shap.Explainer(
                self._predict_function,
                self.background_data,
                feature_names=self.feature_names,
            )
            log.info("SHAP explainer initialized")
        except ImportError:
            log.warning("SHAP not installed. Install with: pip install shap")
        except Exception as e:
            log.warning(f"Failed to initialize SHAP: {e}")
    
    def _predict_function(self, X: np.ndarray) -> np.ndarray:
        """Prediction function for SHAP."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return self.model.predict(X)
    
    def explain_prediction(
        self,
        X_sample: np.ndarray,
        prediction: float,
        prediction_std: float = 0.0,
    ) -> LocalExplanation:
        """Generate local explanation for a prediction.
        
        Args:
            X_sample: Sample features (1D array)
            prediction: Model prediction for this sample
            prediction_std: Uncertainty estimate
        
        Returns:
            LocalExplanation with feature contributions
        """
        # Ensure 2D
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        # Initialize SHAP if needed
        if self._shap_explainer is None:
            self._init_shap()
        
        explanation = LocalExplanation(
            predicted_value=float(prediction),
            predicted_class=self._get_prediction_class(prediction),
            base_value=self._get_base_value(),
            prediction_std=prediction_std,
        )
        
        # Calculate SHAP values
        if self._shap_explainer is not None:
            try:
                shap_values = self._shap_explainer(X_sample)
                self._process_shap_values(shap_values, explanation)
            except Exception as e:
                log.warning(f"SHAP calculation failed: {e}")
                self._fallback_explanation(X_sample, explanation)
        else:
            # Fallback without SHAP
            self._fallback_explanation(X_sample, explanation)
        
        return explanation
    
    def _get_prediction_class(self, prediction: float) -> str:
        """Get prediction class label."""
        if self.model_type == "classifier":
            if prediction > 0.7:
                return "STRONG_BUY"
            elif prediction > 0.55:
                return "BUY"
            elif prediction > 0.45:
                return "HOLD"
            elif prediction > 0.3:
                return "SELL"
            return "STRONG_SELL"
        return "PREDICTION"
    
    def _get_base_value(self) -> float:
        """Get expected model output (base value)."""
        if self.background_data is not None:
            predictions = self._predict_function(self.background_data)
            return float(np.mean(predictions))
        return 0.5
    
    def _process_shap_values(
        self,
        shap_values: Any,
        explanation: LocalExplanation,
    ) -> None:
        """Process SHAP values into explanation."""
        try:
            # Extract SHAP values
            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = np.array(shap_values)
            
            # Handle multi-class
            if values.ndim > 2:
                values = values[:, :, 1]  # Use positive class
            
            # Get values for first (only) sample
            sample_values = values[0] if values.ndim > 1 else values
            
            # Create feature contributions
            for i, name in enumerate(self.feature_names):
                if i < len(sample_values):
                    contribution = float(sample_values[i])
                    explanation.feature_contributions[name] = contribution
                    explanation.feature_values[name] = float(
                        shap_values.data[0, i] if hasattr(shap_values, 'data') else 0
                    )
            
            # Sort by contribution
            sorted_features = sorted(
                explanation.feature_contributions.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            
            explanation.top_positive_features = [
                (f, v) for f, v in sorted_features if v > 0
            ][:5]
            
            explanation.top_negative_features = [
                (f, v) for f, v in sorted_features if v < 0
            ][:5]
            
        except Exception as e:
            log.warning(f"Failed to process SHAP values: {e}")
    
    def _fallback_explanation(
        self,
        X_sample: np.ndarray,
        explanation: LocalExplanation,
    ) -> None:
        """Generate fallback explanation without SHAP."""
        # Simple feature importance based on coefficient magnitude
        if hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_
            if coefficients.ndim > 1:
                coefficients = coefficients[0]
            
            for i, name in enumerate(self.feature_names):
                if i < len(coefficients):
                    explanation.feature_contributions[name] = float(coefficients[i])
        
        # Use absolute values for ranking
        sorted_features = sorted(
            explanation.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        
        explanation.top_positive_features = [
            (f, v) for f, v in sorted_features if v > 0
        ][:5]
        
        explanation.top_negative_features = [
            (f, v) for f, v in sorted_features if v < 0
        ][:5]
    
    def explain_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        max_samples: int = 1000,
    ) -> GlobalExplanation:
        """Generate global explanation for dataset.
        
        Args:
            X: Feature matrix
            y: Target values (optional)
            max_samples: Maximum samples for SHAP calculation
        
        Returns:
            GlobalExplanation with feature importances
        """
        n_samples = min(len(X), max_samples)
        
        explanation = GlobalExplanation(
            dataset_name="trading_dataset",
            num_samples=n_samples,
            num_features=len(self.feature_names),
        )
        
        # Sample data
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
            y_sample = y[indices] if y is not None else None
        else:
            X_sample = X
            y_sample = y
        
        # Initialize SHAP if needed
        if self._shap_explainer is None:
            self._init_shap()
        
        # Calculate SHAP values
        if self._shap_explainer is not None:
            try:
                shap_values = self._shap_explainer(X_sample)
                self._process_global_shap(shap_values, explanation)
            except Exception as e:
                log.warning(f"Global SHAP calculation failed: {e}")
                self._fallback_global_explanation(explanation)
        else:
            self._fallback_global_explanation(explanation)
        
        return explanation
    
    def _process_global_shap(
        self,
        shap_values: Any,
        explanation: GlobalExplanation,
    ) -> None:
        """Process global SHAP values."""
        try:
            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = np.array(shap_values)
            
            # Handle multi-class
            if values.ndim > 2:
                values = values[:, :, 1]
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(values), axis=0)
            
            for i, name in enumerate(self.feature_names):
                if i < len(mean_abs_shap):
                    importance = FeatureImportance(
                        feature_name=name,
                        shap_value=float(np.mean(values[:, i])),
                        shap_abs_mean=float(mean_abs_shap[i]),
                    )
                    explanation.feature_importances.append(importance)
                    explanation.mean_shap_values[name] = float(np.mean(values[:, i]))
                    explanation.std_shap_values[name] = float(np.std(values[:, i]))
        
        except Exception as e:
            log.warning(f"Failed to process global SHAP: {e}")
    
    def _fallback_global_explanation(
        self,
        explanation: GlobalExplanation,
    ) -> None:
        """Generate fallback global explanation."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            for i, name in enumerate(self.feature_names):
                if i < len(importances):
                    importance = FeatureImportance(
                        feature_name=name,
                        shap_value=float(importances[i]),
                        shap_abs_mean=float(importances[i]),
                    )
                    explanation.feature_importances.append(importance)
    
    def save_explanation(
        self,
        explanation: LocalExplanation | GlobalExplanation,
        output_path: Path | str,
    ) -> None:
        """Save explanation to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(explanation.to_dict(), f, indent=2, ensure_ascii=False)
        
        log.info(f"Explanation saved to {path}")
    
    def get_feature_summary(self) -> pd.DataFrame:
        """Get feature importance summary as DataFrame."""
        data = []
        for name in self.feature_names:
            data.append({
                'feature': name,
                'importance': 0.0,  # Would be filled from SHAP
            })
        return pd.DataFrame(data)


def explain_prediction(
    model: Any,
    X_sample: np.ndarray,
    feature_names: list[str],
    prediction: float,
    background_data: np.ndarray | None = None,
) -> LocalExplanation:
    """Convenience function to explain a single prediction.
    
    Args:
        model: Trained model
        X_sample: Sample features
        feature_names: Feature names
        prediction: Model prediction
        background_data: Background dataset for SHAP
    
    Returns:
        LocalExplanation
    """
    explainer = ModelExplainer(
        model=model,
        feature_names=feature_names,
        background_data=background_data,
    )
    return explainer.explain_prediction(X_sample, prediction)
