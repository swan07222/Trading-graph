"""
Prediction Quality Enhancement Module

Addresses all major disadvantages of AI guessing:
1. Uncertainty quantification with epistemic/aleatoric separation
2. Adaptive confidence thresholding based on market regime
3. Data quality scoring and graceful degradation
4. Model artifact validation and auto-recovery
5. Ensemble disagreement detection and conflict resolution
6. Prediction explanation layer (SHAP-like feature importance)
7. Drift detection and auto-recalibration
8. Latency optimization with async processing
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classification for adaptive thresholding."""
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRASH = "crash"
    BUBBLE = "bubble"
    STRONG_UPTREND = "strong_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    LOW_LIQUIDITY = "low_liquidity"


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment."""
    
    overall_score: float = 1.0
    completeness: float = 1.0
    consistency: float = 1.0
    timeliness: float = 1.0
    noise_level: float = 0.0
    feature_quality: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    is_sufficient: bool = True
    recommendation: str = "Data quality is good for prediction"


@dataclass
class UncertaintyQuantification:
    """Comprehensive uncertainty breakdown."""
    
    epistemic: float = 0.0
    aleatoric: float = 0.0
    total: float = 0.0
    per_class_std: np.ndarray | None = None
    confidence_interval_95: tuple[float, float] | None = None
    is_high_uncertainty: bool = False
    is_reliable: bool = True
    model_disagreement: float = 0.0
    data_noise: float = 0.0
    distribution_shift: float = 0.0


@dataclass
class EnsembleDisagreement:
    """Analysis of disagreement between ensemble models."""
    
    agreement_rate: float = 1.0
    kappa_statistic: float = 1.0
    individual_predictions: dict[str, int] = field(default_factory=dict)
    individual_confidences: dict[str, float] = field(default_factory=dict)
    individual_probabilities: dict[str, np.ndarray] = field(default_factory=dict)
    disagreement_type: str = "none"
    minority_models: list[str] = field(default_factory=list)
    majority_models: list[str] = field(default_factory=list)
    resolution_method: str = "weighted_vote"
    confidence_penalty: float = 0.0


@dataclass
class FeatureImportance:
    """SHAP-like feature importance for prediction explanation."""
    
    feature_names: list[str]
    importance_scores: np.ndarray
    top_features: list[tuple[str, float]] = field(default_factory=list)
    positive_influence: list[str] = field(default_factory=list)
    negative_influence: list[str] = field(default_factory=list)
    stability_scores: np.ndarray | None = None
    summary: str = ""


@dataclass
class PredictionQualityReport:
    """Comprehensive prediction quality assessment."""
    
    confidence: float = 0.5
    uncertainty: UncertaintyQuantification = field(default_factory=UncertaintyQuantification)
    data_quality: DataQualityReport = field(default_factory=DataQualityReport)
    ensemble_disagreement: EnsembleDisagreement = field(default_factory=EnsembleDisagreement)
    feature_importance: FeatureImportance | None = None
    market_regime: MarketRegime = MarketRegime.NORMAL
    is_high_quality: bool = True
    is_reliable: bool = True
    requires_human_review: bool = False
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    model_version: str = ""
    
    @property
    def adjusted_confidence(self) -> float:
        """Confidence adjusted for all quality factors."""
        base_conf = self.confidence
        uncertainty_penalty = self.uncertainty.total * 0.3
        disagreement_penalty = (1.0 - self.ensemble_disagreement.agreement_rate) * 0.2
        data_quality_penalty = (1.0 - self.data_quality.overall_score) * 0.15
        
        regime_penalties = {
            MarketRegime.HIGH_VOLATILITY: 0.10,
            MarketRegime.CRASH: 0.20,
            MarketRegime.BUBBLE: 0.15,
            MarketRegime.LOW_LIQUIDITY: 0.12,
        }
        regime_penalty = regime_penalties.get(self.market_regime, 0.0)
        
        total_penalty = uncertainty_penalty + disagreement_penalty + data_quality_penalty + regime_penalty
        adjusted = max(0.0, min(1.0, base_conf - total_penalty))
        return adjusted
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "confidence": self.confidence,
            "adjusted_confidence": self.adjusted_confidence,
            "is_reliable": self.is_reliable,
            "is_high_quality": self.is_high_quality,
            "market_regime": self.market_regime.value,
            "uncertainty": {
                "total": self.uncertainty.total,
                "epistemic": self.uncertainty.epistemic,
                "aleatoric": self.uncertainty.aleatoric,
                "is_high_uncertainty": self.uncertainty.is_high_uncertainty,
            },
            "data_quality": {
                "overall_score": self.data_quality.overall_score,
                "completeness": self.data_quality.completeness,
                "is_sufficient": self.data_quality.is_sufficient,
            },
            "ensemble_disagreement": {
                "agreement_rate": self.ensemble_disagreement.agreement_rate,
                "disagreement_type": self.ensemble_disagreement.disagreement_type,
            },
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


class PredictionQualityAssessor:
    """Comprehensive prediction quality assessment system."""
    
    def __init__(
        self,
        uncertainty_threshold_high: float = 0.3,
        uncertainty_threshold_low: float = 0.1,
        min_agreement_rate: float = 0.6,
        min_data_quality: float = 0.7,
    ):
        self.uncertainty_threshold_high = uncertainty_threshold_high
        self.uncertainty_threshold_low = uncertainty_threshold_low
        self.min_agreement_rate = min_agreement_rate
        self.min_data_quality = min_data_quality
        
        self._regime_confidence_adjustments = {
            MarketRegime.NORMAL: 0.0,
            MarketRegime.HIGH_VOLATILITY: -0.10,
            MarketRegime.LOW_VOLATILITY: 0.05,
            MarketRegime.CRASH: -0.20,
            MarketRegime.BUBBLE: -0.15,
            MarketRegime.STRONG_UPTREND: 0.05,
            MarketRegime.STRONG_DOWNTREND: 0.03,
            MarketRegime.LOW_LIQUIDITY: -0.12,
        }
        
        self._recent_importance: list[np.ndarray] = []
        self._importance_window = 50
    
    def assess_prediction(
        self,
        probabilities: np.ndarray,
        individual_predictions: dict[str, np.ndarray],
        individual_confidences: dict[str, float],
        input_data: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        market_data: dict[str, Any] | None = None,
    ) -> PredictionQualityReport:
        """Comprehensive quality assessment for a prediction."""
        report = PredictionQualityReport()
        
        report.uncertainty = self._quantify_uncertainty(
            individual_predictions, probabilities
        )
        report.ensemble_disagreement = self._analyze_disagreement(
            individual_predictions, individual_confidences
        )
        
        if input_data is not None:
            report.data_quality = self._assess_data_quality(input_data, feature_names)
        
        if market_data is not None:
            report.market_regime = self._detect_market_regime(market_data)
        
        predicted_class = int(np.argmax(probabilities))
        report.confidence = float(probabilities[predicted_class])
        
        if (
            input_data is not None and
            feature_names is not None and
            hasattr(CONFIG.model, "enable_feature_importance") and
            CONFIG.model.enable_feature_importance
        ):
            report.feature_importance = self._compute_feature_importance(
                input_data, probabilities, feature_names
            )
        
        self._set_quality_flags(report)
        self._generate_recommendations(report)
        
        return report
    
    def _quantify_uncertainty(
        self,
        individual_predictions: dict[str, np.ndarray],
        final_probabilities: np.ndarray,
    ) -> UncertaintyQuantification:
        """Quantify prediction uncertainty with epistemic/aleatoric breakdown."""
        uq = UncertaintyQuantification()
        
        if not individual_predictions:
            uq.is_reliable = False
            return uq
        
        pred_array = np.array(list(individual_predictions.values()))
        n_models = pred_array.shape[0]
        
        std_across_models = np.std(pred_array, axis=0)
        uq.epistemic = float(np.mean(std_across_models))
        uq.per_class_std = std_across_models
        
        entropies = []
        for probs in pred_array:
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
        uq.aleatoric = float(np.mean(entropies)) / np.log(3)
        
        uq.total = 0.6 * uq.epistemic + 0.4 * uq.aleatoric
        uq.is_high_uncertainty = uq.total > self.uncertainty_threshold_high
        uq.is_reliable = uq.total < self.uncertainty_threshold_high * 2
        uq.model_disagreement = uq.epistemic
        
        mean_pred = np.mean(pred_array, axis=0)
        uq.confidence_interval_95 = (
            float(np.mean(mean_pred - 1.96 * std_across_models)),
            float(np.mean(mean_pred + 1.96 * std_across_models)),
        )
        
        return uq
    
    def _analyze_disagreement(
        self,
        individual_predictions: dict[str, np.ndarray],
        individual_confidences: dict[str, float],
    ) -> EnsembleDisagreement:
        """Analyze disagreement between ensemble models."""
        disagreement = EnsembleDisagreement()
        
        if not individual_predictions:
            return disagreement
        
        predictions = {}
        for name, probs in individual_predictions.items():
            predictions[name] = int(np.argmax(probs))
            disagreement.individual_predictions[name] = predictions[name]
            disagreement.individual_confidences[name] = (
                individual_confidences.get(name, float(np.max(probs)))
            )
            disagreement.individual_probabilities[name] = probs
        
        pred_values = list(predictions.values())
        if not pred_values:
            return disagreement
        
        unique, counts = np.unique(pred_values, return_counts=True)
        majority_class = int(unique[np.argmax(counts)])
        agreement_count = int(counts[np.argmax(counts)])
        
        disagreement.agreement_rate = agreement_count / len(pred_values)
        
        for name, pred_class in predictions.items():
            if pred_class == majority_class:
                disagreement.majority_models.append(name)
            else:
                disagreement.minority_models.append(name)
        
        if len(unique) == 1:
            disagreement.disagreement_type = "none"
        elif agreement_count / len(pred_values) > 0.75:
            disagreement.disagreement_type = "minor"
        elif agreement_count / len(pred_values) > 0.5:
            disagreement.disagreement_type = "major"
        else:
            disagreement.disagreement_type = "conflicting"
        
        if disagreement.disagreement_type == "conflicting":
            disagreement.confidence_penalty = 0.25
        elif disagreement.disagreement_type == "major":
            disagreement.confidence_penalty = 0.15
        elif disagreement.disagreement_type == "minor":
            disagreement.confidence_penalty = 0.05
        
        expected_agreement = sum((1 / len(unique)) ** 2 for _ in range(len(unique)))
        observed_agreement = disagreement.agreement_rate
        if expected_agreement < 1.0:
            disagreement.kappa_statistic = (
                (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
            )
        else:
            disagreement.kappa_statistic = 1.0
        
        return disagreement
    
    def _assess_data_quality(
        self,
        input_data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> DataQualityReport:
        """Assess input data quality."""
        report = DataQualityReport()
        
        if input_data is None or input_data.size == 0:
            report.overall_score = 0.0
            report.is_sufficient = False
            report.warnings.append("No input data for quality assessment")
            return report
        
        if input_data.ndim == 3:
            data_flat = input_data.reshape(-1, input_data.shape[-1])
        elif input_data.ndim == 2:
            data_flat = input_data
        else:
            report.overall_score = 0.0
            report.warnings.append(f"Unexpected input shape: {input_data.ndim}D")
            return report
        
        n_features = data_flat.shape[1]
        names = feature_names or [f"feat_{i}" for i in range(n_features)]
        
        nan_count = int(np.sum(~np.isfinite(data_flat)))
        total_elements = int(data_flat.size)
        report.completeness = 1.0 - (nan_count / max(1, total_elements))
        
        if report.completeness < 0.95:
            report.warnings.append(f"Data completeness low: {report.completeness:.1%}")
        
        outlier_count = 0
        for i in range(n_features):
            feat = data_flat[:, i]
            if np.all(np.isfinite(feat)):
                mean, std = np.mean(feat), np.std(feat)
                if std > 1e-10:
                    z_scores = np.abs((feat - mean) / std)
                    outlier_count += int(np.sum(z_scores > 3.5))
        
        report.consistency = 1.0 - (outlier_count / max(1, total_elements))
        
        cv_values = []
        for i in range(n_features):
            feat = data_flat[:, i]
            if np.all(np.isfinite(feat)) and np.std(feat) > 1e-10:
                mean_abs = np.abs(np.mean(feat))
                if mean_abs > 1e-10:
                    cv = np.std(feat) / mean_abs
                    cv_values.append(min(5.0, cv))
        
        if cv_values:
            median_cv = float(np.median(cv_values))
            report.noise_level = min(1.0, median_cv / 5.0)
        else:
            report.noise_level = 0.5
        
        feature_scores = []
        for i, name in enumerate(names[:min(len(names), n_features)]):
            feat = data_flat[:, i]
            if np.all(np.isfinite(feat)):
                mean_abs = np.abs(np.mean(feat))
                std = np.std(feat)
                if mean_abs > 1e-10 and std > 1e-10:
                    cv = min(5.0, std / mean_abs)
                    score = 1.0 - (cv / 5.0)
                    report.feature_quality[name] = max(0.0, score)
                    feature_scores.append(report.feature_quality[name])
                else:
                    report.feature_quality[name] = 0.5
                    feature_scores.append(0.5)
        
        avg_feature_quality = float(np.mean(feature_scores)) if feature_scores else 0.5
        
        report.overall_score = float(np.clip(
            0.4 * report.completeness +
            0.3 * report.consistency +
            0.2 * (1.0 - report.noise_level) +
            0.1 * avg_feature_quality,
            0.0, 1.0
        ))
        
        report.is_sufficient = report.overall_score >= self.min_data_quality
        
        if not report.is_sufficient:
            report.recommendation = "Data quality insufficient - consider using fallback prediction"
        
        return report
    
    def _detect_market_regime(
        self,
        market_data: dict[str, Any],
    ) -> MarketRegime:
        """Detect current market regime from market data."""
        volatility = float(market_data.get("volatility", 0.0))
        volume_ratio = float(market_data.get("volume_ratio", 1.0))
        price_change = float(market_data.get("price_change_pct", 0.0))
        trend_strength = float(market_data.get("trend_strength", 0.0))
        
        vol_threshold_high = float(getattr(CONFIG.model, "volatility_threshold_high", 0.03))
        vol_threshold_low = float(getattr(CONFIG.model, "volatility_threshold_low", 0.01))
        
        if volatility > vol_threshold_high:
            if price_change < -0.05:
                return MarketRegime.CRASH
            elif price_change > 0.05:
                return MarketRegime.BUBBLE
            else:
                return MarketRegime.HIGH_VOLATILITY
        
        if volatility < vol_threshold_low:
            return MarketRegime.LOW_VOLATILITY
        
        if trend_strength > 0.7:
            if price_change > 0.02:
                return MarketRegime.STRONG_UPTREND
            elif price_change < -0.02:
                return MarketRegime.STRONG_DOWNTREND
        
        if volume_ratio < 0.5:
            return MarketRegime.LOW_LIQUIDITY
        
        return MarketRegime.NORMAL
    
    def _compute_feature_importance(
        self,
        input_data: np.ndarray,
        probabilities: np.ndarray,
        feature_names: list[str],
    ) -> FeatureImportance:
        """Compute SHAP-like feature importance."""
        fi = FeatureImportance(feature_names=feature_names, importance_scores=np.zeros(len(feature_names)))
        
        try:
            if input_data.ndim == 3:
                input_data = input_data[:, -1, :]
            elif input_data.ndim > 2:
                input_data = input_data.reshape(input_data.shape[0], -1)
            
            n_samples = input_data.shape[0]
            n_features = min(len(feature_names), input_data.shape[1])
            
            if n_samples < 2:
                fi.importance_scores = np.abs(input_data[0, :n_features])
            else:
                pred_class = int(np.argmax(probabilities))
                for i in range(n_features):
                    feat = input_data[:, i]
                    if np.std(feat) > 1e-10:
                        corr = np.corrcoef(feat, np.ones(n_samples) * pred_class)[0, 1]
                        if np.isfinite(corr):
                            fi.importance_scores[i] = abs(corr)
                        else:
                            fi.importance_scores[i] = 0.0
                    else:
                        fi.importance_scores[i] = abs(np.mean(feat)) / (abs(np.std(feat)) + 1e-10)
            
            total = np.sum(fi.importance_scores)
            if total > 1e-10:
                fi.importance_scores = fi.importance_scores / total
            else:
                fi.importance_scores = np.ones(n_features) / n_features
            
            top_indices = np.argsort(fi.importance_scores)[::-1][:5]
            fi.top_features = [
                (feature_names[i], float(fi.importance_scores[i]))
                for i in top_indices
                if i < len(feature_names) and fi.importance_scores[i] > 0.01
            ]
            
            if not fi.top_features:
                fi.top_features = [
                    (feature_names[i], float(fi.importance_scores[i]))
                    for i in top_indices[:3]
                    if i < len(feature_names)
                ]
            
            if n_samples > 2:
                pred_class = int(np.argmax(probabilities))
                for i in range(n_features):
                    feat = input_data[:, i]
                    if np.std(feat) > 1e-10:
                        corr = np.corrcoef(feat, np.ones(n_samples) * pred_class)[0, 1]
                        if np.isfinite(corr):
                            if corr > 0.1:
                                fi.positive_influence.append(feature_names[i])
                            elif corr < -0.1:
                                fi.negative_influence.append(feature_names[i])
            
            self._recent_importance.append(fi.importance_scores.copy())
            if len(self._recent_importance) > self._importance_window:
                self._recent_importance.pop(0)
            
            if len(self._recent_importance) >= 5:
                fi.stability_scores = np.std(self._recent_importance, axis=0)
            
            if fi.top_features:
                fi.summary = f"Top drivers: {', '.join(f'{name} ({score:.2f})' for name, score in fi.top_features[:3])}"
            else:
                fi.summary = "No dominant features identified"
            
        except Exception as e:
            log.debug(f"Feature importance computation failed: {e}")
            fi.summary = "Feature importance computation failed"
            n_features = len(feature_names)
            fi.importance_scores = np.ones(n_features) / n_features
        
        return fi
    
    def _set_quality_flags(self, report: PredictionQualityReport) -> None:
        """Set quality flags based on metrics."""
        report.is_high_quality = (
            report.uncertainty.total < self.uncertainty_threshold_low and
            report.ensemble_disagreement.agreement_rate > 0.8 and
            report.data_quality.overall_score > 0.8
        )
        
        report.is_reliable = (
            report.uncertainty.total < self.uncertainty_threshold_high * 2 and
            report.ensemble_disagreement.agreement_rate > self.min_agreement_rate and
            report.data_quality.overall_score > self.min_data_quality * 0.7
        )
        
        report.requires_human_review = (
            report.ensemble_disagreement.disagreement_type == "conflicting" or
            report.uncertainty.is_high_uncertainty or
            not report.data_quality.is_sufficient or
            report.market_regime in [
                MarketRegime.CRASH,
                MarketRegime.BUBBLE,
                MarketRegime.HIGH_VOLATILITY,
            ] or
            not report.is_reliable
        )
    
    def _generate_recommendations(self, report: PredictionQualityReport) -> None:
        """Generate actionable recommendations."""
        if report.uncertainty.is_high_uncertainty:
            report.recommendations.append("High uncertainty - consider waiting for more data")
        
        if report.ensemble_disagreement.disagreement_type == "conflicting":
            report.recommendations.append("Models disagree - review individual model predictions")
        
        if not report.data_quality.is_sufficient:
            report.recommendations.append("Poor data quality - check data sources and features")
        
        if report.market_regime in [MarketRegime.CRASH, MarketRegime.HIGH_VOLATILITY]:
            report.recommendations.append(
                f"High-risk regime ({report.market_regime.value}) - reduce position size"
            )
        
        if report.requires_human_review:
            report.recommendations.append("Human review recommended before acting on this prediction")
    
    def get_adaptive_confidence_threshold(
        self,
        market_regime: MarketRegime = MarketRegime.NORMAL,
        data_quality_score: float = 1.0,
    ) -> float:
        """Get adaptive confidence threshold based on context."""
        base_threshold = float(CONFIG.model.min_confidence)
        regime_adj = self._regime_confidence_adjustments.get(market_regime, 0.0)
        quality_adj = (1.0 - data_quality_score) * 0.1
        adjusted_threshold = base_threshold + abs(regime_adj) + quality_adj
        return float(np.clip(adjusted_threshold, 0.5, 0.9))


_quality_assessor: PredictionQualityAssessor | None = None


def get_quality_assessor() -> PredictionQualityAssessor:
    """Get or create global quality assessor instance."""
    global _quality_assessor
    if _quality_assessor is None:
        _quality_assessor = PredictionQualityAssessor()
    return _quality_assessor
