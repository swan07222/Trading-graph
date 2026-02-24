"""
Confidence Calibration and Uncertainty Quantification

Addresses disadvantages:
- Only 70%+ confidence threshold - meaning 30%+ could be wrong
- Relies on historical patterns; no guarantee future performance matches
- Model predictions are probabilistic, not deterministic

Features:
- Calibrated confidence scores (confidence = actual accuracy)
- Uncertainty bands (prediction intervals)
- Ensemble disagreement as uncertainty metric
- Dynamic confidence thresholds by regime
- Prediction quality tracking
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level classification."""
    VERY_LOW = "very_low"  # 0-40%
    LOW = "low"  # 40-55%
    MEDIUM = "medium"  # 55-70%
    HIGH = "high"  # 70-85%
    VERY_HIGH = "very_high"  # 85-100%


@dataclass
class CalibratedPrediction:
    """Prediction with calibrated confidence and uncertainty."""
    symbol: str
    timestamp: datetime
    signal: str
    raw_confidence: float  # Model's raw confidence
    calibrated_confidence: float  # Calibrated confidence
    uncertainty: float  # Prediction uncertainty (0-1)
    prediction_interval_lower: float  # Lower bound of price prediction
    prediction_interval_upper: float  # Upper bound of price prediction
    confidence_level: ConfidenceLevel
    ensemble_disagreement: float  # How much ensemble models disagree
    regime: str
    is_reliable: bool  # Whether prediction meets reliability threshold
    notes: str = ""

    @property
    def prediction_range(self) -> float:
        """Get prediction range (uncertainty in price terms)."""
        return self.prediction_interval_upper - self.prediction_interval_lower

    @property
    def uncertainty_pct(self) -> float:
        """Get uncertainty as percentage of prediction center."""
        center = (self.prediction_interval_lower + self.prediction_interval_upper) / 2
        if center > 0:
            return (self.prediction_range / 2) / center
        return 1.0


@dataclass
class ConfidenceBucket:
    """Tracking predictions in a confidence bucket."""
    min_confidence: float
    max_confidence: float
    predictions: list[CalibratedPrediction] = field(default_factory=list)
    correct: int = 0
    total: int = 0

    @property
    def empirical_accuracy(self) -> float:
        """Calculate actual accuracy for this bucket."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    @property
    def calibration_error(self) -> float:
        """Calculate calibration error."""
        expected = (self.min_confidence + self.max_confidence) / 2
        return abs(self.empirical_accuracy - expected)


class ConfidenceCalibrator:
    """
    Calibrate model confidence scores to actual accuracy.

    Uses isotonic regression / histogram binning to map
    raw confidence scores to calibrated probabilities.
    """

    def __init__(
        self,
        n_buckets: int = 10,
        min_samples_per_bucket: int = 30,
    ) -> None:
        self.n_buckets = n_buckets
        self.min_samples_per_bucket = min_samples_per_bucket

        self._lock = threading.RLock()
        self._buckets: list[ConfidenceBucket] = self._create_buckets()
        self._all_predictions: list[CalibratedPrediction] = []
        self._calibration_map: dict[str, float] = {}  # symbol -> calibration factor

    def _create_buckets(self) -> list[ConfidenceBucket]:
        """Create confidence buckets."""
        edges = np.linspace(0.0, 1.0, self.n_buckets + 1)
        buckets = []
        for i in range(self.n_buckets):
            buckets.append(
                ConfidenceBucket(
                    min_confidence=edges[i],
                    max_confidence=edges[i + 1],
                )
            )
        return buckets

    def record_prediction(
        self,
        prediction: CalibratedPrediction,
    ) -> None:
        """Record prediction for calibration tracking."""
        with self._lock:
            self._all_predictions.append(prediction)

            # Find appropriate bucket
            for bucket in self._buckets:
                if bucket.min_confidence <= prediction.raw_confidence < bucket.max_confidence:
                    bucket.total += 1
                    break

            # Keep last 10000 predictions
            if len(self._all_predictions) > 10000:
                self._all_predictions = self._all_predictions[-10000:]

    def mark_outcome(
        self,
        prediction: CalibratedPrediction,
        was_correct: bool,
    ) -> None:
        """Mark prediction outcome for calibration."""
        with self._lock:
            # Find and update bucket
            for bucket in self._buckets:
                if bucket.min_confidence <= prediction.raw_confidence < bucket.max_confidence:
                    if was_correct:
                        bucket.correct += 1
                    break

            # Update calibration map
            self._update_calibration_map()

    def _update_calibration_map(self) -> None:
        """Update calibration mapping based on bucket accuracies."""
        for bucket in self._buckets:
            if bucket.total >= self.min_samples_per_bucket:
                # Map midpoint confidence to empirical accuracy
                midpoint = (bucket.min_confidence + bucket.max_confidence) / 2
                self._calibration_map[midpoint] = bucket.empirical_accuracy

    def calibrate(self, raw_confidence: float, symbol: str = None) -> float:
        """
        Calibrate raw confidence score.

        Args:
            raw_confidence: Model's raw confidence (0-1)
            symbol: Optional symbol for symbol-specific calibration

        Returns:
            Calibrated confidence (0-1)
        """
        with self._lock:
            # Symbol-specific calibration if available
            if symbol and symbol in self._calibration_map:
                base_calibration = self._calibration_map[symbol]
            else:
                # Find calibration from nearest bucket
                if not self._calibration_map:
                    return raw_confidence  # No calibration data yet

                # Linear interpolation
                sorted_points = sorted(self._calibration_map.keys())
                if raw_confidence <= sorted_points[0]:
                    return self._calibration_map[sorted_points[0]]
                if raw_confidence >= sorted_points[-1]:
                    return self._calibration_map[sorted_points[-1]]

                # Find bracketing points
                for i in range(len(sorted_points) - 1):
                    low = sorted_points[i]
                    high = sorted_points[i + 1]
                    if low <= raw_confidence < high:
                        # Linear interpolation
                        t = (raw_confidence - low) / (high - low)
                        cal_low = self._calibration_map[low]
                        cal_high = self._calibration_map[high]
                        return cal_low + t * (cal_high - cal_low)

            return raw_confidence

    def get_calibration_report(self) -> dict:
        """Get calibration quality report."""
        with self._lock:
            buckets_data = []
            for bucket in self._buckets:
                if bucket.total > 0:
                    buckets_data.append({
                        "range": f"{bucket.min_confidence:.0%}-{bucket.max_confidence:.0%}",
                        "empirical_accuracy": round(bucket.empirical_accuracy, 3),
                        "expected_accuracy": round(
                            (bucket.min_confidence + bucket.max_confidence) / 2, 3
                        ),
                        "calibration_error": round(bucket.calibration_error, 3),
                        "count": bucket.total,
                    })

            total_predictions = sum(b.total for b in self._buckets)
            correct_predictions = sum(b.correct for b in self._buckets)

            overall_accuracy = (
                correct_predictions / total_predictions if total_predictions > 0 else 0.0
            )

            # Calculate ECE (Expected Calibration Error)
            ece = 0.0
            if total_predictions > 0:
                for bucket in self._buckets:
                    if bucket.total > 0:
                        weight = bucket.total / total_predictions
                        ece += weight * bucket.calibration_error

            return {
                "total_predictions": total_predictions,
                "overall_accuracy": round(overall_accuracy, 3),
                "expected_calibration_error": round(ece, 3),
                "buckets": buckets_data,
            }


class UncertaintyEstimator:
    """
    Estimate prediction uncertainty using multiple methods.

    Methods:
    - Ensemble disagreement
    - Historical error distribution
    - Regime-based uncertainty
    - Input data quality
    """

    def __init__(
        self,
        confidence_level: float = 0.90,
    ) -> None:
        self.confidence_level = confidence_level

        self._lock = threading.RLock()
        self._historical_errors: dict[str, list[float]] = {}
        self._ensemble_disagreements: list[float] = []

    def record_ensemble_predictions(
        self,
        symbol: str,
        predictions: list[float],
        actual: float,
    ) -> None:
        """Record ensemble predictions for uncertainty estimation."""
        with self._lock:
            # Calculate ensemble disagreement (std of predictions)
            if len(predictions) > 1:
                disagreement = np.std(predictions)
                self._ensemble_disagreements.append(disagreement)

                # Keep last 1000
                if len(self._ensemble_disagreements) > 1000:
                    self._ensemble_disagreements = self._ensemble_disagreements[-1000:]

            # Calculate prediction error
            if predictions:
                avg_prediction = np.mean(predictions)
                if avg_prediction > 0:
                    error = abs(actual - avg_prediction) / avg_prediction

                    if symbol not in self._historical_errors:
                        self._historical_errors[symbol] = []
                    self._historical_errors[symbol].append(error)

                    # Keep last 100 per symbol
                    if len(self._historical_errors[symbol]) > 100:
                        self._historical_errors[symbol] = self._historical_errors[symbol][-100:]

    def estimate_uncertainty(
        self,
        symbol: str,
        ensemble_predictions: list[float],
        current_price: float,
        regime: str = None,
    ) -> tuple[float, float, float]:
        """
        Estimate prediction uncertainty.

        Args:
            symbol: Stock code
            ensemble_predictions: List of predictions from ensemble
            current_price: Current price
            regime: Current market regime

        Returns:
            (uncertainty_0_1, lower_bound, upper_bound)
        """
        if not ensemble_predictions or current_price <= 0:
            return 1.0, 0.0, 0.0

        with self._lock:
            # 1. Ensemble disagreement
            disagreement = np.std(ensemble_predictions)
            mean_pred = np.mean(ensemble_predictions)

            if mean_pred > 0:
                disagreement_pct = disagreement / mean_pred
            else:
                disagreement_pct = 1.0

            # 2. Historical error for symbol
            historical_error = 0.05  # Default 5%
            if symbol in self._historical_errors and self._historical_errors[symbol]:
                errors = self._historical_errors[symbol]
                historical_error = np.percentile(errors, self.confidence_level * 100)

            # 3. Combine uncertainties (conservative approach)
            uncertainty = max(disagreement_pct, historical_error)

            # 4. Regime adjustment (higher uncertainty in crisis)
            regime_multiplier = 1.0
            if regime:
                if "crisis" in regime.lower() or "bear_high" in regime.lower():
                    regime_multiplier = 1.5
                elif "bull_low" in regime.lower():
                    regime_multiplier = 0.8

            uncertainty *= regime_multiplier
            uncertainty = np.clip(uncertainty, 0.01, 0.50)  # Cap at 50%

            # 5. Calculate prediction interval
            z_score = 1.645  # 90% confidence
            interval_half_width = z_score * uncertainty * current_price

            lower_bound = current_price - interval_half_width
            upper_bound = current_price + interval_half_width

            return uncertainty, lower_bound, upper_bound

    def get_disagreement_stats(self) -> dict:
        """Get ensemble disagreement statistics."""
        with self._lock:
            if not self._ensemble_disagreements:
                return {"count": 0}

            return {
                "count": len(self._ensemble_disagreements),
                "mean": round(np.mean(self._ensemble_disagreements), 4),
                "std": round(np.std(self._ensemble_disagreements), 4),
                "median": round(np.median(self._ensemble_disagreements), 4),
                "p90": round(np.percentile(self._ensemble_disagreements, 90), 4),
            }


class DynamicThresholdManager:
    """
    Dynamically adjust confidence thresholds based on conditions.

    Higher thresholds in:
    - High volatility regimes
    - Low model accuracy periods
    - High uncertainty predictions

    Lower thresholds in:
    - Stable bull markets
    - High model accuracy periods
    - Low uncertainty predictions
    """

    def __init__(
        self,
        base_threshold: float = 0.70,
        min_threshold: float = 0.55,
        max_threshold: float = 0.90,
    ) -> None:
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self._lock = threading.RLock()
        self._regime_thresholds: dict[str, float] = {}
        self._symbol_thresholds: dict[str, float] = {}
        self._accuracy_history: list[tuple[datetime, float]] = []

    def update_regime_threshold(
        self,
        regime: str,
        volatility: float,
        recent_accuracy: float,
    ) -> None:
        """Update threshold for specific regime."""
        with self._lock:
            # Start with base threshold
            threshold = self.base_threshold

            # Adjust for volatility (higher vol = higher threshold)
            vol_adjustment = min(volatility * 0.5, 0.10)
            threshold += vol_adjustment

            # Adjust for accuracy (lower accuracy = higher threshold)
            if recent_accuracy < 0.55:
                threshold += 0.10
            elif recent_accuracy < 0.60:
                threshold += 0.05
            elif recent_accuracy > 0.70:
                threshold -= 0.05

            # Clamp to bounds
            threshold = np.clip(threshold, self.min_threshold, self.max_threshold)

            self._regime_thresholds[regime] = threshold

    def update_symbol_threshold(
        self,
        symbol: str,
        recent_accuracy: float,
        uncertainty: float,
    ) -> None:
        """Update threshold for specific symbol."""
        with self._lock:
            threshold = self.base_threshold

            # Adjust for symbol-specific accuracy
            if recent_accuracy < 0.50:
                threshold += 0.15
            elif recent_accuracy < 0.60:
                threshold += 0.05
            elif recent_accuracy > 0.70:
                threshold -= 0.05

            # Adjust for uncertainty
            threshold += uncertainty * 0.2

            # Clamp to bounds
            threshold = np.clip(threshold, self.min_threshold, self.max_threshold)

            self._symbol_thresholds[symbol] = threshold

    def get_threshold(
        self,
        symbol: str = None,
        regime: str = None,
        uncertainty: float = None,
    ) -> float:
        """Get dynamic confidence threshold."""
        with self._lock:
            threshold = self.base_threshold

            # Apply regime adjustment
            if regime and regime in self._regime_thresholds:
                threshold = self._regime_thresholds[regime]

            # Apply symbol adjustment
            if symbol and symbol in self._symbol_thresholds:
                symbol_threshold = self._symbol_thresholds[symbol]
                # Average with base if we have regime threshold
                if regime and regime in self._regime_thresholds:
                    threshold = (threshold + symbol_threshold) / 2
                else:
                    threshold = symbol_threshold

            # Apply uncertainty adjustment
            if uncertainty is not None:
                threshold += uncertainty * 0.1
                threshold = np.clip(threshold, self.min_threshold, self.max_threshold)

            return threshold

    def should_trade(
        self,
        confidence: float,
        symbol: str = None,
        regime: str = None,
        uncertainty: float = None,
    ) -> tuple[bool, str]:
        """
        Check if prediction confidence meets threshold for trading.

        Returns:
            (should_trade, reason)
        """
        threshold = self.get_threshold(symbol, regime, uncertainty)

        if confidence >= threshold:
            return True, f"Confidence {confidence:.0%} >= threshold {threshold:.0%}"

        gap = threshold - confidence
        return False, f"Confidence {confidence:.0%} below threshold {threshold:.0%} (gap: {gap:.0%})"


def create_calibrated_prediction(
    symbol: str,
    signal: str,
    raw_confidence: float,
    current_price: float,
    ensemble_predictions: list[float],
    regime: str,
    calibrator: ConfidenceCalibrator = None,
    uncertainty_estimator: UncertaintyEstimator = None,
) -> CalibratedPrediction:
    """Create a calibrated prediction with uncertainty bands."""
    # Calibrate confidence
    if calibrator:
        calibrated_confidence = calibrator.calibrate(raw_confidence, symbol)
    else:
        calibrated_confidence = raw_confidence

    # Estimate uncertainty
    if uncertainty_estimator and ensemble_predictions:
        uncertainty, lower, upper = uncertainty_estimator.estimate_uncertainty(
            symbol, ensemble_predictions, current_price, regime
        )
    else:
        # Default uncertainty based on confidence
        uncertainty = 1.0 - calibrated_confidence
        uncertainty = np.clip(uncertainty, 0.05, 0.30)
        lower = current_price * (1 - uncertainty)
        upper = current_price * (1 + uncertainty)

    # Determine confidence level
    if calibrated_confidence < 0.40:
        level = ConfidenceLevel.VERY_LOW
    elif calibrated_confidence < 0.55:
        level = ConfidenceLevel.LOW
    elif calibrated_confidence < 0.70:
        level = ConfidenceLevel.MEDIUM
    elif calibrated_confidence < 0.85:
        level = ConfidenceLevel.HIGH
    else:
        level = ConfidenceLevel.VERY_HIGH

    # Calculate ensemble disagreement
    if ensemble_predictions and len(ensemble_predictions) > 1:
        disagreement = np.std(ensemble_predictions) / np.mean(ensemble_predictions)
    else:
        disagreement = 0.0

    # Determine reliability
    is_reliable = (
        calibrated_confidence >= 0.70 and
        uncertainty <= 0.20 and
        disagreement <= 0.10
    )

    return CalibratedPrediction(
        symbol=symbol,
        timestamp=datetime.now(),
        signal=signal,
        raw_confidence=raw_confidence,
        calibrated_confidence=calibrated_confidence,
        uncertainty=uncertainty,
        prediction_interval_lower=lower,
        prediction_interval_upper=upper,
        confidence_level=level,
        ensemble_disagreement=disagreement,
        regime=regime,
        is_reliable=is_reliable,
    )
