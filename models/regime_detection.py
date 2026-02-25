"""
Model Regime Detection and Adaptive Retraining

Addresses disadvantages:
- ML models can overfit historical data
- Market regime changes can invalidate trained models
- Requires continuous retraining
- No guarantee models adapt quickly to black swan events

Features:
- Real-time market regime detection (bull/bear/high-vol/crisis)
- Model performance monitoring with decay detection
- Adaptive retraining triggers based on regime changes
- Ensemble model weighting by regime
- Black swan detection and response
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    BULL_LOW_VOL = "bull_low_vol"  # Bull market, low volatility
    BULL_HIGH_VOL = "bull_high_vol"  # Bull market, high volatility
    BEAR_LOW_VOL = "bear_low_vol"  # Bear market, low volatility
    BEAR_HIGH_VOL = "bear_high_vol"  # Bear market, high volatility
    SIDEWAYS = "sideways"  # Range-bound market
    CRISIS = "crisis"  # Extreme volatility, crash
    TRANSITION = "transition"  # Regime change in progress


@dataclass
class RegimeMetrics:
    """Current regime metrics."""
    regime: MarketRegime
    confidence: float  # 0-1 confidence in regime classification
    trend_strength: float  # -1 to 1 (negative = bear, positive = bull)
    volatility_level: float  # Annualized volatility
    momentum: float  # Price momentum
    volume_trend: float  # Volume trend
    market_stress: float  # 0-1 stress indicator
    timestamp: datetime
    lookback_days: int = 20


@dataclass
class ModelPerformance:
    """Model performance tracking."""
    symbol: str
    model_id: str
    regime: MarketRegime
    predictions_count: int = 0
    correct_predictions: int = 0
    avg_confidence: float = 0.0
    avg_error_pct: float = 0.0
    last_prediction_time: datetime | None = None
    last_retrain_time: datetime | None = None
    train_samples: int = 0

    @property
    def accuracy(self) -> float:
        """Calculate prediction accuracy."""
        if self.predictions_count == 0:
            return 0.0
        return self.correct_predictions / self.predictions_count

    @property
    def needs_retrain(self) -> bool:
        """Check if model needs retraining."""
        # Retrain if accuracy below threshold or too few samples
        if self.train_samples < 100:
            return True
        if self.accuracy < 0.55:
            return True
        if self.avg_error_pct > 0.05:
            return True
        return False


class RegimeDetector:
    """
    Real-time market regime detection.

    Uses multiple indicators to classify market state:
    - Price trend (moving averages, momentum)
    - Volatility (realized vol, VIX-like indicators)
    - Volume patterns
    - Market breadth
    - Stress indicators
    """

    def __init__(
        self,
        lookback_days: int = 20,
        volatility_threshold: float = 0.30,
        crisis_volatility_threshold: float = 0.60,
    ) -> None:
        self.lookback_days = lookback_days
        self.volatility_threshold = volatility_threshold
        self.crisis_volatility_threshold = crisis_volatility_threshold

        self._lock = threading.RLock()
        self._price_history: dict[str, list[tuple[datetime, float]]] = {}
        self._volume_history: dict[str, list[tuple[datetime, float]]] = {}
        self._current_regime: RegimeMetrics | None = None
        self._regime_history: list[RegimeMetrics] = []

    def update_price(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        volume: float = 0.0,
    ) -> None:
        """Update price data for regime detection."""
        with self._lock:
            if symbol not in self._price_history:
                self._price_history[symbol] = []
            if symbol not in self._volume_history:
                self._volume_history[symbol] = []

            self._price_history[symbol].append((timestamp, price))
            if volume > 0:
                self._volume_history[symbol].append((timestamp, volume))

            # Keep only lookback period
            cutoff = datetime.now() - timedelta(days=self.lookback_days * 2)
            self._price_history[symbol] = [
                (ts, p) for ts, p in self._price_history[symbol]
                if ts > cutoff
            ]
            self._volume_history[symbol] = [
                (ts, v) for ts, v in self._volume_history[symbol]
                if ts > cutoff
            ]

    def detect_regime(self, symbol: str = "market") -> RegimeMetrics:
        """
        Detect current market regime.

        Args:
            symbol: Stock code or "market" for broad market

        Returns:
            RegimeMetrics with current regime classification
        """
        with self._lock:
            prices = self._price_history.get(symbol, [])
            volumes = self._volume_history.get(symbol, [])

        if len(prices) < 5:
            return RegimeMetrics(
                regime=MarketRegime.TRANSITION,
                confidence=0.0,
                trend_strength=0.0,
                volatility_level=0.0,
                momentum=0.0,
                volume_trend=0.0,
                market_stress=0.0,
                timestamp=datetime.now(),
                lookback_days=self.lookback_days,
            )

        # Extract price series
        timestamps, price_series = zip(*prices, strict=False)
        price_array = np.array(price_series)

        # Calculate trend (bull/bear)
        trend_strength = self._calculate_trend(price_array)

        # Calculate volatility
        volatility = self._calculate_volatility(price_array)

        # Calculate momentum
        momentum = self._calculate_momentum(price_array)

        # Calculate volume trend
        volume_trend = self._calculate_volume_trend(volumes)

        # Calculate market stress
        stress = self._calculate_stress(price_array, volatility)

        # Classify regime
        regime, confidence = self._classify_regime(
            trend_strength, volatility, momentum, stress
        )

        metrics = RegimeMetrics(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_level=volatility,
            momentum=momentum,
            volume_trend=volume_trend,
            market_stress=stress,
            timestamp=datetime.now(),
            lookback_days=self.lookback_days,
        )

        with self._lock:
            self._current_regime = metrics
            self._regime_history.append(metrics)

            # Keep last 1000 regime readings
            if len(self._regime_history) > 1000:
                self._regime_history = self._regime_history[-1000:]

        return metrics

    def _calculate_trend(self, prices: np.ndarray) -> float:
        """Calculate trend strength (-1 to 1)."""
        if len(prices) < 10:
            return 0.0

        # Simple trend: (current - past) / past
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)

        if long_ma == 0:
            return 0.0

        trend = (short_ma - long_ma) / long_ma
        return np.clip(trend * 10, -1.0, 1.0)  # Scale to -1 to 1

    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate annualized volatility."""
        if len(prices) < 5:
            return 0.0

        returns = np.diff(prices) / prices[:-1]
        daily_vol = np.std(returns)

        # Annualize (assuming daily data)
        annualized_vol = daily_vol * np.sqrt(252)

        return annualized_vol

    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate price momentum (-1 to 1)."""
        if len(prices) < 10:
            return 0.0

        # Momentum: rate of change over lookback
        lookback = min(10, len(prices) - 1)
        if prices[-lookback - 1] == 0:
            return 0.0

        roc = (prices[-1] - prices[-lookback - 1]) / prices[-lookback - 1]
        return np.clip(roc * 5, -1.0, 1.0)  # Scale to -1 to 1

    def _calculate_volume_trend(self, volumes: list) -> float:
        """Calculate volume trend (-1 to 1)."""
        if len(volumes) < 10:
            return 0.0

        vol_series = np.array([v for _, v in volumes])
        recent_vol = np.mean(vol_series[-5:])
        older_vol = np.mean(vol_series[-20:-5]) if len(vol_series) >= 20 else recent_vol

        if older_vol == 0:
            return 0.0

        trend = (recent_vol - older_vol) / older_vol
        return np.clip(trend * 2, -1.0, 1.0)

    def _calculate_stress(self, prices: np.ndarray, volatility: float) -> float:
        """Calculate market stress indicator (0-1)."""
        if len(prices) < 5:
            return 0.0

        # Stress based on recent large moves
        returns = np.diff(prices) / prices[:-1]
        large_moves = np.sum(np.abs(returns) > 0.03)  # >3% moves
        stress_from_moves = min(large_moves / len(returns), 1.0)

        # Stress from volatility
        stress_from_vol = min(volatility / self.crisis_volatility_threshold, 1.0)

        return 0.5 * stress_from_moves + 0.5 * stress_from_vol

    def _classify_regime(
        self,
        trend: float,
        volatility: float,
        momentum: float,
        stress: float,
    ) -> tuple[MarketRegime, float]:
        """Classify market regime based on metrics."""
        # Crisis detection (highest priority)
        if stress > 0.7 or volatility > self.crisis_volatility_threshold:
            return MarketRegime.CRISIS, min(stress, 1.0)

        # Sideways market
        if abs(trend) < 0.1:
            return MarketRegime.SIDEWAYS, 1.0 - abs(trend) * 10

        # Bull/Bear classification
        is_bull = trend > 0
        is_high_vol = volatility > self.volatility_threshold

        if is_bull:
            regime = MarketRegime.BULL_HIGH_VOL if is_high_vol else MarketRegime.BULL_LOW_VOL
        else:
            regime = MarketRegime.BEAR_HIGH_VOL if is_high_vol else MarketRegime.BEAR_LOW_VOL

        # Confidence based on trend strength and momentum agreement
        trend_confidence = min(abs(trend) * 5, 1.0)
        momentum_agreement = 1.0 if (trend > 0) == (momentum > 0) else 0.5
        confidence = 0.6 * trend_confidence + 0.4 * momentum_agreement

        return regime, confidence

    def is_regime_change(
        self,
        current_regime: MarketRegime,
        lookback_changes: int = 5,
    ) -> bool:
        """Detect if regime is changing."""
        with self._lock:
            if len(self._regime_history) <= lookback_changes:
                return False

            recent_regimes = [
                m.regime for m in self._regime_history[-lookback_changes:]
            ]
            previous_regime = self._regime_history[-lookback_changes - 1].regime

            # Regime change if majority of recent readings differ from before
            different_count = sum(1 for r in recent_regimes if r != previous_regime)
            return different_count >= lookback_changes // 2

    def get_regime_history(
        self,
        since: datetime = None,
        limit: int = 100,
    ) -> list[RegimeMetrics]:
        """Get regime history."""
        with self._lock:
            history = self._regime_history.copy()

        if since:
            history = [m for m in history if m.timestamp >= since]

        return history[-limit:]


class AdaptiveRetrainer:
    """
    Adaptive model retraining system.

    Triggers retraining based on:
    - Model performance decay
    - Regime changes
    - Time-based schedules
    - Black swan events
    """

    def __init__(
        self,
        regime_detector: RegimeDetector,
        min_samples_for_retrain: int = 100,
        performance_decay_threshold: float = 0.10,
        retrain_cooldown_hours: int = 1,
    ) -> None:
        self.regime_detector = regime_detector
        self.min_samples_for_retrain = min_samples_for_retrain
        self.performance_decay_threshold = performance_decay_threshold
        self.retrain_cooldown_hours = retrain_cooldown_hours

        self._lock = threading.RLock()
        self._model_performance: dict[str, ModelPerformance] = {}
        self._last_retrain: dict[str, datetime] = {}
        self._baseline_accuracies: dict[str, float] = {}
        self._retrain_triggers: list[dict] = []

    def record_prediction(
        self,
        model_id: str,
        symbol: str,
        predicted_signal: str,
        actual_signal: str,
        confidence: float,
        error_pct: float,
        regime: MarketRegime,
    ) -> None:
        """Record prediction for performance tracking."""
        with self._lock:
            key = f"{model_id}:{symbol}"

            if key not in self._model_performance:
                self._model_performance[key] = ModelPerformance(
                    symbol=symbol,
                    model_id=model_id,
                    regime=regime,
                )

            perf = self._model_performance[key]
            perf.predictions_count += 1

            if predicted_signal == actual_signal:
                perf.correct_predictions += 1

            # Update running averages
            n = perf.predictions_count
            perf.avg_confidence = (
                (perf.avg_confidence * (n - 1) + confidence) / n
            )
            perf.avg_error_pct = (
                (perf.avg_error_pct * (n - 1) + error_pct) / n
            )
            perf.last_prediction_time = datetime.now()
            perf.regime = regime

            # Set baseline accuracy after enough samples
            if perf.predictions_count == 50 and key not in self._baseline_accuracies:
                self._baseline_accuracies[key] = perf.accuracy

    def record_model_load(
        self,
        model_id: str,
        symbol: str,
        train_samples: int,
    ) -> None:
        """Record model loading."""
        with self._lock:
            key = f"{model_id}:{symbol}"

            if key not in self._model_performance:
                self._model_performance[key] = ModelPerformance(
                    symbol=symbol,
                    model_id=model_id,
                    regime=MarketRegime.TRANSITION,
                )

            self._model_performance[key].train_samples = train_samples
            self._model_performance[key].last_retrain_time = datetime.now()

    def should_retrain(
        self,
        model_id: str,
        symbol: str,
        current_regime: MarketRegime,
    ) -> tuple[bool, str]:
        """
        Check if model should be retrained.

        Returns:
            (should_retrain, reason)
        """
        key = f"{model_id}:{symbol}"

        with self._lock:
            perf = self._model_performance.get(key)

            # Check cooldown
            last_retrain = self._last_retrain.get(key)
            if last_retrain:
                cooldown_end = last_retrain + timedelta(
                    hours=self.retrain_cooldown_hours
                )
                if datetime.now() < cooldown_end:
                    return False, "In cooldown period"

            # New model with insufficient samples
            if perf and perf.train_samples < self.min_samples_for_retrain:
                return True, f"Insufficient training samples: {perf.train_samples}"

            # Performance decay
            if perf and key in self._baseline_accuracies:
                baseline = self._baseline_accuracies[key]
                decay = baseline - perf.accuracy

                if decay > self.performance_decay_threshold:
                    return True, f"Performance decay: {decay:.1%}"

            # Regime change detection
            if perf and perf.regime != current_regime:
                if current_regime == MarketRegime.CRISIS:
                    return True, "Crisis regime detected"
                if self.regime_detector.is_regime_change(current_regime):
                    return True, f"Regime change: {perf.regime.value} -> {current_regime.value}"

        return False, "No retrain needed"

    def record_retrain(self, model_id: str, symbol: str) -> None:
        """Record retraining event."""
        key = f"{model_id}:{symbol}"

        with self._lock:
            self._last_retrain[key] = datetime.now()

            if key in self._model_performance:
                self._model_performance[key].last_retrain_time = datetime.now()

            trigger = {
                "timestamp": datetime.now().isoformat(),
                "model_id": model_id,
                "symbol": symbol,
                "reason": "Retrained",
            }
            self._retrain_triggers.append(trigger)

            # Keep last 100 triggers
            if len(self._retrain_triggers) > 100:
                self._retrain_triggers = self._retrain_triggers[-100:]

    def trigger_black_swan_retrain(self, symbol: str = None) -> None:
        """Trigger immediate retraining for all models on black swan event."""
        log.critical("🚨 BLACK SWAN EVENT - Triggering emergency retraining")

        with self._lock:
            trigger = {
                "timestamp": datetime.now().isoformat(),
                "model_id": "all",
                "symbol": symbol or "market",
                "reason": "Black swan event",
                "emergency": True,
            }
            self._retrain_triggers.append(trigger)

            # Reset cooldowns
            self._last_retrain.clear()

    def get_model_performance(self, model_id: str = None, symbol: str = None) -> dict:
        """Get model performance statistics."""
        with self._lock:
            performances = self._model_performance.copy()

        if model_id and symbol:
            key = f"{model_id}:{symbol}"
            if key in performances:
                return {key: performances[key]}
            return {}

        return {
            k: {
                "symbol": v.symbol,
                "model_id": v.model_id,
                "regime": v.regime.value,
                "accuracy": round(v.accuracy, 3),
                "predictions": v.predictions_count,
                "avg_confidence": round(v.avg_confidence, 3),
                "avg_error_pct": round(v.avg_error_pct, 4),
                "train_samples": v.train_samples,
                "needs_retrain": v.needs_retrain,
            }
            for k, v in performances.items()
        }

    def get_retrain_history(self, limit: int = 50) -> list[dict]:
        """Get retraining history."""
        with self._lock:
            return self._retrain_triggers[-limit:]


@dataclass
class EnsembleWeight:
    """Model weight in ensemble by regime."""
    model_id: str
    weight_by_regime: dict[MarketRegime, float]
    default_weight: float = 1.0


class RegimeAwareEnsemble:
    """
    Ensemble model weighting based on market regime.

    Different models perform better in different regimes.
    This dynamically weights models based on current regime.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model_weights: dict[str, EnsembleWeight] = {}
        self._regime_detector = RegimeDetector()
        self._current_regime: MarketRegime | None = None

    def register_model(
        self,
        model_id: str,
        weights_by_regime: dict[MarketRegime, float],
        default_weight: float = 1.0,
    ) -> None:
        """Register model with regime-specific weights."""
        with self._lock:
            self._model_weights[model_id] = EnsembleWeight(
                model_id=model_id,
                weight_by_regime=weights_by_regime,
                default_weight=default_weight,
            )

    def update_regime(self, symbol: str = "market") -> MarketRegime:
        """Update current regime detection."""
        metrics = self._regime_detector.detect_regime(symbol)
        with self._lock:
            self._current_regime = metrics.regime
        return metrics.regime

    def get_model_weights(self) -> dict[str, float]:
        """Get current model weights based on regime."""
        with self._lock:
            if self._current_regime is None:
                # Default equal weights
                return {
                    model_id: w.default_weight
                    for model_id, w in self._model_weights.items()
                }

            weights = {}
            for model_id, ensemble_weight in self._model_weights.items():
                weight = ensemble_weight.weight_by_regime.get(
                    self._current_regime,
                    ensemble_weight.default_weight,
                )
                weights[model_id] = weight

            # Normalize weights
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

            return weights

    def get_regime_info(self) -> dict:
        """Get current regime information."""
        with self._lock:
            return {
                "regime": self._current_regime.value if self._current_regime else None,
                "model_count": len(self._model_weights),
                "weights": self.get_model_weights(),
            }
