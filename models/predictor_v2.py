"""
Streamlined Predictor with Regime Detection

Key improvements for 70%+ accuracy:
1. Regime-adaptive thresholds
2. Ensemble confidence weighting
3. Cost-aware predictions
4. Quality filters
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core_v2 import Prediction, Signal
from models.regime import MarketRegimeDetector, RegimeType


@dataclass
class PredictionConfig:
    """Configuration for prediction."""
    min_confidence: float = 0.55
    use_regime: bool = True
    cost_aware: bool = True
    commission_rate: float = 0.0003
    slippage_bps: float = 2.0


class StreamlinedPredictor:
    """
    Streamlined prediction with regime detection.
    
    Focus on quality over quantity:
    - Only predict when confidence is high
    - Adjust thresholds based on regime
    - Account for trading costs
    """
    
    def __init__(
        self,
        model_dir: Path | None = None,
        config: PredictionConfig | None = None,
    ):
        self.model_dir = model_dir or Path("./models_saved")
        self.config = config or PredictionConfig()
        self.regime_detector = MarketRegimeDetector()
        
        # Cache for regime results
        self._regime_cache: dict[str, tuple[float, Any]] = {}
        self._regime_cache_ttl = 300.0  # 5 minutes
    
    def predict(
        self,
        stock_code: str,
        bars: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> Prediction | None:
        """
        Generate prediction with regime awareness.
        
        Args:
            stock_code: Stock code (6 digits)
            bars: OHLCV bar data
            features: Pre-computed features (optional)
            
        Returns:
            Prediction object or None if confidence too low
        """
        if bars is None or len(bars) < 60:
            return None
        
        # Detect regime
        regime_result = self._detect_regime_cached(stock_code, bars)
        
        # Get adaptive thresholds
        if self.config.use_regime:
            thresholds = self.regime_detector.get_adaptive_thresholds(
                regime_result.regime
            )
            min_confidence = thresholds["min_confidence"]
        else:
            min_confidence = self.config.min_confidence
        
        # Generate raw prediction from ensemble
        raw_prediction = self._generate_raw_prediction(
            stock_code, bars, features
        )
        
        if raw_prediction is None:
            return None
        
        # Apply regime adjustments
        adjusted_prediction = self._apply_regime_adjustments(
            raw_prediction, regime_result
        )
        
        # Apply cost filter
        if self.config.cost_aware:
            if not self._passes_cost_filter(adjusted_prediction):
                return None
        
        # Final confidence check
        if adjusted_prediction.confidence < min_confidence:
            return None
        
        return adjusted_prediction
    
    def _detect_regime_cached(
        self,
        stock_code: str,
        bars: pd.DataFrame,
    ) -> Any:
        """Detect regime with caching."""
        import time
        
        now = time.time()
        
        # Check cache
        if stock_code in self._regime_cache:
            cache_time, cache_result = self._regime_cache[stock_code]
            if now - cache_time < self._regime_cache_ttl:
                return cache_result
        
        # Detect regime
        result = self.regime_detector.detect(bars)
        
        # Cache result
        self._regime_cache[stock_code] = (now, result)
        
        return result
    
    def _generate_raw_prediction(
        self,
        stock_code: str,
        bars: pd.DataFrame,
        features: pd.DataFrame | None,
    ) -> Prediction | None:
        """
        Generate raw prediction from model ensemble.
        
        This is a simplified version - in production, this would
        load and run the actual model ensemble.
        """
        # For demonstration, use feature-based heuristic
        # In production, replace with actual model inference
        
        if features is None:
            features = self._compute_features(bars)
        
        if features is None or len(features) < 10:
            return None
        
        # Get latest features
        latest = features.iloc[-1]
        
        # Simple ensemble of signals
        signals = []
        weights = []
        
        # Momentum signal
        if "returns" in latest:
            mom_signal = np.sign(latest["returns"])
            mom_strength = min(abs(latest["returns"]) / 3, 1.0)
            signals.append(mom_signal)
            weights.append(mom_strength * 0.3)
        
        # Mean reversion signal
        if "price_to_ma20" in latest:
            mr_signal = -np.sign(latest["price_to_ma20"])
            mr_strength = min(abs(latest["price_to_ma20"]) / 5, 1.0)
            signals.append(mr_signal)
            weights.append(mr_strength * 0.25)
        
        # RSI signal
        if "rsi_14" in latest:
            rsi = latest["rsi_14"]
            if rsi < 0.3:
                rsi_signal = 1.0
            elif rsi > 0.7:
                rsi_signal = -1.0
            else:
                rsi_signal = 0.0
            rsi_strength = max(0, 0.5 - abs(rsi - 0.5))
            signals.append(rsi_signal)
            weights.append(rsi_strength * 0.25)
        
        # Volatility signal
        if "volatility_20" in latest:
            vol = latest["volatility_20"]
            if vol > 5:
                vol_signal = -0.5  # High vol = reduce confidence
            else:
                vol_signal = 0.3
            signals.append(vol_signal)
            weights.append(0.2)
        
        if not signals or sum(weights) < 0.3:
            return None
        
        # Weighted ensemble
        signals = np.array(signals)
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-8)
        
        ensemble_score = float(np.dot(signals, weights))
        
        # Convert to signal and confidence
        if ensemble_score > 0.3:
            signal = Signal.BUY
            confidence = 0.5 + ensemble_score * 0.5
        elif ensemble_score < -0.3:
            signal = Signal.SELL
            confidence = 0.5 + abs(ensemble_score) * 0.5
        else:
            signal = Signal.HOLD
            confidence = 0.5 + abs(ensemble_score) * 0.3
        
        confidence = min(0.95, max(0.5, confidence))
        
        current_price = float(bars["close"].iloc[-1])
        
        return Prediction(
            stock_code=stock_code,
            stock_name="",
            signal=signal,
            confidence=confidence,
            current_price=current_price,
            target_price=current_price * (1 + ensemble_score * 0.05),
            stop_loss=current_price * (1 - abs(ensemble_score) * 0.03),
        )
    
    def _compute_features(self, bars: pd.DataFrame) -> pd.DataFrame | None:
        """Compute technical features."""
        try:
            from data.features import FeatureEngine
            engine = FeatureEngine()
            return engine.create_features(bars)
        except Exception:
            return None
    
    def _apply_regime_adjustments(
        self,
        prediction: Prediction,
        regime_result: Any,
    ) -> Prediction:
        """Adjust prediction based on regime."""
        # Adjust confidence based on regime reliability
        regime_accuracy = regime_result.historical_accuracy
        
        # Boost confidence in high-accuracy regimes
        if regime_accuracy > 0.65:
            prediction.confidence = min(0.95, prediction.confidence * 1.1)
        elif regime_accuracy < 0.55:
            prediction.confidence = max(0.5, prediction.confidence * 0.9)
        
        # Set regime
        prediction.regime = regime_result.regime.value
        
        # Adjust target/stop based on volatility
        if regime_result.volatility_level == "HIGH":
            # Wider targets/stops in high volatility
            prediction.target_price = prediction.current_price * 1.08
            prediction.stop_loss = prediction.current_price * 0.94
        elif regime_result.volatility_level == "LOW":
            # Tighter targets/stops in low volatility
            prediction.target_price = prediction.current_price * 1.04
            prediction.stop_loss = prediction.current_price * 0.97
        
        return prediction
    
    def _passes_cost_filter(self, prediction: Prediction) -> bool:
        """Check if prediction passes cost-aware filter."""
        if prediction.signal == Signal.HOLD:
            return True
        
        # Calculate expected return
        if prediction.signal == Signal.BUY:
            expected_return = (
                prediction.target_price - prediction.current_price
            ) / prediction.current_price
        else:
            expected_return = (
                prediction.current_price - prediction.target_price
            ) / prediction.current_price
        
        # Calculate costs
        total_cost = (
            self.config.commission_rate * 2 +  # Entry + exit
            self.config.slippage_bps / 10000 +
            0.001  # Stamp duty (CN market)
        )
        
        # Net expected return
        net_return = expected_return - total_cost
        
        # Require positive net return with margin
        min_net_return = 0.005  # 0.5% minimum
        
        return net_return >= min_net_return
    
    def get_prediction_quality(self, stock_code: str) -> dict[str, Any]:
        """Get prediction quality metrics for a stock."""
        regime_result = self._regime_cache.get(stock_code, (None, None))[1]
        
        if regime_result is None:
            return {"quality": "UNKNOWN", "confidence": 0.0}
        
        return {
            "regime": regime_result.regime.value,
            "regime_confidence": regime_result.confidence,
            "historical_accuracy": regime_result.historical_accuracy,
            "recommended_threshold": regime_result.recommended_threshold,
            "quality": (
                "HIGH" if regime_result.historical_accuracy > 0.65
                else "MEDIUM" if regime_result.historical_accuracy > 0.55
                else "LOW"
            ),
        }


def create_predictor(
    model_dir: str | None = None,
    min_confidence: float = 0.55,
    use_regime: bool = True,
) -> StreamlinedPredictor:
    """Factory function to create predictor."""
    config = PredictionConfig(
        min_confidence=min_confidence,
        use_regime=use_regime,
        cost_aware=True,
    )
    
    return StreamlinedPredictor(
        model_dir=Path(model_dir) if model_dir else None,
        config=config,
    )
