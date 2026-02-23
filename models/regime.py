"""Market Regime Detection Module.

Detects market regimes to improve prediction accuracy through:
1. Adaptive thresholds based on volatility
2. Model routing based on regime
3. Confidence adjustment based on regime reliability

Regimes:
- BULL_STRONG: Strong uptrend, low volatility
- BULL_WEAK: Weak uptrend, moderate volatility  
- BEAR_STRONG: Strong downtrend, high volatility
- BEAR_WEAK: Weak downtrend, moderate volatility
- RANGE_UP: Range-bound with upward bias
- RANGE_DOWN: Range-bound with downward bias
- CHOPPY: No clear direction, high noise
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class RegimeType(Enum):
    BULL_STRONG = "BULL_STRONG"
    BULL_WEAK = "BULL_WEAK"
    BEAR_STRONG = "BEAR_STRONG"
    BEAR_WEAK = "BEAR_WEAK"
    RANGE_UP = "RANGE_UP"
    RANGE_DOWN = "RANGE_DOWN"
    CHOPPY = "CHOPPY"


@dataclass
class RegimeResult:
    """Regime detection result."""
    regime: RegimeType
    confidence: float  # 0.0 to 1.0
    trend_strength: float  # -1 to 1
    volatility_level: str  # LOW, MEDIUM, HIGH
    recommended_threshold: float  # Adaptive threshold for this regime
    
    # Reliability metrics for prediction accuracy
    historical_accuracy: float = 0.0  # Historical prediction accuracy in this regime
    avg_return: float = 0.0  # Average return when following signals in this regime


class MarketRegimeDetector:
    """Market regime detection using multiple indicators.
    
    Uses:
    - Moving average relationships for trend
    - ATR for volatility
    - ADX for trend strength
    - Price position for range detection
    """
    
    def __init__(
        self,
        trend_period: int = 20,
        volatility_period: int = 14,
        adx_period: int = 14,
    ) -> None:
        self.trend_period = trend_period
        self.volatility_period = volatility_period
        self.adx_period = adx_period
        
        # Regime-specific thresholds (calibrated for CN A-shares)
        self.regime_thresholds = {
            RegimeType.BULL_STRONG: {"threshold": 0.55, "accuracy_target": 0.72},
            RegimeType.BULL_WEAK: {"threshold": 0.58, "accuracy_target": 0.65},
            RegimeType.BEAR_STRONG: {"threshold": 0.60, "accuracy_target": 0.70},
            RegimeType.BEAR_WEAK: {"threshold": 0.62, "accuracy_target": 0.62},
            RegimeType.RANGE_UP: {"threshold": 0.52, "accuracy_target": 0.58},
            RegimeType.RANGE_DOWN: {"threshold": 0.55, "accuracy_target": 0.55},
            RegimeType.CHOPPY: {"threshold": 0.65, "accuracy_target": 0.50},  # Avoid trading
        }
    
    def detect(self, df: pd.DataFrame) -> RegimeResult:
        """Detect current market regime.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            RegimeResult with regime and confidence
        """
        if len(df) < self.trend_period * 2:
            return self._default_regime()
        
        # Calculate indicators
        df = df.copy()
        ma_fast = df["close"].rolling(self.trend_period // 2).mean()
        ma_slow = df["close"].rolling(self.trend_period).mean()
        ma_long = df["close"].rolling(self.trend_period * 2).mean()
        
        # Trend direction and strength
        trend_score = self._calc_trend_score(df, ma_fast, ma_slow, ma_long)
        
        # Volatility
        vol_level = self._calc_volatility_level(df)
        
        # Trend strength (ADX-like)
        trend_strength = self._calc_trend_strength(df)
        
        # Range detection
        is_ranging = self._is_ranging_market(df, ma_slow)
        
        # Determine regime
        regime = self._classify_regime(
            trend_score, vol_level, trend_strength, is_ranging
        )
        
        # Calculate confidence
        confidence = self._calc_confidence(
            regime, trend_score, vol_level, trend_strength
        )
        
        # Get recommended threshold
        threshold_config = self.regime_thresholds.get(
            regime, {"threshold": 0.60}
        )
        
        return RegimeResult(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_score,
            volatility_level=vol_level,
            recommended_threshold=threshold_config["threshold"],
            historical_accuracy=threshold_config["accuracy_target"],
        )
    
    def _calc_trend_score(
        self,
        df: pd.DataFrame,
        ma_fast: pd.Series,
        ma_slow: pd.Series,
        ma_long: pd.Series,
    ) -> float:
        """Calculate trend score from -1 (strong bear) to 1 (strong bull)."""
        close = df["close"].iloc[-1]
        
        # Position relative to MAs
        if ma_slow.iloc[-1] > 0:
            pos_vs_slow = (close - ma_slow.iloc[-1]) / ma_slow.iloc[-1]
        else:
            pos_vs_slow = 0
        
        if ma_long.iloc[-1] > 0:
            pos_vs_long = (close - ma_long.iloc[-1]) / ma_long.iloc[-1]
        else:
            pos_vs_long = 0
        
        # MA alignment
        ma_bullish = 1.0 if ma_fast.iloc[-1] > ma_slow.iloc[-1] > ma_long.iloc[-1] else 0
        ma_bearish = 1.0 if ma_fast.iloc[-1] < ma_slow.iloc[-1] < ma_long.iloc[-1] else 0
        
        # Combine signals
        score = (
            0.4 * np.tanh(pos_vs_slow * 100) +
            0.3 * np.tanh(pos_vs_long * 100) +
            0.3 * (ma_bullish - ma_bearish)
        )
        
        return float(np.clip(score, -1, 1))
    
    def _calc_volatility_level(self, df: pd.DataFrame) -> str:
        """Calculate volatility level."""
        # ATR-based volatility
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(self.volatility_period).mean()
        
        # Normalized volatility (ATR / price)
        vol_ratio = (atr / df["close"]).iloc[-1] * 100
        
        # Calibrated for CN A-shares (10% daily limit)
        if vol_ratio < 1.5:
            return "LOW"
        elif vol_ratio < 3.0:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _calc_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength (ADX-like)."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # Directional movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # True range
        high_low = high - low
        high_close_prev = (high - close.shift(1)).abs()
        low_close_prev = (low - close.shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Smoothed averages
        atr = tr.rolling(self.adx_period).mean()
        plus_di = 100 * plus_dm.rolling(self.adx_period).mean() / (atr + 1e-8)
        minus_di = 100 * minus_dm.rolling(self.adx_period).mean() / (atr + 1e-8)
        
        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(self.adx_period).mean()
        
        return float(adx.iloc[-1] / 100)  # Normalize to 0-1
    
    def _is_ranging_market(
        self,
        df: pd.DataFrame,
        ma: pd.Series,
    ) -> bool:
        """Detect if market is ranging."""
        close = df["close"]
        
        # Check how often price crosses MA
        above_ma = (close > ma).astype(int)
        crosses = above_ma.diff().abs().sum()
        
        # Calculate price range
        highest = close.rolling(self.trend_period).max()
        lowest = close.rolling(self.trend_period).min()
        range_pct = (highest - lowest) / lowest
        
        # Ranging if: many crosses + moderate range
        is_ranging = (
            crosses >= self.trend_period * 0.4 and
            range_pct.iloc[-1] < 0.15
        )
        
        return bool(is_ranging)
    
    def _classify_regime(
        self,
        trend_score: float,
        vol_level: str,
        trend_strength: float,
        is_ranging: bool,
    ) -> RegimeType:
        """Classify regime based on indicators."""
        # Ranging market
        if is_ranging:
            if trend_score > 0.1:
                return RegimeType.RANGE_UP
            elif trend_score < -0.1:
                return RegimeType.RANGE_DOWN
            else:
                return RegimeType.CHOPPY
        
        # Trending market
        if abs(trend_score) > 0.5 and trend_strength > 0.4:
            if trend_score > 0:
                return RegimeType.BULL_STRONG
            else:
                return RegimeType.BEAR_STRONG
        elif abs(trend_score) > 0.2:
            if trend_score > 0:
                return RegimeType.BULL_WEAK
            else:
                return RegimeType.BEAR_WEAK
        else:
            return RegimeType.CHOPPY
    
    def _calc_confidence(
        self,
        regime: RegimeType,
        trend_score: float,
        vol_level: str,
        trend_strength: float,
    ) -> float:
        """Calculate regime detection confidence."""
        base_confidence = 0.5
        
        # Trend strength adds confidence
        base_confidence += trend_strength * 0.3
        
        # Extreme scores add confidence
        base_confidence += abs(trend_score) * 0.2
        
        # Low volatility adds confidence (clearer signals)
        if vol_level == "LOW":
            base_confidence += 0.1
        elif vol_level == "HIGH":
            base_confidence -= 0.1
        
        # Choppy regime has lower confidence
        if regime == RegimeType.CHOPPY:
            base_confidence *= 0.7
        
        return float(np.clip(base_confidence, 0.0, 1.0))
    
    def _default_regime(self) -> RegimeResult:
        """Return default regime when insufficient data."""
        return RegimeResult(
            regime=RegimeType.CHOPPY,
            confidence=0.3,
            trend_strength=0.0,
            volatility_level="MEDIUM",
            recommended_threshold=0.65,
            historical_accuracy=0.50,
        )
    
    def get_adaptive_thresholds(self, regime: RegimeType) -> dict[str, float]:
        """Get adaptive thresholds for label creation based on regime.
        
        Returns thresholds that account for:
        - Typical price movements in this regime
        - Required confidence for trading
        - Expected accuracy
        """
        base = self.regime_thresholds.get(
            regime,
            {"threshold": 0.60, "accuracy_target": 0.60}
        )
        
        return {
            "up_threshold": base["threshold"] * 0.01,  # 0.6% for strong bull
            "down_threshold": base["threshold"] * 0.01,
            "min_confidence": base["threshold"],
            "accuracy_target": base["accuracy_target"],
        }


def detect_regime(df: pd.DataFrame) -> RegimeResult:
    """Convenience function for regime detection."""
    detector = MarketRegimeDetector()
    return detector.detect(df)
