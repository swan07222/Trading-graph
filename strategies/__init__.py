"""Trading strategies for Trading Graph.

This package contains various trading strategies that can be enabled
and configured for auto-trading.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from core.types import OrderSide


class SignalStrength(Enum):
    """Signal strength levels."""

    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


@dataclass
class Signal:
    """Trading signal from a strategy."""

    strategy_name: str
    symbol: str
    side: OrderSide | None
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    entry_price: float | None = None
    target_price: float | None = None
    stop_loss: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_valid(self) -> bool:
        """Check if signal is valid."""
        return self.side is not None and self.confidence > 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "side": self.side.value if self.side else None,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "timestamp": self.timestamp.isoformat(),
            **self.metadata,
        }


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.
    
    FIX REGIME AWARENESS: Added regime-aware confidence adjustment
    to improve signal quality across different market conditions.

    All strategies must implement the generate_signal method.

    Example:
        class MyStrategy(BaseStrategy):
            def generate_signal(self, data: dict) -> Signal | None:
                # Your logic here
                return Signal(...)
    """

    name: str = "base"
    description: str = "Base strategy"
    version: str = "1.0.0"

    # Strategy parameters (override in subclass)
    params: dict[str, Any] = {}

    # Minimum confidence threshold (0.0 to 1.0)
    min_confidence: float = 0.5
    
    # Regime-specific confidence adjustments
    _regime_multipliers: dict[str, float] = {
        "bull_low_vol": 1.05,      # Slight boost in stable bull markets
        "bull_high_vol": 0.95,     # Reduce confidence in volatile bull
        "bear_low_vol": 0.90,      # Cautious in stable bear
        "bear_high_vol": 0.80,     # Very cautious in volatile bear
        "crisis": 0.70,            # Minimum confidence in crisis
        "sideways": 0.85,          # Reduce confidence in choppy markets
    }

    @abstractmethod
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        """Generate trading signal from market data.

        Args:
            data: Market data including:
                - bars: List of OHLCV bars
                - indicators: Pre-computed technical indicators
                - position: Current position (if any)
                - portfolio: Portfolio state

        Returns:
            Signal object if signal generated, None otherwise
        """
        pass
    
    def _detect_market_regime(self, data: dict[str, Any]) -> str:
        """Detect current market regime from data.
        
        Uses price action and volatility to classify regime:
        - bull_low_vol: Uptrend with low volatility
        - bull_high_vol: Uptrend with high volatility
        - bear_low_vol: Downtrend with low volatility
        - bear_high_vol: Downtrend with high volatility
        - crisis: Extreme volatility/drawdown
        - sideways: No clear trend
        
        Returns:
            Regime string
        """
        bars = data.get("bars", [])
        if len(bars) < 60:
            return "sideways"  # Default with insufficient data
        
        closes = [b["close"] for b in bars]
        
        # Calculate trend (60-day return)
        if len(closes) >= 60:
            trend_return = (closes[-1] - closes[-60]) / closes[-60]
        else:
            trend_return = (closes[-1] - closes[0]) / closes[0] if closes else 0.0
        
        # Calculate volatility (20-day std of returns)
        if len(closes) >= 21:
            returns = [
                (closes[i] - closes[i-1]) / closes[i-1]
                for i in range(len(closes)-20, len(closes))
            ]
            volatility = float(np.std(returns)) if returns else 0.0
        else:
            volatility = 0.02  # Default
        
        # Classify regime
        if volatility > 0.04:  # >4% daily vol = extreme
            return "crisis"
        elif volatility > 0.025:  # >2.5% = high vol
            if trend_return > 0.05:
                return "bull_high_vol"
            elif trend_return < -0.05:
                return "bear_high_vol"
            else:
                return "sideways"
        else:  # Low vol
            if trend_return > 0.03:
                return "bull_low_vol"
            elif trend_return < -0.03:
                return "bear_low_vol"
            else:
                return "sideways"
    
    def _apply_regime_adjustment(
        self,
        confidence: float,
        regime: str = None,
    ) -> float:
        """Adjust confidence based on market regime.
        
        FIX REGIME AWARENESS: Different regimes require different
        confidence thresholds for actionable signals.
        
        Args:
            confidence: Raw signal confidence
            regime: Market regime (auto-detected if None)
            
        Returns:
            Regime-adjusted confidence
        """
        if regime is None:
            regime = "sideways"
        
        multiplier = self._regime_multipliers.get(regime, 1.0)
        adjusted = confidence * multiplier
        
        # Ensure confidence stays in valid range
        return float(np.clip(adjusted, 0.0, 1.0))

    def validate_signal(self, signal: Signal) -> bool:
        """Validate a generated signal.

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid
        """
        if not signal.is_valid:
            return False
        if signal.confidence < self.min_confidence:
            return False
        return True

    def get_info(self) -> dict[str, Any]:
        """Get strategy information."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "params": self.params,
            "min_confidence": self.min_confidence,
        }


# Registry for available strategies
STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register_strategy(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    """Decorator to register a strategy."""
    if hasattr(cls, "name") and cls.name:
        STRATEGY_REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> BaseStrategy | None:
    """Get strategy instance by name."""
    strategy_class = STRATEGY_REGISTRY.get(name)
    if strategy_class:
        return strategy_class()
    return None


def list_strategies() -> list[str]:
    """List all registered strategies."""
    return list(STRATEGY_REGISTRY.keys())


# Import all strategies to register them (after class definitions)
from strategies.bollinger_breakout import BollingerBreakoutStrategy  # noqa: E402
from strategies.earnings_momentum import EarningsMomentumStrategy  # noqa: E402
from strategies.gap_and_go import GapAndGoStrategy  # noqa: E402
from strategies.golden_cross import GoldenCrossStrategy  # noqa: E402
from strategies.macd_divergence import MACDDivergenceStrategy  # noqa: E402
from strategies.mean_reversion import MeanReversionStrategy  # noqa: E402
from strategies.momentum_breakout import MomentumBreakoutStrategy  # noqa: E402
from strategies.rsi_oversold import RSIOversoldStrategy  # noqa: E402
from strategies.support_resistance import SupportResistanceStrategy  # noqa: E402
from strategies.trend_following import TrendFollowingStrategy  # noqa: E402
from strategies.volume_profile import VolumeProfileStrategy  # noqa: E402
from strategies.vwap_reversion import VWAPReversionStrategy  # noqa: E402

__all__ = [
    "BaseStrategy",
    "Signal",
    "SignalStrength",
    "STRATEGY_REGISTRY",
    "register_strategy",
    "get_strategy",
    "list_strategies",
    "MomentumBreakoutStrategy",
    "MeanReversionStrategy",
    "TrendFollowingStrategy",
    "VolumeProfileStrategy",
    "SupportResistanceStrategy",
    "MACDDivergenceStrategy",
    "BollingerBreakoutStrategy",
    "RSIOversoldStrategy",
    "GoldenCrossStrategy",
    "GapAndGoStrategy",
    "VWAPReversionStrategy",
    "EarningsMomentumStrategy",
]
