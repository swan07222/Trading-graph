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
# noqa: E402 (imports must be after class definitions to avoid circular imports)
from strategies.bollinger_breakout import BollingerBreakoutStrategy
from strategies.earnings_momentum import EarningsMomentumStrategy
from strategies.gap_and_go import GapAndGoStrategy
from strategies.golden_cross import GoldenCrossStrategy
from strategies.macd_divergence import MACDDivergenceStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum_breakout import MomentumBreakoutStrategy
from strategies.rsi_oversold import RSIOversoldStrategy
from strategies.support_resistance import SupportResistanceStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.volume_profile import VolumeProfileStrategy
from strategies.vwap_reversion import VWAPReversionStrategy

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
