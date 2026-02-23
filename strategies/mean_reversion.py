"""
Mean Reversion Strategy.

Trades based on the principle that prices tend to revert to their mean.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from core.types import OrderSide
from strategies import BaseStrategy, Signal, SignalStrength, register_strategy


@register_strategy
class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy.

    Buys when price is significantly below moving average,
    sells when significantly above.

    Parameters:
        ma_period: Moving average period (default: 20)
        zscore_threshold: Z-score threshold for entry (default: 2.0)
    """

    name = "mean_reversion"
    description = "Mean reversion using Z-score"
    version = "1.0.0"

    params = {
        "ma_period": 20,
        "zscore_threshold": 2.0,
    }

    min_confidence = 0.55

    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        """Generate mean reversion signal."""
        bars = data.get("bars", [])
        if len(bars) < self.params["ma_period"] + 10:
            return None

        closes = np.array([b["close"] for b in bars[-self.params["ma_period"] * 2 :]])
        current_close = closes[-1]

        # Calculate moving average and standard deviation
        ma = np.mean(closes[-self.params["ma_period"] :])
        std = np.std(closes[-self.params["ma_period"] :])

        if std == 0:
            return None

        # Calculate Z-score
        zscore = (current_close - ma) / std

        # Check for oversold (buy signal)
        if zscore < -self.params["zscore_threshold"]:
            confidence = min(1.0, abs(zscore) / 4.0)
            if confidence >= self.min_confidence:
                strength = self._calculate_strength(abs(zscore))
                target = ma  # Target is mean reversion
                stop = current_close * 0.95

                return Signal(
                    strategy_name=self.name,
                    symbol=data.get("symbol", ""),
                    side=OrderSide.BUY,
                    strength=strength,
                    confidence=confidence,
                    entry_price=current_close,
                    target_price=target,
                    stop_loss=stop,
                    metadata={"zscore": zscore, "ma": ma, "std": std},
                )

        # Check for overbought (sell signal)
        elif zscore > self.params["zscore_threshold"]:
            confidence = min(1.0, abs(zscore) / 4.0)
            if confidence >= self.min_confidence:
                strength = self._calculate_strength(abs(zscore))
                target = ma
                stop = current_close * 1.05

                return Signal(
                    strategy_name=self.name,
                    symbol=data.get("symbol", ""),
                    side=OrderSide.SELL,
                    strength=strength,
                    confidence=confidence,
                    entry_price=current_close,
                    target_price=target,
                    stop_loss=stop,
                    metadata={"zscore": zscore, "ma": ma, "std": std},
                )

        return None

    def _calculate_strength(self, zscore_abs: float) -> SignalStrength:
        """Calculate signal strength based on Z-score."""
        if zscore_abs > 3.5:
            return SignalStrength.VERY_STRONG
        elif zscore_abs > 2.5:
            return SignalStrength.STRONG
        elif zscore_abs > 2.0:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK
