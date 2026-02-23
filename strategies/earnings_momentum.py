"""Earnings Momentum Strategy - Post-earnings announcement drift."""
from __future__ import annotations

from typing import Any

import numpy as np

from core.types import OrderSide
from strategies import BaseStrategy, Signal, SignalStrength, register_strategy


@register_strategy
class EarningsMomentumStrategy(BaseStrategy):
    """Post-earnings announcement drift strategy."""
    name = "earnings_momentum"
    description = "Trade post-earnings momentum"
    version = "1.0.0"
    params = {"earnings_lookback": 5, "momentum_period": 10}
    min_confidence = 0.6

    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        bars = data.get("bars", [])
        if len(bars) < self.params["earnings_lookback"] + self.params["momentum_period"]:
            return None
        # Check for earnings event in metadata
        earnings_date = data.get("metadata", {}).get("last_earnings_date")
        if not earnings_date:
            return None
        closes = np.array([b["close"] for b in bars[-(self.params["earnings_lookback"] + self.params["momentum_period"]) :]])
        # Calculate post-earnings momentum
        pre_earnings = closes[0]  # Simplified: assume first bar is pre-earnings
        post_earnings = closes[-1]
        momentum = (post_earnings - pre_earnings) / pre_earnings
        # Strong positive momentum
        if momentum > 0.05:  # 5% move
            confidence = min(1.0, 0.5 + momentum * 5)
            if confidence >= self.min_confidence:
                return Signal(self.name, data.get("symbol", ""), OrderSide.BUY,
                    SignalStrength.STRONG, confidence, entry_price=post_earnings,
                    target_price=post_earnings * 1.1, stop_loss=post_earnings * 0.95,
                    metadata={"momentum": momentum, "earnings_date": earnings_date})
        # Strong negative momentum
        elif momentum < -0.05:
            confidence = min(1.0, 0.5 + abs(momentum) * 5)
            if confidence >= self.min_confidence:
                return Signal(self.name, data.get("symbol", ""), OrderSide.SELL,
                    SignalStrength.STRONG, confidence, entry_price=post_earnings,
                    target_price=post_earnings * 0.9, stop_loss=post_earnings * 1.05,
                    metadata={"momentum": momentum, "earnings_date": earnings_date})
        return None
