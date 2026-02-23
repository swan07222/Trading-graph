"""Golden Cross Strategy - Long-term trend change detection."""
from __future__ import annotations

from typing import Any

import numpy as np

from core.types import OrderSide
from strategies import BaseStrategy, Signal, SignalStrength, register_strategy


@register_strategy
class GoldenCrossStrategy(BaseStrategy):
    """Golden Cross / Death Cross long-term signals."""
    name = "golden_cross"
    description = "50/200 day moving average crossover"
    version = "1.0.0"
    params = {"short_ma": 50, "long_ma": 200}
    min_confidence = 0.65

    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        bars = data.get("bars", [])
        if len(bars) < self.params["long_ma"] + 5:
            return None
        closes = np.array([b["close"] for b in bars[-self.params["long_ma"] * 2 :]])
        short_ma = np.mean(closes[-self.params["short_ma"] :])
        long_ma = np.mean(closes[-self.params["long_ma"] :])
        prev_short = np.mean(closes[-self.params["short_ma"] - 10 : -10])
        prev_long = np.mean(closes[-self.params["long_ma"] - 10 : -10])
        current_close = closes[-1]
        # Golden Cross
        if prev_short <= prev_long and short_ma > long_ma:
            return Signal(self.name, data.get("symbol", ""), OrderSide.BUY,
                SignalStrength.VERY_STRONG, 0.8, entry_price=current_close,
                target_price=current_close * 1.2, stop_loss=long_ma * 0.95,
                metadata={"short_ma": short_ma, "long_ma": long_ma})
        # Death Cross
        elif prev_short >= prev_long and short_ma < long_ma:
            return Signal(self.name, data.get("symbol", ""), OrderSide.SELL,
                SignalStrength.VERY_STRONG, 0.8, entry_price=current_close,
                target_price=current_close * 0.8, stop_loss=long_ma * 1.05,
                metadata={"short_ma": short_ma, "long_ma": long_ma})
        return None
