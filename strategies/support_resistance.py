"""Support/Resistance Strategy - Identifies key price levels."""
from __future__ import annotations

from typing import Any

import numpy as np

from core.types import OrderSide
from strategies import BaseStrategy, Signal, SignalStrength, register_strategy


@register_strategy
class SupportResistanceStrategy(BaseStrategy):
    """Support and Resistance level trading."""
    name = "support_resistance"
    description = "Trade bounces off support/resistance levels"
    version = "1.0.0"
    params = {"lookback": 50, "tolerance": 0.02}
    min_confidence = 0.55

    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        bars = data.get("bars", [])
        if len(bars) < self.params["lookback"]:
            return None
        highs = np.array([b["high"] for b in bars[-self.params["lookback"] :]])
        lows = np.array([b["low"] for b in bars[-self.params["lookback"] :]])
        closes = np.array([b["close"] for b in bars])
        current_close = closes[-1]
        # Find significant levels
        resistance = np.percentile(highs, 80)
        support = np.percentile(lows, 20)
        # Bounce off support
        if abs(current_close - support) / support < self.params["tolerance"] and current_close > support:
            confidence = 0.6 + (1 - (current_close - support) / support) * 0.3
            if confidence >= self.min_confidence:
                return Signal(self.name, data.get("symbol", ""), OrderSide.BUY,
                    SignalStrength.MODERATE, min(1.0, confidence), entry_price=current_close,
                    target_price=resistance, stop_loss=support * 0.98)
        # Rejection at resistance
        elif abs(current_close - resistance) / resistance < self.params["tolerance"] and current_close < resistance:
            confidence = 0.6 + (1 - (resistance - current_close) / resistance) * 0.3
            if confidence >= self.min_confidence:
                return Signal(self.name, data.get("symbol", ""), OrderSide.SELL,
                    SignalStrength.MODERATE, min(1.0, confidence), entry_price=current_close,
                    target_price=support, stop_loss=resistance * 1.02)
        return None
