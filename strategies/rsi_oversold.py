"""RSI Oversold/Overbought Strategy."""
from __future__ import annotations

from typing import Any

import numpy as np

from core.types import OrderSide
from strategies import BaseStrategy, Signal, SignalStrength, register_strategy


@register_strategy
class RSIOversoldStrategy(BaseStrategy):
    """RSI-based mean reversion strategy."""
    name = "rsi_oversold"
    description = "Trade RSI oversold/overbought conditions"
    version = "1.0.0"
    params = {"period": 14, "oversold": 30, "overbought": 70}
    min_confidence = 0.5

    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        bars = data.get("bars", [])
        if len(bars) < self.params["period"] + 10:
            return None
        closes = np.array([b["close"] for b in bars[-self.params["period"] * 2 :]])
        # Calculate RSI
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-self.params["period"] :])
        avg_loss = np.mean(losses[-self.params["period"] :])
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        current_close = closes[-1]
        # Oversold condition
        if rsi < self.params["oversold"]:
            confidence = 0.5 + (self.params["oversold"] - rsi) / 100
            if confidence >= self.min_confidence:
                return Signal(self.name, data.get("symbol", ""), OrderSide.BUY,
                    SignalStrength.MODERATE, min(1.0, confidence), entry_price=current_close,
                    target_price=current_close * 1.05, stop_loss=current_close * 0.97,
                    metadata={"rsi": rsi})
        # Overbought condition
        elif rsi > self.params["overbought"]:
            confidence = 0.5 + (rsi - self.params["overbought"]) / 100
            if confidence >= self.min_confidence:
                return Signal(self.name, data.get("symbol", ""), OrderSide.SELL,
                    SignalStrength.MODERATE, min(1.0, confidence), entry_price=current_close,
                    target_price=current_close * 0.95, stop_loss=current_close * 1.03,
                    metadata={"rsi": rsi})
        return None
