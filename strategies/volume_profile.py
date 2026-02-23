"""Volume Profile Strategy - Analyzes volume at price levels."""
from __future__ import annotations
from typing import Any
import numpy as np
from core.types import OrderSide
from strategies import BaseStrategy, Signal, SignalStrength, register_strategy

@register_strategy
class VolumeProfileStrategy(BaseStrategy):
    """Volume Profile analysis for support/resistance."""
    name = "volume_profile"
    description = "Volume-weighted support and resistance"
    version = "1.0.0"
    params = {"lookback": 30, "bins": 20}
    min_confidence = 0.5

    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        bars = data.get("bars", [])
        if len(bars) < self.params["lookback"]:
            return None
        recent = bars[-self.params["lookback"] :]
        closes = np.array([b["close"] for b in recent])
        volumes = np.array([b["volume"] for b in recent])
        current_close = closes[-1]
        # Calculate VWAP
        vwap = np.sum(closes * volumes) / np.sum(volumes)
        # Volume-weighted price deviation
        deviation = (current_close - vwap) / vwap
        if deviation < -0.03:  # 3% below VWAP
            confidence = min(1.0, abs(deviation) * 15)
            if confidence >= self.min_confidence:
                return Signal(self.name, data.get("symbol", ""), OrderSide.BUY,
                    SignalStrength.MODERATE, confidence, entry_price=current_close,
                    target_price=vwap, stop_loss=current_close * 0.97, metadata={"vwap": vwap})
        elif deviation > 0.03:  # 3% above VWAP
            confidence = min(1.0, abs(deviation) * 15)
            if confidence >= self.min_confidence:
                return Signal(self.name, data.get("symbol", ""), OrderSide.SELL,
                    SignalStrength.MODERATE, confidence, entry_price=current_close,
                    target_price=vwap, stop_loss=current_close * 1.03, metadata={"vwap": vwap})
        return None
