"""VWAP Reversion Strategy - Mean reversion to VWAP."""
from __future__ import annotations

from typing import Any

import numpy as np

from core.types import OrderSide
from strategies import BaseStrategy, Signal, SignalStrength, register_strategy


@register_strategy
class VWAPReversionStrategy(BaseStrategy):
    """VWAP mean reversion strategy."""
    name = "vwap_reversion"
    description = "Trade reversion to VWAP"
    version = "1.0.0"
    params = {"lookback": 20, "std_threshold": 2.0}
    min_confidence = 0.55

    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        bars = data.get("bars", [])
        if len(bars) < self.params["lookback"]:
            return None
        closes = np.array([b["close"] for b in bars[-self.params["lookback"] :]])
        volumes = np.array([b["volume"] for b in bars[-self.params["lookback"] :]])
        # Calculate VWAP and standard deviation
        vwap = np.sum(closes * volumes) / np.sum(volumes)
        std = np.std(closes)
        current_close = closes[-1]
        # Calculate bands
        upper_band = vwap + self.params["std_threshold"] * std
        lower_band = vwap - self.params["std_threshold"] * std
        # Reversion from upper band
        if current_close > upper_band:
            confidence = min(1.0, 0.5 + (current_close - upper_band) / vwap * 20)
            return Signal(self.name, data.get("symbol", ""), OrderSide.SELL,
                SignalStrength.MODERATE, confidence, entry_price=current_close,
                target_price=vwap, stop_loss=upper_band * 1.02,
                metadata={"vwap": vwap, "upper": upper_band, "lower": lower_band})
        # Reversion from lower band
        elif current_close < lower_band:
            confidence = min(1.0, 0.5 + (lower_band - current_close) / vwap * 20)
            return Signal(self.name, data.get("symbol", ""), OrderSide.BUY,
                SignalStrength.MODERATE, confidence, entry_price=current_close,
                target_price=vwap, stop_loss=lower_band * 0.98,
                metadata={"vwap": vwap, "upper": upper_band, "lower": lower_band})
        return None
