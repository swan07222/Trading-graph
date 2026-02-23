"""Bollinger Bands Breakout Strategy."""
from __future__ import annotations
from typing import Any
import numpy as np
from core.types import OrderSide
from strategies import BaseStrategy, Signal, SignalStrength, register_strategy

@register_strategy
class BollingerBreakoutStrategy(BaseStrategy):
    """Bollinger Bands breakout with volume confirmation."""
    name = "bollinger_breakout"
    description = "Trade breakouts from Bollinger Bands"
    version = "1.0.0"
    params = {"period": 20, "std_dev": 2.0}
    min_confidence = 0.55

    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        bars = data.get("bars", [])
        if len(bars) < self.params["period"] + 5:
            return None
        closes = np.array([b["close"] for b in bars[-self.params["period"] * 2 :]])
        volumes = np.array([b["volume"] for b in bars[-self.params["period"] * 2 :]])
        ma = np.mean(closes[-self.params["period"] :])
        std = np.std(closes[-self.params["period"] :])
        upper = ma + self.params["std_dev"] * std
        lower = ma - self.params["std_dev"] * std
        current_close = closes[-1]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-self.params["period"] :])
        # Breakout above
        if current_close > upper and current_volume > avg_volume * 1.5:
            confidence = min(1.0, 0.5 + (current_close - upper) / upper * 10)
            return Signal(self.name, data.get("symbol", ""), OrderSide.BUY,
                SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MODERATE,
                confidence, entry_price=current_close, target_price=upper * 1.05,
                stop_loss=ma, metadata={"upper": upper, "lower": lower, "ma": ma})
        # Breakout below
        elif current_close < lower and current_volume > avg_volume * 1.5:
            confidence = min(1.0, 0.5 + (lower - current_close) / lower * 10)
            return Signal(self.name, data.get("symbol", ""), OrderSide.SELL,
                SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MODERATE,
                confidence, entry_price=current_close, target_price=lower * 0.95,
                stop_loss=ma, metadata={"upper": upper, "lower": lower, "ma": ma})
        return None
