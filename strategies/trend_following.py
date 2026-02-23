"""Trend Following Strategy - Follows established trends using moving averages."""
from __future__ import annotations
from typing import Any
import numpy as np
from core.types import OrderSide
from strategies import BaseStrategy, Signal, SignalStrength, register_strategy

@register_strategy
class TrendFollowingStrategy(BaseStrategy):
    """Trend Following using moving average crossovers."""
    name = "trend_following"
    description = "Trend following with dual moving averages"
    version = "1.0.0"
    params = {"fast_ma": 10, "slow_ma": 30}
    min_confidence = 0.55

    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        """Generate trend following signal."""
        # FIX #17: Add input validation
        if not isinstance(data, dict):
            return None
        
        bars = data.get("bars", [])
        if not isinstance(bars, list) or len(bars) < self.params["slow_ma"] + 5:
            return None
        
        try:
            closes = np.array([b["close"] for b in bars[-self.params["slow_ma"] * 2 :]])
            if len(closes) < self.params["slow_ma"]:
                return None
        except (KeyError, TypeError, ValueError):
            return None
        
        current_close = closes[-1]
        if current_close <= 0:
            return None
        
        fast_ma = np.mean(closes[-self.params["fast_ma"] :])
        slow_ma = np.mean(closes[-self.params["slow_ma"] :])
        prev_fast = np.mean(closes[-self.params["fast_ma"] - 5 : -5])
        prev_slow = np.mean(closes[-self.params["slow_ma"] - 5 : -5])
        
        # Golden cross (fast crosses above slow)
        if prev_fast <= prev_slow and fast_ma > slow_ma:
            confidence = min(1.0, 0.5 + (fast_ma - slow_ma) / slow_ma * 10)
            if confidence >= self.min_confidence:
                return Signal(
                    self.name, data.get("symbol", ""), OrderSide.BUY,
                    SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MODERATE,
                    confidence, entry_price=current_close, target_price=current_close * 1.1,
                    stop_loss=current_close * 0.95, metadata={"fast_ma": fast_ma, "slow_ma": slow_ma}
                )
        # Death cross (fast crosses below slow)
        elif prev_fast >= prev_slow and fast_ma < slow_ma:
            confidence = min(1.0, 0.5 + (slow_ma - fast_ma) / slow_ma * 10)
            if confidence >= self.min_confidence:
                return Signal(
                    self.name, data.get("symbol", ""), OrderSide.SELL,
                    SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MODERATE,
                    confidence, entry_price=current_close, target_price=current_close * 0.9,
                    stop_loss=current_close * 1.05, metadata={"fast_ma": fast_ma, "slow_ma": slow_ma}
                )
        return None
