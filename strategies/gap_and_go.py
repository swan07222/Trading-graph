"""Gap and Go Strategy - Trades opening gaps."""
from __future__ import annotations

from typing import Any

import numpy as np

from core.types import OrderSide
from strategies import BaseStrategy, Signal, SignalStrength, register_strategy


@register_strategy
class GapAndGoStrategy(BaseStrategy):
    """Gap and Go momentum strategy."""
    name = "gap_and_go"
    description = "Trade continuation of opening gaps"
    version = "1.0.0"
    params = {"min_gap_pct": 2.0, "lookback": 20}
    min_confidence = 0.55

    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        bars = data.get("bars", [])
        if len(bars) < self.params["lookback"] + 2:
            return None
        closes = np.array([b["close"] for b in bars[-self.params["lookback"] - 2 :]])
        opens = np.array([b["open"] for b in bars[-self.params["lookback"] - 2 :]])
        # Calculate gap
        prev_close = closes[-2]
        curr_open = opens[-1]
        gap_pct = (curr_open - prev_close) / prev_close * 100
        avg_range = np.mean([bars[i]["high"] - bars[i]["low"] for i in range(-self.params["lookback"] - 2, -2)])
        # Gap up
        if gap_pct > self.params["min_gap_pct"]:
            confidence = min(1.0, 0.5 + gap_pct / 10)
            if confidence >= self.min_confidence:
                return Signal(self.name, data.get("symbol", ""), OrderSide.BUY,
                    SignalStrength.MODERATE, confidence, entry_price=curr_open,
                    target_price=curr_open + avg_range, stop_loss=prev_close,
                    metadata={"gap_pct": gap_pct})
        # Gap down
        elif gap_pct < -self.params["min_gap_pct"]:
            confidence = min(1.0, 0.5 + abs(gap_pct) / 10)
            if confidence >= self.min_confidence:
                return Signal(self.name, data.get("symbol", ""), OrderSide.SELL,
                    SignalStrength.MODERATE, confidence, entry_price=curr_open,
                    target_price=curr_open - avg_range, stop_loss=prev_close,
                    metadata={"gap_pct": gap_pct})
        return None
