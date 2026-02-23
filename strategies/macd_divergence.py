"""MACD Divergence Strategy - Detects divergences between price and MACD."""
from __future__ import annotations
from typing import Any
import numpy as np
from core.types import OrderSide
from strategies import BaseStrategy, Signal, SignalStrength, register_strategy

@register_strategy
class MACDDivergenceStrategy(BaseStrategy):
    """MACD Divergence detection for reversals."""
    name = "macd_divergence"
    description = "Detect bullish/bearish MACD divergences"
    version = "1.0.0"
    params = {"fast": 12, "slow": 26, "signal": 9}
    min_confidence = 0.6

    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        bars = data.get("bars", [])
        if len(bars) < self.params["slow"] + 20:
            return None
        closes = np.array([b["close"] for b in bars])
        # Calculate MACD
        ema_fast = self._ema(closes, self.params["fast"])
        ema_slow = self._ema(closes, self.params["slow"])
        macd_line = ema_fast - ema_slow
        macd_signal = self._ema(macd_line, self.params["signal"])
        macd_hist = macd_line - macd_signal
        # Look for divergence (simplified)
        if len(macd_hist) < 10:
            return None
        # Bullish divergence: price lower low, MACD higher low
        price_low = np.min(closes[-10:])
        macd_low = np.min(macd_hist[-10:])
        if closes[-1] < price_low and macd_hist[-1] > macd_low:
            return Signal(self.name, data.get("symbol", ""), OrderSide.BUY,
                SignalStrength.STRONG, 0.7, entry_price=closes[-1],
                target_price=closes[-1] * 1.05, stop_loss=closes[-1] * 0.97)
        # Bearish divergence
        price_high = np.max(closes[-10:])
        macd_high = np.max(macd_hist[-10:])
        if closes[-1] > price_high and macd_hist[-1] < macd_high:
            return Signal(self.name, data.get("symbol", ""), OrderSide.SELL,
                SignalStrength.STRONG, 0.7, entry_price=closes[-1],
                target_price=closes[-1] * 0.95, stop_loss=closes[-1] * 1.03)
        return None

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        multiplier = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
