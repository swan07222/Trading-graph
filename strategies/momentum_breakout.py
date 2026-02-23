"""
Momentum Breakout Strategy.

Identifies stocks breaking out of consolidation with high volume.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from core.types import OrderSide
from strategies import BaseStrategy, Signal, SignalStrength, register_strategy


@register_strategy
class MomentumBreakoutStrategy(BaseStrategy):
    """
    Momentum Breakout Strategy.

    Identifies stocks breaking out of N-day highs with volume confirmation.

    Signals:
        BUY: Price breaks above N-day high with volume > M-day avg
        SELL: Price breaks below N-day low with volume > M-day avg

    Parameters:
        lookback_days: Number of days for high/low (default: 20)
        volume_multiplier: Volume threshold multiplier (default: 1.5)
        min_confidence: Minimum confidence threshold (default: 0.6)
    """

    name = "momentum_breakout"
    description = "Momentum breakout with volume confirmation"
    version = "1.0.0"

    params = {
        "lookback_days": 20,
        "volume_multiplier": 1.5,
    }

    min_confidence = 0.6

    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        """Generate breakout signal."""
        bars = data.get("bars", [])
        if len(bars) < self.params["lookback_days"] + 5:
            return None

        # Get recent bars
        recent = bars[-self.params["lookback_days"] - 1 : -1]
        current = bars[-1]

        # Calculate N-day high/low
        highs = [b["high"] for b in recent]
        lows = [b["low"] for b in recent]
        volumes = [b["volume"] for b in recent]

        n_day_high = max(highs)
        n_day_low = min(lows)
        avg_volume = sum(volumes) / len(volumes)

        current_close = current["close"]
        current_volume = current["volume"]

        # Check for breakout above
        if current_close > n_day_high and current_volume > avg_volume * self.params["volume_multiplier"]:
            # Calculate confidence based on volume and breakout magnitude
            breakout_pct = (current_close - n_day_high) / n_day_high
            volume_ratio = current_volume / avg_volume

            confidence = min(1.0, 0.5 + breakout_pct * 10 + (volume_ratio - 1) * 0.2)

            if confidence >= self.min_confidence:
                strength = self._calculate_strength(breakout_pct, volume_ratio)
                target = n_day_high * 1.05  # 5% target
                stop = n_day_high * 0.97  # 3% stop

                return Signal(
                    strategy_name=self.name,
                    symbol=data.get("symbol", ""),
                    side=OrderSide.BUY,
                    strength=strength,
                    confidence=confidence,
                    entry_price=current_close,
                    target_price=target,
                    stop_loss=stop,
                    metadata={
                        "breakout_pct": breakout_pct,
                        "volume_ratio": volume_ratio,
                        "n_day_high": n_day_high,
                    },
                )

        # Check for breakdown below
        elif current_close < n_day_low and current_volume > avg_volume * self.params["volume_multiplier"]:
            breakdown_pct = (n_day_low - current_close) / n_day_low
            volume_ratio = current_volume / avg_volume

            confidence = min(1.0, 0.5 + breakdown_pct * 10 + (volume_ratio - 1) * 0.2)

            if confidence >= self.min_confidence:
                strength = self._calculate_strength(breakdown_pct, volume_ratio)
                target = n_day_low * 0.95
                stop = n_day_low * 1.03

                return Signal(
                    strategy_name=self.name,
                    symbol=data.get("symbol", ""),
                    side=OrderSide.SELL,
                    strength=strength,
                    confidence=confidence,
                    entry_price=current_close,
                    target_price=target,
                    stop_loss=stop,
                    metadata={
                        "breakdown_pct": breakdown_pct,
                        "volume_ratio": volume_ratio,
                        "n_day_low": n_day_low,
                    },
                )

        return None

    def _calculate_strength(self, breakout_pct: float, volume_ratio: float) -> SignalStrength:
        """Calculate signal strength."""
        score = breakout_pct * 100 + volume_ratio
        if score > 3:
            return SignalStrength.VERY_STRONG
        elif score > 2:
            return SignalStrength.STRONG
        elif score > 1:
            return SignalStrength.MODERATE
        elif score > 0.5:
            return SignalStrength.WEAK
        return SignalStrength.VERY_WEAK
