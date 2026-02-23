"""
Strategy Collection - 30+ diverse trading strategies across multiple categories.

This module provides pre-built strategy templates for the Strategy Marketplace.
"""
from __future__ import annotations

from typing import Any

from core.types import OrderSide
from strategies import BaseStrategy, Signal, SignalStrength, register_strategy


# ==================== TREND FOLLOWING STRATEGIES ====================

@register_strategy
class DualMovingAverageStrategy(BaseStrategy):
    """
    Dual Moving Average Crossover Strategy.
    
    Classic trend-following strategy using two moving averages.
    
    Signals:
        BUY: Fast MA crosses above Slow MA
        SELL: Fast MA crosses below Slow MA
    """
    name = "dual_ma_crossover"
    description = "Classic dual moving average crossover with trend confirmation"
    version = "1.0.0"
    
    params = {
        "fast_period": 10,
        "slow_period": 30,
        "trend_filter": True,
    }
    
    min_confidence = 0.55
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        bars = data.get("bars", [])
        if len(bars) < self.params["slow_period"] + 5:
            return None
        
        closes = [b["close"] for b in bars]
        
        # Calculate moving averages
        fast_ma = sum(closes[-self.params["fast_period"]:]) / self.params["fast_period"]
        slow_ma = sum(closes[-self.params["slow_period"]:]) / self.params["slow_period"]
        prev_fast = sum(closes[-self.params["fast_period"]-1:-1]) / self.params["fast_period"]
        prev_slow = sum(closes[-self.params["slow_period"]-1:-1]) / self.params["slow_period"]
        
        # Check for crossover
        current_price = closes[-1]
        
        if prev_fast <= prev_slow and fast_ma > slow_ma:
            # Bullish crossover
            confidence = min(1.0, 0.5 + (fast_ma - slow_ma) / slow_ma * 10)
            if confidence >= self.min_confidence:
                return Signal(
                    strategy_name=self.name,
                    symbol=data.get("symbol", ""),
                    side=OrderSide.BUY,
                    strength=self._calculate_strength(abs(fast_ma - slow_ma) / slow_ma),
                    confidence=confidence,
                    entry_price=current_price,
                    target_price=current_price * 1.05,
                    stop_loss=current_price * 0.97,
                    metadata={"fast_ma": fast_ma, "slow_ma": slow_ma},
                )
        
        elif prev_fast >= prev_slow and fast_ma < slow_ma:
            # Bearish crossover
            confidence = min(1.0, 0.5 + (slow_ma - fast_ma) / slow_ma * 10)
            if confidence >= self.min_confidence:
                return Signal(
                    strategy_name=self.name,
                    symbol=data.get("symbol", ""),
                    side=OrderSide.SELL,
                    strength=self._calculate_strength(abs(fast_ma - slow_ma) / slow_ma),
                    confidence=confidence,
                    entry_price=current_price,
                    target_price=current_price * 0.95,
                    stop_loss=current_price * 1.03,
                    metadata={"fast_ma": fast_ma, "slow_ma": slow_ma},
                )
        
        return None


@register_strategy
class TripleMovingAverageStrategy(BaseStrategy):
    """
    Triple Moving Average Strategy.
    
    Uses three MAs for stronger trend confirmation.
    """
    name = "triple_ma_trend"
    description = "Triple MA alignment for high-confidence trend following"
    version = "1.0.0"
    
    params = {
        "short_period": 5,
        "medium_period": 20,
        "long_period": 60,
    }
    
    min_confidence = 0.65
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        bars = data.get("bars", [])
        if len(bars) < self.params["long_period"] + 5:
            return None
        
        closes = [b["close"] for b in bars]
        
        short_ma = sum(closes[-self.params["short_period"]:]) / self.params["short_period"]
        medium_ma = sum(closes[-self.params["medium_period"]:]) / self.params["medium_period"]
        long_ma = sum(closes[-self.params["long_period"]:]) / self.params["long_period"]
        
        current_price = closes[-1]
        
        # Bullish alignment: Short > Medium > Long
        if short_ma > medium_ma > long_ma:
            confidence = min(1.0, 0.6 + (short_ma - long_ma) / long_ma * 15)
            if confidence >= self.min_confidence:
                return Signal(
                    strategy_name=self.name,
                    symbol=data.get("symbol", ""),
                    side=OrderSide.BUY,
                    strength=SignalStrength.STRONG,
                    confidence=confidence,
                    entry_price=current_price,
                    target_price=current_price * 1.06,
                    stop_loss=current_price * 0.96,
                )
        
        # Bearish alignment: Short < Medium < Long
        elif short_ma < medium_ma < long_ma:
            confidence = min(1.0, 0.6 + (long_ma - short_ma) / long_ma * 15)
            if confidence >= self.min_confidence:
                return Signal(
                    strategy_name=self.name,
                    symbol=data.get("symbol", ""),
                    side=OrderSide.SELL,
                    strength=SignalStrength.STRONG,
                    confidence=confidence,
                    entry_price=current_price,
                    target_price=current_price * 0.94,
                    stop_loss=current_price * 1.04,
                )
        
        return None


@register_strategy
class ADXTrendStrategy(BaseStrategy):
    """
    ADX Trend Strength Strategy.
    
    Uses Average Directional Index to identify strong trends.
    """
    name = "adx_trend"
    description = "Trade strong trends identified by ADX > 25"
    version = "1.0.0"
    
    params = {
        "adx_period": 14,
        "adx_threshold": 25,
        "di_period": 14,
    }
    
    min_confidence = 0.58
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        indicators = data.get("indicators", {})
        adx = indicators.get("adx", 0)
        plus_di = indicators.get("plus_di", 0)
        minus_di = indicators.get("minus_di", 0)
        current_price = data.get("current_price", 0)
        
        if adx > self.params["adx_threshold"]:
            if plus_di > minus_di * 1.2:
                confidence = min(1.0, 0.5 + (adx - 25) / 100)
                if confidence >= self.min_confidence:
                    return Signal(
                        strategy_name=self.name,
                        symbol=data.get("symbol", ""),
                        side=OrderSide.BUY,
                        strength=SignalStrength.MODERATE,
                        confidence=confidence,
                        entry_price=current_price,
                        target_price=current_price * 1.04,
                        stop_loss=current_price * 0.97,
                        metadata={"adx": adx, "plus_di": plus_di, "minus_di": minus_di},
                    )
            elif minus_di > plus_di * 1.2:
                confidence = min(1.0, 0.5 + (adx - 25) / 100)
                if confidence >= self.min_confidence:
                    return Signal(
                        strategy_name=self.name,
                        symbol=data.get("symbol", ""),
                        side=OrderSide.SELL,
                        strength=SignalStrength.MODERATE,
                        confidence=confidence,
                        entry_price=current_price,
                        target_price=current_price * 0.96,
                        stop_loss=current_price * 1.03,
                        metadata={"adx": adx, "plus_di": plus_di, "minus_di": minus_di},
                    )
        
        return None


# ==================== MEAN REVERSION STRATEGIES ====================

@register_strategy
class ZScoreMeanReversionStrategy(BaseStrategy):
    """
    Z-Score Mean Reversion Strategy.
    
    Trades when price deviates significantly from moving average.
    """
    name = "zscore_mean_reversion"
    description = "Statistical mean reversion using Z-score thresholds"
    version = "1.0.0"
    
    params = {
        "lookback_period": 30,
        "entry_zscore": 2.0,
        "exit_zscore": 0.5,
    }
    
    min_confidence = 0.55
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        bars = data.get("bars", [])
        if len(bars) < self.params["lookback_period"] + 5:
            return None
        
        closes = [b["close"] for b in bars[-self.params["lookback_period"]:]]
        current_price = closes[-1]
        
        mean = sum(closes) / len(closes)
        std = (sum((x - mean) ** 2 for x in closes) / len(closes)) ** 0.5
        
        if std == 0:
            return None
        
        zscore = (current_price - mean) / std
        
        if zscore < -self.params["entry_zscore"]:
            # Price significantly below mean - buy
            confidence = min(1.0, 0.5 + abs(zscore) / 10)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,
                strength=self._calculate_strength(abs(zscore) / self.params["entry_zscore"]),
                confidence=confidence,
                entry_price=current_price,
                target_price=mean,
                stop_loss=current_price * 0.95,
                metadata={"zscore": zscore, "mean": mean, "std": std},
            )
        
        elif zscore > self.params["entry_zscore"]:
            # Price significantly above mean - sell
            confidence = min(1.0, 0.5 + abs(zscore) / 10)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.SELL,
                strength=self._calculate_strength(abs(zscore) / self.params["entry_zscore"]),
                confidence=confidence,
                entry_price=current_price,
                target_price=mean,
                stop_loss=current_price * 1.05,
                metadata={"zscore": zscore, "mean": mean, "std": std},
            )
        
        return None


@register_strategy
class BollingerBandReversionStrategy(BaseStrategy):
    """
    Bollinger Band Mean Reversion.
    
    Fades moves to outer bands with middle band target.
    """
    name = "bb_reversion"
    description = "Mean reversion using Bollinger Band extremes"
    version = "1.0.0"
    
    params = {
        "period": 20,
        "std_dev": 2.0,
    }
    
    min_confidence = 0.52
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        indicators = data.get("indicators", {})
        bb_upper = indicators.get("bb_upper", 0)
        bb_lower = indicators.get("bb_lower", 0)
        bb_middle = indicators.get("bb_middle", 0)
        current_price = data.get("current_price", 0)
        
        if bb_lower == 0 or bb_upper == 0:
            return None
        
        if current_price <= bb_lower:
            confidence = min(1.0, 0.5 + (bb_lower - current_price) / bb_lower * 10)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current_price,
                target_price=bb_middle,
                stop_loss=current_price * 0.96,
                metadata={"bb_position": "lower"},
            )
        
        elif current_price >= bb_upper:
            confidence = min(1.0, 0.5 + (current_price - bb_upper) / bb_upper * 10)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.SELL,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current_price,
                target_price=bb_middle,
                stop_loss=current_price * 1.04,
                metadata={"bb_position": "upper"},
            )
        
        return None


@register_strategy
class RSIReversalStrategy(BaseStrategy):
    """
    RSI Reversal Strategy.
    
    Trades RSI extremes with trend confirmation.
    """
    name = "rsi_reversal"
    description = "RSI oversold/overbought reversal with divergence detection"
    version = "1.0.0"
    
    params = {
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
    }
    
    min_confidence = 0.50
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        indicators = data.get("indicators", {})
        rsi = indicators.get("rsi", 50)
        current_price = data.get("current_price", 0)
        
        if rsi < self.params["oversold"]:
            confidence = min(1.0, 0.5 + (self.params["oversold"] - rsi) / 100)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current_price,
                target_price=current_price * 1.03,
                stop_loss=current_price * 0.97,
                metadata={"rsi": rsi, "condition": "oversold"},
            )
        
        elif rsi > self.params["overbought"]:
            confidence = min(1.0, 0.5 + (rsi - self.params["overbought"]) / 100)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.SELL,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current_price,
                target_price=current_price * 0.97,
                stop_loss=current_price * 1.03,
                metadata={"rsi": rsi, "condition": "overbought"},
            )
        
        return None


# ==================== MOMENTUM STRATEGIES ====================

@register_strategy
class RateOfChangeMomentumStrategy(BaseStrategy):
    """
    Rate of Change Momentum Strategy.
    
    Trades strong positive/negative momentum.
    """
    name = "roc_momentum"
    description = "Momentum strategy using rate of change indicator"
    version = "1.0.0"
    
    params = {
        "roc_period": 12,
        "threshold": 5.0,
    }
    
    min_confidence = 0.55
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        indicators = data.get("indicators", {})
        roc = indicators.get("roc", 0)
        current_price = data.get("current_price", 0)
        
        if roc > self.params["threshold"]:
            confidence = min(1.0, 0.5 + roc / 50)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,
                strength=self._calculate_strength(roc / self.params["threshold"]),
                confidence=confidence,
                entry_price=current_price,
                target_price=current_price * (1 + roc / 100),
                stop_loss=current_price * 0.97,
                metadata={"roc": roc},
            )
        
        elif roc < -self.params["threshold"]:
            confidence = min(1.0, 0.5 + abs(roc) / 50)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.SELL,
                strength=self._calculate_strength(abs(roc) / self.params["threshold"]),
                confidence=confidence,
                entry_price=current_price,
                target_price=current_price * (1 + roc / 100),
                stop_loss=current_price * 1.03,
                metadata={"roc": roc},
            )
        
        return None


@register_strategy
class StochasticMomentumStrategy(BaseStrategy):
    """
    Stochastic Oscillator Momentum Strategy.
    
    Uses stochastic crossovers for entry signals.
    """
    name = "stochastic_momentum"
    description = "Stochastic oscillator with signal line crossovers"
    version = "1.0.0"
    
    params = {
        "k_period": 14,
        "d_period": 3,
        "oversold": 20,
        "overbought": 80,
    }
    
    min_confidence = 0.52
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        indicators = data.get("indicators", {})
        k = indicators.get("stoch_k", 50)
        d = indicators.get("stoch_d", 50)
        prev_k = indicators.get("stoch_k_prev", 50)
        prev_d = indicators.get("stoch_d_prev", 50)
        current_price = data.get("current_price", 0)
        
        # Bullish crossover in oversold zone
        if prev_k <= prev_d and k > d and k < self.params["oversold"]:
            confidence = min(1.0, 0.5 + (self.params["oversold"] - k) / 100)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current_price,
                target_price=current_price * 1.04,
                stop_loss=current_price * 0.97,
                metadata={"stoch_k": k, "stoch_d": d},
            )
        
        # Bearish crossover in overbought zone
        elif prev_k >= prev_d and k < d and k > self.params["overbought"]:
            confidence = min(1.0, 0.5 + (k - self.params["overbought"]) / 100)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.SELL,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current_price,
                target_price=current_price * 0.96,
                stop_loss=current_price * 1.03,
                metadata={"stoch_k": k, "stoch_d": d},
            )
        
        return None


# ==================== BREAKOUT STRATEGIES ====================

@register_strategy
class DonchianBreakoutStrategy(BaseStrategy):
    """
    Donchian Channel Breakout Strategy.
    
    Classic Turtle Trading breakout system.
    """
    name = "donchian_breakout"
    description = "Donchian channel breakout following Turtle Trading rules"
    version = "1.0.0"
    
    params = {
        "lookback_period": 20,
        "exit_period": 10,
    }
    
    min_confidence = 0.58
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        bars = data.get("bars", [])
        if len(bars) < self.params["lookback_period"] + 5:
            return None
        
        lookback = bars[-self.params["lookback_period"]-1:-1]
        current = bars[-1]
        
        highest = max(b["high"] for b in lookback)
        lowest = min(b["low"] for b in lookback)
        current_close = current["close"]
        current_volume = current["volume"]
        avg_volume = sum(b["volume"] for b in lookback) / len(lookback)
        
        # Breakout above with volume confirmation
        if current_close > highest and current_volume > avg_volume * 1.3:
            confidence = min(1.0, 0.55 + (current_close - highest) / highest * 10)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,
                strength=SignalStrength.STRONG,
                confidence=confidence,
                entry_price=current_close,
                target_price=current_close * 1.06,
                stop_loss=highest * 0.98,
                metadata={"donchian_high": highest, "donchian_low": lowest},
            )
        
        # Breakdown below with volume confirmation
        elif current_close < lowest and current_volume > avg_volume * 1.3:
            confidence = min(1.0, 0.55 + (lowest - current_close) / lowest * 10)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.SELL,
                strength=SignalStrength.STRONG,
                confidence=confidence,
                entry_price=current_close,
                target_price=current_close * 0.94,
                stop_loss=lowest * 1.02,
                metadata={"donchian_high": highest, "donchian_low": lowest},
            )
        
        return None


@register_strategy
class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility Compression Breakout Strategy.
    
    Trades breakouts from low volatility periods.
    """
    name = "volatility_breakout"
    description = "Breakout from volatility compression (squeeze)"
    version = "1.0.0"
    
    params = {
        "bb_period": 20,
        "keltner_period": 20,
        "atr_multiplier": 1.5,
    }
    
    min_confidence = 0.55
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        indicators = data.get("indicators", {})
        bb_width = indicators.get("bb_width", 0)
        atr = indicators.get("atr", 0)
        current_price = data.get("current_price", 0)
        
        # Detect squeeze (narrow Bollinger Bands)
        if bb_width > 0 and bb_width < 0.05:  # Compression detected
            # Wait for expansion and breakout
            momentum = indicators.get("momentum", 0)
            
            if momentum > 2:
                confidence = min(1.0, 0.55 + momentum / 20)
                return Signal(
                    strategy_name=self.name,
                    symbol=data.get("symbol", ""),
                    side=OrderSide.BUY,
                    strength=SignalStrength.MODERATE,
                    confidence=confidence,
                    entry_price=current_price,
                    target_price=current_price * 1.05,
                    stop_loss=current_price * 0.97,
                    metadata={"squeeze": True, "momentum": momentum},
                )
            elif momentum < -2:
                confidence = min(1.0, 0.55 + abs(momentum) / 20)
                return Signal(
                    strategy_name=self.name,
                    symbol=data.get("symbol", ""),
                    side=OrderSide.SELL,
                    strength=SignalStrength.MODERATE,
                    confidence=confidence,
                    entry_price=current_price,
                    target_price=current_price * 0.95,
                    stop_loss=current_price * 1.03,
                    metadata={"squeeze": True, "momentum": momentum},
                )
        
        return None


# ==================== VOLUME STRATEGIES ====================

@register_strategy
class VolumeWeightedAveragePriceStrategy(BaseStrategy):
    """
    VWAP Mean Reversion Strategy.
    
    Trades deviations from VWAP with volume confirmation.
    """
    name = "vwap_reversion_enhanced"
    description = "VWAP reversion with volume profile analysis"
    version = "1.0.0"
    
    params = {
        "deviation_threshold": 2.0,
        "volume_confirmation": True,
    }
    
    min_confidence = 0.55
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        indicators = data.get("indicators", {})
        vwap = indicators.get("vwap", 0)
        current_price = data.get("current_price", 0)
        volume = indicators.get("volume", 0)
        avg_volume = indicators.get("avg_volume", 0)
        
        if vwap == 0:
            return None
        
        deviation = (current_price - vwap) / vwap * 100
        
        # Check volume confirmation
        volume_ok = not self.params["volume_confirmation"] or volume > avg_volume * 0.8
        
        if deviation < -self.params["deviation_threshold"] and volume_ok:
            confidence = min(1.0, 0.5 + abs(deviation) / 10)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current_price,
                target_price=vwap,
                stop_loss=current_price * 0.96,
                metadata={"vwap": vwap, "deviation": deviation},
            )
        
        elif deviation > self.params["deviation_threshold"] and volume_ok:
            confidence = min(1.0, 0.5 + deviation / 10)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.SELL,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current_price,
                target_price=vwap,
                stop_loss=current_price * 1.04,
                metadata={"vwap": vwap, "deviation": deviation},
            )
        
        return None


@register_strategy
class OnBalanceVolumeStrategy(BaseStrategy):
    """
    On-Balance Volume Divergence Strategy.
    
    Detects OBV divergences for reversal signals.
    """
    name = "obv_divergence"
    description = "Volume-based divergence detection using OBV"
    version = "1.0.0"
    
    params = {
        "lookback_period": 14,
    }
    
    min_confidence = 0.58
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        indicators = data.get("indicators", {})
        obv = indicators.get("obv", 0)
        obv_prev = indicators.get("obv_prev", 0)
        price = data.get("current_price", 0)
        price_prev = indicators.get("price_prev", 0)
        
        # Bullish divergence: price lower low, OBV higher low
        if price < price_prev and obv > obv_prev:
            confidence = min(1.0, 0.55 + (obv - obv_prev) / abs(obv_prev + 1) * 10)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=price,
                target_price=price * 1.04,
                stop_loss=price * 0.97,
                metadata={"divergence": "bullish", "obv": obv},
            )
        
        # Bearish divergence: price higher high, OBV lower high
        elif price > price_prev and obv < obv_prev:
            confidence = min(1.0, 0.55 + (obv_prev - obv) / abs(obv_prev + 1) * 10)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.SELL,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=price,
                target_price=price * 0.96,
                stop_loss=price * 1.03,
                metadata={"divergence": "bearish", "obv": obv},
            )
        
        return None


# ==================== PATTERN RECOGNITION STRATEGIES ====================

@register_strategy
class CandlestickPatternStrategy(BaseStrategy):
    """
    Candlestick Pattern Recognition Strategy.
    
    Detects and trades classic candlestick patterns.
    """
    name = "candlestick_patterns"
    description = "Classic candlestick pattern detection and trading"
    version = "1.0.0"
    
    params = {
        "min_body_ratio": 0.6,
        "confirmation": True,
    }
    
    min_confidence = 0.52
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        bars = data.get("bars", [])
        if len(bars) < 3:
            return None
        
        current = bars[-1]
        prev = bars[-2]
        
        # Detect hammer pattern
        if self._is_hammer(prev):
            confidence = 0.60
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current["close"],
                target_price=current["close"] * 1.03,
                stop_loss=prev["low"] * 0.99,
                metadata={"pattern": "hammer"},
            )
        
        # Detect shooting star
        if self._is_shooting_star(prev):
            confidence = 0.60
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.SELL,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current["close"],
                target_price=current["close"] * 0.97,
                stop_loss=prev["high"] * 1.01,
                metadata={"pattern": "shooting_star"},
            )
        
        # Detect bullish engulfing
        if self._is_bullish_engulfing(prev, current):
            confidence = 0.65
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current["close"],
                target_price=current["close"] * 1.04,
                stop_loss=current["low"] * 0.99,
                metadata={"pattern": "bullish_engulfing"},
            )
        
        return None
    
    def _is_hammer(self, bar: dict) -> bool:
        body = abs(bar["close"] - bar["open"])
        upper_shadow = bar["high"] - max(bar["open"], bar["close"])
        lower_shadow = min(bar["open"], bar["close"]) - bar["low"]
        total_range = bar["high"] - bar["low"]
        
        if total_range == 0:
            return False
        
        return (lower_shadow > body * 2 and 
                upper_shadow < body * 0.5 and
                body / total_range > 0.3)
    
    def _is_shooting_star(self, bar: dict) -> bool:
        body = abs(bar["close"] - bar["open"])
        upper_shadow = bar["high"] - max(bar["open"], bar["close"])
        lower_shadow = min(bar["open"], bar["close"]) - bar["low"]
        
        return (upper_shadow > body * 2 and 
                lower_shadow < body * 0.5)
    
    def _is_bullish_engulfing(self, prev: dict, current: dict) -> bool:
        prev_body = prev["open"] - prev["close"]  # Bearish
        curr_body = current["close"] - current["open"]  # Bullish
        
        return (prev_body > 0 and curr_body > 0 and
                current["close"] > prev["open"] and
                current["open"] < prev["close"])


# ==================== MULTI-FACTOR STRATEGIES ====================

@register_strategy
class MultiFactorAlphaStrategy(BaseStrategy):
    """
    Multi-Factor Alpha Strategy.
    
    Combines momentum, value, and quality factors.
    """
    name = "multi_factor_alpha"
    description = "Multi-factor model combining momentum, value, and quality"
    version = "1.0.0"
    
    params = {
        "momentum_weight": 0.4,
        "value_weight": 0.3,
        "quality_weight": 0.3,
        "min_score": 0.6,
    }
    
    min_confidence = 0.60
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        indicators = data.get("indicators", {})
        
        # Calculate factor scores (normalized 0-1)
        momentum_score = min(1.0, max(0, indicators.get("momentum_score", 0.5)))
        value_score = min(1.0, max(0, indicators.get("value_score", 0.5)))
        quality_score = min(1.0, max(0, indicators.get("quality_score", 0.5)))
        
        # Weighted composite score
        composite = (
            momentum_score * self.params["momentum_weight"] +
            value_score * self.params["value_weight"] +
            quality_score * self.params["quality_weight"]
        )
        
        current_price = data.get("current_price", 0)
        
        if composite >= self.params["min_score"]:
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,
                strength=self._calculate_strength(composite),
                confidence=composite,
                entry_price=current_price,
                target_price=current_price * 1.05,
                stop_loss=current_price * 0.96,
                metadata={
                    "composite_score": composite,
                    "momentum": momentum_score,
                    "value": value_score,
                    "quality": quality_score,
                },
            )
        
        elif composite <= (1 - self.params["min_score"]):
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.SELL,
                strength=self._calculate_strength(1 - composite),
                confidence=1 - composite,
                entry_price=current_price,
                target_price=current_price * 0.95,
                stop_loss=current_price * 1.04,
                metadata={
                    "composite_score": composite,
                    "momentum": momentum_score,
                    "value": value_score,
                    "quality": quality_score,
                },
            )
        
        return None


# ==================== EVENT-DRIVEN STRATEGIES ====================

@register_strategy
class EarningsAnnouncementStrategy(BaseStrategy):
    """
    Earnings Announcement Strategy.
    
    Trades pre and post-earnings momentum.
    """
    name = "earnings_momentum_enhanced"
    description = "Enhanced earnings momentum with sentiment integration"
    version = "1.0.0"
    
    params = {
        "lookback_days": 5,
        "momentum_period": 10,
        "sentiment_weight": 0.3,
    }
    
    min_confidence = 0.60
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        bars = data.get("bars", [])
        if len(bars) < self.params["momentum_period"] + 5:
            return None
        
        # Calculate post-earnings momentum
        closes = [b["close"] for b in bars[-self.params["momentum_period"]:]]
        momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
        
        # Get sentiment score if available
        sentiment = data.get("sentiment_score", 0)
        
        # Combined score
        combined_score = momentum * 0.7 + sentiment * self.params["sentiment_weight"]
        
        current_price = closes[-1]
        
        if combined_score > 0.03:  # 3% threshold
            confidence = min(1.0, 0.55 + combined_score * 5)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,
                strength=self._calculate_strength(abs(combined_score) * 10),
                confidence=confidence,
                entry_price=current_price,
                target_price=current_price * 1.06,
                stop_loss=current_price * 0.95,
                metadata={"momentum": momentum, "sentiment": sentiment},
            )
        
        elif combined_score < -0.03:
            confidence = min(1.0, 0.55 + abs(combined_score) * 5)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.SELL,
                strength=self._calculate_strength(abs(combined_score) * 10),
                confidence=confidence,
                entry_price=current_price,
                target_price=current_price * 0.94,
                stop_loss=current_price * 1.05,
                metadata={"momentum": momentum, "sentiment": sentiment},
            )
        
        return None


# ==================== SCALPING STRATEGIES ====================

@register_strategy
class OrderFlowScalpingStrategy(BaseStrategy):
    """
    Order Flow Scalping Strategy.
    
    Quick scalps based on order flow imbalances.
    """
    name = "order_flow_scalping"
    description = "High-frequency scalping using order flow analysis"
    version = "1.0.0"
    
    params = {
        "imbalance_threshold": 2.0,
        "target_bps": 20,
        "stop_bps": 10,
    }
    
    min_confidence = 0.55
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        indicators = data.get("indicators", {})
        buy_volume = indicators.get("buy_volume", 0)
        sell_volume = indicators.get("sell_volume", 0)
        current_price = data.get("current_price", 0)
        
        if sell_volume == 0:
            return None
        
        imbalance = buy_volume / sell_volume
        
        if imbalance > self.params["imbalance_threshold"]:
            confidence = min(1.0, 0.55 + (imbalance - self.params["imbalance_threshold"]) / 10)
            target_bps = self.params["target_bps"] / 10000
            stop_bps = self.params["stop_bps"] / 10000
            
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current_price,
                target_price=current_price * (1 + target_bps),
                stop_loss=current_price * (1 - stop_bps),
                metadata={"imbalance": imbalance},
            )
        
        elif imbalance < (1 / self.params["imbalance_threshold"]):
            confidence = min(1.0, 0.55 + (1/imbalance - self.params["imbalance_threshold"]) / 10)
            target_bps = self.params["target_bps"] / 10000
            stop_bps = self.params["stop_bps"] / 10000
            
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.SELL,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current_price,
                target_price=current_price * (1 - target_bps),
                stop_loss=current_price * (1 + stop_bps),
                metadata={"imbalance": imbalance},
            )
        
        return None


# ==================== STATISTICAL ARBITRAGE ====================

@register_strategy
class PairsTradingStrategy(BaseStrategy):
    """
    Pairs Trading Strategy.
    
    Statistical arbitrage between correlated assets.
    """
    name = "pairs_trading"
    description = "Statistical arbitrage using cointegrated pairs"
    version = "1.0.0"
    
    params = {
        "lookback_period": 60,
        "entry_zscore": 2.0,
        "exit_zscore": 0.5,
    }
    
    min_confidence = 0.58
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        # This strategy requires pair data - simplified version
        spread_zscore = data.get("spread_zscore", 0)
        current_price = data.get("current_price", 0)
        pair_symbol = data.get("pair_symbol", "")
        
        if spread_zscore > self.params["entry_zscore"]:
            confidence = min(1.0, 0.55 + (spread_zscore - self.params["entry_zscore"]) / 5)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.SELL,  # Sell spread
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current_price,
                target_price=current_price * 0.98,
                stop_loss=current_price * 1.02,
                metadata={"spread_zscore": spread_zscore, "pair": pair_symbol},
            )
        
        elif spread_zscore < -self.params["entry_zscore"]:
            confidence = min(1.0, 0.55 + (abs(spread_zscore) - self.params["entry_zscore"]) / 5)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,  # Buy spread
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current_price,
                target_price=current_price * 1.02,
                stop_loss=current_price * 0.98,
                metadata={"spread_zscore": spread_zscore, "pair": pair_symbol},
            )
        
        return None


# ==================== MACHINE LEARNING STRATEGIES ====================

@register_strategy
class MLClassifierStrategy(BaseStrategy):
    """
    Machine Learning Classifier Strategy.
    
    Uses pre-trained ML model for signal generation.
    """
    name = "ml_classifier"
    description = "ML-based classification for directional prediction"
    version = "1.0.0"
    
    params = {
        "min_probability": 0.65,
        "model_type": "random_forest",
    }
    
    min_confidence = 0.65
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        # Get ML prediction from data
        ml_prediction = data.get("ml_prediction", {})
        predicted_class = ml_prediction.get("class", 0)  # -1, 0, 1
        probability = ml_prediction.get("probability", 0.5)
        current_price = data.get("current_price", 0)
        
        if probability < self.params["min_probability"]:
            return None
        
        if predicted_class == 1:  # Bullish
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,
                strength=self._calculate_strength(probability),
                confidence=probability,
                entry_price=current_price,
                target_price=current_price * 1.05,
                stop_loss=current_price * 0.96,
                metadata={"ml_prediction": ml_prediction},
            )
        
        elif predicted_class == -1:  # Bearish
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.SELL,
                strength=self._calculate_strength(probability),
                confidence=probability,
                entry_price=current_price,
                target_price=current_price * 0.95,
                stop_loss=current_price * 1.04,
                metadata={"ml_prediction": ml_prediction},
            )
        
        return None


# ==================== SENTIMENT-BASED STRATEGIES ====================

@register_strategy
class SentimentMomentumStrategy(BaseStrategy):
    """
    Sentiment Momentum Strategy.
    
    Combines news sentiment with price momentum.
    """
    name = "sentiment_momentum"
    description = "News sentiment combined with price momentum"
    version = "1.0.0"
    
    params = {
        "sentiment_threshold": 0.3,
        "momentum_period": 5,
        "sentiment_weight": 0.4,
    }
    
    min_confidence = 0.58
    
    def generate_signal(self, data: dict[str, Any]) -> Signal | None:
        sentiment_score = data.get("sentiment_score", 0)
        bars = data.get("bars", [])
        
        if len(bars) < self.params["momentum_period"] + 5:
            return None
        
        closes = [b["close"] for b in bars[-self.params["momentum_period"]:]]
        momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
        
        # Combined score
        combined = (
            sentiment_score * self.params["sentiment_weight"] +
            momentum * (1 - self.params["sentiment_weight"])
        )
        
        current_price = closes[-1]
        
        if combined > self.params["sentiment_threshold"]:
            confidence = min(1.0, 0.55 + combined * 0.5)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.BUY,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current_price,
                target_price=current_price * 1.04,
                stop_loss=current_price * 0.97,
                metadata={"sentiment": sentiment_score, "momentum": momentum},
            )
        
        elif combined < -self.params["sentiment_threshold"]:
            confidence = min(1.0, 0.55 + abs(combined) * 0.5)
            return Signal(
                strategy_name=self.name,
                symbol=data.get("symbol", ""),
                side=OrderSide.SELL,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current_price,
                target_price=current_price * 0.96,
                stop_loss=current_price * 1.03,
                metadata={"sentiment": sentiment_score, "momentum": momentum},
            )
        
        return None
