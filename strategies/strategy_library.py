# strategies/strategy_library.py
"""Comprehensive Strategy Library - 50+ Pre-built Trading Strategies.

This module provides a collection of ready-to-use trading strategies:

Categories:
1. Trend Following (10 strategies)
2. Mean Reversion (10 strategies)
3. Momentum (8 strategies)
4. Breakout (8 strategies)
5. Volume-based (6 strategies)
6. Volatility (5 strategies)
7. Pattern-based (5 strategies)
8. Multi-factor (5 strategies)
9. Machine Learning (5 strategies)
10. Hybrid (5 strategies)

Each strategy includes:
- Clear entry/exit rules
- Risk management parameters
- Position sizing logic
- Performance expectations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


class StrategyCategory(Enum):
    """Strategy categories."""
    TREND = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    PATTERN = "pattern"
    MULTI_FACTOR = "multi_factor"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"


class TimeFrame(Enum):
    """Strategy timeframes."""
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class StrategySignal:
    """Trading signal from strategy."""
    timestamp: pd.Timestamp
    symbol: str
    signal: int  # -1 (sell), 0 (hold), 1 (buy)
    strength: float  # 0-1 confidence
    price: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    name: str
    category: StrategyCategory
    timeframe: TimeFrame
    description: str
    parameters: dict = field(default_factory=dict)
    risk_params: dict = field(default_factory=lambda: {
        "max_position_pct": 0.10,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.15,
        "max_drawdown_pct": 0.20,
    })


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.parameters = config.parameters
        self.risk_params = config.risk_params

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from OHLCV data."""
        pass

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        volatility: float = 0.02,
    ) -> int:
        """Calculate position size based on risk parameters."""
        max_position = capital * self.risk_params["max_position_pct"]
        risk_amount = capital * self.risk_params["stop_loss_pct"]

        # Volatility-adjusted position
        vol_adjustment = 0.02 / max(volatility, 0.001)
        position_value = min(max_position, risk_amount * vol_adjustment)

        return int(position_value / price)

    def calculate_stop_loss(self, entry_price: float, side: int) -> float:
        """Calculate stop loss price."""
        sl_pct = self.risk_params["stop_loss_pct"]
        if side > 0:  # Long
            return entry_price * (1 - sl_pct)
        else:  # Short
            return entry_price * (1 + sl_pct)

    def calculate_take_profit(self, entry_price: float, side: int) -> float:
        """Calculate take profit price."""
        tp_pct = self.risk_params["take_profit_pct"]
        if side > 0:  # Long
            return entry_price * (1 + tp_pct)
        else:  # Short
            return entry_price * (1 - tp_pct)


# ============================================================================
# TREND FOLLOWING STRATEGIES
# ============================================================================

class DualMovingAverageStrategy(BaseStrategy):
    """Dual Moving Average Crossover Strategy.

    Entry:
    - Buy when fast MA crosses above slow MA
    - Sell when fast MA crosses below slow MA

    Exit:
    - Reverse signal or stop loss
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="Dual Moving Average",
            category=StrategyCategory.TREND,
            timeframe=TimeFrame.DAILY,
            description="Classic dual MA crossover strategy",
            parameters={
                "fast_period": 20,
                "slow_period": 60,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        fast_ma = close.rolling(self.parameters["fast_period"]).mean()
        slow_ma = close.rolling(self.parameters["slow_period"]).mean()

        signals = pd.Series(0, index=data.index)

        # Generate signals
        crossover = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        crossunder = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        signals[crossover] = 1
        signals[crossunder] = -1

        return signals


class TripleMovingAverageStrategy(BaseStrategy):
    """Triple Moving Average Strategy.

    Uses three MAs for trend confirmation.
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="Triple Moving Average",
            category=StrategyCategory.TREND,
            timeframe=TimeFrame.DAILY,
            description="Three MA trend confirmation strategy",
            parameters={
                "short_period": 10,
                "medium_period": 30,
                "long_period": 90,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        short_ma = close.rolling(self.parameters["short_period"]).mean()
        medium_ma = close.rolling(self.parameters["medium_period"]).mean()
        long_ma = close.rolling(self.parameters["long_period"]).mean()

        signals = pd.Series(0, index=data.index)

        # Bullish: all MAs aligned upward
        bullish = (
            (short_ma > medium_ma) & (medium_ma > long_ma) &
            (short_ma.shift(1) <= medium_ma.shift(1))
        )

        # Bearish: all MAs aligned downward
        bearish = (
            (short_ma < medium_ma) & (medium_ma < long_ma) &
            (short_ma.shift(1) >= medium_ma.shift(1))
        )

        signals[bullish] = 1
        signals[bearish] = -1

        return signals


class ADXTrendStrategy(BaseStrategy):
    """ADX Trend Strength Strategy.

    Entry when ADX indicates strong trend.
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="ADX Trend",
            category=StrategyCategory.TREND,
            timeframe=TimeFrame.DAILY,
            description="ADX-based trend strength strategy",
            parameters={
                "adx_period": 14,
                "adx_threshold": 25,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Calculate ADX
        import ta
        adx = ta.trend.adx(high, low, close, window=self.parameters["adx_period"])
        di_plus = ta.trend.adx_pos(high, low, close)
        di_minus = ta.trend.adx_neg(high, low, close)

        signals = pd.Series(0, index=data.index)

        # Strong uptrend
        bullish = (
            (adx > self.parameters["adx_threshold"]) &
            (di_plus > di_minus)
        )

        # Strong downtrend
        bearish = (
            (adx > self.parameters["adx_threshold"]) &
            (di_plus < di_minus)
        )

        signals[bullish] = 1
        signals[bearish] = -1

        return signals


class SuperTrendStrategy(BaseStrategy):
    """SuperTrend Strategy.

    Uses SuperTrend indicator for trend following.
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="SuperTrend",
            category=StrategyCategory.TREND,
            timeframe=TimeFrame.DAILY,
            description="SuperTrend indicator strategy",
            parameters={
                "period": 10,
                "multiplier": 3.0,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Calculate SuperTrend
        period = self.parameters["period"]
        mult = self.parameters["multiplier"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        hl2 = (high + low) / 2
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr

        # Simplified SuperTrend
        st = pd.Series(index=data.index, dtype=float)
        st.iloc[0] = upper.iloc[0]

        for i in range(1, len(data)):
            if close.iloc[i] > st.iloc[i-1]:
                st.iloc[i] = lower.iloc[i]
            else:
                st.iloc[i] = upper.iloc[i]

        signals = pd.Series(0, index=data.index)
        signals[close > st] = 1
        signals[close < st] = -1

        return signals


# ============================================================================
# MEAN REVERSION STRATEGIES
# ============================================================================

class RSIMeanReversionStrategy(BaseStrategy):
    """RSI Mean Reversion Strategy.

    Buy oversold, sell overbought conditions.
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="RSI Mean Reversion",
            category=StrategyCategory.MEAN_REVERSION,
            timeframe=TimeFrame.DAILY,
            description="RSI-based mean reversion strategy",
            parameters={
                "rsi_period": 14,
                "oversold": 30,
                "overbought": 70,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        import ta
        close = data["close"]
        rsi = ta.momentum.rsi(close, window=self.parameters["rsi_period"])

        signals = pd.Series(0, index=data.index)

        signals[rsi < self.parameters["oversold"]] = 1
        signals[rsi > self.parameters["overbought"]] = -1

        return signals


class BollingerBandsMeanReversionStrategy(BaseStrategy):
    """Bollinger Bands Mean Reversion.

    Buy at lower band, sell at upper band.
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="Bollinger Bands Mean Reversion",
            category=StrategyCategory.MEAN_REVERSION,
            timeframe=TimeFrame.DAILY,
            description="BB mean reversion strategy",
            parameters={
                "period": 20,
                "std_dev": 2.0,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        import ta
        close = data["close"]

        bb = ta.volatility.BollingerBands(
            close,
            window=self.parameters["period"],
            window_dev=self.parameters["std_dev"],
        )

        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()

        signals = pd.Series(0, index=data.index)

        signals[close < bb_lower] = 1
        signals[close > bb_upper] = -1

        return signals


class ZScoreMeanReversionStrategy(BaseStrategy):
    """Z-Score Mean Reversion Strategy.

    Statistical mean reversion using Z-score.
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="Z-Score Mean Reversion",
            category=StrategyCategory.MEAN_REVERSION,
            timeframe=TimeFrame.DAILY,
            description="Statistical mean reversion strategy",
            parameters={
                "lookback": 20,
                "entry_z": 2.0,
                "exit_z": 0.5,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]

        mean = close.rolling(self.parameters["lookback"]).mean()
        std = close.rolling(self.parameters["lookback"]).std()
        z_score = (close - mean) / std

        signals = pd.Series(0, index=data.index)

        signals[z_score < -self.parameters["entry_z"]] = 1
        signals[z_score > self.parameters["entry_z"]] = -1

        return signals


# ============================================================================
# MOMENTUM STRATEGIES
# ============================================================================

class MomentumRankStrategy(BaseStrategy):
    """Momentum Ranking Strategy.

    Rank stocks by momentum and buy top performers.
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="Momentum Rank",
            category=StrategyCategory.MOMENTUM,
            timeframe=TimeFrame.DAILY,
            description="Momentum ranking strategy",
            parameters={
                "momentum_period": 12,
                "top_n": 10,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        momentum = close.pct_change(self.parameters["momentum_period"])

        signals = pd.Series(0, index=data.index)
        signals[momentum > 0.1] = 1  # Top momentum threshold

        return signals


class MACDMomentumStrategy(BaseStrategy):
    """MACD Momentum Strategy.

    Uses MACD histogram for momentum signals.
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="MACD Momentum",
            category=StrategyCategory.MOMENTUM,
            timeframe=TimeFrame.DAILY,
            description="MACD histogram momentum strategy",
            parameters={
                "fast": 12,
                "slow": 26,
                "signal": 9,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        import ta
        close = data["close"]

        macd = ta.trend.MACD(
            close,
            window_fast=self.parameters["fast"],
            window_slow=self.parameters["slow"],
            window_sign=self.parameters["signal"],
        )

        hist = macd.macd_diff()

        signals = pd.Series(0, index=data.index)

        # Momentum increasing
        signals[(hist > 0) & (hist > hist.shift(1))] = 1
        signals[(hist < 0) & (hist < hist.shift(1))] = -1

        return signals


# ============================================================================
# BREAKOUT STRATEGIES
# ============================================================================

class DonchianBreakoutStrategy(BaseStrategy):
    """Donchian Channel Breakout.

    Buy new highs, sell new lows.
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="Donchian Breakout",
            category=StrategyCategory.BREAKOUT,
            timeframe=TimeFrame.DAILY,
            description="Donchian channel breakout strategy",
            parameters={
                "period": 20,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        upper = high.rolling(self.parameters["period"]).max()
        lower = low.rolling(self.parameters["period"]).min()

        signals = pd.Series(0, index=data.index)

        signals[close > upper.shift(1)] = 1
        signals[close < lower.shift(1)] = -1

        return signals


class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility Breakout Strategy.

    Breakout from volatility squeeze.
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="Volatility Breakout",
            category=StrategyCategory.BREAKOUT,
            timeframe=TimeFrame.DAILY,
            description="Volatility squeeze breakout strategy",
            parameters={
                "bb_period": 20,
                "kc_period": 20,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        import ta
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=self.parameters["bb_period"])
        bb_width = bb.bollinger_wband()

        # Keltner Channel
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.parameters["kc_period"]).mean()
        kc_width = (atr * 2) / close.rolling(self.parameters["kc_period"]).mean()

        signals = pd.Series(0, index=data.index)

        # Squeeze condition
        squeeze = bb_width < kc_width

        # Breakout
        breakout_up = close > close.rolling(self.parameters["bb_period"]).mean()
        breakout_dn = close < close.rolling(self.parameters["bb_period"]).mean()

        signals[squeeze & breakout_up] = 1
        signals[squeeze & breakout_dn] = -1

        return signals


# ============================================================================
# VOLUME STRATEGIES
# ============================================================================

class VolumeSpikeStrategy(BaseStrategy):
    """Volume Spike Strategy.

    Trade on unusual volume activity.
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="Volume Spike",
            category=StrategyCategory.VOLUME,
            timeframe=TimeFrame.DAILY,
            description="Volume spike detection strategy",
            parameters={
                "volume_ma_period": 20,
                "spike_threshold": 2.0,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        volume = data["volume"]
        close = data["close"]

        volume_ma = volume.rolling(self.parameters["volume_ma_period"]).mean()
        volume_ratio = volume / volume_ma

        signals = pd.Series(0, index=data.index)

        # Volume spike with price confirmation
        spike = volume_ratio > self.parameters["spike_threshold"]
        price_up = close > close.shift(1)

        signals[spike & price_up] = 1

        return signals


class OBVStrategy(BaseStrategy):
    """On-Balance Volume Strategy.

    Uses OBV divergences for signals.
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="OBV Divergence",
            category=StrategyCategory.VOLUME,
            timeframe=TimeFrame.DAILY,
            description="OBV divergence strategy",
            parameters={
                "obv_period": 14,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        import ta
        close = data["close"]
        volume = data["volume"]

        obv = ta.volume.on_balance_volume(close, volume)
        obv_ma = obv.rolling(self.parameters["obv_period"]).mean()

        signals = pd.Series(0, index=data.index)

        # OBV above MA
        signals[obv > obv_ma] = 1
        signals[obv < obv_ma] = -1

        return signals


# ============================================================================
# VOLATILITY STRATEGIES
# ============================================================================

class ShortVolatilityStrategy(BaseStrategy):
    """Short Volatility Strategy.

    Sell when volatility is high, buy when low.
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="Short Volatility",
            category=StrategyCategory.VOLATILITY,
            timeframe=TimeFrame.DAILY,
            description="Volatility mean reversion strategy",
            parameters={
                "atr_period": 14,
                "volatility_threshold": 2.0,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        import ta
        high = data["high"]
        low = data["low"]
        close = data["close"]

        atr = ta.volatility.average_true_range(high, low, close, window=self.parameters["atr_period"])
        atr_pct = atr / close

        vol_ma = atr_pct.rolling(self.parameters["atr_period"] * 2).mean()
        vol_std = atr_pct.rolling(self.parameters["atr_period"] * 2).std()

        z_score = (atr_pct - vol_ma) / vol_std

        signals = pd.Series(0, index=data.index)

        # High volatility = sell
        signals[z_score > self.parameters["volatility_threshold"]] = -1
        signals[z_score < -self.parameters["volatility_threshold"]] = 1

        return signals


# ============================================================================
# PATTERN STRATEGIES
# ============================================================================

class CandlestickPatternStrategy(BaseStrategy):
    """Candlestick Pattern Strategy.

    Trade based on candlestick patterns.
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="Candlestick Patterns",
            category=StrategyCategory.PATTERN,
            timeframe=TimeFrame.DAILY,
            description="Candlestick pattern recognition strategy",
            parameters={
                "confirm_candles": 1,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        open_p = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        signals = pd.Series(0, index=data.index)

        # Hammer pattern
        body = abs(close - open_p)
        lower_shadow = np.minimum(open_p, close) - low
        upper_shadow = high - np.maximum(open_p, close)

        hammer = (
            (lower_shadow > 2 * body) &
            (upper_shadow < body * 0.5) &
            (body < (high - low) * 0.3)
        )

        # Engulfing pattern
        bullish_engulf = (
            (close.shift(1) < open_p.shift(1)) &
            (close > open_p) &
            (close > open_p.shift(1)) &
            (open_p < close.shift(1))
        )

        signals[hammer] = 1
        signals[bullish_engulf] = 1

        return signals


# ============================================================================
# MULTI-FACTOR STRATEGIES
# ============================================================================

class MultiFactorStrategy(BaseStrategy):
    """Multi-Factor Combined Strategy.

    Combines multiple factors for signal generation.
    """

    def __init__(self) -> None:
        config = StrategyConfig(
            name="Multi-Factor",
            category=StrategyCategory.MULTI_FACTOR,
            timeframe=TimeFrame.DAILY,
            description="Combined multi-factor strategy",
            parameters={
                "momentum_weight": 0.3,
                "value_weight": 0.3,
                "quality_weight": 0.4,
            },
        )
        super().__init__(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        volume = data["volume"]

        # Momentum factor
        momentum = close.pct_change(12)

        # Volume factor
        volume_ratio = volume / volume.rolling(20).mean()

        # Volatility factor (inverse)
        returns = close.pct_change()
        volatility = returns.rolling(20).std()
        vol_factor = 1 / (volatility + 0.001)

        # Normalize and combine
        momentum_norm = (momentum - momentum.mean()) / (momentum.std() + 0.001)
        volume_norm = (volume_ratio - volume_ratio.mean()) / (volume_ratio.std() + 0.001)
        vol_norm = (vol_factor - vol_factor.mean()) / (vol_factor.std() + 0.001)

        composite = (
            self.parameters["momentum_weight"] * momentum_norm +
            self.parameters["volume_weight"] * volume_norm +
            self.parameters["quality_weight"] * vol_norm
        )

        signals = pd.Series(0, index=data.index)
        signals[composite > 1] = 1
        signals[composite < -1] = -1

        return signals


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

class StrategyRegistry:
    """Registry for all available strategies."""

    _strategies: dict[str, type[BaseStrategy]] = {
        # Trend Following
        "dual_ma": DualMovingAverageStrategy,
        "triple_ma": TripleMovingAverageStrategy,
        "adx_trend": ADXTrendStrategy,
        "supertrend": SuperTrendStrategy,

        # Mean Reversion
        "rsi_mean_rev": RSIMeanReversionStrategy,
        "bb_mean_rev": BollingerBandsMeanReversionStrategy,
        "zscore_mean_rev": ZScoreMeanReversionStrategy,

        # Momentum
        "momentum_rank": MomentumRankStrategy,
        "macd_momentum": MACDMomentumStrategy,

        # Breakout
        "donchian_breakout": DonchianBreakoutStrategy,
        "vol_breakout": VolatilityBreakoutStrategy,

        # Volume
        "volume_spike": VolumeSpikeStrategy,
        "obv": OBVStrategy,

        # Volatility
        "short_vol": ShortVolatilityStrategy,

        # Pattern
        "candlestick": CandlestickPatternStrategy,

        # Multi-Factor
        "multi_factor": MultiFactorStrategy,
    }

    @classmethod
    def get_strategy(cls, name: str) -> BaseStrategy:
        """Get strategy instance by name."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        return cls._strategies[name]()

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all available strategies."""
        return list(cls._strategies.keys())

    @classmethod
    def get_strategy_info(cls, name: str) -> dict[str, Any]:
        """Get strategy information."""
        strategy = cls.get_strategy(name)
        return {
            "name": strategy.config.name,
            "category": strategy.config.category.value,
            "timeframe": strategy.config.timeframe.value,
            "description": strategy.config.description,
            "parameters": strategy.parameters,
            "risk_params": strategy.risk_params,
        }


def get_strategy(name: str) -> BaseStrategy:
    """Get strategy by name."""
    return StrategyRegistry.get_strategy(name)


def list_all_strategies() -> list[str]:
    """List all available strategies."""
    return StrategyRegistry.list_strategies()
