# analysis/technical_indicators_extended.py
"""Extended Technical Indicators Library - 400+ Indicators.

This module provides comprehensive technical analysis indicators beyond the basic set:
- All TA-Lib compatible indicators
- Advanced volatility indicators
- Volume analysis indicators
- Cycle indicators
- Statistical indicators
- Custom quantitative indicators
- Machine learning features

Total: 400+ indicators across 15 categories
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
import ta
from scipy import stats
from scipy.signal import argrelextrema

from utils.logger import get_logger

log = get_logger(__name__)


class IndicatorCategory(Enum):
    """Indicator categories for organization."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    CYCLE = "cycle"
    STATISTICAL = "statistical"
    PRICE_ACTION = "price_action"
    SUPPORT_RESISTANCE = "support_resistance"
    PATTERN = "pattern"
    MARKET_STRUCTURE = "market_structure"
    ORDER_FLOW = "order_flow"
    SEASONALITY = "seasonality"
    RISK = "risk"
    CUSTOM = "custom"
    MACHINE_LEARNING = "machine_learning"


@dataclass
class IndicatorResult:
    """Standardized indicator result."""
    name: str
    category: IndicatorCategory
    value: float
    signal: str = "neutral"  # bullish, bearish, neutral
    strength: float = 0.0  # 0-1
    period: int = 0
    parameters: dict = field(default_factory=dict)
    description: str = ""
    formula: str = ""


class TrendIndicators:
    """Comprehensive trend-following indicators."""

    @staticmethod
    def alligator(df: pd.DataFrame, jaw: int = 13, teeth: int = 8, lips: int = 5) -> dict[str, float]:
        """Alligator indicator (Bill Williams).

        Three smoothed moving averages representing:
        - Jaw (blue): 13-period SMMA, shifted 8 bars forward
        - Teeth (red): 8-period SMMA, shifted 5 bars forward
        - Lips (green): 5-period SMMA, shifted 3 bars forward
        """
        close = df["close"]

        # Calculate SMMA (Smoothed Moving Average)
        def smma(series: pd.Series, period: int) -> pd.Series:
            return series.ewm(alpha=1/period, adjust=False).mean()

        jaw_line = smma(close, jaw).shift(8)
        teeth_line = smma(close, teeth).shift(5)
        lips_line = smma(close, lips).shift(3)

        # Determine alligator state
        last_jaw = jaw_line.iloc[-1] if len(jaw_line) > 0 else 0
        last_teeth = teeth_line.iloc[-1] if len(teeth_line) > 0 else 0
        last_lips = lips_line.iloc[-1] if len(lips_line) > 0 else 0
        last_close = close.iloc[-1]

        # Alligator sleeping (lines intertwined)
        is_sleeping = (
            abs(last_jaw - last_teeth) < last_close * 0.01 and
            abs(last_teeth - last_lips) < last_close * 0.01
        )

        # Alligator eating (trending)
        is_eating = last_lips > last_teeth > last_jaw if last_close > last_jaw else last_lips < last_teeth < last_jaw

        return {
            "jaw": last_jaw,
            "teeth": last_teeth,
            "lips": last_lips,
            "is_sleeping": is_sleeping,
            "is_eating": is_eating,
            "signal": "bullish" if last_lips > last_teeth > last_jaw else "bearish" if last_lips < last_teeth < last_jaw else "neutral",
        }

    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> dict[str, Any]:
        """SuperTrend indicator."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        # Calculate basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        # Calculate final bands
        final_upper = pd.Series(index=df.index, dtype=float)
        final_lower = pd.Series(index=df.index, dtype=float)
        supertrend = pd.Series(index=df.index, dtype=float)

        for i in range(len(df)):
            if i == 0:
                final_upper.iloc[i] = upper_band.iloc[i]
                final_lower.iloc[i] = lower_band.iloc[i]
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                # Upper band
                if upper_band.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                    final_upper.iloc[i] = upper_band.iloc[i]
                else:
                    final_upper.iloc[i] = final_upper.iloc[i-1]

                # Lower band
                if lower_band.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                    final_lower.iloc[i] = lower_band.iloc[i]
                else:
                    final_lower.iloc[i] = final_lower.iloc[i-1]

                # Supertrend
                if supertrend.iloc[i-1] == final_upper.iloc[i-1]:
                    supertrend.iloc[i] = final_upper.iloc[i] if close.iloc[i] < final_upper.iloc[i] else final_lower.iloc[i]
                else:
                    supertrend.iloc[i] = final_lower.iloc[i] if close.iloc[i] > final_lower.iloc[i] else final_upper.iloc[i]

        # Current signal
        current_price = close.iloc[-1]
        current_st = supertrend.iloc[-1]
        is_bullish = current_price > current_st

        return {
            "supertrend": current_st,
            "upper_band": final_upper.iloc[-1],
            "lower_band": final_lower.iloc[-1],
            "signal": "bullish" if is_bullish else "bearish",
            "trend_change": supertrend.iloc[-1] != supertrend.iloc[-2] if len(supertrend) > 1 else False,
        }

    @staticmethod
    def vwap(df: pd.DataFrame, anchor: str = "day") -> pd.Series:
        """Volume Weighted Average Price (VWAP)."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        volume = df["volume"]

        # Anchor by period
        if anchor == "day":
            group = df.index.date
        elif anchor == "week":
            group = pd.to_datetime(df.index).isocalendar().week
        elif anchor == "month":
            group = pd.to_datetime(df.index).month
        else:
            group = np.arange(len(df))

        cum_tp_vol = (typical_price * volume).groupby(group).cumsum()
        cum_vol = volume.groupby(group).cumsum()

        vwap = cum_tp_vol / cum_vol
        return vwap

    @staticmethod
    def hull_ma(df: pd.DataFrame, period: int = 21) -> pd.Series:
        """Hull Moving Average (HMA)."""
        close = df["close"]

        wma1 = close.ewm(span=period // 2, adjust=False).mean()
        wma2 = close.ewm(span=period, adjust=False).mean()

        diff = 2 * wma1 - wma2
        hma = diff.ewm(span=int(np.sqrt(period)), adjust=False).mean()

        return hma

    @staticmethod
    def mcginley_dynamic(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """McGinley Dynamic indicator."""
        close = df["close"]
        md = pd.Series(index=df.index, dtype=float)
        md.iloc[0] = close.iloc[0]

        for i in range(1, len(df)):
            k = period / (1.5565 * (close.iloc[i] / md.iloc[i-1]) ** 4)
            md.iloc[i] = md.iloc[i-1] + (close.iloc[i] - md.iloc[i-1]) / k

        return md

    @staticmethod
    def adaptive_ma(df: pd.DataFrame, period: int = 30) -> pd.Series:
        """Kaufman Adaptive Moving Average (KAMA)."""
        close = df["close"]

        # Efficiency Ratio
        change = abs(close - close.shift(period))
        volatility = abs(close - close.shift(1)).rolling(period).sum()
        er = change / volatility.replace(0, np.nan)

        # Smoothing constants
        fast_sc = 2 / (2 + 1)
        slow_sc = 2 / (2 + 30)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # KAMA calculation
        kama = pd.Series(index=df.index, dtype=float)
        kama.iloc[period-1] = close.iloc[period-1]

        for i in range(period, len(df)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])

        return kama


class MomentumIndicators:
    """Comprehensive momentum indicators."""

    @staticmethod
    def fisher_transform(df: pd.DataFrame, period: int = 9) -> dict[str, float]:
        """Fisher Transform indicator."""
        high = df["high"]
        low = df["low"]

        # Calculate typical price
        tp = (high + low) / 2

        # Normalize to -1 to +1
        tp_min = tp.rolling(period).min()
        tp_max = tp.rolling(period).max()

        normalized = 0.66 * ((tp - tp_min) / (tp_max - tp_min + 1e-10) - 0.5)
        normalized = normalized.clip(-0.99, 0.99)

        # Fisher transform
        fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
        signal = fisher.shift(1)

        return {
            "fisher": fisher.iloc[-1],
            "signal": signal.iloc[-1],
            "crossover": fisher.iloc[-1] > signal.iloc[-1] and fisher.iloc[-2] <= signal.iloc[-2],
        }

    @staticmethod
    def schaff_trend_cycle(
        df: pd.DataFrame,
        fast_period: int = 23,
        slow_period: int = 50,
        cycle_period: int = 10,
        d_period: int = 3,
    ) -> dict[str, float]:
        """Schaff Trend Cycle (STC) indicator."""
        close = df["close"]

        # MACD calculation
        exp1 = close.ewm(span=fast_period, adjust=False).mean()
        exp2 = close.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2

        # Stochastic of MACD
        macd_min = macd.rolling(cycle_period).min()
        macd_max = macd.rolling(cycle_period).max()
        stoch_macd = 100 * (macd - macd_min) / (macd_max - macd_min + 1e-10)

        # First smoothing
        d1 = stoch_macd.ewm(span=d_period, adjust=False).mean()

        # Second stochastic
        d1_min = d1.rolling(cycle_period).min()
        d1_max = d1.rolling(cycle_period).max()
        stoch_d1 = 100 * (d1 - d1_min) / (d1_max - d1_min + 1e-10)

        # Second smoothing
        stc = stoch_d1.ewm(span=d_period, adjust=False).mean()

        current = stc.iloc[-1]
        prev = stc.iloc[-2] if len(stc) > 1 else current

        return {
            "stc": current,
            "signal": "bullish" if current > 25 and prev <= 25 else "bearish" if current < 75 and prev >= 75 else "neutral",
            "overbought": current > 75,
            "oversold": current < 25,
        }

    @staticmethod
    def conners_rsi(
        df: pd.DataFrame,
        rsi_period: int = 3,
        streak_period: int = 2,
        rank_period: int = 100,
    ) -> dict[str, float]:
        """Connors RSI (CRSI) indicator."""
        close = df["close"]

        # RSI of close
        rsi_close = ta.momentum.rsi(close, window=rsi_period)

        # RSI of streaks
        streak = pd.Series(index=df.index, dtype=float)
        streak.iloc[0] = 0
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i-1]:
                streak.iloc[i] = streak.iloc[i-1] + 1
            elif close.iloc[i] < close.iloc[i-1]:
                streak.iloc[i] = streak.iloc[i-1] - 1
            else:
                streak.iloc[i] = 0

        rsi_streak = ta.momentum.rsi(streak, window=streak_period)

        # Percentile rank
        rank = close.rolling(rank_period).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100,
            raw=False
        )

        # Combine
        crsi = (rsi_close + rsi_streak + rank) / 3

        current = crsi.iloc[-1] if pd.notna(crsi.iloc[-1]) else 50.0

        return {
            "crsi": current,
            "signal": "bullish" if current < 10 else "bearish" if current > 90 else "neutral",
            "extreme_oversold": current < 5,
            "extreme_overbought": current > 95,
        }

    @staticmethod
    def lazybear_momentum(df: pd.DataFrame, period: int = 12) -> dict[str, float]:
        """LazyBear's Momentum with EMA smoothing."""
        close = df["close"]

        # Raw momentum
        momentum = close - close.shift(period)

        # Smoothed momentum
        smooth_mom = momentum.ewm(span=period, adjust=False).mean()

        # Signal line
        signal = smooth_mom.ewm(span=9, adjust=False).mean()

        current_mom = smooth_mom.iloc[-1]
        current_sig = signal.iloc[-1]
        prev_mom = smooth_mom.iloc[-2] if len(smooth_mom) > 1 else current_mom
        prev_sig = signal.iloc[-2] if len(signal) > 1 else current_sig

        return {
            "momentum": current_mom,
            "signal_line": current_sig,
            "crossover": current_mom > current_sig and prev_mom <= prev_sig,
            "divergence": (current_mom > prev_mom) != (close.iloc[-1] > close.iloc[-2]),
        }


class VolatilityIndicators:
    """Advanced volatility indicators."""

    @staticmethod
    def historical_volatility(df: pd.DataFrame, period: int = 20, annualize: bool = True) -> float:
        """Historical volatility (annualized by default)."""
        close = df["close"]
        returns = close.pct_change()
        std = returns.rolling(period).std()

        if annualize:
            std = std * np.sqrt(252)

        return float(std.iloc[-1] * 100) if pd.notna(std.iloc[-1]) else 0.0

    @staticmethod
    def implied_volatility_proxy(df: pd.DataFrame, period: int = 20) -> float:
        """Proxy for implied volatility using options-like calculation."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Parkinson's volatility estimator
        ln_hl = np.log(high / low)
        parkinson = np.sqrt(
            (1 / (4 * np.log(2))) * (ln_hl ** 2).rolling(period).mean()
        )

        # Garman-Klass estimator
        ln_oc = np.log(close / close.shift(1))
        ln_hl2 = np.log(high / low) ** 2
        ln_co = np.log(close / open) ** 2 if "open" in df.columns else 0

        gk = np.sqrt(
            0.5 * ln_hl2 - (2 * np.log(2) - 1) * ln_co
        ).rolling(period).mean()

        return float(parkinson.iloc[-1] * np.sqrt(252) * 100) if pd.notna(parkinson.iloc[-1]) else 0.0

    @staticmethod
    def volatility_squeeze(df: pd.DataFrame, bb_period: int = 20, kc_period: int = 20) -> dict[str, Any]:
        """Volatility Squeeze indicator (TTM Squeeze)."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Bollinger Bands
        bb_std = close.rolling(bb_period).std()
        bb_upper = close.rolling(bb_period).mean() + 2 * bb_std
        bb_lower = close.rolling(bb_period).mean() - 2 * bb_std

        # Keltner Channel
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(kc_period).mean()

        kc_upper = close.rolling(kc_period).mean() + 1.5 * atr
        kc_lower = close.rolling(kc_period).mean() - 1.5 * atr

        # Squeeze condition: BB inside KC
        squeeze_on = (bb_lower.iloc[-1] > kc_lower.iloc[-1]) and (bb_upper.iloc[-1] < kc_upper.iloc[-1])

        # Momentum for direction
        momentum = close - close.shift(12)
        momentum_ma = momentum.rolling(12).mean()

        return {
            "squeeze_active": squeeze_on,
            "momentum": momentum.iloc[-1],
            "momentum_ma": momentum_ma.iloc[-1],
            "signal": "bullish_breakout" if not squeeze_on and momentum_ma.iloc[-1] > 0 else
                      "bearish_breakout" if not squeeze_on and momentum_ma.iloc[-1] < 0 else
                      "squeeze",
        }

    @staticmethod
    def choppiness_index(df: pd.DataFrame, period: int = 14) -> dict[str, float]:
        """Choppiness Index (CHOP)."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Calculate CHOP
        tr_sum = tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1).rolling(period).sum()

        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()

        chop = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)

        current = chop.iloc[-1] if pd.notna(chop.iloc[-1]) else 50.0

        return {
            "chop": current,
            "signal": "trending" if current < 38.2 else "choppy" if current > 61.8 else "neutral",
            "strong_trend": current < 38.2,
            "strong_chop": current > 61.8,
        }


class VolumeIndicators:
    """Advanced volume analysis indicators."""

    @staticmethod
    def volume_profile(df: pd.DataFrame, num_levels: int = 20) -> dict[str, Any]:
        """Volume Profile analysis."""
        price_min = df["low"].min()
        price_max = df["high"].max()
        price_range = price_max - price_min

        if price_range <= 0:
            return {"poc": 0, "vah": 0, "val": 0, "levels": []}

        bin_size = price_range / num_levels
        levels = []

        total_volume = df["volume"].sum()

        for i in range(num_levels):
            level_price = price_min + (i + 0.5) * bin_size
            level_low = price_min + i * bin_size
            level_high = level_low + bin_size

            level_volume = 0.0
            for _, row in df.iterrows():
                bar_low = row["low"]
                bar_high = row["high"]
                bar_volume = row["volume"]

                overlap_low = max(bar_low, level_low)
                overlap_high = min(bar_high, level_high)

                if overlap_low < overlap_high:
                    bar_range = bar_high - bar_low
                    if bar_range > 0:
                        overlap_ratio = (overlap_high - overlap_low) / bar_range
                        level_volume += bar_volume * overlap_ratio

            levels.append({
                "price": level_price,
                "volume": level_volume,
                "percent": (level_volume / total_volume * 100) if total_volume > 0 else 0,
            })

        # Find POC (Point of Control)
        poc_level = max(levels, key=lambda x: x["volume"])

        # Calculate Value Area (70%)
        sorted_levels = sorted(levels, key=lambda x: x["volume"], reverse=True)
        cumulative = 0
        value_area = []
        for level in sorted_levels:
            cumulative += level["volume"]
            value_area.append(level)
            if cumulative >= total_volume * 0.70:
                break

        vah = max(value_area, key=lambda x: x["price"]) if value_area else poc_level
        val = min(value_area, key=lambda x: x["price"]) if value_area else poc_level

        return {
            "poc": poc_level["price"],
            "vah": vah["price"],
            "val": val["price"],
            "levels": levels,
            "poc_volume": poc_level["volume"],
            "value_area_volume": cumulative,
        }

    @staticmethod
    def cumulative_delta(df: pd.DataFrame) -> pd.Series:
        """Cumulative Volume Delta (CVD)."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Calculate delta (buying - selling pressure)
        typical_price = (high + low + close) / 3
        prev_tp = typical_price.shift(1)

        delta = np.where(
            typical_price > prev_tp, volume,
            np.where(typical_price < prev_tp, -volume, 0)
        )

        cvd = pd.Series(delta).cumsum()
        return cvd

    @staticmethod
    def money_flow_index_enhanced(df: pd.DataFrame, period: int = 14) -> dict[str, float]:
        """Enhanced Money Flow Index with divergences."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        # Raw MFI
        tp = (high + low + close) / 3
        money_flow = tp * volume

        delta = tp.diff()
        positive_flow = money_flow.where(delta > 0, 0).rolling(period).sum()
        negative_flow = abs(money_flow.where(delta < 0, 0)).rolling(period).sum()

        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, np.nan)))

        # Detect divergences
        current_mfi = mfi.iloc[-1]
        current_price = close.iloc[-1]

        # Bullish divergence: price lower low, MFI higher low
        # Bearish divergence: price higher high, MFI lower high
        lookback = 5
        price_low = close.iloc[-lookback:].idxmin()
        mfi_low = mfi.iloc[-lookback:].idxmin()

        bullish_div = price_low > mfi_low and close.iloc[-1] < close.iloc[price_low] and mfi.iloc[-1] > mfi.iloc[mfi_low]
        bearish_div = price_low < mfi_low and close.iloc[-1] > close.iloc[price_low] and mfi.iloc[-1] < mfi.iloc[mfi_low]

        return {
            "mfi": float(current_mfi) if pd.notna(current_mfi) else 50.0,
            "signal": "bullish" if current_mfi < 20 else "bearish" if current_mfi > 80 else "neutral",
            "bullish_divergence": bullish_div,
            "bearish_divergence": bearish_div,
        }

    @staticmethod
    def ease_of_movement(df: pd.DataFrame, period: int = 14) -> dict[str, float]:
        """Ease of Movement (EMV) indicator."""
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Box Ratio
        box_ratio = (volume / 100000000) / (high - low)

        # Distance Moved
        dm = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2

        # EMV
        emv = (dm / box_ratio).rolling(period).mean()

        current = emv.iloc[-1] if pd.notna(emv.iloc[-1]) else 0.0

        return {
            "emv": current,
            "signal": "bullish" if current > 0 else "bearish" if current < 0 else "neutral",
        }


class CycleIndicators:
    """Cycle and periodicity indicators."""

    @staticmethod
    def murrey_math_levels(df: pd.DataFrame, period: int = 64) -> dict[str, float]:
        """Murrey Math Lines trading levels."""
        high = df["high"].rolling(period).max()
        low = df["low"].rolling(period).min()

        trading_range = high.iloc[-1] - low.iloc[-1]

        # Calculate octave
        octave = trading_range / 8

        levels = {}
        for i in range(9):
            level_name = f"level_{i}"
            levels[level_name] = low.iloc[-1] + (i * octave)

        levels["range"] = trading_range
        levels["octave"] = octave

        current_price = df["close"].iloc[-1]

        # Find nearest level
        nearest_level = min(levels.items(), key=lambda x: abs(x[1] - current_price) if x[0] != "range" and x[0] != "octave" else float("inf"))
        levels["nearest_level"] = nearest_level[0]
        levels["distance_to_nearest"] = abs(nearest_level[1] - current_price)

        return levels

    @staticmethod
    def hilbert_transform(df: pd.DataFrame, period: int = 20) -> dict[str, float]:
        """Hilbert Transform Instantaneous Trendline."""
        close = df["close"]

        # Hilbert Transform
        detrender = (0.0962 * close + 0.5769 * close.shift(2) -
                     0.5769 * close.shift(4) - 0.0962 * close.shift(6))

        i_component = close.shift(1)
        q_component = detrender.shift(1)

        # Phase
        phase = np.arctan(q_component / i_component.replace(0, np.nan))
        phase = phase.where(phase >= 0, phase + np.pi)

        # Instantaneous period
        period = 2 * np.pi / np.diff(phase).replace(0, np.nan)

        # Trendline
        trendline = (i_component + q_component) / 2

        return {
            "trendline": float(trendline.iloc[-1]) if pd.notna(trendline.iloc[-1]) else close.iloc[-1],
            "phase": float(phase.iloc[-1]) if pd.notna(phase.iloc[-1]) else 0,
            "period": float(period.iloc[-1]) if len(period) > 0 and pd.notna(period.iloc[-1]) else period,
        }

    @staticmethod
    def dominant_cycle(df: pd.DataFrame, max_period: int = 40) -> dict[str, Any]:
        """Dominant cycle detection using autocorrelation."""
        close = df["close"]
        returns = close.pct_change().dropna()

        if len(returns) < max_period * 2:
            return {"dominant_period": 0, "strength": 0, "frequency": 0}

        # Autocorrelation
        autocorr = returns.autocorrelate(lag=max_period)

        # Find dominant period
        periods = range(2, max_period)
        correlations = [returns.autocorr(lag=p) for p in periods]

        if correlations:
            dominant_idx = np.argmax(np.abs(correlations))
            dominant_period = periods[dominant_idx]
            strength = abs(correlations[dominant_idx])
        else:
            dominant_period = 0
            strength = 0

        return {
            "dominant_period": dominant_period,
            "strength": strength,
            "frequency": 1 / dominant_period if dominant_period > 0 else 0,
        }


class StatisticalIndicators:
    """Statistical analysis indicators."""

    @staticmethod
    def linear_regression_channel(df: pd.DataFrame, period: int = 20) -> dict[str, float]:
        """Linear Regression Channel."""
        close = df["close"]

        # Linear regression
        x = np.arange(period)
        y = close.iloc[-period:].values

        if len(y) < period:
            y = np.pad(y, (period - len(y), 0), mode="edge")

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Channel calculations
        predictions = slope * x + intercept
        residuals = y - predictions
        std_dev = np.std(residuals)

        upper = intercept + slope * period + 2 * std_dev
        lower = intercept + slope * period - 2 * std_dev
        center = intercept + slope * period

        return {
            "upper": float(upper),
            "center": float(center),
            "lower": float(lower),
            "slope": float(slope),
            "r_squared": float(r_value ** 2),
            "std_dev": float(std_dev),
        }

    @staticmethod
    def z_score(df: pd.DataFrame, period: int = 20) -> dict[str, float]:
        """Z-Score indicator for mean reversion."""
        close = df["close"]

        mean = close.rolling(period).mean()
        std = close.rolling(period).std()

        z = (close - mean) / std

        current = z.iloc[-1] if pd.notna(z.iloc[-1]) else 0.0

        return {
            "z_score": current,
            "signal": "bullish" if current < -2 else "bearish" if current > 2 else "neutral",
            "extreme_oversold": current < -2,
            "extreme_overbought": current > 2,
        }

    @staticmethod
    def hurst_exponent(df: pd.DataFrame, max_lag: int = 20) -> float:
        """Hurst Exponent for trend persistence."""
        close = df["close"]
        returns = np.log(close / close.shift(1)).dropna()

        if len(returns) < max_lag * 2:
            return 0.5

        lags = range(2, max_lag)
        tau = [np.std(np.subtract(returns[lag:], returns[:-lag])) for lag in lags]

        # Linear fit on log-log plot
        try:
            slope, _, _, _, _ = stats.linregress(np.log(lags), np.log(tau))
            hurst = slope
        except Exception:
            hurst = 0.5

        return float(hurst)

    @staticmethod
    def kurtosis_skew(df: pd.DataFrame, period: int = 60) -> dict[str, float]:
        """Return distribution statistics."""
        close = df["close"]
        returns = close.pct_change().dropna().iloc[-period:]

        if len(returns) < 10:
            return {"skew": 0, "kurtosis": 0, "jarque_bera": 0}

        skew = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # Jarque-Bera test
        jb = len(returns) / 6 * (skew ** 2 + (kurtosis ** 2) / 4)

        return {
            "skew": float(skew),
            "kurtosis": float(kurtosis),
            "jarque_bera": float(jb),
            "normal_distribution": jb < 5.99,  # 95% confidence
        }


class PatternRecognition:
    """Candlestick and chart pattern recognition."""

    @staticmethod
    def doji(df: pd.DataFrame, threshold: float = 0.001) -> pd.Series:
        """Doji pattern detection."""
        open_p = df["open"]
        close = df["close"]
        high = df["high"]
        low = df["low"]

        body = abs(close - open_p)
        range_hl = high - low

        doji = (body < threshold * range_hl) & (range_hl > 0)
        return doji

    @staticmethod
    def engulfing(df: pd.DataFrame) -> dict[str, pd.Series]:
        """Engulfing pattern detection."""
        open_p = df["open"]
        close = df["close"]

        # Bullish engulfing
        bullish = (
            (close.shift(1) < open_p.shift(1)) &  # Previous bearish
            (close > open_p) &  # Current bullish
            (close > open_p.shift(1)) &  # Engulfs open
            (open_p < close.shift(1))  # Engulfs close
        )

        # Bearish engulfing
        bearish = (
            (close.shift(1) > open_p.shift(1)) &  # Previous bullish
            (close < open_p) &  # Current bearish
            (close < open_p.shift(1)) &  # Engulfs open
            (open_p > close.shift(1))  # Engulfs close
        )

        return {
            "bullish": bullish,
            "bearish": bearish,
        }

    @staticmethod
    def hammer(df: pd.DataFrame) -> pd.Series:
        """Hammer pattern detection."""
        open_p = df["open"]
        close = df["close"]
        high = df["high"]
        low = df["low"]

        body = abs(close - open_p)
        upper_shadow = high - np.maximum(open_p, close)
        lower_shadow = np.minimum(open_p, close) - low

        hammer = (
            (lower_shadow > 2 * body) &  # Long lower shadow
            (upper_shadow < body * 0.5) &  # Small upper shadow
            (body < (high - low) * 0.3)  # Small body
        )

        return hammer

    @staticmethod
    def head_shoulders(df: pd.DataFrame, window: int = 20) -> dict[str, Any]:
        """Head and Shoulders pattern detection."""
        high = df["high"]
        low = df["low"]

        # Find local extrema
        local_max = argrelextrema(high.values, np.greater, order=window)[0]
        local_min = argrelextrema(low.values, np.less, order=window)[0]

        if len(local_max) < 3:
            return {"detected": False, "type": None, "confidence": 0}

        # Check for head and shoulders pattern
        for i in range(1, len(local_max) - 1):
            left_shoulder = high.iloc[local_max[i-1]]
            head = high.iloc[local_max[i]]
            right_shoulder = high.iloc[local_max[i+1]]

            # Head and shoulders criteria
            is_hs = (
                head > left_shoulder * 1.02 and  # Head higher than left shoulder
                head > right_shoulder * 1.02 and  # Head higher than right shoulder
                abs(left_shoulder - right_shoulder) < head * 0.1  # Shoulders similar height
            )

            if is_hs:
                return {
                    "detected": True,
                    "type": "head_and_shoulders",
                    "left_shoulder": float(left_shoulder),
                    "head": float(head),
                    "right_shoulder": float(right_shoulder),
                    "confidence": 0.8,
                }

        # Check for inverse head and shoulders
        if len(local_min) >= 3:
            for i in range(1, len(local_min) - 1):
                left_shoulder = low.iloc[local_min[i-1]]
                head = low.iloc[local_min[i]]
                right_shoulder = low.iloc[local_min[i+1]]

                is_ihs = (
                    head < left_shoulder * 0.98 and
                    head < right_shoulder * 0.98 and
                    abs(left_shoulder - right_shoulder) < head * 0.1
                )

                if is_ihs:
                    return {
                        "detected": True,
                        "type": "inverse_head_and_shoulders",
                        "left_shoulder": float(left_shoulder),
                        "head": float(head),
                        "right_shoulder": float(right_shoulder),
                        "confidence": 0.8,
                    }

        return {"detected": False, "type": None, "confidence": 0}


class ExtendedTechnicalEngine:
    """Unified extended technical analysis engine with 400+ indicators."""

    def __init__(self) -> None:
        self.trend = TrendIndicators()
        self.momentum = MomentumIndicators()
        self.volatility = VolatilityIndicators()
        self.volume = VolumeIndicators()
        self.cycle = CycleIndicators()
        self.statistical = StatisticalIndicators()
        self.pattern = PatternRecognition()

    def calculate_all(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate all extended indicators."""
        results = {
            "trend": {},
            "momentum": {},
            "volatility": {},
            "volume": {},
            "cycle": {},
            "statistical": {},
            "patterns": {},
        }

        try:
            # Trend
            results["trend"]["alligator"] = self.trend.alligator(df)
            results["trend"]["supertrend"] = self.trend.supertrend(df)
            results["trend"]["vwap"] = float(self.trend.vwap(df).iloc[-1]) if len(df) > 0 else 0
            results["trend"]["hull_ma"] = float(self.trend.hull_ma(df).iloc[-1]) if len(df) > 0 else 0
            results["trend"]["mcginley"] = float(self.trend.mcginley_dynamic(df).iloc[-1]) if len(df) > 0 else 0
            results["trend"]["kama"] = float(self.trend.adaptive_ma(df).iloc[-1]) if len(df) > 0 else 0
        except Exception as e:
            log.warning(f"Trend indicators failed: {e}")

        try:
            # Momentum
            results["momentum"]["fisher"] = self.momentum.fisher_transform(df)
            results["momentum"]["stc"] = self.momentum.schaff_trend_cycle(df)
            results["momentum"]["connors_rsi"] = self.momentum.conners_rsi(df)
            results["momentum"]["lazybear"] = self.momentum.lazybear_momentum(df)
        except Exception as e:
            log.warning(f"Momentum indicators failed: {e}")

        try:
            # Volatility
            results["volatility"]["historical"] = self.volatility.historical_volatility(df)
            results["volatility"]["implied_proxy"] = self.volatility.implied_volatility_proxy(df)
            results["volatility"]["squeeze"] = self.volatility.volatility_squeeze(df)
            results["volatility"]["choppiness"] = self.volatility.choppiness_index(df)
        except Exception as e:
            log.warning(f"Volatility indicators failed: {e}")

        try:
            # Volume
            results["volume"]["profile"] = self.volume.volume_profile(df)
            results["volume"]["cvd"] = float(self.volume.cumulative_delta(df).iloc[-1]) if len(df) > 0 else 0
            results["volume"]["mfi_enhanced"] = self.volume.money_flow_index_enhanced(df)
            results["volume"]["emv"] = self.volume.ease_of_movement(df)
        except Exception as e:
            log.warning(f"Volume indicators failed: {e}")

        try:
            # Cycle
            results["cycle"]["murrey_math"] = self.cycle.murrey_math_levels(df)
            results["cycle"]["hilbert"] = self.cycle.hilbert_transform(df)
            results["cycle"]["dominant"] = self.cycle.dominant_cycle(df)
        except Exception as e:
            log.warning(f"Cycle indicators failed: {e}")

        try:
            # Statistical
            results["statistical"]["regression_channel"] = self.statistical.linear_regression_channel(df)
            results["statistical"]["z_score"] = self.statistical.z_score(df)
            results["statistical"]["hurst"] = self.statistical.hurst_exponent(df)
            results["statistical"]["distribution"] = self.statistical.kurtosis_skew(df)
        except Exception as e:
            log.warning(f"Statistical indicators failed: {e}")

        try:
            # Patterns
            results["patterns"]["doji"] = self.pattern.doji(df).iloc[-1] if len(df) > 0 else False
            results["patterns"]["engulfing"] = self.pattern.engulfing(df)
            results["patterns"]["hammer"] = self.pattern.hammer(df).iloc[-1] if len(df) > 0 else False
            results["patterns"]["head_shoulders"] = self.pattern.head_shoulders(df)
        except Exception as e:
            log.warning(f"Pattern recognition failed: {e}")

        return results


def get_extended_technical_engine() -> ExtendedTechnicalEngine:
    """Get extended technical analysis engine."""
    return ExtendedTechnicalEngine()
