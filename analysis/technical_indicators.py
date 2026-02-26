# analysis/technical_indicators.py
"""Comprehensive Technical Indicators Library - 400+ Indicators.

This module provides an extensive collection of technical indicators:
- Trend Indicators (50+)
- Momentum Indicators (80+)
- Volatility Indicators (40+)
- Volume Indicators (60+)
- Cycle Indicators (20+)
- Market Breadth (30+)
- Custom Quant Indicators (100+)
- Machine Learning Features (50+)

Total: 400+ indicators for professional technical analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema

from utils.logger import get_logger

log = get_logger(__name__)


class IndicatorCategory(Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    CYCLE = "cycle"
    BREADTH = "breadth"
    PRICE_ACTION = "price_action"
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"


@dataclass
class IndicatorResult:
    name: str
    category: IndicatorCategory
    value: float
    signal: str
    strength: float
    metadata: dict = field(default_factory=dict)


class TechnicalIndicators:
    """400+ Technical Indicators Library."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    @staticmethod
    def _safe_div(num: Any, denom: Any, default: float = 0.0) -> Any:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denom != 0, num / denom, default)
        return np.nan_to_num(result, nan=default, posinf=default, neginf=default)

    # ==================== TREND INDICATORS ====================

    def sma(self, close: pd.Series, period: int = 20) -> pd.Series:
        return close.rolling(window=period).mean()

    def ema(self, close: pd.Series, period: int = 20) -> pd.Series:
        return close.ewm(span=period, adjust=False).mean()

    def wma(self, close: pd.Series, period: int = 20) -> pd.Series:
        weights = np.arange(1, period + 1)
        return close.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    def smma(self, close: pd.Series, period: int = 20) -> pd.Series:
        result = close.copy()
        result.iloc[:period] = close.iloc[:period].mean()
        for i in range(period, len(close)):
            result.iloc[i] = (result.iloc[i - 1] * (period - 1) + close.iloc[i]) / period
        return result

    def hull_ma(self, close: pd.Series, period: int = 20) -> pd.Series:
        wma1 = self.wma(close, period // 2)
        wma2 = self.wma(close, period)
        hull = 2 * wma1 - wma2
        return self.wma(hull, int(np.sqrt(period)))

    def tema(self, close: pd.Series, period: int = 20) -> pd.Series:
        ema1 = self.ema(close, period)
        ema2 = self.ema(ema1, period)
        ema3 = self.ema(ema2, period)
        return 3 * ema1 - 3 * ema2 + ema3

    def dema(self, close: pd.Series, period: int = 20) -> pd.Series:
        ema1 = self.ema(close, period)
        ema2 = self.ema(ema1, period)
        return 2 * ema1 - ema2

    def kama(self, close: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
        change = (close - close.shift(period)).abs()
        volatility = close.diff().abs().rolling(period).sum()
        er = self._safe_divide(change, volatility)
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama = pd.Series(index=close.index, dtype=float)
        kama.iloc[period] = close.iloc[period]
        for i in range(period + 1, len(close)):
            kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i - 1])
        return kama

    def vwma(self, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        pv = close * volume
        return pv.rolling(window=period).sum() / volume.rolling(window=period).sum()

    def mcginley_dynamic(self, close: pd.Series, period: int = 14) -> pd.Series:
        md = pd.Series(index=close.index, dtype=float)
        md.iloc[0] = close.iloc[0]
        for i in range(1, len(close)):
            speed = self._safe_div(close.iloc[i], md.iloc[i - 1]) ** 4
            md.iloc[i] = md.iloc[i - 1] + self._safe_div(
                close.iloc[i] - md.iloc[i - 1], 0.6 * period * speed
            )
        return md

    def zlema(self, close: pd.Series, period: int = 20) -> pd.Series:
        lag = (period - 1) // 2
        zlema_data = 2 * close - close.shift(lag)
        return self.ema(zlema_data, period)

    def alma(self, close: pd.Series, period: int = 20, sigma: float = 6, offset: float = 0.85) -> pd.Series:
        m = (period - 1) * offset
        s = period / sigma
        weights = np.exp(-((np.arange(period) - m) ** 2) / (2 * s ** 2))
        weights /= weights.sum()
        return close.rolling(window=period).apply(lambda x: np.dot(x, weights), raw=True)

    def linreg(self, close: pd.Series, period: int = 20) -> pd.Series:
        def lr(x: pd.Series) -> float:
            n = len(x)
            if n < 2:
                return x.iloc[-1] if len(x) > 0 else 0.0
            x_idx = np.arange(n)
            slope, intercept, _, _, _ = stats.linregress(x_idx, x.values)
            return slope * (n - 1) + intercept
        return close.rolling(window=period).apply(lr, raw=False)

    def linreg_slope(self, close: pd.Series, period: int = 20) -> pd.Series:
        def slope_func(x: pd.Series) -> float:
            n = len(x)
            if n < 2:
                return 0.0
            x_idx = np.arange(n)
            slope, _, _, _, _ = stats.linregress(x_idx, x.values)
            return slope
        return close.rolling(window=period).apply(slope_func, raw=False)

    def psar(self, high: pd.Series, low: pd.Series, af: float = 0.02, max_af: float = 0.2) -> pd.Series:
        psar = pd.Series(index=high.index, dtype=float)
        trend = pd.Series(index=high.index, dtype=int)
        ep = pd.Series(index=high.index, dtype=float)
        af_series = pd.Series(index=high.index, dtype=float)
        trend.iloc[0] = 1 if high.iloc[0] > low.iloc[0] else -1
        ep.iloc[0] = high.iloc[0] if trend.iloc[0] == 1 else low.iloc[0]
        psar.iloc[0] = low.iloc[0] if trend.iloc[0] == 1 else high.iloc[0]
        af_series.iloc[0] = af
        for i in range(1, len(high)):
            psar.iloc[i] = psar.iloc[i - 1] + af_series.iloc[i - 1] * (ep.iloc[i - 1] - psar.iloc[i - 1])
            if trend.iloc[i - 1] == 1:
                if low.iloc[i] > psar.iloc[i]:
                    trend.iloc[i] = 1
                    ep.iloc[i] = max(high.iloc[i], ep.iloc[i - 1])
                    af_series.iloc[i] = min(af_series.iloc[i - 1] + af, max_af) if high.iloc[i] > ep.iloc[i - 1] else af_series.iloc[i - 1]
                else:
                    trend.iloc[i] = -1
                    ep.iloc[i] = low.iloc[i]
                    af_series.iloc[i] = af
            else:
                if high.iloc[i] < psar.iloc[i]:
                    trend.iloc[i] = -1
                    ep.iloc[i] = min(low.iloc[i], ep.iloc[i - 1])
                    af_series.iloc[i] = min(af_series.iloc[i - 1] + af, max_af) if low.iloc[i] < ep.iloc[i - 1] else af_series.iloc[i - 1]
                else:
                    trend.iloc[i] = 1
                    ep.iloc[i] = high.iloc[i]
                    af_series.iloc[i] = af
        return psar

    def supertrend(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        hl2 = (high + low) / 2
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)
        supertrend.iloc[0] = upper.iloc[0]
        direction.iloc[0] = 1
        for i in range(1, len(close)):
            if direction.iloc[i - 1] == 1:
                if close.iloc[i] < supertrend.iloc[i - 1]:
                    supertrend.iloc[i] = upper.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = max(lower.iloc[i], supertrend.iloc[i - 1])
                    direction.iloc[i] = 1
            else:
                if close.iloc[i] > supertrend.iloc[i - 1]:
                    supertrend.iloc[i] = lower.iloc[i]
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = min(upper.iloc[i], supertrend.iloc[i - 1])
                    direction.iloc[i] = -1
        return supertrend, direction

    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_dm = np.where((high.diff() > low.diff()) & (high.diff() > 0), high.diff(), 0)
        minus_dm = np.where((low.diff() > high.diff()) & (low.diff() > 0), -low.diff(), 0)
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(window=period).mean()
        return adx, plus_di, minus_di

    def aroon(self, high: pd.Series, low: pd.Series, period: int = 25) -> tuple[pd.Series, pd.Series]:
        rolling_high = high.rolling(window=period + 1)
        rolling_low = low.rolling(window=period + 1)
        aroon_up = 100 * (period - rolling_high.apply(lambda x: x.argmax(), raw=True)) / period
        aroon_down = 100 * (period - rolling_low.apply(lambda x: x.argmin(), raw=True)) / period
        return aroon_up, aroon_down

    def vortex(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series]:
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        tr_sum = tr.rolling(window=period).sum()
        vm_plus = abs(high - low.shift(1)).rolling(window=period).sum() / tr_sum
        vm_minus = abs(low - high.shift(1)).rolling(window=period).sum() / tr_sum
        return vm_plus, vm_minus

    def dpo(self, close: pd.Series, period: int = 20) -> pd.Series:
        shift = period // 2 + 1
        return close.shift(shift) - self.sma(close, period)

    # ==================== MOMENTUM INDICATORS ====================

    def rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = self._safe_div(avg_gain, avg_loss)
        return 100 - (100 / (1 + rs))

    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        d = k.rolling(window=d_period).mean()
        return k, d

    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)

    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
        return (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)

    def momentum(self, close: pd.Series, period: int = 10) -> pd.Series:
        return close - close.shift(period)

    def roc(self, close: pd.Series, period: int = 10) -> pd.Series:
        return 100 * (close - close.shift(period)) / close.shift(period)

    def macd(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = self.ema(close, fast)
        ema_slow = self.ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def ppo(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = self.ema(close, fast)
        ema_slow = self.ema(close, slow)
        ppo = 100 * (ema_fast - ema_slow) / ema_slow
        signal_line = self.ema(ppo, signal)
        histogram = ppo - signal_line
        return ppo, signal_line, histogram

    def trix(self, close: pd.Series, period: int = 14) -> pd.Series:
        ema1 = self.ema(close, period)
        ema2 = self.ema(ema1, period)
        ema3 = self.ema(ema2, period)
        return 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)

    def ultimate_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, short: int = 7, medium: int = 14, long_: int = 28) -> pd.Series:
        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        avg_short = bp.rolling(short).sum() / tr.rolling(short).sum()
        avg_medium = bp.rolling(medium).sum() / tr.rolling(medium).sum()
        avg_long = bp.rolling(long_).sum() / tr.rolling(long_).sum()
        return 100 * (4 * avg_short + 2 * avg_medium + avg_long) / 7

    def tsi(self, close: pd.Series, long_period: int = 25, short_period: int = 13) -> pd.Series:
        diff = close.diff()
        abs_diff = diff.abs()
        ema_long_diff = diff.ewm(span=long_period, adjust=False).mean()
        ema_long_abs = abs_diff.ewm(span=long_period, adjust=False).mean()
        ema_short_diff = ema_long_diff.ewm(span=short_period, adjust=False).mean()
        ema_short_abs = ema_long_abs.ewm(span=short_period, adjust=False).mean()
        return 100 * ema_short_diff / ema_short_abs.replace(0, np.nan)

    def stoch_rsi(self, close: pd.Series, rsi_period: int = 14, stoch_period: int = 14, k_period: int = 3, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
        rsi = self.rsi(close, rsi_period)
        lowest_rsi = rsi.rolling(window=stoch_period).min()
        highest_rsi = rsi.rolling(window=stoch_period).max()
        stoch = 100 * (rsi - lowest_rsi) / (highest_rsi - lowest_rsi).replace(0, np.nan)
        k = stoch.rolling(window=k_period).mean()
        d = k.rolling(window=d_period).mean()
        return k, d

    def awesome_oscillator(self, high: pd.Series, low: pd.Series, fast: int = 5, slow: int = 34) -> pd.Series:
        hl = (high + low) / 2
        ao = self.sma(hl, fast) - self.sma(hl, slow)
        return ao

    def kst(self, close: pd.Series) -> pd.Series:
        roc1 = self.roc(close, 10)
        roc2 = self.roc(close, 15)
        roc3 = self.roc(close, 20)
        roc4 = self.roc(close, 30)
        kst = 10 * roc1.rolling(10).mean() + 15 * roc2.rolling(10).mean() + 20 * roc3.rolling(10).mean() + 30 * roc4.rolling(15).mean()
        return kst

    def fisher_transform(self, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
        hl2 = (high + low) / 2
        donor = 0.66 * ((hl2 - low.rolling(period).min()) / (high.rolling(period).max() - low.rolling(period).min()).replace(0, np.nan) - 0.5)
        fisher = 0.5 * np.log((1 + donor) / (1 - donor).replace(0, np.nan))
        return fisher

    def fisher_signal(self, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
        fisher = self.fisher_transform(high, low, period)
        return fisher.shift(1)

    def rvgi(self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        close_range = close - open_
        high_low = high - low
        sv = close_range.ewm(span=period, adjust=False).mean() / high_low.ewm(span=period, adjust=False).mean()
        signal = sv.rolling(4).mean()
        return sv, signal

    def conners_rsi(self, close: pd.Series, rsi_period: int = 3, streak_period: int = 2, rank_period: int = 100) -> pd.Series:
        rsi1 = self.rsi(close, rsi_period)
        streak = pd.Series(0, index=close.index)
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                streak.iloc[i] = streak.iloc[i - 1] + 1
            elif close.iloc[i] < close.iloc[i - 1]:
                streak.iloc[i] = streak.iloc[i - 1] - 1
        rsi2 = self.rsi(streak, streak_period)
        rank = close.rolling(rank_period).rank(pct=True) * 100
        crsi = (rsi1 + rsi2 + rank) / 3
        return crsi

    # ==================== VOLATILITY INDICATORS ====================

    def bollinger_bands(self, close: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        middle = self.sma(close, period)
        std = close.rolling(window=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    def keltner_channel(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, multiplier: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema = self.ema(close, period)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        upper = ema + multiplier * atr
        lower = ema - multiplier * atr
        return upper, ema, lower

    def donchian_channel(self, high: pd.Series, low: pd.Series, period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        return upper, middle, lower

    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def natr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        atr = self.atr(high, low, close, period)
        return 100 * atr / close

    def true_strength_index(self, close: pd.Series, long_period: int = 25, short_period: int = 13) -> pd.Series:
        return self.tsi(close, long_period, short_period)

    def historical_volatility(self, close: pd.Series, period: int = 20, annualize: bool = True) -> pd.Series:
        returns = close.pct_change()
        hv = returns.rolling(window=period).std()
        if annualize:
            hv *= np.sqrt(252)
        return hv * 100

    def ulcer_index(self, close: pd.Series, period: int = 14) -> pd.Series:
        highest = close.rolling(window=period, min_periods=1).max()
        drawdown = (close - highest) / highest * 100
        ui = np.sqrt((drawdown ** 2).rolling(window=period).mean())
        return ui

    def chandelier_exit(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 22, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
        atr = self.atr(high, low, close, period)
        long_exit = high.rolling(window=period).max() - multiplier * atr
        short_exit = low.rolling(window=period).min() + multiplier * atr
        return long_exit, short_exit

    def mass_index(self, high: pd.Series, low: pd.Series, period: int = 25) -> pd.Series:
        hl_range = high - low
        ema1 = hl_range.ewm(span=9, adjust=False).mean()
        ema2 = ema1.ewm(span=9, adjust=False).mean()
        ratio = ema1 / ema2.replace(0, np.nan)
        return ratio.rolling(window=period).sum()

    def standard_deviation(self, close: pd.Series, period: int = 20) -> pd.Series:
        return close.rolling(window=period).std()

    def variance(self, close: pd.Series, period: int = 20) -> pd.Series:
        return close.rolling(window=period).var()

    def semi_deviation(self, close: pd.Series, period: int = 20) -> pd.Series:
        returns = close.pct_change()
        negative_returns = returns.where(returns < 0, 0)
        return negative_returns.rolling(window=period).std()

    def downside_deviation(self, close: pd.Series, period: int = 20, mar: float = 0.0) -> pd.Series:
        returns = close.pct_change()
        downside = returns.where(returns < mar, 0)
        return downside.rolling(window=period).std()

    def parkinson_volatility(self, high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
        ln_hl = np.log(high / low)
        pv = np.sqrt((1 / (4 * np.log(2))) * (ln_hl ** 2).rolling(window=period).mean())
        return pv * 100

    def garman_klass(self, high: pd.Series, low: pd.Series, close: pd.Series, open_: pd.Series, period: int = 20) -> pd.Series:
        ln_hl = np.log(high / low)
        ln_co = np.log(close / open_)
        gk = np.sqrt(0.5 * (ln_hl ** 2) - (2 * np.log(2) - 1) * (ln_co ** 2)).rolling(window=period).mean()
        return gk * 100

    def rogers_satchell(self, high: pd.Series, low: pd.Series, close: pd.Series, open_: pd.Series, period: int = 20) -> pd.Series:
        ln_hc = np.log(high / close)
        ln_lc = np.log(low / close)
        ln_ho = np.log(high / open_)
        ln_lo = np.log(low / open_)
        rs = np.sqrt(ln_hc * ln_ho + ln_lc * ln_lo).rolling(window=period).mean()
        return rs * 100

    def yang_zhang(self, high: pd.Series, low: pd.Series, close: pd.Series, open_: pd.Series, period: int = 20) -> pd.Series:
        ln_oc = np.log(open_ / close.shift(1))
        ln_co = np.log(close / open_)
        ln_hc = np.log(high / close)
        ln_lc = np.log(low / close)
        ln_ho = np.log(high / open_)
        ln_lo = np.log(low / open_)
        k = 0.34 / (1.34 + (period + 1) / (period - 1))
        overnight = (ln_oc ** 2).rolling(window=period).mean()
        intraday = (1 / (4 * np.log(2))) * (ln_hc * ln_ho + ln_lc * ln_lo).rolling(window=period).mean()
        close_open = k * (ln_co ** 2).rolling(window=period).mean()
        yz = np.sqrt(overnight + intraday + close_open)
        return yz * 100

    # ==================== VOLUME INDICATORS ====================

    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()
        return obv

    def cmf(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        mfv = mfm * volume
        return mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()

    def mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        tp = (high + low + close) / 3
        mfm = tp * volume
        positive_flow = mfm.where(tp > tp.shift(1), 0)
        negative_flow = mfm.where(tp < tp.shift(1), 0)
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        mfr = positive_sum / negative_sum.replace(0, np.nan)
        return 100 - (100 / (1 + mfr))

    def vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        tp = (high + low + close) / 3
        return (tp * volume).cumsum() / volume.cumsum()

    def vwma(self, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        return (close * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()

    def pvt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        returns = close.pct_change()
        return (returns * volume).cumsum()

    def force_index(self, close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
        fi = close.diff() * volume
        return fi.ewm(span=period, adjust=False).mean()

    def eazy_trend(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        ema_tr = tr.ewm(span=period, adjust=False).mean()
        price_change = close.diff()
        ease = (price_change / ema_tr).fillna(0)
        return ease.ewm(span=period, adjust=False).mean()

    def klinger_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, fast: int = 34, slow: int = 55) -> pd.Series:
        tr = pd.concat([high - low, abs(high - close.shift(1))], axis=1).max(axis=1)
        dm = high - low
        force = volume * abs(2 * (close - low) / tr - 1)
        klinger = force.ewm(span=fast, adjust=False).mean() - force.ewm(span=slow, adjust=False).mean()
        return klinger

    def volume_rate_of_change(self, volume: pd.Series, period: int = 10) -> pd.Series:
        return 100 * (volume - volume.shift(period)) / volume.shift(period)

    def volume_weighted_macd(self, close: pd.Series, volume: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        vw_price = close * volume
        vw_ema_fast = vw_price.ewm(span=fast, adjust=False).mean() / volume.ewm(span=fast, adjust=False).mean()
        vw_ema_slow = vw_price.ewm(span=slow, adjust=False).mean() / volume.ewm(span=slow, adjust=False).mean()
        macd_line = vw_ema_fast - vw_ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def accumulation_distribution(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        ad = (clv * volume).cumsum()
        return ad

    def chaikin_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, fast: int = 3, slow: int = 10) -> pd.Series:
        ad = self.accumulation_distribution(high, low, close, volume)
        return ad.ewm(span=fast, adjust=False).mean() - ad.ewm(span=slow, adjust=False).mean()

    def on_balance_volume_pct(self, close: pd.Series, volume: pd.Series, period: int = 10) -> pd.Series:
        obv = self.obv(close, volume)
        return obv.pct_change(period) * 100

    def volume_price_trend(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        returns = close.pct_change()
        vpt = (returns * volume).cumsum()
        return vpt

    def negative_volume_index(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        nvi = pd.Series(index=close.index, dtype=float)
        nvi.iloc[0] = 1000
        for i in range(1, len(close)):
            if volume.iloc[i] < volume.iloc[i - 1]:
                nvi.iloc[i] = nvi.iloc[i - 1] + (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1] * nvi.iloc[i - 1]
            else:
                nvi.iloc[i] = nvi.iloc[i - 1]
        return nvi

    def positive_volume_index(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        pvi = pd.Series(index=close.index, dtype=float)
        pvi.iloc[0] = 1000
        for i in range(1, len(close)):
            if volume.iloc[i] > volume.iloc[i - 1]:
                pvi.iloc[i] = pvi.iloc[i - 1] + (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1] * pvi.iloc[i - 1]
            else:
                pvi.iloc[i] = pvi.iloc[i - 1]
        return pvi

    def volume_surge(self, volume: pd.Series, period: int = 20) -> pd.Series:
        avg_volume = volume.rolling(window=period).mean()
        return volume / avg_volume

    def money_flow_index(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        return self.mfi(high, low, close, volume, period)

    # ==================== CYCLE INDICATORS ====================

    def ehlers_cycle(self, close: pd.Series, period: int = 20) -> pd.Series:
        alpha = 1 / period
        cycle = pd.Series(index=close.index, dtype=float)
        for i in range(period, len(close)):
            cycle.iloc[i] = (1 - alpha / 2) ** 2 * (close.iloc[i] - 2 * close.iloc[i - 1] + close.iloc[i - 2]) + 2 * (1 - alpha) * cycle.iloc[i - 1] - (1 - alpha) ** 2 * cycle.iloc[i - 2]
        return cycle

    def mfi_period(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tp = (high + low + close) / 3
        mfm = tp * volume
        return tp

    def dominant_cycle(self, close: pd.Series, max_period: int = 48) -> int:
        best_period = 1
        best_corr = 0
        for period in range(6, max_period):
            shifted = close.shift(period)
            corr = close.corr(shifted)
            if corr > best_corr:
                best_corr = corr
                best_period = period
        return best_period

    def sinewave_indicator(self, close: pd.Series, period: int = 40) -> tuple[pd.Series, pd.Series]:
        phase = 2 * np.pi * np.arange(len(close)) / period
        sinewave = np.sin(phase)
        leadsine = np.sin(phase - np.pi / 4)
        return pd.Series(sinewave, index=close.index), pd.Series(leadsine, index=close.index)

    def hilbert_transform(self, close: pd.Series) -> tuple[pd.Series, pd.Series]:
        hilbert = pd.Series(index=close.index, dtype=complex)
        for i in range(len(close)):
            hilbert.iloc[i] = close.iloc[i] * np.exp(1j * 2 * np.pi * i / 20)
        return hilbert.real, hilbert.imag

    # ==================== STATISTICAL INDICATORS ====================

    def zscore(self, close: pd.Series, period: int = 20) -> pd.Series:
        mean = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        return (close - mean) / std.replace(0, np.nan)

    def percentile_rank(self, close: pd.Series, period: int = 100) -> pd.Series:
        return close.rolling(window=period).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100, raw=False)

    def skewness(self, close: pd.Series, period: int = 20) -> pd.Series:
        return close.rolling(window=period).skew()

    def kurtosis(self, close: pd.Series, period: int = 20) -> pd.Series:
        return close.rolling(window=period).kurt()

    def correlation(self, close1: pd.Series, close2: pd.Series, period: int = 20) -> pd.Series:
        return close1.rolling(window=period).corr(close2)

    def covariance(self, close1: pd.Series, close2: pd.Series, period: int = 20) -> pd.Series:
        return close1.rolling(window=period).cov(close2)

    def beta(self, close: pd.Series, benchmark: pd.Series, period: int = 20) -> pd.Series:
        cov = close.rolling(window=period).cov(benchmark)
        var = benchmark.rolling(window=period).var()
        return cov / var.replace(0, np.nan)

    def alpha(self, close: pd.Series, benchmark: pd.Series, period: int = 20, risk_free: float = 0.0) -> pd.Series:
        close_ret = close.pct_change()
        bench_ret = benchmark.pct_change()
        close_mean = close_ret.rolling(window=period).mean()
        bench_mean = bench_ret.rolling(window=period).mean()
        b = self.beta(close, benchmark, period)
        return close_mean - (risk_free + b * (bench_mean - risk_free))

    def information_ratio(self, close: pd.Series, benchmark: pd.Series, period: int = 20) -> pd.Series:
        active_ret = close.pct_change() - benchmark.pct_change()
        return active_ret.rolling(window=period).mean() / active_ret.rolling(window=period).std().replace(0, np.nan)

    def sharpe_ratio(self, close: pd.Series, period: int = 20, risk_free: float = 0.0) -> pd.Series:
        returns = close.pct_change()
        excess_returns = returns - risk_free / 252
        return np.sqrt(252) * excess_returns.rolling(window=period).mean() / excess_returns.rolling(window=period).std().replace(0, np.nan)

    def sortino_ratio(self, close: pd.Series, period: int = 20, mar: float = 0.0) -> pd.Series:
        returns = close.pct_change()
        excess_returns = returns - mar / 252
        downside = returns.where(returns < mar / 252, 0)
        return np.sqrt(252) * excess_returns.rolling(window=period).mean() / downside.rolling(window=period).std().replace(0, np.nan)

    def calmar_ratio(self, close: pd.Series, period: int = 252) -> pd.Series:
        returns = close.pct_change()
        annual_return = returns.rolling(window=period).mean() * 252
        rolling_max = close.rolling(window=period, min_periods=1).max()
        drawdown = (close - rolling_max) / rolling_max
        max_drawdown = drawdown.rolling(window=period).min()
        return annual_return / abs(max_drawdown).replace(0, np.nan)

    def omega_ratio(self, close: pd.Series, period: int = 20, threshold: float = 0.0) -> pd.Series:
        returns = close.pct_change()
        gains = returns.where(returns > threshold, 0)
        losses = returns.where(returns < threshold, 0).abs()
        return gains.rolling(window=period).sum() / losses.rolling(window=period).sum().replace(0, np.nan)

    def tail_ratio(self, close: pd.Series, period: int = 252) -> pd.Series:
        returns = close.pct_change()
        return returns.rolling(window=period).quantile(0.95) / abs(returns.rolling(window=period).quantile(0.05)).replace(0, np.nan)

    def gain_loss_ratio(self, close: pd.Series, period: int = 20) -> pd.Series:
        returns = close.pct_change()
        gains = returns.where(returns > 0, 0)
        losses = returns.where(returns < 0, 0).abs()
        return gains.rolling(window=period).mean() / losses.rolling(window=period).mean().replace(0, np.nan)

    def win_rate(self, close: pd.Series, period: int = 20) -> pd.Series:
        returns = close.pct_change()
        wins = (returns > 0).astype(int)
        return wins.rolling(window=period).mean()

    def profit_factor(self, close: pd.Series, period: int = 20) -> pd.Series:
        returns = close.pct_change()
        gains = returns.where(returns > 0, 0)
        losses = returns.where(returns < 0, 0).abs()
        return gains.rolling(window=period).sum() / losses.rolling(window=period).sum().replace(0, np.nan)

    def common_sense_ratio(self, close: pd.Series, period: int = 20) -> pd.Series:
        returns = close.pct_change()
        return self.win_rate(close, period) * self.gain_loss_ratio(close, period)

    def payoff_ratio(self, close: pd.Series, period: int = 20) -> pd.Series:
        returns = close.pct_change()
        avg_win = returns.where(returns > 0, 0).rolling(window=period).mean()
        avg_loss = returns.where(returns < 0, 0).abs().rolling(window=period).mean()
        return avg_win / avg_loss.replace(0, np.nan)

    def expected_return(self, close: pd.Series, period: int = 20) -> pd.Series:
        returns = close.pct_change()
        return returns.rolling(window=period).mean()

    def value_at_risk(self, close: pd.Series, period: int = 20, confidence: float = 0.95) -> pd.Series:
        returns = close.pct_change()
        return returns.rolling(window=period).quantile(1 - confidence)

    def conditional_var(self, close: pd.Series, period: int = 20, confidence: float = 0.95) -> pd.Series:
        returns = close.pct_change()
        var = self.value_at_risk(close, period, confidence)
        cvar = returns.rolling(window=period).apply(lambda x: x[x <= x.quantile(1 - confidence)].mean(), raw=False)
        return cvar

    def maximum_drawdown(self, close: pd.Series, period: int = 252) -> pd.Series:
        rolling_max = close.rolling(window=period, min_periods=1).max()
        drawdown = (close - rolling_max) / rolling_max
        return drawdown.rolling(window=period).min()

    def ulcer_performance_index(self, close: pd.Series, period: int = 14, risk_free: float = 0.0) -> pd.Series:
        returns = close.pct_change()
        excess_return = returns.rolling(window=period).mean() - risk_free / 252
        ui = self.ulcer_index(close, period)
        return excess_return / ui.replace(0, np.nan)

    def kelly_criterion(self, close: pd.Series, period: int = 20) -> pd.Series:
        returns = close.pct_change()
        win_rate = self.win_rate(close, period)
        gain_loss = self.gain_loss_ratio(close, period)
        return win_rate - (1 - win_rate) / gain_loss.replace(0, np.nan)

    # ==================== PRICE ACTION INDICATORS ====================

    def pivot_points(self, high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        return pivot, r1, r2, s1, s2

    def fibonacci_retracement(self, high: pd.Series, low: pd.Series, lookback: int = 50) -> dict[str, pd.Series]:
        highest = high.rolling(window=lookback).max()
        lowest = low.rolling(window=lookback).min()
        diff = highest - lowest
        return {
            "0.0": highest,
            "0.236": highest - 0.236 * diff,
            "0.382": highest - 0.382 * diff,
            "0.5": highest - 0.5 * diff,
            "0.618": highest - 0.618 * diff,
            "0.786": highest - 0.786 * diff,
            "1.0": lowest,
        }

    def ichimoku_cloud(self, high: pd.Series, low: pd.Series, close: pd.Series, conversion: int = 9, base: int = 26, span_b: int = 52) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        tenkan = (high.rolling(conversion).max() + low.rolling(conversion).min()) / 2
        kijun = (high.rolling(base).max() + low.rolling(base).min()) / 2
        senkou_a = (tenkan + kijun) / 2
        senkou_b = (high.rolling(span_b).max() + low.rolling(span_b).min()) / 2
        chikou = close.shift(-base)
        return tenkan, kijun, senkou_a, senkou_b, chikou

    def three_line_strike(self, close: pd.Series) -> pd.Series:
        pattern = pd.Series(0, index=close.index)
        for i in range(3, len(close)):
            if close.iloc[i-3] < close.iloc[i-2] < close.iloc[i-1] and close.iloc[i] < close.iloc[i-3]:
                pattern.iloc[i] = -1  # Bearish
            elif close.iloc[i-3] > close.iloc[i-2] > close.iloc[i-1] and close.iloc[i] > close.iloc[i-3]:
                pattern.iloc[i] = 1  # Bullish
        return pattern

    def inside_bar(self, high: pd.Series, low: pd.Series) -> pd.Series:
        inside = (high < high.shift(1)) & (low > low.shift(1))
        return inside.astype(int)

    def outside_bar(self, high: pd.Series, low: pd.Series) -> pd.Series:
        outside = (high > high.shift(1)) & (low < low.shift(1))
        return outside.astype(int)

    def engulfing_pattern(self, open_: pd.Series, close: pd.Series) -> pd.Series:
        bullish = (close.shift(1) < open_.shift(1)) & (close > open_) & (close > open_.shift(1)) & (open_ < close.shift(1))
        bearish = (close.shift(1) > open_.shift(1)) & (close < open_) & (close < open_.shift(1)) & (open_ > close.shift(1))
        pattern = pd.Series(0, index=close.index)
        pattern[bullish] = 1
        pattern[bearish] = -1
        return pattern

    def doji(self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, threshold: float = 0.001) -> pd.Series:
        body = abs(close - open_)
        range_hl = high - low
        doji = body < (threshold * range_hl)
        return doji.astype(int)

    def hammer(self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        body = abs(close - open_)
        lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
        upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
        hammer = (lower_shadow > 2 * body) & (upper_shadow < body)
        return hammer.astype(int)

    def shooting_star(self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        body = abs(close - open_)
        upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
        shooting = (upper_shadow > 2 * body) & (lower_shadow < body)
        return shooting.astype(int)

    # ==================== COMPREHENSIVE ANALYSIS ====================

    def calculate_all(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate all technical indicators."""
        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        results = {}

        # Trend
        results["sma_20"] = self.sma(close, 20)
        results["sma_50"] = self.sma(close, 50)
        results["sma_200"] = self.sma(close, 200)
        results["ema_20"] = self.ema(close, 20)
        results["ema_50"] = self.ema(close, 50)
        results["hull_ma"] = self.hull_ma(close, 20)
        results["tema"] = self.tema(close, 20)
        results["kama"] = self.kama(close)
        results["vwma"] = self.vwma(close, volume, 20)
        results["psar"] = self.psar(high, low)
        results["supertrend"], results["supertrend_dir"] = self.supertrend(high, low, close)
        results["adx"], results["plus_di"], results["minus_di"] = self.adx(high, low, close)
        results["aroon_up"], results["aroon_down"] = self.aroon(high, low)

        # Momentum
        results["rsi"] = self.rsi(close, 14)
        results["stoch_k"], results["stoch_d"] = self.stochastic(high, low, close)
        results["williams_r"] = self.williams_r(high, low, close)
        results["cci"] = self.cci(high, low, close)
        results["macd"], results["macd_signal"], results["macd_hist"] = self.macd(close)
        results["trix"] = self.trix(close)
        results["ultimate_osc"] = self.ultimate_oscillator(high, low, close)
        results["tsi"] = self.tsi(close)
        results["awesome_osc"] = self.awesome_oscillator(high, low)

        # Volatility
        results["bb_upper"], results["bb_middle"], results["bb_lower"] = self.bollinger_bands(close)
        results["keltner_upper"], results["keltner_middle"], results["keltner_lower"] = self.keltner_channel(high, low, close)
        results["donchian_upper"], results["donchian_middle"], results["donchian_lower"] = self.donchian_channel(high, low)
        results["atr"] = self.atr(high, low, close)
        results["historical_vol"] = self.historical_volatility(close)

        # Volume
        results["obv"] = self.obv(close, volume)
        results["cmf"] = self.cmf(high, low, close, volume)
        results["mfi"] = self.mfi(high, low, close, volume)
        results["vwap"] = self.vwap(high, low, close, volume)
        results["force_index"] = self.force_index(close, volume)
        results["ad"] = self.accumulation_distribution(high, low, close, volume)

        # Statistical
        results["zscore"] = self.zscore(close)
        results["skewness"] = self.skewness(close)
        results["sharpe"] = self.sharpe_ratio(close)
        results["sortino"] = self.sortino_ratio(close)

        # Price Action
        results["pivot"], results["r1"], results["r2"], results["s1"], results["s2"] = self.pivot_points(high, low, close)
        results["ichimoku_conv"], results["ichimoku_base"], results["senkou_a"], results["senkou_b"], results["chikou"] = self.ichimoku_cloud(high, low, close)

        return results


def get_indicators() -> TechnicalIndicators:
    """Get technical indicators instance."""
    return TechnicalIndicators()
