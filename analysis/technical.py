# analysis/technical.py
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import ta

from utils.logger import get_logger

log = get_logger(__name__)

class TrendDirection(Enum):
    STRONG_UP = "strong_uptrend"
    UP = "uptrend"
    SIDEWAYS = "sideways"
    DOWN = "downtrend"
    STRONG_DOWN = "strong_downtrend"

class SignalStrength(Enum):
    STRONG = 3
    MODERATE = 2
    WEAK = 1
    NONE = 0

@dataclass
class TechnicalSignal:
    """Technical analysis signal."""
    indicator: str
    signal: str  # "buy", "sell", "neutral"
    strength: SignalStrength
    value: float
    description: str

@dataclass
class SupportResistance:
    """Support and resistance levels."""
    support_1: float
    support_2: float
    support_3: float
    resistance_1: float
    resistance_2: float
    resistance_3: float
    pivot: float

@dataclass
class TechnicalSummary:
    """Complete technical analysis summary."""
    trend: TrendDirection
    trend_strength: float
    signals: list[TechnicalSignal]
    support_resistance: SupportResistance
    overall_signal: str  # "buy", "sell", "neutral"
    overall_score: float  # -100 to +100
    indicators: dict[str, float]

class TechnicalAnalyzer:
    """Comprehensive technical analysis engine."""

    def __init__(self) -> None:
        self.min_data_points = 60
        self._required_columns = ("open", "high", "low", "close", "volume")
        self._indicator_names = (
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_9", "ema_21", "ema_55", "ema_100", "ema_200",
            "macd", "macd_signal", "macd_hist", "macd_hist_prev",
            "ppo", "ppo_signal", "ppo_hist", "trix",
            "rsi_14", "rsi_7", "rsi_21", "stoch_k", "stoch_d", "stoch_rsi",
            "uo", "tsi", "kama",
            "bb_upper", "bb_middle", "bb_lower", "bb_pct", "bb_width",
            "keltner_upper", "keltner_middle", "keltner_lower",
            "donchian_upper", "donchian_middle", "donchian_lower",
            "ichimoku_conv", "ichimoku_base", "ichimoku_a", "ichimoku_b",
            "adx", "di_plus", "di_minus",
            "atr_14", "mfi", "cci", "williams_r", "cmf", "force_index",
            "roc_10", "obv", "vwap",
            "volatility_20", "atr_pct", "momentum_20",
            "volume_ratio", "close", "prev_close", "change_pct",
        )

    def list_supported_indicators(self) -> list[str]:
        """Return a stable list of supported indicator names."""
        return list(self._indicator_names)

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and sanitize OHLCV input before indicator computation."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")

        missing = [c for c in self._required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        out = df.copy()
        for col in self._required_columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.replace([np.inf, -np.inf], np.nan)
        out["open"] = out["open"].fillna(out["close"])
        out = out.dropna(subset=["high", "low", "close", "volume"])
        if out.empty:
            raise ValueError("No valid OHLCV rows after sanitization")

        if not out.index.is_monotonic_increasing:
            out = out.sort_index()
        return out

    def analyze(self, df: pd.DataFrame) -> TechnicalSummary:
        """Perform complete technical analysis."""
        df = self._prepare_dataframe(df)
        if len(df) < self.min_data_points:
            raise ValueError(f"Need at least {self.min_data_points} data points")

        df = df.copy()

        indicators = self._calculate_indicators(df)
        signals = self._generate_signals(df, indicators)
        trend, trend_strength = self._analyze_trend(df, indicators)
        sr_levels = self._calculate_support_resistance(df)
        overall_score, overall_signal = self._calculate_overall(signals, trend)

        return TechnicalSummary(
            trend=trend,
            trend_strength=trend_strength,
            signals=signals,
            support_resistance=sr_levels,
            overall_signal=overall_signal,
            overall_score=overall_score,
            indicators=indicators
        )

    def _calculate_indicators(self, df: pd.DataFrame) -> dict[str, float]:
        """Calculate all technical indicators."""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        indicators = {}

        def safe_last(series: pd.Series, default: float = 0.0) -> float:
            if len(series) and pd.notna(series.iloc[-1]):
                return float(series.iloc[-1])
            return float(default)

        indicators['sma_5'] = safe_last(close.rolling(5).mean())
        indicators['sma_10'] = safe_last(close.rolling(10).mean())
        indicators['sma_20'] = safe_last(close.rolling(20).mean())
        indicators['sma_50'] = safe_last(close.rolling(50).mean())
        indicators['ema_9'] = safe_last(close.ewm(span=9, adjust=False).mean())
        indicators['ema_21'] = safe_last(close.ewm(span=21, adjust=False).mean())
        indicators['ema_55'] = safe_last(close.ewm(span=55, adjust=False).mean())
        indicators['ema_100'] = safe_last(close.ewm(span=100, adjust=False).mean())
        indicators['ema_200'] = safe_last(close.ewm(span=200, adjust=False).mean())

        if len(df) >= 200:
            indicators['sma_200'] = safe_last(close.rolling(200).mean())
        else:
            indicators['sma_200'] = safe_last(close.rolling(len(df)).mean())

        macd = ta.trend.MACD(close)
        macd_diff = macd.macd_diff()
        indicators['macd'] = safe_last(macd.macd())
        indicators['macd_signal'] = safe_last(macd.macd_signal())
        indicators['macd_hist'] = safe_last(macd_diff)
        indicators['macd_hist_prev'] = float(macd_diff.iloc[-2]) if len(macd_diff) > 1 and pd.notna(macd_diff.iloc[-2]) else 0.0
        indicators['ppo'] = safe_last(ta.momentum.ppo(close))
        indicators['ppo_signal'] = safe_last(ta.momentum.ppo_signal(close))
        indicators['ppo_hist'] = safe_last(ta.momentum.ppo_hist(close))
        indicators['trix'] = safe_last(ta.trend.trix(close))

        indicators['rsi_14'] = safe_last(ta.momentum.rsi(close, window=14))
        indicators['rsi_7'] = safe_last(ta.momentum.rsi(close, window=7))
        indicators['rsi_21'] = safe_last(ta.momentum.rsi(close, window=21))

        stoch = ta.momentum.StochasticOscillator(high, low, close)
        indicators['stoch_k'] = safe_last(stoch.stoch())
        indicators['stoch_d'] = safe_last(stoch.stoch_signal())
        indicators['stoch_rsi'] = safe_last(ta.momentum.stochrsi(close, window=14, smooth1=3, smooth2=3))
        indicators['uo'] = safe_last(ta.momentum.ultimate_oscillator(high, low, close))
        indicators['tsi'] = safe_last(ta.momentum.tsi(close))
        try:
            indicators['kama'] = safe_last(ta.momentum.kama(close))
        except Exception:
            indicators['kama'] = 0.0

        bb = ta.volatility.BollingerBands(close)
        indicators['bb_upper'] = safe_last(bb.bollinger_hband())
        indicators['bb_middle'] = safe_last(bb.bollinger_mavg())
        indicators['bb_lower'] = safe_last(bb.bollinger_lband())
        indicators['bb_pct'] = safe_last(bb.bollinger_pband())
        indicators['bb_width'] = safe_last(bb.bollinger_wband())

        kc = ta.volatility.KeltnerChannel(high, low, close)
        indicators['keltner_upper'] = safe_last(kc.keltner_channel_hband())
        indicators['keltner_middle'] = safe_last(kc.keltner_channel_mband())
        indicators['keltner_lower'] = safe_last(kc.keltner_channel_lband())

        dc = ta.volatility.DonchianChannel(high, low, close, window=20)
        indicators['donchian_upper'] = safe_last(dc.donchian_channel_hband())
        indicators['donchian_middle'] = safe_last(dc.donchian_channel_mband())
        indicators['donchian_lower'] = safe_last(dc.donchian_channel_lband())

        ichimoku = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
        indicators['ichimoku_conv'] = safe_last(ichimoku.ichimoku_conversion_line())
        indicators['ichimoku_base'] = safe_last(ichimoku.ichimoku_base_line())
        indicators['ichimoku_a'] = safe_last(ichimoku.ichimoku_a())
        indicators['ichimoku_b'] = safe_last(ichimoku.ichimoku_b())

        indicators['adx'] = safe_last(ta.trend.adx(high, low, close))
        indicators['di_plus'] = safe_last(ta.trend.adx_pos(high, low, close))
        indicators['di_minus'] = safe_last(ta.trend.adx_neg(high, low, close))
        indicators['atr_14'] = safe_last(ta.volatility.average_true_range(
            high, low, close, window=14
        ))

        vol_ma20 = safe_last(volume.rolling(20).mean(), 1.0)
        indicators['volume_ratio'] = float(volume.iloc[-1] / vol_ma20) if vol_ma20 > 0 else 1.0

        indicators['mfi'] = safe_last(ta.volume.money_flow_index(high, low, close, volume))
        indicators['cmf'] = safe_last(ta.volume.chaikin_money_flow(high, low, close, volume))
        indicators['force_index'] = safe_last(ta.volume.force_index(close, volume))

        indicators['cci'] = safe_last(ta.trend.cci(high, low, close))
        indicators['williams_r'] = safe_last(ta.momentum.williams_r(
            high, low, close, lbp=14
        ))
        indicators['roc_10'] = safe_last(ta.momentum.roc(close, window=10))
        indicators['momentum_20'] = safe_last(close.pct_change(20) * 100.0)
        indicators['volatility_20'] = safe_last(
            close.pct_change().rolling(20).std() * np.sqrt(252) * 100.0
        )
        indicators['obv'] = safe_last(ta.volume.on_balance_volume(close, volume))
        indicators['vwap'] = safe_last(ta.volume.volume_weighted_average_price(
            high, low, close, volume, window=20
        ))

        indicators['close'] = close.iloc[-1]
        indicators['prev_close'] = close.iloc[-2] if len(df) > 1 else close.iloc[-1]
        indicators['change_pct'] = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(df) > 1 else 0
        indicators['atr_pct'] = (
            (indicators['atr_14'] / indicators['close']) * 100.0
            if indicators['close'] > 0
            else 0.0
        )

        for key, value in indicators.items():
            try:
                if pd.isna(value) or not np.isfinite(float(value)):
                    indicators[key] = 0.0
            except Exception:
                indicators[key] = 0.0

        return indicators

    def _generate_signals(self, df: pd.DataFrame, ind: dict[str, float]) -> list[TechnicalSignal]:
        """Generate trading signals from indicators."""
        signals = []
        close = ind['close']

        rsi = ind['rsi_14']
        if rsi < 30:
            signals.append(TechnicalSignal("RSI", "buy", SignalStrength.STRONG, rsi, f"RSI oversold ({rsi:.0f})"))
        elif rsi > 70:
            signals.append(TechnicalSignal("RSI", "sell", SignalStrength.STRONG, rsi, f"RSI overbought ({rsi:.0f})"))
        else:
            signals.append(TechnicalSignal("RSI", "neutral", SignalStrength.NONE, rsi, f"RSI neutral ({rsi:.0f})"))

        if ind['macd'] > ind['macd_signal']:
            strength = SignalStrength.STRONG if ind['macd_hist'] > ind['macd_hist_prev'] else SignalStrength.MODERATE
            signals.append(TechnicalSignal("MACD", "buy", strength, ind['macd'], "MACD above signal"))
        else:
            strength = SignalStrength.STRONG if ind['macd_hist'] < ind['macd_hist_prev'] else SignalStrength.MODERATE
            signals.append(TechnicalSignal("MACD", "sell", strength, ind['macd'], "MACD below signal"))

        if ind['ppo'] > ind['ppo_signal']:
            signals.append(TechnicalSignal("PPO", "buy", SignalStrength.WEAK, ind['ppo'], "PPO above signal"))
        elif ind['ppo'] < ind['ppo_signal']:
            signals.append(TechnicalSignal("PPO", "sell", SignalStrength.WEAK, ind['ppo'], "PPO below signal"))

        if close > ind['sma_20']:
            signals.append(TechnicalSignal("SMA", "buy", SignalStrength.MODERATE, ind['sma_20'], "Price above SMA 20"))
        else:
            signals.append(TechnicalSignal("SMA", "sell", SignalStrength.MODERATE, ind['sma_20'], "Price below SMA 20"))

        if close > ind['ema_21'] > ind['ema_55'] > ind['ema_200']:
            signals.append(TechnicalSignal("EMA Trend", "buy", SignalStrength.MODERATE, ind['ema_21'], "Bullish EMA stack"))
        elif close < ind['ema_21'] < ind['ema_55'] < ind['ema_200']:
            signals.append(TechnicalSignal("EMA Trend", "sell", SignalStrength.MODERATE, ind['ema_21'], "Bearish EMA stack"))

        if ind['adx'] > 25:
            if ind['di_plus'] > ind['di_minus']:
                signals.append(TechnicalSignal("ADX", "buy", SignalStrength.STRONG, ind['adx'], f"Strong uptrend (ADX={ind['adx']:.0f})"))
            else:
                signals.append(TechnicalSignal("ADX", "sell", SignalStrength.STRONG, ind['adx'], f"Strong downtrend (ADX={ind['adx']:.0f})"))

        if ind['stoch_k'] < 20:
            signals.append(TechnicalSignal("Stochastic", "buy", SignalStrength.MODERATE, ind['stoch_k'], "Stochastic oversold"))
        elif ind['stoch_k'] > 80:
            signals.append(TechnicalSignal("Stochastic", "sell", SignalStrength.MODERATE, ind['stoch_k'], "Stochastic overbought"))

        if ind['bb_pct'] < 0:
            signals.append(TechnicalSignal("BB", "buy", SignalStrength.MODERATE, ind['bb_lower'], "Price below lower BB"))
        elif ind['bb_pct'] > 1:
            signals.append(TechnicalSignal("BB", "sell", SignalStrength.MODERATE, ind['bb_upper'], "Price above upper BB"))
        elif ind['bb_width'] < 8 and ind['adx'] > 20:
            side = "buy" if ind['di_plus'] >= ind['di_minus'] else "sell"
            signals.append(TechnicalSignal("BB Squeeze", side, SignalStrength.WEAK, ind['bb_width'], "Volatility squeeze with trend pressure"))

        if ind['mfi'] < 20:
            signals.append(TechnicalSignal("MFI", "buy", SignalStrength.MODERATE, ind['mfi'], "Money flow oversold"))
        elif ind['mfi'] > 80:
            signals.append(TechnicalSignal("MFI", "sell", SignalStrength.MODERATE, ind['mfi'], "Money flow overbought"))

        if ind['cmf'] > 0.1:
            signals.append(TechnicalSignal("CMF", "buy", SignalStrength.WEAK, ind['cmf'], "Positive money flow"))
        elif ind['cmf'] < -0.1:
            signals.append(TechnicalSignal("CMF", "sell", SignalStrength.WEAK, ind['cmf'], "Negative money flow"))

        if ind['uo'] < 30:
            signals.append(TechnicalSignal("Ultimate Osc", "buy", SignalStrength.WEAK, ind['uo'], "Ultimate oscillator oversold"))
        elif ind['uo'] > 70:
            signals.append(TechnicalSignal("Ultimate Osc", "sell", SignalStrength.WEAK, ind['uo'], "Ultimate oscillator overbought"))

        if ind['cci'] < -100:
            signals.append(TechnicalSignal("CCI", "buy", SignalStrength.WEAK, ind['cci'], "CCI oversold"))
        elif ind['cci'] > 100:
            signals.append(TechnicalSignal("CCI", "sell", SignalStrength.WEAK, ind['cci'], "CCI overbought"))

        if close > ind['donchian_upper'] and ind['volume_ratio'] >= 1.2:
            signals.append(TechnicalSignal("Donchian", "buy", SignalStrength.STRONG, ind['donchian_upper'], "Breakout above Donchian high"))
        elif close < ind['donchian_lower'] and ind['volume_ratio'] >= 1.2:
            signals.append(TechnicalSignal("Donchian", "sell", SignalStrength.STRONG, ind['donchian_lower'], "Breakdown below Donchian low"))

        if close > ind['ichimoku_base'] and ind['ichimoku_conv'] > ind['ichimoku_base']:
            signals.append(TechnicalSignal("Ichimoku", "buy", SignalStrength.WEAK, ind['ichimoku_base'], "Price above base and conversion > base"))
        elif close < ind['ichimoku_base'] and ind['ichimoku_conv'] < ind['ichimoku_base']:
            signals.append(TechnicalSignal("Ichimoku", "sell", SignalStrength.WEAK, ind['ichimoku_base'], "Price below base and conversion < base"))

        return signals

    def _analyze_trend(self, df: pd.DataFrame, ind: dict[str, float]) -> tuple[TrendDirection, float]:
        """Analyze overall trend."""
        close = ind['close']
        scores = []

        # Short-term
        if close > ind['sma_5'] > ind['sma_10']:
            scores.append(2)
        elif close < ind['sma_5'] < ind['sma_10']:
            scores.append(-2)
        else:
            scores.append(0)

        # Medium-term
        if close > ind['sma_20'] > ind['sma_50']:
            scores.append(2)
        elif close < ind['sma_20'] < ind['sma_50']:
            scores.append(-2)
        else:
            scores.append(0)

        # Long-term
        if ind['sma_50'] > ind['sma_200']:
            scores.append(2)
        else:
            scores.append(-2)

        if ind['adx'] > 25:
            if ind['di_plus'] > ind['di_minus']:
                scores.append(1)
            else:
                scores.append(-1)

        total_score = sum(scores)
        trend_strength = min(abs(total_score) / 8, 1.0)

        if total_score >= 5:
            trend = TrendDirection.STRONG_UP
        elif total_score >= 2:
            trend = TrendDirection.UP
        elif total_score <= -5:
            trend = TrendDirection.STRONG_DOWN
        elif total_score <= -2:
            trend = TrendDirection.DOWN
        else:
            trend = TrendDirection.SIDEWAYS

        return trend, trend_strength

    def _calculate_support_resistance(self, df: pd.DataFrame) -> SupportResistance:
        """Calculate support and resistance using pivot points."""
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]

        pivot = (high + low + close) / 3

        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)

        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)

        return SupportResistance(
            support_1=round(s1, 2),
            support_2=round(s2, 2),
            support_3=round(s3, 2),
            resistance_1=round(r1, 2),
            resistance_2=round(r2, 2),
            resistance_3=round(r3, 2),
            pivot=round(pivot, 2)
        )

    def _calculate_overall(self, signals: list[TechnicalSignal], trend: TrendDirection) -> tuple[float, str]:
        """Calculate overall score and signal."""
        score = 0.0

        for technical_signal in signals:
            weight = technical_signal.strength.value
            if technical_signal.signal == "buy":
                score += weight * 10
            elif technical_signal.signal == "sell":
                score -= weight * 10

        trend_scores = {
            TrendDirection.STRONG_UP: 20,
            TrendDirection.UP: 10,
            TrendDirection.SIDEWAYS: 0,
            TrendDirection.DOWN: -10,
            TrendDirection.STRONG_DOWN: -20
        }
        score += trend_scores.get(trend, 0)

        score = max(-100, min(100, score))

        if score >= 30:
            overall_signal = "buy"
        elif score <= -30:
            overall_signal = "sell"
        else:
            overall_signal = "neutral"

        return float(score), overall_signal
