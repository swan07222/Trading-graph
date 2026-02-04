"""
Technical Analysis - Advanced indicators and pattern recognition
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import ta

from utils.logger import log


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
    """Technical analysis signal"""
    indicator: str
    signal: str  # "buy", "sell", "neutral"
    strength: SignalStrength
    value: float
    description: str


@dataclass
class SupportResistance:
    """Support and resistance levels"""
    support_1: float
    support_2: float
    support_3: float
    resistance_1: float
    resistance_2: float
    resistance_3: float
    pivot: float


@dataclass
class TechnicalSummary:
    """Complete technical analysis summary"""
    trend: TrendDirection
    trend_strength: float
    signals: List[TechnicalSignal]
    support_resistance: SupportResistance
    overall_signal: str  # "buy", "sell", "neutral"
    overall_score: float  # -100 to +100
    indicators: Dict[str, float]


class TechnicalAnalyzer:
    """Comprehensive technical analysis engine"""
    
    def __init__(self):
        self.min_data_points = 60
    
    def analyze(self, df: pd.DataFrame) -> TechnicalSummary:
        """Perform complete technical analysis"""
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
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate all technical indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        indicators = {}
        
        # Moving Averages
        indicators['sma_5'] = close.rolling(5).mean().iloc[-1]
        indicators['sma_10'] = close.rolling(10).mean().iloc[-1]
        indicators['sma_20'] = close.rolling(20).mean().iloc[-1]
        indicators['sma_50'] = close.rolling(50).mean().iloc[-1]
        
        if len(df) >= 200:
            indicators['sma_200'] = close.rolling(200).mean().iloc[-1]
        else:
            indicators['sma_200'] = close.rolling(len(df)).mean().iloc[-1]
        
        # MACD
        macd = ta.trend.MACD(close)
        indicators['macd'] = macd.macd().iloc[-1]
        indicators['macd_signal'] = macd.macd_signal().iloc[-1]
        indicators['macd_hist'] = macd.macd_diff().iloc[-1]
        indicators['macd_hist_prev'] = macd.macd_diff().iloc[-2] if len(df) > 1 else 0
        
        # RSI
        indicators['rsi_14'] = ta.momentum.rsi(close, window=14).iloc[-1]
        indicators['rsi_7'] = ta.momentum.rsi(close, window=7).iloc[-1]
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        indicators['stoch_k'] = stoch.stoch().iloc[-1]
        indicators['stoch_d'] = stoch.stoch_signal().iloc[-1]
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close)
        indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
        indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
        indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
        indicators['bb_pct'] = bb.bollinger_pband().iloc[-1]
        
        # ADX
        indicators['adx'] = ta.trend.adx(high, low, close).iloc[-1]
        indicators['di_plus'] = ta.trend.adx_pos(high, low, close).iloc[-1]
        indicators['di_minus'] = ta.trend.adx_neg(high, low, close).iloc[-1]
        
        # Volume
        vol_ma20 = volume.rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
        
        # MFI
        indicators['mfi'] = ta.volume.money_flow_index(high, low, close, volume).iloc[-1]
        
        # CCI
        indicators['cci'] = ta.trend.cci(high, low, close).iloc[-1]
        
        # Price data
        indicators['close'] = close.iloc[-1]
        indicators['prev_close'] = close.iloc[-2] if len(df) > 1 else close.iloc[-1]
        indicators['change_pct'] = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(df) > 1 else 0
        
        return indicators
    
    def _generate_signals(self, df: pd.DataFrame, ind: Dict[str, float]) -> List[TechnicalSignal]:
        """Generate trading signals from indicators"""
        signals = []
        close = ind['close']
        
        # RSI
        rsi = ind['rsi_14']
        if rsi < 30:
            signals.append(TechnicalSignal("RSI", "buy", SignalStrength.STRONG, rsi, f"RSI oversold ({rsi:.0f})"))
        elif rsi > 70:
            signals.append(TechnicalSignal("RSI", "sell", SignalStrength.STRONG, rsi, f"RSI overbought ({rsi:.0f})"))
        else:
            signals.append(TechnicalSignal("RSI", "neutral", SignalStrength.NONE, rsi, f"RSI neutral ({rsi:.0f})"))
        
        # MACD
        if ind['macd'] > ind['macd_signal']:
            strength = SignalStrength.STRONG if ind['macd_hist'] > ind['macd_hist_prev'] else SignalStrength.MODERATE
            signals.append(TechnicalSignal("MACD", "buy", strength, ind['macd'], "MACD above signal"))
        else:
            strength = SignalStrength.STRONG if ind['macd_hist'] < ind['macd_hist_prev'] else SignalStrength.MODERATE
            signals.append(TechnicalSignal("MACD", "sell", strength, ind['macd'], "MACD below signal"))
        
        # Moving Averages
        if close > ind['sma_20']:
            signals.append(TechnicalSignal("SMA", "buy", SignalStrength.MODERATE, ind['sma_20'], "Price above SMA 20"))
        else:
            signals.append(TechnicalSignal("SMA", "sell", SignalStrength.MODERATE, ind['sma_20'], "Price below SMA 20"))
        
        # ADX
        if ind['adx'] > 25:
            if ind['di_plus'] > ind['di_minus']:
                signals.append(TechnicalSignal("ADX", "buy", SignalStrength.STRONG, ind['adx'], f"Strong uptrend (ADX={ind['adx']:.0f})"))
            else:
                signals.append(TechnicalSignal("ADX", "sell", SignalStrength.STRONG, ind['adx'], f"Strong downtrend (ADX={ind['adx']:.0f})"))
        
        # Stochastic
        if ind['stoch_k'] < 20:
            signals.append(TechnicalSignal("Stochastic", "buy", SignalStrength.MODERATE, ind['stoch_k'], "Stochastic oversold"))
        elif ind['stoch_k'] > 80:
            signals.append(TechnicalSignal("Stochastic", "sell", SignalStrength.MODERATE, ind['stoch_k'], "Stochastic overbought"))
        
        # Bollinger Bands
        if ind['bb_pct'] < 0:
            signals.append(TechnicalSignal("BB", "buy", SignalStrength.MODERATE, ind['bb_lower'], "Price below lower BB"))
        elif ind['bb_pct'] > 1:
            signals.append(TechnicalSignal("BB", "sell", SignalStrength.MODERATE, ind['bb_upper'], "Price above upper BB"))
        
        return signals
    
    def _analyze_trend(self, df: pd.DataFrame, ind: Dict[str, float]) -> Tuple[TrendDirection, float]:
        """Analyze overall trend"""
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
        
        # ADX
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
        """Calculate support and resistance using pivot points"""
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
    
    def _calculate_overall(self, signals: List[TechnicalSignal], trend: TrendDirection) -> Tuple[float, str]:
        """Calculate overall score and signal"""
        score = 0
        
        for signal in signals:
            weight = signal.strength.value
            if signal.signal == "buy":
                score += weight * 10
            elif signal.signal == "sell":
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
            signal = "buy"
        elif score <= -30:
            signal = "sell"
        else:
            signal = "neutral"
        
        return score, signal