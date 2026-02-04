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
    """
    Comprehensive technical analysis engine
    
    Features:
    - Trend analysis
    - Momentum indicators
    - Volume analysis
    - Support/Resistance detection
    - Pattern recognition
    - Divergence detection
    """
    
    def __init__(self):
        self.min_data_points = 60
    
    def analyze(self, df: pd.DataFrame) -> TechnicalSummary:
        """
        Perform complete technical analysis
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            TechnicalSummary with all analysis results
        """
        if len(df) < self.min_data_points:
            raise ValueError(f"Need at least {self.min_data_points} data points")
        
        df = df.copy()
        
        # Calculate all indicators
        indicators = self._calculate_indicators(df)
        
        # Generate signals
        signals = self._generate_signals(df, indicators)
        
        # Determine trend
        trend, trend_strength = self._analyze_trend(df, indicators)
        
        # Calculate support/resistance
        sr_levels = self._calculate_support_resistance(df)
        
        # Calculate overall score
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
        
        # === Trend Indicators ===
        # Moving Averages
        indicators['sma_5'] = close.rolling(5).mean().iloc[-1]
        indicators['sma_10'] = close.rolling(10).mean().iloc[-1]
        indicators['sma_20'] = close.rolling(20).mean().iloc[-1]
        indicators['sma_50'] = close.rolling(50).mean().iloc[-1]
        indicators['sma_200'] = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else close.rolling(len(df)).mean().iloc[-1]
        
        # EMA
        indicators['ema_12'] = close.ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = close.ewm(span=26).mean().iloc[-1]
        indicators['ema_50'] = close.ewm(span=50).mean().iloc[-1]
        
        # MACD
        macd = ta.trend.MACD(close)
        indicators['macd'] = macd.macd().iloc[-1]
        indicators['macd_signal'] = macd.macd_signal().iloc[-1]
        indicators['macd_hist'] = macd.macd_diff().iloc[-1]
        indicators['macd_hist_prev'] = macd.macd_diff().iloc[-2]
        
        # ADX (Trend Strength)
        indicators['adx'] = ta.trend.adx(high, low, close).iloc[-1]
        indicators['di_plus'] = ta.trend.adx_pos(high, low, close).iloc[-1]
        indicators['di_minus'] = ta.trend.adx_neg(high, low, close).iloc[-1]
        
        # Parabolic SAR
        psar = ta.trend.PSARIndicator(high, low, close)
        indicators['psar'] = psar.psar().iloc[-1]
        indicators['psar_up'] = psar.psar_up().iloc[-1] if not pd.isna(psar.psar_up().iloc[-1]) else 0
        indicators['psar_down'] = psar.psar_down().iloc[-1] if not pd.isna(psar.psar_down().iloc[-1]) else 0
        
        # === Momentum Indicators ===
        # RSI
        indicators['rsi_14'] = ta.momentum.rsi(close, window=14).iloc[-1]
        indicators['rsi_7'] = ta.momentum.rsi(close, window=7).iloc[-1]
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        indicators['stoch_k'] = stoch.stoch().iloc[-1]
        indicators['stoch_d'] = stoch.stoch_signal().iloc[-1]
        
        # Williams %R
        indicators['williams_r'] = ta.momentum.williams_r(high, low, close).iloc[-1]
        
        # CCI
        indicators['cci'] = ta.trend.cci(high, low, close).iloc[-1]
        
        # ROC
        indicators['roc'] = ta.momentum.roc(close, window=10).iloc[-1]
        
        # Ultimate Oscillator
        indicators['uo'] = ta.momentum.ultimate_oscillator(high, low, close).iloc[-1]
        
        # === Volatility Indicators ===
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close)
        indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
        indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
        indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle'] * 100
        indicators['bb_pct'] = bb.bollinger_pband().iloc[-1]
        
        # ATR
        indicators['atr'] = ta.volatility.average_true_range(high, low, close).iloc[-1]
        indicators['atr_pct'] = indicators['atr'] / close.iloc[-1] * 100
        
        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(high, low, close)
        indicators['kc_upper'] = kc.keltner_channel_hband().iloc[-1]
        indicators['kc_lower'] = kc.keltner_channel_lband().iloc[-1]
        
        # === Volume Indicators ===
        # OBV
        obv = ta.volume.on_balance_volume(close, volume)
        indicators['obv'] = obv.iloc[-1]
        indicators['obv_sma'] = obv.rolling(20).mean().iloc[-1]
        
        # MFI
        indicators['mfi'] = ta.volume.money_flow_index(high, low, close, volume).iloc[-1]
        
        # VWAP (approximation)
        typical_price = (high + low + close) / 3
        indicators['vwap'] = (typical_price * volume).rolling(20).sum().iloc[-1] / volume.rolling(20).sum().iloc[-1]
        
        # Volume SMA
        indicators['volume_sma'] = volume.rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = volume.iloc[-1] / indicators['volume_sma']
        
        # === Price Data ===
        indicators['close'] = close.iloc[-1]
        indicators['open'] = df['open'].iloc[-1]
        indicators['high'] = high.iloc[-1]
        indicators['low'] = low.iloc[-1]
        indicators['prev_close'] = close.iloc[-2]
        indicators['change_pct'] = (close.iloc[-1] / close.iloc[-2] - 1) * 100
        
        # 52-week high/low
        if len(df) >= 252:
            indicators['high_52w'] = high.tail(252).max()
            indicators['low_52w'] = low.tail(252).min()
        else:
            indicators['high_52w'] = high.max()
            indicators['low_52w'] = low.min()
        
        indicators['pct_from_high'] = (close.iloc[-1] / indicators['high_52w'] - 1) * 100
        indicators['pct_from_low'] = (close.iloc[-1] / indicators['low_52w'] - 1) * 100
        
        return indicators
    
    def _generate_signals(self, df: pd.DataFrame, ind: Dict[str, float]) -> List[TechnicalSignal]:
        """Generate trading signals from indicators"""
        signals = []
        close = ind['close']
        
        # === Moving Average Signals ===
        # Price vs SMA 20
        if close > ind['sma_20'] * 1.02:
            signals.append(TechnicalSignal(
                indicator="SMA 20",
                signal="buy",
                strength=SignalStrength.MODERATE,
                value=ind['sma_20'],
                description=f"Price above SMA 20 (¥{ind['sma_20']:.2f})"
            ))
        elif close < ind['sma_20'] * 0.98:
            signals.append(TechnicalSignal(
                indicator="SMA 20",
                signal="sell",
                strength=SignalStrength.MODERATE,
                value=ind['sma_20'],
                description=f"Price below SMA 20 (¥{ind['sma_20']:.2f})"
            ))
        
        # Golden/Death Cross (SMA 50 vs SMA 200)
        if ind['sma_50'] > ind['sma_200']:
            signals.append(TechnicalSignal(
                indicator="MA Cross",
                signal="buy",
                strength=SignalStrength.STRONG,
                value=0,
                description="Golden Cross: SMA 50 > SMA 200"
            ))
        else:
            signals.append(TechnicalSignal(
                indicator="MA Cross",
                signal="sell",
                strength=SignalStrength.STRONG,
                value=0,
                description="Death Cross: SMA 50 < SMA 200"
            ))
        
        # === MACD Signals ===
        if ind['macd'] > ind['macd_signal']:
            strength = SignalStrength.STRONG if ind['macd_hist'] > ind['macd_hist_prev'] else SignalStrength.MODERATE
            signals.append(TechnicalSignal(
                indicator="MACD",
                signal="buy",
                strength=strength,
                value=ind['macd'],
                description="MACD above signal line"
            ))
        else:
            strength = SignalStrength.STRONG if ind['macd_hist'] < ind['macd_hist_prev'] else SignalStrength.MODERATE
            signals.append(TechnicalSignal(
                indicator="MACD",
                signal="sell",
                strength=strength,
                value=ind['macd'],
                description="MACD below signal line"
            ))
        
        # === RSI Signals ===
        rsi = ind['rsi_14']
        if rsi < 30:
            signals.append(TechnicalSignal(
                indicator="RSI",
                signal="buy",
                strength=SignalStrength.STRONG,
                value=rsi,
                description=f"RSI oversold ({rsi:.1f})"
            ))
        elif rsi < 40:
            signals.append(TechnicalSignal(
                indicator="RSI",
                signal="buy",
                strength=SignalStrength.WEAK,
                value=rsi,
                description=f"RSI approaching oversold ({rsi:.1f})"
            ))
        elif rsi > 70:
            signals.append(TechnicalSignal(
                indicator="RSI",
                signal="sell",
                strength=SignalStrength.STRONG,
                value=rsi,
                description=f"RSI overbought ({rsi:.1f})"
            ))
        elif rsi > 60:
            signals.append(TechnicalSignal(
                indicator="RSI",
                signal="sell",
                strength=SignalStrength.WEAK,
                value=rsi,
                description=f"RSI approaching overbought ({rsi:.1f})"
            ))
        else:
            signals.append(TechnicalSignal(
                indicator="RSI",
                signal="neutral",
                strength=SignalStrength.NONE,
                value=rsi,
                description=f"RSI neutral ({rsi:.1f})"
            ))
        
        # === Stochastic Signals ===
        stoch_k = ind['stoch_k']
        stoch_d = ind['stoch_d']
        
        if stoch_k < 20 and stoch_k > stoch_d:
            signals.append(TechnicalSignal(
                indicator="Stochastic",
                signal="buy",
                strength=SignalStrength.STRONG,
                value=stoch_k,
                description=f"Stochastic bullish crossover in oversold ({stoch_k:.1f})"
            ))
        elif stoch_k > 80 and stoch_k < stoch_d:
            signals.append(TechnicalSignal(
                indicator="Stochastic",
                signal="sell",
                strength=SignalStrength.STRONG,
                value=stoch_k,
                description=f"Stochastic bearish crossover in overbought ({stoch_k:.1f})"
            ))
        
        # === Bollinger Bands Signals ===
        bb_pct = ind['bb_pct']
        if bb_pct < 0:
            signals.append(TechnicalSignal(
                indicator="Bollinger Bands",
                signal="buy",
                strength=SignalStrength.MODERATE,
                value=ind['bb_lower'],
                description="Price below lower Bollinger Band"
            ))
        elif bb_pct > 1:
            signals.append(TechnicalSignal(
                indicator="Bollinger Bands",
                signal="sell",
                strength=SignalStrength.MODERATE,
                value=ind['bb_upper'],
                description="Price above upper Bollinger Band"
            ))
        
        # === CCI Signals ===
        cci = ind['cci']
        if cci < -100:
            signals.append(TechnicalSignal(
                indicator="CCI",
                signal="buy",
                strength=SignalStrength.MODERATE,
                value=cci,
                description=f"CCI oversold ({cci:.1f})"
            ))
        elif cci > 100:
            signals.append(TechnicalSignal(
                indicator="CCI",
                signal="sell",
                strength=SignalStrength.MODERATE,
                value=cci,
                description=f"CCI overbought ({cci:.1f})"
            ))
        
        # === MFI Signals ===
        mfi = ind['mfi']
        if mfi < 20:
            signals.append(TechnicalSignal(
                indicator="MFI",
                signal="buy",
                strength=SignalStrength.MODERATE,
                value=mfi,
                description=f"MFI oversold ({mfi:.1f})"
            ))
        elif mfi > 80:
            signals.append(TechnicalSignal(
                indicator="MFI",
                signal="sell",
                strength=SignalStrength.MODERATE,
                value=mfi,
                description=f"MFI overbought ({mfi:.1f})"
            ))
        
        # === Volume Signals ===
        vol_ratio = ind['volume_ratio']
        if vol_ratio > 2 and ind['change_pct'] > 0:
            signals.append(TechnicalSignal(
                indicator="Volume",
                signal="buy",
                strength=SignalStrength.MODERATE,
                value=vol_ratio,
                description=f"High volume bullish ({vol_ratio:.1f}x average)"
            ))
        elif vol_ratio > 2 and ind['change_pct'] < 0:
            signals.append(TechnicalSignal(
                indicator="Volume",
                signal="sell",
                strength=SignalStrength.MODERATE,
                value=vol_ratio,
                description=f"High volume bearish ({vol_ratio:.1f}x average)"
            ))
        
        # === ADX/DMI Signals ===
        if ind['adx'] > 25:
            if ind['di_plus'] > ind['di_minus']:
                signals.append(TechnicalSignal(
                    indicator="ADX/DMI",
                    signal="buy",
                    strength=SignalStrength.STRONG if ind['adx'] > 40 else SignalStrength.MODERATE,
                    value=ind['adx'],
                    description=f"Strong uptrend (ADX: {ind['adx']:.1f})"
                ))
            else:
                signals.append(TechnicalSignal(
                    indicator="ADX/DMI",
                    signal="sell",
                    strength=SignalStrength.STRONG if ind['adx'] > 40 else SignalStrength.MODERATE,
                    value=ind['adx'],
                    description=f"Strong downtrend (ADX: {ind['adx']:.1f})"
                ))
        
        return signals
    
    def _analyze_trend(self, df: pd.DataFrame, ind: Dict[str, float]) -> Tuple[TrendDirection, float]:
        """Analyze overall trend"""
        close = ind['close']
        
        # Multiple timeframe trend analysis
        scores = []
        
        # Short-term (5/10 MA)
        if close > ind['sma_5'] > ind['sma_10']:
            scores.append(2)
        elif close < ind['sma_5'] < ind['sma_10']:
            scores.append(-2)
        else:
            scores.append(0)
        
        # Medium-term (20/50 MA)
        if close > ind['sma_20'] > ind['sma_50']:
            scores.append(2)
        elif close < ind['sma_20'] < ind['sma_50']:
            scores.append(-2)
        else:
            scores.append(0)
        
        # Long-term (50/200 MA)
        if ind['sma_50'] > ind['sma_200']:
            scores.append(2)
        else:
            scores.append(-2)
        
        # MACD trend
        if ind['macd'] > 0 and ind['macd'] > ind['macd_signal']:
            scores.append(1)
        elif ind['macd'] < 0 and ind['macd'] < ind['macd_signal']:
            scores.append(-1)
        else:
            scores.append(0)
        
        # ADX trend strength
        if ind['adx'] > 25:
            adx_multiplier = 1 + (ind['adx'] - 25) / 50
            if ind['di_plus'] > ind['di_minus']:
                scores.append(1 * adx_multiplier)
            else:
                scores.append(-1 * adx_multiplier)
        
        total_score = sum(scores)
        max_score = 10  # Approximate maximum
        trend_strength = abs(total_score) / max_score
        
        if total_score >= 6:
            trend = TrendDirection.STRONG_UP
        elif total_score >= 3:
            trend = TrendDirection.UP
        elif total_score <= -6:
            trend = TrendDirection.STRONG_DOWN
        elif total_score <= -3:
            trend = TrendDirection.DOWN
        else:
            trend = TrendDirection.SIDEWAYS
        
        return trend, min(trend_strength, 1.0)
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> SupportResistance:
        """Calculate support and resistance levels using pivot points"""
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        # Standard pivot point
        pivot = (high + low + close) / 3
        
        # Support levels
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        # Resistance levels
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
    
    def _calculate_overall(self, signals: List[TechnicalSignal], 
                           trend: TrendDirection) -> Tuple[float, str]:
        """Calculate overall score and signal"""
        score = 0
        
        # Weight signals by strength
        for signal in signals:
            weight = signal.strength.value
            if signal.signal == "buy":
                score += weight * 10
            elif signal.signal == "sell":
                score -= weight * 10
        
        # Add trend bias
        trend_scores = {
            TrendDirection.STRONG_UP: 20,
            TrendDirection.UP: 10,
            TrendDirection.SIDEWAYS: 0,
            TrendDirection.DOWN: -10,
            TrendDirection.STRONG_DOWN: -20
        }
        score += trend_scores.get(trend, 0)
        
        # Normalize to -100 to +100
        score = max(-100, min(100, score))
        
        if score >= 30:
            signal = "buy"
        elif score <= -30:
            signal = "sell"
        else:
            signal = "neutral"
        
        return score, signal
    
    def detect_divergence(self, df: pd.DataFrame, indicator: str = "rsi") -> Optional[str]:
        """
        Detect bullish or bearish divergence
        
        Returns: "bullish", "bearish", or None
        """
        close = df['close'].tail(20)
        
        if indicator == "rsi":
            ind = ta.momentum.rsi(df['close'], window=14).tail(20)
        elif indicator == "macd":
            ind = ta.trend.MACD(df['close']).macd_diff().tail(20)
        else:
            return None
        
        # Find peaks and troughs
        price_higher_high = close.iloc[-1] > close.iloc[-10]
        price_lower_low = close.iloc[-1] < close.iloc[-10]
        ind_higher_high = ind.iloc[-1] > ind.iloc[-10]
        ind_lower_low = ind.iloc[-1] < ind.iloc[-10]
        
        # Bearish divergence: price makes higher high, indicator makes lower high
        if price_higher_high and not ind_higher_high:
            return "bearish"
        
        # Bullish divergence: price makes lower low, indicator makes higher low
        if price_lower_low and not ind_lower_low:
            return "bullish"
        
        return None
    
    def detect_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect candlestick patterns"""
        patterns = []
        
        if len(df) < 5:
            return patterns
        
        o = df['open'].values
        h = df['high'].values
        c = df['close'].values
        l = df['low'].values
        
        # Doji
        body = abs(c[-1] - o[-1])
        total_range = h[-1] - l[-1]
        if total_range > 0 and body / total_range < 0.1:
            patterns.append("Doji")
        
        # Hammer (bullish)
        lower_shadow = min(o[-1], c[-1]) - l[-1]
        upper_shadow = h[-1] - max(o[-1], c[-1])
        if lower_shadow > body * 2 and upper_shadow < body * 0.5:
            patterns.append("Hammer (bullish)")
        
        # Shooting Star (bearish)
        if upper_shadow > body * 2 and lower_shadow < body * 0.5:
            patterns.append("Shooting Star (bearish)")
        
        # Engulfing patterns
        if len(df) >= 2:
            prev_body = abs(c[-2] - o[-2])
            curr_body = abs(c[-1] - o[-1])
            
            # Bullish engulfing
            if c[-2] < o[-2] and c[-1] > o[-1] and curr_body > prev_body:
                if c[-1] > o[-2] and o[-1] < c[-2]:
                    patterns.append("Bullish Engulfing")
            
            # Bearish engulfing
            if c[-2] > o[-2] and c[-1] < o[-1] and curr_body > prev_body:
                if c[-1] < o[-2] and o[-1] > c[-2]:
                    patterns.append("Bearish Engulfing")
        
        # Three white soldiers / Three black crows
        if len(df) >= 3:
            if all(c[-i] > o[-i] for i in range(1, 4)) and all(c[-i] > c[-i-1] for i in range(1, 3)):
                patterns.append("Three White Soldiers (bullish)")
            
            if all(c[-i] < o[-i] for i in range(1, 4)) and all(c[-i] < c[-i-1] for i in range(1, 3)):
                patterns.append("Three Black Crows (bearish)")
        
        return patterns