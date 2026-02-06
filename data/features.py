# data/features.py
"""
Feature Engineering - Technical indicators WITHOUT look-ahead bias

FIXED: Removed all center=True and future-looking operations
"""
import pandas as pd
import numpy as np
from typing import List
import ta

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class FeatureEngine:
    """
    Creates technical analysis features from OHLCV data.
    ALL features use only past data (strictly causal).
    """
    
    FEATURE_NAMES = [
        # Returns (2)
        'returns', 'log_returns',
        # Volatility (4)
        'volatility_5', 'volatility_10', 'volatility_20', 'volatility_ratio',
        # Moving Averages (4)
        'ma_ratio_5_20', 'ma_ratio_10_50', 'ma_ratio_20_60', 'price_to_ma20',
        # Momentum (10)
        'rsi_14', 'rsi_7', 'stoch_k', 'stoch_d', 'williams_r',
        'momentum_5', 'momentum_10', 'momentum_20', 'roc_10', 'uo',
        # MACD (3)
        'macd_line', 'macd_signal', 'macd_hist',
        # Bollinger Bands (3)
        'bb_position', 'bb_width', 'bb_pct',
        # Volume (5)
        'volume_ratio', 'volume_ma_ratio', 'obv_slope', 'mfi', 'vwap_ratio',
        # Trend (4)
        'adx', 'cci', 'trend_strength', 'di_diff',
        # Price Position (4) - FIXED: no center=True
        'price_position_20', 'price_position_60',
        'distance_from_high', 'distance_from_low',
        # Candlestick (3)
        'body_size', 'upper_shadow', 'lower_shadow',
        # ATR (2)
        'atr_pct', 'atr_ratio',
        # Additional (3)
        'gap', 'range_pct', 'close_position',
    ]
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all technical features - strictly causal."""
        df = df.copy()
        
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        open_price = df['open']
        
        # === Returns ===
        df['returns'] = close.pct_change() * 100
        df['log_returns'] = np.log(close / close.shift(1)) * 100
        
        # === Volatility (all use past data only) ===
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-8)
        
        # === Moving Averages ===
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma60 = close.rolling(60).mean()
        
        df['ma_ratio_5_20'] = (ma5 / ma20 - 1) * 100
        df['ma_ratio_10_50'] = (ma10 / ma50 - 1) * 100
        df['ma_ratio_20_60'] = (ma20 / ma60 - 1) * 100
        df['price_to_ma20'] = (close / ma20 - 1) * 100
        
        # === Momentum ===
        df['rsi_14'] = ta.momentum.rsi(close, window=14) / 100 - 0.5
        df['rsi_7'] = ta.momentum.rsi(close, window=7) / 100 - 0.5
        
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        df['stoch_k'] = stoch.stoch() / 100 - 0.5
        df['stoch_d'] = stoch.stoch_signal() / 100 - 0.5
        
        df['williams_r'] = ta.momentum.williams_r(high, low, close) / 100 + 0.5
        df['momentum_5'] = (close / close.shift(5) - 1) * 100
        df['momentum_10'] = (close / close.shift(10) - 1) * 100
        df['momentum_20'] = (close / close.shift(20) - 1) * 100
        df['roc_10'] = ta.momentum.roc(close, window=10)
        df['uo'] = ta.momentum.ultimate_oscillator(high, low, close) / 100 - 0.5
        
        # === MACD ===
        macd = ta.trend.MACD(close)
        df['macd_line'] = macd.macd() / close * 100
        df['macd_signal'] = macd.macd_signal() / close * 100
        df['macd_hist'] = macd.macd_diff() / close * 100
        
        # === Bollinger Bands ===
        bb = ta.volatility.BollingerBands(close)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_range = bb_upper - bb_lower + 1e-8
        df['bb_position'] = ((close - bb_lower) / bb_range * 2 - 1).clip(-1, 1)
        df['bb_width'] = bb_range / close * 100
        df['bb_pct'] = bb.bollinger_pband()
        
        # === Volume ===
        vol_ma20 = volume.rolling(20).mean()
        df['volume_ratio'] = np.log(volume / vol_ma20 + 0.01)
        df['volume_ma_ratio'] = volume / vol_ma20
        
        obv = ta.volume.on_balance_volume(close, volume)
        df['obv_slope'] = obv.pct_change(5) * 100
        df['mfi'] = ta.volume.money_flow_index(high, low, close, volume) / 100 - 0.5
        
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        df['vwap_ratio'] = (close / vwap - 1) * 100
        
        # === Trend ===
        df['adx'] = ta.trend.adx(high, low, close) / 100
        df['cci'] = ta.trend.cci(high, low, close) / 200
        df['trend_strength'] = df['adx'] * np.sign(df['ma_ratio_5_20'])
        
        di_plus = ta.trend.adx_pos(high, low, close)
        di_minus = ta.trend.adx_neg(high, low, close)
        df['di_diff'] = (di_plus - di_minus) / 100
        
        # === Price Position (FIXED: no center=True, use shift(1) for completed windows) ===
        high_20 = high.rolling(20).max().shift(1)  # Completed window, excludes current bar
        low_20 = low.rolling(20).min().shift(1)
        high_60 = high.rolling(60).max().shift(1)
        low_60 = low.rolling(60).min().shift(1)
        
        df['price_position_20'] = ((close - low_20) / (high_20 - low_20 + 1e-8) * 2 - 1).clip(-1, 1)
        df['price_position_60'] = ((close - low_60) / (high_60 - low_60 + 1e-8) * 2 - 1).clip(-1, 1)
        df['distance_from_high'] = (high_20 - close) / close * 100
        df['distance_from_low'] = (close - low_20) / close * 100
        
        # === Candlestick ===
        df['body_size'] = abs(close - open_price) / open_price * 100
        df['upper_shadow'] = (high - pd.concat([open_price, close], axis=1).max(axis=1)) / open_price * 100
        df['lower_shadow'] = (pd.concat([open_price, close], axis=1).min(axis=1) - low) / open_price * 100
        
        # === ATR ===
        atr = ta.volatility.average_true_range(high, low, close)
        atr_long = ta.volatility.average_true_range(high, low, close, window=50)
        df['atr_pct'] = atr / close * 100
        df['atr_ratio'] = atr / (atr_long + 1e-8)
        
        # === Additional ===
        df['gap'] = (open_price / close.shift(1) - 1) * 100
        df['range_pct'] = (high - low) / close * 100
        df['close_position'] = (close - low) / (high - low + 1e-8)
        
        # === Cleanup ===
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        log.info(f"Created {len(self.FEATURE_NAMES)} features, {len(df)} valid samples")
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names"""
        return self.FEATURE_NAMES.copy()