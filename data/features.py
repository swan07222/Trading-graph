"""
Feature Engineering - Create technical indicators
"""
import pandas as pd
import numpy as np
from typing import List
import ta

from config import CONFIG
from utils.logger import log


class FeatureEngine:
    """
    Creates technical analysis features from OHLCV data
    All features are normalized to similar scales
    """
    
    # List of feature names created
    FEATURE_NAMES = [
        # Returns
        'returns', 'log_returns',
        # Volatility
        'volatility_5', 'volatility_20', 'volatility_ratio',
        # Moving Averages
        'ma_ratio_5_20', 'ma_ratio_20_60', 'price_to_ma20',
        # Momentum
        'rsi_14', 'rsi_6', 'stoch_k', 'stoch_d', 'williams_r',
        'momentum_5', 'momentum_20', 'roc_10',
        # MACD
        'macd_line', 'macd_signal', 'macd_hist',
        # Bollinger Bands
        'bb_position', 'bb_width',
        # Volume
        'volume_ratio', 'volume_ma_ratio', 'obv_slope', 'mfi',
        # Trend
        'adx', 'cci', 'trend_strength',
        # Price Position
        'price_position_20', 'price_position_60',
        'distance_from_high', 'distance_from_low',
        # Candlestick
        'body_size', 'upper_shadow', 'lower_shadow',
        # ATR
        'atr_pct',
    ]
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all technical features
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added feature columns
        """
        df = df.copy()
        
        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        
        # === Returns ===
        df['returns'] = df['close'].pct_change() * 100
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1)) * 100
        
        # === Volatility ===
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-8)
        
        # === Moving Averages ===
        ma5 = df['close'].rolling(5).mean()
        ma20 = df['close'].rolling(20).mean()
        ma60 = df['close'].rolling(60).mean()
        
        df['ma_ratio_5_20'] = (ma5 / ma20 - 1) * 100
        df['ma_ratio_20_60'] = (ma20 / ma60 - 1) * 100
        df['price_to_ma20'] = (df['close'] / ma20 - 1) * 100
        
        # === Momentum Indicators ===
        # RSI
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14) / 100 - 0.5
        df['rsi_6'] = ta.momentum.rsi(df['close'], window=6) / 100 - 0.5
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch() / 100 - 0.5
        df['stoch_d'] = stoch.stoch_signal() / 100 - 0.5
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close']) / 100 + 0.5
        
        # Momentum
        df['momentum_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
        df['momentum_20'] = (df['close'] / df['close'].shift(20) - 1) * 100
        
        # ROC
        df['roc_10'] = ta.momentum.roc(df['close'], window=10)
        
        # === MACD ===
        macd = ta.trend.MACD(df['close'])
        df['macd_line'] = macd.macd() / df['close'] * 100
        df['macd_signal'] = macd.macd_signal() / df['close'] * 100
        df['macd_hist'] = macd.macd_diff() / df['close'] * 100
        
        # === Bollinger Bands ===
        bb = ta.volatility.BollingerBands(df['close'])
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_range = bb_upper - bb_lower + 1e-8
        df['bb_position'] = ((df['close'] - bb_lower) / bb_range * 2 - 1).clip(-1, 1)
        df['bb_width'] = bb_range / df['close'] * 100
        
        # === Volume ===
        vol_ma20 = df['volume'].rolling(20).mean()
        df['volume_ratio'] = np.log(df['volume'] / vol_ma20 + 0.01)
        df['volume_ma_ratio'] = df['volume'] / vol_ma20
        
        # OBV
        obv = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_slope'] = obv.pct_change(5) * 100
        
        # MFI
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume']) / 100 - 0.5
        
        # === Trend ===
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close']) / 100
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close']) / 200
        df['trend_strength'] = df['adx'] * np.sign(df['ma_ratio_5_20'])
        
        # === Price Position ===
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        high_60 = df['high'].rolling(60).max()
        low_60 = df['low'].rolling(60).min()
        
        df['price_position_20'] = ((df['close'] - low_20) / (high_20 - low_20 + 1e-8) * 2 - 1).clip(-1, 1)
        df['price_position_60'] = ((df['close'] - low_60) / (high_60 - low_60 + 1e-8) * 2 - 1).clip(-1, 1)
        df['distance_from_high'] = (high_20 - df['close']) / df['close'] * 100
        df['distance_from_low'] = (df['close'] - low_20) / df['close'] * 100
        
        # === Candlestick Patterns ===
        df['body_size'] = abs(df['close'] - df['open']) / df['open'] * 100
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open'] * 100
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open'] * 100
        
        # === ATR ===
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['atr_pct'] = atr / df['close'] * 100
        
        # === Clean up ===
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        log.info(f"Created {len(self.FEATURE_NAMES)} features, {len(df)} samples")
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names"""
        return self.FEATURE_NAMES.copy()