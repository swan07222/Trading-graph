# data/features.py
"""
Feature Engineering - Technical indicators WITHOUT look-ahead bias

FIXED: Removed all center=True and future-looking operations
All features use only past data (strictly causal)
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import warnings

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)

warnings.filterwarnings('ignore', category=RuntimeWarning)


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
        # Price Position (4)
        'price_position_20', 'price_position_60',
        'distance_from_high', 'distance_from_low',
        # Candlestick (3)
        'body_size', 'upper_shadow', 'lower_shadow',
        # ATR (2)
        'atr_pct', 'atr_ratio',
        # Additional (3)
        'gap', 'range_pct', 'close_position',
    ]
    
    def __init__(self):
        self._ta_available = False
        self._ta = None
        try:
            import ta
            self._ta = ta
            self._ta_available = True
        except ImportError:
            log.warning("'ta' library not available. Using basic features only.")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all technical features - strictly causal.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with all features added
        """
        df = df.copy()
        
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        open_price = df['open']
        
        # === Returns ===
        df['returns'] = close.pct_change() * 100
        with np.errstate(divide='ignore', invalid='ignore'):
            df['log_returns'] = np.log(close / close.shift(1)) * 100
        
        # === Volatility ===
        df['volatility_5'] = df['returns'].rolling(5, min_periods=1).std()
        df['volatility_10'] = df['returns'].rolling(10, min_periods=1).std()
        df['volatility_20'] = df['returns'].rolling(20, min_periods=1).std()
        df['volatility_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-8)
        
        # === Moving Averages ===
        ma5 = close.rolling(5, min_periods=1).mean()
        ma10 = close.rolling(10, min_periods=1).mean()
        ma20 = close.rolling(20, min_periods=1).mean()
        ma50 = close.rolling(50, min_periods=1).mean()
        ma60 = close.rolling(60, min_periods=1).mean()
        
        df['ma_ratio_5_20'] = (ma5 / ma20 - 1) * 100
        df['ma_ratio_10_50'] = (ma10 / ma50 - 1) * 100
        df['ma_ratio_20_60'] = (ma20 / ma60 - 1) * 100
        df['price_to_ma20'] = (close / ma20 - 1) * 100
        
        if self._ta_available:
            self._add_ta_features(df, close, high, low, volume, open_price)
        else:
            self._add_basic_features(df, close, high, low, volume, open_price)
        
        # === Cleanup ===
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().fillna(0)
        
        for col in self.FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        
        log.debug(f"Created {len(self.FEATURE_NAMES)} features, {len(df)} samples")
        
        return df
    
    def _add_ta_features(self, df, close, high, low, volume, open_price):
        """Add features using ta library"""
        ta = self._ta
        
        # === Momentum ===
        try:
            df['rsi_14'] = ta.momentum.rsi(close, window=14, fillna=True) / 100 - 0.5
            df['rsi_7'] = ta.momentum.rsi(close, window=7, fillna=True) / 100 - 0.5
        except Exception:
            df['rsi_14'] = 0.0
            df['rsi_7'] = 0.0
        
        try:
            stoch = ta.momentum.StochasticOscillator(high, low, close, fillna=True)
            df['stoch_k'] = stoch.stoch() / 100 - 0.5
            df['stoch_d'] = stoch.stoch_signal() / 100 - 0.5
        except Exception:
            df['stoch_k'] = 0.0
            df['stoch_d'] = 0.0
        
        try:
            df['williams_r'] = ta.momentum.williams_r(high, low, close, fillna=True) / 100 + 0.5
        except Exception:
            df['williams_r'] = 0.0
        
        df['momentum_5'] = (close / close.shift(5) - 1) * 100
        df['momentum_10'] = (close / close.shift(10) - 1) * 100
        df['momentum_20'] = (close / close.shift(20) - 1) * 100
        
        try:
            df['roc_10'] = ta.momentum.roc(close, window=10, fillna=True)
        except Exception:
            df['roc_10'] = 0.0
        
        try:
            df['uo'] = ta.momentum.ultimate_oscillator(high, low, close, fillna=True) / 100 - 0.5
        except Exception:
            df['uo'] = 0.0
        
        # === MACD ===
        try:
            macd = ta.trend.MACD(close, fillna=True)
            df['macd_line'] = macd.macd() / close * 100
            df['macd_signal'] = macd.macd_signal() / close * 100
            df['macd_hist'] = macd.macd_diff() / close * 100
        except Exception:
            df['macd_line'] = 0.0
            df['macd_signal'] = 0.0
            df['macd_hist'] = 0.0
        
        # === Bollinger Bands ===
        try:
            bb = ta.volatility.BollingerBands(close, fillna=True)
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            bb_range = bb_upper - bb_lower + 1e-8
            df['bb_position'] = ((close - bb_lower) / bb_range * 2 - 1).clip(-1, 1)
            df['bb_width'] = bb_range / close * 100
            df['bb_pct'] = bb.bollinger_pband()
        except Exception:
            df['bb_position'] = 0.0
            df['bb_width'] = 0.0
            df['bb_pct'] = 0.5
        
        # === Volume ===
        vol_ma20 = volume.rolling(20, min_periods=1).mean()
        df['volume_ratio'] = np.log(volume / vol_ma20 + 0.01)
        df['volume_ma_ratio'] = volume / (vol_ma20 + 1)
        
        try:
            obv = ta.volume.on_balance_volume(close, volume)
            df['obv_slope'] = obv.pct_change(5) * 100
        except Exception:
            df['obv_slope'] = 0.0
        
        try:
            df['mfi'] = ta.volume.money_flow_index(high, low, close, volume, fillna=True) / 100 - 0.5
        except Exception:
            df['mfi'] = 0.0
        
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(20, min_periods=1).sum() / (volume.rolling(20, min_periods=1).sum() + 1)
        df['vwap_ratio'] = (close / vwap - 1) * 100
        
        # === Trend ===
        try:
            df['adx'] = ta.trend.adx(high, low, close, fillna=True) / 100
        except Exception:
            df['adx'] = 0.0
        
        try:
            df['cci'] = ta.trend.cci(high, low, close, fillna=True) / 200
        except Exception:
            df['cci'] = 0.0
        
        df['trend_strength'] = df['adx'] * np.sign(df['ma_ratio_5_20'])
        
        try:
            di_plus = ta.trend.adx_pos(high, low, close, fillna=True)
            di_minus = ta.trend.adx_neg(high, low, close, fillna=True)
            df['di_diff'] = (di_plus - di_minus) / 100
        except Exception:
            df['di_diff'] = 0.0
        
        # === Price Position (FIXED: using shift(1) for completed windows) ===
        high_20 = high.rolling(20, min_periods=1).max().shift(1)
        low_20 = low.rolling(20, min_periods=1).min().shift(1)
        high_60 = high.rolling(60, min_periods=1).max().shift(1)
        low_60 = low.rolling(60, min_periods=1).min().shift(1)
        
        df['price_position_20'] = ((close - low_20) / (high_20 - low_20 + 1e-8) * 2 - 1).clip(-1, 1)
        df['price_position_60'] = ((close - low_60) / (high_60 - low_60 + 1e-8) * 2 - 1).clip(-1, 1)
        df['distance_from_high'] = (high_20 - close) / close * 100
        df['distance_from_low'] = (close - low_20) / close * 100
        
        # === Candlestick ===
        df['body_size'] = abs(close - open_price) / (open_price + 1e-8) * 100
        df['upper_shadow'] = (high - pd.concat([open_price, close], axis=1).max(axis=1)) / (open_price + 1e-8) * 100
        df['lower_shadow'] = (pd.concat([open_price, close], axis=1).min(axis=1) - low) / (open_price + 1e-8) * 100
        
        # === ATR ===
        try:
            atr = ta.volatility.average_true_range(high, low, close, fillna=True)
            atr_long = ta.volatility.average_true_range(high, low, close, window=50, fillna=True)
            df['atr_pct'] = atr / close * 100
            df['atr_ratio'] = atr / (atr_long + 1e-8)
        except Exception:
            df['atr_pct'] = 0.0
            df['atr_ratio'] = 1.0
        
        # === Additional ===
        df['gap'] = (open_price / close.shift(1) - 1) * 100
        df['range_pct'] = (high - low) / close * 100
        df['close_position'] = (close - low) / (high - low + 1e-8)
    
    def _add_basic_features(self, df, close, high, low, volume, open_price):
        """Add basic features without ta library"""
        # RSI calculation
        def calc_rsi(series, period):
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        df['rsi_14'] = calc_rsi(close, 14) / 100 - 0.5
        df['rsi_7'] = calc_rsi(close, 7) / 100 - 0.5
        
        # Stochastic
        low_14 = low.rolling(14, min_periods=1).min()
        high_14 = high.rolling(14, min_periods=1).max()
        df['stoch_k'] = ((close - low_14) / (high_14 - low_14 + 1e-8)) - 0.5
        df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=1).mean()
        
        df['williams_r'] = -df['stoch_k']
        
        df['momentum_5'] = (close / close.shift(5) - 1) * 100
        df['momentum_10'] = (close / close.shift(10) - 1) * 100
        df['momentum_20'] = (close / close.shift(20) - 1) * 100
        df['roc_10'] = (close / close.shift(10) - 1) * 100
        df['uo'] = 0.0
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        df['macd_line'] = macd / close * 100
        df['macd_signal'] = signal / close * 100
        df['macd_hist'] = (macd - signal) / close * 100
        
        # Bollinger Bands
        ma20 = close.rolling(20, min_periods=1).mean()
        std20 = close.rolling(20, min_periods=1).std()
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        bb_range = bb_upper - bb_lower + 1e-8
        
        df['bb_position'] = ((close - bb_lower) / bb_range * 2 - 1).clip(-1, 1)
        df['bb_width'] = bb_range / close * 100
        df['bb_pct'] = (close - bb_lower) / bb_range
        
        # Volume
        vol_ma20 = volume.rolling(20, min_periods=1).mean()
        df['volume_ratio'] = np.log(volume / vol_ma20 + 0.01)
        df['volume_ma_ratio'] = volume / (vol_ma20 + 1)
        df['obv_slope'] = 0.0
        df['mfi'] = 0.0
        
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(20, min_periods=1).sum() / (volume.rolling(20, min_periods=1).sum() + 1)
        df['vwap_ratio'] = (close / vwap - 1) * 100
        
        # Trend
        df['adx'] = 0.25
        df['cci'] = 0.0
        df['trend_strength'] = 0.0
        df['di_diff'] = 0.0
        
        # Price Position
        high_20 = high.rolling(20, min_periods=1).max().shift(1)
        low_20 = low.rolling(20, min_periods=1).min().shift(1)
        high_60 = high.rolling(60, min_periods=1).max().shift(1)
        low_60 = low.rolling(60, min_periods=1).min().shift(1)
        
        df['price_position_20'] = ((close - low_20) / (high_20 - low_20 + 1e-8) * 2 - 1).clip(-1, 1)
        df['price_position_60'] = ((close - low_60) / (high_60 - low_60 + 1e-8) * 2 - 1).clip(-1, 1)
        df['distance_from_high'] = (high_20 - close) / close * 100
        df['distance_from_low'] = (close - low_20) / close * 100
        
        # Candlestick
        df['body_size'] = abs(close - open_price) / (open_price + 1e-8) * 100
        df['upper_shadow'] = (high - pd.concat([open_price, close], axis=1).max(axis=1)) / (open_price + 1e-8) * 100
        df['lower_shadow'] = (pd.concat([open_price, close], axis=1).min(axis=1) - low) / (open_price + 1e-8) * 100
        
        # ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=1).mean()
        atr_long = tr.rolling(50, min_periods=1).mean()
        
        df['atr_pct'] = atr / close * 100
        df['atr_ratio'] = atr / (atr_long + 1e-8)
        
        # Additional
        df['gap'] = (open_price / close.shift(1) - 1) * 100
        df['range_pct'] = (high - low) / close * 100
        df['close_position'] = (close - low) / (high - low + 1e-8)
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names"""
        return self.FEATURE_NAMES.copy()