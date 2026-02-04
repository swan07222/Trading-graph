"""
Advanced Feature Engineering for Maximum Prediction Accuracy
Includes market microstructure, alternative data, and regime detection
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import ta
from scipy import stats
from scipy.signal import argrelextrema

from utils.logger import log


class AdvancedFeatureEngine:
    """
    State-of-the-art feature engineering including:
    - Market microstructure features
    - Order flow imbalance
    - Regime detection
    - Cross-sectional features
    - Fractal/chaos features
    - Information-theoretic features
    """
    
    FEATURE_GROUPS = {
        'basic': [
            'returns', 'log_returns', 'realized_volatility', 'parkinson_vol',
            'garman_klass_vol', 'rogers_satchell_vol'
        ],
        'momentum': [
            'rsi_14', 'rsi_7', 'rsi_21', 'stoch_k', 'stoch_d', 'williams_r',
            'roc_5', 'roc_10', 'roc_20', 'momentum_5', 'momentum_10', 'momentum_20',
            'tsi', 'ultimate_oscillator', 'awesome_oscillator'
        ],
        'trend': [
            'macd', 'macd_signal', 'macd_hist', 'macd_cross',
            'adx', 'di_plus', 'di_minus', 'di_cross',
            'aroon_up', 'aroon_down', 'aroon_diff',
            'cci', 'dpo', 'kst', 'ichimoku_a', 'ichimoku_b', 'ichimoku_signal'
        ],
        'volatility': [
            'atr_pct', 'bb_width', 'bb_position', 'kc_width', 'kc_position',
            'donchian_width', 'volatility_ratio', 'chaikin_volatility'
        ],
        'volume': [
            'volume_ratio', 'obv_slope', 'mfi', 'cmf', 'eom', 'vpt',
            'nvi', 'pvi', 'force_index', 'volume_price_trend'
        ],
        'microstructure': [
            'spread_estimate', 'kyle_lambda', 'amihud_illiquidity',
            'order_imbalance', 'vpin', 'volume_clock'
        ],
        'pattern': [
            'support_distance', 'resistance_distance', 'pivot_position',
            'trend_strength', 'trend_consistency', 'breakout_score'
        ],
        'regime': [
            'regime_volatility', 'regime_trend', 'regime_mean_reversion',
            'hurst_exponent', 'entropy', 'fractal_dimension'
        ],
        'cross_sectional': [
            'relative_strength', 'momentum_rank', 'volatility_rank',
            'mean_reversion_score'
        ]
    }
    
    def __init__(self):
        self.feature_names = []
        self._build_feature_list()
    
    def _build_feature_list(self):
        """Build complete feature list"""
        self.feature_names = []
        for group, features in self.FEATURE_GROUPS.items():
            self.feature_names.extend(features)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features"""
        df = df.copy()
        
        # Basic features
        df = self._create_basic_features(df)
        
        # Momentum features
        df = self._create_momentum_features(df)
        
        # Trend features
        df = self._create_trend_features(df)
        
        # Volatility features
        df = self._create_volatility_features(df)
        
        # Volume features
        df = self._create_volume_features(df)
        
        # Microstructure features
        df = self._create_microstructure_features(df)
        
        # Pattern features
        df = self._create_pattern_features(df)
        
        # Regime features
        df = self._create_regime_features(df)
        
        # Clean up
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        log.info(f"Created {len(self.feature_names)} features, {len(df)} samples")
        
        return df
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic price and return features"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Returns
        df['returns'] = close.pct_change() * 100
        df['log_returns'] = np.log(close / close.shift(1)) * 100
        
        # Volatility estimators
        df['realized_volatility'] = df['returns'].rolling(20).std()
        
        # Parkinson volatility (uses high-low)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            ((np.log(high / low)) ** 2).rolling(20).mean()
        ) * 100
        
        # Garman-Klass volatility
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / df['open']) ** 2
        df['garman_klass_vol'] = np.sqrt(
            0.5 * log_hl.rolling(20).mean() - 
            (2 * np.log(2) - 1) * log_co.rolling(20).mean()
        ) * 100
        
        # Rogers-Satchell volatility
        log_ho = np.log(high / df['open'])
        log_hc = np.log(high / close)
        log_lo = np.log(low / df['open'])
        log_lc = np.log(low / close)
        df['rogers_satchell_vol'] = np.sqrt(
            (log_ho * log_hc + log_lo * log_lc).rolling(20).mean()
        ) * 100
        
        return df
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI variations
        df['rsi_14'] = ta.momentum.rsi(close, window=14) / 100 - 0.5
        df['rsi_7'] = ta.momentum.rsi(close, window=7) / 100 - 0.5
        df['rsi_21'] = ta.momentum.rsi(close, window=21) / 100 - 0.5
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        df['stoch_k'] = stoch.stoch() / 100 - 0.5
        df['stoch_d'] = stoch.stoch_signal() / 100 - 0.5
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(high, low, close) / 100 + 0.5
        
        # ROC
        df['roc_5'] = ta.momentum.roc(close, window=5)
        df['roc_10'] = ta.momentum.roc(close, window=10)
        df['roc_20'] = ta.momentum.roc(close, window=20)
        
        # Momentum
        df['momentum_5'] = (close / close.shift(5) - 1) * 100
        df['momentum_10'] = (close / close.shift(10) - 1) * 100
        df['momentum_20'] = (close / close.shift(20) - 1) * 100
        
        # TSI
        df['tsi'] = ta.momentum.tsi(close) / 100
        
        # Ultimate Oscillator
        df['ultimate_oscillator'] = ta.momentum.ultimate_oscillator(high, low, close) / 100 - 0.5
        
        # Awesome Oscillator
        df['awesome_oscillator'] = ta.momentum.awesome_oscillator(high, low) / close * 100
        
        return df
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # MACD
        macd = ta.trend.MACD(close)
        df['macd'] = macd.macd() / close * 100
        df['macd_signal'] = macd.macd_signal() / close * 100
        df['macd_hist'] = macd.macd_diff() / close * 100
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(float) - 0.5
        
        # ADX
        df['adx'] = ta.trend.adx(high, low, close) / 100
        df['di_plus'] = ta.trend.adx_pos(high, low, close) / 100
        df['di_minus'] = ta.trend.adx_neg(high, low, close) / 100
        df['di_cross'] = (df['di_plus'] > df['di_minus']).astype(float) - 0.5
        
        # Aroon
        aroon = ta.trend.AroonIndicator(high, low)
        df['aroon_up'] = aroon.aroon_up() / 100 - 0.5
        df['aroon_down'] = aroon.aroon_down() / 100 - 0.5
        df['aroon_diff'] = (df['aroon_up'] - df['aroon_down'])
        
        # CCI
        df['cci'] = ta.trend.cci(high, low, close) / 200
        
        # DPO
        df['dpo'] = ta.trend.dpo(close) / close * 100
        
        # KST
        df['kst'] = ta.trend.kst(close) / close * 100
        
        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(high, low)
        df['ichimoku_a'] = (ichimoku.ichimoku_a() - close) / close * 100
        df['ichimoku_b'] = (ichimoku.ichimoku_b() - close) / close * 100
        df['ichimoku_signal'] = (
            (close > ichimoku.ichimoku_a()) & 
            (close > ichimoku.ichimoku_b())
        ).astype(float) - 0.5
        
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR
        df['atr_pct'] = ta.volatility.average_true_range(high, low, close) / close * 100
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        df['bb_width'] = (bb_upper - bb_lower) / close * 100
        df['bb_position'] = ((close - bb_lower) / (bb_upper - bb_lower + 1e-8) * 2 - 1).clip(-1, 1)
        
        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(high, low, close)
        kc_upper = kc.keltner_channel_hband()
        kc_lower = kc.keltner_channel_lband()
        df['kc_width'] = (kc_upper - kc_lower) / close * 100
        df['kc_position'] = ((close - kc_lower) / (kc_upper - kc_lower + 1e-8) * 2 - 1).clip(-1, 1)
        
        # Donchian Channel
        dc = ta.volatility.DonchianChannel(high, low, close)
        dc_upper = dc.donchian_channel_hband()
        dc_lower = dc.donchian_channel_lband()
        df['donchian_width'] = (dc_upper - dc_lower) / close * 100
        
        # Volatility ratio
        df['volatility_ratio'] = df['realized_volatility'] / df['realized_volatility'].rolling(60).mean()
        
        # Chaikin Volatility
        hl_ema = (high - low).ewm(span=10).mean()
        df['chaikin_volatility'] = (hl_ema / hl_ema.shift(10) - 1) * 100
        
        return df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Volume ratio
        vol_ma = volume.rolling(20).mean()
        df['volume_ratio'] = np.log(volume / vol_ma + 0.01)
        
        # OBV
        obv = ta.volume.on_balance_volume(close, volume)
        df['obv_slope'] = obv.pct_change(5) * 100
        
        # MFI
        df['mfi'] = ta.volume.money_flow_index(high, low, close, volume) / 100 - 0.5
        
        # CMF
        df['cmf'] = ta.volume.chaikin_money_flow(high, low, close, volume)
        
        # EOM
        df['eom'] = ta.volume.ease_of_movement(high, low, volume) / close * 1000
        
        # VPT
        vpt = ta.volume.volume_price_trend(close, volume)
        df['vpt'] = vpt.pct_change(10) * 100
        
        # NVI and PVI
        df['nvi'] = ta.volume.negative_volume_index(close, volume).pct_change(20) * 100
        df['pvi'] = ta.volume.positive_volume_index(close, volume).pct_change(20) * 100
        
        # Force Index
        df['force_index'] = ta.volume.force_index(close, volume) / (close * vol_ma) * 100
        
        # Volume-Price Trend
        df['volume_price_trend'] = ((close - close.shift(1)) / close.shift(1) * volume).rolling(20).sum() / vol_ma
        
        return df
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Spread estimate (Corwin-Schultz)
        gamma = (np.log(high / low)) ** 2
        beta = gamma + gamma.shift(1)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
        df['spread_estimate'] = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        df['spread_estimate'] = df['spread_estimate'].clip(0, 0.1)
        
        # Kyle's Lambda (price impact)
        returns = close.pct_change()
        signed_volume = volume * np.sign(returns)
        df['kyle_lambda'] = returns.rolling(20).cov(signed_volume) / signed_volume.rolling(20).var()
        df['kyle_lambda'] = df['kyle_lambda'].clip(-0.001, 0.001) * 1000
        
        # Amihud Illiquidity
        df['amihud_illiquidity'] = (np.abs(returns) / (volume * close)).rolling(20).mean() * 1e9
        df['amihud_illiquidity'] = df['amihud_illiquidity'].clip(0, 100)
        
        # Order imbalance proxy
        df['order_imbalance'] = (close - low) / (high - low + 1e-8) * 2 - 1
        
        # VPIN (Volume-Synchronized Probability of Informed Trading) - simplified
        buy_volume = volume * ((close - low) / (high - low + 1e-8))
        sell_volume = volume - buy_volume
        df['vpin'] = (np.abs(buy_volume - sell_volume) / volume).rolling(20).mean()
        
        # Volume clock (time between volume buckets)
        cum_vol = volume.cumsum()
        bucket_size = volume.rolling(20).sum() / 20
        df['volume_clock'] = (cum_vol / bucket_size).diff().clip(0, 5)
        
        return df
    
    def _create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pattern recognition features"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Support and Resistance
        window = 20
        local_max = high.rolling(window, center=True).max()
        local_min = low.rolling(window, center=True).min()
        
        df['resistance_distance'] = (local_max - close) / close * 100
        df['support_distance'] = (close - local_min) / close * 100
        
        # Pivot position
        pivot = (high + low + close) / 3
        df['pivot_position'] = (close - pivot.shift(1)) / pivot.shift(1) * 100
        
        # Trend strength
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        df['trend_strength'] = (ma20 - ma50) / close * 100
        
        # Trend consistency (how often price moves in trend direction)
        trend_dir = np.sign(df['trend_strength'])
        return_dir = np.sign(close.pct_change())
        df['trend_consistency'] = (trend_dir == return_dir).rolling(20).mean() * 2 - 1
        
        # Breakout score
        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()
        range_20 = high_20 - low_20
        df['breakout_score'] = (close - (high_20 + low_20) / 2) / (range_20 / 2 + 1e-8)
        
        return df
    
    def _create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market regime detection features"""
        close = df['close']
        returns = close.pct_change()
        
        # Volatility regime
        vol = returns.rolling(20).std()
        vol_ma = vol.rolling(60).mean()
        df['regime_volatility'] = (vol / vol_ma).clip(0.5, 2) - 1
        
        # Trend regime
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        df['regime_trend'] = np.sign(ma20 - ma60) * (np.abs(ma20 - ma60) / close * 100).clip(0, 10) / 10
        
        # Mean reversion regime
        zscore = (close - ma20) / (returns.rolling(20).std() * np.sqrt(20) + 1e-8)
        df['regime_mean_reversion'] = -zscore.clip(-3, 3) / 3
        
        # Hurst exponent (simplified)
        def hurst(ts, max_lag=20):
            lags = range(2, max_lag)
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            if min(tau) <= 0:
                return 0.5
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        
        df['hurst_exponent'] = returns.rolling(100).apply(
            lambda x: hurst(x.values) if len(x.dropna()) >= 50 else 0.5,
            raw=False
        )
        df['hurst_exponent'] = df['hurst_exponent'].fillna(0.5)
        
        # Entropy (market randomness)
        def entropy(ts, bins=10):
            if len(ts.dropna()) < 20:
                return 0.5
            hist, _ = np.histogram(ts.dropna(), bins=bins, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist + 1e-8)) / np.log(bins)
        
        df['entropy'] = returns.rolling(50).apply(
            lambda x: entropy(x),
            raw=False
        )
        df['entropy'] = df['entropy'].fillna(0.5)
        
        # Fractal dimension (simplified)
        def fractal_dim(ts):
            if len(ts.dropna()) < 20:
                return 1.5
            n = len(ts)
            max_k = min(n // 4, 10)
            lengths = []
            for k in range(1, max_k + 1):
                length = 0
                for m in range(k):
                    idx = np.arange(m, n, k)
                    if len(idx) > 1:
                        length += np.sum(np.abs(np.diff(ts.iloc[idx])))
                lengths.append(length * (n - 1) / (k * ((n - m - 1) // k) * k))
            
            if len(lengths) < 2 or min(lengths) <= 0:
                return 1.5
            
            x = np.log(np.arange(1, len(lengths) + 1))
            y = np.log(np.array(lengths) + 1e-8)
            slope = np.polyfit(x, y, 1)[0]
            return 2 + slope
        
        df['fractal_dimension'] = close.rolling(50).apply(
            lambda x: fractal_dim(x),
            raw=False
        )
        df['fractal_dimension'] = (df['fractal_dimension'].fillna(1.5) - 1) / 1  # Normalize to ~0-1
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get all feature column names"""
        all_features = []
        for features in self.FEATURE_GROUPS.values():
            all_features.extend(features)
        return all_features