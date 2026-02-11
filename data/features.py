# data/features.py
"""
Feature Engineering - Technical indicators WITHOUT look-ahead bias.

All features are strictly causal: only past and current completed bar data
is used. No center=True, no future-looking operations.

Changes from original:
- Proper min_periods to avoid garbage early-row features
- Sentinel NaN fill per feature type (not blanket 0)
- Consistent RSI/Stochastic between ta and basic paths
- Vectorized candlestick math (np.maximum/np.minimum)
- Input length validation
- Per-module warning suppression (not global)
- Explicit feature scale documentation
"""
import warnings
from typing import List

import numpy as np
import pandas as pd

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class FeatureEngine:
    """
    Creates technical analysis features from OHLCV data.

    ALL features use only past data (strictly causal).
    """

    FEATURE_NAMES: List[str] = [
        # Returns (2)
        "returns", "log_returns",
        # Volatility (4)
        "volatility_5", "volatility_10", "volatility_20", "volatility_ratio",
        # Moving Averages (4)
        "ma_ratio_5_20", "ma_ratio_10_50", "ma_ratio_20_60", "price_to_ma20",
        # Momentum (10)
        "rsi_14", "rsi_7", "stoch_k", "stoch_d", "williams_r",
        "momentum_5", "momentum_10", "momentum_20", "roc_10", "uo",
        # MACD (3)
        "macd_line", "macd_signal", "macd_hist",
        # Bollinger Bands (3)
        "bb_position", "bb_width", "bb_pct",
        # Volume (5)
        "volume_ratio", "volume_ma_ratio", "obv_slope", "mfi", "vwap_ratio",
        # Trend (4)
        "adx", "cci", "trend_strength", "di_diff",
        # Price Position (4)
        "price_position_20", "price_position_60",
        "distance_from_high", "distance_from_low",
        # Candlestick (3)
        "body_size", "upper_shadow", "lower_shadow",
        # ATR (2)
        "atr_pct", "atr_ratio",
        # Additional (3)
        "gap", "range_pct", "close_position",
    ]

    # Minimum rows needed to produce meaningful features
    MIN_ROWS = 60

    def __init__(self):
        self._ta = None
        self._ta_available = False
        try:
            import ta  # noqa: F811

            self._ta = ta
            self._ta_available = True
        except ImportError:
            log.warning("'ta' library not available — using built-in indicators")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all technical features — strictly causal.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]

        Returns:
            Copy of *df* with feature columns appended.

        Raises:
            ValueError: on missing columns or insufficient rows.
        """
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(df) < self.MIN_ROWS:
            raise ValueError(
                f"Need >= {self.MIN_ROWS} rows for meaningful features, "
                f"got {len(df)}"
            )

        # Suppress RuntimeWarning only inside this method
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = self._build(df)

        return result

    def get_feature_columns(self) -> List[str]:
        """Return a copy of the canonical feature name list."""
        return self.FEATURE_NAMES.copy()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Ensure numeric
        for col in ("open", "high", "low", "close", "volume"):
            out[col] = pd.to_numeric(out[col], errors="coerce")

        close = out["close"]
        high = out["high"]
        low = out["low"]
        volume = out["volume"]
        open_price = out["open"]

        # === Returns ===
        out["returns"] = close.pct_change() * 100
        with np.errstate(divide="ignore", invalid="ignore"):
            out["log_returns"] = np.log(close / close.shift(1)) * 100

        # === Volatility (require full window to avoid garbage) ===
        out["volatility_5"] = out["returns"].rolling(5, min_periods=5).std()
        out["volatility_10"] = out["returns"].rolling(10, min_periods=10).std()
        out["volatility_20"] = out["returns"].rolling(20, min_periods=20).std()
        out["volatility_ratio"] = out["volatility_5"] / (
            out["volatility_20"] + 1e-8
        )

        # === Moving Averages ===
        ma5 = close.rolling(5, min_periods=5).mean()
        ma10 = close.rolling(10, min_periods=10).mean()
        ma20 = close.rolling(20, min_periods=20).mean()
        ma50 = close.rolling(50, min_periods=50).mean()
        ma60 = close.rolling(60, min_periods=60).mean()

        out["ma_ratio_5_20"] = (ma5 / ma20 - 1) * 100
        out["ma_ratio_10_50"] = (ma10 / ma50 - 1) * 100
        out["ma_ratio_20_60"] = (ma20 / ma60 - 1) * 100
        out["price_to_ma20"] = (close / ma20 - 1) * 100

        # === Indicator suite (consistent between ta / basic) ===
        self._add_momentum(out, close, high, low, volume)
        self._add_macd(out, close)
        self._add_bollinger(out, close)
        self._add_volume_features(out, close, high, low, volume)
        self._add_trend(out, close, high, low)
        self._add_price_position(out, close, high, low)
        self._add_candlestick(out, close, high, low, open_price)
        self._add_atr(out, close, high, low)
        self._add_additional(out, close, high, low, open_price)

        # === Cleanup ===
        out = out.replace([np.inf, -np.inf], np.nan)

        # Forward-fill first (propagate last valid), then fill remaining
        # leading NaNs with feature-appropriate defaults
        out[self.FEATURE_NAMES] = (
            out[self.FEATURE_NAMES].ffill()
        )

        # Fill remaining NaN with 0 for ratio/momentum features,
        # 0.5 for bounded [0,1] features
        bounded_features = {
            "bb_pct", "close_position", "adx",
        }
        for col in self.FEATURE_NAMES:
            if col not in out.columns:
                out[col] = 0.0
            elif col in bounded_features:
                out[col] = out[col].fillna(0.5)
            else:
                out[col] = out[col].fillna(0.0)

        log.debug(
            f"Created {len(self.FEATURE_NAMES)} features, {len(out)} samples"
        )
        return out

    # ------------------------------------------------------------------
    # RSI (consistent Wilder smoothing in both paths)
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
        """Wilder-smoothed RSI — same algorithm regardless of ta library."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100.0 - (100.0 / (1.0 + rs))

    # ------------------------------------------------------------------
    # Stochastic (consistent in both paths)
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_stochastic(
        close: pd.Series, high: pd.Series, low: pd.Series,
        k_period: int = 14, d_period: int = 3,
    ) -> tuple:
        """Raw %K and %D with proper min_periods."""
        low_n = low.rolling(k_period, min_periods=k_period).min()
        high_n = high.rolling(k_period, min_periods=k_period).max()
        k = (close - low_n) / (high_n - low_n + 1e-8) * 100
        d = k.rolling(d_period, min_periods=d_period).mean()
        return k, d

    # ------------------------------------------------------------------
    # Momentum features
    # ------------------------------------------------------------------

    def _add_momentum(
        self, df: pd.DataFrame,
        close: pd.Series, high: pd.Series,
        low: pd.Series, volume: pd.Series,
    ):
        # RSI — always use our own Wilder-smoothed implementation
        df["rsi_14"] = self._calc_rsi(close, 14) / 100 - 0.5
        df["rsi_7"] = self._calc_rsi(close, 7) / 100 - 0.5

        # Stochastic — always use our own implementation
        k, d = self._calc_stochastic(close, high, low)
        df["stoch_k"] = k / 100 - 0.5
        df["stoch_d"] = d / 100 - 0.5

        # Williams %R = -(100 - %K) rewritten
        df["williams_r"] = -(100 - k) / 100 + 0.5

        # Momentum
        df["momentum_5"] = (close / close.shift(5) - 1) * 100
        df["momentum_10"] = (close / close.shift(10) - 1) * 100
        df["momentum_20"] = (close / close.shift(20) - 1) * 100

        # Rate of change
        df["roc_10"] = (close / close.shift(10) - 1) * 100

        # Ultimate Oscillator — always compute manually for consistency
        df["uo"] = self._calc_ultimate_oscillator(close, high, low) / 100 - 0.5

    @staticmethod
    def _calc_ultimate_oscillator(
        close: pd.Series, high: pd.Series, low: pd.Series,
        s: int = 7, m: int = 14, l: int = 28,
    ) -> pd.Series:
        """Ultimate Oscillator (Williams) — no external dependency."""
        prev_close = close.shift(1)
        bp = close - np.minimum(low, prev_close)
        tr = np.maximum(high, prev_close) - np.minimum(low, prev_close)

        avg_s = bp.rolling(s, min_periods=s).sum() / (
            tr.rolling(s, min_periods=s).sum() + 1e-10
        )
        avg_m = bp.rolling(m, min_periods=m).sum() / (
            tr.rolling(m, min_periods=m).sum() + 1e-10
        )
        avg_l = bp.rolling(l, min_periods=l).sum() / (
            tr.rolling(l, min_periods=l).sum() + 1e-10
        )
        uo = 100 * (4 * avg_s + 2 * avg_m + avg_l) / 7
        return uo

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------

    @staticmethod
    def _add_macd(df: pd.DataFrame, close: pd.Series):
        ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
        ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
        hist = macd_line - signal

        # Normalize by price to make scale-invariant
        df["macd_line"] = macd_line / close * 100
        df["macd_signal"] = signal / close * 100
        df["macd_hist"] = hist / close * 100

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------

    @staticmethod
    def _add_bollinger(df: pd.DataFrame, close: pd.Series, window: int = 20):
        ma = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        band_range = upper - lower + 1e-8

        df["bb_position"] = ((close - lower) / band_range * 2 - 1).clip(-1, 1)
        df["bb_width"] = band_range / close * 100
        df["bb_pct"] = (close - lower) / band_range

    # ------------------------------------------------------------------
    # Volume features
    # ------------------------------------------------------------------

    @staticmethod
    def _add_volume_features(
        df: pd.DataFrame,
        close: pd.Series, high: pd.Series,
        low: pd.Series, volume: pd.Series,
    ):
        vol_ma20 = volume.rolling(20, min_periods=20).mean()

        # Fixed: epsilon in denominator, not added to ratio
        df["volume_ratio"] = np.log(volume / (vol_ma20 + 1e-8) + 1e-8)
        df["volume_ma_ratio"] = volume / (vol_ma20 + 1e-8)

        # OBV slope
        sign = np.sign(close.diff())
        obv = (sign * volume).cumsum()
        df["obv_slope"] = obv.pct_change(5) * 100

        # MFI (manual — consistent regardless of ta library)
        typical = (high + low + close) / 3
        raw_mf = typical * volume
        delta_tp = typical.diff()
        pos_mf = raw_mf.where(delta_tp > 0, 0.0).rolling(14, min_periods=14).sum()
        neg_mf = raw_mf.where(delta_tp <= 0, 0.0).rolling(14, min_periods=14).sum()
        mfi = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-10))
        df["mfi"] = mfi / 100 - 0.5

        # VWAP ratio — fixed epsilon in denominator
        typical_price = (high + low + close) / 3
        cum_tp_vol = (typical_price * volume).rolling(20, min_periods=20).sum()
        cum_vol = volume.rolling(20, min_periods=20).sum()
        vwap = cum_tp_vol / (cum_vol + 1e-8)
        df["vwap_ratio"] = (close / vwap - 1) * 100

    # ------------------------------------------------------------------
    # Trend features
    # ------------------------------------------------------------------

    def _add_trend(
        self, df: pd.DataFrame,
        close: pd.Series, high: pd.Series, low: pd.Series,
    ):
        adx_val, di_plus, di_minus = self._calc_adx(close, high, low)
        df["adx"] = adx_val / 100
        df["di_diff"] = (di_plus - di_minus) / 100

        # CCI
        typical = (high + low + close) / 3
        ma_tp = typical.rolling(20, min_periods=20).mean()
        mad = typical.rolling(20, min_periods=20).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        df["cci"] = (typical - ma_tp) / (0.015 * mad + 1e-10) / 200

        # Trend strength — handle sign(0) → use small epsilon
        ma5 = close.rolling(5, min_periods=5).mean()
        ma20 = close.rolling(20, min_periods=20).mean()
        ma_ratio = (ma5 / ma20 - 1) * 100
        df["trend_strength"] = df["adx"] * np.sign(ma_ratio)

    @staticmethod
    def _calc_adx(
        close: pd.Series, high: pd.Series, low: pd.Series,
        period: int = 14,
    ) -> tuple:
        """Compute ADX, +DI, -DI with Wilder smoothing."""
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm = pd.Series(plus_dm, index=close.index)
        minus_dm = pd.Series(minus_dm, index=close.index)

        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        smooth_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        smooth_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        di_plus = 100 * smooth_plus / (atr + 1e-10)
        di_minus = 100 * smooth_minus / (atr + 1e-10)

        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        return adx, di_plus, di_minus

    # ------------------------------------------------------------------
    # Price position
    # ------------------------------------------------------------------

    @staticmethod
    def _add_price_position(
        df: pd.DataFrame,
        close: pd.Series, high: pd.Series, low: pd.Series,
    ):
        # shift(1): window of COMPLETED bars only (excludes current bar)
        high_20 = high.rolling(20, min_periods=20).max().shift(1)
        low_20 = low.rolling(20, min_periods=20).min().shift(1)
        high_60 = high.rolling(60, min_periods=60).max().shift(1)
        low_60 = low.rolling(60, min_periods=60).min().shift(1)

        df["price_position_20"] = (
            (close - low_20) / (high_20 - low_20 + 1e-8) * 2 - 1
        ).clip(-1, 1)
        df["price_position_60"] = (
            (close - low_60) / (high_60 - low_60 + 1e-8) * 2 - 1
        ).clip(-1, 1)
        df["distance_from_high"] = (high_20 - close) / (close + 1e-8) * 100
        df["distance_from_low"] = (close - low_20) / (close + 1e-8) * 100

    # ------------------------------------------------------------------
    # Candlestick
    # ------------------------------------------------------------------

    @staticmethod
    def _add_candlestick(
        df: pd.DataFrame,
        close: pd.Series, high: pd.Series,
        low: pd.Series, open_price: pd.Series,
    ):
        # Vectorized (no pd.concat overhead)
        body_top = np.maximum(open_price.values, close.values)
        body_bottom = np.minimum(open_price.values, close.values)
        denom = np.where(open_price.values > 1e-4, open_price.values, 1e-4)

        df["body_size"] = np.abs(close.values - open_price.values) / denom * 100
        df["upper_shadow"] = (high.values - body_top) / denom * 100
        df["lower_shadow"] = (body_bottom - low.values) / denom * 100

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    @staticmethod
    def _add_atr(
        df: pd.DataFrame,
        close: pd.Series, high: pd.Series, low: pd.Series,
    ):
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr_14 = tr.rolling(14, min_periods=14).mean()
        atr_50 = tr.rolling(50, min_periods=50).mean()

        df["atr_pct"] = atr_14 / (close + 1e-8) * 100
        df["atr_ratio"] = atr_14 / (atr_50 + 1e-8)

    # ------------------------------------------------------------------
    # Additional
    # ------------------------------------------------------------------

    @staticmethod
    def _add_additional(
        df: pd.DataFrame,
        close: pd.Series, high: pd.Series,
        low: pd.Series, open_price: pd.Series,
    ):
        df["gap"] = (open_price / close.shift(1) - 1) * 100
        df["range_pct"] = (high - low) / (close + 1e-8) * 100
        df["close_position"] = (close - low) / (high - low + 1e-8)