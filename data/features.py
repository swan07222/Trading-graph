# data/features.py
import warnings

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# Consistent epsilon for division-by-zero protection
_EPS = 1e-8

class FeatureEngine:
    """Creates technical analysis features from OHLCV data.

    ALL features use only past data (strictly causal).
    """

    FEATURE_NAMES: list[str] = [
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

    # Features that are bounded in [0, 1] or [-1, 1] — fill NaN with 0.5
    _UNIT_INTERVAL_FEATURES = frozenset({
        "bb_pct",
        "close_position",
        "adx",
    })
    _ZERO_CENTERED_BOUNDED_FEATURES = frozenset({
        "price_position_20",
        "price_position_60",
        "bb_position",
    })

    # Features that are centered around 0 — fill NaN with 0.0
    # (all features not in _BOUNDED_FEATURES)

    # Must be >= max lookback window (ma60 needs 60 rows)
    MIN_ROWS = 60

    def __init__(self) -> None:
        self._ta = None
        self._ta_available = False
        try:
            import ta  # noqa: F811

            self._ta = ta
            self._ta_available = True
        except ImportError:
            log.warning("'ta' library not available — using built-in indicators")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all technical features — strictly causal.

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

        # FIX #7: Validate data quality before feature computation
        # Check for all-NaN columns
        for col in required:
            if df[col].isna().all():
                raise ValueError(f"Column '{col}' contains only NaN values")
            # Check for non-positive prices (shouldn't happen in valid data)
            if col in ["open", "high", "low", "close"]:
                if (df[col] <= 0).all():
                    raise ValueError(f"Column '{col}' contains only non-positive values")
        
        # Check for infinite values in critical columns
        for col in required:
            if not np.isfinite(df[col]).all():
                log.warning(f"Column '{col}' contains infinite values, will be handled")

        # Suppress RuntimeWarning only inside this method
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = self._build(df)

        return result

    def get_feature_columns(self) -> list[str]:
        """Return a copy of the canonical feature name list."""
        return self.FEATURE_NAMES.copy()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features with comprehensive NaN handling.
        
        FIX #7: Added explicit NaN handling after each feature computation
        to prevent feature corruption leading to bad predictions.
        """
        out = df.copy()

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
            ratio = close / close.shift(1)
            # FIX: Handle negative/zero ratios correctly for log returns
            # When price decreases, ratio < 1, log should be negative
            # When ratio <= 0 (shouldn't happen with valid prices), use NaN
            # Also handle case where shift(1) produces NaN on first row
            ratio_clean = ratio.where(ratio > 0, np.nan)
            out["log_returns"] = np.log(ratio_clean) * 100
        
        # FIX #7: Explicitly handle NaN in returns
        out["returns"] = out["returns"].fillna(0.0)
        out["log_returns"] = out["log_returns"].fillna(0.0)

        # === Volatility (require full window to avoid garbage) ===
        out["volatility_5"] = out["returns"].rolling(5, min_periods=5).std()
        out["volatility_10"] = out["returns"].rolling(10, min_periods=10).std()
        out["volatility_20"] = out["returns"].rolling(20, min_periods=20).std()
        out["volatility_ratio"] = out["volatility_5"] / (
            out["volatility_20"] + _EPS
        )
        
        # FIX #7: Handle NaN in volatility features
        out["volatility_5"] = out["volatility_5"].fillna(0.0)
        out["volatility_10"] = out["volatility_10"].fillna(0.0)
        out["volatility_20"] = out["volatility_20"].fillna(0.0)
        out["volatility_ratio"] = out["volatility_ratio"].fillna(1.0)  # Neutral ratio

        # === Moving Averages ===
        ma5 = close.rolling(5, min_periods=5).mean()
        ma10 = close.rolling(10, min_periods=10).mean()
        ma20 = close.rolling(20, min_periods=20).mean()
        ma50 = close.rolling(50, min_periods=50).mean()
        ma60 = close.rolling(60, min_periods=60).mean()

        out["ma_ratio_5_20"] = (ma5 / (ma20 + _EPS) - 1) * 100
        out["ma_ratio_10_50"] = (ma10 / (ma50 + _EPS) - 1) * 100
        out["ma_ratio_20_60"] = (ma20 / (ma60 + _EPS) - 1) * 100
        out["price_to_ma20"] = (close / (ma20 + _EPS) - 1) * 100
        
        # FIX #7: Handle NaN in MA ratios
        for col in ["ma_ratio_5_20", "ma_ratio_10_50", "ma_ratio_20_60", "price_to_ma20"]:
            out[col] = out[col].fillna(0.0)

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

        # FIX MATYPE: Ensure all feature columns are numeric before ffill
        for col in self.FEATURE_NAMES:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        # Forward-fill first (propagate last valid), then fill remaining
        # leading NaNs with feature-appropriate defaults
        out[self.FEATURE_NAMES] = (
            out[self.FEATURE_NAMES].ffill()
        )

        # FIX BOUNDED: Fill remaining NaN with appropriate defaults
        # - Bounded [0,1] or [-1,1] features → 0.5 (neutral midpoint)
        # - Ratio/momentum/unbounded features → 0.0 (no signal)
        for col in self.FEATURE_NAMES:
            if col not in out.columns:
                out[col] = 0.0
                log.debug(f"Feature '{col}' was missing — filled with 0.0")
            elif col in self._UNIT_INTERVAL_FEATURES:
                out[col] = out[col].fillna(0.5)
            elif col in self._ZERO_CENTERED_BOUNDED_FEATURES:
                out[col] = out[col].fillna(0.0)
            else:
                out[col] = out[col].fillna(0.0)

        # FIX FEATCOUNT: Verify all features were created
        missing_features = set(self.FEATURE_NAMES) - set(out.columns)
        if missing_features:
            log.error(
                f"Feature creation incomplete — missing: {missing_features}. "
                f"This is a bug in FeatureEngine._build()."
            )
            for col in missing_features:
                out[col] = 0.0

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
        rs = avg_gain / (avg_loss + _EPS)
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
        k = (close - low_n) / (high_n - low_n + _EPS) * 100
        d = k.rolling(d_period, min_periods=d_period).mean()
        return k, d

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _add_momentum(
        self, df: pd.DataFrame,
        close: pd.Series, high: pd.Series,
        low: pd.Series, volume: pd.Series,
    ) -> None:
        # RSI — always use our own Wilder-smoothed implementation
        df["rsi_14"] = self._calc_rsi(close, 14) / 100 - 0.5
        df["rsi_7"] = self._calc_rsi(close, 7) / 100 - 0.5

        # Stochastic — always use our own implementation
        k, d = self._calc_stochastic(close, high, low)
        df["stoch_k"] = k / 100 - 0.5
        df["stoch_d"] = d / 100 - 0.5

        # Williams %R = -(100 - %K) rewritten
        df["williams_r"] = -(100 - k) / 100 + 0.5

        # Momentum — guard against zero/NaN from shift
        shifted_5 = close.shift(5)
        shifted_10 = close.shift(10)
        shifted_20 = close.shift(20)

        df["momentum_5"] = (close / (shifted_5 + _EPS) - 1) * 100
        df["momentum_10"] = (close / (shifted_10 + _EPS) - 1) * 100
        df["momentum_20"] = (close / (shifted_20 + _EPS) - 1) * 100

        df["roc_10"] = (close / (shifted_10 + _EPS) - 1) * 100

        # Ultimate Oscillator — always compute manually for consistency
        df["uo"] = self._calc_ultimate_oscillator(close, high, low) / 100 - 0.5

    @staticmethod
    def _calc_ultimate_oscillator(
        close: pd.Series, high: pd.Series, low: pd.Series,
        s: int = 7, m: int = 14, long_window: int = 28,
    ) -> pd.Series:
        """Ultimate Oscillator (Williams) — no external dependency.

        FIX UONAN: prev_close from shift(1) produces NaN on first row.
        np.minimum/np.maximum propagate NaN correctly, and rolling sums
        with min_periods handle it. No special action needed beyond
        ensuring min_periods is set.
        """
        prev_close = close.shift(1)
        bp = close - pd.Series(
            np.minimum(low.values, prev_close.values),
            index=close.index,
        )
        tr = pd.Series(
            np.maximum(high.values, prev_close.values),
            index=close.index,
        ) - pd.Series(
            np.minimum(low.values, prev_close.values),
            index=close.index,
        )

        avg_s = bp.rolling(s, min_periods=s).sum() / (
            tr.rolling(s, min_periods=s).sum() + _EPS
        )
        avg_m = bp.rolling(m, min_periods=m).sum() / (
            tr.rolling(m, min_periods=m).sum() + _EPS
        )
        avg_l = bp.rolling(long_window, min_periods=long_window).sum() / (
            tr.rolling(long_window, min_periods=long_window).sum() + _EPS
        )
        uo = 100 * (4 * avg_s + 2 * avg_m + avg_l) / 7
        return uo

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def _add_macd(df: pd.DataFrame, close: pd.Series) -> None:
        ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
        ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
        hist = macd_line - signal

        # Normalize by price to make scale-invariant
        # FIX DIVZERO: Use epsilon in denominator
        df["macd_line"] = macd_line / (close + _EPS) * 100
        df["macd_signal"] = signal / (close + _EPS) * 100
        df["macd_hist"] = hist / (close + _EPS) * 100

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def _add_bollinger(df: pd.DataFrame, close: pd.Series, window: int = 20) -> None:
        ma = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        band_range = upper - lower + _EPS

        df["bb_position"] = ((close - lower) / band_range * 2 - 1).clip(-1, 1)
        df["bb_width"] = band_range / (close + _EPS) * 100
        df["bb_pct"] = ((close - lower) / band_range).clip(0, 1)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def _add_volume_features(
        df: pd.DataFrame,
        close: pd.Series, high: pd.Series,
        low: pd.Series, volume: pd.Series,
    ) -> None:
        vol_ma20 = volume.rolling(20, min_periods=20).mean()

        # FIX VOLRATIO: Keep halted/zero-volume bars near neutral instead of
        # emitting extreme finite outliers from log(eps).
        vol_ratio_raw = volume / (vol_ma20 + _EPS)
        vol_ratio_safe = vol_ratio_raw.clip(lower=1e-3, upper=1e3)
        inactive_mask = (volume <= 0) | (vol_ma20 <= 0)
        vol_ratio_safe = vol_ratio_safe.where(~inactive_mask, 1.0)
        df["volume_ratio"] = np.log(vol_ratio_safe)
        df["volume_ma_ratio"] = vol_ratio_raw.where(np.isfinite(vol_ratio_raw), 0.0)

        # FIX OBVSLOPE: OBV slope handles zero OBV gracefully
        sign = np.sign(close.diff())
        obv = (sign * volume).cumsum()
        # Use diff instead of pct_change to avoid division by zero OBV
        obv_diff = obv.diff(5)
        obv_lag = obv.shift(5).abs() + _EPS
        df["obv_slope"] = (obv_diff / obv_lag) * 100

        # MFI (manual — consistent regardless of ta library)
        typical = (high + low + close) / 3
        raw_mf = typical * volume
        delta_tp = typical.diff()
        pos_mf = raw_mf.where(delta_tp > 0, 0.0).rolling(14, min_periods=14).sum()
        # Unchanged typical price contributes to neither positive nor negative flow.
        neg_mf = raw_mf.where(delta_tp < 0, 0.0).rolling(14, min_periods=14).sum()
        mfi = 100 - 100 / (1 + pos_mf / (neg_mf + _EPS))
        df["mfi"] = mfi / 100 - 0.5

        typical_price = (high + low + close) / 3
        cum_tp_vol = (typical_price * volume).rolling(20, min_periods=20).sum()
        cum_vol = volume.rolling(20, min_periods=20).sum()
        vwap = cum_tp_vol / (cum_vol + _EPS)
        df["vwap_ratio"] = (close / (vwap + _EPS) - 1) * 100

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _add_trend(
        self, df: pd.DataFrame,
        close: pd.Series, high: pd.Series, low: pd.Series,
    ) -> None:
        adx_val, di_plus, di_minus = self._calc_adx(close, high, low)
        df["adx"] = adx_val / 100
        df["di_diff"] = (di_plus - di_minus) / 100

        typical = (high + low + close) / 3
        ma_tp = typical.rolling(20, min_periods=20).mean()
        mad = typical.rolling(20, min_periods=20).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        df["cci"] = (typical - ma_tp) / (0.015 * mad + _EPS) / 200

        ma5 = close.rolling(5, min_periods=5).mean()
        ma20 = close.rolling(20, min_periods=20).mean()
        ma_ratio = (ma5 / (ma20 + _EPS) - 1) * 100
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

        plus_dm = pd.Series(plus_dm, index=close.index, dtype=np.float64)
        minus_dm = pd.Series(minus_dm, index=close.index, dtype=np.float64)

        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        smooth_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        smooth_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        di_plus = 100 * smooth_plus / (atr + _EPS)
        di_minus = 100 * smooth_minus / (atr + _EPS)

        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + _EPS)
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        return adx, di_plus, di_minus

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def _add_price_position(
        df: pd.DataFrame,
        close: pd.Series, high: pd.Series, low: pd.Series,
    ) -> None:
        # shift(1): window of COMPLETED bars only (excludes current bar)
        high_20 = high.rolling(20, min_periods=20).max().shift(1)
        low_20 = low.rolling(20, min_periods=20).min().shift(1)
        high_60 = high.rolling(60, min_periods=60).max().shift(1)
        low_60 = low.rolling(60, min_periods=60).min().shift(1)

        range_20 = high_20 - low_20 + _EPS
        range_60 = high_60 - low_60 + _EPS

        df["price_position_20"] = (
            (close - low_20) / range_20 * 2 - 1
        ).clip(-1, 1)
        df["price_position_60"] = (
            (close - low_60) / range_60 * 2 - 1
        ).clip(-1, 1)
        df["distance_from_high"] = (high_20 - close) / (close + _EPS) * 100
        df["distance_from_low"] = (close - low_20) / (close + _EPS) * 100

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def _add_candlestick(
        df: pd.DataFrame,
        close: pd.Series, high: pd.Series,
        low: pd.Series, open_price: pd.Series,
    ) -> None:
        # Vectorized (no pd.concat overhead)
        close_vals = close.values.astype(np.float64)
        open_vals = open_price.values.astype(np.float64)
        high_vals = high.values.astype(np.float64)
        low_vals = low.values.astype(np.float64)

        body_top = np.maximum(open_vals, close_vals)
        body_bottom = np.minimum(open_vals, close_vals)
        denom = np.where(open_vals > _EPS, open_vals, _EPS)

        df["body_size"] = np.abs(close_vals - open_vals) / denom * 100
        df["upper_shadow"] = (high_vals - body_top) / denom * 100
        df["lower_shadow"] = (body_bottom - low_vals) / denom * 100

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def _add_atr(
        df: pd.DataFrame,
        close: pd.Series, high: pd.Series, low: pd.Series,
    ) -> None:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr_14 = tr.rolling(14, min_periods=14).mean()
        atr_50 = tr.rolling(50, min_periods=50).mean()

        df["atr_pct"] = atr_14 / (close + _EPS) * 100
        df["atr_ratio"] = atr_14 / (atr_50 + _EPS)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def _add_additional(
        df: pd.DataFrame,
        close: pd.Series, high: pd.Series,
        low: pd.Series, open_price: pd.Series,
    ) -> None:
        # FIX GAPNAN: gap uses shift(1) which produces NaN on first row.
        # This is expected — the NaN will be handled by ffill + fillna(0)
        # in _build(). Using epsilon in denominator for safety.
        prev_close = close.shift(1)
        df["gap"] = (open_price / (prev_close + _EPS) - 1) * 100
        # Zero out gap where prev_close is NaN (first row) to avoid
        df.loc[prev_close.isna(), "gap"] = 0.0

        df["range_pct"] = (high - low) / (close + _EPS) * 100
        df["close_position"] = (close - low) / (high - low + _EPS)
