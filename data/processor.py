# data/processor.py
from __future__ import annotations

import copy
import pickle
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)

# Module-level constant for feature scaling clamp
FEATURE_CLIP_VALUE = 5.0

# Class label names — dynamically generated if NUM_CLASSES != 3
_DEFAULT_LABEL_NAMES = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

class FeatureEngineProtocol(Protocol):
    """Protocol for type-checking feature engine compatibility."""

    MIN_ROWS: int

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def get_feature_columns(self) -> list[str]: ...

class RealtimeBuffer:
    """
    Thread-safe circular buffer for real-time data streaming.
    Maintains the last N bars for sequence construction.

    All public methods are individually thread-safe via RLock.
    For compound check-then-act patterns, callers should use
    ``with buffer.lock:`` to hold the lock across multiple calls.
    """

    def __init__(self, max_size: int | None = None):
        self.max_size = max_size or CONFIG.SEQUENCE_LENGTH * 2
        self._buffer: deque = deque(maxlen=self.max_size)
        self._lock = threading.RLock()
        self._last_update: datetime | None = None

    @property
    def lock(self) -> threading.RLock:
        """Expose lock for compound atomic operations."""
        return self._lock

    def append(self, row: dict[str, float]) -> None:
        """Add a new bar to the buffer."""
        with self._lock:
            self._buffer.append(row)
            self._last_update = datetime.now()

    def extend(self, rows: list[dict[str, float]]) -> None:
        """Add multiple bars to the buffer."""
        with self._lock:
            for row in rows:
                self._buffer.append(row)
            if rows:
                self._last_update = datetime.now()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert buffer to DataFrame."""
        with self._lock:
            if not self._buffer:
                return pd.DataFrame()
            return pd.DataFrame(list(self._buffer))

    def get_latest(self, n: int | None = None) -> pd.DataFrame:
        """Get the latest N bars as DataFrame."""
        n = n or CONFIG.SEQUENCE_LENGTH
        with self._lock:
            if len(self._buffer) < n:
                return pd.DataFrame()
            data = list(self._buffer)[-n:]
            return pd.DataFrame(data)

    def is_ready(self, min_bars: int | None = None) -> bool:
        """Check if buffer has enough data for prediction."""
        min_bars = min_bars or CONFIG.SEQUENCE_LENGTH
        with self._lock:
            return len(self._buffer) >= min_bars

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._last_update = None

    def append_and_snapshot(
        self, row: dict[str, float], min_required: int
    ) -> pd.DataFrame | None:
        """
        Atomically append a row and return a DataFrame snapshot
        if the buffer has >= min_required rows, else None.

        This eliminates the TOCTOU race between is_ready() and
        to_dataframe().
        """
        with self._lock:
            self._buffer.append(row)
            self._last_update = datetime.now()
            if len(self._buffer) < min_required:
                return None
            return pd.DataFrame(list(self._buffer))

    def initialize_from_dataframe(self, df: pd.DataFrame) -> int:
        """
        Atomically clear and initialize buffer from DataFrame.

        Returns the number of rows added.

        FIX THREAD: This replaces the separate clear() + append() calls
        that had a race condition window.
        """
        with self._lock:
            self._buffer.clear()
            count = 0
            for _, row in df.iterrows():
                self._buffer.append(row.to_dict())
                count += 1
            if count > 0:
                self._last_update = datetime.now()
            return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def last_update(self) -> datetime | None:
        with self._lock:
            return self._last_update

class DataProcessor:
    """
    Thread-safe data processor with proper leakage prevention
    and real-time inference support.

    CRITICAL RULES:
    1. fit_scaler() must be called ONLY with training data
    2. Labels near split boundaries are invalidated
    3. Embargo gap prevents any information flow
    4. Real-time predictions use pre-fitted scaler
    5. split_temporal_single_stock computes features WITHIN each split
       when a feature_engine is provided (prevents feature leakage)

    REAL-TIME SUPPORT:
    - prepare_realtime_sequence(): For live bar-by-bar prediction
    - prepare_inference_sequence(): For batch inference
    - RealtimeBuffer: For streaming data management
    """

    def __init__(self) -> None:
        self.scaler: RobustScaler | None = None
        self._fitted = False
        self._lock = threading.RLock()
        self._n_features: int | None = None
        self._fit_samples: int = 0

        self._interval: str = "1d"
        self._horizon: int = CONFIG.PREDICTION_HORIZON
        self._scaler_version: str = ""

        # Real-time buffers per stock
        self._realtime_buffers: dict[str, RealtimeBuffer] = {}
        self._buffer_lock = threading.RLock()

    # =========================================================================
    # =========================================================================

    def create_labels(
        self,
        df: pd.DataFrame,
        horizon: int | None = None,
        up_thresh: float | None = None,
        down_thresh: float | None = None,
        profit_aware: bool | None = None,
    ) -> pd.DataFrame:
        """
        Create classification labels based on future returns.

        Labels:
            0 = DOWN     (return <= down_thresh %)
            1 = NEUTRAL  (down_thresh% < return < up_thresh%)
            2 = UP       (return >= up_thresh %)

        Thresholds are in **percentage points** (e.g. 2.0 means 2%).

        IMPORTANT: The last ``horizon`` rows will have NaN labels.

        Raises:
            ValueError: If thresholds are inconsistent or ``close`` is missing.
        """
        horizon = int(horizon if horizon is not None else CONFIG.PREDICTION_HORIZON)
        up_thresh = float(up_thresh if up_thresh is not None else CONFIG.UP_THRESHOLD)
        down_thresh = float(
            down_thresh if down_thresh is not None else CONFIG.DOWN_THRESHOLD
        )

        # --- threshold validation (Issue 6) ---
        if up_thresh <= 0:
            raise ValueError(f"up_thresh must be positive, got {up_thresh}")
        if down_thresh >= 0:
            raise ValueError(f"down_thresh must be negative, got {down_thresh}")
        if down_thresh >= up_thresh:
            raise ValueError(
                f"down_thresh ({down_thresh}) must be < up_thresh ({up_thresh})"
            )

        df = df.copy()

        if "close" not in df.columns:
            raise ValueError("DataFrame must have 'close' column")

        close = pd.to_numeric(df["close"], errors="coerce")
        future_price = close.shift(-horizon)
        future_return = (future_price / close - 1) * 100

        # Optional: cost-aware neutral band to optimize for tradable moves.
        use_profit_aware = (
            bool(profit_aware)
            if profit_aware is not None
            else bool(getattr(CONFIG.precision, "profit_aware_labels", False))
        )
        if use_profit_aware:
            trading_cost_pct = (
                float(getattr(CONFIG.trading, "commission", 0.0))
                + float(getattr(CONFIG.trading, "slippage", 0.0))
                + float(getattr(CONFIG.trading, "stamp_tax", 0.0))
            ) * 100.0
            cost_buffer = float(
                getattr(CONFIG.precision, "label_cost_buffer_pct", 0.0)
            )
            min_edge = float(
                getattr(CONFIG.precision, "min_label_edge_pct", 0.0)
            )
            required_edge = max(min_edge, trading_cost_pct + cost_buffer)
            up_thresh = max(up_thresh, required_edge)
            down_thresh = min(down_thresh, -required_edge)

        df["label"] = np.nan  # Start as float so NaN is representable
        df.loc[future_return >= up_thresh, "label"] = 2.0  # UP
        df.loc[future_return <= down_thresh, "label"] = 0.0  # DOWN
        # Fill remaining non-NaN positions with NEUTRAL
        valid_return = future_return.notna()
        neutral_mask = valid_return & df["label"].isna()
        df.loc[neutral_mask, "label"] = 1.0  # NEUTRAL

        df["future_return"] = future_return

        # Invalidate last ``horizon`` rows (no future data available)
        if horizon > 0 and len(df) > horizon:
            df.iloc[-horizon:, df.columns.get_loc("label")] = np.nan
            df.iloc[-horizon:, df.columns.get_loc("future_return")] = np.nan

        return df

    # =========================================================================
    # =========================================================================

    def fit_scaler(
        self,
        features: np.ndarray,
        interval: str | None = None,
        horizon: int | None = None,
    ) -> DataProcessor:
        """
        Fit scaler on training data ONLY.
        Thread-safe.
        """
        with self._lock:
            if features.ndim != 2:
                raise ValueError(
                    f"Features must be 2D, got shape {features.shape}"
                )

            valid_mask = ~(
                np.isnan(features).any(axis=1)
                | np.isinf(features).any(axis=1)
            )
            clean_features = features[valid_mask]

            if len(clean_features) < 10:
                raise ValueError(
                    f"Insufficient valid samples for scaler: {len(clean_features)}"
                )

            self.scaler = RobustScaler()
            self.scaler.fit(clean_features)

            self._fitted = True
            self._n_features = features.shape[1]
            self._fit_samples = len(clean_features)
            self._interval = str(interval or self._interval)
            self._horizon = int(horizon or self._horizon)
            self._scaler_version = datetime.now().strftime("%Y%m%d_%H%M%S")

            log.info(
                f"Scaler fitted: {self._fit_samples} samples, "
                f"{self._n_features} features, "
                f"interval={self._interval}, horizon={self._horizon}"
            )

        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using the fitted scaler.
        Clips extreme values to [-FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE]
        for numerical stability.

        FIX TRANSFORM: Handles 1D input gracefully by reshaping.
        """
        with self._lock:
            if not self._fitted:
                raise RuntimeError(
                    "Scaler not fitted! Call fit_scaler() first or load_scaler()."
                )

            # FIX TRANSFORM: Handle 1D input
            original_ndim = features.ndim
            if original_ndim == 1:
                features = features.reshape(1, -1)

            if features.shape[-1] != self._n_features:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self._n_features}, "
                    f"got {features.shape[-1]}"
                )

            original_shape = features.shape
            is_3d = features.ndim == 3

            if is_3d:
                features_2d = features.reshape(-1, features.shape[-1])
            else:
                features_2d = features

            nan_mask = np.isnan(features_2d) | np.isinf(features_2d)
            features_clean = np.nan_to_num(
                features_2d, nan=0.0, posinf=0.0, neginf=0.0
            )

            transformed = self.scaler.transform(features_clean)
            transformed = np.clip(
                transformed, -FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE
            )

            transformed[nan_mask] = 0.0

            if is_3d:
                transformed = transformed.reshape(original_shape)

            result = transformed.astype(np.float32)

            # Restore original ndim if input was 1D
            if original_ndim == 1:
                result = result.reshape(-1)

            return result

    def save_scaler(
        self,
        path: str | Path | None = None,
        interval: str | None = None,
        horizon: int | None = None,
    ) -> None:
        """
        Save scaler atomically with metadata.

        If ``interval`` or ``horizon`` are provided, the instance metadata
        is updated to match (keeps instance and saved file consistent).

        FIX SCALER: Deep-copy the scaler INSIDE the lock so that
        serialization (which happens outside the lock for performance)
        operates on a frozen snapshot. Without this, a concurrent
        fit_scaler() call could mutate self.scaler mid-pickle.
        """
        if not self._fitted:
            log.warning("Scaler not fitted, nothing to save")
            return

        with self._lock:
            if interval is not None:
                self._interval = str(interval)
            if horizon is not None:
                self._horizon = int(horizon)

            save_interval = self._interval
            save_horizon = self._horizon

            if path is None:
                path = CONFIG.MODEL_DIR / f"scaler_{save_interval}_{save_horizon}.pkl"

            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # FIX SCALER: Deep-copy scaler inside lock so serialization
            # operates on a frozen snapshot, not the live object
            data = {
                "scaler": copy.deepcopy(self.scaler),
                "n_features": self._n_features,
                "fit_samples": self._fit_samples,
                "fitted": True,
                "interval": save_interval,
                "horizon": save_horizon,
                "version": self._scaler_version,
                "saved_at": datetime.now().isoformat(),
            }

        # Now safe to serialize outside lock — data is a snapshot
        try:
            from utils.atomic_io import atomic_pickle_dump
            atomic_pickle_dump(path, data)
        except ImportError:
            tmp_path = path.with_suffix(".pkl.tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                try:
                    import os
                    os.fsync(f.fileno())
                except OSError:
                    pass
            tmp_path.replace(path)

        log.info(
            f"Scaler saved: {path} (interval={save_interval}, horizon={save_horizon})"
        )

    def load_scaler(
        self,
        path: str | Path | None = None,
        interval: str | None = None,
        horizon: int | None = None,
    ) -> bool:
        """
        Load saved scaler for inference.

        WARNING: Uses pickle.load which can execute arbitrary code.
        Only load scalers from trusted sources.
        """
        if path is None:
            interval = interval or "1d"
            horizon = horizon or CONFIG.PREDICTION_HORIZON
            path = str(
                CONFIG.MODEL_DIR / f"scaler_{interval}_{horizon}.pkl"
            )

        path = Path(path)
        if not path.exists():
            log.warning(f"Scaler not found: {path}")
            return False

        try:
            # WARNING: pickle.load is inherently unsafe for untrusted data.
            with open(path, "rb") as f:
                data = pickle.load(f)  # noqa: S301

            if not isinstance(data, dict) or "scaler" not in data:
                log.error(f"Invalid scaler file format: {path}")
                return False

            if not isinstance(data["scaler"], RobustScaler):
                log.error(f"Loaded object is not a RobustScaler: {type(data['scaler'])}")
                return False

            with self._lock:
                self.scaler = data["scaler"]
                self._n_features = data.get("n_features")
                self._fit_samples = data.get("fit_samples", 0)
                self._fitted = data.get("fitted", True)
                self._interval = data.get("interval", "1d")
                self._horizon = data.get(
                    "horizon", CONFIG.PREDICTION_HORIZON
                )
                self._scaler_version = data.get("version", "")

            log.info(
                f"Scaler loaded: {path} "
                f"({self._n_features} features, "
                f"interval={self._interval}, horizon={self._horizon})"
            )
            return True

        except Exception as e:
            log.error(f"Failed to load scaler: {e}")
            return False

    @property
    def is_fitted(self) -> bool:
        """Check if scaler is fitted."""
        with self._lock:
            return self._fitted

    @property
    def n_features(self) -> int | None:
        """Get number of features."""
        with self._lock:
            return self._n_features

    # =========================================================================
    # =========================================================================

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        fit_scaler: bool = False,
        return_index: bool = False,
        interval: str | None = None,
        horizon: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Prepare sequences for training/validation.

        Args:
            df: DataFrame with feature columns and 'label' column.
            feature_cols: List of feature column names.
            fit_scaler: If True, fit scaler on valid rows before transforming.
            return_index: If True, also return the DatetimeIndex of each sample.
            interval: Data interval metadata (passed to fit_scaler if fitting).
            horizon: Prediction horizon metadata (passed to fit_scaler if fitting).

        Returns:
            (X, y, returns) or (X, y, returns, index) if return_index=True.

        Note:
            Empty results return arrays with correct ndim:
            - X: shape (0, seq_len, n_features)
            - y: shape (0,)
            - returns: shape (0,)
        """
        seq_len = int(CONFIG.SEQUENCE_LENGTH)
        num_classes = int(CONFIG.NUM_CLASSES)
        n_features = len(feature_cols)

        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        if "label" not in df.columns:
            raise ValueError(
                "DataFrame must have 'label' column. "
                "Call create_labels() first."
            )

        # FIX SHAPE: Define properly shaped empty arrays for early return
        empty_X = np.zeros((0, seq_len, n_features), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.int64)
        empty_r = np.zeros((0,), dtype=np.float32)
        empty_idx = pd.DatetimeIndex([])

        if len(df) < seq_len:
            if return_index:
                return empty_X, empty_y, empty_r, empty_idx
            return empty_X, empty_y, empty_r

        features = df[feature_cols].values.astype(np.float32)

        labels = df["label"].values.astype(np.float64)

        returns = (
            df["future_return"].values.astype(np.float64)
            if "future_return" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )

        if fit_scaler:
            valid_mask = ~np.isnan(labels)
            if valid_mask.sum() > 10:
                self.fit_scaler(
                    features[valid_mask],
                    interval=interval,
                    horizon=horizon,
                )

        if self._fitted:
            features = self.transform(features)
        else:
            # FIX WARN: Log warning when features pass through unscaled
            if not fit_scaler:
                log.warning(
                    "Scaler not fitted and fit_scaler=False — "
                    "features will be clipped but NOT scaled. "
                    "Model predictions may be unreliable."
                )
            features = np.clip(
                np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0),
                -FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE,
            ).astype(np.float32)

        X_list: list[np.ndarray] = []
        y_list: list[int] = []
        r_list: list[float] = []
        idx_list: list[Any] = []
        skipped_invalid = 0

        for i in range(seq_len - 1, len(features)):
            if np.isnan(labels[i]):
                continue

            label_val = int(labels[i])
            if label_val < 0 or label_val >= num_classes:
                skipped_invalid += 1
                continue

            seq = features[i - seq_len + 1: i + 1]

            if len(seq) == seq_len:
                # FIX SEQNAN: Replace NaN in sequences with 0 instead of
                # skipping the entire sequence. After scaling, occasional
                # NaN can appear from edge effects; discarding them wastes
                # valid labels. Zero is the neutral value after RobustScaler.
                if np.isnan(seq).any():
                    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

                X_list.append(seq)
                y_list.append(label_val)
                r_list.append(
                    float(returns[i])
                    if not np.isnan(returns[i])
                    else 0.0
                )
                idx_list.append(df.index[i])

        if skipped_invalid > 0:
            log.warning(
                f"Skipped {skipped_invalid} samples with labels "
                f"outside [0, {num_classes})"
            )

        if not X_list:
            if return_index:
                return empty_X, empty_y, empty_r, empty_idx
            return empty_X, empty_y, empty_r

        out = (
            np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.int64),
            np.array(r_list, dtype=np.float32),
        )

        if return_index:
            return (*out, pd.DatetimeIndex(idx_list))
        return out

    def prepare_forecast_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        horizon: int,
        fit_scaler: bool = False,
        return_index: bool = False,
        interval: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Prepare multi-step forecasting dataset.

        Returns:
            (X, Y) or (X, Y, index) where:
            - X has shape (N, seq_len, n_features)
            - Y has shape (N, horizon) containing future percentage returns

        Note:
            Empty results return properly shaped arrays:
            - X: shape (0, seq_len, n_features)
            - Y: shape (0, horizon)
        """
        seq_len = int(CONFIG.SEQUENCE_LENGTH)
        horizon = int(horizon)
        n_features = len(feature_cols)

        # FIX SHAPE: Define properly shaped empty arrays
        empty_X = np.zeros((0, seq_len, n_features), dtype=np.float32)
        empty_Y = np.zeros((0, horizon), dtype=np.float32)
        empty_idx = pd.DatetimeIndex([])

        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        if "close" not in df.columns:
            raise ValueError(
                "DataFrame must have 'close' column for forecasting"
            )

        if len(df) < seq_len + horizon:
            if return_index:
                return empty_X, empty_Y, empty_idx
            return empty_X, empty_Y

        features = df[feature_cols].values.astype(np.float32)
        close = (
            pd.to_numeric(df["close"], errors="coerce")
            .values.astype(np.float64)
        )

        if fit_scaler:
            valid_mask = ~np.isnan(features).any(axis=1)
            if valid_mask.sum() > 10:
                self.fit_scaler(
                    features[valid_mask],
                    interval=interval,
                    horizon=horizon,
                )

        if self._fitted:
            features = self.transform(features)
        else:
            features = np.clip(
                np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0),
                -FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE,
            ).astype(np.float32)

        X_list: list[np.ndarray] = []
        Y_list: list[np.ndarray] = []
        idx_list: list[Any] = []

        n = len(df)
        for i in range(seq_len - 1, n):
            if i + horizon >= n:
                break

            seq = features[i - seq_len + 1: i + 1]
            if len(seq) != seq_len:
                continue

            # FIX SEQNAN: Replace NaN instead of skipping
            if np.isnan(seq).any():
                seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

            c0 = float(close[i])
            if not np.isfinite(c0) or c0 <= 0:
                continue

            fut = close[i + 1: i + horizon + 1]
            if len(fut) != horizon:
                continue

            if np.isnan(fut).any():
                continue

            y_vals = (fut / c0 - 1.0) * 100.0

            X_list.append(seq)
            Y_list.append(y_vals.astype(np.float32))
            idx_list.append(df.index[i])

        if not X_list:
            if return_index:
                return empty_X, empty_Y, empty_idx
            return empty_X, empty_Y

        X = np.array(X_list, dtype=np.float32)
        Y = np.array(Y_list, dtype=np.float32)

        if return_index:
            return X, Y, pd.DatetimeIndex(idx_list)
        return X, Y

    # =========================================================================
    # REAL-TIME INFERENCE
    # =========================================================================

    def prepare_inference_sequence(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> np.ndarray:
        """
        Prepare a single sequence for inference using the last
        SEQUENCE_LENGTH rows.

        Returns:
            np.ndarray of shape (1, seq_len, n_features).
        """
        seq_len = int(CONFIG.SEQUENCE_LENGTH)

        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        if len(df) < seq_len:
            raise ValueError(
                f"Need at least {seq_len} rows, got {len(df)}"
            )

        df_seq = df.tail(seq_len)
        features = df_seq[feature_cols].values.astype(np.float32)

        features = np.nan_to_num(
            features, nan=0.0, posinf=0.0, neginf=0.0
        )

        if self._fitted:
            features = self.transform(features)
        else:
            log.warning("Scaler not fitted - using clipped raw features")
            features = np.clip(
                features, -FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE
            ).astype(np.float32)

        return features[np.newaxis, :, :]

    def _get_feature_lookback(self) -> int:
        """
        Safely retrieve ``CONFIG.data.feature_lookback`` with fallback.
        (Issue 7 fix)
        """
        data_config = getattr(CONFIG, "data", None)
        if data_config is not None:
            return int(getattr(data_config, "feature_lookback", 60))
        return 60

    def prepare_realtime_sequence(
        self,
        code: str,
        new_bar: dict[str, float],
        feature_cols: list[str],
        feature_engine: Any | None = None,
    ) -> np.ndarray | None:
        """
        Prepare sequence for real-time prediction with streaming data.
        Uses config-driven feature lookback instead of magic number.

        Thread-safe: uses atomic append_and_snapshot to avoid TOCTOU race.
        """
        seq_len = int(CONFIG.SEQUENCE_LENGTH)
        feature_lookback = self._get_feature_lookback()
        min_required = seq_len + feature_lookback

        with self._buffer_lock:
            if code not in self._realtime_buffers:
                self._realtime_buffers[code] = RealtimeBuffer(
                    max_size=min_required * 2
                )
            buffer = self._realtime_buffers[code]

        # Atomic append + readiness check + snapshot
        df = buffer.append_and_snapshot(new_bar, min_required)
        if df is None:
            return None

        if feature_engine is not None:
            try:
                engine_min_rows = getattr(feature_engine, "MIN_ROWS", 60)
                if len(df) < engine_min_rows:
                    log.debug(
                        f"Insufficient rows for feature engine: "
                        f"{len(df)} < {engine_min_rows}"
                    )
                    return None
                df = feature_engine.create_features(df)
            except Exception as e:
                log.warning(
                    f"Feature calculation failed for {code}: {e}"
                )
                return None

        missing = set(feature_cols) - set(df.columns)
        if missing:
            log.warning(f"Missing features for {code}: {missing}")
            return None

        try:
            return self.prepare_inference_sequence(df, feature_cols)
        except Exception as e:
            log.warning(
                f"Sequence preparation failed for {code}: {e}"
            )
            return None

    def prepare_batch_inference(
        self,
        dataframes: dict[str, pd.DataFrame],
        feature_cols: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """Prepare batch inference for multiple stocks."""
        X_list: list[np.ndarray] = []
        codes: list[str] = []

        for code, df in dataframes.items():
            try:
                seq = self.prepare_inference_sequence(df, feature_cols)
                X_list.append(seq[0])
                codes.append(code)
            except Exception as e:
                log.debug(f"Failed to prepare {code}: {e}")

        if not X_list:
            return np.array([]), []

        return np.array(X_list, dtype=np.float32), codes

    def get_realtime_buffer(
        self, code: str
    ) -> RealtimeBuffer | None:
        """Get the realtime buffer for a stock."""
        with self._buffer_lock:
            return self._realtime_buffers.get(code)

    def clear_realtime_buffer(self, code: str | None = None) -> None:
        """Clear realtime buffer(s)."""
        with self._buffer_lock:
            if code:
                if code in self._realtime_buffers:
                    self._realtime_buffers[code].clear()
            else:
                self._realtime_buffers.clear()

    def initialize_realtime_buffer(
        self, code: str, df: pd.DataFrame
    ) -> None:
        """
        Initialize realtime buffer with historical data.

        FIX THREAD: Uses atomic initialize_from_dataframe to prevent
        race conditions between clear() and append() calls.
        """
        with self._buffer_lock:
            if code not in self._realtime_buffers:
                self._realtime_buffers[code] = RealtimeBuffer()
            buffer = self._realtime_buffers[code]

        # Atomic initialization under buffer's internal lock
        count = buffer.initialize_from_dataframe(df)

        log.debug(
            f"Initialized buffer for {code} with {count} bars"
        )

    # =========================================================================
    # =========================================================================

    def split_temporal(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        fit_scaler_on_train: bool = True,
        horizon: int | None = None,
        feature_engine: Any | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Alias for split_temporal_single_stock."""
        return self.split_temporal_single_stock(
            df,
            feature_cols,
            fit_scaler_on_train,
            horizon,
            feature_engine=feature_engine,
        )

    def split_temporal_single_stock(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        fit_scaler_on_train: bool = True,
        horizon: int | None = None,
        feature_engine: Any | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split a single stock temporally with embargo gap between splits.

        CRITICAL FIX (Issue 1 + Issue 2):
        -  When ``feature_engine`` is provided, raw OHLCV data is split
           FIRST, then features are computed WITHIN each split so that
           rolling-window indicators cannot leak across boundaries.
        -  Extra ``feature_lookback`` rows are prepended to val/test
           raw data so rolling windows can warm up; those rows are then
           invalidated via NaN labels before sequence creation.
        -  Warmup length uses ``CONFIG.data.feature_lookback`` (the true
           max indicator lookback) instead of ``seq_len``.
        """
        seq_len = int(CONFIG.SEQUENCE_LENGTH)
        n_features = len(feature_cols)

        # FIX SHAPE: Define properly shaped empty arrays
        empty: tuple[np.ndarray, np.ndarray, np.ndarray] = (
            np.zeros((0, seq_len, n_features), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

        if not feature_cols:
            log.error("feature_cols cannot be empty")
            return {"train": empty, "val": empty, "test": empty}

        if not df.index.is_monotonic_increasing:
            log.warning(
                "DataFrame index is not sorted chronologically — sorting now"
            )
            df = df.sort_index()

        n = len(df)
        horizon = int(horizon or CONFIG.PREDICTION_HORIZON)
        embargo = int(CONFIG.EMBARGO_BARS)
        feature_lookback = self._get_feature_lookback()

        min_required = seq_len + horizon + embargo + 50
        if n < min_required:
            log.warning(
                f"Insufficient data for splitting: {n} rows, "
                f"need {min_required}"
            )
            return {"train": empty, "val": empty, "test": empty}

        # ---- validate feature_cols early when feature_engine is provided ----
        if feature_engine is not None:
            try:
                engine_cols = set(feature_engine.get_feature_columns())
                requested_cols = set(feature_cols)
                missing_from_engine = requested_cols - engine_cols
                if missing_from_engine:
                    log.warning(
                        f"feature_cols contains columns not produced by "
                        f"feature_engine: {missing_from_engine}"
                    )
            except Exception as e:
                log.warning(f"Could not validate feature_cols against engine: {e}")

        # ---- temporal boundaries ----
        # Ensure each split can produce at least one sequence:
        # len(split) >= seq_len + horizon.
        min_split_rows = seq_len + horizon
        min_train_rows = min_split_rows + 1

        requested_train_end = int(n * float(CONFIG.TRAIN_RATIO))
        requested_val_end = int(n * float(CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))

        effective_embargo = embargo
        max_train_end = n - 2 * (effective_embargo + min_split_rows)
        if max_train_end < min_train_rows:
            # If configured embargo is too strict for data length, shrink it.
            effective_embargo = max(
                0, (n - min_train_rows - 2 * min_split_rows) // 2
            )
            max_train_end = n - 2 * (effective_embargo + min_split_rows)

        if max_train_end < min_train_rows:
            log.warning(
                f"Insufficient data for temporal split: n={n}, "
                f"need >= {min_train_rows + 2 * min_split_rows}"
            )
            return {"train": empty, "val": empty, "test": empty}

        train_end = max(min(requested_train_end, max_train_end), min_train_rows)
        val_start = train_end + effective_embargo

        min_val_end = val_start + min_split_rows
        max_val_end = n - (effective_embargo + min_split_rows)
        if max_val_end < min_val_end:
            log.warning(
                f"Could not allocate val/test splits: "
                f"min_val_end={min_val_end}, max_val_end={max_val_end}"
            )
            return {"train": empty, "val": empty, "test": empty}

        val_end = min(max(requested_val_end, min_val_end), max_val_end)
        test_start = val_end + effective_embargo

        # ---- decide whether to recompute features per split ----
        if feature_engine is not None:
            # Split RAW data, include lookback prefix for rolling warmup
            train_raw = df.iloc[:train_end].copy()

            val_raw_begin = max(0, val_start - feature_lookback)
            val_raw = df.iloc[val_raw_begin:val_end].copy()

            test_raw_begin = max(0, test_start - feature_lookback)
            test_raw = df.iloc[test_raw_begin:].copy()

            min_rows = getattr(feature_engine, "MIN_ROWS", 60)

            try:
                if len(train_raw) < min_rows:
                    log.warning(
                        f"Train split too small for features: "
                        f"{len(train_raw)} < {min_rows}"
                    )
                    return {"train": empty, "val": empty, "test": empty}
                train_df = feature_engine.create_features(train_raw)
            except ValueError as e:
                log.warning(f"Train feature creation failed: {e}")
                return {"train": empty, "val": empty, "test": empty}

            try:
                if len(val_raw) >= min_rows:
                    val_df = feature_engine.create_features(val_raw)
                else:
                    val_df = pd.DataFrame()
            except ValueError as e:
                log.warning(f"Val feature creation failed: {e}")
                val_df = pd.DataFrame()

            try:
                if len(test_raw) >= min_rows:
                    test_df = feature_engine.create_features(test_raw)
                else:
                    test_df = pd.DataFrame()
            except ValueError as e:
                log.warning(f"Test feature creation failed: {e}")
                test_df = pd.DataFrame()

            warmup_val = val_start - val_raw_begin
            warmup_test = test_start - test_raw_begin
        else:
            # Features already on df — legacy path (warn about leakage)
            log.warning(
                "No feature_engine provided — features may leak "
                "across splits"
            )
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[val_start:val_end].copy()
            test_df = df.iloc[test_start:].copy()

            # Use feature_lookback as warmup (Issue 2)
            warmup_val = min(feature_lookback, len(val_df) // 4) if len(val_df) > 0 else 0
            warmup_test = min(feature_lookback, len(test_df) // 4) if len(test_df) > 0 else 0

        # ---- validate feature_cols exist in computed DataFrames ----
        for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if len(split_df) > 0:
                missing = set(feature_cols) - set(split_df.columns)
                if missing:
                    log.error(
                        f"Split '{name}' is missing feature columns: {missing}. "
                        f"Available: {sorted(split_df.columns.tolist())}"
                    )
                    if name == "train":
                        return {"train": empty, "val": empty, "test": empty}

        # ---- create labels WITHIN each split ----
        if len(train_df) > 0:
            train_df = self.create_labels(train_df, horizon=horizon)
        if len(val_df) > 0:
            val_df = self.create_labels(val_df, horizon=horizon)
        if len(test_df) > 0:
            test_df = self.create_labels(test_df, horizon=horizon)

        # ---- invalidate warmup rows in val/test ----
        # FIX WARMUP2: Clamp warmup to actual DataFrame length
        if warmup_val > 0 and len(val_df) > 0 and "label" in val_df.columns:
            clamp_val = min(warmup_val, len(val_df))
            if clamp_val > 0:
                val_df.iloc[
                    :clamp_val, val_df.columns.get_loc("label")
                ] = np.nan
        if warmup_test > 0 and len(test_df) > 0 and "label" in test_df.columns:
            clamp_test = min(warmup_test, len(test_df))
            if clamp_test > 0:
                test_df.iloc[
                    :clamp_test, test_df.columns.get_loc("label")
                ] = np.nan

        # ---- fit scaler on TRAIN only ----
        if fit_scaler_on_train and len(train_df) >= seq_len:
            if "label" in train_df.columns:
                avail_cols = [c for c in feature_cols if c in train_df.columns]
                if len(avail_cols) == len(feature_cols):
                    train_features = train_df[feature_cols].values
                    valid_mask = ~train_df["label"].isna()
                    if int(valid_mask.sum()) > 10:
                        self.fit_scaler(
                            train_features[valid_mask],
                            interval=self._interval,
                            horizon=horizon,
                        )

        # ---- prepare sequences for each split ----
        results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for name, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            if len(split_df) >= min_split_rows:
                try:
                    X, y, r = self.prepare_sequences(
                        split_df, feature_cols, fit_scaler=False
                    )
                    results[name] = (X, y, r)
                    if len(X) > 0:
                        log.info(
                            f"Split '{name}': {len(X)} samples, "
                            f"class dist: {self.get_class_distribution(y)}"
                        )
                    else:
                        log.debug(f"Split '{name}': 0 valid samples")
                except Exception as e:
                    log.warning(f"Sequence preparation failed for split '{name}': {e}")
                    results[name] = empty
            else:
                results[name] = empty
                log.debug(
                    f"Split '{name}': 0 samples (insufficient data: {len(split_df)} rows)"
                )

        for name in ("train", "val", "test"):
            if name not in results:
                results[name] = empty

        return results

    # =========================================================================
    # =========================================================================

    def prepare_single_sequence(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> np.ndarray:
        """Alias for prepare_inference_sequence."""
        return self.prepare_inference_sequence(df, feature_cols)

    def get_class_distribution(self, y: np.ndarray) -> dict[str, int]:
        """
        Get class distribution for logging.
        Dynamically handles any NUM_CLASSES value.

        FIX BINCOUNT: Guards against negative labels which would crash bincount.
        """
        num_classes = int(CONFIG.NUM_CLASSES)

        if len(y) == 0:
            dist: dict[str, int] = {}
            for i in range(num_classes):
                label_name = _DEFAULT_LABEL_NAMES.get(i, f"CLASS_{i}")
                dist[label_name] = 0
            dist["total"] = 0
            return dist

        y_int = y.astype(int)

        # FIX BINCOUNT: Filter out invalid labels before bincount
        valid_mask = (y_int >= 0) & (y_int < num_classes)
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum()
            log.warning(
                f"get_class_distribution: {invalid_count} labels outside "
                f"[0, {num_classes}) — excluding from distribution"
            )
            y_int = y_int[valid_mask]

        if len(y_int) == 0:
            dist = {}
            for i in range(num_classes):
                label_name = _DEFAULT_LABEL_NAMES.get(i, f"CLASS_{i}")
                dist[label_name] = 0
            dist["total"] = 0
            return dist

        counts = np.bincount(y_int, minlength=num_classes)

        dist = {}
        for i in range(num_classes):
            label_name = _DEFAULT_LABEL_NAMES.get(i, f"CLASS_{i}")
            dist[label_name] = int(counts[i]) if i < len(counts) else 0
        dist["total"] = int(len(y))  # Original length including invalid

        return dist

    def get_scaler_info(self) -> dict[str, Any]:
        """Get scaler metadata."""
        with self._lock:
            return {
                "fitted": self._fitted,
                "n_features": self._n_features,
                "fit_samples": self._fit_samples,
                "interval": self._interval,
                "horizon": self._horizon,
                "version": self._scaler_version,
            }

    def validate_features(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> bool:
        """Validate that DataFrame has all required features."""
        missing = set(feature_cols) - set(df.columns)
        if missing:
            log.warning(f"Missing features: {missing}")
            return False
        return True

# REAL-TIME PREDICTION HELPER

class RealtimePredictor:
    """
    High-level helper for real-time AI predictions.

    Combines FeatureEngine, DataProcessor (with scaler), and
    EnsembleModel for end-to-end live prediction.
    """

    def __init__(
        self,
        interval: str = "1m",
        horizon: int = 30,
        auto_load: bool = True,
    ):
        self.interval = str(interval).lower()
        self.horizon = int(horizon)

        self.processor = DataProcessor()
        self.feature_engine: Any | None = None
        self.ensemble: Any | None = None
        self.forecaster: Any | None = None

        self._feature_cols: list[str] = []
        self._loaded = False
        self._loading = False
        self._lock = threading.RLock()
        self._device: str = "cpu"

        if auto_load:
            self.load_models()

    def load_models(self) -> bool:
        """
        Load all required models for prediction.

        FIX RACE: Uses double-check locking to prevent redundant loads
        when multiple threads call load_models() simultaneously.
        """
        # Fast path: already loaded
        if self._loaded:
            return True

        with self._lock:
            # Double-check under lock
            if self._loaded:
                return True

            if self._loading:
                log.debug("load_models() already in progress, waiting...")
                return self._loaded

            self._loading = True

        try:
            return self._do_load_models()
        finally:
            with self._lock:
                self._loading = False

    def _do_load_models(self) -> bool:
        """Internal model loading implementation."""
        from data.features import FeatureEngine
        from models.ensemble import EnsembleModel

        try:
            self.feature_engine = FeatureEngine()
            self._feature_cols = (
                self.feature_engine.get_feature_columns()
            )

            scaler_path = (
                CONFIG.MODEL_DIR
                / f"scaler_{self.interval}_{self.horizon}.pkl"
            )
            if not self.processor.load_scaler(str(scaler_path)):
                log.warning(f"Failed to load scaler: {scaler_path}")
                return False

            if (
                self.processor.n_features is not None
                and self.processor.n_features != len(self._feature_cols)
            ):
                log.error(
                    f"Feature count mismatch: scaler expects "
                    f"{self.processor.n_features}, engine produces "
                    f"{len(self._feature_cols)}"
                )
                return False

            ensemble_path = (
                CONFIG.MODEL_DIR
                / f"ensemble_{self.interval}_{self.horizon}.pt"
            )
            if ensemble_path.exists():
                self.ensemble = EnsembleModel(
                    input_size=self.processor.n_features
                    or len(self._feature_cols)
                )
                if not self.ensemble.load(str(ensemble_path)):
                    log.warning(
                        f"Failed to load ensemble: {ensemble_path}"
                    )
                    self.ensemble = None
                else:
                    self._device = self.ensemble.device

            forecast_path = (
                CONFIG.MODEL_DIR
                / f"forecast_{self.interval}_{self.horizon}.pt"
            )
            if forecast_path.exists():
                self._load_forecaster(forecast_path)

            with self._lock:
                self._loaded = True

            log.info(
                f"Models loaded: interval={self.interval}, "
                f"horizon={self.horizon}, device={self._device}"
            )
            return True

        except Exception as e:
            log.error(f"Failed to load models: {e}")
            return False

    def _load_forecaster(self, path: Path) -> None:
        """
        Load TCN forecaster model.

        The forecaster outputs ``horizon`` regression values (not class
        probabilities), so ``num_classes`` in the TCN is set to the
        horizon value from the saved checkpoint.

        WARNING: Uses weights_only=False which can execute arbitrary code.
        Only load from trusted model files.
        """
        import torch

        from models.networks import TCNModel

        try:
            data = torch.load(
                path, map_location="cpu", weights_only=False
            )

            required_keys = {"input_size", "horizon", "arch", "state_dict"}
            if not required_keys.issubset(data.keys()):
                log.warning(
                    f"Forecaster checkpoint missing keys: "
                    f"{required_keys - set(data.keys())}"
                )
                return

            output_size = int(data["horizon"])

            self.forecaster = TCNModel(
                input_size=int(data["input_size"]),
                hidden_size=int(data["arch"]["hidden_size"]),
                num_classes=output_size,
                dropout=float(data["arch"]["dropout"]),
            )
            self.forecaster.load_state_dict(data["state_dict"])
            self.forecaster.eval()

            # FIX DEVICE: Move forecaster to same device as ensemble
            self.forecaster.to(self._device)

            log.info(
                f"Forecaster loaded: {path} "
                f"(output_size={output_size}, device={self._device})"
            )
        except Exception as e:
            log.warning(f"Failed to load forecaster: {e}")
            self.forecaster = None

    def predict(
        self,
        df: pd.DataFrame,
        include_forecast: bool = True,
    ) -> dict[str, Any] | None:
        """
        Make real-time prediction.

        FIX LOCK: Does NOT hold self._lock during inference, only during
        state checks. This prevents blocking other threads during the
        potentially slow forward pass.

        Args:
            df: DataFrame with OHLCV data (must have >= MIN_ROWS rows
                for feature computation + SEQUENCE_LENGTH for sequence).
            include_forecast: Whether to include multi-step forecast.

        Returns:
            Dict with signal, confidence, probabilities, etc. or None on failure.
        """
        # Check loaded state under lock (fast)
        with self._lock:
            is_loaded = self._loaded

        if not is_loaded:
            if not self.load_models():
                return None

        # FIX LOCK: Perform inference WITHOUT holding lock
        return self._do_predict(df, include_forecast)

    def _do_predict(
        self,
        df: pd.DataFrame,
        include_forecast: bool,
    ) -> dict[str, Any] | None:
        """
        Internal prediction logic.

        FIX LOCK: No longer requires self._lock to be held.
        The models are only modified during load_models() which is
        protected by its own locking. During prediction, models are
        read-only (eval mode).
        """
        try:
            min_rows = getattr(self.feature_engine, "MIN_ROWS", 60)
            if len(df) < min_rows:
                log.warning(
                    f"Insufficient data for prediction: "
                    f"{len(df)} rows < {min_rows} minimum"
                )
                return None

            df_features = self.feature_engine.create_features(df.copy())
            X = self.processor.prepare_inference_sequence(
                df_features, self._feature_cols
            )

            num_classes = int(CONFIG.NUM_CLASSES)
            result: dict[str, Any] = {
                "timestamp": datetime.now(),
                "signal": "HOLD",
                "confidence": 0.0,
                "probabilities": [1.0 / num_classes] * num_classes,
                "predicted_class": 1,
            }

            if self.ensemble is not None:
                pred = self.ensemble.predict(X)

                result["probabilities"] = pred.probabilities.tolist()
                result["predicted_class"] = pred.predicted_class
                result["confidence"] = pred.confidence
                result["entropy"] = pred.entropy
                result["agreement"] = pred.agreement

                min_conf = float(CONFIG.MIN_CONFIDENCE)
                strong_buy = float(CONFIG.STRONG_BUY_THRESHOLD)
                strong_sell = float(CONFIG.STRONG_SELL_THRESHOLD)

                if pred.predicted_class == 2 and pred.confidence >= min_conf:
                    if pred.confidence >= strong_buy:
                        result["signal"] = "STRONG_BUY"
                    else:
                        result["signal"] = "BUY"
                elif pred.predicted_class == 0 and pred.confidence >= min_conf:
                    if pred.confidence >= strong_sell:
                        result["signal"] = "STRONG_SELL"
                    else:
                        result["signal"] = "SELL"
                else:
                    result["signal"] = "HOLD"

            if include_forecast and self.forecaster is not None:
                import torch

                self.forecaster.eval()
                with torch.inference_mode():
                    X_tensor = torch.FloatTensor(X).to(self._device)
                    forecast, _ = self.forecaster(X_tensor)
                    result["forecast"] = (
                        forecast[0].cpu().numpy().tolist()
                    )

            return result

        except Exception as e:
            log.error(f"Prediction failed: {e}")
            return None

    def predict_batch(
        self,
        dataframes: dict[str, pd.DataFrame],
        include_forecast: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Make predictions for multiple stocks."""
        results: dict[str, dict[str, Any]] = {}

        for code, df in dataframes.items():
            pred = self.predict(
                df, include_forecast=include_forecast
            )
            if pred is not None:
                results[code] = pred

        return results

    def update_and_predict(
        self,
        code: str,
        new_bar: dict[str, float],
    ) -> dict[str, Any] | None:
        """Update buffer with new bar and make prediction."""
        with self._lock:
            is_loaded = self._loaded

        if not is_loaded:
            if not self.load_models():
                return None

        # FIX LOCK: No lock held during prediction
        return self._do_update_and_predict(code, new_bar)

    def _do_update_and_predict(
        self,
        code: str,
        new_bar: dict[str, float],
    ) -> dict[str, Any] | None:
        """Internal update and predict logic."""
        X = self.processor.prepare_realtime_sequence(
            code, new_bar, self._feature_cols, self.feature_engine
        )

        if X is None:
            return None

        result: dict[str, Any] = {
            "code": code,
            "timestamp": datetime.now(),
            "signal": "HOLD",
            "confidence": 0.0,
        }

        if self.ensemble is not None:
            pred = self.ensemble.predict(X)
            result["probabilities"] = pred.probabilities.tolist()
            result["predicted_class"] = pred.predicted_class
            result["confidence"] = pred.confidence

            min_conf = float(CONFIG.MIN_CONFIDENCE)

            if pred.predicted_class == 2 and pred.confidence >= min_conf:
                result["signal"] = "BUY"
            elif pred.predicted_class == 0 and pred.confidence >= min_conf:
                result["signal"] = "SELL"

        return result

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        with self._lock:
            return self._loaded

    @property
    def device(self) -> str:
        """Get the device models are running on."""
        with self._lock:
            return self._device
