# data/processor.py
"""
Data Processor - Prepare data for training and REAL-TIME inference

CRITICAL FEATURES:
- Scaler fitted ONLY on training data (no leakage)
- Proper embargo gap between splits
- Feature warmup invalidation in val/test splits
- Real-time sequence preparation for live predictions
- Multi-horizon forecasting support
- Thread-safe operations
- Streaming data support
- Config-driven feature lookback (no magic numbers)
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Union, Any
from sklearn.preprocessing import RobustScaler
import pickle
from pathlib import Path
import threading
from datetime import datetime
from collections import deque

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class RealtimeBuffer:
    """
    Thread-safe circular buffer for real-time data streaming.
    Maintains the last N bars for sequence construction.
    """
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or CONFIG.SEQUENCE_LENGTH * 2
        self._buffer: deque = deque(maxlen=self.max_size)
        self._lock = threading.RLock()
        self._last_update: Optional[datetime] = None
    
    def append(self, row: Dict[str, float]):
        """Add a new bar to the buffer"""
        with self._lock:
            self._buffer.append(row)
            self._last_update = datetime.now()
    
    def extend(self, rows: List[Dict[str, float]]):
        """Add multiple bars to the buffer"""
        with self._lock:
            for row in rows:
                self._buffer.append(row)
            self._last_update = datetime.now()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert buffer to DataFrame"""
        with self._lock:
            if not self._buffer:
                return pd.DataFrame()
            return pd.DataFrame(list(self._buffer))
    
    def get_latest(self, n: int = None) -> pd.DataFrame:
        """Get the latest N bars as DataFrame"""
        n = n or CONFIG.SEQUENCE_LENGTH
        with self._lock:
            if len(self._buffer) < n:
                return pd.DataFrame()
            data = list(self._buffer)[-n:]
            return pd.DataFrame(data)
    
    def is_ready(self, min_bars: int = None) -> bool:
        """Check if buffer has enough data for prediction"""
        min_bars = min_bars or CONFIG.SEQUENCE_LENGTH
        with self._lock:
            return len(self._buffer) >= min_bars
    
    def clear(self):
        """Clear the buffer"""
        with self._lock:
            self._buffer.clear()
            self._last_update = None
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
    
    @property
    def last_update(self) -> Optional[datetime]:
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
    
    REAL-TIME SUPPORT:
    - prepare_realtime_sequence(): For live bar-by-bar prediction
    - prepare_inference_sequence(): For batch inference
    - RealtimeBuffer: For streaming data management
    """
    
    def __init__(self):
        self.scaler: Optional[RobustScaler] = None
        self._fitted = False
        self._lock = threading.RLock()
        self._n_features: Optional[int] = None
        self._fit_samples: int = 0
        
        # Metadata for model compatibility
        self._interval: str = "1d"
        self._horizon: int = CONFIG.PREDICTION_HORIZON
        self._scaler_version: str = ""
        
        # Real-time buffers per stock
        self._realtime_buffers: Dict[str, RealtimeBuffer] = {}
        self._buffer_lock = threading.RLock()
    
    # =========================================================================
    # LABEL CREATION
    # =========================================================================
    
    def create_labels(self, 
                      df: pd.DataFrame,
                      horizon: int = None,
                      up_thresh: float = None,
                      down_thresh: float = None) -> pd.DataFrame:
        """
        Create classification labels based on future returns.
        
        Labels:
            0 = DOWN (return <= down_thresh)
            1 = NEUTRAL (down_thresh < return < up_thresh)
            2 = UP (return >= up_thresh)
        
        IMPORTANT: The last `horizon` rows will have NaN labels.
        """
        horizon = int(horizon or CONFIG.PREDICTION_HORIZON)
        up_thresh = float(up_thresh or CONFIG.UP_THRESHOLD)
        down_thresh = float(down_thresh or CONFIG.DOWN_THRESHOLD)
        
        df = df.copy()
        
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")
        
        close = pd.to_numeric(df['close'], errors='coerce')
        future_price = close.shift(-horizon)
        future_return = (future_price / close - 1) * 100
        
        df['label'] = 1  # Default: NEUTRAL
        df.loc[future_return >= up_thresh, 'label'] = 2  # UP
        df.loc[future_return <= down_thresh, 'label'] = 0  # DOWN
        df['future_return'] = future_return
        
        # Invalidate last `horizon` rows (no future data available)
        if horizon > 0 and len(df) > horizon:
            df.iloc[-horizon:, df.columns.get_loc('label')] = np.nan
            df.iloc[-horizon:, df.columns.get_loc('future_return')] = np.nan
        
        return df
    
    # =========================================================================
    # SCALER OPERATIONS
    # =========================================================================
    
    def fit_scaler(self, features: np.ndarray, 
                   interval: str = None, 
                   horizon: int = None) -> 'DataProcessor':
        """
        Fit scaler on training data ONLY.
        Thread-safe.
        """
        with self._lock:
            if features.ndim != 2:
                raise ValueError(f"Features must be 2D, got shape {features.shape}")
            
            valid_mask = ~(np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1))
            clean_features = features[valid_mask]
            
            if len(clean_features) < 10:
                raise ValueError(f"Insufficient valid samples for scaler: {len(clean_features)}")
            
            self.scaler = RobustScaler()
            self.scaler.fit(clean_features)
            
            self._fitted = True
            self._n_features = features.shape[1]
            self._fit_samples = len(clean_features)
            self._interval = str(interval or "1d")
            self._horizon = int(horizon or CONFIG.PREDICTION_HORIZON)
            self._scaler_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            log.info(f"Scaler fitted: {self._fit_samples} samples, "
                    f"{self._n_features} features, "
                    f"interval={self._interval}, horizon={self._horizon}")
        
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using the fitted scaler.
        Clips extreme values to [-5, 5] for numerical stability.
        """
        with self._lock:
            if not self._fitted:
                raise RuntimeError("Scaler not fitted! Call fit_scaler() first or load_scaler().")
            
            if features.shape[-1] != self._n_features:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self._n_features}, "
                    f"got {features.shape[-1]}"
                )
            
            original_shape = features.shape
            
            if features.ndim == 3:
                features_2d = features.reshape(-1, features.shape[-1])
            else:
                features_2d = features.copy()
            
            nan_mask = np.isnan(features_2d) | np.isinf(features_2d)
            features_2d = np.nan_to_num(features_2d, nan=0.0, posinf=0.0, neginf=0.0)
            
            transformed = self.scaler.transform(features_2d)
            transformed = np.clip(transformed, -5, 5)
            
            transformed[nan_mask] = 0.0
            
            if len(original_shape) == 3:
                transformed = transformed.reshape(original_shape)
            
            return transformed.astype(np.float32)
    
    def save_scaler(self, path: str = None, 
                    interval: str = None, 
                    horizon: int = None):
        """Save scaler atomically with metadata."""
        if not self._fitted:
            log.warning("Scaler not fitted, nothing to save")
            return
        
        interval = interval or self._interval
        horizon = horizon or self._horizon
        
        if path is None:
            path = CONFIG.MODEL_DIR / f"scaler_{interval}_{horizon}.pkl"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            data = {
                'scaler': self.scaler,
                'n_features': self._n_features,
                'fit_samples': self._fit_samples,
                'fitted': True,
                'interval': str(interval),
                'horizon': int(horizon),
                'version': self._scaler_version,
                'saved_at': datetime.now().isoformat(),
            }
        
        try:
            from utils.atomic_io import atomic_pickle_dump
            atomic_pickle_dump(path, data)
        except ImportError:
            tmp_path = path.with_suffix('.pkl.tmp')
            with open(tmp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(path)
        
        log.info(f"Scaler saved: {path} (interval={interval}, horizon={horizon})")
    
    def load_scaler(self, path: str = None, 
                    interval: str = None, 
                    horizon: int = None) -> bool:
        """Load saved scaler for inference."""
        if path is None:
            interval = interval or "1d"
            horizon = horizon or CONFIG.PREDICTION_HORIZON
            path = str(CONFIG.MODEL_DIR / f"scaler_{interval}_{horizon}.pkl")
        
        if not Path(path).exists():
            log.warning(f"Scaler not found: {path}")
            return False
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            with self._lock:
                self.scaler = data['scaler']
                self._n_features = data.get('n_features')
                self._fit_samples = data.get('fit_samples', 0)
                self._fitted = data.get('fitted', True)
                self._interval = data.get('interval', '1d')
                self._horizon = data.get('horizon', CONFIG.PREDICTION_HORIZON)
                self._scaler_version = data.get('version', '')
            
            log.info(f"Scaler loaded: {path} "
                    f"({self._n_features} features, "
                    f"interval={self._interval}, horizon={self._horizon})")
            return True
            
        except Exception as e:
            log.error(f"Failed to load scaler: {e}")
            return False
    
    @property
    def is_fitted(self) -> bool:
        """Check if scaler is fitted"""
        with self._lock:
            return self._fitted
    
    @property
    def n_features(self) -> Optional[int]:
        """Get number of features"""
        with self._lock:
            return self._n_features
    
    # =========================================================================
    # TRAINING SEQUENCE PREPARATION
    # =========================================================================
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        fit_scaler: bool = False,
        return_index: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], 
               Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]]:
        """
        Prepare sequences for training/validation.
        """
        seq_len = int(CONFIG.SEQUENCE_LENGTH)

        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        if 'label' not in df.columns:
            raise ValueError("DataFrame must have 'label' column. Call create_labels() first.")

        features = df[feature_cols].values.astype(np.float32)
        labels = df['label'].values
        returns = df['future_return'].values if 'future_return' in df.columns else np.zeros(len(df))

        if fit_scaler:
            valid_mask = ~np.isnan(labels)
            if valid_mask.sum() > 10:
                self.fit_scaler(features[valid_mask])

        if self._fitted:
            features = self.transform(features)

        X, y, r, idx = [], [], [], []

        for i in range(seq_len - 1, len(features)):
            if np.isnan(labels[i]):
                continue

            seq = features[i - seq_len + 1 : i + 1]
            
            if len(seq) == seq_len and not np.isnan(seq).any():
                X.append(seq)
                y.append(int(labels[i]))
                r.append(float(returns[i]) if not np.isnan(returns[i]) else 0.0)
                idx.append(df.index[i])

        if not X:
            empty = (np.array([]), np.array([]), np.array([]))
            if return_index:
                return (*empty, pd.DatetimeIndex([]))
            return empty

        out = (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int64),
            np.array(r, dtype=np.float32),
        )
        
        if return_index:
            return (*out, pd.DatetimeIndex(idx))
        return out
    
    def prepare_forecast_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        horizon: int,
        fit_scaler: bool = False,
        return_index: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], 
               Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
        """
        Prepare multi-step forecasting dataset.
        """
        seq_len = int(CONFIG.SEQUENCE_LENGTH)
        horizon = int(horizon)

        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        if "close" not in df.columns:
            raise ValueError("DataFrame must have 'close' column for forecasting")

        features = df[feature_cols].values.astype(np.float32)
        close = pd.to_numeric(df["close"], errors="coerce").values.astype(np.float32)

        if fit_scaler:
            valid_mask = ~np.isnan(features).any(axis=1)
            if valid_mask.sum() > 10:
                self.fit_scaler(features[valid_mask])

        if self._fitted:
            features = self.transform(features)

        X, Y, idx = [], [], []

        n = len(df)
        for i in range(seq_len - 1, n):
            if i + horizon >= n:
                break

            seq = features[i - seq_len + 1: i + 1]
            if len(seq) != seq_len or np.isnan(seq).any():
                continue

            c0 = float(close[i])
            if not np.isfinite(c0) or c0 <= 0:
                continue

            fut = close[i + 1: i + horizon + 1]
            if len(fut) != horizon or np.isnan(fut).any():
                continue

            y = (fut / c0 - 1.0) * 100.0

            X.append(seq)
            Y.append(y.astype(np.float32))
            idx.append(df.index[i])

        if not X:
            if return_index:
                return np.array([]), np.array([]), pd.DatetimeIndex([])
            return np.array([]), np.array([])

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)

        if return_index:
            return X, Y, pd.DatetimeIndex(idx)
        return X, Y
    
    # =========================================================================
    # REAL-TIME INFERENCE
    # =========================================================================
    
    def prepare_inference_sequence(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> np.ndarray:
        """
        Prepare a single sequence for inference using the last SEQUENCE_LENGTH rows.
        """
        seq_len = int(CONFIG.SEQUENCE_LENGTH)

        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        if len(df) < seq_len:
            raise ValueError(f"Need at least {seq_len} rows, got {len(df)}")

        df_seq = df.tail(seq_len).copy()
        features = df_seq[feature_cols].values.astype(np.float32)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if self._fitted:
            features = self.transform(features)
        else:
            log.warning("Scaler not fitted - using clipped raw features")
            features = np.clip(features, -5, 5).astype(np.float32)

        return features[np.newaxis, :, :]
    
    def prepare_realtime_sequence(
        self,
        code: str,
        new_bar: Dict[str, float],
        feature_cols: List[str],
        feature_engine: Any = None
    ) -> Optional[np.ndarray]:
        """
        Prepare sequence for real-time prediction with streaming data.
        Uses config-driven feature lookback instead of magic number.
        """
        seq_len = int(CONFIG.SEQUENCE_LENGTH)
        
        # Get feature lookback from config
        feature_lookback = getattr(CONFIG, 'data', None)
        feature_lookback = feature_lookback.feature_lookback if feature_lookback else 60
        min_required = seq_len + feature_lookback
        
        # Get or create buffer for this stock
        with self._buffer_lock:
            if code not in self._realtime_buffers:
                self._realtime_buffers[code] = RealtimeBuffer(max_size=seq_len * 3)
            buffer = self._realtime_buffers[code]
        
        buffer.append(new_bar)
        
        if not buffer.is_ready(min_required):
            return None
        
        df = buffer.to_dataframe()
        
        if feature_engine is not None:
            try:
                df = feature_engine.create_features(df)
            except Exception as e:
                log.warning(f"Feature calculation failed for {code}: {e}")
                return None
        
        missing = set(feature_cols) - set(df.columns)
        if missing:
            log.warning(f"Missing features for {code}: {missing}")
            return None
        
        try:
            return self.prepare_inference_sequence(df, feature_cols)
        except Exception as e:
            log.warning(f"Sequence preparation failed for {code}: {e}")
            return None
    
    def prepare_batch_inference(
        self,
        dataframes: Dict[str, pd.DataFrame],
        feature_cols: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare batch inference for multiple stocks."""
        X_list = []
        codes = []
        
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
    
    def get_realtime_buffer(self, code: str) -> Optional[RealtimeBuffer]:
        """Get the realtime buffer for a stock"""
        with self._buffer_lock:
            return self._realtime_buffers.get(code)
    
    def clear_realtime_buffer(self, code: str = None):
        """Clear realtime buffer(s)"""
        with self._buffer_lock:
            if code:
                if code in self._realtime_buffers:
                    self._realtime_buffers[code].clear()
            else:
                self._realtime_buffers.clear()
    
    def initialize_realtime_buffer(
        self,
        code: str,
        df: pd.DataFrame
    ):
        """Initialize realtime buffer with historical data."""
        with self._buffer_lock:
            if code not in self._realtime_buffers:
                self._realtime_buffers[code] = RealtimeBuffer()
            buffer = self._realtime_buffers[code]
        
        buffer.clear()
        
        for _, row in df.iterrows():
            bar = row.to_dict()
            buffer.append(bar)
        
        log.debug(f"Initialized buffer for {code} with {len(buffer)} bars")
    
    # =========================================================================
    # TEMPORAL SPLITTING
    # =========================================================================
    
    def split_temporal(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        fit_scaler_on_train: bool = True,
        horizon: int = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Alias for split_temporal_single_stock."""
        return self.split_temporal_single_stock(
            df, feature_cols, fit_scaler_on_train, horizon
        )
    
    def split_temporal_single_stock(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        fit_scaler_on_train: bool = True,
        horizon: int = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split a single stock temporally with embargo gap between splits.
        Includes feature warmup invalidation in val/test.
        """
        n = len(df)
        horizon = int(horizon or CONFIG.PREDICTION_HORIZON)
        embargo = int(CONFIG.EMBARGO_BARS)
        seq_len = int(CONFIG.SEQUENCE_LENGTH)

        min_required = seq_len + horizon + embargo + 50
        if n < min_required:
            log.warning(f"Insufficient data for splitting: {n} rows, need {min_required}")
            empty = (np.array([]), np.array([]), np.array([]))
            return {"train": empty, "val": empty, "test": empty}

        # Calculate temporal boundaries with embargo GAP
        train_end = int(n * float(CONFIG.TRAIN_RATIO))
        val_end = int(n * float(CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))

        val_start = min(n, train_end + embargo)
        test_start = min(n, val_end + embargo)

        # Split data
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:val_end].copy()
        test_df = df.iloc[test_start:].copy()

        # Create labels WITHIN each split (prevents leakage)
        train_df = self.create_labels(train_df, horizon=horizon)
        val_df = self.create_labels(val_df, horizon=horizon)
        test_df = self.create_labels(test_df, horizon=horizon)

        # Invalidate first seq_len labels in val/test where features
        # may depend on data from prior split (feature lookback contamination)
        feature_warmup = min(seq_len, len(val_df) // 4) if len(val_df) > seq_len else 0
        if feature_warmup > 0 and 'label' in val_df.columns:
            val_df.iloc[:feature_warmup, val_df.columns.get_loc('label')] = np.nan
        feature_warmup_test = min(seq_len, len(test_df) // 4) if len(test_df) > seq_len else 0
        if feature_warmup_test > 0 and 'label' in test_df.columns:
            test_df.iloc[:feature_warmup_test, test_df.columns.get_loc('label')] = np.nan

        # Fit scaler on TRAIN only
        if fit_scaler_on_train and len(train_df) >= seq_len:
            train_features = train_df[feature_cols].values
            valid_mask = ~train_df["label"].isna()
            if int(valid_mask.sum()) > 10:
                self.fit_scaler(train_features[valid_mask], horizon=horizon)

        # Prepare sequences for each split
        results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        
        for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if len(split_df) >= seq_len + 5:
                X, y, r = self.prepare_sequences(split_df, feature_cols, fit_scaler=False)
                results[name] = (X, y, r)
                log.debug(f"Split '{name}': {len(X)} samples")
            else:
                results[name] = (np.array([]), np.array([]), np.array([]))
                log.debug(f"Split '{name}': 0 samples (insufficient data)")

        return results
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def prepare_single_sequence(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Alias for prepare_inference_sequence"""
        return self.prepare_inference_sequence(df, feature_cols)
    
    def get_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """Get class distribution for logging"""
        if len(y) == 0:
            return {'DOWN': 0, 'NEUTRAL': 0, 'UP': 0, 'total': 0}
        
        y_int = y.astype(int)
        counts = np.bincount(y_int, minlength=CONFIG.NUM_CLASSES)
        
        return {
            'DOWN': int(counts[0]),
            'NEUTRAL': int(counts[1]),
            'UP': int(counts[2]),
            'total': int(len(y))
        }
    
    def get_scaler_info(self) -> Dict[str, Any]:
        """Get scaler metadata"""
        with self._lock:
            return {
                'fitted': self._fitted,
                'n_features': self._n_features,
                'fit_samples': self._fit_samples,
                'interval': self._interval,
                'horizon': self._horizon,
                'version': self._scaler_version,
            }
    
    def validate_features(self, df: pd.DataFrame, feature_cols: List[str]) -> bool:
        """Validate that DataFrame has all required features"""
        missing = set(feature_cols) - set(df.columns)
        if missing:
            log.warning(f"Missing features: {missing}")
            return False
        return True


# =============================================================================
# REAL-TIME PREDICTION HELPER
# =============================================================================

class RealtimePredictor:
    """
    High-level helper for real-time AI predictions.
    """
    
    def __init__(
        self,
        interval: str = "1m",
        horizon: int = 30,
        auto_load: bool = True
    ):
        self.interval = str(interval).lower()
        self.horizon = int(horizon)
        
        self.processor = DataProcessor()
        self.feature_engine = None
        self.ensemble = None
        self.forecaster = None
        
        self._feature_cols: List[str] = []
        self._loaded = False
        self._lock = threading.RLock()
        
        if auto_load:
            self.load_models()
    
    def load_models(self) -> bool:
        """Load all required models for prediction"""
        from data.features import FeatureEngine
        from models.ensemble import EnsembleModel
        
        with self._lock:
            try:
                self.feature_engine = FeatureEngine()
                self._feature_cols = self.feature_engine.get_feature_columns()
                
                scaler_path = CONFIG.MODEL_DIR / f"scaler_{self.interval}_{self.horizon}.pkl"
                if not self.processor.load_scaler(str(scaler_path)):
                    log.warning(f"Failed to load scaler: {scaler_path}")
                    return False
                
                ensemble_path = CONFIG.MODEL_DIR / f"ensemble_{self.interval}_{self.horizon}.pt"
                if ensemble_path.exists():
                    self.ensemble = EnsembleModel(
                        input_size=self.processor.n_features or len(self._feature_cols)
                    )
                    if not self.ensemble.load(str(ensemble_path)):
                        log.warning(f"Failed to load ensemble: {ensemble_path}")
                        self.ensemble = None
                
                forecast_path = CONFIG.MODEL_DIR / f"forecast_{self.interval}_{self.horizon}.pt"
                if forecast_path.exists():
                    self._load_forecaster(forecast_path)
                
                self._loaded = True
                log.info(f"Models loaded: interval={self.interval}, horizon={self.horizon}")
                return True
                
            except Exception as e:
                log.error(f"Failed to load models: {e}")
                return False
    
    def _load_forecaster(self, path: Path):
        """Load TCN forecaster model"""
        import torch
        from models.networks import TCNModel
        
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            
            self.forecaster = TCNModel(
                input_size=data['input_size'],
                hidden_size=data['arch']['hidden_size'],
                num_classes=data['horizon'],
                dropout=data['arch']['dropout']
            )
            self.forecaster.load_state_dict(data['state_dict'])
            self.forecaster.eval()
            
            log.info(f"Forecaster loaded: {path}")
        except Exception as e:
            log.warning(f"Failed to load forecaster: {e}")
            self.forecaster = None
    
    def predict(
        self,
        df: pd.DataFrame,
        include_forecast: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Make real-time prediction."""
        if not self._loaded:
            if not self.load_models():
                return None
        
        with self._lock:
            try:
                df = self.feature_engine.create_features(df.copy())
                X = self.processor.prepare_inference_sequence(df, self._feature_cols)
                
                result = {
                    'timestamp': datetime.now(),
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'probabilities': [0.33, 0.34, 0.33],
                    'predicted_class': 1,
                }
                
                if self.ensemble:
                    pred = self.ensemble.predict(X)
                    
                    result['probabilities'] = pred.probabilities.tolist()
                    result['predicted_class'] = pred.predicted_class
                    result['confidence'] = pred.confidence
                    result['entropy'] = pred.entropy
                    result['agreement'] = pred.agreement
                    
                    if pred.predicted_class == 2 and pred.confidence >= CONFIG.MIN_CONFIDENCE:
                        if pred.confidence >= CONFIG.STRONG_BUY_THRESHOLD:
                            result['signal'] = 'STRONG_BUY'
                        else:
                            result['signal'] = 'BUY'
                    elif pred.predicted_class == 0 and pred.confidence >= CONFIG.MIN_CONFIDENCE:
                        if pred.confidence >= CONFIG.STRONG_SELL_THRESHOLD:
                            result['signal'] = 'STRONG_SELL'
                        else:
                            result['signal'] = 'SELL'
                    else:
                        result['signal'] = 'HOLD'
                
                if include_forecast and self.forecaster:
                    import torch
                    
                    self.forecaster.eval()
                    with torch.inference_mode():
                        X_tensor = torch.FloatTensor(X)
                        forecast, _ = self.forecaster(X_tensor)
                        result['forecast'] = forecast[0].cpu().numpy().tolist()
                
                return result
                
            except Exception as e:
                log.error(f"Prediction failed: {e}")
                return None
    
    def predict_batch(
        self,
        dataframes: Dict[str, pd.DataFrame],
        include_forecast: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Make predictions for multiple stocks."""
        results = {}
        
        for code, df in dataframes.items():
            pred = self.predict(df, include_forecast=include_forecast)
            if pred:
                results[code] = pred
        
        return results
    
    def update_and_predict(
        self,
        code: str,
        new_bar: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Update buffer with new bar and make prediction."""
        X = self.processor.prepare_realtime_sequence(
            code, new_bar, self._feature_cols, self.feature_engine
        )
        
        if X is None:
            return None
        
        with self._lock:
            result = {
                'code': code,
                'timestamp': datetime.now(),
                'signal': 'HOLD',
                'confidence': 0.0,
            }
            
            if self.ensemble:
                pred = self.ensemble.predict(X)
                result['probabilities'] = pred.probabilities.tolist()
                result['predicted_class'] = pred.predicted_class
                result['confidence'] = pred.confidence
                
                if pred.predicted_class == 2 and pred.confidence >= CONFIG.MIN_CONFIDENCE:
                    result['signal'] = 'BUY'
                elif pred.predicted_class == 0 and pred.confidence >= CONFIG.MIN_CONFIDENCE:
                    result['signal'] = 'SELL'
            
            return result