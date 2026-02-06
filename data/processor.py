# data/processor.py
"""
Data Processor - Prepare data for training WITHOUT data leakage

CRITICAL FIXES:
- Scaler fitted ONLY on training data
- Proper embargo gap between splits  
- Labels truncated at split boundaries
- Consistent sequence construction
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import RobustScaler
import pickle
from pathlib import Path
import threading

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class DataProcessor:
    """
    Thread-safe data processor with proper leakage prevention.
    
    CRITICAL RULES:
    1. fit_scaler() must be called ONLY with training data
    2. Labels near split boundaries are invalidated
    3. Embargo gap prevents any information flow
    """
    
    def __init__(self):
        self.scaler: Optional[RobustScaler] = None
        self._fitted = False
        self._lock = threading.Lock()
        self._n_features: Optional[int] = None
        self._fit_samples: int = 0
    
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
        horizon = horizon or CONFIG.PREDICTION_HORIZON
        up_thresh = up_thresh or CONFIG.UP_THRESHOLD
        down_thresh = down_thresh or CONFIG.DOWN_THRESHOLD
        
        df = df.copy()
        
        # Future return over horizon
        future_price = df['close'].shift(-horizon)
        future_return = (future_price / df['close'] - 1) * 100
        
        # Create labels
        df['label'] = 1  # Default: NEUTRAL
        df.loc[future_return >= up_thresh, 'label'] = 2  # UP
        df.loc[future_return <= down_thresh, 'label'] = 0  # DOWN
        df['future_return'] = future_return
        
        # Mark last horizon rows as invalid (no future data)
        df.iloc[-horizon:, df.columns.get_loc('label')] = np.nan
        df.iloc[-horizon:, df.columns.get_loc('future_return')] = np.nan
        
        return df
    
    def fit_scaler(self, features: np.ndarray) -> 'DataProcessor':
        """
        Fit scaler on training data ONLY.
        Thread-safe.
        """
        with self._lock:
            if features.ndim != 2:
                raise ValueError(f"Features must be 2D, got shape {features.shape}")
            
            # Remove any NaN rows before fitting
            valid_mask = ~np.isnan(features).any(axis=1)
            clean_features = features[valid_mask]
            
            if len(clean_features) < 10:
                raise ValueError(f"Insufficient valid samples for scaler: {len(clean_features)}")
            
            self.scaler = RobustScaler()
            self.scaler.fit(clean_features)
            
            self._fitted = True
            self._n_features = features.shape[1]
            self._fit_samples = len(clean_features)
            
            log.info(f"Scaler fitted on {self._fit_samples} samples, {self._n_features} features")
        
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using the fitted scaler.
        Clips extreme values to [-5, 5] for stability.
        """
        with self._lock:
            if not self._fitted:
                raise RuntimeError("Scaler not fitted! Call fit_scaler() first.")
            
            if features.shape[-1] != self._n_features:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self._n_features}, "
                    f"got {features.shape[-1]}"
                )
            
            original_shape = features.shape
            if features.ndim == 3:
                features_2d = features.reshape(-1, features.shape[-1])
            else:
                features_2d = features
            
            # Handle NaN values
            nan_mask = np.isnan(features_2d)
            features_2d = np.nan_to_num(features_2d, nan=0.0)
            
            transformed = self.scaler.transform(features_2d)
            transformed = np.clip(transformed, -5, 5)
            
            # Restore NaN positions
            transformed[nan_mask] = 0.0
            
            if len(original_shape) == 3:
                transformed = transformed.reshape(original_shape)
            
            return transformed.astype(np.float32)
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        fit_scaler: bool = False,
        return_index: bool = False
    ):
        seq_len = CONFIG.SEQUENCE_LENGTH

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
                idx.append(df.index[i])  # aligned end-of-sequence timestamp

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
    
    def prepare_inference_sequence(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> np.ndarray:
        """
        Prepare a single sequence for inference using the last SEQUENCE_LENGTH rows.
        Robust to NaN/Inf and missing columns.
        """
        seq_len = CONFIG.SEQUENCE_LENGTH

        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns for inference: {missing_cols}")

        if len(df) < seq_len:
            raise ValueError(f"Need at least {seq_len} rows, got {len(df)}")

        df_seq = df.tail(seq_len).copy()
        features = df_seq[feature_cols].values.astype(np.float32)

        # Replace inf/nan safely (keep stable behavior)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if self._fitted:
            features = self.transform(features)
        else:
            log.warning("Scaler not fitted - using raw features (clipped)")
            features = np.clip(features, -5, 5).astype(np.float32)

        return features[np.newaxis, :, :]
    
    # Alias for backward compatibility
    def prepare_single_sequence(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Alias for prepare_inference_sequence"""
        return self.prepare_inference_sequence(df, feature_cols)
    
    def split_temporal(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        fit_scaler_on_train: bool = True
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Alias for split_temporal_single_stock for backward compatibility
        """
        return self.split_temporal_single_stock(df, feature_cols, fit_scaler_on_train)
    
    def split_temporal_single_stock(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        fit_scaler_on_train: bool = True
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split a single stock's data temporally with proper embargo.

        FIX:
        - clamp split boundaries so we never slice negative ranges
        - keep leakage rules intact
        """
        n = len(df)
        horizon = CONFIG.PREDICTION_HORIZON
        embargo = CONFIG.EMBARGO_BARS

        train_end_raw = int(n * CONFIG.TRAIN_RATIO)
        val_end_raw = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))

        cut_train = max(0, train_end_raw - horizon - embargo)
        cut_val = max(cut_train, val_end_raw - horizon - embargo)

        train_df = df.iloc[:cut_train].copy()
        val_df = df.iloc[train_end_raw:cut_val].copy()
        test_df = df.iloc[val_end_raw:].copy()

        train_df = self.create_labels(train_df)
        val_df = self.create_labels(val_df)
        test_df = self.create_labels(test_df)

        if fit_scaler_on_train and len(train_df) >= CONFIG.SEQUENCE_LENGTH:
            train_features = train_df[feature_cols].values
            valid_mask = ~train_df["label"].isna()
            if int(valid_mask.sum()) > 10:
                self.fit_scaler(train_features[valid_mask])

        results = {}
        for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if len(split_df) >= CONFIG.SEQUENCE_LENGTH + 5:
                X, y, r = self.prepare_sequences(split_df, feature_cols, fit_scaler=False)
                results[name] = (X, y, r)
            else:
                results[name] = (np.array([]), np.array([]), np.array([]))

        return results
    
    def save_scaler(self, path: str = None):
        """Save scaler atomically"""
        if not self._fitted:
            log.warning("Scaler not fitted, nothing to save")
            return

        from pathlib import Path
        
        path = Path(path) if path else (CONFIG.MODEL_DIR / "scaler.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            data = {
                'scaler': self.scaler,
                'n_features': self._n_features,
                'fit_samples': self._fit_samples,
                'fitted': True
            }

        try:
            from utils.atomic_io import atomic_pickle_dump
            atomic_pickle_dump(path, data)
        except ImportError:
            # Fallback to regular pickle
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        log.info(f"Scaler saved to {path}")
    
    def load_scaler(self, path: str = None) -> bool:
        """Load saved scaler for inference"""
        path = path or str(CONFIG.MODEL_DIR / "scaler.pkl")
        
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
            
            log.info(f"Scaler loaded from {path}")
            return True
            
        except Exception as e:
            log.error(f"Failed to load scaler: {e}")
            return False
    
    def get_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """Get class distribution for logging"""
        if len(y) == 0:
            return {'DOWN': 0, 'NEUTRAL': 0, 'UP': 0, 'total': 0}
        counts = np.bincount(y.astype(int), minlength=CONFIG.NUM_CLASSES)
        return {
            'DOWN': int(counts[0]),
            'NEUTRAL': int(counts[1]),
            'UP': int(counts[2]),
            'total': int(len(y))
        }