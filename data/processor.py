"""
Data Processor - Prepare data for training WITHOUT data leakage

FIXED Issues:
- Scaler fitted ONLY on training data
- Proper embargo gap between splits
- Consistent sequence construction for train and inference
- No look-ahead bias

Author: AI Trading System v3.0
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import RobustScaler
import pickle
from pathlib import Path
import threading

from config import CONFIG
from utils.logger import log


class DataProcessor:
    """
    Thread-safe data processor with proper scaler handling.
    
    CRITICAL RULES:
    1. fit_scaler() must be called ONLY with training data
    2. transform() uses the fitted scaler for all splits
    3. Embargo gap prevents label leakage at split boundaries
    4. Sequence construction is identical for train and inference
    """
    
    def __init__(self):
        self.scaler: Optional[RobustScaler] = None
        self._fitted = False
        self._lock = threading.Lock()
        
        # Store statistics for validation
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
        
        IMPORTANT: The last `horizon` rows will have NaN labels
        and should be excluded from training.
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
        df.loc[df.index[-horizon:], 'label'] = np.nan
        df.loc[df.index[-horizon:], 'future_return'] = np.nan
        
        return df
    
    def fit_scaler(self, features: np.ndarray) -> 'DataProcessor':
        """
        Fit scaler on training data ONLY.
        
        This must be called with training features BEFORE any transform() calls.
        Thread-safe.
        """
        with self._lock:
            if features.ndim != 2:
                raise ValueError(f"Features must be 2D, got shape {features.shape}")
            
            self.scaler = RobustScaler()
            self.scaler.fit(features)
            
            self._fitted = True
            self._n_features = features.shape[1]
            self._fit_samples = features.shape[0]
            
            log.info(f"Scaler fitted on {self._fit_samples} samples, {self._n_features} features")
        
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using the fitted scaler.
        
        Clips extreme values to [-5, 5] for numerical stability.
        Thread-safe.
        """
        with self._lock:
            if not self._fitted:
                raise RuntimeError(
                    "Scaler not fitted! Call fit_scaler() with training data first."
                )
            
            if features.shape[-1] != self._n_features:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self._n_features}, "
                    f"got {features.shape[-1]}"
                )
            
            # Handle both 2D and 3D inputs
            original_shape = features.shape
            if features.ndim == 3:
                # Reshape (batch, seq, features) -> (batch*seq, features)
                features_2d = features.reshape(-1, features.shape[-1])
            else:
                features_2d = features
            
            transformed = self.scaler.transform(features_2d)
            transformed = np.clip(transformed, -5, 5)
            
            if len(original_shape) == 3:
                transformed = transformed.reshape(original_shape)
            
            return transformed.astype(np.float32)
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        fit_scaler: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequences for training or inference.
        
        SEQUENCE CONSTRUCTION (consistent for train and inference):
        - For each valid label at index i, the input sequence is:
          features[i - seq_len + 1 : i + 1]  (includes row i)
        - This means the sequence ENDS at the row where we make the prediction
        
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names
            fit_scaler: If True, fit scaler on this data (ONLY for training!)
            
        Returns:
            X: (n_samples, seq_len, n_features)
            y: (n_samples,) labels
            r: (n_samples,) future returns for backtesting
        """
        seq_len = CONFIG.SEQUENCE_LENGTH
        
        # Validate
        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        if 'label' not in df.columns:
            raise ValueError("DataFrame must have 'label' column. Call create_labels() first.")
        
        # Extract arrays
        features = df[feature_cols].values.astype(np.float32)
        labels = df['label'].values
        returns = df['future_return'].values if 'future_return' in df.columns else np.zeros(len(df))
        
        # Fit scaler if requested (training data only!)
        if fit_scaler:
            # Only fit on rows with valid labels
            valid_mask = ~np.isnan(labels)
            self.fit_scaler(features[valid_mask])
        
        # Transform features
        if self._fitted:
            features = self.transform(features)
        else:
            log.warning("Scaler not fitted - using raw features (not recommended)")
        
        # Create sequences
        X, y, r = [], [], []
        
        for i in range(seq_len - 1, len(features)):
            # Skip if label is invalid (NaN)
            if np.isnan(labels[i]):
                continue
            
            # Sequence: [i - seq_len + 1, i] inclusive
            seq = features[i - seq_len + 1 : i + 1]
            
            if len(seq) == seq_len:
                X.append(seq)
                y.append(int(labels[i]))
                r.append(float(returns[i]) if not np.isnan(returns[i]) else 0.0)
        
        if not X:
            return np.array([]), np.array([]), np.array([])
        
        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int64),
            np.array(r, dtype=np.float32)
        )
    
    def prepare_inference_sequence(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> np.ndarray:
        """
        Prepare a single sequence for inference.
        
        Uses the last SEQUENCE_LENGTH rows of the DataFrame.
        Consistent with training sequence construction.
        
        Returns:
            X: (1, seq_len, n_features)
        """
        seq_len = CONFIG.SEQUENCE_LENGTH
        
        if len(df) < seq_len:
            raise ValueError(
                f"Need at least {seq_len} rows for inference, got {len(df)}"
            )
        
        # Take last seq_len rows
        df_seq = df.tail(seq_len)
        features = df_seq[feature_cols].values.astype(np.float32)
        
        # Transform
        if self._fitted:
            features = self.transform(features)
        else:
            log.warning("Scaler not fitted - inference may be inaccurate")
            features = np.clip(features, -5, 5)
        
        return features[np.newaxis, :, :]
    
    def split_temporal(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split data temporally with proper embargo.
        
        CRITICAL: 
        - No shuffling (time series!)
        - Embargo gap between splits prevents label leakage
        - Scaler fitted only on training portion
        
        Returns:
            Dict with 'train', 'val', 'test' keys, each containing (X, y, r)
        """
        n = len(df)
        embargo = CONFIG.EMBARGO_BARS
        
        # Calculate split points
        train_end = int(n * CONFIG.TRAIN_RATIO)
        val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
        
        # Apply embargo: skip EMBARGO_BARS after train and val boundaries
        train_df = df.iloc[:train_end - embargo]
        val_df = df.iloc[train_end:val_end - embargo]
        test_df = df.iloc[val_end:]
        
        log.info(f"Temporal split with embargo={embargo}:")
        log.info(f"  Train: rows 0-{train_end - embargo - 1} ({len(train_df)} rows)")
        log.info(f"  Val: rows {train_end}-{val_end - embargo - 1} ({len(val_df)} rows)")
        log.info(f"  Test: rows {val_end}-{n-1} ({len(test_df)} rows)")
        
        # Fit scaler on training data only
        train_features = train_df[feature_cols].values
        valid_mask = ~train_df['label'].isna()
        self.fit_scaler(train_features[valid_mask])
        
        # Prepare sequences for each split
        results = {}
        
        for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            if len(split_df) >= CONFIG.SEQUENCE_LENGTH:
                X, y, r = self.prepare_sequences(split_df, feature_cols, fit_scaler=False)
                results[name] = (X, y, r)
                log.info(f"  {name}: {len(X)} sequences")
            else:
                results[name] = (np.array([]), np.array([]), np.array([]))
                log.warning(f"  {name}: insufficient data for sequences")
        
        return results
    
    def save_scaler(self, path: str = None):
        """Save fitted scaler for inference"""
        if not self._fitted:
            log.warning("Scaler not fitted, nothing to save")
            return
        
        path = path or str(CONFIG.MODEL_DIR / "scaler.pkl")
        
        with self._lock:
            data = {
                'scaler': self.scaler,
                'n_features': self._n_features,
                'fit_samples': self._fit_samples,
                'fitted': True
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        
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
        counts = np.bincount(y.astype(int), minlength=CONFIG.NUM_CLASSES)
        return {
            'DOWN': int(counts[0]),
            'NEUTRAL': int(counts[1]),
            'UP': int(counts[2]),
            'total': int(len(y))
        }
    
    def get_class_weights(self, y: np.ndarray) -> np.ndarray:
        """Calculate balanced class weights"""
        counts = np.bincount(y.astype(int), minlength=CONFIG.NUM_CLASSES)
        weights = 1.0 / (counts + 1)
        return weights / weights.sum()