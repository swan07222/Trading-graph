"""
Data Processor - Prepare data for training without data leakage

CRITICAL: This processor properly handles scaling to prevent data leakage:
- Scaler is fitted ONLY on training data
- Same scaler is applied to validation and test data
- Scaler is saved with the model for inference

Author: AI Trading System
Version: 2.0
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import RobustScaler
import pickle
from pathlib import Path

from config import CONFIG
from utils.logger import log


class DataProcessor:
    """
    Data processor with proper scaler handling to prevent data leakage
    
    Usage:
        processor = DataProcessor()
        
        # For training:
        X_train, y_train, r_train = processor.prepare_sequences(train_df, feature_cols, fit_scaler=True)
        X_val, y_val, r_val = processor.prepare_sequences(val_df, feature_cols, fit_scaler=False)
        processor.save_scaler()
        
        # For inference:
        processor.load_scaler()
        X = processor.prepare_sequences(df, feature_cols, fit_scaler=False)
    """
    
    def __init__(self):
        self.scaler: Optional[RobustScaler] = None
        self._fitted = False
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create classification labels based on future returns
        
        Labels:
            0 = DOWN (return <= -2%)
            1 = NEUTRAL (-2% < return < 2%)
            2 = UP (return >= 2%)
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            DataFrame with added 'label' and 'future_return' columns
        """
        df = df.copy()
        
        # Calculate future return
        future_price = df['close'].shift(-CONFIG.PREDICTION_HORIZON)
        future_return = (future_price / df['close'] - 1) * 100
        
        # Create labels
        df['label'] = 1  # NEUTRAL by default
        df.loc[future_return >= CONFIG.UP_THRESHOLD, 'label'] = 2  # UP
        df.loc[future_return <= CONFIG.DOWN_THRESHOLD, 'label'] = 0  # DOWN
        df['future_return'] = future_return
        
        # Remove rows without valid labels (last PREDICTION_HORIZON rows)
        df = df.iloc[:-CONFIG.PREDICTION_HORIZON]
        
        return df
    
    def fit_scaler(self, features: np.ndarray) -> 'DataProcessor':
        """
        Fit scaler on training data ONLY
        
        This should be called with training data features before
        calling prepare_sequences for any split.
        
        Args:
            features: Training features array (n_samples, n_features)
            
        Returns:
            self for chaining
        """
        self.scaler = RobustScaler()
        self.scaler.fit(features)
        
        # Also store simple statistics for fallback
        self._feature_means = np.mean(features, axis=0)
        self._feature_stds = np.std(features, axis=0) + 1e-8
        
        self._fitted = True
        log.info(f"Scaler fitted on {len(features)} samples with {features.shape[1]} features")
        
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler
        
        Args:
            features: Features array to transform
            
        Returns:
            Transformed features, clipped to [-5, 5]
        """
        if not self._fitted:
            raise RuntimeError(
                "Scaler not fitted! Call fit_scaler with training data first, "
                "or load a saved scaler with load_scaler()"
            )
        
        transformed = self.scaler.transform(features)
        
        # Clip extreme values to prevent instability
        transformed = np.clip(transformed, -5, 5)
        
        return transformed
    
    def prepare_sequences(self,
                          df: pd.DataFrame,
                          feature_cols: List[str],
                          fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequences for model training/inference
        
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names
            fit_scaler: If True, fit scaler on this data (ONLY for training data!)
            
        Returns:
            X: Sequences array (n_samples, sequence_length, n_features)
            y: Labels array (n_samples,)
            r: Future returns array (n_samples,) for trading simulation
        """
        # Extract raw features
        features = df[feature_cols].values.astype(np.float32)
        labels = df['label'].values.astype(np.int64)
        returns = df['future_return'].values.astype(np.float32)
        
        # Handle scaling
        if fit_scaler:
            self.fit_scaler(features)
        
        if self._fitted:
            features = self.transform(features)
        else:
            # Fallback: simple standardization (not recommended)
            log.warning("Using fallback standardization - scaler not fitted!")
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-8
            features = np.clip((features - mean) / std, -5, 5)
        
        # Create sequences
        seq_len = CONFIG.SEQUENCE_LENGTH
        X, y, r = [], [], []
        
        for i in range(seq_len, len(features)):
            X.append(features[i-seq_len:i])
            y.append(labels[i])
            r.append(returns[i])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        r = np.array(r, dtype=np.float32)
        
        log.debug(f"Created {len(X)} sequences of shape {X.shape[1:]}")
        
        return X, y, r
    
    def prepare_single_sequence(self,
                                df: pd.DataFrame,
                                feature_cols: List[str]) -> np.ndarray:
        """
        Prepare a single sequence for inference
        
        Args:
            df: DataFrame with at least SEQUENCE_LENGTH rows
            feature_cols: Feature column names
            
        Returns:
            Single sequence array (1, sequence_length, n_features)
        """
        if len(df) < CONFIG.SEQUENCE_LENGTH:
            raise ValueError(
                f"Need at least {CONFIG.SEQUENCE_LENGTH} rows, got {len(df)}"
            )
        
        # Get last SEQUENCE_LENGTH rows
        df_seq = df.tail(CONFIG.SEQUENCE_LENGTH)
        features = df_seq[feature_cols].values.astype(np.float32)
        
        # Transform
        if self._fitted:
            features = self.transform(features)
        else:
            log.warning("Scaler not fitted - prediction may be inaccurate!")
            features = np.clip(features, -5, 5)
        
        return features[np.newaxis, :, :]
    
    def save_scaler(self, path: str = None):
        """
        Save fitted scaler for inference
        
        Args:
            path: Path to save scaler (default: MODEL_DIR/scaler.pkl)
        """
        if not self._fitted:
            log.warning("Scaler not fitted, nothing to save")
            return
        
        path = path or str(CONFIG.MODEL_DIR / "scaler.pkl")
        
        scaler_data = {
            'scaler': self.scaler,
            'feature_means': self._feature_means,
            'feature_stds': self._feature_stds,
            'fitted': self._fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(scaler_data, f)
        
        log.info(f"Scaler saved to {path}")
    
    def load_scaler(self, path: str = None) -> bool:
        """
        Load saved scaler for inference
        
        Args:
            path: Path to load scaler from
            
        Returns:
            True if loaded successfully
        """
        path = path or str(CONFIG.MODEL_DIR / "scaler.pkl")
        
        if not Path(path).exists():
            log.warning(f"Scaler not found at {path}")
            return False
        
        try:
            with open(path, 'rb') as f:
                scaler_data = pickle.load(f)
            
            if isinstance(scaler_data, dict):
                self.scaler = scaler_data['scaler']
                self._feature_means = scaler_data.get('feature_means')
                self._feature_stds = scaler_data.get('feature_stds')
                self._fitted = scaler_data.get('fitted', True)
            else:
                # Old format - just the scaler
                self.scaler = scaler_data
                self._fitted = True
            
            log.info(f"Scaler loaded from {path}")
            return True
            
        except Exception as e:
            log.error(f"Failed to load scaler: {e}")
            return False
    
    def split_data_temporal(self,
                            X: np.ndarray,
                            y: np.ndarray,
                            r: np.ndarray) -> Tuple:
        """
        Split data maintaining temporal order (NO SHUFFLING)
        
        This is critical for time series - shuffling would cause data leakage!
        
        Args:
            X: Feature sequences
            y: Labels
            r: Returns
            
        Returns:
            Tuple of (X_train, y_train, r_train, X_val, y_val, r_val, X_test, y_test, r_test)
        """
        n = len(X)
        train_end = int(n * CONFIG.TRAIN_RATIO)
        val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
        
        X_train, y_train, r_train = X[:train_end], y[:train_end], r[:train_end]
        X_val, y_val, r_val = X[train_end:val_end], y[train_end:val_end], r[train_end:val_end]
        X_test, y_test, r_test = X[val_end:], y[val_end:], r[val_end:]
        
        log.info(f"Temporal split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return (X_train, y_train, r_train,
                X_val, y_val, r_val,
                X_test, y_test, r_test)
    
    def get_class_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Calculate class weights for imbalanced data
        
        Args:
            y: Labels array
            
        Returns:
            Array of class weights
        """
        counts = np.bincount(y.astype(int), minlength=CONFIG.NUM_CLASSES)
        weights = 1.0 / (counts + 1)
        weights = weights / weights.sum()
        
        log.debug(f"Class weights: DOWN={weights[0]:.3f}, NEUTRAL={weights[1]:.3f}, UP={weights[2]:.3f}")
        
        return weights
    
    def get_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """Get class distribution"""
        counts = np.bincount(y.astype(int), minlength=CONFIG.NUM_CLASSES)
        return {
            'DOWN': int(counts[0]),
            'NEUTRAL': int(counts[1]),
            'UP': int(counts[2])
        }