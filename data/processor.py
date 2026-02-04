"""
Data Processor - Prepare data for training without data leakage

Critical Design Decisions:
1. Scaler is ONLY fitted on training data
2. Same scaler is applied to validation/test data
3. Scaler is saved with model for consistent inference
4. NO shuffling of time series data

Author: AI Trading System
Version: 2.0
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import RobustScaler
import pickle
from pathlib import Path

from config import CONFIG
from utils.logger import log


class DataProcessor:
    """
    Data processor that prevents data leakage.
    
    IMPORTANT: The scaler must be:
    1. Fitted ONLY on training data
    2. Applied (transform only) to validation/test data
    3. Saved and loaded for inference
    
    Usage:
        processor = DataProcessor()
        
        # Fit scaler on training features
        processor.fit_scaler(train_features)
        
        # Transform all splits
        X_train = processor.transform(train_features)
        X_val = processor.transform(val_features)  # Note: transform, not fit_transform!
        X_test = processor.transform(test_features)
        
        # Save for inference
        processor.save_scaler()
    """
    
    def __init__(self):
        self.scaler: Optional[RobustScaler] = None
        self._fitted = False
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create classification labels based on future returns.
        
        Labels:
            0 = DOWN (return <= -threshold)
            1 = NEUTRAL (between thresholds)
            2 = UP (return >= +threshold)
            
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            DataFrame with added 'label' and 'future_return' columns
        """
        df = df.copy()
        
        # Calculate future return over prediction horizon
        future_price = df['close'].shift(-CONFIG.PREDICTION_HORIZON)
        future_return = (future_price / df['close'] - 1) * 100
        
        # Create labels
        df['label'] = 1  # Default: NEUTRAL
        df.loc[future_return >= CONFIG.UP_THRESHOLD, 'label'] = 2  # UP
        df.loc[future_return <= CONFIG.DOWN_THRESHOLD, 'label'] = 0  # DOWN
        df['future_return'] = future_return
        
        # Remove rows without valid labels (last PREDICTION_HORIZON rows)
        df = df.iloc[:-CONFIG.PREDICTION_HORIZON]
        
        return df
    
    def fit_scaler(self, features: np.ndarray) -> 'DataProcessor':
        """
        Fit the scaler on training data ONLY.
        
        This method should ONLY be called with training features,
        never with validation or test features (that would be data leakage).
        
        Args:
            features: Training features array of shape (n_samples, n_features)
            
        Returns:
            self for method chaining
        """
        self.scaler = RobustScaler()
        self.scaler.fit(features)
        
        # Also store simple statistics for fallback
        self._feature_means = np.mean(features, axis=0)
        self._feature_stds = np.std(features, axis=0) + 1e-8
        
        self._fitted = True
        log.info(f"Scaler fitted on {len(features)} training samples with {features.shape[1]} features")
        
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using the fitted scaler.
        
        Args:
            features: Features array to transform
            
        Returns:
            Transformed and clipped features
            
        Raises:
            RuntimeError: If scaler hasn't been fitted
        """
        if not self._fitted:
            raise RuntimeError(
                "Scaler not fitted! Call fit_scaler() with training data first, "
                "or load a saved scaler with load_scaler()."
            )
        
        transformed = self.scaler.transform(features)
        
        # Clip extreme values to prevent numerical issues
        transformed = np.clip(transformed, -5, 5)
        
        return transformed
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit scaler and transform in one step.
        
        WARNING: Only use this for training data!
        For validation/test data, use transform() only.
        """
        self.fit_scaler(features)
        return self.transform(features)
    
    def prepare_sequences(self, 
                          df: pd.DataFrame,
                          feature_cols: List[str],
                          fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequences for model training/inference.
        
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names
            fit_scaler: If True, fit scaler on this data (ONLY for training data!)
            
        Returns:
            X: Sequences of shape (n_samples, sequence_length, n_features)
            y: Labels of shape (n_samples,)
            r: Future returns of shape (n_samples,) for trading simulation
        """
        # Extract raw features
        features = df[feature_cols].values
        labels = df['label'].values
        returns = df['future_return'].values
        
        # Handle scaling
        if fit_scaler:
            features = self.fit_transform(features)
        elif self._fitted:
            features = self.transform(features)
        else:
            log.warning("Scaler not fitted - using per-batch normalization (not recommended)")
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
            features = np.clip(features, -5, 5)
        
        # Create sequences
        seq_len = CONFIG.SEQUENCE_LENGTH
        X, y, r = [], [], []
        
        for i in range(seq_len, len(features)):
            X.append(features[i-seq_len:i])
            y.append(labels[i])
            r.append(returns[i])
        
        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int64),
            np.array(r, dtype=np.float32)
        )
    
    def save_scaler(self, path: str = None) -> bool:
        """
        Save fitted scaler for later use (inference).
        
        Args:
            path: Save path (defaults to MODEL_DIR/scaler.pkl)
            
        Returns:
            True if successful
        """
        if not self._fitted:
            log.warning("Cannot save: scaler not fitted")
            return False
        
        path = Path(path) if path else CONFIG.MODEL_DIR / "scaler.pkl"
        
        try:
            scaler_data = {
                'scaler': self.scaler,
                'feature_means': self._feature_means,
                'feature_stds': self._feature_stds,
                'version': '2.0'
            }
            
            with open(path, 'wb') as f:
                pickle.dump(scaler_data, f)
            
            log.info(f"Scaler saved to {path}")
            return True
            
        except Exception as e:
            log.error(f"Failed to save scaler: {e}")
            return False
    
    def load_scaler(self, path: str = None) -> bool:
        """
        Load a previously saved scaler.
        
        Args:
            path: Load path (defaults to MODEL_DIR/scaler.pkl)
            
        Returns:
            True if successful
        """
        path = Path(path) if path else CONFIG.MODEL_DIR / "scaler.pkl"
        
        if not path.exists():
            log.warning(f"Scaler file not found: {path}")
            return False
        
        try:
            with open(path, 'rb') as f:
                scaler_data = pickle.load(f)
            
            # Handle both old and new formats
            if isinstance(scaler_data, dict):
                self.scaler = scaler_data['scaler']
                self._feature_means = scaler_data.get('feature_means')
                self._feature_stds = scaler_data.get('feature_stds')
            else:
                # Old format: just the scaler
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
                            r: np.ndarray,
                            train_ratio: float = None,
                            val_ratio: float = None) -> Tuple:
        """
        Split data maintaining temporal order (NO SHUFFLING).
        
        This is critical for time series - shuffling would cause data leakage
        as the model would learn from "future" data.
        
        Args:
            X: Feature sequences
            y: Labels
            r: Returns
            train_ratio: Training set ratio (default from config)
            val_ratio: Validation set ratio (default from config)
            
        Returns:
            Tuple of (X_train, y_train, r_train, X_val, y_val, r_val, X_test, y_test, r_test)
        """
        train_ratio = train_ratio or CONFIG.TRAIN_RATIO
        val_ratio = val_ratio or CONFIG.VAL_RATIO
        
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        X_train, y_train, r_train = X[:train_end], y[:train_end], r[:train_end]
        X_val, y_val, r_val = X[train_end:val_end], y[train_end:val_end], r[train_end:val_end]
        X_test, y_test, r_test = X[val_end:], y[val_end:], r[val_end:]
        
        log.info(f"Temporal split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Log class distribution
        for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            if len(labels) > 0:
                dist = np.bincount(labels, minlength=CONFIG.NUM_CLASSES)
                log.debug(f"  {name} classes: DOWN={dist[0]}, NEUTRAL={dist[1]}, UP={dist[2]}")
        
        return (X_train, y_train, r_train,
                X_val, y_val, r_val,
                X_test, y_test, r_test)
    
    def get_class_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Calculate class weights for imbalanced data.
        
        Uses inverse frequency weighting to handle class imbalance.
        
        Args:
            y: Label array
            
        Returns:
            Weight array of shape (n_classes,)
        """
        counts = np.bincount(y, minlength=CONFIG.NUM_CLASSES)
        
        # Inverse frequency with smoothing
        weights = 1.0 / (counts + 1)
        weights = weights / weights.sum()  # Normalize
        
        log.debug(f"Class weights: {weights}")
        return weights