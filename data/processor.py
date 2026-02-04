"""
Data Processor - Clean and prepare data for training
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import RobustScaler

from config import CONFIG
from utils.logger import log


class DataProcessor:
    """
    Processes raw OHLCV data into training-ready format
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self._fitted = False
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create classification labels based on future returns
        
        Labels:
            0 = DOWN (return <= -2%)
            1 = NEUTRAL (-2% < return < 2%)
            2 = UP (return >= 2%)
        """
        df = df.copy()
        
        # Calculate future return
        future_price = df['close'].shift(-CONFIG.PREDICTION_HORIZON)
        future_return = (future_price / df['close'] - 1) * 100
        
        # Create labels
        df['label'] = 1  # NEUTRAL
        df.loc[future_return >= CONFIG.UP_THRESHOLD, 'label'] = 2  # UP
        df.loc[future_return <= CONFIG.DOWN_THRESHOLD, 'label'] = 0  # DOWN
        df['future_return'] = future_return
        
        # Remove rows without valid labels
        df = df.iloc[:-CONFIG.PREDICTION_HORIZON]
        
        return df
    
    def prepare_sequences(self, 
                          df: pd.DataFrame,
                          feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequences for model training
        
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names
            
        Returns:
            X: Shape (n_samples, sequence_length, n_features)
            y: Shape (n_samples,)
            returns: Shape (n_samples,) - for trading simulation
        """
        # Extract features
        features = df[feature_cols].values
        labels = df['label'].values
        returns = df['future_return'].values
        
        # Normalize features
        if not self._fitted:
            features = self.scaler.fit_transform(features)
            self._fitted = True
        else:
            features = self.scaler.transform(features)
        
        # Clip outliers
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
    
    def split_data(self, 
                   X: np.ndarray, 
                   y: np.ndarray,
                   r: np.ndarray) -> Tuple:
        """
        Split data into train/val/test sets
        Maintains temporal order (no shuffling)
        """
        n = len(X)
        train_end = int(n * CONFIG.TRAIN_RATIO)
        val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
        
        X_train, y_train, r_train = X[:train_end], y[:train_end], r[:train_end]
        X_val, y_val, r_val = X[train_end:val_end], y[train_end:val_end], r[train_end:val_end]
        X_test, y_test, r_test = X[val_end:], y[val_end:], r[val_end:]
        
        log.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return (X_train, y_train, r_train,
                X_val, y_val, r_val,
                X_test, y_test, r_test)
    
    def get_class_weights(self, y: np.ndarray) -> np.ndarray:
        """Calculate class weights for imbalanced data"""
        counts = np.bincount(y, minlength=CONFIG.NUM_CLASSES)
        weights = 1.0 / (counts + 1)
        weights = weights / weights.sum()
        return weights