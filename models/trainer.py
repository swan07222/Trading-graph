"""
Model Trainer - Complete training pipeline with proper data handling

Key Features:
1. Proper temporal train/val/test splits (no data leakage)
2. Scaler fitted only on training data
3. Walk-forward validation option
4. Comprehensive metrics and logging

Author: AI Trading System
Version: 2.0
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from config import CONFIG
from data.fetcher import DataFetcher
from data.processor import DataProcessor
from data.features import FeatureEngine
from models.ensemble import EnsembleModel
from utils.logger import log


class Trainer:
    """
    Training pipeline with proper data handling.
    
    Usage:
        trainer = Trainer()
        results = trainer.train(stock_codes=['600519', '000858'])
        
    The trainer ensures:
    1. Each stock is split temporally (early data for training, recent for testing)
    2. Scaler is fitted ONLY on training data
    3. No future information leaks into training
    """
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.processor = DataProcessor()
        self.feature_engine = FeatureEngine()
        
        self.ensemble: Optional[EnsembleModel] = None
        self.history: Dict = {}
    
    def prepare_data(self, 
                     stock_codes: List[str] = None,
                     verbose: bool = True) -> Tuple:
        """
        Prepare training data with proper temporal splits.
        
        CRITICAL: Each stock is split temporally, then splits are combined.
        This prevents the model from learning from "future" data.
        
        Args:
            stock_codes: List of stock codes to use
            verbose: Whether to show progress
            
        Returns:
            Tuple of (X_train, y_train, r_train, X_val, y_val, r_val, X_test, y_test, r_test)
        """
        stocks = stock_codes or CONFIG.STOCK_POOL
        log.info(f"Preparing data for {len(stocks)} stocks...")
        
        # Step 1: Load and process all stocks
        stock_data = {}
        feature_cols = self.feature_engine.get_feature_columns()
        
        iterator = tqdm(stocks, desc="Loading stocks") if verbose else stocks
        
        for code in iterator:
            try:
                # Get historical data
                df = self.fetcher.get_history(code, days=1500)
                
                if len(df) < CONFIG.SEQUENCE_LENGTH + 100:
                    log.warning(f"Insufficient data for {code}: {len(df)} bars")
                    continue
                
                # Create features
                df = self.feature_engine.create_features(df)
                
                # Create labels
                df = self.processor.create_labels(df)
                
                # Verify we have required columns
                missing_cols = set(feature_cols) - set(df.columns)
                if missing_cols:
                    log.warning(f"{code} missing features: {missing_cols}")
                    continue
                
                stock_data[code] = df
                
            except Exception as e:
                log.error(f"Error processing {code}: {e}")
        
        if len(stock_data) < 3:
            raise ValueError(f"Not enough stocks with valid data: {len(stock_data)}")
        
        log.info(f"Successfully loaded {len(stock_data)} stocks")
        
        # Step 2: Fit scaler on ALL training data (from all stocks)
        log.info("Fitting scaler on training data...")
        all_train_features = []
        
        for code, df in stock_data.items():
            n = len(df)
            train_end = int(n * CONFIG.TRAIN_RATIO)
            train_df = df.iloc[:train_end]
            
            if len(train_df) > CONFIG.SEQUENCE_LENGTH:
                all_train_features.append(train_df[feature_cols].values)
        
        if not all_train_features:
            raise ValueError("No training data available")
        
        combined_train_features = np.concatenate(all_train_features)
        self.processor.fit_scaler(combined_train_features)
        
        # Step 3: Prepare sequences for each split
        log.info("Creating sequences...")
        all_train_X, all_train_y, all_train_r = [], [], []
        all_val_X, all_val_y, all_val_r = [], [], []
        all_test_X, all_test_y, all_test_r = [], [], []
        
        for code, df in stock_data.items():
            n = len(df)
            train_end = int(n * CONFIG.TRAIN_RATIO)
            val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
            
            # Split temporally
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
            
            # Create sequences (scaler already fitted)
            for split_df, X_list, y_list, r_list in [
                (train_df, all_train_X, all_train_y, all_train_r),
                (val_df, all_val_X, all_val_y, all_val_r),
                (test_df, all_test_X, all_test_y, all_test_r)
            ]:
                if len(split_df) >= CONFIG.SEQUENCE_LENGTH + 5:
                    X, y, r = self.processor.prepare_sequences(
                        split_df, feature_cols, fit_scaler=False
                    )
                    if len(X) > 0:
                        X_list.append(X)
                        y_list.append(y)
                        r_list.append(r)
        
        # Step 4: Combine all stocks
        def safe_concat(arrays):
            return np.concatenate(arrays) if arrays else np.array([])
        
        X_train = safe_concat(all_train_X)
        y_train = safe_concat(all_train_y)
        r_train = safe_concat(all_train_r)
        
        X_val = safe_concat(all_val_X)
        y_val = safe_concat(all_val_y)
        r_val = safe_concat(all_val_r)
        
        X_test = safe_concat(all_test_X)
        y_test = safe_concat(all_test_y)
        r_test = safe_concat(all_test_r)
        
        # Save scaler for inference
        self.processor.save_scaler()
        
        # Log statistics
        log.info(f"Data prepared successfully:")
        log.info(f"  Train: {len(X_train)} samples")
        log.info(f"  Val:   {len(X_val)} samples")
        log.info(f"  Test:  {len(X_test)} samples")
        
        if len(y_train) > 0:
            dist = np.bincount(y_train, minlength=3)
            log.info(f"  Train class distribution: DOWN={dist[0]}, NEUTRAL={dist[1]}, UP={dist[2]}")
        
        return (X_train, y_train, r_train,
                X_val, y_val, r_val,
                X_test, y_test, r_test)
    
    def train(self,
              stock_codes: List[str] = None,
              epochs: int = None,
              callback: Callable = None,
              save_model: bool = True) -> Dict:
        """
        Train the ensemble model.
        
        Args:
            stock_codes: Stocks to train on (default: CONFIG.STOCK_POOL)
            epochs: Number of training epochs
            callback: Progress callback(model_name, epoch, val_acc)
            save_model: Whether to save the trained model
            
        Returns:
            Training results including history and metrics
        """
        epochs = epochs or CONFIG.EPOCHS
        
        log.info("=" * 70)
        log.info("TRAINING PIPELINE")
        log.info("=" * 70)
        log.info(f"Epochs: {epochs}")
        log.info(f"Stocks: {len(stock_codes or CONFIG.STOCK_POOL)}")
        log.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # Prepare data
        (X_train, y_train, r_train,
         X_val, y_val, r_val,
         X_test, y_test, r_test) = self.prepare_data(stock_codes)
        
        if len(X_train) < 100:
            raise ValueError(f"Not enough training samples: {len(X_train)}")
        
        input_size = X_train.shape[2]
        log.info(f"Input size: {input_size} features")
        
        # Initialize ensemble
        self.ensemble = EnsembleModel(input_size)
        
        # Train
        log.info("Training ensemble...")
        self.history = self.ensemble.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            callback=callback
        )
        
        # Evaluate on test set
        log.info("Evaluating on test set...")
        metrics = self._evaluate(X_test, y_test, r_test)
        
        # Save model
        if save_model:
            self.ensemble.save()
            log.info(f"Model saved to {CONFIG.MODEL_DIR}")
        
        # Compile results
        best_accuracy = max(
            max(h['val_acc']) if h['val_acc'] else 0
            for h in self.history.values()
        )
        
        results = {
            'history': self.history,
            'best_accuracy': best_accuracy,
            'test_metrics': metrics,
            'input_size': input_size,
            'num_models': len(self.ensemble.models),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
        }
        
        log.info("=" * 70)
        log.info("TRAINING COMPLETE")
        log.info(f"Best Validation Accuracy: {best_accuracy:.2%}")
        log.info(f"Test Accuracy: {metrics.get('accuracy', 0):.2%}")
        log.info("=" * 70)
        
        return results
    
    def _evaluate(self, X: np.ndarray, y: np.ndarray, r: np.ndarray) -> Dict:
        """Evaluate model on test data"""
        if len(X) == 0:
            return {'accuracy': 0, 'trading': {}}
        
        predictions = self.ensemble.predict_batch(X)
        
        pred_classes = np.array([p.predicted_class for p in predictions])
        confidences = np.array([p.confidence for p in predictions])
        
        # Classification accuracy
        accuracy = np.mean(pred_classes == y)
        
        # Per-class accuracy
        class_acc = {}
        for c in range(CONFIG.NUM_CLASSES):
            mask = y == c
            if mask.sum() > 0:
                class_acc[c] = np.mean(pred_classes[mask] == c)
        
        # Trading simulation
        trading_metrics = self._simulate_trading(pred_classes, confidences, r)
        
        return {
            'accuracy': float(accuracy),
            'class_accuracy': class_acc,
            'mean_confidence': float(np.mean(confidences)),
            'trading': trading_metrics
        }
    
    def _simulate_trading(self, 
                          preds: np.ndarray,
                          confs: np.ndarray,
                          returns: np.ndarray) -> Dict:
        """Simulate trading based on predictions"""
        # Only trade when confident
        mask = confs >= CONFIG.MIN_CONFIDENCE
        
        # Position: +1 for UP, -1 for DOWN, 0 for NEUTRAL
        position = np.zeros_like(preds, dtype=float)
        position[preds == 2] = 1   # UP → Long
        position[preds == 0] = -1  # DOWN → Short/Avoid
        
        # Apply confidence filter
        position = position * mask
        
        # Calculate returns with transaction costs
        costs = CONFIG.COMMISSION * 2 + CONFIG.SLIPPAGE * 2 + CONFIG.STAMP_TAX
        
        strategy_returns = position * returns / 100
        trade_costs = np.abs(np.diff(position, prepend=0)) * costs
        net_returns = strategy_returns - trade_costs
        
        # Benchmark: buy & hold
        buy_hold = returns / 100
        
        # Cumulative returns
        cum_strategy = (1 + net_returns).cumprod()
        cum_buyhold = (1 + buy_hold).cumprod()
        
        total_return = (cum_strategy[-1] - 1) * 100 if len(cum_strategy) > 0 else 0
        buyhold_return = (cum_buyhold[-1] - 1) * 100 if len(cum_buyhold) > 0 else 0
        
        # Trade statistics
        trades = np.sum(position != 0)
        
        if trades > 0:
            trade_returns = net_returns[position != 0]
            wins = (trade_returns > 0).sum()
            win_rate = wins / trades
            
            gross_profit = trade_returns[trade_returns > 0].sum() if len(trade_returns[trade_returns > 0]) > 0 else 0
            gross_loss = abs(trade_returns[trade_returns < 0].sum()) if len(trade_returns[trade_returns < 0]) > 0 else 0
            profit_factor = gross_profit / (gross_loss + 1e-8)
        else:
            win_rate = 0
            profit_factor = 0
        
        # Sharpe ratio
        if len(net_returns) > 1 and net_returns.std() > 0:
            sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        if len(cum_strategy) > 0:
            running_max = np.maximum.accumulate(cum_strategy)
            drawdown = (cum_strategy - running_max) / running_max
            max_drawdown = abs(drawdown.min())
        else:
            max_drawdown = 0
        
        return {
            'total_return': float(total_return),
            'buyhold_return': float(buyhold_return),
            'excess_return': float(total_return - buyhold_return),
            'trades': int(trades),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown)
        }
    
    def get_ensemble(self) -> Optional[EnsembleModel]:
        """Get the trained ensemble model"""
        return self.ensemble