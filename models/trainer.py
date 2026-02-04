"""
Model Trainer - Handles the complete training pipeline
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Callable
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
    Handles complete training pipeline:
    1. Data collection
    2. Feature engineering
    3. Model training
    4. Evaluation
    5. Model saving
    """
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.processor = DataProcessor()
        self.feature_engine = FeatureEngine()
        
        self.ensemble: Optional[EnsembleModel] = None
        self.history: Dict = {}
    
    def prepare_data(self, 
                     stock_codes: List[str] = None,
                     verbose: bool = True) -> tuple:
        """
        Prepare training data from multiple stocks
        
        Args:
            stock_codes: List of stock codes to use
            verbose: Whether to show progress
            
        Returns:
            Tuple of (X_train, y_train, r_train, X_val, y_val, r_val, X_test, y_test, r_test)
        """
        stocks = stock_codes or CONFIG.STOCK_POOL
        
        log.info(f"Preparing data for {len(stocks)} stocks...")
        
        all_X, all_y, all_r = [], [], []
        
        iterator = tqdm(stocks, desc="Loading stocks") if verbose else stocks
        
        for code in iterator:
            try:
                # Get historical data
                df = self.fetcher.get_history(code)
                
                if len(df) < CONFIG.SEQUENCE_LENGTH + 50:
                    log.warning(f"Insufficient data for {code}: {len(df)} bars")
                    continue
                
                # Create features
                df = self.feature_engine.create_features(df)
                
                # Create labels
                df = self.processor.create_labels(df)
                
                # Prepare sequences
                feature_cols = self.feature_engine.get_feature_columns()
                X, y, r = self.processor.prepare_sequences(df, feature_cols)
                
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
                    all_r.append(r)
                    
            except Exception as e:
                log.error(f"Error processing {code}: {e}")
        
        if not all_X:
            raise ValueError("No data available for training")
        
        # Concatenate all data
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        r = np.concatenate(all_r, axis=0)
        
        # Shuffle
        idx = np.random.permutation(len(X))
        X, y, r = X[idx], y[idx], r[idx]
        
        log.info(f"Total samples: {len(X)}")
        log.info(f"Class distribution: DOWN={np.sum(y==0)}, NEUTRAL={np.sum(y==1)}, UP={np.sum(y==2)}")
        
        # Split data
        return self.processor.split_data(X, y, r)
    
    def train(self,
              stock_codes: List[str] = None,
              epochs: int = None,
              callback: Callable = None,
              save_model: bool = True) -> Dict:
        """
        Train the ensemble model
        
        Args:
            stock_codes: Stocks to train on
            epochs: Number of epochs
            callback: Progress callback(model_name, epoch, val_acc)
            save_model: Whether to save the trained model
            
        Returns:
            Training results including history and metrics
        """
        epochs = epochs or CONFIG.EPOCHS
        
        log.info("=" * 60)
        log.info("Starting Training Pipeline")
        log.info("=" * 60)
        
        # Prepare data
        (X_train, y_train, r_train,
         X_val, y_val, r_val,
         X_test, y_test, r_test) = self.prepare_data(stock_codes)
        
        input_size = X_train.shape[2]
        
        # Initialize ensemble
        self.ensemble = EnsembleModel(input_size)
        
        # Train
        log.info(f"Training ensemble with {len(self.ensemble.models)} models...")
        
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
            'num_models': len(self.ensemble.models)
        }
        
        log.info("=" * 60)
        log.info(f"Training Complete! Best accuracy: {best_accuracy:.2%}")
        log.info("=" * 60)
        
        return results
    
    def _evaluate(self, X: np.ndarray, y: np.ndarray, r: np.ndarray) -> Dict:
        """Evaluate model on test data"""
        predictions = self.ensemble.predict_batch(X)
        
        pred_classes = np.array([p.predicted_class for p in predictions])
        confidences = np.array([p.confidence for p in predictions])
        
        # Classification metrics
        accuracy = np.mean(pred_classes == y)
        
        # Per-class accuracy
        class_acc = {}
        for c in range(CONFIG.NUM_CLASSES):
            mask = y == c
            if mask.sum() > 0:
                class_acc[c] = np.mean(pred_classes[mask] == c)
            else:
                class_acc[c] = 0
        
        # Trading simulation
        trading_metrics = self._simulate_trading(pred_classes, confidences, r)
        
        return {
            'accuracy': accuracy,
            'class_accuracy': class_acc,
            'mean_confidence': np.mean(confidences),
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
        position[preds == 2] = 1   # UP -> Long
        position[preds == 0] = -1  # DOWN -> Short/Avoid
        
        # Apply confidence filter
        position = position * mask
        
        # Calculate returns
        costs = CONFIG.COMMISSION * 2 + CONFIG.SLIPPAGE * 2 + CONFIG.STAMP_TAX
        
        strategy_returns = position * returns / 100
        trade_costs = np.abs(np.diff(position, prepend=0)) * costs
        net_returns = strategy_returns - trade_costs
        
        # Buy & hold benchmark
        buy_hold = returns / 100
        
        # Cumulative returns
        cum_strategy = (1 + net_returns).cumprod()
        cum_buyhold = (1 + buy_hold).cumprod()
        
        total_return = (cum_strategy[-1] - 1) * 100
        buyhold_return = (cum_buyhold[-1] - 1) * 100
        
        # Trade statistics
        trades = np.sum(position != 0)
        
        if trades > 0:
            trade_returns = net_returns[position != 0]
            wins = (trade_returns > 0).sum()
            win_rate = wins / trades
            
            gross_profit = trade_returns[trade_returns > 0].sum()
            gross_loss = abs(trade_returns[trade_returns < 0].sum())
            profit_factor = gross_profit / (gross_loss + 1e-8)
        else:
            win_rate = 0
            profit_factor = 0
        
        # Sharpe ratio
        if net_returns.std() > 0:
            sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        running_max = np.maximum.accumulate(cum_strategy)
        drawdown = (cum_strategy - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        return {
            'total_return': total_return,
            'buyhold_return': buyhold_return,
            'excess_return': total_return - buyhold_return,
            'trades': int(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }
    
    def get_ensemble(self) -> Optional[EnsembleModel]:
        """Get the trained ensemble model"""
        return self.ensemble