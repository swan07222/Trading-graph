"""
Model Trainer - Complete Training Pipeline

FIXED Issues:
- Proper temporal split per stock (no data leakage)
- Scaler fitted only on training data
- No shuffling of time series data
- Scaler saved with model for inference

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
    Complete training pipeline with proper data handling
    
    Features:
    - Multi-stock data collection
    - Proper temporal train/val/test split (no leakage)
    - Scaler fitting only on training data
    - Model training with callbacks
    - Performance evaluation
    - Model persistence
    
    Usage:
        trainer = Trainer()
        results = trainer.train(stock_codes=['600519', '000858'], epochs=100)
    """
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.processor = DataProcessor()
        self.feature_engine = FeatureEngine()
        
        self.ensemble: Optional[EnsembleModel] = None
        self.history: Dict = {}
        self.input_size: int = 0
    
    def prepare_data(self,
                     stock_codes: List[str] = None,
                     min_samples_per_stock: int = 100,
                     verbose: bool = True) -> Tuple:
        """
        Prepare training data with proper temporal split
        
        CRITICAL: Each stock is split temporally BEFORE combining.
        This prevents any data leakage between train/val/test sets.
        
        Args:
            stock_codes: List of stock codes to use
            min_samples_per_stock: Minimum samples required per stock
            verbose: Show progress bar
            
        Returns:
            Tuple of (X_train, y_train, r_train, X_val, y_val, r_val, X_test, y_test, r_test)
        """
        stocks = stock_codes or CONFIG.STOCK_POOL
        
        log.info(f"Preparing data for {len(stocks)} stocks...")
        log.info(f"Temporal split: Train={CONFIG.TRAIN_RATIO:.0%}, "
                f"Val={CONFIG.VAL_RATIO:.0%}, Test={CONFIG.TEST_RATIO:.0%}")
        
        # Phase 1: Collect and process data for each stock
        stock_data: Dict[str, Dict] = {}
        
        iterator = tqdm(stocks, desc="Loading stocks") if verbose else stocks
        
        for code in iterator:
            try:
                # Fetch historical data
                df = self.fetcher.get_history(code, days=1500)
                
                if len(df) < CONFIG.SEQUENCE_LENGTH + min_samples_per_stock:
                    log.warning(f"Insufficient data for {code}: {len(df)} bars")
                    continue
                
                # Create features
                df = self.feature_engine.create_features(df)
                
                # Create labels
                df = self.processor.create_labels(df)
                
                # Check if we have enough data after processing
                if len(df) < CONFIG.SEQUENCE_LENGTH + 50:
                    log.warning(f"Insufficient processed data for {code}")
                    continue
                
                stock_data[code] = {
                    'df': df,
                    'samples': len(df) - CONFIG.SEQUENCE_LENGTH
                }
                
            except Exception as e:
                log.error(f"Error processing {code}: {e}")
        
        if not stock_data:
            raise ValueError("No valid stock data available for training")
        
        log.info(f"Successfully loaded {len(stock_data)} stocks")
        
        # Phase 2: Fit scaler on training portion of ALL stocks
        log.info("Fitting scaler on training data...")
        
        feature_cols = self.feature_engine.get_feature_columns()
        all_train_features = []
        
        for code, data in stock_data.items():
            df = data['df']
            n = len(df)
            train_end = int(n * CONFIG.TRAIN_RATIO)
            
            train_df = df.iloc[:train_end]
            train_features = train_df[feature_cols].values
            all_train_features.append(train_features)
        
        # Combine and fit scaler
        combined_train_features = np.concatenate(all_train_features, axis=0)
        self.processor.fit_scaler(combined_train_features)
        
        log.info(f"Scaler fitted on {len(combined_train_features)} training samples")
        
        # Phase 3: Create sequences for each split
        all_train = {'X': [], 'y': [], 'r': []}
        all_val = {'X': [], 'y': [], 'r': []}
        all_test = {'X': [], 'y': [], 'r': []}
        
        for code, data in stock_data.items():
            df = data['df']
            n = len(df)
            
            # Temporal split points
            train_end = int(n * CONFIG.TRAIN_RATIO)
            val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
            
            # Split dataframes
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
            
            # Create sequences for each split
            for split_name, split_df, storage in [
                ('train', train_df, all_train),
                ('val', val_df, all_val),
                ('test', test_df, all_test)
            ]:
                if len(split_df) >= CONFIG.SEQUENCE_LENGTH + 5:
                    X, y, r = self.processor.prepare_sequences(
                        split_df, 
                        feature_cols, 
                        fit_scaler=False  # Already fitted!
                    )
                    if len(X) > 0:
                        storage['X'].append(X)
                        storage['y'].append(y)
                        storage['r'].append(r)
        
        # Phase 4: Combine all stocks
        def combine_arrays(storage: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if not storage['X']:
                return np.array([]), np.array([]), np.array([])
            return (
                np.concatenate(storage['X']),
                np.concatenate(storage['y']),
                np.concatenate(storage['r'])
            )
        
        X_train, y_train, r_train = combine_arrays(all_train)
        X_val, y_val, r_val = combine_arrays(all_val)
        X_test, y_test, r_test = combine_arrays(all_test)
        
        # Store input size for model creation
        self.input_size = X_train.shape[2] if len(X_train) > 0 else 0
        
        # Log statistics
        log.info(f"Data prepared:")
        log.info(f"  Train: {len(X_train)} samples")
        log.info(f"  Val:   {len(X_val)} samples")
        log.info(f"  Test:  {len(X_test)} samples")
        log.info(f"  Input size: {self.input_size} features")
        
        # Log class distribution
        if len(y_train) > 0:
            dist = self.processor.get_class_distribution(y_train)
            log.info(f"  Class distribution: DOWN={dist['DOWN']}, "
                    f"NEUTRAL={dist['NEUTRAL']}, UP={dist['UP']}")
        
        # Save scaler for inference
        self.processor.save_scaler()
        
        return (X_train, y_train, r_train,
                X_val, y_val, r_val,
                X_test, y_test, r_test)
    
    def train(self,
              stock_codes: List[str] = None,
              epochs: int = None,
              batch_size: int = None,
              model_names: List[str] = None,
              callback: Callable = None,
              save_model: bool = True) -> Dict:
        """
        Train the ensemble model
        
        Args:
            stock_codes: Stocks to train on (default: CONFIG.STOCK_POOL)
            epochs: Training epochs (default: CONFIG.EPOCHS)
            batch_size: Batch size (default: CONFIG.BATCH_SIZE)
            model_names: Which models to train (default: all)
            callback: Progress callback(model_name, epoch, val_acc)
            save_model: Whether to save the trained model
            
        Returns:
            Training results dict with history and metrics
        """
        epochs = epochs or CONFIG.EPOCHS
        batch_size = batch_size or CONFIG.BATCH_SIZE
        
        log.info("=" * 70)
        log.info("STARTING TRAINING PIPELINE")
        log.info("=" * 70)
        
        start_time = datetime.now()
        
        # Prepare data
        (X_train, y_train, r_train,
         X_val, y_val, r_val,
         X_test, y_test, r_test) = self.prepare_data(stock_codes)
        
        if len(X_train) == 0:
            raise ValueError("No training data available")
        
        # Initialize ensemble
        self.ensemble = EnsembleModel(
            input_size=self.input_size,
            model_names=model_names
        )
        
        log.info(f"Training ensemble with {len(self.ensemble.models)} models...")
        log.info(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Train
        self.history = self.ensemble.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            callback=callback
        )
        
        # Evaluate on test set
        log.info("Evaluating on test set...")
        test_metrics = self._evaluate(X_test, y_test, r_test)
        
        # Save model
        if save_model:
            self.ensemble.save()
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds() / 60
        
        # Compile results
        best_accuracy = max(
            max(h.get('val_acc', [0])) if h.get('val_acc') else 0
            for h in self.history.values()
        )
        
        results = {
            'history': self.history,
            'best_val_accuracy': best_accuracy,
            'test_metrics': test_metrics,
            'input_size': self.input_size,
            'num_models': len(self.ensemble.models),
            'training_time_minutes': training_time,
            'epochs': epochs,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
        
        log.info("=" * 70)
        log.info(f"TRAINING COMPLETE")
        log.info(f"  Best Val Accuracy: {best_accuracy:.2%}")
        log.info(f"  Test Accuracy: {test_metrics.get('accuracy', 0):.2%}")
        log.info(f"  Training Time: {training_time:.1f} minutes")
        log.info("=" * 70)
        
        return results
    
    def _evaluate(self, X: np.ndarray, y: np.ndarray, r: np.ndarray) -> Dict:
        """Evaluate model on test data"""
        if len(X) == 0:
            return {'accuracy': 0, 'trading': {}}
        
        # Get predictions
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
        confidence_mask = confs >= CONFIG.MIN_CONFIDENCE
        
        # Position: +1 for UP, -1 for DOWN, 0 for NEUTRAL
        position = np.zeros_like(preds, dtype=float)
        position[preds == 2] = 1   # UP -> Long
        position[preds == 0] = -1  # DOWN -> Short/Avoid
        
        # Apply confidence filter
        position = position * confidence_mask
        
        # Calculate returns with costs
        costs = CONFIG.COMMISSION * 2 + CONFIG.SLIPPAGE * 2 + CONFIG.STAMP_TAX
        
        strategy_returns = position * returns / 100
        
        # Trading costs only when position changes
        position_changes = np.abs(np.diff(position, prepend=0))
        trade_costs = position_changes * costs
        
        net_returns = strategy_returns - trade_costs
        
        # Buy & hold benchmark
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
            
            gross_profit = trade_returns[trade_returns > 0].sum()
            gross_loss = abs(trade_returns[trade_returns < 0].sum())
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
            drawdown = (cum_strategy - running_max) / (running_max + 1e-8)
            max_drawdown = abs(drawdown.min())
        else:
            max_drawdown = 0
        
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
    
    def save_training_report(self, results: Dict, path: str = None):
        """Save training report to file"""
        import json
        
        path = path or str(CONFIG.DATA_DIR / "training_report.json")
        
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        report = convert(results)
        report['timestamp'] = datetime.now().isoformat()
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        log.info(f"Training report saved to {path}")