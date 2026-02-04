"""
Model Trainer - Complete Training Pipeline

FIXED Issues:
- Proper temporal split per stock (no data leakage)
- Scaler fitted only on training data
- Labels created WITHIN each split
- Scaler saved with model for inference
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
    
    CRITICAL: Labels are created WITHIN each temporal split to prevent leakage.
    """
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.processor = DataProcessor()
        self.feature_engine = FeatureEngine()
        
        self.ensemble: Optional[EnsembleModel] = None
        self.history: Dict = {}
        self.input_size: int = 0
    
    def prepare_data(
        self,
        stock_codes: List[str] = None,
        min_samples_per_stock: int = 100,
        verbose: bool = True
    ) -> Tuple:
        """
        Prepare training data with proper temporal split
        
        CRITICAL: Each stock is split temporally BEFORE labeling.
        Labels are created WITHIN each split to prevent leakage.
        """
        stocks = stock_codes or CONFIG.STOCK_POOL
        
        log.info(f"Preparing data for {len(stocks)} stocks...")
        log.info(f"Temporal split: Train={CONFIG.TRAIN_RATIO:.0%}, "
                f"Val={CONFIG.VAL_RATIO:.0%}, Test={CONFIG.TEST_RATIO:.0%}")
        
        # Phase 1: Collect raw data and create features (NO LABELS YET)
        stock_data: Dict[str, Dict] = {}
        feature_cols = self.feature_engine.get_feature_columns()
        
        iterator = tqdm(stocks, desc="Loading stocks") if verbose else stocks
        
        for code in iterator:
            try:
                df = self.fetcher.get_history(code, days=1500)
                
                if len(df) < CONFIG.SEQUENCE_LENGTH + min_samples_per_stock:
                    log.warning(f"Insufficient data for {code}: {len(df)} bars")
                    continue
                
                # Create features ONLY (no labels)
                df = self.feature_engine.create_features(df)
                
                if len(df) < CONFIG.SEQUENCE_LENGTH + 50:
                    log.warning(f"Insufficient processed data for {code}")
                    continue
                
                stock_data[code] = {'df': df}
                
            except Exception as e:
                log.error(f"Error processing {code}: {e}")
        
        if not stock_data:
            raise ValueError("No valid stock data available for training")
        
        log.info(f"Successfully loaded {len(stock_data)} stocks")
        
        # Phase 2: Split each stock temporally BEFORE labeling
        # Then collect training features for scaler fitting
        
        all_train_features = []
        split_data = {}
        
        horizon = CONFIG.PREDICTION_HORIZON
        embargo = CONFIG.EMBARGO_BARS
        
        for code, data in stock_data.items():
            df = data['df']
            n = len(df)
            
            # Calculate split points with embargo
            train_end = int(n * CONFIG.TRAIN_RATIO) - horizon - embargo
            val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO)) - horizon - embargo
            
            if train_end < CONFIG.SEQUENCE_LENGTH + 20:
                log.warning(f"Insufficient training data for {code}")
                continue
            
            # Split raw data BEFORE labeling
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[int(n * CONFIG.TRAIN_RATIO):val_end].copy()
            test_df = df.iloc[int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO)):].copy()
            
            # Create labels WITHIN each split
            train_df = self.processor.create_labels(train_df)
            val_df = self.processor.create_labels(val_df)
            test_df = self.processor.create_labels(test_df)
            
            split_data[code] = {
                'train': train_df,
                'val': val_df,
                'test': test_df
            }
            
            # Collect training features for scaler
            train_features = train_df[feature_cols].values
            valid_mask = ~train_df['label'].isna()
            if valid_mask.sum() > 0:
                all_train_features.append(train_features[valid_mask])
        
        if not all_train_features:
            raise ValueError("No valid training data after split")
        
        # Phase 3: Fit scaler on training data ONLY
        log.info("Fitting scaler on training data...")
        combined_train_features = np.concatenate(all_train_features, axis=0)
        self.processor.fit_scaler(combined_train_features)
        log.info(f"Scaler fitted on {len(combined_train_features)} training samples")
        
        # Phase 4: Create sequences for each split
        all_train = {'X': [], 'y': [], 'r': []}
        all_val = {'X': [], 'y': [], 'r': []}
        all_test = {'X': [], 'y': [], 'r': []}
        
        for code, splits in split_data.items():
            for split_name, split_df, storage in [
                ('train', splits['train'], all_train),
                ('val', splits['val'], all_val),
                ('test', splits['test'], all_test)
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
        
        # Phase 5: Combine all stocks
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
        
        self.input_size = X_train.shape[2] if len(X_train) > 0 else 0
        
        # Log statistics
        log.info(f"Data prepared:")
        log.info(f"  Train: {len(X_train)} samples")
        log.info(f"  Val:   {len(X_val)} samples")
        log.info(f"  Test:  {len(X_test)} samples")
        log.info(f"  Input size: {self.input_size} features")
        
        if len(y_train) > 0:
            dist = self.processor.get_class_distribution(y_train)
            log.info(f"  Class distribution: DOWN={dist['DOWN']}, "
                    f"NEUTRAL={dist['NEUTRAL']}, UP={dist['UP']}")
        
        # Save scaler for inference
        self.processor.save_scaler()
        
        return (X_train, y_train, r_train,
                X_val, y_val, r_val,
                X_test, y_test, r_test)
    
    def train(
        self,
        stock_codes: List[str] = None,
        epochs: int = None,
        batch_size: int = None,
        model_names: List[str] = None,
        callback: Callable = None,
        stop_flag: Callable = None,
        save_model: bool = True
    ) -> Dict:
        """Train the ensemble model"""
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
            callback=callback,
            stop_flag=stop_flag
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
        best_accuracy = 0
        for h in self.history.values():
            if h.get('val_acc'):
                best_accuracy = max(best_accuracy, max(h['val_acc']))
        
        results = {
            'history': self.history,
            'best_accuracy': best_accuracy,
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
        
        predictions = self.ensemble.predict_batch(X)
        
        pred_classes = np.array([p.predicted_class for p in predictions])
        confidences = np.array([p.confidence for p in predictions])
        
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
    
    def _simulate_trading(
        self,
        preds: np.ndarray,
        confs: np.ndarray,
        returns: np.ndarray
    ) -> Dict:
        """Simulate trading with CORRECT compounding"""
        confidence_mask = confs >= CONFIG.MIN_CONFIDENCE
        
        # Position: +1 for UP prediction (long only)
        position = np.zeros_like(preds, dtype=float)
        position[preds == 2] = 1  # UP -> Long
        position = position * confidence_mask
        
        # Daily returns (decimal, not percentage)
        daily_returns = returns / 100
        
        # Strategy returns with transaction costs
        costs = CONFIG.COMMISSION * 2 + CONFIG.SLIPPAGE * 2 + CONFIG.STAMP_TAX
        
        # Calculate position changes for cost application
        position_changes = np.abs(np.diff(position, prepend=0))
        
        # Net daily returns: position * market_return - costs_on_trade
        strategy_returns = position * daily_returns - position_changes * costs
        
        # CORRECT: Compound returns multiplicatively
        strategy_equity = np.cumprod(1 + strategy_returns)
        buyhold_equity = np.cumprod(1 + daily_returns)
        
        total_return = (strategy_equity[-1] - 1) * 100 if len(strategy_equity) > 0 else 0
        buyhold_return = (buyhold_equity[-1] - 1) * 100 if len(buyhold_equity) > 0 else 0
        
        # Trade-level statistics (FIXED: compound per-trade)
        trades = []
        in_trade = False
        trade_equity = 1.0
        
        for i in range(len(position)):
            if position[i] > 0:
                if not in_trade:
                    in_trade = True
                    trade_equity = 1.0
                trade_equity *= (1 + strategy_returns[i])
            else:
                if in_trade:
                    trades.append(trade_equity - 1)  # Trade return
                    in_trade = False
                    trade_equity = 1.0
        
        if in_trade:
            trades.append(trade_equity - 1)
        
        # Statistics
        num_trades = len(trades)
        if num_trades > 0:
            wins = [t for t in trades if t > 0]
            losses = [t for t in trades if t < 0]
            win_rate = len(wins) / num_trades
            
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 1e-8
            profit_factor = gross_profit / gross_loss
        else:
            win_rate = 0
            profit_factor = 0
        
        # Risk metrics
        if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
            sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        running_max = np.maximum.accumulate(strategy_equity)
        drawdown = (strategy_equity - running_max) / (running_max + 1e-8)
        max_drawdown = abs(np.min(drawdown))
        
        return {
            'total_return': total_return,
            'buyhold_return': buyhold_return,
            'excess_return': total_return - buyhold_return,
            'trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }
    
    def get_ensemble(self) -> Optional[EnsembleModel]:
        return self.ensemble
    
    def save_training_report(self, results: Dict, path: str = None):
        """Save training report to file"""
        import json
        
        path = path or str(CONFIG.DATA_DIR / "training_report.json")
        
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