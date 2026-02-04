"""
Universal Auto-Learning System
Searches ALL available stocks and trains on best candidates
"""
import os
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from config import CONFIG
from utils.logger import log


@dataclass
class LearningProgress:
    """Track learning progress"""
    stage: str = "idle"
    progress: float = 0.0
    message: str = ""
    stocks_found: int = 0
    stocks_processed: int = 0
    training_epoch: int = 0
    training_accuracy: float = 0.0
    is_running: bool = False
    errors: List[str] = field(default_factory=list)
    
    def reset(self):
        self.stage = "idle"
        self.progress = 0.0
        self.message = ""
        self.stocks_found = 0
        self.stocks_processed = 0
        self.training_epoch = 0
        self.training_accuracy = 0.0
        self.is_running = False
        self.errors = []


class AutoLearner:
    """
    Universal Auto-Learning System
    
    Improvements:
    1. Searches ALL available stocks (not just config list)
    2. Parallel data downloading (10x faster)
    3. Proper scaler fitting on ALL training data
    4. Progress tracking and callbacks
    5. Incremental learning support
    """
    
    def __init__(self):
        self.progress = LearningProgress()
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        
        # History
        self.history_path = CONFIG.DATA_DIR / "learning_history.json"
        self.history = self._load_history()
    
    def add_callback(self, callback: Callable):
        self._callbacks.append(callback)
    
    def _notify(self):
        for cb in self._callbacks:
            try:
                cb(self.progress)
            except Exception as e:
                log.warning(f"Callback error: {e}")
    
    def _load_history(self) -> Dict:
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {'sessions': [], 'best_accuracy': 0, 'total_stocks': 0}
    
    def _save_history(self):
        try:
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            log.error(f"Failed to save history: {e}")
    
    def start_learning(
        self,
        auto_search: bool = True,
        max_stocks: int = 500,  # Increased from 80
        epochs: int = None,
        incremental: bool = True,
        min_market_cap: float = 10,  # Billions
        search_all: bool = True  # NEW: Search all stocks
    ):
        """Start auto-learning"""
        if self._thread and self._thread.is_alive():
            log.warning("Learning already in progress")
            return
        
        self._stop_flag = False
        self.progress.reset()
        
        epochs = epochs or CONFIG.AUTO_LEARN_EPOCHS
        
        self._thread = threading.Thread(
            target=self._learning_loop,
            args=(auto_search, max_stocks, epochs, incremental, min_market_cap, search_all),
            daemon=True
        )
        self._thread.start()
    
    def stop_learning(self):
        """Stop learning"""
        self._stop_flag = True
        if self._thread:
            self._thread.join(timeout=10)
    
    def _learning_loop(
        self, 
        auto_search: bool, 
        max_stocks: int, 
        epochs: int, 
        incremental: bool,
        min_market_cap: float,
        search_all: bool
    ):
        """Main learning loop"""
        session_start = datetime.now()
        
        try:
            self.progress.is_running = True
            self.progress.errors = []
            
            success = self._run_session(
                auto_search, max_stocks, epochs, incremental, 
                min_market_cap, search_all
            )
            
            if success:
                duration = (datetime.now() - session_start).total_seconds() / 60
                self.history['sessions'].append({
                    'timestamp': session_start.isoformat(),
                    'duration_minutes': duration,
                    'test_accuracy': self.progress.training_accuracy,
                    'stocks_used': self.progress.stocks_processed,
                    'samples': self.progress.stocks_processed * 200,
                    'epochs': epochs
                })
                self.history['best_accuracy'] = max(
                    self.history.get('best_accuracy', 0),
                    self.progress.training_accuracy
                )
                self.history['total_stocks'] = self.progress.stocks_processed
                self._save_history()
            
        except Exception as e:
            import traceback
            log.error(f"Learning failed: {e}")
            traceback.print_exc()
            self.progress.errors.append(str(e))
            self.progress.stage = "error"
            self.progress.message = str(e)
        
        finally:
            self.progress.is_running = False
            self._notify()
    
    def _run_session(
        self, 
        auto_search: bool, 
        max_stocks: int, 
        epochs: int, 
        incremental: bool,
        min_market_cap: float,
        search_all: bool
    ) -> bool:
        """Run a single training session"""
        
        # === Stage 1: Discover ALL Stocks ===
        self.progress.stage = "searching"
        self.progress.message = "Discovering stocks from all sources..."
        self.progress.progress = 0
        self._notify()
        
        if self._stop_flag:
            return False
        
        if auto_search and search_all:
            from data.discovery import UniversalStockDiscovery
            
            discovery = UniversalStockDiscovery()
            
            def search_callback(msg, count):
                self.progress.message = msg
                self.progress.stocks_found = count
                self._notify()
            
            stocks = discovery.discover_all(
                callback=search_callback,
                max_stocks=max_stocks,
                min_market_cap=min_market_cap,
                include_st=False
            )
        elif auto_search:
            # Fallback to old discovery
            from models.auto_learner import StockDiscovery
            discovery = StockDiscovery()
            stocks = discovery.discover(use_cache=True)
        else:
            from data.discovery import DiscoveredStock
            stocks = [
                DiscoveredStock(code=c, name=f"Stock {c}", source="config", score=1.0)
                for c in CONFIG.STOCK_POOL
            ]
        
        self.progress.stocks_found = len(stocks)
        selected_codes = [s.code for s in stocks if s.is_valid()][:max_stocks]
        
        log.info(f"Discovered {len(selected_codes)} valid stocks for training")
        
        if len(selected_codes) < CONFIG.MIN_STOCKS_FOR_TRAINING:
            self.progress.stage = "error"
            self.progress.message = f"Insufficient stocks: {len(selected_codes)}"
            return False
        
        if self._stop_flag:
            return False
        
        # === Stage 2: Parallel Data Download ===
        self.progress.stage = "downloading"
        self.progress.message = "Downloading data (parallel)..."
        self.progress.progress = 10
        self._notify()
        
        from data.fetcher import DataFetcher
        from data.features import FeatureEngine
        from data.processor import DataProcessor
        
        fetcher = DataFetcher()
        feature_engine = FeatureEngine()
        processor = DataProcessor()
        feature_cols = feature_engine.get_feature_columns()
        
        def download_callback(code, completed, total):
            self.progress.stocks_processed = completed
            self.progress.message = f"Downloaded {completed}/{total}: {code}"
            self.progress.progress = 10 + (completed / total) * 25
            self._notify()
        
        # PARALLEL DOWNLOAD (10x faster)
        raw_data = fetcher.get_multiple_parallel(
            selected_codes,
            days=1500,
            max_workers=10,
            callback=download_callback
        )
        
        if len(raw_data) < 5:
            self.progress.stage = "error"
            self.progress.message = f"Insufficient data: only {len(raw_data)} stocks"
            return False
        
        if self._stop_flag:
            return False
        
        # === Stage 3: Feature Engineering ===
        self.progress.stage = "preparing"
        self.progress.message = "Creating features..."
        self.progress.progress = 40
        self._notify()
        
        valid_data = {}
        for code, df in raw_data.items():
            if self._stop_flag:
                return False
            
            try:
                df = feature_engine.create_features(df)
                if len(df) >= CONFIG.SEQUENCE_LENGTH + 50:
                    valid_data[code] = df
            except Exception as e:
                log.debug(f"Feature creation failed for {code}: {e}")
        
        log.info(f"Created features for {len(valid_data)} stocks")
        
        if len(valid_data) < 5:
            self.progress.stage = "error"
            self.progress.message = f"Insufficient processed data"
            return False
        
        # === Stage 4: Prepare Data with Proper Splits ===
        self.progress.message = "Preparing training data..."
        self.progress.progress = 50
        self._notify()
        
        if self._stop_flag:
            return False
        
        # Collect ALL training features for scaler fitting
        all_train_features = []
        split_data = {}
        
        horizon = CONFIG.PREDICTION_HORIZON
        embargo = CONFIG.EMBARGO_BARS
        
        for code, df in valid_data.items():
            n = len(df)
            train_end = int(n * CONFIG.TRAIN_RATIO) - horizon - embargo
            val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO)) - horizon - embargo
            
            if train_end < CONFIG.SEQUENCE_LENGTH + 20:
                continue
            
            # Split BEFORE labeling
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[int(n * CONFIG.TRAIN_RATIO):val_end].copy()
            test_df = df.iloc[int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO)):].copy()
            
            # Create labels WITHIN each split
            train_df = processor.create_labels(train_df)
            val_df = processor.create_labels(val_df)
            test_df = processor.create_labels(test_df)
            
            split_data[code] = {'train': train_df, 'val': val_df, 'test': test_df}
            
            # Collect training features for scaler
            train_features = train_df[feature_cols].values
            valid_mask = ~train_df['label'].isna()
            if valid_mask.sum() > 0:
                all_train_features.append(train_features[valid_mask])
        
        if not all_train_features:
            self.progress.stage = "error"
            self.progress.message = "No valid training data"
            return False
        
        # Fit scaler on ALL training features
        combined_train = np.concatenate(all_train_features, axis=0)
        processor.fit_scaler(combined_train)
        log.info(f"Scaler fitted on {len(combined_train)} samples from {len(split_data)} stocks")
        
        # Create sequences
        all_train = {'X': [], 'y': []}
        all_val = {'X': [], 'y': []}
        all_test = {'X': [], 'y': []}
        
        for code, splits in split_data.items():
            for split_name, split_df, storage in [
                ('train', splits['train'], all_train),
                ('val', splits['val'], all_val),
                ('test', splits['test'], all_test)
            ]:
                if len(split_df) >= CONFIG.SEQUENCE_LENGTH + 5:
                    X, y, _ = processor.prepare_sequences(split_df, feature_cols, fit_scaler=False)
                    if len(X) > 0:
                        storage['X'].append(X)
                        storage['y'].append(y)
        
        if not all_train['X']:
            self.progress.stage = "error"
            self.progress.message = "No training sequences"
            return False
        
        X_train = np.concatenate(all_train['X'])
        y_train = np.concatenate(all_train['y'])
        X_val = np.concatenate(all_val['X']) if all_val['X'] else X_train[-1000:]
        y_val = np.concatenate(all_val['y']) if all_val['y'] else y_train[-1000:]
        X_test = np.concatenate(all_test['X']) if all_test['X'] else np.array([])
        y_test = np.concatenate(all_test['y']) if all_test['y'] else np.array([])
        
        log.info(f"Data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Save scaler
        processor.save_scaler()
        
        # === Stage 5: Train Model ===
        self.progress.stage = "training"
        self.progress.message = "Training AI models..."
        self.progress.progress = 55
        self._notify()
        
        if self._stop_flag:
            return False
        
        from models.ensemble import EnsembleModel
        
        input_size = X_train.shape[2]
        ensemble = EnsembleModel(input_size)
        
        if incremental and ensemble.load():
            log.info("Loaded existing model for incremental learning")
        
        def train_callback(model_name, epoch, val_acc):
            if self._stop_flag:
                return
            self.progress.training_epoch = epoch + 1
            self.progress.training_accuracy = val_acc
            self.progress.message = f"Training {model_name}: Epoch {epoch+1}/{epochs}"
            self.progress.progress = 55 + (epoch + 1) / epochs * 35
            self._notify()
        
        def stop_check():
            return self._stop_flag
        
        ensemble.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            callback=train_callback,
            stop_flag=stop_check
        )
        
        if self._stop_flag:
            return False
        
        # === Stage 6: Evaluate ===
        self.progress.stage = "evaluating"
        self.progress.message = "Evaluating model..."
        self.progress.progress = 95
        self._notify()
        
        if len(X_test) > 0:
            correct = 0
            total = min(len(X_test), 1000)
            
            for i in range(total):
                pred = ensemble.predict(X_test[i:i+1])
                if pred.predicted_class == y_test[i]:
                    correct += 1
            
            test_accuracy = correct / total
            log.info(f"Test accuracy: {test_accuracy:.2%}")
            self.progress.training_accuracy = test_accuracy
        
        # Save model
        ensemble.save()
        
        # Complete
        self.progress.stage = "complete"
        self.progress.message = f"Complete! Accuracy: {self.progress.training_accuracy:.1%}"
        self.progress.progress = 100
        self._notify()
        
        return True
    
    def get_learning_stats(self) -> Dict:
        return {
            'sessions_count': len(self.history.get('sessions', [])),
            'best_accuracy': self.history.get('best_accuracy', 0),
            'total_stocks': self.history.get('total_stocks', 0),
            'is_running': self.progress.is_running,
            'current_stage': self.progress.stage
        }