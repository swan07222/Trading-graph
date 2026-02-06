# models/auto_learner.py
"""
Continuous Auto-Learning System
"""
import os
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from config.settings import CONFIG  # FIXED
from utils.logger import get_logger
from utils.cancellation import CancellationToken, CancelledException
from data.fetcher import get_fetcher

log = get_logger(__name__)


@dataclass
class LearningProgress:
    """Track learning progress"""
    stage: str = "idle"
    progress: float = 0.0
    message: str = ""
    stocks_found: int = 0
    stocks_processed: int = 0
    stocks_total: int = 0
    training_epoch: int = 0
    training_total_epochs: int = 0
    training_accuracy: float = 0.0
    validation_accuracy: float = 0.0
    is_running: bool = False
    is_paused: bool = False
    errors: List[str] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    
    # Continuous learning stats
    total_training_sessions: int = 0
    total_stocks_learned: int = 0
    total_training_hours: float = 0.0
    best_accuracy_ever: float = 0.0
    
    def reset(self):
        self.stage = "idle"
        self.progress = 0.0
        self.message = ""
        self.stocks_processed = 0
        self.training_epoch = 0
        self.is_running = False
        self.is_paused = False
        self.errors = []
    
    def to_dict(self) -> Dict:
        return {
            'stage': self.stage,
            'progress': self.progress,
            'message': self.message,
            'stocks_found': self.stocks_found,
            'stocks_processed': self.stocks_processed,
            'training_epoch': self.training_epoch,
            'training_accuracy': self.training_accuracy,
            'is_running': self.is_running,
            'total_sessions': self.total_training_sessions,
            'best_accuracy': self.best_accuracy_ever,
        }


class ContinuousLearner:
    """
    Continuous learning system
    """
    
    MODE_FULL = "full"
    MODE_INCREMENTAL = "incremental"
    MODE_ONLINE = "online"
    
    def __init__(self):
        self.progress = LearningProgress()
        self._cancel_token = CancellationToken()
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[LearningProgress], None]] = []
        self._lock = threading.RLock()
        
        self._stock_queue: Queue = Queue()
        self._data_queue: Queue = Queue()
        
        self._processed_stocks: Set[str] = set()
        self._failed_stocks: Set[str] = set()
        
        self.history_path = CONFIG.data_dir / "continuous_learning_history.json"
        self.state_path = CONFIG.data_dir / "learner_state.json"
        self._load_state()
    
    def add_callback(self, callback: Callable[[LearningProgress], None]):
        with self._lock:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[LearningProgress], None]):
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def _notify(self):
        self.progress.last_update = datetime.now()
        with self._lock:
            callbacks = self._callbacks.copy()
        
        for cb in callbacks:
            try:
                cb(self.progress)
            except Exception as e:
                log.warning(f"Callback error: {e}")
    
    def _update_progress(self, stage: str = None, message: str = None, 
                         progress: float = None, **kwargs):
        if stage:
            self.progress.stage = stage
        if message:
            self.progress.message = message
        if progress is not None:
            self.progress.progress = progress
        
        for key, value in kwargs.items():
            if hasattr(self.progress, key):
                setattr(self.progress, key, value)
        
        self._notify()
    
    def start(
        self,
        mode: str = MODE_FULL,
        max_stocks: int = None,
        epochs_per_cycle: int = 10,
        min_market_cap: float = 10,
        include_all_markets: bool = True,
        continuous: bool = True,
        learning_while_trading: bool = True,
        interval: str = "1m",
        prediction_horizon: int = 30,
        lookback_bars: int = 3000,
        cycle_interval_seconds: int = 900,  # 15 minutes
        incremental: bool = True,
    ):
        """
        Continuous auto-learning:
        - Discover liquid stocks
        - Prefetch intraday (dealing) bars into DB
        - Train incrementally (few epochs) repeatedly until stopped
        """
        if self._thread and self._thread.is_alive():
            if self.progress.is_paused:
                self.resume()
                return
            log.warning("Learning already in progress")
            return

        self._cancel_token = CancellationToken()
        self.progress.reset()
        self.progress.is_running = True

        self._thread = threading.Thread(
            target=self._continuous_learning_loop,
            args=(
                mode,
                max_stocks,
                epochs_per_cycle,
                min_market_cap,
                include_all_markets,
                continuous,
                learning_while_trading,
                str(interval).lower(),
                int(prediction_horizon),
                int(lookback_bars),
                int(cycle_interval_seconds),
                bool(incremental),
            ),
            daemon=True
        )
        self._thread.start()

        log.info(
            f"Continuous learning started: interval={interval}, horizon={prediction_horizon}, "
            f"lookback={lookback_bars}, cycle={cycle_interval_seconds}s, incremental={incremental}"
        )
    
    def run(
        self,
        mode: str = MODE_FULL,
        max_stocks: int = None,
        epochs_per_cycle: int = 50,
        min_market_cap: float = 10,
        include_all_markets: bool = True,
        continuous: bool = False,
        learning_while_trading: bool = False,
        auto_search: bool = True,
        search_all: bool = True,
        epochs: int = None,          
        **_unused_kwargs             
    ):
        """Synchronous run for CLI"""
        self.start(
            mode=mode,
            max_stocks=max_stocks or 500,
            epochs_per_cycle=epochs_per_cycle,
            min_market_cap=min_market_cap,
            include_all_markets=include_all_markets,
            continuous=continuous,
            learning_while_trading=learning_while_trading
        )
        
        if self._thread:
            self._thread.join()
        
        return self.progress
    
    def stop(self):
        """Stop learning gracefully"""
        self._cancel_token.cancel()
        if self._thread:
            self._thread.join(timeout=30)
        self._save_state()
        self.progress.is_running = False
        self._notify()
        log.info("Continuous learning stopped")
    
    def pause(self):
        self.progress.is_paused = True
        self._notify()
        log.info("Learning paused")
    
    def resume(self):
        self.progress.is_paused = False
        self._notify()
        log.info("Learning resumed")
    
    def _continuous_learning_loop(
        self,
        mode: str,
        max_stocks: int,
        epochs_per_cycle: int,
        min_market_cap: float,
        include_all_markets: bool,
        continuous: bool,
        learning_while_trading: bool,
        interval: str,
        prediction_horizon: int,
        lookback_bars: int,
        cycle_interval_seconds: int,
        incremental: bool,
    ):
        """Main continuous learning loop: discover -> prefetch intraday -> train -> repeat until stop."""
        cycle = 0

        try:
            while not self._cancel_token.is_cancelled:
                cycle += 1
                self._update_progress(
                    stage="cycle_start",
                    message=f"=== Learning Cycle {cycle} (intraday dealing: {interval}) ===",
                    progress=0.0
                )

                while self.progress.is_paused and not self._cancel_token.is_cancelled:
                    time.sleep(1)

                if self._cancel_token.is_cancelled:
                    break

                ok = self._run_learning_cycle(
                    mode=mode,
                    max_stocks=max_stocks or 200,
                    epochs=max(1, int(epochs_per_cycle)),
                    min_market_cap=min_market_cap,
                    include_all_markets=include_all_markets,
                    interval=interval,
                    prediction_horizon=int(prediction_horizon),
                    lookback_bars=int(lookback_bars),
                    incremental=bool(incremental),
                )

                if ok:
                    self.progress.total_training_sessions += 1
                    self._save_state()

                if not continuous:
                    break

                # Wait between cycles, but remain cancellable
                wait_s = int(cycle_interval_seconds)
                self._update_progress(
                    stage="waiting",
                    message=f"Waiting {wait_s}s until next cycle... (press Stop to cancel)",
                    progress=100.0
                )

                for _ in range(wait_s):
                    if self._cancel_token.is_cancelled:
                        break
                    time.sleep(1)

        except CancelledException:
            log.info("Learning cancelled")
        except Exception as e:
            log.error(f"Learning error: {e}")
            import traceback
            traceback.print_exc()
            self.progress.errors.append(str(e))
        finally:
            self.progress.is_running = False
            self._save_state()
            self._notify()
    
    def _run_learning_cycle(
        self,
        mode: str,
        max_stocks: int,
        epochs: int,
        min_market_cap: float,
        include_all_markets: bool,
        interval: str,
        prediction_horizon: int,
        lookback_bars: int,
        incremental: bool,
    ) -> bool:
        """One continuous cycle: discover -> prefetch intraday -> train."""
        start_time = datetime.now()

        try:
            # 1) Discover liquid stocks continuously
            self._update_progress(
                stage="discovering",
                message="Discovering liquid stocks (high dealing activity)...",
                progress=5.0
            )

            from data.discovery import UniversalStockDiscovery
            discovery = UniversalStockDiscovery()

            def cb(msg, count):
                if self._cancel_token.is_cancelled:
                    raise CancelledException()
                self._update_progress(message=msg, stocks_found=count)

            stocks = discovery.discover_all(
                callback=cb,
                max_stocks=int(max_stocks),
                min_market_cap=float(min_market_cap),
                include_st=False,
            ) or []

            codes = [s.code for s in stocks if getattr(s, "code", "")] or CONFIG.stock_pool[: int(max_stocks)]
            codes = [str(c).zfill(6) for c in codes]
            self.progress.stocks_found = len(codes)
            self.progress.stocks_total = len(codes)

            if not codes:
                self._update_progress(stage="error", message="No stocks discovered")
                return False

            # 2) Prefetch intraday dealing bars into DB
            self._update_progress(
                stage="downloading",
                message=f"Prefetching dealing data: {interval} bars for {len(codes)} stocks...",
                progress=20.0
            )

            from data.fetcher import get_fetcher
            fetcher = get_fetcher()

            ok_cnt = 0
            for i, code in enumerate(codes, start=1):
                if self._cancel_token.is_cancelled:
                    raise CancelledException()

                try:
                    df = fetcher.get_history(
                        code,
                        interval=interval,
                        bars=int(lookback_bars),
                        days=int(lookback_bars),
                        use_cache=True,
                        update_db=True,
                    )
                    if df is not None and not df.empty:
                        ok_cnt += 1
                except Exception:
                    pass

                self.progress.stocks_processed = i
                pct = 20.0 + 35.0 * (i / max(1, len(codes)))
                self._update_progress(
                    message=f"Prefetched {interval} {i}/{len(codes)}: {code}",
                    progress=pct
                )

            if ok_cnt < max(5, int(0.1 * len(codes))):
                self._update_progress(stage="error", message=f"Too few intraday histories fetched: {ok_cnt}")
                return False

            # 3) Train (incremental) on dealing bars
            self._update_progress(
                stage="training",
                message=f"Training incrementally (epochs={epochs}) on intraday dealing features...",
                progress=60.0,
                training_total_epochs=int(epochs),
            )

            from models.trainer import Trainer
            trainer = Trainer()

            def train_cb(model_name, epoch_idx, val_acc):
                if self._cancel_token.is_cancelled:
                    raise CancelledException()
                self.progress.training_epoch = int(epoch_idx) + 1
                self.progress.validation_accuracy = float(val_acc)
                self._update_progress(
                    message=f"Training {model_name}: epoch {epoch_idx+1}/{epochs}",
                    progress=60.0 + 35.0 * ((epoch_idx + 1) / max(1, epochs))
                )

            out = trainer.train(
                stock_codes=codes,
                epochs=int(epochs),
                callback=train_cb,
                stop_flag=self._cancel_token,
                save_model=True,
                incremental=bool(incremental),
                interval=str(interval).lower(),
                prediction_horizon=int(prediction_horizon),
                lookback_bars=int(lookback_bars),
            )

            acc = float(out.get("best_accuracy", 0.0))
            self.progress.training_accuracy = acc
            self._update_progress(
                stage="complete",
                message=f"Cycle complete. best_val_acc={acc:.1%}, prefetched={ok_cnt}/{len(codes)}",
                progress=100.0
            )

            # Update stats
            duration_h = (datetime.now() - start_time).total_seconds() / 3600.0
            self.progress.total_training_hours += duration_h
            self.progress.total_stocks_learned += len(codes)
            if acc > self.progress.best_accuracy_ever:
                self.progress.best_accuracy_ever = acc

            return True

        except CancelledException:
            raise
        except Exception as e:
            self._update_progress(stage="error", message=str(e))
            self.progress.errors.append(str(e))
            return False
    
    def _discover_all_stocks(
        self,
        max_stocks: int,
        min_market_cap: float,
        include_all_markets: bool
    ) -> List:
        """Discover all available stocks from multiple sources"""
        from data.discovery import UniversalStockDiscovery
        
        discovery = UniversalStockDiscovery()
        
        def search_callback(msg, count):
            self._update_progress(
                message=msg,
                stocks_found=count
            )
            
            if self._cancel_token.is_cancelled:
                raise CancelledException()
        
        stocks = discovery.discover_all(
            callback=search_callback,
            max_stocks=max_stocks,
            min_market_cap=min_market_cap,
            include_st=False
        )
        
        valid = [s for s in stocks if s.is_valid()]
        
        log.info(f"Discovered {len(valid)} valid stocks")
        return valid
    
    def _download_all_data(self, stocks: List) -> Dict[str, pd.DataFrame]:
        """Download data for all stocks in parallel"""
        from data.fetcher import DataFetcher
        from data.features import FeatureEngine
        
        fetcher = DataFetcher()
        feature_engine = FeatureEngine()
        
        codes = [s.code for s in stocks]
        
        def download_callback(code, completed, total):
            self.progress.stocks_processed = completed
            self._update_progress(
                message=f"Downloaded {completed}/{total}: {code}",
                progress=10 + (completed / total) * 30
            )
            
            if self._cancel_token.is_cancelled:
                raise CancelledException()
        
        raw_data = fetcher.get_multiple_parallel(
            codes,
            days=1500,
            callback=download_callback,
            max_workers=CONFIG.data.parallel_downloads
        )
        
        valid_data = {}
        seq_len = CONFIG.model.sequence_length
        
        for code, df in raw_data.items():
            if self._cancel_token.is_cancelled:
                raise CancelledException()
            
            try:
                df = feature_engine.create_features(df)
                if len(df) >= seq_len + 50:
                    valid_data[code] = df
                    self._processed_stocks.add(code)
            except Exception as e:
                log.debug(f"Feature creation failed for {code}: {e}")
                self._failed_stocks.add(code)
        
        log.info(f"Processed {len(valid_data)} stocks with features")
        return valid_data
    
    def _prepare_training_data(self, data: Dict[str, pd.DataFrame]):
        """Prepare training data with proper splits"""
        from data.features import FeatureEngine
        from data.processor import DataProcessor
        
        feature_engine = FeatureEngine()
        processor = DataProcessor()
        feature_cols = feature_engine.get_feature_columns()
        
        horizon = CONFIG.model.prediction_horizon
        embargo = CONFIG.model.embargo_bars
        train_ratio = CONFIG.model.train_ratio
        val_ratio = CONFIG.model.val_ratio
        seq_len = CONFIG.model.sequence_length
        
        all_train_features = []
        split_data = {}
        
        for code, df in data.items():
            if self._cancel_token.is_cancelled:
                raise CancelledException()
            
            n = len(df)
            train_end = int(n * train_ratio) - horizon - embargo
            val_end = int(n * (train_ratio + val_ratio)) - horizon - embargo
            
            if train_end < seq_len + 20:
                continue
            
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[int(n * train_ratio):val_end].copy()
            test_df = df.iloc[int(n * (train_ratio + val_ratio)):].copy()
            
            train_df = processor.create_labels(train_df)
            val_df = processor.create_labels(val_df)
            test_df = processor.create_labels(test_df)
            
            split_data[code] = {'train': train_df, 'val': val_df, 'test': test_df}
            
            train_features = train_df[feature_cols].values
            valid_mask = ~train_df['label'].isna()
            if valid_mask.sum() > 0:
                all_train_features.append(train_features[valid_mask])
        
        if not all_train_features:
            self._update_progress(stage="error", message="No valid training data")
            return None
        
        combined_train = np.concatenate(all_train_features, axis=0)
        processor.fit_scaler(combined_train)
        log.info(f"Scaler fitted on {len(combined_train)} samples")
        
        all_train = {'X': [], 'y': []}
        all_val = {'X': [], 'y': []}
        all_test = {'X': [], 'y': []}
        
        for code, splits in split_data.items():
            if self._cancel_token.is_cancelled:
                raise CancelledException()
            
            for split_name, split_df, storage in [
                ('train', splits['train'], all_train),
                ('val', splits['val'], all_val),
                ('test', splits['test'], all_test)
            ]:
                if len(split_df) >= seq_len + 5:
                    X, y, _ = processor.prepare_sequences(split_df, feature_cols, fit_scaler=False)
                    if len(X) > 0:
                        storage['X'].append(X)
                        storage['y'].append(y)
        
        if not all_train['X']:
            self._update_progress(stage="error", message="No training sequences")
            return None
        
        processor.save_scaler()
        
        return {
            'X_train': np.concatenate(all_train['X']),
            'y_train': np.concatenate(all_train['y']),
            'X_val': np.concatenate(all_val['X']) if all_val['X'] else None,
            'y_val': np.concatenate(all_val['y']) if all_val['y'] else None,
            'X_test': np.concatenate(all_test['X']) if all_test['X'] else None,
            'y_test': np.concatenate(all_test['y']) if all_test['y'] else None,
            'processor': processor
        }
    
    def _train_model(self, train_data: Dict, epochs: int, mode: str) -> float:
        from models.ensemble import EnsembleModel

        X_train = train_data['X_train']
        y_train = train_data['y_train']
        X_val = train_data['X_val']
        y_val = train_data['y_val']
        X_test = train_data['X_test']
        y_test = train_data['y_test']

        input_size = X_train.shape[2]
        ensemble = EnsembleModel(input_size)

        if mode == self.MODE_INCREMENTAL:
            ensemble.load()

        if X_val is None or len(X_val) == 0:
            split = int(len(X_train) * 0.85)
            X_val, y_val = X_train[split:], y_train[split:]
            X_train, y_train = X_train[:split], y_train[:split]

        def train_callback(model_name, epoch, val_acc):
            if self._cancel_token.is_cancelled:
                raise CancelledException()
            self.progress.training_epoch = epoch + 1
            self.progress.validation_accuracy = float(val_acc)
            self._update_progress(
                message=f"Training {model_name}: Epoch {epoch+1}/{epochs}",
                progress=50 + (epoch + 1) / epochs * 40
            )

        ensemble.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            callback=train_callback,
            stop_flag=self._cancel_token
        )

        ensemble.calibrate(X_val, y_val)
        ensemble.save()

        if X_test is not None and len(X_test) > 0:
            preds = ensemble.predict_batch(X_test[:1000])
            pred_cls = np.array([p.predicted_class for p in preds])
            return float(np.mean(pred_cls == y_test[:len(pred_cls)]))

        return float(self.progress.validation_accuracy)
    
    def _save_state(self):
        """Save learner state for resume"""
        state = {
            'total_sessions': self.progress.total_training_sessions,
            'total_stocks': self.progress.total_stocks_learned,
            'total_hours': self.progress.total_training_hours,
            'best_accuracy': self.progress.best_accuracy_ever,
            'processed_stocks': list(self._processed_stocks),
            'failed_stocks': list(self._failed_stocks),
            'last_save': datetime.now().isoformat()
        }
        
        try:
            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load previous learner state"""
        if not self.state_path.exists():
            return
        
        try:
            with open(self.state_path, 'r') as f:
                state = json.load(f)
            
            self.progress.total_training_sessions = state.get('total_sessions', 0)
            self.progress.total_stocks_learned = state.get('total_stocks', 0)
            self.progress.total_training_hours = state.get('total_hours', 0.0)
            self.progress.best_accuracy_ever = state.get('best_accuracy', 0.0)
            self._processed_stocks = set(state.get('processed_stocks', []))
            self._failed_stocks = set(state.get('failed_stocks', []))
            
            log.info(f"Loaded learner state: {self.progress.total_training_sessions} sessions")
        except Exception as e:
            log.warning(f"Failed to load state: {e}")
    
    def get_stats(self) -> Dict:
        """Get learning statistics"""
        return {
            'is_running': self.progress.is_running,
            'is_paused': self.progress.is_paused,
            'current_stage': self.progress.stage,
            'progress': self.progress.progress,
            'total_sessions': self.progress.total_training_sessions,
            'total_stocks': self.progress.total_stocks_learned,
            'total_hours': self.progress.total_training_hours,
            'best_accuracy': self.progress.best_accuracy_ever,
            'current_accuracy': self.progress.training_accuracy,
            'errors': self.progress.errors[-10:],
        }


# Backward compatibility alias
AutoLearner = ContinuousLearner