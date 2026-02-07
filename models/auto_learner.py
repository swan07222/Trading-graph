# models/auto_learner.py
"""
Continuous Auto-Learning System

Automatically discovers stocks, fetches real-time dealing data,
and trains AI models for graph prediction.
"""
import os
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Set, Any
from dataclasses import dataclass, field
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from config.settings import CONFIG
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
    current_interval: str = "1d"
    current_horizon: int = 5
    
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
            'training_total_epochs': self.training_total_epochs,
            'training_accuracy': self.training_accuracy,
            'validation_accuracy': self.validation_accuracy,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'total_sessions': self.total_training_sessions,
            'best_accuracy': self.best_accuracy_ever,
            'interval': self.current_interval,
            'horizon': self.current_horizon,
        }


class ContinuousLearner:
    """
    Continuous learning system that:
    1. Discovers liquid stocks from internet sources
    2. Fetches real-time dealing/intraday data
    3. Trains AI models incrementally
    4. Runs continuously until stopped
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
        
        self.history_path = CONFIG.DATA_DIR / "continuous_learning_history.json"
        self.state_path = CONFIG.DATA_DIR / "learner_state.json"
        self._load_state()
    
    def add_callback(self, callback: Callable[[LearningProgress], None]):
        """Add progress callback"""
        with self._lock:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[LearningProgress], None]):
        """Remove progress callback"""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def _notify(self):
        """Notify all callbacks of progress update"""
        self.progress.last_update = datetime.now()
        with self._lock:
            callbacks = self._callbacks.copy()
        
        for cb in callbacks:
            try:
                cb(self.progress)
            except Exception as e:
                log.debug(f"Callback error: {e}")
    
    def _update_progress(self, stage: str = None, message: str = None, 
                         progress: float = None, **kwargs):
        """Update progress and notify callbacks"""
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
    
    def _should_stop(self) -> bool:
        """Check if learning should stop"""
        return self._cancel_token.is_cancelled
    
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
        cycle_interval_seconds: int = 900,
        incremental: bool = True,
    ):
        """
        Start continuous auto-learning.
        
        Args:
            mode: Learning mode (full, incremental, online)
            max_stocks: Maximum stocks to process per cycle
            epochs_per_cycle: Training epochs per cycle
            min_market_cap: Minimum market cap filter (billions)
            include_all_markets: Include all market types
            continuous: Run continuously (loop)
            learning_while_trading: Allow learning during trading hours
            interval: Data interval (1m, 5m, 15m, 30m, 60m, 1d)
            prediction_horizon: Bars ahead to predict
            lookback_bars: Historical bars to fetch
            cycle_interval_seconds: Seconds between learning cycles
            incremental: Continue from previous training
        """
        if self._thread and self._thread.is_alive():
            if self.progress.is_paused:
                self.resume()
                return
            log.warning("Learning already in progress")
            return

        # Reset cancel token
        self._cancel_token = CancellationToken()
        self.progress.reset()
        self.progress.is_running = True
        self.progress.current_interval = str(interval)
        self.progress.current_horizon = int(prediction_horizon)

        self._thread = threading.Thread(
            target=self._continuous_learning_loop,
            args=(
                mode,
                max_stocks or 200,
                max(1, int(epochs_per_cycle)),
                float(min_market_cap),
                bool(include_all_markets),
                bool(continuous),
                bool(learning_while_trading),
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
            f"Continuous learning started: interval={interval}, "
            f"horizon={prediction_horizon}, lookback={lookback_bars}, "
            f"cycle={cycle_interval_seconds}s, incremental={incremental}"
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
        interval: str = "1d",
        prediction_horizon: int = 5,
        lookback_bars: int = 1500,
        **kwargs
    ):
        """Synchronous run for CLI usage"""
        self.start(
            mode=mode,
            max_stocks=max_stocks or 500,
            epochs_per_cycle=epochs_per_cycle,
            min_market_cap=min_market_cap,
            include_all_markets=include_all_markets,
            continuous=continuous,
            learning_while_trading=learning_while_trading,
            interval=interval,
            prediction_horizon=prediction_horizon,
            lookback_bars=lookback_bars,
        )
        
        if self._thread:
            self._thread.join()
        
        return self.progress
    
    def stop(self):
        """Stop learning gracefully"""
        log.info("Stopping continuous learning...")
        self._cancel_token.cancel()
        
        if self._thread:
            self._thread.join(timeout=30)
        
        self._save_state()
        self.progress.is_running = False
        self._notify()
        log.info("Continuous learning stopped")
    
    def pause(self):
        """Pause learning"""
        self.progress.is_paused = True
        self._notify()
        log.info("Learning paused")
    
    def resume(self):
        """Resume paused learning"""
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
        """Main continuous learning loop"""
        cycle = 0

        try:
            while not self._should_stop():
                cycle += 1
                self._update_progress(
                    stage="cycle_start",
                    message=f"=== Learning Cycle {cycle} ({interval} interval) ===",
                    progress=0.0
                )

                # Check if paused
                while self.progress.is_paused and not self._should_stop():
                    time.sleep(1)

                if self._should_stop():
                    break

                # Run one learning cycle
                success = self._run_learning_cycle(
                    mode=mode,
                    max_stocks=max_stocks,
                    epochs=epochs_per_cycle,
                    min_market_cap=min_market_cap,
                    include_all_markets=include_all_markets,
                    interval=interval,
                    prediction_horizon=prediction_horizon,
                    lookback_bars=lookback_bars,
                    incremental=incremental,
                )

                if success:
                    self.progress.total_training_sessions += 1
                    self._save_state()

                if not continuous:
                    break

                # Wait between cycles
                self._update_progress(
                    stage="waiting",
                    message=f"Waiting {cycle_interval_seconds}s until next cycle...",
                    progress=100.0
                )

                for _ in range(cycle_interval_seconds):
                    if self._should_stop():
                        break
                    time.sleep(1)

        except CancelledException:
            log.info("Learning cancelled by user")
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
        """
        Run one complete learning cycle:
        1. Discover liquid stocks from internet
        2. Fetch dealing/intraday data
        3. Train AI models
        """
        start_time = datetime.now()

        try:
            # === Step 1: Discover stocks ===
            self._update_progress(
                stage="discovering",
                message="Discovering liquid stocks from internet...",
                progress=5.0
            )

            codes = self._discover_stocks(
                max_stocks=max_stocks,
                min_market_cap=min_market_cap,
            )

            if not codes:
                self._update_progress(
                    stage="error", 
                    message="No stocks discovered"
                )
                return False

            self.progress.stocks_found = len(codes)
            self.progress.stocks_total = len(codes)

            # === Step 2: Prefetch data ===
            self._update_progress(
                stage="downloading",
                message=f"Fetching {interval} data for {len(codes)} stocks...",
                progress=20.0
            )

            fetcher = get_fetcher()
            ok_count = 0

            for i, code in enumerate(codes, start=1):
                if self._should_stop():
                    raise CancelledException()

                try:
                    df = fetcher.get_history(
                        code,
                        interval=interval,
                        bars=lookback_bars,
                        days=lookback_bars,
                        use_cache=True,
                        update_db=True,
                    )
                    if df is not None and not df.empty and len(df) >= 100:
                        ok_count += 1
                except Exception as e:
                    log.debug(f"Failed to fetch {code}: {e}")

                self.progress.stocks_processed = i
                pct = 20.0 + 35.0 * (i / len(codes))
                self._update_progress(
                    message=f"Fetched {i}/{len(codes)}: {code}",
                    progress=pct
                )

            if ok_count < max(5, int(0.1 * len(codes))):
                self._update_progress(
                    stage="error", 
                    message=f"Too few stocks with data: {ok_count}"
                )
                return False

            log.info(f"Successfully fetched data for {ok_count}/{len(codes)} stocks")

            # === Step 3: Train models ===
            self._update_progress(
                stage="training",
                message=f"Training AI models (epochs={epochs})...",
                progress=60.0,
                training_total_epochs=epochs,
            )

            from models.trainer import Trainer
            trainer = Trainer()

            def train_callback(model_name, epoch_idx, val_acc):
                if self._should_stop():
                    raise CancelledException()
                    
                self.progress.training_epoch = epoch_idx + 1
                self.progress.validation_accuracy = float(val_acc)
                self._update_progress(
                    message=f"Training {model_name}: epoch {epoch_idx+1}/{epochs}",
                    progress=60.0 + 35.0 * ((epoch_idx + 1) / max(1, epochs))
                )

            result = trainer.train(
                stock_codes=codes,
                epochs=epochs,
                callback=train_callback,
                stop_flag=self._cancel_token,
                save_model=True,
                incremental=incremental,
                interval=interval,
                prediction_horizon=prediction_horizon,
                lookback_bars=lookback_bars,
            )

            if result.get("status") == "cancelled":
                raise CancelledException()

            acc = float(result.get("best_accuracy", 0.0))
            self.progress.training_accuracy = acc

            # Update stats
            duration_hours = (datetime.now() - start_time).total_seconds() / 3600.0
            self.progress.total_training_hours += duration_hours
            self.progress.total_stocks_learned += len(codes)
            
            if acc > self.progress.best_accuracy_ever:
                self.progress.best_accuracy_ever = acc

            self._update_progress(
                stage="complete",
                message=f"Cycle complete! Best accuracy: {acc:.1%}, "
                       f"Stocks: {ok_count}/{len(codes)}",
                progress=100.0
            )

            return True

        except CancelledException:
            raise
        except Exception as e:
            log.error(f"Learning cycle error: {e}")
            import traceback
            traceback.print_exc()
            self._update_progress(stage="error", message=str(e))
            self.progress.errors.append(str(e))
            return False
    
    def _discover_stocks(
        self, 
        max_stocks: int, 
        min_market_cap: float
    ) -> List[str]:
        """
        Discover stocks from internet sources.
        Uses UniversalStockDiscovery for comprehensive search.
        """
        try:
            from data.discovery import UniversalStockDiscovery
            
            discovery = UniversalStockDiscovery()
            
            def search_callback(msg, count):
                if self._should_stop():
                    raise CancelledException()
                self._update_progress(
                    message=msg,
                    stocks_found=count
                )
            
            stocks = discovery.discover_all(
                callback=search_callback,
                max_stocks=max_stocks,
                min_market_cap=min_market_cap,
                include_st=False
            )
            
            if stocks:
                codes = [s.code for s in stocks if s.is_valid()]
                log.info(f"Discovered {len(codes)} valid stocks")
                return codes
            
        except CancelledException:
            raise
        except Exception as e:
            log.warning(f"Stock discovery failed: {e}")
        
        # Fallback to CONFIG stock pool
        log.info("Using fallback stock pool")
        return CONFIG.STOCK_POOL[:max_stocks]
    
    def _save_state(self):
        """Save learner state for resume"""
        state = {
            'total_sessions': self.progress.total_training_sessions,
            'total_stocks': self.progress.total_stocks_learned,
            'total_hours': self.progress.total_training_hours,
            'best_accuracy': self.progress.best_accuracy_ever,
            'processed_stocks': list(self._processed_stocks),
            'failed_stocks': list(self._failed_stocks),
            'last_interval': self.progress.current_interval,
            'last_horizon': self.progress.current_horizon,
            'last_save': datetime.now().isoformat()
        }
        
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
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
            self.progress.current_interval = state.get('last_interval', '1d')
            self.progress.current_horizon = state.get('last_horizon', 5)
            self._processed_stocks = set(state.get('processed_stocks', []))
            self._failed_stocks = set(state.get('failed_stocks', []))
            
            log.info(f"Loaded learner state: {self.progress.total_training_sessions} sessions, "
                    f"best accuracy: {self.progress.best_accuracy_ever:.1%}")
        except Exception as e:
            log.warning(f"Failed to load state: {e}")
    
    def get_stats(self) -> Dict:
        """Get learning statistics"""
        return {
            'is_running': self.progress.is_running,
            'is_paused': self.progress.is_paused,
            'current_stage': self.progress.stage,
            'progress': self.progress.progress,
            'message': self.progress.message,
            'total_sessions': self.progress.total_training_sessions,
            'total_stocks': self.progress.total_stocks_learned,
            'total_hours': self.progress.total_training_hours,
            'best_accuracy': self.progress.best_accuracy_ever,
            'current_accuracy': self.progress.training_accuracy,
            'validation_accuracy': self.progress.validation_accuracy,
            'current_interval': self.progress.current_interval,
            'current_horizon': self.progress.current_horizon,
            'errors': self.progress.errors[-10:],
        }


# Backward compatibility alias
AutoLearner = ContinuousLearner