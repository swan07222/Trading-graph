"""
Auto-Learning System - Continuous Learning with Proper Leakage Prevention

FEATURES:
- Continuous background learning
- Discovery caching (24h)
- Proper embargo-based splits
- True incremental learning (loads existing model)
- Fallback to cached data when offline

Author: AI Trading System v3.0
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


@dataclass
class StockInfo:
    """Discovered stock information"""
    code: str
    name: str
    source: str
    score: float
    discovered_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        self.code = self._clean_code(self.code)
    
    def _clean_code(self, code: str) -> str:
        if not code:
            return ""
        code = str(code).strip()
        for prefix in ['sh', 'sz', 'SH', 'SZ']:
            code = code.replace(prefix, '')
        code = code.replace('.', '').replace('-', '')
        if code.isdigit():
            return code.zfill(6)
        return ""
    
    def is_valid(self) -> bool:
        if not self.code or len(self.code) != 6 or not self.code.isdigit():
            return False
        valid_prefixes = ['60', '00', '30', '68']
        return any(self.code.startswith(p) for p in valid_prefixes)


class DiscoveryCache:
    """Cache for discovered stocks to avoid repeated API calls"""
    
    def __init__(self):
        self.cache_file = CONFIG.CACHE_DIR / "discovery_cache.json"
        self._cache: Dict = {}
        self._load()
    
    def _load(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check if cache is still valid
                    cached_time = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
                    if datetime.now() - cached_time < timedelta(hours=CONFIG.DISCOVERY_CACHE_HOURS):
                        self._cache = data
                        log.info(f"Loaded discovery cache with {len(data.get('stocks', []))} stocks")
            except Exception as e:
                log.warning(f"Failed to load discovery cache: {e}")
    
    def _save(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            log.warning(f"Failed to save discovery cache: {e}")
    
    def get_cached_stocks(self) -> List[StockInfo]:
        """Get cached stocks if still valid"""
        if not self._cache:
            return []
        
        stocks = []
        for s in self._cache.get('stocks', []):
            stock = StockInfo(
                code=s['code'],
                name=s['name'],
                source=s['source'],
                score=s['score']
            )
            if stock.is_valid():
                stocks.append(stock)
        return stocks
    
    def update_cache(self, stocks: List[StockInfo]):
        """Update cache with new stocks"""
        self._cache = {
            'timestamp': datetime.now().isoformat(),
            'stocks': [
                {'code': s.code, 'name': s.name, 'source': s.source, 'score': s.score}
                for s in stocks if s.is_valid()
            ]
        }
        self._save()


class StockDiscovery:
    """Discover stocks from various sources with caching"""
    
    def __init__(self):
        self.cache = DiscoveryCache()
        self._rate_limit = 2.0
        self._last_request = 0
        
        try:
            import akshare as ak
            self._ak = ak
        except ImportError:
            self._ak = None
    
    def _wait(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_request = time.time()
    
    def discover(self, use_cache: bool = True, callback: Callable = None) -> List[StockInfo]:
        """
        Discover stocks from internet or cache.
        Falls back to CONFIG.STOCK_POOL if all else fails.
        """
        # Try cache first
        if use_cache:
            cached = self.cache.get_cached_stocks()
            if len(cached) >= CONFIG.MIN_STOCKS_FOR_TRAINING:
                log.info(f"Using {len(cached)} cached stocks")
                return cached
        
        # Try online discovery
        all_stocks = []
        
        if self._ak:
            sources = [
                ("涨幅榜", self._find_gainers, 50),
                ("跌幅榜", self._find_losers, 30),
                ("成交额榜", self._find_volume, 50),
            ]
            
            for name, finder, limit in sources:
                if callback:
                    callback(f"搜索 {name}...", len(all_stocks))
                
                try:
                    self._wait()
                    stocks = finder(limit)
                    valid = [s for s in stocks if s.is_valid()]
                    all_stocks.extend(valid)
                    log.info(f"Found {len(valid)} from {name}")
                except Exception as e:
                    log.warning(f"Failed to search {name}: {e}")
        
        # Deduplicate
        unique = {}
        for stock in all_stocks:
            if stock.code not in unique or stock.score > unique[stock.code].score:
                unique[stock.code] = stock
        
        result = sorted(unique.values(), key=lambda x: x.score, reverse=True)
        
        # Update cache if we found stocks
        if len(result) >= 5:
            self.cache.update_cache(result)
        
        # Fallback to default pool
        if len(result) < CONFIG.MIN_STOCKS_FOR_TRAINING:
            log.warning("Insufficient stocks discovered, using default pool")
            for code in CONFIG.STOCK_POOL:
                if code not in unique:
                    result.append(StockInfo(
                        code=code,
                        name=f"Stock {code}",
                        source="config",
                        score=0.5
                    ))
        
        log.info(f"Total stocks for training: {len(result)}")
        return result
    
    def _find_gainers(self, limit: int) -> List[StockInfo]:
        df = self._ak.stock_zh_a_spot_em()
        df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
        df = df.dropna(subset=['涨跌幅'])
        df = df.sort_values('涨跌幅', ascending=False).head(limit)
        
        return [
            StockInfo(
                code=str(row['代码']),
                name=str(row.get('名称', '')),
                source="涨幅榜",
                score=min(abs(float(row['涨跌幅'])) / 10, 1.0)
            )
            for _, row in df.iterrows()
        ]
    
    def _find_losers(self, limit: int) -> List[StockInfo]:
        df = self._ak.stock_zh_a_spot_em()
        df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
        df = df.dropna(subset=['涨跌幅'])
        df = df.sort_values('涨跌幅', ascending=True).head(limit)
        
        return [
            StockInfo(
                code=str(row['代码']),
                name=str(row.get('名称', '')),
                source="跌幅榜",
                score=min(abs(float(row['涨跌幅'])) / 10, 0.8)
            )
            for _, row in df.iterrows()
        ]
    
    def _find_volume(self, limit: int) -> List[StockInfo]:
        df = self._ak.stock_zh_a_spot_em()
        df['成交额'] = pd.to_numeric(df['成交额'], errors='coerce')
        df = df.dropna(subset=['成交额'])
        df = df.sort_values('成交额', ascending=False).head(limit)
        
        return [
            StockInfo(
                code=str(row['代码']),
                name=str(row.get('名称', '')),
                source="成交额榜",
                score=min(float(row['成交额']) / 1e10, 1.0)
            )
            for _, row in df.iterrows()
        ]


class AutoLearner:
    """
    Automatic Learning System with Continuous Operation
    
    Features:
    - Background continuous learning
    - Proper leakage-free splits (uses DataProcessor.split_temporal_single_stock)
    - True incremental learning (loads existing model)
    - Discovery caching
    - Progress callbacks
    """
    
    def __init__(self):
        self.discovery = StockDiscovery()
        self.progress = LearningProgress()
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        self._continuous = False
        
        # History tracking
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
        max_stocks: int = 80,
        epochs: int = None,
        incremental: bool = True,
        continuous: bool = False
    ):
        """
        Start learning process.
        
        Args:
            auto_search: Search internet for stocks
            max_stocks: Maximum stocks to train on
            epochs: Training epochs
            incremental: Load and continue from existing model
            continuous: Keep running and retraining periodically
        """
        if self._thread and self._thread.is_alive():
            log.warning("Learning already in progress")
            return
        
        self._stop_flag = False
        self._continuous = continuous
        self.progress.reset()
        
        epochs = epochs or CONFIG.AUTO_LEARN_EPOCHS
        
        self._thread = threading.Thread(
            target=self._learning_loop,
            args=(auto_search, max_stocks, epochs, incremental),
            daemon=True
        )
        self._thread.start()
    
    def stop_learning(self):
        self._stop_flag = True
        self._continuous = False
        if self._thread:
            self._thread.join(timeout=10)
    
    def _learning_loop(self, auto_search: bool, max_stocks: int, epochs: int, incremental: bool):
        """Main learning loop - can run continuously"""
        while not self._stop_flag:
            session_start = datetime.now()
            
            try:
                self.progress.is_running = True
                self.progress.errors = []
                
                success = self._run_single_session(auto_search, max_stocks, epochs, incremental)
                
                if success:
                    # Record session
                    duration = (datetime.now() - session_start).total_seconds() / 60
                    self.history['sessions'].append({
                        'timestamp': session_start.isoformat(),
                        'duration_minutes': duration,
                        'accuracy': self.progress.training_accuracy,
                        'stocks_used': self.progress.stocks_processed,
                        'epochs': epochs
                    })
                    self.history['best_accuracy'] = max(
                        self.history.get('best_accuracy', 0),
                        self.progress.training_accuracy
                    )
                    self._save_history()
                
            except Exception as e:
                import traceback
                log.error(f"Learning session failed: {e}")
                traceback.print_exc()
                self.progress.errors.append(str(e))
            
            finally:
                self.progress.is_running = False
                self._notify()
            
            # If continuous mode, wait and repeat
            if self._continuous and not self._stop_flag:
                self.progress.stage = "waiting"
                self.progress.message = f"Next session in {CONFIG.AUTO_LEARN_INTERVAL_HOURS} hours"
                self._notify()
                
                # Wait in small increments to allow stopping
                wait_seconds = CONFIG.AUTO_LEARN_INTERVAL_HOURS * 3600
                for _ in range(int(wait_seconds / 10)):
                    if self._stop_flag:
                        break
                    time.sleep(10)
            else:
                break
    
    def _run_single_session(self, auto_search: bool, max_stocks: int, epochs: int, incremental: bool) -> bool:
        """Run a single training session with proper leakage prevention"""
        
        # === Stage 1: Discover Stocks ===
        self.progress.stage = "searching"
        self.progress.message = "搜索股票..."
        self.progress.progress = 0
        self._notify()
        
        if auto_search:
            def search_cb(msg, count):
                self.progress.message = msg
                self.progress.stocks_found = count
                self._notify()
            
            stocks = self.discovery.discover(use_cache=True, callback=search_cb)
        else:
            stocks = [
                StockInfo(code=c, name=f"Stock {c}", source="config", score=1.0)
                for c in CONFIG.STOCK_POOL
            ]
        
        self.progress.stocks_found = len(stocks)
        selected_codes = [s.code for s in stocks[:max_stocks] if s.is_valid()]
        
        if len(selected_codes) < CONFIG.MIN_STOCKS_FOR_TRAINING:
            self.progress.stage = "error"
            self.progress.message = f"Insufficient stocks: {len(selected_codes)}"
            return False
        
        if self._stop_flag:
            return False
        
        # === Stage 2: Download Data ===
        self.progress.stage = "downloading"
        self.progress.message = "下载数据..."
        self.progress.progress = 20
        self._notify()
        
        from data.fetcher import DataFetcher
        from data.features import FeatureEngine
        from data.processor import DataProcessor
        
        fetcher = DataFetcher()
        feature_engine = FeatureEngine()
        processor = DataProcessor()
        
        valid_data = {}
        
        for i, code in enumerate(selected_codes):
            if self._stop_flag:
                return False
            
            self.progress.stocks_processed = i + 1
            self.progress.message = f"下载 {code} ({i+1}/{len(selected_codes)})"
            self.progress.progress = 20 + (i + 1) / len(selected_codes) * 20
            self._notify()
            
            try:
                df = fetcher.get_history(code, days=1500, use_cache=True)
                
                if df is not None and len(df) >= 200:
                    df = feature_engine.create_features(df)
                    if len(df) >= CONFIG.SEQUENCE_LENGTH + 50:
                        valid_data[code] = df
                        log.debug(f"Loaded {code}: {len(df)} bars")
                        
            except Exception as e:
                log.warning(f"Failed to load {code}: {e}")
        
        if len(valid_data) < 5:
            self.progress.stage = "error"
            self.progress.message = f"Insufficient valid data: {len(valid_data)} stocks"
            return False
        
        # === Stage 3: Prepare Data with Proper Splits ===
        self.progress.stage = "preparing"
        self.progress.message = "准备训练数据..."
        self.progress.progress = 45
        self._notify()
        
        feature_cols = feature_engine.get_feature_columns()
        
        all_train = {'X': [], 'y': []}
        all_val = {'X': [], 'y': []}
        all_test = {'X': [], 'y': []}
        
        first_stock = True
        
        for code, df in valid_data.items():
            if self._stop_flag:
                return False
            
            try:
                # Use PROPER split that handles embargo correctly
                splits = processor.split_temporal_single_stock(
                    df, feature_cols, 
                    fit_scaler_on_train=first_stock  # Fit scaler only on first stock's train data
                )
                first_stock = False
                
                for split_name, storage in [('train', all_train), ('val', all_val), ('test', all_test)]:
                    X, y, _ = splits[split_name]
                    if len(X) > 0:
                        storage['X'].append(X)
                        storage['y'].append(y)
                        
            except Exception as e:
                log.warning(f"Failed to process {code}: {e}")
        
        if not all_train['X']:
            self.progress.stage = "error"
            self.progress.message = "No training data available"
            return False
        
        X_train = np.concatenate(all_train['X'])
        y_train = np.concatenate(all_train['y'])
        X_val = np.concatenate(all_val['X']) if all_val['X'] else X_train[-100:]
        y_val = np.concatenate(all_val['y']) if all_val['y'] else y_train[-100:]
        X_test = np.concatenate(all_test['X']) if all_test['X'] else np.array([])
        y_test = np.concatenate(all_test['y']) if all_test['y'] else np.array([])
        
        log.info(f"Data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Save scaler
        processor.save_scaler()
        
        # === Stage 4: Train Model ===
        self.progress.stage = "training"
        self.progress.message = "训练AI模型..."
        self.progress.progress = 50
        self._notify()
        
        from models.ensemble import EnsembleModel
        
        input_size = X_train.shape[2]
        ensemble = EnsembleModel(input_size)
        
        # TRUE incremental: load existing model if available
        if incremental:
            if ensemble.load():
                log.info("Loaded existing model for incremental learning")
            else:
                log.info("No existing model, training from scratch")
        
        def train_callback(model_name, epoch, val_acc):
            if self._stop_flag:
                return
            self.progress.training_epoch = epoch + 1
            self.progress.training_accuracy = val_acc
            self.progress.message = f"训练 {model_name}: Epoch {epoch+1}/{epochs}"
            self.progress.progress = 50 + (epoch + 1) / epochs * 40
            self._notify()
        
        ensemble.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            callback=train_callback,
            stop_flag=lambda: self._stop_flag
        )
        
        if self._stop_flag:
            return False
        
        # === Stage 5: Evaluate ===
        self.progress.stage = "evaluating"
        self.progress.message = "评估模型..."
        self.progress.progress = 95
        self._notify()
        
        if len(X_test) > 0:
            correct = 0
            total = min(len(X_test), 500)
            
            for i in range(total):
                pred = ensemble.predict(X_test[i:i+1])
                if pred.predicted_class == y_test[i]:
                    correct += 1
            
            test_accuracy = correct / total
            log.info(f"Test accuracy: {test_accuracy:.2%}")
            self.progress.training_accuracy = test_accuracy
        
        # Save model
        ensemble.save()
        
        # === Stage 6: Complete ===
        self.progress.stage = "complete"
        self.progress.message = f"完成！准确率: {self.progress.training_accuracy:.1%}"
        self.progress.progress = 100
        self._notify()
        
        return True
    
    def get_learning_stats(self) -> Dict:
        return {
            'sessions_count': len(self.history.get('sessions', [])),
            'best_accuracy': self.history.get('best_accuracy', 0),
            'total_stocks': self.history.get('total_stocks', 0),
            'last_update': self.history.get('last_update'),
            'is_running': self.progress.is_running,
            'current_stage': self.progress.stage
        }