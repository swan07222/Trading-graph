"""
Auto-Learning System - Automatic Stock Discovery and Training

FIXED Issues:
- Robust stock discovery with multiple sources
- Proper error handling and fallback
- Better stock code extraction
- Fallback to CONFIG.STOCK_POOL when internet fails

Author: AI Trading System
Version: 2.0
"""
import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import requests

from config import CONFIG
from utils.logger import log

# Try importing akshare
try:
    import akshare as ak
    AKSHARE_OK = True
except ImportError:
    AKSHARE_OK = False
    log.warning("akshare not installed - some features limited")


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
        """Reset progress"""
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
    """Stock information from discovery"""
    code: str
    name: str
    source: str
    reason: str
    score: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        # Clean and validate code
        self.code = self._clean_code(self.code)
    
    def _clean_code(self, code: str) -> str:
        """Clean stock code"""
        if not code:
            return ""
        code = str(code).strip()
        # Remove exchange prefixes/suffixes
        for prefix in ['sh', 'sz', 'SH', 'SZ']:
            code = code.replace(prefix, '')
        code = code.replace('.', '').replace('-', '')
        # Pad to 6 digits
        if code.isdigit():
            return code.zfill(6)
        return ""
    
    def is_valid(self) -> bool:
        """Check if stock code is valid"""
        if not self.code or len(self.code) != 6:
            return False
        if not self.code.isdigit():
            return False
        # Valid A-share prefixes
        valid_prefixes = ['60', '00', '30', '68']  # SH, SZ main, ChiNext, STAR
        return any(self.code.startswith(p) for p in valid_prefixes)


class InternetStockFinder:
    """
    Find stocks from various internet sources
    
    Sources:
    1. Top gainers (涨幅榜)
    2. Top losers (跌幅榜)
    3. High volume (成交额榜)
    4. Trending/Hot stocks
    5. Analyst recommendations
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self._rate_limit = 2.0  # seconds between requests
        self._last_request = 0
        self._timeout = 30  # seconds
    
    def _wait(self):
        """Rate limiting"""
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_request = time.time()
    
    def find_all_stocks(self, callback: Callable = None) -> List[StockInfo]:
        """
        Find stocks from all sources with fallback
        
        Args:
            callback: Progress callback(message, progress_pct)
            
        Returns:
            List of StockInfo objects
        """
        all_stocks = []
        
        sources = [
            ("涨幅榜 (Top Gainers)", self.find_top_gainers, 50),
            ("跌幅榜 (Top Losers)", self.find_top_losers, 30),
            ("成交额榜 (High Volume)", self.find_top_volume, 50),
            ("机构推荐 (Analyst Picks)", self.find_analyst_picks, 30),
        ]
        
        total_sources = len(sources)
        
        for i, (name, finder, limit) in enumerate(sources):
            progress = ((i + 1) / total_sources) * 100
            
            if callback:
                callback(f"搜索 {name}...", progress)
            
            try:
                self._wait()
                stocks = finder(limit=limit)
                
                # Filter valid stocks
                valid_stocks = [s for s in stocks if s.is_valid()]
                all_stocks.extend(valid_stocks)
                
                log.info(f"Found {len(valid_stocks)} valid stocks from {name}")
                
            except Exception as e:
                log.warning(f"Failed to search {name}: {e}")
        
        # Remove duplicates, keep highest score
        unique = {}
        for stock in all_stocks:
            if stock.code not in unique or stock.score > unique[stock.code].score:
                unique[stock.code] = stock
        
        result = sorted(unique.values(), key=lambda x: x.score, reverse=True)
        
        log.info(f"Total unique valid stocks found: {len(result)}")
        
        return result
    
    def find_top_gainers(self, limit: int = 50) -> List[StockInfo]:
        """Find top gaining stocks"""
        if not AKSHARE_OK:
            return []
        
        try:
            df = ak.stock_zh_a_spot_em()
            
            if df is None or df.empty:
                return []
            
            # Sort by change percent
            df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
            df = df.dropna(subset=['涨跌幅'])
            df = df.sort_values('涨跌幅', ascending=False).head(limit)
            
            stocks = []
            for _, row in df.iterrows():
                code = str(row.get('代码', '')).strip()
                change = float(row.get('涨跌幅', 0))
                
                stocks.append(StockInfo(
                    code=code,
                    name=str(row.get('名称', '')),
                    source="涨幅榜",
                    reason=f"今日涨幅 {change:+.2f}%",
                    score=min(abs(change) / 10, 1.0)
                ))
            
            return stocks
            
        except Exception as e:
            log.error(f"find_top_gainers error: {e}")
            return []
    
    def find_top_losers(self, limit: int = 30) -> List[StockInfo]:
        """Find top losing stocks (potential rebounds)"""
        if not AKSHARE_OK:
            return []
        
        try:
            df = ak.stock_zh_a_spot_em()
            
            if df is None or df.empty:
                return []
            
            df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
            df = df.dropna(subset=['涨跌幅'])
            df = df.sort_values('涨跌幅', ascending=True).head(limit)
            
            stocks = []
            for _, row in df.iterrows():
                code = str(row.get('代码', '')).strip()
                change = float(row.get('涨跌幅', 0))
                
                stocks.append(StockInfo(
                    code=code,
                    name=str(row.get('名称', '')),
                    source="跌幅榜",
                    reason=f"今日跌幅 {change:.2f}% (反弹机会)",
                    score=min(abs(change) / 10, 0.8)
                ))
            
            return stocks
            
        except Exception as e:
            log.error(f"find_top_losers error: {e}")
            return []
    
    def find_top_volume(self, limit: int = 50) -> List[StockInfo]:
        """Find high volume stocks"""
        if not AKSHARE_OK:
            return []
        
        try:
            df = ak.stock_zh_a_spot_em()
            
            if df is None or df.empty:
                return []
            
            df['成交额'] = pd.to_numeric(df['成交额'], errors='coerce')
            df = df.dropna(subset=['成交额'])
            df = df.sort_values('成交额', ascending=False).head(limit)
            
            stocks = []
            for _, row in df.iterrows():
                code = str(row.get('代码', '')).strip()
                amount = float(row.get('成交额', 0))
                amount_b = amount / 1e8  # Convert to 亿
                
                stocks.append(StockInfo(
                    code=code,
                    name=str(row.get('名称', '')),
                    source="成交额榜",
                    reason=f"成交额 {amount_b:.1f}亿",
                    score=min(amount_b / 100, 1.0)
                ))
            
            return stocks
            
        except Exception as e:
            log.error(f"find_top_volume error: {e}")
            return []
    
    def find_analyst_picks(self, limit: int = 30) -> List[StockInfo]:
        """Find analyst recommended stocks"""
        if not AKSHARE_OK:
            return []
        
        try:
            df = ak.stock_rank_forecast_cninfo()
            
            if df is None or df.empty:
                return []
            
            stocks = []
            for _, row in df.head(limit * 2).iterrows():
                # Try different column names
                code = None
                for col in ['代码', '股票代码', 'code', 'symbol', 'CODE']:
                    if col in row.index:
                        code = str(row[col]).strip()
                        break
                
                if not code:
                    continue
                
                # Get name
                name = ''
                for col in ['名称', '股票名称', 'name', 'NAME']:
                    if col in row.index:
                        name = str(row[col])
                        break
                
                stocks.append(StockInfo(
                    code=code,
                    name=name,
                    source="机构推荐",
                    reason="分析师推荐",
                    score=0.8
                ))
            
            # Filter valid stocks
            return [s for s in stocks if s.is_valid()][:limit]
            
        except Exception as e:
            log.warning(f"find_analyst_picks error: {e}")
            return []
    
    def get_fallback_stocks(self) -> List[StockInfo]:
        """Get fallback stocks from config when internet fails"""
        log.info("Using fallback stock pool from config")
        
        return [
            StockInfo(
                code=code,
                name=f"Stock {code}",
                source="config",
                reason="Default stock pool",
                score=1.0
            )
            for code in CONFIG.STOCK_POOL
        ]


class AutoLearner:
    """
    Automatic Learning System
    
    Features:
    1. Auto-search internet for stocks
    2. Auto-download historical data
    3. Auto-train AI models
    4. Continuous learning support
    5. Performance tracking
    """
    
    def __init__(self):
        self.finder = InternetStockFinder()
        self.progress = LearningProgress()
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        
        # Learning history
        self.history_path = CONFIG.DATA_DIR / "learning_history.json"
        self.history = self._load_history()
    
    def add_callback(self, callback: Callable):
        """Add progress callback"""
        self._callbacks.append(callback)
    
    def _notify(self):
        """Notify all callbacks"""
        for cb in self._callbacks:
            try:
                cb(self.progress)
            except Exception as e:
                log.warning(f"Callback error: {e}")
    
    def _load_history(self) -> Dict:
        """Load learning history"""
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Failed to load history: {e}")
        
        return {
            'sessions': [],
            'best_accuracy': 0,
            'total_stocks': 0,
            'last_update': None
        }
    
    def _save_history(self):
        """Save learning history"""
        try:
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            log.error(f"Failed to save history: {e}")
    
    def start_learning(self,
                       auto_search: bool = True,
                       max_stocks: int = 80,
                       epochs: int = 100,
                       incremental: bool = True):
        """
        Start auto-learning process
        
        Args:
            auto_search: Search internet for stocks
            max_stocks: Maximum stocks to include
            epochs: Training epochs
            incremental: Use incremental learning
        """
        if self._thread and self._thread.is_alive():
            log.warning("Learning already in progress")
            return
        
        self._stop_flag = False
        self.progress.reset()
        
        self._thread = threading.Thread(
            target=self._learning_loop,
            args=(auto_search, max_stocks, epochs, incremental),
            daemon=True
        )
        self._thread.start()
    
    def stop_learning(self):
        """Stop the learning process"""
        self._stop_flag = True
        if self._thread:
            self._thread.join(timeout=10)
    
    def _learning_loop(self, auto_search: bool, max_stocks: int,
                       epochs: int, incremental: bool):
        """Main learning loop"""
        session_start = datetime.now()
        
        try:
            self.progress.is_running = True
            self.progress.errors = []
            
            # =====================================
            # STAGE 1: Search for stocks
            # =====================================
            self.progress.stage = "searching"
            self.progress.message = "搜索股票..."
            self.progress.progress = 0
            self._notify()
            
            if auto_search:
                def search_callback(msg, prog):
                    self.progress.message = msg
                    self.progress.progress = prog * 0.2  # 0-20%
                    self._notify()
                
                stocks = self.finder.find_all_stocks(callback=search_callback)
            else:
                stocks = self.finder.get_fallback_stocks()
            
            # Fallback if no stocks found
            if len(stocks) < 5:
                log.warning("Insufficient stocks from internet, using fallback")
                stocks = self.finder.get_fallback_stocks()
            
            self.progress.stocks_found = len(stocks)
            
            if self._stop_flag:
                self._set_stopped()
                return
            
            # Select top stocks
            selected_codes = [s.code for s in stocks[:max_stocks]]
            log.info(f"Selected {len(selected_codes)} stocks for training")
            
            # =====================================
            # STAGE 2: Download data
            # =====================================
            self.progress.stage = "downloading"
            self.progress.message = "下载股票数据..."
            self._notify()
            
            from data.fetcher import DataFetcher
            fetcher = DataFetcher()
            
            valid_data = {}
            failed_codes = []
            
            for i, code in enumerate(selected_codes):
                if self._stop_flag:
                    self._set_stopped()
                    return
                
                self.progress.stocks_processed = i + 1
                self.progress.message = f"下载 {code} ({i+1}/{len(selected_codes)})"
                self.progress.progress = 20 + (i + 1) / len(selected_codes) * 30  # 20-50%
                self._notify()
                
                try:
                    df = fetcher.get_history(code, days=1500, use_cache=True)
                    
                    if df is not None and len(df) >= 200:
                        valid_data[code] = df
                        log.debug(f"Downloaded {code}: {len(df)} bars")
                    else:
                        failed_codes.append(code)
                        log.warning(f"Insufficient data for {code}")
                        
                except Exception as e:
                    failed_codes.append(code)
                    log.warning(f"Failed to download {code}: {e}")
            
            log.info(f"Valid stocks with data: {len(valid_data)}")
            
            if len(valid_data) < 5:
                self.progress.message = "数据不足，无法训练"
                self.progress.stage = "error"
                self.progress.errors.append("Insufficient data for training")
                self._notify()
                return
            
            # =====================================
            # STAGE 3: Prepare training data
            # =====================================
            self.progress.stage = "preparing"
            self.progress.message = "准备训练数据..."
            self.progress.progress = 50
            self._notify()
            
            from data.features import FeatureEngine
            from data.processor import DataProcessor
            
            feature_engine = FeatureEngine()
            processor = DataProcessor()
            
            # Phase 1: Create features for all stocks
            processed_data = {}
            
            for i, (code, df) in enumerate(valid_data.items()):
                if self._stop_flag:
                    self._set_stopped()
                    return
                
                self.progress.message = f"处理特征 {code} ({i+1}/{len(valid_data)})"
                self.progress.progress = 50 + (i + 1) / len(valid_data) * 5  # 50-55%
                self._notify()
                
                try:
                    df = feature_engine.create_features(df)
                    df = processor.create_labels(df)
                    processed_data[code] = df
                except Exception as e:
                    log.warning(f"Failed to process {code}: {e}")
            
            if len(processed_data) < 3:
                self.progress.message = "处理后数据不足"
                self.progress.stage = "error"
                self._notify()
                return
            
            # Phase 2: Fit scaler on training data
            self.progress.message = "准备缩放器..."
            self._notify()
            
            feature_cols = feature_engine.get_feature_columns()
            all_train_features = []
            
            for code, df in processed_data.items():
                n = len(df)
                train_end = int(n * CONFIG.TRAIN_RATIO)
                train_df = df.iloc[:train_end]
                all_train_features.append(train_df[feature_cols].values)
            
            combined_features = np.concatenate(all_train_features)
            processor.fit_scaler(combined_features)
            
            # Phase 3: Create sequences
            self.progress.message = "创建序列..."
            self._notify()
            
            all_train = {'X': [], 'y': [], 'r': []}
            all_val = {'X': [], 'y': [], 'r': []}
            all_test = {'X': [], 'y': [], 'r': []}
            
            for code, df in processed_data.items():
                n = len(df)
                train_end = int(n * CONFIG.TRAIN_RATIO)
                val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
                
                for split_name, start_idx, end_idx, storage in [
                    ('train', 0, train_end, all_train),
                    ('val', train_end, val_end, all_val),
                    ('test', val_end, n, all_test)
                ]:
                    split_df = df.iloc[start_idx:end_idx]
                    
                    if len(split_df) >= CONFIG.SEQUENCE_LENGTH + 5:
                        X, y, r = processor.prepare_sequences(
                            split_df, feature_cols, fit_scaler=False
                        )
                        if len(X) > 0:
                            storage['X'].append(X)
                            storage['y'].append(y)
                            storage['r'].append(r)
            
            # Combine arrays
            X_train = np.concatenate(all_train['X']) if all_train['X'] else np.array([])
            y_train = np.concatenate(all_train['y']) if all_train['y'] else np.array([])
            
            X_val = np.concatenate(all_val['X']) if all_val['X'] else np.array([])
            y_val = np.concatenate(all_val['y']) if all_val['y'] else np.array([])
            
            X_test = np.concatenate(all_test['X']) if all_test['X'] else np.array([])
            y_test = np.concatenate(all_test['y']) if all_test['y'] else np.array([])
            
            if len(X_train) < 100:
                self.progress.message = "训练数据不足"
                self.progress.stage = "error"
                self._notify()
                return
            
            log.info(f"Data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            # Save scaler
            processor.save_scaler()
            
            # =====================================
            # STAGE 4: Train models
            # =====================================
            self.progress.stage = "training"
            self.progress.message = "训练AI模型..."
            self.progress.progress = 60
            self._notify()
            
            from models.ensemble import EnsembleModel
            
            input_size = X_train.shape[2]
            
            # Load existing model if incremental
            if incremental:
                try:
                    existing = EnsembleModel(input_size)
                    if existing.load():
                        log.info("Loaded existing model for incremental learning")
                except:
                    pass
            
            # Create ensemble
            ensemble = EnsembleModel(input_size)
            
            def training_callback(model_name, epoch, val_acc):
                if self._stop_flag:
                    return
                
                self.progress.training_epoch = epoch + 1
                self.progress.training_accuracy = val_acc
                self.progress.message = f"训练 {model_name}: Epoch {epoch+1}/{epochs}"
                self.progress.progress = 60 + (epoch + 1) / epochs * 35  # 60-95%
                self._notify()
            
            history = ensemble.train(
                X_train, y_train,
                X_val, y_val,
                epochs=epochs,
                callback=training_callback
            )
            
            if self._stop_flag:
                self._set_stopped()
                return
            
            # =====================================
            # STAGE 5: Evaluate and save
            # =====================================
            self.progress.stage = "evaluating"
            self.progress.message = "评估模型..."
            self.progress.progress = 95
            self._notify()
            
            # Test accuracy
            correct = 0
            total = len(X_test)
            
            for i in range(min(total, 1000)):  # Limit evaluation samples
                pred = ensemble.predict(X_test[i:i+1])
                if pred.predicted_class == y_test[i]:
                    correct += 1
            
            test_accuracy = correct / min(total, 1000) if total > 0 else 0
            log.info(f"Test accuracy: {test_accuracy:.2%}")
            
            # Save model
            ensemble.save()
            
            # =====================================
            # STAGE 6: Complete
            # =====================================
            self.progress.stage = "complete"
            self.progress.message = f"训练完成！准确率: {test_accuracy:.1%}"
            self.progress.progress = 100
            self.progress.training_accuracy = test_accuracy
            self._notify()
            
            # Save session to history
            duration = (datetime.now() - session_start).total_seconds() / 60
            
            session = {
                'timestamp': session_start.isoformat(),
                'duration_minutes': duration,
                'stocks_searched': self.progress.stocks_found,
                'stocks_used': len(valid_data),
                'samples': len(X_train) + len(X_val) + len(X_test),
                'epochs': epochs,
                'test_accuracy': test_accuracy,
                'incremental': incremental
            }
            
            self.history['sessions'].append(session)
            self.history['best_accuracy'] = max(
                self.history.get('best_accuracy', 0),
                test_accuracy
            )
            self.history['total_stocks'] = len(set(
                s.get('stocks_used', 0) for s in self.history.get('sessions', [])
            ))
            self.history['last_update'] = datetime.now().isoformat()
            self._save_history()
            
            log.info(f"Auto-learning completed in {duration:.1f} minutes!")
            
        except Exception as e:
            import traceback
            log.error(f"Auto-learning failed: {e}")
            traceback.print_exc()
            
            self.progress.stage = "error"
            self.progress.message = f"错误: {str(e)}"
            self.progress.errors.append(str(e))
            self._notify()
        
        finally:
            self.progress.is_running = False
            self._notify()
    
    def _set_stopped(self):
        """Set stopped state"""
        self.progress.stage = "idle"
        self.progress.message = "已停止"
        self.progress.is_running = False
        self._notify()
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics"""
        return {
            'sessions_count': len(self.history.get('sessions', [])),
            'best_accuracy': self.history.get('best_accuracy', 0),
            'total_stocks': self.history.get('total_stocks', 0),
            'last_update': self.history.get('last_update'),
            'current_progress': self.progress
        }