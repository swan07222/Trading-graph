"""
Auto-Learning AI System
Automatically searches internet, downloads data, and trains models
"""
import os
import sys
import json
import time
import random
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# Web scraping
import requests
from bs4 import BeautifulSoup

try:
    import akshare as ak
    AKSHARE_OK = True
except ImportError:
    AKSHARE_OK = False

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


@dataclass
class StockInfo:
    """Stock information from web search"""
    code: str
    name: str
    source: str
    reason: str  # Why this stock was found (trending, news, etc.)
    score: float  # Relevance score
    timestamp: datetime = field(default_factory=datetime.now)


class InternetStockFinder:
    """
    Automatically find stocks from various internet sources
    
    Sources:
    1. Trending stocks (涨幅榜, 热门股票)
    2. News mentions
    3. Analyst recommendations
    4. Sector leaders
    5. New highs/lows
    6. Volume breakouts
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self._rate_limit = 1.0  # seconds between requests
        self._last_request = 0
    
    def _wait(self):
        """Rate limiting"""
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_request = time.time()
    
    def find_all_stocks(self, callback: Callable = None) -> List[StockInfo]:
        """
        Find stocks from all sources
        
        Args:
            callback: Optional progress callback(message, progress)
        """
        all_stocks = []
        
        sources = [
            ("涨幅榜", self.find_top_gainers),
            ("跌幅榜", self.find_top_losers),
            ("成交额榜", self.find_top_volume),
            ("热门股票", self.find_trending),
            ("机构推荐", self.find_analyst_picks),
            ("行业龙头", self.find_sector_leaders),
            ("创新高", self.find_new_highs),
            ("放量突破", self.find_volume_breakouts),
        ]
        
        for i, (name, finder) in enumerate(sources):
            try:
                if callback:
                    callback(f"搜索 {name}...", (i + 1) / len(sources) * 100)
                
                stocks = finder()
                all_stocks.extend(stocks)
                log.info(f"Found {len(stocks)} stocks from {name}")
                
            except Exception as e:
                log.warning(f"Failed to search {name}: {e}")
        
        # Remove duplicates, keep highest score
        unique = {}
        for stock in all_stocks:
            if stock.code not in unique or stock.score > unique[stock.code].score:
                unique[stock.code] = stock
        
        result = sorted(unique.values(), key=lambda x: x.score, reverse=True)
        log.info(f"Total unique stocks found: {len(result)}")
        
        return result
    
    def find_top_gainers(self, limit: int = 50) -> List[StockInfo]:
        """Find top gaining stocks today"""
        if not AKSHARE_OK:
            return []
        
        self._wait()
        
        try:
            df = ak.stock_zh_a_spot_em()
            df = df.sort_values('涨跌幅', ascending=False).head(limit)
            
            stocks = []
            for _, row in df.iterrows():
                code = str(row['代码']).zfill(6)
                stocks.append(StockInfo(
                    code=code,
                    name=row['名称'],
                    source="涨幅榜",
                    reason=f"今日涨幅 {row['涨跌幅']:.2f}%",
                    score=min(row['涨跌幅'] / 10, 1.0)  # Normalize
                ))
            
            return stocks
            
        except Exception as e:
            log.error(f"find_top_gainers error: {e}")
            return []
    
    def find_top_losers(self, limit: int = 30) -> List[StockInfo]:
        """Find top losing stocks (potential rebounds)"""
        if not AKSHARE_OK:
            return []
        
        self._wait()
        
        try:
            df = ak.stock_zh_a_spot_em()
            df = df.sort_values('涨跌幅', ascending=True).head(limit)
            
            stocks = []
            for _, row in df.iterrows():
                code = str(row['代码']).zfill(6)
                stocks.append(StockInfo(
                    code=code,
                    name=row['名称'],
                    source="跌幅榜",
                    reason=f"今日跌幅 {row['涨跌幅']:.2f}% (反弹机会)",
                    score=min(abs(row['涨跌幅']) / 10, 0.8)
                ))
            
            return stocks
            
        except Exception as e:
            log.error(f"find_top_losers error: {e}")
            return []
    
    def find_top_volume(self, limit: int = 50) -> List[StockInfo]:
        """Find top volume stocks"""
        if not AKSHARE_OK:
            return []
        
        self._wait()
        
        try:
            df = ak.stock_zh_a_spot_em()
            df = df.sort_values('成交额', ascending=False).head(limit)
            
            stocks = []
            for _, row in df.iterrows():
                code = str(row['代码']).zfill(6)
                volume_b = row['成交额'] / 1e8  # Convert to 亿
                stocks.append(StockInfo(
                    code=code,
                    name=row['名称'],
                    source="成交额榜",
                    reason=f"成交额 {volume_b:.1f}亿",
                    score=min(volume_b / 100, 1.0)
                ))
            
            return stocks
            
        except Exception as e:
            log.error(f"find_top_volume error: {e}")
            return []
    
    def find_trending(self, limit: int = 30) -> List[StockInfo]:
        """Find trending/hot stocks"""
        if not AKSHARE_OK:
            return []
        
        self._wait()
        
        try:
            # Try to get hot stocks
            df = ak.stock_hot_rank_em()
            
            stocks = []
            for i, row in df.head(limit).iterrows():
                code = str(row.get('代码', row.get('股票代码', ''))).zfill(6)
                if len(code) != 6:
                    continue
                    
                stocks.append(StockInfo(
                    code=code,
                    name=row.get('名称', row.get('股票名称', '')),
                    source="热门股票",
                    reason=f"热度排名 #{i+1}",
                    score=1.0 - (i / limit)
                ))
            
            return stocks
            
        except Exception as e:
            log.warning(f"find_trending error: {e}")
            return []
    
    def find_analyst_picks(self, limit: int = 30) -> List[StockInfo]:
        """Find stocks with analyst recommendations"""
        if not AKSHARE_OK:
            return []
        
        self._wait()
        
        try:
            df = ak.stock_rank_forecast_cninfo()
            
            stocks = []
            for _, row in df.head(limit).iterrows():
                code = str(row.get('代码', '')).zfill(6)
                if len(code) != 6:
                    continue
                    
                stocks.append(StockInfo(
                    code=code,
                    name=row.get('名称', ''),
                    source="机构推荐",
                    reason="分析师推荐",
                    score=0.8
                ))
            
            return stocks
            
        except Exception as e:
            log.warning(f"find_analyst_picks error: {e}")
            return []
    
    def find_sector_leaders(self, limit: int = 30) -> List[StockInfo]:
        """Find sector/industry leaders"""
        if not AKSHARE_OK:
            return []
        
        self._wait()
        
        try:
            # Get industry data
            df = ak.stock_board_industry_name_em()
            
            stocks = []
            for _, row in df.head(limit).iterrows():
                # Get leader stocks for this industry
                try:
                    industry_name = row['板块名称']
                    leader_df = ak.stock_board_industry_cons_em(symbol=industry_name)
                    
                    if len(leader_df) > 0:
                        top = leader_df.head(3)  # Top 3 in each industry
                        for _, lrow in top.iterrows():
                            code = str(lrow.get('代码', '')).zfill(6)
                            if len(code) == 6:
                                stocks.append(StockInfo(
                                    code=code,
                                    name=lrow.get('名称', ''),
                                    source="行业龙头",
                                    reason=f"{industry_name} 行业龙头",
                                    score=0.9
                                ))
                except:
                    pass
            
            return stocks[:limit]
            
        except Exception as e:
            log.warning(f"find_sector_leaders error: {e}")
            return []
    
    def find_new_highs(self, limit: int = 30) -> List[StockInfo]:
        """Find stocks at 52-week highs"""
        if not AKSHARE_OK:
            return []
        
        self._wait()
        
        try:
            df = ak.stock_zh_a_spot_em()
            
            stocks = []
            for _, row in df.iterrows():
                code = str(row['代码']).zfill(6)
                current = row['最新价']
                high_52w = row.get('52周最高', current * 1.1)
                
                # If current is within 2% of 52-week high
                if current >= high_52w * 0.98:
                    stocks.append(StockInfo(
                        code=code,
                        name=row['名称'],
                        source="创新高",
                        reason=f"接近52周新高 ¥{high_52w:.2f}",
                        score=0.85
                    ))
            
            return sorted(stocks, key=lambda x: x.score, reverse=True)[:limit]
            
        except Exception as e:
            log.warning(f"find_new_highs error: {e}")
            return []
    
    def find_volume_breakouts(self, limit: int = 30) -> List[StockInfo]:
        """Find stocks with volume breakouts"""
        if not AKSHARE_OK:
            return []
        
        self._wait()
        
        try:
            df = ak.stock_zh_a_spot_em()
            df = df[df['换手率'] > 5]  # High turnover
            df = df.sort_values('换手率', ascending=False).head(limit)
            
            stocks = []
            for _, row in df.iterrows():
                code = str(row['代码']).zfill(6)
                turnover = row['换手率']
                
                stocks.append(StockInfo(
                    code=code,
                    name=row['名称'],
                    source="放量突破",
                    reason=f"换手率 {turnover:.1f}%",
                    score=min(turnover / 20, 1.0)
                ))
            
            return stocks
            
        except Exception as e:
            log.warning(f"find_volume_breakouts error: {e}")
            return []


class AutoLearner:
    """
    Automatic Learning System
    
    Features:
    1. Auto-search internet for stocks
    2. Auto-download historical data
    3. Auto-train AI models
    4. Continuous learning (incremental updates)
    5. Model versioning
    6. Performance tracking
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
            except:
                pass
    
    def _load_history(self) -> Dict:
        """Load learning history"""
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
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
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            log.error(f"Failed to save history: {e}")
    
    def start_learning(self, 
                       auto_search: bool = True,
                       max_stocks: int = 100,
                       epochs: int = 100,
                       incremental: bool = True):
        """
        Start auto-learning process
        
        Args:
            auto_search: Search internet for stocks
            max_stocks: Maximum stocks to include
            epochs: Training epochs
            incremental: Use incremental learning (keep old knowledge)
        """
        if self._thread and self._thread.is_alive():
            log.warning("Learning already in progress")
            return
        
        self._stop_flag = False
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
            self._thread.join(timeout=5)
    
    def _learning_loop(self, auto_search: bool, max_stocks: int, 
                       epochs: int, incremental: bool):
        """Main learning loop"""
        try:
            self.progress.is_running = True
            self.progress.errors = []
            
            session_start = datetime.now()
            
            # =====================================
            # STAGE 1: Search for stocks
            # =====================================
            self.progress.stage = "searching"
            self.progress.message = "搜索互联网寻找股票..."
            self.progress.progress = 0
            self._notify()
            
            if auto_search:
                def search_callback(msg, prog):
                    self.progress.message = msg
                    self.progress.progress = prog * 0.2  # 0-20%
                    self._notify()
                
                stocks = self.finder.find_all_stocks(callback=search_callback)
                self.progress.stocks_found = len(stocks)
            else:
                # Use default stock pool
                stocks = [
                    StockInfo(code=code, name="", source="config", reason="Default pool", score=1.0)
                    for code in CONFIG.STOCK_POOL
                ]
                self.progress.stocks_found = len(stocks)
            
            if self._stop_flag:
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
            for i, code in enumerate(selected_codes):
                if self._stop_flag:
                    return
                
                self.progress.stocks_processed = i + 1
                self.progress.message = f"下载 {code} ({i+1}/{len(selected_codes)})"
                self.progress.progress = 20 + (i + 1) / len(selected_codes) * 30  # 20-50%
                self._notify()
                
                try:
                    df = fetcher.get_history(code, days=1500, use_cache=False)
                    if len(df) >= 200:  # Minimum data requirement
                        valid_data[code] = df
                        log.debug(f"Downloaded {code}: {len(df)} days")
                except Exception as e:
                    log.warning(f"Failed to download {code}: {e}")
                    self.progress.errors.append(f"{code}: {str(e)[:50]}")
            
            log.info(f"Valid stocks with data: {len(valid_data)}")
            
            if len(valid_data) < 5:
                self.progress.message = "数据不足，无法训练"
                self.progress.stage = "error"
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
            
            all_X, all_y, all_r = [], [], []
            
            for i, (code, df) in enumerate(valid_data.items()):
                if self._stop_flag:
                    return
                
                self.progress.message = f"处理 {code} ({i+1}/{len(valid_data)})"
                self.progress.progress = 50 + (i + 1) / len(valid_data) * 10  # 50-60%
                self._notify()
                
                try:
                    df = feature_engine.create_features(df)
                    df = processor.create_labels(df)
                    
                    feature_cols = feature_engine.get_feature_columns()
                    X, y, r = processor.prepare_sequences(df, feature_cols)
                    
                    if len(X) > 0:
                        all_X.append(X)
                        all_y.append(y)
                        all_r.append(r)
                        
                except Exception as e:
                    log.warning(f"Failed to process {code}: {e}")
            
            if not all_X:
                self.progress.message = "无法准备训练数据"
                self.progress.stage = "error"
                self._notify()
                return
            
            X = np.concatenate(all_X)
            y = np.concatenate(all_y)
            r = np.concatenate(all_r)
            
            # Shuffle
            idx = np.random.permutation(len(X))
            X, y, r = X[idx], y[idx], r[idx]
            
            # Split
            n = len(X)
            train_end = int(n * 0.7)
            val_end = int(n * 0.85)
            
            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:val_end], y[train_end:val_end]
            X_test, y_test = X[val_end:], y[val_end:]
            
            log.info(f"Training data: {len(X_train)} samples")
            log.info(f"Validation data: {len(X_val)} samples")
            log.info(f"Test data: {len(X_test)} samples")
            
            # =====================================
            # STAGE 4: Train models
            # =====================================
            self.progress.stage = "training"
            self.progress.message = "训练AI模型..."
            self.progress.progress = 60
            self._notify()
            
            import torch
            from models.ensemble import EnsembleModel
            
            input_size = X_train.shape[2]
            
            # Load existing model for incremental learning
            if incremental:
                try:
                    existing_model = EnsembleModel(input_size)
                    if existing_model.load():
                        log.info("Loaded existing model for incremental learning")
                except:
                    pass
            
            # Create new ensemble
            ensemble = EnsembleModel(input_size)
            
            def training_callback(model_name, epoch, val_acc):
                self.progress.training_epoch = epoch + 1
                self.progress.training_accuracy = val_acc
                self.progress.message = f"训练 {model_name}: Epoch {epoch+1}, 准确率 {val_acc:.1%}"
                self.progress.progress = 60 + (epoch + 1) / epochs * 35  # 60-95%
                self._notify()
            
            history = ensemble.train(
                X_train, y_train,
                X_val, y_val,
                epochs=epochs,
                callback=training_callback
            )
            
            if self._stop_flag:
                return
            
            # =====================================
            # STAGE 5: Evaluate and save
            # =====================================
            self.progress.stage = "evaluating"
            self.progress.message = "评估模型性能..."
            self.progress.progress = 95
            self._notify()
            
            # Test accuracy
            correct = 0
            for i in range(len(X_test)):
                pred = ensemble.predict(X_test[i:i+1])
                if pred.predicted_class == y_test[i]:
                    correct += 1
            
            test_accuracy = correct / len(X_test)
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
            session = {
                'timestamp': session_start.isoformat(),
                'duration_minutes': (datetime.now() - session_start).total_seconds() / 60,
                'stocks_searched': self.progress.stocks_found,
                'stocks_used': len(valid_data),
                'samples': len(X),
                'epochs': epochs,
                'test_accuracy': test_accuracy,
                'incremental': incremental
            }
            
            self.history['sessions'].append(session)
            self.history['best_accuracy'] = max(self.history['best_accuracy'], test_accuracy)
            self.history['total_stocks'] = len(set(
                code for s in self.history['sessions'] 
                for code in valid_data.keys()
            ))
            self.history['last_update'] = datetime.now().isoformat()
            self._save_history()
            
            log.info("Auto-learning completed successfully!")
            
        except Exception as e:
            log.error(f"Auto-learning failed: {e}")
            import traceback
            traceback.print_exc()
            
            self.progress.stage = "error"
            self.progress.message = f"错误: {str(e)}"
            self.progress.errors.append(str(e))
            self._notify()
        
        finally:
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


class ContinuousLearner:
    """
    Continuous Learning System
    Automatically updates model based on new data and trading results
    """
    
    def __init__(self, auto_learner: AutoLearner):
        self.auto_learner = auto_learner
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Schedule
        self.schedule = {
            'daily_update': True,      # Update data daily
            'weekly_retrain': True,    # Retrain weekly
            'learn_from_trades': True  # Learn from trading results
        }
        
        # Trading feedback
        self.trade_results: List[Dict] = []
    
    def start(self):
        """Start continuous learning"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info("Continuous learning started")
    
    def stop(self):
        """Stop continuous learning"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def add_trade_result(self, stock_code: str, prediction: int, 
                         actual: int, profit_pct: float):
        """Add trading result for learning"""
        self.trade_results.append({
            'code': stock_code,
            'prediction': prediction,
            'actual': actual,
            'profit_pct': profit_pct,
            'timestamp': datetime.now().isoformat()
        })
        
        # Learn from mistakes
        if prediction != actual:
            log.info(f"Learning from wrong prediction: {stock_code}")
    
    def _run_loop(self):
        """Main loop for continuous learning"""
        last_daily = datetime.now() - timedelta(days=1)
        last_weekly = datetime.now() - timedelta(weeks=1)
        
        while self._running:
            now = datetime.now()
            
            # Daily data update
            if self.schedule['daily_update']:
                if (now - last_daily).days >= 1:
                    if now.hour >= 18:  # After market close
                        self._daily_update()
                        last_daily = now
            
            # Weekly retrain
            if self.schedule['weekly_retrain']:
                if (now - last_weekly).days >= 7:
                    if now.weekday() == 5:  # Saturday
                        self._weekly_retrain()
                        last_weekly = now
            
            time.sleep(3600)  # Check hourly
    
    def _daily_update(self):
        """Daily data update"""
        log.info("Running daily data update...")
        
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        
        for code in CONFIG.STOCK_POOL:
            try:
                fetcher.get_history(code, days=30, use_cache=False)
            except:
                pass
    
    def _weekly_retrain(self):
        """Weekly model retrain"""
        log.info("Running weekly retrain...")
        
        self.auto_learner.start_learning(
            auto_search=True,
            max_stocks=50,
            epochs=50,
            incremental=True
        )