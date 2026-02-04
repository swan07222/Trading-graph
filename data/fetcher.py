"""
High-Performance Data Fetcher with Parallel Downloads
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import time
import pickle
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests

from loguru import logger
from config import CONFIG
from data.cache_manager import TieredCache, cached


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retry logic"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator


@dataclass
class Quote:
    """Real-time quote"""
    code: str
    name: str
    price: float
    open: float
    high: float
    low: float
    volume: int
    amount: float
    change_pct: float
    timestamp: datetime
    source: str = ""


class DataFetcher:
    """High-performance data fetcher with caching and parallel downloads"""
    
    def __init__(self):
        self._cache = TieredCache()
        self._ak = None
        self._yf = None
        self._rate_limiter = threading.Semaphore(5)  # Max 5 concurrent
        self._last_request = {}
        self._min_interval = 0.2  # 200ms between requests per source
        
        self._init_sources()
    
    def _init_sources(self):
        try:
            import akshare as ak
            self._ak = ak
            logger.info("AkShare initialized")
        except ImportError:
            logger.warning("AkShare not available")
        
        try:
            import yfinance as yf
            self._yf = yf
            logger.info("Yahoo Finance initialized")
        except ImportError:
            logger.warning("yfinance not available")
    
    def _rate_limit(self, source: str):
        """Per-source rate limiting"""
        now = time.time()
        last = self._last_request.get(source, 0)
        wait = self._min_interval - (now - last)
        if wait > 0:
            time.sleep(wait)
        self._last_request[source] = time.time()
    
    def _clean_code(self, code: str) -> str:
        code = str(code).strip()
        for prefix in ['sh', 'sz', 'SH', 'SZ', '.SS', '.SZ']:
            code = code.replace(prefix, '')
        return code.zfill(6)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def get_history(
        self, 
        code: str, 
        days: int = 500,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get historical data with caching"""
        code = self._clean_code(code)
        cache_key = f"history:{code}:{days}"
        
        # Check cache
        if use_cache:
            cached = self._cache.get(cache_key, max_age_hours=CONFIG.DISCOVERY_CACHE_HOURS)
            if cached is not None and len(cached) >= min(days, 100):
                return cached.tail(days)
        
        # Fetch fresh
        df = self._fetch_history_akshare(code, days)
        
        if df.empty and self._yf:
            df = self._fetch_history_yahoo(code, days)
        
        if not df.empty:
            self._cache.set(cache_key, df)
        
        return df
    
    def _fetch_history_akshare(self, code: str, days: int) -> pd.DataFrame:
        """Fetch from AkShare"""
        if not self._ak:
            return pd.DataFrame()
        
        try:
            with self._rate_limiter:
                self._rate_limit('akshare')
                
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")
                
                df = self._ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                
                if df is None or df.empty:
                    return pd.DataFrame()
                
                # Rename columns
                column_map = {
                    '日期': 'date', '开盘': 'open', '收盘': 'close',
                    '最高': 'high', '最低': 'low', '成交量': 'volume',
                    '成交额': 'amount', '涨跌幅': 'change_pct', '换手率': 'turnover'
                }
                df = df.rename(columns=column_map)
                
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                
                for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna(subset=['close', 'volume'])
                df = df[df['volume'] > 0]
                
                return df.tail(days)
                
        except Exception as e:
            logger.debug(f"AkShare failed for {code}: {e}")
            return pd.DataFrame()
    
    def _fetch_history_yahoo(self, code: str, days: int) -> pd.DataFrame:
        """Fetch from Yahoo Finance"""
        if not self._yf:
            return pd.DataFrame()
        
        try:
            with self._rate_limiter:
                self._rate_limit('yahoo')
                
                # Convert to Yahoo symbol
                suffix = '.SS' if code.startswith('6') else '.SZ'
                symbol = f"{code}{suffix}"
                
                ticker = self._yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=int(days * 1.5))
                
                df = ticker.history(start=start_date, end=end_date)
                
                if df.empty:
                    return pd.DataFrame()
                
                df = df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                })
                
                df.index.name = 'date'
                df = df[['open', 'high', 'low', 'close', 'volume']].copy()
                df['amount'] = df['close'] * df['volume']
                
                return df.tail(days)
                
        except Exception as e:
            logger.debug(f"Yahoo failed for {code}: {e}")
            return pd.DataFrame()
    
    def get_realtime(self, code: str) -> Optional[Quote]:
        """Get realtime quote"""
        code = self._clean_code(code)
        
        if self._ak:
            try:
                with self._rate_limiter:
                    self._rate_limit('akshare_rt')
                    df = self._ak.stock_zh_a_spot_em()
                    row = df[df['代码'] == code]
                    
                    if not row.empty:
                        r = row.iloc[0]
                        return Quote(
                            code=code,
                            name=str(r.get('名称', '')),
                            price=float(r.get('最新价', 0) or 0),
                            open=float(r.get('今开', 0) or 0),
                            high=float(r.get('最高', 0) or 0),
                            low=float(r.get('最低', 0) or 0),
                            volume=int(r.get('成交量', 0) or 0),
                            amount=float(r.get('成交额', 0) or 0),
                            change_pct=float(r.get('涨跌幅', 0) or 0),
                            timestamp=datetime.now(),
                            source='akshare'
                        )
            except Exception as e:
                logger.debug(f"Realtime failed for {code}: {e}")
        
        # Fallback to last historical
        df = self.get_history(code, days=5)
        if not df.empty:
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            change_pct = (last['close'] / prev['close'] - 1) * 100 if prev['close'] > 0 else 0
            
            return Quote(
                code=code,
                name=f"Stock {code}",
                price=float(last['close']),
                open=float(last['open']),
                high=float(last['high']),
                low=float(last['low']),
                volume=int(last['volume']),
                amount=float(last.get('amount', 0)),
                change_pct=change_pct,
                timestamp=datetime.now(),
                source='cache'
            )
        
        return None
    
    def get_multiple_parallel(
        self, 
        codes: List[str], 
        days: int = 500,
        max_workers: int = 10,
        callback: callable = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple stocks in parallel (MAJOR SPEED IMPROVEMENT)
        """
        results = {}
        total = len(codes)
        completed = 0
        lock = threading.Lock()
        
        def fetch_one(code: str) -> Tuple[str, pd.DataFrame]:
            df = self.get_history(code, days)
            return code, df
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_one, code): code for code in codes}
            
            for future in as_completed(futures):
                code = futures[future]
                try:
                    code, df = future.result()
                    if not df.empty and len(df) >= 100:
                        results[code] = df
                except Exception as e:
                    logger.warning(f"Failed to fetch {code}: {e}")
                
                with lock:
                    completed += 1
                    if callback:
                        callback(code, completed, total)
        
        logger.info(f"Parallel fetch complete: {len(results)}/{total} successful")
        return results
    
    def get_stock_name(self, code: str) -> str:
        """Get stock name"""
        quote = self.get_realtime(code)
        return quote.name if quote else code
    
    def clear_cache(self, code: str = None):
        """Clear cache"""
        if code:
            code = self._clean_code(code)
            # Would need to implement per-key deletion
        else:
            self._cache.clear()
        logger.info("Cache cleared")