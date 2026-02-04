"""
Data Fetcher - Get stock data from multiple sources
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass
import time
import pickle
from pathlib import Path

from loguru import logger

try:
    import akshare as ak
    AKSHARE_OK = True
except ImportError:
    AKSHARE_OK = False
    logger.warning("akshare not installed")

from config import CONFIG


@dataclass
class Quote:
    """Real-time quote data"""
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


class DataFetcher:
    """
    Fetches stock data from various sources
    with caching and error handling
    """
    
    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = 4  # hours
        self._rate_limit = 0.5  # seconds between requests
        self._last_request = 0
    
    def _wait_rate_limit(self):
        """Respect rate limits"""
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_request = time.time()
    
    def get_history(self, 
                    code: str, 
                    days: int = 500,
                    use_cache: bool = True) -> pd.DataFrame:
        """
        Get historical OHLCV data
        
        Args:
            code: Stock code (e.g., "600519")
            days: Number of days to fetch
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns: open, high, low, close, volume, amount
        """
        code = self._clean_code(code)
        cache_key = f"{code}_{days}"
        
        # Check cache
        if use_cache and cache_key in self._cache:
            cache_age = (datetime.now() - self._cache_time.get(cache_key, datetime.min))
            if cache_age.total_seconds() / 3600 < self._cache_ttl:
                logger.debug(f"Cache hit for {code}")
                return self._cache[cache_key].copy()
        
        # Fetch from source
        df = self._fetch_akshare(code, days)
        
        if df.empty:
            # Try loading from disk cache
            df = self._load_disk_cache(code)
        
        if not df.empty:
            self._cache[cache_key] = df.copy()
            self._cache_time[cache_key] = datetime.now()
            self._save_disk_cache(code, df)
        
        return df
    
    def _fetch_akshare(self, code: str, days: int) -> pd.DataFrame:
        """Fetch from AKShare"""
        if not AKSHARE_OK:
            return pd.DataFrame()
        
        self._wait_rate_limit()
        
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
            
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # Forward adjusted
            )
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # Standardize columns
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '涨跌幅': 'change_pct',
                '换手率': 'turnover'
            })
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Ensure numeric
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove invalid rows
            df = df.dropna(subset=['close', 'volume'])
            df = df[df['high'] >= df['low']]
            
            logger.info(f"Fetched {len(df)} bars for {code}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {code}: {e}")
            return pd.DataFrame()
    
    def get_realtime(self, code: str) -> Optional[Quote]:
        """Get real-time quote"""
        if not AKSHARE_OK:
            return None
        
        self._wait_rate_limit()
        
        try:
            df = ak.stock_zh_a_spot_em()
            row = df[df['代码'] == code]
            
            if row.empty:
                return None
            
            r = row.iloc[0]
            
            return Quote(
                code=code,
                name=str(r.get('名称', '')),
                price=float(r.get('最新价', 0)),
                open=float(r.get('今开', 0)),
                high=float(r.get('最高', 0)),
                low=float(r.get('最低', 0)),
                volume=int(r.get('成交量', 0)),
                amount=float(r.get('成交额', 0)),
                change_pct=float(r.get('涨跌幅', 0)),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to get quote for {code}: {e}")
            return None
    
    def get_stock_list(self) -> pd.DataFrame:
        """Get list of all A-share stocks"""
        if not AKSHARE_OK:
            return pd.DataFrame()
        
        try:
            df = ak.stock_zh_a_spot_em()
            return df[['代码', '名称']].rename(columns={'代码': 'code', '名称': 'name'})
        except:
            return pd.DataFrame()
    
    def get_stock_name(self, code: str) -> str:
        """Get stock name by code"""
        quote = self.get_realtime(code)
        return quote.name if quote else code
    
    def _clean_code(self, code: str) -> str:
        """Clean stock code"""
        return str(code).replace('sh', '').replace('sz', '').replace('.', '').strip().zfill(6)
    
    def _save_disk_cache(self, code: str, df: pd.DataFrame):
        """Save to disk cache"""
        try:
            path = CONFIG.DATA_DIR / f"{code}.pkl"
            df.to_pickle(path)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_disk_cache(self, code: str) -> pd.DataFrame:
        """Load from disk cache"""
        try:
            path = CONFIG.DATA_DIR / f"{code}.pkl"
            if path.exists():
                df = pd.read_pickle(path)
                logger.debug(f"Loaded {code} from disk cache")
                return df
        except:
            pass
        return pd.DataFrame()