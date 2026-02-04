"""
Robust Data Fetcher - Real Data with Multiple Sources and Fallback

Features:
- Multiple data source fallback (AkShare → Yahoo Finance)
- Automatic retry with exponential backoff
- Intelligent caching (memory + disk)
- Rate limiting to avoid API bans
- Proper error handling and logging

Author: AI Trading System
Version: 2.0
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import time
import pickle
import hashlib
from pathlib import Path
from functools import wraps
import requests

from loguru import logger

from config import CONFIG


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        # Log warning and retry
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * 2, max_delay)  # Exponential backoff with cap
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")
            
            raise last_exception
        return wrapper
    return decorator


@dataclass
class Quote:
    """
    Real-time quote data structure.
    
    Attributes:
        code: Stock code (e.g., "600519")
        name: Stock name (e.g., "贵州茅台")
        price: Current price
        open: Opening price
        high: Day high
        low: Day low
        volume: Trading volume (shares)
        amount: Trading amount (currency)
        change_pct: Price change percentage
        timestamp: Quote timestamp
    """
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


class DataSourceBase:
    """Base class for all data sources"""
    name: str = "base"
    priority: int = 0  # Lower = higher priority
    
    def get_history(self, code: str, days: int) -> pd.DataFrame:
        """Get historical OHLCV data"""
        raise NotImplementedError
    
    def get_realtime(self, code: str) -> Optional[Quote]:
        """Get real-time quote"""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if this data source is available"""
        return True


class AkShareDataSource(DataSourceBase):
    """
    AkShare data source for Chinese A-shares.
    Primary data source with comprehensive coverage.
    """
    name = "akshare"
    priority = 1
    
    def __init__(self):
        self._available = False
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
        })
        
        try:
            import akshare
            self._ak = akshare
            self._available = True
            logger.info("AkShare data source initialized")
        except ImportError:
            logger.warning("AkShare not installed. Run: pip install akshare")
    
    def is_available(self) -> bool:
        return self._available
    
    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def get_history(self, code: str, days: int) -> pd.DataFrame:
        """
        Get historical daily OHLCV data from AkShare.
        
        Args:
            code: Stock code (e.g., "600519")
            days: Number of trading days to fetch
            
        Returns:
            DataFrame with columns: open, high, low, close, volume, amount
        """
        if not self._available:
            raise RuntimeError("AkShare not available")
        
        end_date = datetime.now().strftime("%Y%m%d")
        # Fetch extra days to account for weekends/holidays
        start_date = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")
        
        df = self._ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"  # Forward adjusted prices
        )
        
        if df is None or df.empty:
            raise ValueError(f"No data returned for {code}")
        
        # Standardize column names (Chinese → English)
        column_map = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '涨跌幅': 'change_pct',
            '换手率': 'turnover'
        }
        df = df.rename(columns=column_map)
        
        # Set date as index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Data quality checks
        df = df.dropna(subset=['close', 'volume'])
        df = df[df['high'] >= df['low']]
        df = df[df['close'] > 0]
        df = df[df['volume'] > 0]
        
        logger.debug(f"AkShare: Got {len(df)} bars for {code}")
        return df.tail(days)
    
    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def get_realtime(self, code: str) -> Optional[Quote]:
        """Get real-time quote from AkShare"""
        if not self._available:
            return None
        
        try:
            df = self._ak.stock_zh_a_spot_em()
            row = df[df['代码'] == code]
            
            if row.empty:
                return None
            
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
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"AkShare realtime failed for {code}: {e}")
            return None


class YahooFinanceDataSource(DataSourceBase):
    """
    Yahoo Finance data source.
    Fallback for when AkShare is unavailable or rate limited.
    Works globally without geographic restrictions.
    """
    name = "yahoo"
    priority = 2
    
    # Map Chinese stock code prefixes to Yahoo Finance suffixes
    EXCHANGE_SUFFIX = {
        '6': '.SS',   # Shanghai Stock Exchange
        '0': '.SZ',   # Shenzhen Stock Exchange
        '3': '.SZ',   # Shenzhen ChiNext
    }
    
    def __init__(self):
        self._available = False
        
        try:
            import yfinance
            self._yf = yfinance
            self._available = True
            logger.info("Yahoo Finance data source initialized")
        except ImportError:
            logger.warning("yfinance not installed. Run: pip install yfinance")
    
    def is_available(self) -> bool:
        return self._available
    
    def _to_yahoo_symbol(self, code: str) -> str:
        """Convert Chinese stock code to Yahoo Finance symbol"""
        code = str(code).zfill(6)
        suffix = self.EXCHANGE_SUFFIX.get(code[0], '.SS')
        return f"{code}{suffix}"
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def get_history(self, code: str, days: int) -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        if not self._available:
            raise RuntimeError("Yahoo Finance not available")
        
        symbol = self._to_yahoo_symbol(code)
        ticker = self._yf.Ticker(symbol)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days * 1.5))
        
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        
        if df.empty:
            raise ValueError(f"No data from Yahoo for {symbol}")
        
        # Standardize column names
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        df.index.name = 'date'
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df['amount'] = df['close'] * df['volume']
        
        # Data quality
        df = df.dropna()
        df = df[df['close'] > 0]
        
        logger.debug(f"Yahoo: Got {len(df)} bars for {code}")
        return df.tail(days)
    
    def get_realtime(self, code: str) -> Optional[Quote]:
        """Get real-time quote from Yahoo Finance"""
        if not self._available:
            return None
        
        try:
            symbol = self._to_yahoo_symbol(code)
            ticker = self._yf.Ticker(symbol)
            info = ticker.info
            
            price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            
            if not price or price <= 0:
                return None
            
            return Quote(
                code=code,
                name=info.get('shortName', code),
                price=float(price),
                open=float(info.get('open', price) or price),
                high=float(info.get('dayHigh', price) or price),
                low=float(info.get('dayLow', price) or price),
                volume=int(info.get('volume', 0) or 0),
                amount=0,
                change_pct=float(info.get('regularMarketChangePercent', 0) or 0),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"Yahoo realtime failed for {code}: {e}")
            return None


class DataFetcher:
    """
    Unified data fetcher with intelligent source selection and caching.
    
    Features:
    - Automatic fallback between data sources
    - Multi-level caching (memory + disk)
    - Rate limiting to prevent API bans
    - Robust error handling
    
    Usage:
        fetcher = DataFetcher()
        df = fetcher.get_history("600519", days=500)
        quote = fetcher.get_realtime("600519")
    """
    
    def __init__(self):
        # Initialize data sources (ordered by priority)
        self._sources: List[DataSourceBase] = []
        
        # Add AkShare (primary for Chinese stocks)
        ak_source = AkShareDataSource()
        if ak_source.is_available():
            self._sources.append(ak_source)
        
        # Add Yahoo Finance (fallback)
        yf_source = YahooFinanceDataSource()
        if yf_source.is_available():
            self._sources.append(yf_source)
        
        if not self._sources:
            logger.error("No data sources available! Install akshare or yfinance.")
        else:
            logger.info(f"Data fetcher initialized with {len(self._sources)} sources: "
                       f"{[s.name for s in self._sources]}")
        
        # Memory cache
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl_hours = 4
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.5  # seconds
    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _clean_code(self, code: str) -> str:
        """Normalize stock code format"""
        code = str(code).strip()
        # Remove exchange prefixes/suffixes
        for prefix in ['sh', 'sz', 'SH', 'SZ']:
            code = code.replace(prefix, '')
        code = code.replace('.', '')
        return code.zfill(6)
    
    def _get_cache_key(self, code: str, days: int) -> str:
        """Generate unique cache key"""
        return f"{code}_{days}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._cache_time:
            return False
        
        age = datetime.now() - self._cache_time[cache_key]
        return age.total_seconds() / 3600 < self._cache_ttl_hours
    
    def _save_to_disk_cache(self, code: str, df: pd.DataFrame):
        """Save data to disk cache"""
        try:
            cache_file = CONFIG.DATA_DIR / f"cache_{code}.pkl"
            cache_data = {
                'data': df,
                'timestamp': datetime.now(),
                'version': '2.0'
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.debug(f"Saved {code} to disk cache")
        except Exception as e:
            logger.warning(f"Failed to save disk cache for {code}: {e}")
    
    def _load_from_disk_cache(self, code: str, max_age_hours: float = 24) -> Optional[pd.DataFrame]:
        """Load data from disk cache if fresh enough"""
        try:
            cache_file = CONFIG.DATA_DIR / f"cache_{code}.pkl"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Handle old cache format
            if isinstance(cache_data, pd.DataFrame):
                return cache_data
            
            # Check cache age
            cache_time = cache_data.get('timestamp', datetime.min)
            age = (datetime.now() - cache_time).total_seconds() / 3600
            
            if age > max_age_hours:
                logger.debug(f"Disk cache for {code} is stale ({age:.1f}h old)")
                return None
            
            logger.debug(f"Loaded {code} from disk cache ({age:.1f}h old)")
            return cache_data['data']
            
        except Exception as e:
            logger.warning(f"Failed to load disk cache for {code}: {e}")
            return None
    
    def get_history(self, 
                    code: str, 
                    days: int = 500,
                    use_cache: bool = True,
                    force_refresh: bool = False) -> pd.DataFrame:
        """
        Get historical OHLCV data for a stock.
        
        Args:
            code: Stock code (e.g., "600519")
            days: Number of trading days to fetch
            use_cache: Whether to use cached data
            force_refresh: Force fetch from source, ignoring cache
            
        Returns:
            DataFrame with columns: open, high, low, close, volume, amount
            Empty DataFrame if all sources fail
        """
        code = self._clean_code(code)
        cache_key = self._get_cache_key(code, days)
        
        # Check memory cache
        if use_cache and not force_refresh and self._is_cache_valid(cache_key):
            logger.debug(f"Memory cache hit for {code}")
            return self._cache[cache_key].copy()
        
        # Check disk cache
        if use_cache and not force_refresh:
            disk_data = self._load_from_disk_cache(code, max_age_hours=self._cache_ttl_hours)
            if disk_data is not None and len(disk_data) >= min(days, 50):
                self._cache[cache_key] = disk_data
                self._cache_time[cache_key] = datetime.now()
                return disk_data.tail(days).copy()
        
        # Fetch from sources
        self._rate_limit()
        
        for source in self._sources:
            try:
                logger.info(f"Fetching {code} from {source.name}...")
                df = source.get_history(code, days)
                
                if df is not None and len(df) >= 30:  # Minimum data requirement
                    logger.info(f"Got {len(df)} bars for {code} from {source.name}")
                    
                    # Update caches
                    self._cache[cache_key] = df.copy()
                    self._cache_time[cache_key] = datetime.now()
                    self._save_to_disk_cache(code, df)
                    
                    return df
                else:
                    logger.warning(f"{source.name} returned insufficient data for {code}")
                    
            except Exception as e:
                logger.warning(f"{source.name} failed for {code}: {e}")
                continue
        
        # All sources failed - try stale disk cache as last resort
        stale_data = self._load_from_disk_cache(code, max_age_hours=168)  # 1 week
        if stale_data is not None:
            logger.warning(f"Using stale cache for {code} (all sources failed)")
            return stale_data.tail(days)
        
        logger.error(f"All data sources failed for {code}")
        return pd.DataFrame()
    
    def get_realtime(self, code: str) -> Optional[Quote]:
        """
        Get real-time quote for a stock.
        
        Args:
            code: Stock code (e.g., "600519")
            
        Returns:
            Quote object or None if unavailable
        """
        code = self._clean_code(code)
        self._rate_limit()
        
        for source in self._sources:
            try:
                quote = source.get_realtime(code)
                if quote and quote.price > 0:
                    return quote
            except Exception as e:
                logger.warning(f"{source.name} realtime failed for {code}: {e}")
                continue
        
        # Fallback: use last close from history
        df = self.get_history(code, days=5, use_cache=True)
        if not df.empty:
            last = df.iloc[-1]
            prev_close = df.iloc[-2]['close'] if len(df) > 1 else last['close']
            
            return Quote(
                code=code,
                name=f"Stock {code}",
                price=float(last['close']),
                open=float(last['open']),
                high=float(last['high']),
                low=float(last['low']),
                volume=int(last['volume']),
                amount=float(last.get('amount', 0)),
                change_pct=float((last['close'] / prev_close - 1) * 100) if prev_close > 0 else 0,
                timestamp=datetime.now()
            )
        
        return None
    
    def get_stock_name(self, code: str) -> str:
        """Get stock name by code"""
        quote = self.get_realtime(code)
        return quote.name if quote else code
    
    def get_multiple(self, 
                     codes: List[str], 
                     days: int = 500,
                     progress_callback=None) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple stocks.
        
        Args:
            codes: List of stock codes
            days: Number of days per stock
            progress_callback: Optional callback(current, total, code)
            
        Returns:
            Dictionary mapping code to DataFrame
        """
        result = {}
        total = len(codes)
        
        for i, code in enumerate(codes):
            if progress_callback:
                progress_callback(i + 1, total, code)
            
            df = self.get_history(code, days)
            if not df.empty:
                result[code] = df
        
        logger.info(f"Fetched data for {len(result)}/{total} stocks")
        return result
    
    def clear_cache(self, code: str = None):
        """Clear cache (memory and optionally disk)"""
        if code:
            code = self._clean_code(code)
            # Clear memory cache for this code
            keys_to_remove = [k for k in self._cache if k.startswith(code)]
            for key in keys_to_remove:
                del self._cache[key]
                if key in self._cache_time:
                    del self._cache_time[key]
            
            # Clear disk cache
            cache_file = CONFIG.DATA_DIR / f"cache_{code}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            
            logger.info(f"Cleared cache for {code}")
        else:
            # Clear all
            self._cache.clear()
            self._cache_time.clear()
            
            for cache_file in CONFIG.DATA_DIR.glob("cache_*.pkl"):
                cache_file.unlink()
            
            logger.info("Cleared all cache")