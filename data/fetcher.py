"""
Robust Data Fetcher - Multiple Real Data Sources with Fallback
NO MOCK DATA - Real market data only

Data Sources (in priority order):
1. AkShare (Primary - Chinese A-shares, free)
2. Yahoo Finance (Fallback - Global, requires yfinance)
3. Tushare (Alternative - Chinese, requires token)
4. Local Cache (Disk cache for offline/backup)

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
import json
from pathlib import Path
from functools import wraps
import requests

from loguru import logger

from config import CONFIG


# ============================================================
# Retry Decorator
# ============================================================

def retry_on_failure(max_retries: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """
    Decorator for retry logic with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
    """
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
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            logger.error(f"{func.__name__} failed after {max_retries} attempts: {last_exception}")
            raise last_exception
        return wrapper
    return decorator


# ============================================================
# Data Models
# ============================================================

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
    source: str = ""  # Which data source provided this


@dataclass
class DataSourceStatus:
    """Track data source health"""
    name: str
    available: bool
    last_success: Optional[datetime] = None
    last_error: Optional[str] = None
    success_count: int = 0
    error_count: int = 0


# ============================================================
# Base Data Source
# ============================================================

class DataSource:
    """Abstract base class for data sources"""
    name: str = "base"
    priority: int = 0
    
    def __init__(self):
        self.status = DataSourceStatus(name=self.name, available=True)
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_history(self, code: str, days: int) -> pd.DataFrame:
        """Get historical OHLCV data"""
        raise NotImplementedError
    
    def get_realtime(self, code: str) -> Optional[Quote]:
        """Get real-time quote"""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if source is available"""
        return self.status.available
    
    def _record_success(self):
        """Record successful request"""
        self.status.last_success = datetime.now()
        self.status.success_count += 1
        self.status.available = True
    
    def _record_error(self, error: str):
        """Record failed request"""
        self.status.last_error = error
        self.status.error_count += 1
        
        # Disable source after too many consecutive errors
        if self.status.error_count > 5:
            recent_success = self.status.last_success
            if recent_success is None or (datetime.now() - recent_success).seconds > 300:
                self.status.available = False
                logger.warning(f"Data source {self.name} disabled due to repeated errors")


# ============================================================
# AkShare Data Source (Primary for Chinese A-shares)
# ============================================================

class AkShareSource(DataSource):
    """
    AkShare - Primary data source for Chinese A-shares
    Free, no API key required
    https://akshare.akfamily.xyz/
    """
    name = "akshare"
    priority = 1
    
    def __init__(self):
        super().__init__()
        self._ak = None
        self._check_availability()
    
    def _check_availability(self):
        """Check if akshare is installed"""
        try:
            import akshare as ak
            self._ak = ak
            self.status.available = True
            logger.info("AkShare data source initialized")
        except ImportError:
            self.status.available = False
            logger.warning("AkShare not installed. Run: pip install akshare")
    
    @retry_on_failure(max_retries=3, delay=3.0)
    def get_history(self, code: str, days: int) -> pd.DataFrame:
        """
        Get historical daily data from AkShare
        
        Args:
            code: Stock code (e.g., "600519")
            days: Number of days to fetch
            
        Returns:
            DataFrame with columns: open, high, low, close, volume, amount
        """
        if not self._ak:
            raise RuntimeError("AkShare not available")
        
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")
            
            logger.debug(f"AkShare: Fetching {code} from {start_date} to {end_date}")
            
            df = self._ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # Forward adjusted prices
            )
            
            if df is None or df.empty:
                raise ValueError(f"No data returned for {code}")
            
            # Standardize column names
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
            
            # Set date index
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Ensure numeric types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Clean invalid data
            df = df.dropna(subset=['close', 'volume'])
            df = df[df['volume'] > 0]
            df = df[df['high'] >= df['low']]
            
            if len(df) < 10:
                raise ValueError(f"Insufficient data for {code}: only {len(df)} valid rows")
            
            self._record_success()
            logger.info(f"AkShare: Got {len(df)} bars for {code}")
            
            return df.tail(days)
            
        except Exception as e:
            self._record_error(str(e))
            raise
    
    @retry_on_failure(max_retries=2, delay=1.5)
    def get_realtime(self, code: str) -> Optional[Quote]:
        """Get real-time quote from AkShare"""
        if not self._ak:
            return None
        
        try:
            df = self._ak.stock_zh_a_spot_em()
            row = df[df['代码'] == code]
            
            if row.empty:
                return None
            
            r = row.iloc[0]
            
            quote = Quote(
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
                source=self.name
            )
            
            self._record_success()
            return quote
            
        except Exception as e:
            self._record_error(str(e))
            return None
    
    def get_stock_list(self) -> pd.DataFrame:
        """Get list of all A-share stocks"""
        if not self._ak:
            return pd.DataFrame()
        
        try:
            df = self._ak.stock_zh_a_spot_em()
            return df[['代码', '名称']].rename(columns={'代码': 'code', '名称': 'name'})
        except:
            return pd.DataFrame()


# ============================================================
# Yahoo Finance Data Source (Global Fallback)
# ============================================================

class YahooFinanceSource(DataSource):
    """
    Yahoo Finance - Fallback source
    Works globally, good for testing and backup
    Requires: pip install yfinance
    """
    name = "yahoo"
    priority = 2
    
    # Map Chinese stock codes to Yahoo symbols
    EXCHANGE_SUFFIX = {
        '6': '.SS',  # Shanghai Stock Exchange
        '0': '.SZ',  # Shenzhen Stock Exchange
        '3': '.SZ',  # Shenzhen ChiNext
    }
    
    def __init__(self):
        super().__init__()
        self._yf = None
        self._check_availability()
    
    def _check_availability(self):
        """Check if yfinance is installed"""
        try:
            import yfinance as yf
            self._yf = yf
            self.status.available = True
            logger.info("Yahoo Finance data source initialized")
        except ImportError:
            self.status.available = False
            logger.warning("yfinance not installed. Run: pip install yfinance")
    
    def _to_yahoo_symbol(self, code: str) -> str:
        """Convert Chinese stock code to Yahoo Finance symbol"""
        code = str(code).zfill(6)
        suffix = self.EXCHANGE_SUFFIX.get(code[0], '.SS')
        return f"{code}{suffix}"
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def get_history(self, code: str, days: int) -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        if not self._yf:
            raise RuntimeError("yfinance not available")
        
        try:
            symbol = self._to_yahoo_symbol(code)
            ticker = self._yf.Ticker(symbol)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=int(days * 1.5))
            
            logger.debug(f"Yahoo: Fetching {symbol}")
            
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data from Yahoo for {symbol}")
            
            # Standardize columns
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            df.index.name = 'date'
            
            # Select and calculate columns
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            df['amount'] = df['close'] * df['volume']
            
            # Clean data
            df = df.dropna()
            df = df[df['volume'] > 0]
            
            if len(df) < 10:
                raise ValueError(f"Insufficient Yahoo data for {code}")
            
            self._record_success()
            logger.info(f"Yahoo: Got {len(df)} bars for {code}")
            
            return df.tail(days)
            
        except Exception as e:
            self._record_error(str(e))
            raise
    
    @retry_on_failure(max_retries=2, delay=1.0)
    def get_realtime(self, code: str) -> Optional[Quote]:
        """Get real-time quote from Yahoo Finance"""
        if not self._yf:
            return None
        
        try:
            symbol = self._to_yahoo_symbol(code)
            ticker = self._yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            price = float(info.get('currentPrice') or info.get('regularMarketPrice') or 0)
            
            if price <= 0:
                return None
            
            quote = Quote(
                code=code,
                name=info.get('shortName', code),
                price=price,
                open=float(info.get('open') or info.get('regularMarketOpen') or 0),
                high=float(info.get('dayHigh') or info.get('regularMarketDayHigh') or 0),
                low=float(info.get('dayLow') or info.get('regularMarketDayLow') or 0),
                volume=int(info.get('volume') or info.get('regularMarketVolume') or 0),
                amount=0,
                change_pct=float(info.get('regularMarketChangePercent') or 0),
                timestamp=datetime.now(),
                source=self.name
            )
            
            self._record_success()
            return quote
            
        except Exception as e:
            self._record_error(str(e))
            return None


# ============================================================
# Main Data Fetcher
# ============================================================

class DataFetcher:
    """
    Robust Data Fetcher with Multiple Sources
    
    Features:
    - Multiple data sources with automatic fallback
    - Memory and disk caching
    - Rate limiting to avoid API blocks
    - Automatic retry on failure
    - Data validation and cleaning
    
    Usage:
        fetcher = DataFetcher()
        df = fetcher.get_history("600519", days=500)
        quote = fetcher.get_realtime("600519")
    """
    
    def __init__(self):
        # Initialize data sources in priority order
        self._sources: List[DataSource] = []
        self._init_sources()
        
        # Memory cache
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl_hours = 4
        
        # Rate limiting
        self._rate_limit = 0.5  # seconds between requests
        self._last_request = 0
        
        logger.info(f"DataFetcher initialized with {len(self._sources)} sources")
    
    def _init_sources(self):
        """Initialize all available data sources"""
        # Add sources in priority order
        sources_to_try = [
            AkShareSource,
            YahooFinanceSource,
        ]
        
        for source_class in sources_to_try:
            try:
                source = source_class()
                if source.is_available():
                    self._sources.append(source)
                    logger.info(f"  ✓ {source.name} available")
                else:
                    logger.warning(f"  ✗ {source.name} not available")
            except Exception as e:
                logger.warning(f"  ✗ Failed to init {source_class.name}: {e}")
        
        if not self._sources:
            logger.error("No data sources available! Install akshare or yfinance.")
    
    def _wait_rate_limit(self):
        """Respect rate limits between requests"""
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_request = time.time()
    
    def _clean_code(self, code: str) -> str:
        """Clean and standardize stock code"""
        code = str(code).strip()
        code = code.replace('sh', '').replace('sz', '')
        code = code.replace('.SS', '').replace('.SZ', '')
        code = code.replace('.', '')
        return code.zfill(6)
    
    def get_history(self,
                    code: str,
                    days: int = 500,
                    use_cache: bool = True) -> pd.DataFrame:
        """
        Get historical OHLCV data with automatic fallback
        
        Args:
            code: Stock code (e.g., "600519")
            days: Number of trading days to fetch
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns: open, high, low, close, volume, amount
            Empty DataFrame if all sources fail
        """
        code = self._clean_code(code)
        cache_key = f"{code}_{days}"
        
        # Check memory cache
        if use_cache and cache_key in self._cache:
            cache_age = (datetime.now() - self._cache_time.get(cache_key, datetime.min))
            if cache_age.total_seconds() / 3600 < self._cache_ttl_hours:
                logger.debug(f"Memory cache hit for {code}")
                return self._cache[cache_key].copy()
        
        # Check disk cache
        disk_df = self._load_disk_cache(code, days)
        if not disk_df.empty and use_cache:
            cache_age = self._get_cache_age(code)
            if cache_age and cache_age.total_seconds() / 3600 < self._cache_ttl_hours:
                logger.debug(f"Disk cache hit for {code}")
                self._cache[cache_key] = disk_df.copy()
                self._cache_time[cache_key] = datetime.now()
                return disk_df.copy()
        
        # Try each source in order
        self._wait_rate_limit()
        
        for source in self._sources:
            if not source.is_available():
                continue
            
            try:
                logger.info(f"Trying {source.name} for {code}...")
                df = source.get_history(code, days)
                
                if not df.empty and len(df) >= 20:
                    logger.info(f"Got {len(df)} bars from {source.name}")
                    
                    # Cache the result
                    self._cache[cache_key] = df.copy()
                    self._cache_time[cache_key] = datetime.now()
                    self._save_disk_cache(code, df)
                    
                    return df
                else:
                    logger.warning(f"{source.name} returned insufficient data for {code}")
                    
            except Exception as e:
                logger.warning(f"{source.name} failed for {code}: {e}")
                continue
        
        # All sources failed - try stale cache
        if not disk_df.empty:
            logger.warning(f"All sources failed for {code}, using stale cache")
            return disk_df
        
        logger.error(f"All data sources failed for {code}")
        return pd.DataFrame()
    
    def get_realtime(self, code: str) -> Optional[Quote]:
        """
        Get real-time quote with automatic fallback
        
        Args:
            code: Stock code
            
        Returns:
            Quote object or None if all sources fail
        """
        code = self._clean_code(code)
        self._wait_rate_limit()
        
        for source in self._sources:
            if not source.is_available():
                continue
            
            try:
                quote = source.get_realtime(code)
                if quote and quote.price > 0:
                    return quote
            except Exception as e:
                logger.debug(f"{source.name} realtime failed: {e}")
                continue
        
        # Fallback: use last historical close
        df = self.get_history(code, days=5, use_cache=True)
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
                source="cache"
            )
        
        return None
    
    def get_stock_name(self, code: str) -> str:
        """Get stock name by code"""
        quote = self.get_realtime(code)
        return quote.name if quote else code
    
    def get_multiple(self, codes: List[str], days: int = 500) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks
        
        Args:
            codes: List of stock codes
            days: Days of history to fetch
            
        Returns:
            Dict mapping code to DataFrame
        """
        result = {}
        total = len(codes)
        
        for i, code in enumerate(codes):
            logger.info(f"Fetching {code} ({i+1}/{total})")
            df = self.get_history(code, days)
            if not df.empty:
                result[code] = df
        
        logger.info(f"Successfully fetched {len(result)}/{total} stocks")
        return result
    
    def _save_disk_cache(self, code: str, df: pd.DataFrame):
        """Save data to disk cache"""
        try:
            path = CONFIG.DATA_DIR / f"{code}.pkl"
            cache_data = {
                'data': df,
                'timestamp': datetime.now(),
                'source': 'fetcher'
            }
            with open(path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.debug(f"Cached {code} to disk")
        except Exception as e:
            logger.warning(f"Failed to cache {code}: {e}")
    
    def _load_disk_cache(self, code: str, days: int) -> pd.DataFrame:
        """Load data from disk cache"""
        try:
            path = CONFIG.DATA_DIR / f"{code}.pkl"
            if not path.exists():
                return pd.DataFrame()
            
            with open(path, 'rb') as f:
                cache_data = pickle.load(f)
            
            if isinstance(cache_data, dict):
                df = cache_data['data']
            else:
                df = cache_data  # Old format compatibility
            
            return df.tail(days)
            
        except Exception as e:
            logger.warning(f"Failed to load cache for {code}: {e}")
            return pd.DataFrame()
    
    def _get_cache_age(self, code: str) -> Optional[timedelta]:
        """Get age of disk cache"""
        try:
            path = CONFIG.DATA_DIR / f"{code}.pkl"
            if not path.exists():
                return None
            
            with open(path, 'rb') as f:
                cache_data = pickle.load(f)
            
            if isinstance(cache_data, dict) and 'timestamp' in cache_data:
                return datetime.now() - cache_data['timestamp']
            
            # Fall back to file modification time
            import os
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            return datetime.now() - mtime
            
        except:
            return None
    
    def get_source_status(self) -> List[DataSourceStatus]:
        """Get status of all data sources"""
        return [source.status for source in self._sources]
    
    def clear_cache(self, code: str = None):
        """Clear cache for specific code or all"""
        if code:
            code = self._clean_code(code)
            # Clear memory cache
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(code)]
            for k in keys_to_remove:
                del self._cache[k]
                if k in self._cache_time:
                    del self._cache_time[k]
            
            # Clear disk cache
            path = CONFIG.DATA_DIR / f"{code}.pkl"
            if path.exists():
                path.unlink()
            
            logger.info(f"Cleared cache for {code}")
        else:
            self._cache.clear()
            self._cache_time.clear()
            
            # Clear all disk cache
            for path in CONFIG.DATA_DIR.glob("*.pkl"):
                try:
                    path.unlink()
                except:
                    pass
            
            logger.info("Cleared all cache")