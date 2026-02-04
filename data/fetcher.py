# data/fetcher.py - COMPLETE REWRITE
"""
Robust Data Fetcher with Multiple Sources and Fallback
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
import requests
from functools import wraps

from loguru import logger

from config import CONFIG


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retry logic with exponential backoff"""
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
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            logger.error(f"{func.__name__} failed after {max_retries} attempts")
            raise last_exception
        return wrapper
    return decorator


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


class DataSource:
    """Base class for data sources"""
    name: str = "base"
    
    def get_history(self, code: str, days: int) -> pd.DataFrame:
        raise NotImplementedError
    
    def get_realtime(self, code: str) -> Optional[Quote]:
        raise NotImplementedError


class AkShareSource(DataSource):
    """AkShare data source for Chinese A-shares"""
    name = "akshare"
    
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        # Increase timeout
        self._timeout = 30
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def get_history(self, code: str, days: int) -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError:
            logger.warning("akshare not installed")
            return pd.DataFrame()
        
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days * 1.5)).strftime("%Y%m%d")
            
            # Use longer timeout
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
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
            
            df = df.dropna(subset=['close', 'volume'])
            df = df[df['high'] >= df['low']]
            
            return df.tail(days)
            
        except Exception as e:
            logger.error(f"AkShare fetch failed for {code}: {e}")
            raise
    
    @retry_on_failure(max_retries=2, delay=1.0)
    def get_realtime(self, code: str) -> Optional[Quote]:
        try:
            import akshare as ak
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
            logger.warning(f"Realtime quote failed for {code}: {e}")
            return None


class YahooFinanceSource(DataSource):
    """Yahoo Finance fallback for testing (works globally)"""
    name = "yahoo"
    
    # Map Chinese codes to Yahoo symbols
    CHINA_SUFFIX_MAP = {
        '6': '.SS',  # Shanghai
        '0': '.SZ',  # Shenzhen
        '3': '.SZ',  # Shenzhen ChiNext
    }
    
    def _to_yahoo_symbol(self, code: str) -> str:
        """Convert Chinese stock code to Yahoo symbol"""
        code = code.zfill(6)
        suffix = self.CHINA_SUFFIX_MAP.get(code[0], '.SS')
        return f"{code}{suffix}"
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def get_history(self, code: str, days: int) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed. Run: pip install yfinance")
            return pd.DataFrame()
        
        try:
            symbol = self._to_yahoo_symbol(code)
            ticker = yf.Ticker(symbol)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days * 1.5)
            
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return pd.DataFrame()
            
            # Standardize columns
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
            
            return df.tail(days)
            
        except Exception as e:
            logger.error(f"Yahoo fetch failed for {code}: {e}")
            raise
    
    def get_realtime(self, code: str) -> Optional[Quote]:
        try:
            import yfinance as yf
            symbol = self._to_yahoo_symbol(code)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return Quote(
                code=code,
                name=info.get('shortName', code),
                price=float(info.get('currentPrice', info.get('regularMarketPrice', 0))),
                open=float(info.get('open', 0)),
                high=float(info.get('dayHigh', 0)),
                low=float(info.get('dayLow', 0)),
                volume=int(info.get('volume', 0)),
                amount=0,
                change_pct=float(info.get('regularMarketChangePercent', 0)),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"Yahoo realtime failed for {code}: {e}")
            return None


class SyntheticDataSource(DataSource):
    """
    Generate synthetic data for testing when all sources fail
    This ensures the system can always run for development/testing
    """
    name = "synthetic"
    
    # Real stock names for realism
    STOCK_NAMES = {
        '600519': '贵州茅台',
        '601318': '中国平安',
        '600036': '招商银行',
        '000858': '五粮液',
        '002594': 'BYD比亚迪',
        '300750': '宁德时代',
    }
    
    def get_history(self, code: str, days: int) -> pd.DataFrame:
        """Generate realistic synthetic stock data"""
        np.random.seed(int(code) % 10000)  # Reproducible per stock
        
        # Generate dates
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=days, freq='B')  # Business days
        
        # Base price depends on stock code
        base_price = 50 + (int(code) % 1000) / 10
        
        # Generate price series with realistic properties
        # Using geometric Brownian motion
        mu = 0.0002  # Daily drift (small positive)
        sigma = 0.02  # Daily volatility
        
        returns = np.random.normal(mu, sigma, days)
        
        # Add some regime changes
        regime_changes = np.random.choice([0, 1], days, p=[0.98, 0.02])
        returns = returns + regime_changes * np.random.normal(0, 0.03, days)
        
        # Generate close prices
        close = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close
        daily_range = np.abs(np.random.normal(0, 0.015, days))
        high = close * (1 + daily_range / 2)
        low = close * (1 - daily_range / 2)
        
        # Open is previous close with gap
        open_prices = np.roll(close, 1) * (1 + np.random.normal(0, 0.005, days))
        open_prices[0] = close[0] * 0.99
        
        # Volume (higher on big moves)
        base_volume = 1000000 + (int(code) % 100) * 10000
        volume_multiplier = 1 + np.abs(returns) * 20
        volume = (base_volume * volume_multiplier * np.random.uniform(0.5, 1.5, days)).astype(int)
        
        df = pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'amount': close * volume
        }, index=dates)
        
        df.index.name = 'date'
        
        logger.info(f"Generated synthetic data for {code}: {len(df)} bars")
        return df
    
    def get_realtime(self, code: str) -> Optional[Quote]:
        """Generate synthetic real-time quote"""
        df = self.get_history(code, 5)
        if df.empty:
            return None
        
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        
        return Quote(
            code=code,
            name=self.STOCK_NAMES.get(code, f'Stock {code}'),
            price=float(last['close']),
            open=float(last['open']),
            high=float(last['high']),
            low=float(last['low']),
            volume=int(last['volume']),
            amount=float(last['amount']),
            change_pct=float((last['close'] / prev['close'] - 1) * 100),
            timestamp=datetime.now()
        )


class DataFetcher:
    """
    Robust data fetcher with multiple sources and fallback
    
    Priority:
    1. Disk cache (if fresh)
    2. AkShare (primary for Chinese stocks)
    3. Yahoo Finance (fallback)
    4. Synthetic data (last resort for testing)
    """
    
    def __init__(self, use_synthetic_fallback: bool = True):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl_hours = 4
        
        # Initialize sources
        self._sources: List[DataSource] = [
            AkShareSource(),
            YahooFinanceSource(),
        ]
        
        if use_synthetic_fallback:
            self._sources.append(SyntheticDataSource())
        
        self._rate_limit = 0.5
        self._last_request = 0
    
    def _wait_rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_request = time.time()
    
    def _clean_code(self, code: str) -> str:
        return str(code).replace('sh', '').replace('sz', '').replace('.', '').strip().zfill(6)
    
    def get_history(self, 
                    code: str, 
                    days: int = 500,
                    use_cache: bool = True) -> pd.DataFrame:
        """
        Get historical data with fallback sources
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
                self._cache[cache_key] = disk_df
                self._cache_time[cache_key] = datetime.now()
                return disk_df.copy()
        
        # Try each source in order
        self._wait_rate_limit()
        
        for source in self._sources:
            try:
                logger.info(f"Trying {source.name} for {code}...")
                df = source.get_history(code, days)
                
                if not df.empty and len(df) >= 50:  # Minimum data requirement
                    logger.info(f"Got {len(df)} bars from {source.name} for {code}")
                    
                    # Cache the result
                    self._cache[cache_key] = df.copy()
                    self._cache_time[cache_key] = datetime.now()
                    self._save_disk_cache(code, df)
                    
                    return df
                    
            except Exception as e:
                logger.warning(f"{source.name} failed for {code}: {e}")
                continue
        
        # If all sources failed, return empty or cached data
        if not disk_df.empty:
            logger.warning(f"All sources failed for {code}, using stale cache")
            return disk_df
        
        logger.error(f"All data sources failed for {code}")
        return pd.DataFrame()
    
    def get_realtime(self, code: str) -> Optional[Quote]:
        """Get real-time quote with fallback"""
        code = self._clean_code(code)
        self._wait_rate_limit()
        
        for source in self._sources:
            try:
                quote = source.get_realtime(code)
                if quote and quote.price > 0:
                    return quote
            except Exception as e:
                logger.warning(f"{source.name} realtime failed for {code}: {e}")
                continue
        
        # Fallback: use last historical close
        df = self.get_history(code, days=5)
        if not df.empty:
            last = df.iloc[-1]
            return Quote(
                code=code,
                name=f"Stock {code}",
                price=float(last['close']),
                open=float(last['open']),
                high=float(last['high']),
                low=float(last['low']),
                volume=int(last['volume']),
                amount=float(last.get('amount', 0)),
                change_pct=0.0,
                timestamp=datetime.now()
            )
        
        return None
    
    def get_stock_name(self, code: str) -> str:
        quote = self.get_realtime(code)
        return quote.name if quote else code
    
    def _save_disk_cache(self, code: str, df: pd.DataFrame):
        try:
            path = CONFIG.DATA_DIR / f"{code}.pkl"
            cache_data = {
                'data': df,
                'timestamp': datetime.now()
            }
            with open(path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache for {code}: {e}")
    
    def _load_disk_cache(self, code: str, days: int) -> pd.DataFrame:
        try:
            path = CONFIG.DATA_DIR / f"{code}.pkl"
            if path.exists():
                with open(path, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                if isinstance(cache_data, dict):
                    df = cache_data['data']
                else:
                    df = cache_data  # Old format
                
                return df.tail(days)
        except Exception as e:
            logger.warning(f"Failed to load cache for {code}: {e}")
        return pd.DataFrame()
    
    def _get_cache_age(self, code: str) -> Optional[timedelta]:
        try:
            path = CONFIG.DATA_DIR / f"{code}.pkl"
            if path.exists():
                with open(path, 'rb') as f:
                    cache_data = pickle.load(f)
                if isinstance(cache_data, dict) and 'timestamp' in cache_data:
                    return datetime.now() - cache_data['timestamp']
        except:
            pass
        return None
    
    def get_multiple(self, codes: List[str], days: int = 500) -> Dict[str, pd.DataFrame]:
        """Get data for multiple stocks"""
        result = {}
        for code in codes:
            df = self.get_history(code, days)
            if not df.empty:
                result[code] = df
        return result