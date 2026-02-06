# data/fetcher.py
"""
High-Performance Data Fetcher with Robust Error Handling
"""
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import pandas as pd
import numpy as np
import requests
import os

from config.settings import CONFIG
from data.cache import get_cache
from data.database import get_database
from core.exceptions import DataFetchError, DataSourceUnavailableError
from utils.logger import get_logger

log = get_logger(__name__)


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_error
        return wrapper
    return decorator


@dataclass
class Quote:
    code: str
    name: str = ""
    price: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    amount: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    bid_vol: int = 0
    ask_vol: int = 0
    timestamp: datetime = None
    source: str = ""
    is_delayed: bool = True
    latency_ms: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class DataSourceStatus:
    """Data source health status"""
    name: str
    available: bool = True
    last_success: datetime = None
    last_error: str = None
    success_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0  # Track consecutive errors
    avg_latency_ms: float = 0.0
    disabled_until: datetime = None  # Temporary disable


class DataSource:
    """Abstract data source with improved error handling"""
    name: str = "base"
    priority: int = 0
    
    def __init__(self):
        self.status = DataSourceStatus(name=self.name)
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self._latencies: List[float] = []
        self._lock = threading.Lock()
    
    def is_available(self) -> bool:
        """Check if source is available (with temporary disable support)"""
        with self._lock:
            if not self.status.available:
                return False
            
            # Check if temporarily disabled
            if self.status.disabled_until:
                if datetime.now() < self.status.disabled_until:
                    return False
                else:
                    # Re-enable after cooldown
                    self.status.disabled_until = None
                    self.status.consecutive_errors = 0
                    log.info(f"Data source {self.name} re-enabled after cooldown")
            
            return True
    
    def get_history(self, code: str, days: int) -> pd.DataFrame:
        raise NotImplementedError
    
    def get_realtime(self, code: str) -> Optional[Quote]:
        return None
    
    def _record_success(self, latency_ms: float = 0):
        with self._lock:
            self.status.last_success = datetime.now()
            self.status.success_count += 1
            self.status.consecutive_errors = 0  # Reset on success
            self.status.available = True
            self.status.disabled_until = None
            
            if latency_ms > 0:
                self._latencies.append(latency_ms)
                if len(self._latencies) > 100:
                    self._latencies.pop(0)
                self.status.avg_latency_ms = np.mean(self._latencies)
    
    def _record_error(self, error: str):
        with self._lock:
            self.status.last_error = error
            self.status.error_count += 1
            self.status.consecutive_errors += 1
            
            # Temporary disable after 10 consecutive errors (not 5)
            if self.status.consecutive_errors >= 10:
                # Disable for 60 seconds, not permanently
                self.status.disabled_until = datetime.now() + timedelta(seconds=60)
                log.warning(f"Data source {self.name} temporarily disabled for 60s")


class SpotCache:
    """Thread-safe cached spot data with TTL"""

    def __init__(self, ttl_seconds: float = 30.0):
        self._cache: Optional[pd.DataFrame] = None
        self._cache_time: float = 0
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._rate_lock = threading.Lock()
        self._ak = None
        
        try:
            import akshare as ak
            self._ak = ak
        except ImportError:
            pass
    
    def get(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Get spot data, refreshing if stale"""
        with self._lock:
            now = time.time()
            
            if not force_refresh and self._cache is not None:
                if now - self._cache_time < self._ttl:
                    return self._cache
            
            if self._ak is None:
                return self._cache
            
            try:
                import socket
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(60)  # 60 second timeout
                
                try:
                    self._cache = self._ak.stock_zh_a_spot_em()
                    self._cache_time = now
                    log.debug(f"Spot cache refreshed: {len(self._cache)} stocks")
                finally:
                    socket.setdefaulttimeout(old_timeout)
                    
            except Exception as e:
                log.warning(f"Spot cache refresh failed: {e}")
            
            return self._cache
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get single stock quote from cache"""
        symbol = str(symbol).strip()
        for prefix in ['sh', 'sz', 'SH', 'SZ', 'bj', 'BJ']:
            symbol = symbol.replace(prefix, '')
        symbol = symbol.replace('.', '').replace('-', '').zfill(6)
        
        df = self.get()
        if df is None or df.empty:
            return None
        
        try:
            row = df[df['代码'] == symbol]
            if row.empty:
                return None
            
            r = row.iloc[0]
            return {
                'code': symbol,
                'name': str(r.get('名称', '')),
                'price': float(r.get('最新价', 0) or 0),
                'open': float(r.get('今开', 0) or 0),
                'high': float(r.get('最高', 0) or 0),
                'low': float(r.get('最低', 0) or 0),
                'close': float(r.get('昨收', 0) or 0),
                'volume': int(r.get('成交量', 0) or 0),
                'amount': float(r.get('成交额', 0) or 0),
                'change': float(r.get('涨跌额', 0) or 0),
                'change_pct': float(r.get('涨跌幅', 0) or 0),
            }
        except Exception:
            return None


_spot_cache: Optional[SpotCache] = None


def get_spot_cache() -> SpotCache:
    global _spot_cache
    if _spot_cache is None:
        _spot_cache = SpotCache(ttl_seconds=30.0)
    return _spot_cache


class AkShareSource(DataSource):
    """AkShare data source - Primary for A-shares"""
    name = "akshare"
    priority = 1
    
    def __init__(self):
        super().__init__()
        self._ak = None
        self._spot_cache = None
        
        try:
            import akshare as ak
            self._ak = ak
            log.info("AkShare initialized")
        except ImportError:
            self.status.available = False
            log.warning("AkShare not available")
    
    def _get_spot_cache(self) -> SpotCache:
        if self._spot_cache is None:
            self._spot_cache = get_spot_cache()
        return self._spot_cache

    def get_history(self, code: str, days: int) -> pd.DataFrame:
        if not self._ak or not self.is_available():
            raise DataSourceUnavailableError("AkShare not available")
        
        start = time.time()
        
        try:
            import socket
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(60)  # 60 second timeout
            
            try:
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")
                
                df = self._ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
            finally:
                socket.setdefaulttimeout(old_timeout)
            
            if df is None or df.empty:
                raise DataFetchError(f"No data for {code}")
            
            # Standardize columns
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
            df = df[df['high'] >= df['low']]
            
            latency = (time.time() - start) * 1000
            self._record_success(latency)
            
            return df.tail(days)
            
        except Exception as e:
            self._record_error(str(e))
            raise
    
    def get_realtime(self, code: str) -> Optional[Quote]:
        if not self._ak or not self.is_available():
            return None
        
        try:
            data = self._get_spot_cache().get_quote(code)
            
            if data is None or data['price'] <= 0:
                return None
            
            return Quote(
                code=code,
                name=data['name'],
                price=data['price'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                volume=data['volume'],
                amount=data['amount'],
                change=data['change'],
                change_pct=data['change_pct'],
                source=self.name
            )
            
        except Exception as e:
            self._record_error(str(e))
            return None
    
    def get_history_instrument(self, inst: dict, days: int, interval: str = "1d") -> pd.DataFrame:
        """
        Instrument-aware history for CN equities with interval support.
        Supported: 1d, 1wk, 1mo (best effort).
        """
        if not self._ak or not self.is_available():
            return pd.DataFrame()

        if inst.get("market") != "CN" or inst.get("asset") != "EQUITY":
            return pd.DataFrame()

        period_map = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}
        period = period_map.get(interval, "daily")

        start = time.time()
        try:
            import socket
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(60)
            try:
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days=int(days * 2.2))).strftime("%Y%m%d")

                df = self._ak.stock_zh_a_hist(
                    symbol=inst["symbol"],
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
            finally:
                socket.setdefaulttimeout(old_timeout)

            if df is None or df.empty:
                return pd.DataFrame()

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
            df = df[df['high'] >= df['low']]

            latency = (time.time() - start) * 1000
            self._record_success(latency)
            return df.tail(days)

        except Exception as e:
            self._record_error(str(e))
            return pd.DataFrame()

    def get_all_stocks(self) -> pd.DataFrame:
        if not self._ak:
            return pd.DataFrame()
        
        try:
            df = self._ak.stock_zh_a_spot_em()
            return df
        except Exception:
            return pd.DataFrame()


class YahooSource(DataSource):
    """Yahoo Finance - Fallback source"""
    name = "yahoo"
    priority = 2
    
    SUFFIX_MAP = {'6': '.SS', '0': '.SZ', '3': '.SZ'}
    
    def __init__(self):
        super().__init__()
        self._yf = None
        try:
            import yfinance as yf
            self._yf = yf
            log.info("Yahoo Finance initialized")
        except ImportError:
            self.status.available = False
            log.warning("yfinance not available")
    
    def _to_yahoo_symbol(self, code: str) -> str:
        suffix = self.SUFFIX_MAP.get(code[0], '.SS')
        return f"{code}{suffix}"
    
    def get_history(self, code: str, days: int) -> pd.DataFrame:
        if not self._yf or not self.is_available():
            raise DataSourceUnavailableError("Yahoo Finance not available")
        
        start = time.time()
        
        try:
            symbol = self._to_yahoo_symbol(code)
            ticker = self._yf.Ticker(symbol)
            
            end = datetime.now()
            start_date = end - timedelta(days=int(days * 1.5))
            
            df = ticker.history(start=start_date, end=end)
            
            if df.empty:
                raise DataFetchError(f"No data from Yahoo for {code}")
            
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
            
            df.index.name = 'date'
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            df['amount'] = df['close'] * df['volume']
            
            df = df.dropna()
            df = df[df['volume'] > 0]
            
            latency = (time.time() - start) * 1000
            self._record_success(latency)
            
            return df.tail(days)
            
        except Exception as e:
            self._record_error(str(e))
            raise
    
    def get_history_instrument(self, inst: dict, days: int, interval: str = "1d") -> pd.DataFrame:
        """
        Instrument-aware yfinance history with interval support.
        Works best for US/HK, also supports CN via yahoo suffix.
        """
        if not self._yf or not self.is_available():
            return pd.DataFrame()

        start = time.time()
        try:
            yahoo_symbol = inst.get("yahoo") or inst.get("symbol")
            if not yahoo_symbol:
                return pd.DataFrame()

            ticker = self._yf.Ticker(yahoo_symbol)

            end = datetime.now()
            start_date = end - timedelta(days=int(days * 2.2))

            df = ticker.history(start=start_date, end=end, interval=interval)

            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })

            df.index.name = 'date'
            keep = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df[keep].copy()
            if 'close' in df.columns and 'volume' in df.columns:
                df['amount'] = df['close'] * df['volume']

            df = df.dropna()
            df = df[df['volume'] > 0]

            latency = (time.time() - start) * 1000
            self._record_success(latency)
            return df.tail(days)

        except Exception as e:
            self._record_error(str(e))
            return pd.DataFrame()

    def get_realtime(self, code: str) -> Optional[Quote]:
        if not self._yf or not self.is_available():
            return None
        
        try:
            symbol = self._to_yahoo_symbol(code)
            ticker = self._yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'regularMarketPrice' not in info:
                return None
            
            return Quote(
                code=code,
                name=info.get('shortName', ''),
                price=float(info.get('regularMarketPrice', 0)),
                open=float(info.get('regularMarketOpen', 0)),
                high=float(info.get('regularMarketDayHigh', 0)),
                low=float(info.get('regularMarketDayLow', 0)),
                close=float(info.get('previousClose', 0)),
                volume=int(info.get('regularMarketVolume', 0)),
                source=self.name
            )
        except Exception as e:
            self._record_error(str(e))
            return None


class DataFetcher:
    """
    High-performance data fetcher with multi-source support
    """
    
    def __init__(self):
        self._sources: List[DataSource] = []
        self._cache = get_cache()
        self._db = get_database()
        self._rate_limiter = threading.Semaphore(CONFIG.data.parallel_downloads)
        self._request_times: Dict[str, float] = {}
        self._min_interval = 0.5  # Increased to 500ms between requests
        
        self._init_sources()
        self._rate_lock = threading.Lock()
    
    def _init_sources(self):
        """Initialize data sources (always include local DB source)."""
        self._sources = []

        # Always-available local DB source to satisfy tests and offline usage
        try:
            class LocalDatabaseSource(DataSource):
                name = "localdb"
                priority = 0

                def __init__(self):
                    super().__init__()
                    self.status.available = True

                def get_history_instrument(self, inst: dict, days: int, interval: str = "1d") -> pd.DataFrame:
                    if inst.get("market") != "CN" or inst.get("asset") != "EQUITY" or interval != "1d":
                        return pd.DataFrame()
                    try:
                        from data.database import get_database
                        db = get_database()
                        df = db.get_bars(inst["symbol"], limit=days)
                        return df.tail(days) if df is not None else pd.DataFrame()
                    except Exception:
                        return pd.DataFrame()

                def get_history(self, code: str, days: int) -> pd.DataFrame:
                    inst = {"market": "CN", "asset": "EQUITY", "symbol": str(code).zfill(6)}
                    return self.get_history_instrument(inst, days=days, interval="1d")

            self._sources.append(LocalDatabaseSource())
        except Exception:
            pass

        # External sources
        for source in [AkShareSource(), YahooSource()]:
            if source.status.available:
                self._sources.append(source)
                log.info(f"Data source {source.name} available")

        self._sources.sort(key=lambda x: x.priority)

        if not self._sources:
            log.error("No data sources available!")
    
    def _rate_limit(self, source: str):
        with self._rate_lock:
            now = time.time()
            last = self._request_times.get(source, 0)
            wait = self._min_interval - (now - last)
            if wait > 0:
                time.sleep(wait)
            self._request_times[source] = time.time()

    def get_history_instrument(self, inst: dict, days: int, interval: str = "1d") -> pd.DataFrame:
        """Local DB source supports CN EQUITY daily bars from MarketDatabase."""
        if inst.get("market") != "CN" or inst.get("asset") != "EQUITY" or interval != "1d":
            return pd.DataFrame()
        try:
            from data.database import get_database
            db = get_database()
            df = db.get_bars(inst["symbol"], limit=days)
            return df.tail(days) if df is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_history(self, code: str, days: int) -> pd.DataFrame:
        """Legacy interface for Local DB source."""
        inst = {"market": "CN", "asset": "EQUITY", "symbol": str(code).zfill(6)}
        return self.get_history_instrument(inst, days=days, interval="1d")

    def _fetch_from_sources_instrument(self, inst: dict, days: int, interval: str = "1d") -> pd.DataFrame:
        """
        Fetch using any source that supports get_history_instrument(inst,...).
        Falls back to legacy get_history(code,days) for CN equity sources.
        """
        with self._rate_limiter:
            for source in self._sources:
                if not source.is_available():
                    continue
                try:
                    self._rate_limit(source.name)

                    # New-style instrument-aware sources
                    fn = getattr(source, "get_history_instrument", None)
                    if callable(fn):
                        df = fn(inst, days=days, interval=interval)
                    else:
                        # Legacy fallback (CN equity only)
                        if inst["market"] == "CN" and inst["asset"] == "EQUITY":
                            df = source.get_history(inst["symbol"], days)
                        else:
                            continue

                    if df is not None and not df.empty and len(df) >= min(days // 2, 50):
                        return df

                except Exception as e:
                    log.warning(f"{source.name} failed for {inst}: {e}")
                    continue

        return pd.DataFrame()
    
    @staticmethod
    def clean_code(code: str) -> str:
        """Robust normalization for CN A-share codes."""
        if code is None:
            return ""

        s = str(code).strip()
        if not s:
            return ""

        s = s.replace(" ", "").replace("-", "").replace("_", "")

        # strip known prefixes (only at start)
        prefixes = ("sh.", "sz.", "bj.", "SH.", "SZ.", "BJ.", "sh", "sz", "bj", "SH", "SZ", "BJ")
        for p in prefixes:
            if s.startswith(p):
                s = s[len(p):]
                break

        # strip known suffixes (only at end)
        suffixes = (".SS", ".SZ", ".BJ", ".ss", ".sz", ".bj")
        for suf in suffixes:
            if s.endswith(suf):
                s = s[:-len(suf)]
                break

        # keep digits only
        digits = "".join(ch for ch in s if ch.isdigit())
        return digits.zfill(6) if digits else ""
    
    def get_history(
        self,
        code: str,
        days: int = 500,
        use_cache: bool = True,
        update_db: bool = True,
        instrument: dict = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Multi-asset/multi-market history fetch.

        - If instrument is None, it will parse from code (CN equity default).
        - Supports offline mode: set env TRADING_OFFLINE=1 to disable network.
        """
        from core.instruments import parse_instrument, instrument_key

        inst = instrument or parse_instrument(code)
        key = instrument_key(inst)
        offline = str(os.environ.get("TRADING_OFFLINE", "0")).lower() in ("1", "true", "yes")

        cache_key = f"history:{key}:{interval}:{days}"

        # Cache
        if use_cache:
            cached_df = self._cache.get(cache_key, CONFIG.data.cache_ttl_hours)
            if cached_df is not None and len(cached_df) >= min(days, 100):
                return cached_df.tail(days)

        # Local DB only works for CN equities in your current schema (daily_bars uses code)
        if inst["market"] == "CN" and inst["asset"] == "EQUITY":
            db_df = self._db.get_bars(inst["symbol"])
            if not db_df.empty and len(db_df) >= days:
                self._cache.set(cache_key, db_df.tail(days))
                return db_df.tail(days)

        if offline:
            # Offline: never hit network sources
            return pd.DataFrame()

        # Fetch from sources (supports instrument-aware sources)
        df = self._fetch_from_sources_instrument(inst, days=days, interval=interval)

        if df is not None and not df.empty:
            self._cache.set(cache_key, df)
            if update_db and inst["market"] == "CN" and inst["asset"] == "EQUITY":
                self._db.upsert_bars(inst["symbol"], df)

        return df if df is not None else pd.DataFrame()
    
    def _fetch_from_sources(self, code: str, days: int) -> pd.DataFrame:
        """Fetch from online sources with fallback"""
        with self._rate_limiter:
            for source in self._sources:
                if not source.is_available():
                    continue
                
                try:
                    self._rate_limit(source.name)
                    df = source.get_history(code, days)
                    
                    if not df.empty and len(df) >= min(days // 2, 50):
                        log.debug(f"Got {len(df)} bars from {source.name} for {code}")
                        return df
                        
                except Exception as e:
                    log.warning(f"{source.name} failed for {code}: {e}")
                    continue
        
        log.error(f"All sources failed for {code}")
        return pd.DataFrame()
    
    def get_realtime(self, code: str, instrument: dict = None) -> Optional[Quote]:
        """
        Multi-asset realtime quote.
        Offline mode returns last close if available (CN only).
        """
        from core.instruments import parse_instrument

        inst = instrument or parse_instrument(code)
        offline = str(os.environ.get("TRADING_OFFLINE", "0")).lower() in ("1", "true", "yes")

        if not offline:
            with self._rate_limiter:
                for source in self._sources:
                    if not source.is_available():
                        continue
                    try:
                        self._rate_limit(f"{source.name}_rt")

                        fn = getattr(source, "get_realtime_instrument", None)
                        if callable(fn):
                            q = fn(inst)
                        else:
                            if inst["market"] == "CN" and inst["asset"] == "EQUITY":
                                q = source.get_realtime(inst["symbol"])
                            else:
                                q = None

                        if q and q.price > 0:
                            return q
                    except Exception:
                        continue

        # Offline fallback: last historical close (CN equity only)
        if inst["market"] == "CN" and inst["asset"] == "EQUITY":
            df = self.get_history(inst["symbol"], days=5, use_cache=True, update_db=False, instrument=inst)
            if not df.empty:
                last = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else last
                change = float(last["close"] - prev["close"])
                change_pct = (change / float(prev["close"]) * 100) if float(prev["close"]) > 0 else 0.0
                return Quote(
                    code=inst["symbol"],
                    name=f"Stock {inst['symbol']}",
                    price=float(last["close"]),
                    open=float(last["open"]),
                    high=float(last["high"]),
                    low=float(last["low"]),
                    close=float(prev["close"]),
                    volume=int(last["volume"]),
                    amount=float(last.get("amount", 0)),
                    change=change,
                    change_pct=change_pct,
                    source="offline_cache",
                )
        return None

    def get_multiple_parallel(
        self,
        codes: List[str],
        days: int = 500,
        callback: Callable[[str, int, int], None] = None,
        max_workers: int = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple stocks in parallel with better error handling"""
        results = {}
        total = len(codes)
        completed = 0
        lock = threading.Lock()
        
        # Reset source status before bulk fetch
        for source in self._sources:
            source.status.consecutive_errors = 0
            source.status.disabled_until = None
        
        def fetch_one(code: str) -> Tuple[str, pd.DataFrame]:
            try:
                df = self.get_history(code, days)
                return code, df
            except Exception as e:
                log.debug(f"Failed to fetch {code}: {e}")
                return code, pd.DataFrame()
        
        # Use fewer workers to avoid overwhelming the API
        workers = min(max_workers or 5, 5)  # Max 5 workers
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch_one, c): c for c in codes}
            
            for future in as_completed(futures):
                code = futures[future]
                try:
                    code, df = future.result(timeout=120)  # 2 minute timeout
                    if not df.empty and len(df) >= CONFIG.data.min_history_days:
                        results[code] = df
                except Exception as e:
                    log.warning(f"Failed to fetch {code}: {e}")
                
                with lock:
                    completed += 1
                    if callback:
                        callback(code, completed, total)
        
        log.info(f"Parallel fetch: {len(results)}/{total} successful")
        return results
        
    def get_all_stocks(self) -> pd.DataFrame:
        """Get list of all available stocks"""
        for source in self._sources:
            if source.name == "akshare" and source.is_available():
                try:
                    df = source.get_all_stocks()
                    if not df.empty:
                        return df
                except Exception as e:
                    log.warning(f"Failed to get stock list: {e}")
            
        return pd.DataFrame()
    
    def get_source_status(self) -> List[DataSourceStatus]:
        """Get status of all data sources"""
        return [s.status for s in self._sources]
    
    def reset_sources(self):
        """Reset all source error counts - call before bulk operations"""
        for source in self._sources:
            with source._lock:
                source.status.consecutive_errors = 0
                source.status.disabled_until = None
                source.status.available = True
        log.info("All data sources reset")


# Global fetcher instance
_fetcher = None


def get_fetcher() -> DataFetcher:
    """Get global fetcher instance"""
    global _fetcher
    if _fetcher is None:
        _fetcher = DataFetcher()
    return _fetcher