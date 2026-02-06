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
    consecutive_errors: int = 0
    avg_latency_ms: float = 0.0
    disabled_until: datetime = None


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
            
            if self.status.disabled_until:
                if datetime.now() < self.status.disabled_until:
                    return False
                else:
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
            self.status.consecutive_errors = 0
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
            
            if self.status.consecutive_errors >= 10:
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
        """
        Non-blocking SpotCache refresh:
        - avoids holding main lock during network call
        - returns last-known data if refresh fails
        """
        now = time.time()

        with self._lock:
            if not force_refresh and self._cache is not None and (now - self._cache_time) < self._ttl:
                return self._cache
            ak = self._ak
            cached = self._cache

        if ak is None:
            return cached

        with self._rate_lock:
            with self._lock:
                if not force_refresh and self._cache is not None and (now - self._cache_time) < self._ttl:
                    return self._cache

            try:
                import socket
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(15)
                try:
                    fresh = ak.stock_zh_a_spot_em()
                finally:
                    socket.setdefaulttimeout(old_timeout)

                with self._lock:
                    if fresh is not None and not fresh.empty:
                        self._cache = fresh
                        self._cache_time = now
                    return self._cache
            except Exception:
                with self._lock:
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
            socket.setdefaulttimeout(60)
            
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
        """Instrument-aware history for CN equities with interval support."""
        if not self._ak or not self.is_available():
            return pd.DataFrame()

        if inst.get("market") != "CN" or inst.get("asset") != "EQUITY":
            return pd.DataFrame()

        start = time.time()
        try:
            import socket
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(60)

            try:
                min_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60"}
                if interval in min_map:
                    df = self._ak.stock_zh_a_hist_min_em(
                        symbol=str(inst["symbol"]).zfill(6),
                        period=min_map[interval],
                        adjust="qfq",
                    )

                    if df is None or df.empty:
                        return pd.DataFrame()

                    col_map_candidates = [
                        {"时间": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"},
                        {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"},
                    ]
                    for cmap in col_map_candidates:
                        if set(cmap.keys()).issubset(set(df.columns)):
                            df = df.rename(columns=cmap)
                            break

                    if "date" not in df.columns:
                        df = df.rename(columns={df.columns[0]: "date"})

                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df = df.dropna(subset=["date"]).set_index("date").sort_index()

                    for c in ["open", "high", "low", "close", "volume", "amount"]:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")

                    df = df.dropna(subset=["close"])
                    if "volume" in df.columns:
                        df = df[df["volume"].fillna(0) >= 0]

                    latency = (time.time() - start) * 1000
                    self._record_success(latency)
                    return df.tail(max(50, int(days)))

                period_map = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}
                period = period_map.get(interval, "daily")

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
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "涨跌幅": "change_pct", "换手率": "turnover"
            }
            df = df.rename(columns=column_map)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["close", "volume"])
            df = df[df["volume"] > 0]
            df = df[df["high"] >= df["low"]]

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
        """Instrument-aware yfinance history with interval support."""
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


class TencentQuoteSource(DataSource):
    """Free realtime quotes via Tencent batch endpoint."""
    name = "tencent"
    priority = 1

    def get_realtime_batch(self, codes: List[str]) -> Dict[str, Quote]:
        if not self.is_available():
            return {}

        from core.constants import get_exchange

        vendor_symbols = []
        vendor_to_code = {}

        for c in codes:
            code6 = str(c).zfill(6)
            ex = get_exchange(code6)
            if ex == "SSE":
                sym = f"sh{code6}"
            elif ex == "SZSE":
                sym = f"sz{code6}"
            elif ex == "BSE":
                sym = f"bj{code6}"
            else:
                continue
            vendor_symbols.append(sym)
            vendor_to_code[sym] = code6

        if not vendor_symbols:
            return {}

        CHUNK = 120
        out: Dict[str, Quote] = {}
        start_all = time.time()

        try:
            for i in range(0, len(vendor_symbols), CHUNK):
                chunk = vendor_symbols[i:i + CHUNK]
                url = "https://qt.gtimg.cn/q=" + ",".join(chunk)

                resp = self._session.get(url, timeout=5)
                text = resp.text

                for line in text.splitlines():
                    if "~" not in line or "=" not in line:
                        continue
                    try:
                        left, right = line.split("=", 1)
                        vendor_sym = left.strip().replace("v_", "")
                        payload = right.strip().strip('";')
                        parts = payload.split("~")
                        if len(parts) < 10:
                            continue

                        code6 = vendor_to_code.get(vendor_sym)
                        if not code6:
                            continue

                        name = parts[1]
                        price = float(parts[3] or 0)
                        prev_close = float(parts[4] or 0)
                        open_px = float(parts[5] or 0)
                        volume = int(float(parts[6] or 0))

                        if price <= 0:
                            continue

                        chg = price - prev_close
                        chg_pct = (chg / prev_close * 100) if prev_close > 0 else 0.0

                        out[code6] = Quote(
                            code=code6,
                            name=name,
                            price=price,
                            open=open_px,
                            high=price,
                            low=price,
                            close=prev_close,
                            volume=volume,
                            amount=0.0,
                            change=chg,
                            change_pct=chg_pct,
                            source=self.name,
                            is_delayed=False,
                            latency_ms=0.0,
                        )
                    except Exception:
                        continue

            latency = (time.time() - start_all) * 1000
            self._record_success(latency)
            for q in out.values():
                q.latency_ms = float(latency)

            return out

        except Exception as e:
            self._record_error(str(e))
            return {}

    def get_realtime(self, code: str) -> Optional[Quote]:
        res = self.get_realtime_batch([code])
        return res.get(str(code).zfill(6))

    def get_history(self, code: str, days: int) -> pd.DataFrame:
        """
        TencentQuoteSource is realtime-only. Return empty DF for history.
        This prevents warning spam and wasted retries in history fetch loops.
        """
        import pandas as pd
        return pd.DataFrame()

    def get_history_instrument(self, inst: dict, days: int, interval: str = "1d") -> pd.DataFrame:
        """
        Realtime-only source: no history.
        """
        import pandas as pd
        return pd.DataFrame()


class DataFetcher:
    """High-performance data fetcher with multi-source support"""
    
    def __init__(self):
        self._sources: List[DataSource] = []
        self._cache = get_cache()
        self._db = get_database()
        self._rate_limiter = threading.Semaphore(CONFIG.data.parallel_downloads)
        self._request_times: Dict[str, float] = {}
        self._min_interval = 0.5

        # last-good quote store (uptime)
        self._last_good_quotes: Dict[str, Quote] = {}
        self._last_good_lock = threading.RLock()

        # micro-caches (latency)
        self._rt_batch_microcache = {"ts": 0.0, "key": None, "data": {}}
        self._rt_single_microcache: Dict[str, Dict[str, object]] = {}

        self._init_sources()
        self._rate_lock = threading.Lock()
    
    def _init_sources(self):
        """Initialize data sources."""
        self._sources = []

        # Local DB source
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

        for source in [AkShareSource(), TencentQuoteSource(), YahooSource()]:
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

    def get_realtime_batch(self, codes: List[str]) -> Dict[str, Quote]:
        """Fast CN realtime quotes for many symbols with uptime fallback."""
        cleaned = [self.clean_code(c) for c in codes]
        cleaned = [c for c in cleaned if c]
        if not cleaned:
            return {}

        offline = str(os.environ.get("TRADING_OFFLINE", "0")).lower() in ("1", "true", "yes")
        if offline:
            return {}

        now = time.time()
        key = ",".join(cleaned)

        # micro-cache 250ms
        try:
            mc = self._rt_batch_microcache
            if mc["key"] == key and (now - float(mc["ts"])) < 0.25:
                data = mc["data"]
                if isinstance(data, dict) and data:
                    return data
        except Exception:
            pass

        result: Dict[str, Quote] = {}

        # 1) batch sources (Tencent)
        for source in self._sources:
            if not source.is_available():
                continue
            fn = getattr(source, "get_realtime_batch", None)
            if callable(fn):
                try:
                    out = fn(cleaned)
                    if isinstance(out, dict) and out:
                        result.update(out)
                        break
                except Exception:
                    continue

        # 2) fill missing from SpotCache
        missing = [c for c in cleaned if c not in result]
        if missing:
            try:
                cache = get_spot_cache()
                df = cache.get()
                if df is not None and not df.empty:
                    for c in missing:
                        q = cache.get_quote(c)
                        if q and q.get("price", 0) > 0:
                            result[c] = Quote(
                                code=c,
                                name=q.get("name", ""),
                                price=float(q["price"]),
                                open=float(q.get("open", 0) or 0),
                                high=float(q.get("high", 0) or 0),
                                low=float(q.get("low", 0) or 0),
                                close=float(q.get("close", 0) or 0),
                                volume=int(q.get("volume", 0) or 0),
                                amount=float(q.get("amount", 0) or 0),
                                change=float(q.get("change", 0) or 0),
                                change_pct=float(q.get("change_pct", 0) or 0),
                                source="spot_cache",
                                is_delayed=False,
                                latency_ms=0.0,
                            )
            except Exception:
                pass

        # 3) last-good fallback if everything failed (uptime)
        if not result:
            with self._last_good_lock:
                fb: Dict[str, Quote] = {}
                for c in cleaned:
                    q = self._last_good_quotes.get(c)
                    if q and q.price > 0:
                        age = (datetime.now() - (q.timestamp or datetime.now())).total_seconds()
                        if age <= 3.0:
                            fb[c] = q
                if fb:
                    return fb

        # update last-good
        if result:
            with self._last_good_lock:
                for c, q in result.items():
                    if q and q.price > 0:
                        self._last_good_quotes[c] = q

        # update micro-cache
        try:
            self._rt_batch_microcache["ts"] = now
            self._rt_batch_microcache["key"] = key
            self._rt_batch_microcache["data"] = result
        except Exception:
            pass

        return result

    def _fetch_from_sources_instrument(self, inst: dict, days: int, interval: str = "1d") -> pd.DataFrame:
        """Fetch using any source that supports get_history_instrument."""
        with self._rate_limiter:
            for source in self._sources:
                if not source.is_available():
                    continue
                try:
                    self._rate_limit(source.name)

                    fn = getattr(source, "get_history_instrument", None)
                    if callable(fn):
                        df = fn(inst, days=days, interval=interval)
                    else:
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

        prefixes = ("sh.", "sz.", "bj.", "SH.", "SZ.", "BJ.", "sh", "sz", "bj", "SH", "SZ", "BJ")
        for p in prefixes:
            if s.startswith(p):
                s = s[len(p):]
                break

        suffixes = (".SS", ".SZ", ".BJ", ".ss", ".sz", ".bj")
        for suf in suffixes:
            if s.endswith(suf):
                s = s[:-len(suf)]
                break

        digits = "".join(ch for ch in s if ch.isdigit())
        return digits.zfill(6) if digits else ""
    
    def get_history(
        self,
        code: str,
        days: int = 500,
        bars: int = None,
        use_cache: bool = True,
        update_db: bool = True,
        instrument: dict = None,
        interval: str = "1d",
        max_age_hours: float = None,
    ) -> pd.DataFrame:
        """
        Key change:
        - For intraday (e.g. 1m): read from local intraday DB FIRST.
        - If insufficient, fetch online, merge, store, return.
        This is the main way to get intraday history quality >=8.5 without paid feeds.
        """
        from core.instruments import parse_instrument, instrument_key

        inst = instrument or parse_instrument(code)
        key = instrument_key(inst)
        interval = str(interval).lower()

        offline = str(os.environ.get("TRADING_OFFLINE", "0")).lower() in ("1", "true", "yes")

        count = int(bars if bars is not None else days)
        count = max(1, count)

        # TTL (intraday should be short)
        if max_age_hours is not None:
            ttl = float(max_age_hours)
        else:
            ttl = float(CONFIG.data.cache_ttl_hours) if interval == "1d" else min(float(CONFIG.data.cache_ttl_hours), 1.0 / 120.0)

        cache_key = f"history:{key}:{interval}:{count}"

        def _clean(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame()
            out = df.copy()
            if not isinstance(out.index, pd.DatetimeIndex):
                out.index = pd.to_datetime(out.index, errors="coerce")
            out = out[~out.index.duplicated(keep="last")].sort_index()

            for c in ("open", "high", "low", "close", "volume", "amount"):
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce")

            if "close" in out.columns:
                out = out.dropna(subset=["close"])
                out = out[out["close"] > 0]
            if "volume" in out.columns:
                out = out[out["volume"].fillna(0) >= 0]
            if "high" in out.columns and "low" in out.columns:
                out = out[out["high"].fillna(0) >= out["low"].fillna(0)]

            if "amount" not in out.columns and "close" in out.columns and "volume" in out.columns:
                out["amount"] = out["close"] * out["volume"]

            return out

        # cache
        if use_cache:
            cached_df = self._cache.get(cache_key, ttl)
            if isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
                cached_df = _clean(cached_df)
                if len(cached_df) >= min(count, 100):
                    return cached_df.tail(count)

        # -------- NEW: intraday DB first --------
        if interval != "1d" and inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            try:
                code6 = str(inst["symbol"]).zfill(6)
                db_df = self._db.get_intraday_bars(code6, interval=interval, limit=count)
                db_df = _clean(db_df)
                if not db_df.empty and len(db_df) >= int(0.8 * count):
                    out = db_df.tail(count)
                    self._cache.set(cache_key, out)
                    return out
            except Exception:
                pass

        # daily CN can use daily DB
        if interval == "1d" and inst["market"] == "CN" and inst["asset"] == "EQUITY":
            db_df = self._db.get_bars(inst["symbol"])
            if db_df is not None and not db_df.empty and len(db_df) >= count:
                out = _clean(db_df).tail(count)
                self._cache.set(cache_key, out)
                return out

        if offline:
            return pd.DataFrame()

        # fetch online
        df = self._fetch_from_sources_instrument(inst, days=count, interval=interval)
        df = _clean(df)
        if df.empty:
            return pd.DataFrame()

        out = df.tail(count)
        self._cache.set(cache_key, out)

        # store intraday into DB for future quality/uptime
        if interval != "1d" and inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            try:
                self._db.upsert_intraday_bars(str(inst["symbol"]).zfill(6), interval, out)
            except Exception:
                pass

        # store daily into DB
        if update_db and interval == "1d" and inst["market"] == "CN" and inst["asset"] == "EQUITY":
            try:
                self._db.upsert_bars(inst["symbol"], out)
            except Exception:
                pass

        return out
    
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
        Lowest-latency single quote.

        Improvements:
        - Use batch path first (Tencent) for CN
        - 250ms microcache
        - last-good fallback if network fails
        """
        from core.instruments import parse_instrument
        inst = instrument or parse_instrument(code)

        offline = str(os.environ.get("TRADING_OFFLINE", "0")).lower() in ("1", "true", "yes")
        if offline:
            return None

        # CN -> try batch route first (fastest)
        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            code6 = str(inst.get("symbol") or "").zfill(6)
            if code6:
                now = time.time()

                # microcache
                try:
                    rec = self._rt_single_microcache.get(code6)
                    if rec and (now - float(rec["ts"])) < 0.25:
                        return rec["q"]
                except Exception:
                    pass

                try:
                    out = self.get_realtime_batch([code6])
                    q = out.get(code6)
                    if q and q.price > 0:
                        self._rt_single_microcache[code6] = {"ts": now, "q": q}
                        return q
                except Exception:
                    pass

                # last-good fallback
                with self._last_good_lock:
                    q = self._last_good_quotes.get(code6)
                    if q and q.price > 0:
                        age = (datetime.now() - (q.timestamp or datetime.now())).total_seconds()
                        if age <= 3.0:
                            return q

        # fallback: try sources one-by-one
        candidates: List[Quote] = []
        with self._rate_limiter:
            for source in self._sources:
                if not source.is_available():
                    continue
                try:
                    fn = getattr(source, "get_realtime_instrument", None)
                    if callable(fn):
                        q = fn(inst)
                    else:
                        q = source.get_realtime(inst.get("symbol", ""))
                    if q and q.price and q.price > 0:
                        candidates.append(q)
                except Exception:
                    continue

        if not candidates:
            return None

        prices = np.array([c.price for c in candidates], dtype=float)
        med = float(np.median(prices))
        good = [c for c in candidates if abs(c.price - med) / max(med, 1e-8) < 0.01]
        pool = good if good else candidates
        pool.sort(key=lambda q: (q.is_delayed, q.latency_ms))
        best = pool[0]

        # update last-good
        try:
            if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
                code6 = str(inst.get("symbol") or "").zfill(6)
                with self._last_good_lock:
                    self._last_good_quotes[code6] = best
        except Exception:
            pass

        return best

    def get_multiple_parallel(
        self,
        codes: List[str],
        days: int = 500,
        callback: Callable[[str, int, int], None] = None,
        max_workers: int = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple stocks in parallel"""
        results = {}
        total = len(codes)
        completed = 0
        lock = threading.Lock()
        
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
        
        workers = min(max_workers or 5, 5)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch_one, c): c for c in codes}
            
            for future in as_completed(futures):
                code = futures[future]
                try:
                    code, df = future.result(timeout=120)
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
        """Reset all source error counts"""
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