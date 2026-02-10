# data/fetcher.py
"""
High-Performance Data Fetcher with Automatic Network Detection

Automatically detects network environment:
- China direct (Astrill OFF): Uses AkShare + Tencent
- VPN active (Astrill ON): Uses Yahoo Finance + Tencent

Network detection is cached and re-checked every 2 minutes.

FIXES APPLIED:
- Issue 16: get_fetcher() is thread-safe with double-checked locking
"""
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import os

import pandas as pd
import numpy as np
import requests

from config.settings import CONFIG
from data.cache import get_cache
from data.database import get_database
from core.exceptions import DataFetchError, DataSourceUnavailableError
from utils.logger import get_logger

log = get_logger(__name__)


# Maximum calendar days each interval can fetch (API limits)
INTERVAL_MAX_DAYS = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "1h": 730,
    "1d": 10000,
    "1wk": 10000,
    "1mo": 10000,
}

# Approximate bars per trading day for each interval
BARS_PER_DAY = {
    "1m": 240,
    "2m": 120,
    "5m": 48,
    "15m": 16,
    "30m": 8,
    "60m": 4,
    "1h": 4,
    "1d": 1,
    "1wk": 0.2,
    "1mo": 0.05,
}


def bars_to_days(bars: int, interval: str) -> int:
    """Convert bar count to calendar days needed, respecting API limits."""
    interval = str(interval).lower()
    bpd = BARS_PER_DAY.get(interval, 1)
    if bpd <= 0:
        bpd = 1
    trading_days = max(1, int(bars / bpd))
    calendar_days = int(trading_days * 1.5) + 2
    max_days = INTERVAL_MAX_DAYS.get(interval, 10000)
    return min(calendar_days, max_days)


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
    """Abstract data source with error tracking"""
    name: str = "base"
    priority: int = 0

    needs_china_direct: bool = False
    needs_vpn: bool = False

    def __init__(self):
        self.status = DataSourceStatus(name=self.name)
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self._latencies: List[float] = []
        self._lock = threading.Lock()

    def is_available(self) -> bool:
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

    def is_suitable_for_network(self) -> bool:
        """Check if this source works in the current network environment."""
        from core.network import get_network_env
        env = get_network_env()

        if self.needs_china_direct and not env.is_china_direct:
            return False
        if self.needs_vpn and not env.is_vpn_active:
            return False
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
            if self.status.consecutive_errors >= 8:
                cooldown = min(30 + self.status.consecutive_errors * 3, 120)
                self.status.disabled_until = datetime.now() + timedelta(seconds=cooldown)
                log.warning(f"Data source {self.name} disabled for {cooldown}s "
                           f"({self.status.consecutive_errors} consecutive errors)")


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
        now = time.time()
        with self._lock:
            if not force_refresh and self._cache is not None and (now - self._cache_time) < self._ttl:
                return self._cache
            ak = self._ak
            cached = self._cache

        if ak is None:
            return cached

        from core.network import get_network_env
        env = get_network_env()
        if not env.eastmoney_ok:
            return cached

        with self._rate_lock:
            with self._lock:
                if not force_refresh and self._cache is not None and (now - self._cache_time) < self._ttl:
                    return self._cache
            try:
                import socket
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(8)
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
        def _to_float(x, default=0.0) -> float:
            try:
                if x is None or pd.isna(x) or (isinstance(x, float) and np.isnan(x)):
                    return float(default)
                return float(x)
            except Exception:
                return float(default)

        def _to_int(x, default=0) -> int:
            try:
                if x is None or pd.isna(x) or (isinstance(x, float) and np.isnan(x)):
                    return int(default)
                return int(float(x))
            except Exception:
                return int(default)

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
                'name': str(r.get('名称', '') or ''),
                'price': _to_float(r.get('最新价', 0)),
                'open': _to_float(r.get('今开', 0)),
                'high': _to_float(r.get('最高', 0)),
                'low': _to_float(r.get('最低', 0)),
                'close': _to_float(r.get('昨收', 0)),
                'volume': _to_int(r.get('成交量', 0)),
                'amount': _to_float(r.get('成交额', 0)),
                'change': _to_float(r.get('涨跌额', 0)),
                'change_pct': _to_float(r.get('涨跌幅', 0)),
            }
        except Exception:
            return None


_spot_cache: Optional[SpotCache] = None
_spot_cache_lock = threading.Lock()


def get_spot_cache() -> SpotCache:
    global _spot_cache
    if _spot_cache is None:
        with _spot_cache_lock:
            if _spot_cache is None:
                _spot_cache = SpotCache(ttl_seconds=30.0)
    return _spot_cache


class AkShareSource(DataSource):
    """AkShare data source - Works ONLY on China direct IP"""
    name = "akshare"
    priority = 1
    needs_china_direct = True

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

    def is_available(self) -> bool:
        if not self._ak:
            return False
        if not self.is_suitable_for_network():
            return False
        return super().is_available()

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
            socket.setdefaulttimeout(15)
            try:
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")
                df = self._ak.stock_zh_a_hist(
                    symbol=code, period="daily",
                    start_date=start_date, end_date=end_date, adjust="qfq"
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
                code=code, name=data['name'], price=data['price'],
                open=data['open'], high=data['high'], low=data['low'],
                close=data['close'], volume=data['volume'], amount=data['amount'],
                change=data['change'], change_pct=data['change_pct'], source=self.name
            )
        except Exception as e:
            self._record_error(str(e))
            return None

    def get_history_instrument(self, inst: dict, days: int, interval: str = "1d") -> pd.DataFrame:
        if not self._ak or not self.is_available():
            return pd.DataFrame()
        if inst.get("market") != "CN" or inst.get("asset") != "EQUITY":
            return pd.DataFrame()

        start = time.time()
        try:
            import socket
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(15)
            try:
                min_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60"}
                if interval in min_map:
                    time.sleep(0.5)
                    df = self._ak.stock_zh_a_hist_min_em(
                        symbol=str(inst["symbol"]).zfill(6),
                        period=min_map[interval], adjust="qfq",
                    )
                    if df is None or df.empty:
                        return pd.DataFrame()

                    col_map_candidates = [
                        {"时间": "date", "开盘": "open", "收盘": "close", "最高": "high",
                         "最低": "low", "成交量": "volume", "成交额": "amount"},
                        {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                         "最低": "low", "成交量": "volume", "成交额": "amount"},
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
                    return df

                period_map = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}
                period = period_map.get(interval, "daily")
                end_date = datetime.now().strftime("%Y%m%d")
                max_cal_days = INTERVAL_MAX_DAYS.get(interval, 10000)
                cal_days = min(int(days * 2.2), max_cal_days)
                start_date = (datetime.now() - timedelta(days=cal_days)).strftime("%Y%m%d")

                df = self._ak.stock_zh_a_hist(
                    symbol=inst["symbol"], period=period,
                    start_date=start_date, end_date=end_date, adjust="qfq"
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
        if not self._ak or not self.is_available():
            return pd.DataFrame()
        try:
            return self._ak.stock_zh_a_spot_em()
        except Exception:
            return pd.DataFrame()


class YahooSource(DataSource):
    """Yahoo Finance - Works ONLY through VPN (foreign IP)"""
    name = "yahoo"
    priority = 1
    needs_vpn = True

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

    def is_available(self) -> bool:
        if not self._yf:
            return False
        if not self.is_suitable_for_network():
            return False
        return super().is_available()

    def _to_yahoo_symbol(self, code: str) -> str:
        code = str(code).zfill(6)
        suffix = self.SUFFIX_MAP.get(code[0], '.SS')
        return f"{code}{suffix}"

    def get_history(self, code: str, days: int) -> pd.DataFrame:
        if not self._yf or not self.is_available():
            raise DataSourceUnavailableError("Yahoo Finance not available")

        start_time = time.time()
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

            latency = (time.time() - start_time) * 1000
            self._record_success(latency)
            return df.tail(days)
        except Exception as e:
            self._record_error(str(e))
            raise

    def get_history_instrument(self, inst: dict, days: int,
                               interval: str = "1d") -> pd.DataFrame:
        if not self._yf or not self.is_available():
            return pd.DataFrame()

        start_time_t = time.time()
        try:
            if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
                code6 = str(inst.get("symbol", "")).zfill(6)
                yahoo_symbol = self._to_yahoo_symbol(code6)
            else:
                yahoo_symbol = inst.get("yahoo") or inst.get("symbol")

            if not yahoo_symbol:
                return pd.DataFrame()

            ticker = self._yf.Ticker(yahoo_symbol)
            max_days = INTERVAL_MAX_DAYS.get(interval, 10000)
            capped_days = min(int(days), max_days)

            yahoo_interval = interval
            if interval == "60m":
                yahoo_interval = "1h"

            if interval in ("1m", "2m", "5m", "15m", "30m", "60m", "1h"):
                period_str = f"{capped_days}d"
                log.debug(f"Yahoo intraday fetch: {yahoo_symbol} "
                         f"interval={yahoo_interval} period={period_str}")
                df = ticker.history(period=period_str, interval=yahoo_interval)
            else:
                end = datetime.now()
                start_date = end - timedelta(days=capped_days)
                df = ticker.history(start=start_date, end=end,
                                   interval=yahoo_interval)

            if df is None or df.empty:
                log.debug(f"Yahoo returned empty for {yahoo_symbol} ({interval})")
                return pd.DataFrame()

            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
            df.index.name = 'date'
            keep = [c for c in ['open', 'high', 'low', 'close', 'volume']
                    if c in df.columns]
            df = df[keep].copy()
            if 'close' in df.columns and 'volume' in df.columns:
                df['amount'] = df['close'] * df['volume']
            df = df.dropna()
            if 'volume' in df.columns:
                df = df[df['volume'] > 0]

            latency = (time.time() - start_time_t) * 1000
            self._record_success(latency)
            log.debug(f"Yahoo OK: {yahoo_symbol} ({interval}): {len(df)} bars")
            return df.tail(days)
        except Exception as e:
            self._record_error(str(e))
            log.debug(f"Yahoo failed for {inst.get('symbol')} ({interval}): {e}")
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
                code=code, name=info.get('shortName', ''),
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
    """Tencent quotes - Works from ANY IP (China or foreign)"""
    name = "tencent"
    priority = 0
    needs_china_direct = False
    needs_vpn = False

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
                            code=code6, name=name, price=price,
                            open=open_px, high=price, low=price,
                            close=prev_close, volume=volume, amount=0.0,
                            change=chg, change_pct=chg_pct,
                            source=self.name, is_delayed=False, latency_ms=0.0,
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
        return pd.DataFrame()

    def get_history_instrument(self, inst: dict, days: int, interval: str = "1d") -> pd.DataFrame:
        return pd.DataFrame()


class DataFetcher:
    """
    High-performance data fetcher with automatic network-aware source selection.
    """

    def __init__(self):
        self._all_sources: List[DataSource] = []
        self._cache = get_cache()
        self._db = get_database()
        self._rate_limiter = threading.Semaphore(CONFIG.data.parallel_downloads)
        self._request_times: Dict[str, float] = {}
        self._min_interval = 0.5
        self._intraday_interval = 1.2

        self._last_good_quotes: Dict[str, Quote] = {}
        self._last_good_lock = threading.RLock()

        self._rt_batch_microcache = {"ts": 0.0, "key": None, "data": {}}
        self._rt_single_microcache: Dict[str, Dict[str, object]] = {}

        self._init_sources()
        self._rate_lock = threading.Lock()

    def _init_sources(self):
        self._all_sources = []

        try:
            class LocalDatabaseSource(DataSource):
                name = "localdb"
                priority = 0
                needs_china_direct = False
                needs_vpn = False

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

            self._all_sources.append(LocalDatabaseSource())
        except Exception:
            pass

        for source_cls in [AkShareSource, TencentQuoteSource, YahooSource]:
            try:
                source = source_cls()
                if source.status.available:
                    self._all_sources.append(source)
                    log.info(f"Data source {source.name} initialized "
                            f"(china_direct={source.needs_china_direct}, "
                            f"vpn={source.needs_vpn})")
            except Exception as e:
                log.warning(f"Failed to init {source_cls.name}: {e}")

        if not self._all_sources:
            log.error("No data sources available!")

    def _get_active_sources(self) -> List[DataSource]:
        active = []
        for s in self._all_sources:
            if s.is_available() and s.is_suitable_for_network():
                active.append(s)
        return sorted(active, key=lambda x: x.priority)

    def _rate_limit(self, source: str, interval: str = "1d"):
        with self._rate_lock:
            now = time.time()
            last = self._request_times.get(source, 0)
            if interval in ("1m", "2m", "5m", "15m", "30m"):
                min_wait = self._intraday_interval
            else:
                min_wait = self._min_interval
            wait = min_wait - (now - last)
            if wait > 0:
                time.sleep(wait)
            self._request_times[source] = time.time()

    def get_realtime_batch(self, codes: List[str]) -> Dict[str, Quote]:
        cleaned = [self.clean_code(c) for c in codes]
        cleaned = [c for c in cleaned if c]
        if not cleaned:
            return {}

        offline = str(os.environ.get("TRADING_OFFLINE", "0")).lower() in ("1", "true", "yes")
        if offline:
            return {}

        # ensure lock exists
        if not hasattr(self, "_rt_cache_lock"):
            self._rt_cache_lock = threading.RLock()

        now = time.time()
        key = ",".join(cleaned)

        # microcache read (locked)
        with self._rt_cache_lock:
            try:
                mc = self._rt_batch_microcache
                if mc["key"] == key and (now - float(mc["ts"])) < 0.25:
                    data = mc["data"]
                    if isinstance(data, dict) and data:
                        return data
            except Exception:
                pass

        result: Dict[str, Quote] = {}

        for source in self._get_active_sources():
            fn = getattr(source, "get_realtime_batch", None)
            if callable(fn):
                try:
                    out = fn(cleaned)
                    if isinstance(out, dict) and out:
                        result.update(out)
                        break
                except Exception:
                    continue

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
                                code=c, name=q.get("name", ""),
                                price=float(q["price"]),
                                open=float(q.get("open", 0) or 0),
                                high=float(q.get("high", 0) or 0),
                                low=float(q.get("low", 0) or 0),
                                close=float(q.get("close", 0) or 0),
                                volume=int(q.get("volume", 0) or 0),
                                amount=float(q.get("amount", 0) or 0),
                                change=float(q.get("change", 0) or 0),
                                change_pct=float(q.get("change_pct", 0) or 0),
                                source="spot_cache", is_delayed=False, latency_ms=0.0,
                            )
            except Exception:
                pass

        if not result:
            with self._last_good_lock:
                for c in cleaned:
                    q = self._last_good_quotes.get(c)
                    if q and q.price > 0:
                        age = (datetime.now() - (q.timestamp or datetime.now())).total_seconds()
                        if age <= 3.0:
                            result[c] = q

        if result:
            with self._last_good_lock:
                for c, q in result.items():
                    if q and q.price > 0:
                        self._last_good_quotes[c] = q

        # microcache write (locked)
        with self._rt_cache_lock:
            try:
                self._rt_batch_microcache["ts"] = now
                self._rt_batch_microcache["key"] = key
                self._rt_batch_microcache["data"] = result
            except Exception:
                pass

        return result

    def _fetch_from_sources_instrument(self, inst: dict, days: int,
                                        interval: str = "1d") -> pd.DataFrame:
        sources = self._get_active_sources()

        if not sources:
            log.warning(f"No active sources available for {inst.get('symbol')} ({interval})")
            return pd.DataFrame()

        source_names = [s.name for s in sources]
        log.debug(f"Active sources for {inst.get('symbol')} ({interval}): {source_names}")

        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            from core.network import get_network_env
            env = get_network_env()

            if env.is_china_direct:
                sources.sort(key=lambda s: (
                    0 if s.name == "akshare" else
                    1 if s.name == "localdb" else 2,
                    s.priority
                ))
            else:
                sources.sort(key=lambda s: (
                    0 if s.name == "yahoo" else
                    1 if s.name == "localdb" else 2,
                    s.priority
                ))

        with self._rate_limiter:
            for source in sources:
                if not source.is_available():
                    log.debug(f"Source {source.name} not available")
                    continue
                if not source.is_suitable_for_network():
                    log.debug(f"Source {source.name} not suitable for current network")
                    continue
                try:
                    self._rate_limit(source.name, interval)
                    fn = getattr(source, "get_history_instrument", None)
                    if callable(fn):
                        df = fn(inst, days=days, interval=interval)
                    else:
                        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
                            df = source.get_history(inst["symbol"], days)
                        else:
                            continue

                    if df is not None and not df.empty:
                        min_required = min(days // 4, 30)
                        if len(df) >= min_required:
                            log.debug(f"Got {len(df)} bars from {source.name} "
                                    f"for {inst.get('symbol')} ({interval})")
                            return df
                        else:
                            log.debug(f"{source.name} returned {len(df)} bars "
                                    f"for {inst.get('symbol')} ({interval}), "
                                    f"need >= {min_required}")
                    else:
                        log.debug(f"{source.name} returned empty for "
                                f"{inst.get('symbol')} ({interval})")
                except Exception as e:
                    log.debug(f"{source.name} failed for "
                            f"{inst.get('symbol')} ({interval}): {e}")
                    continue

        log.warning(f"All sources failed for {inst.get('symbol')} ({interval})")
        return pd.DataFrame()

    @staticmethod
    def clean_code(code: str) -> str:
        if code is None:
            return ""
        s = str(code).strip()
        if not s:
            return ""
        s = s.replace(" ", "").replace("-", "").replace("_", "")
        prefixes = ("sh.", "sz.", "bj.", "SH.", "SZ.", "BJ.",
                    "sh", "sz", "bj", "SH", "SZ", "BJ")
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

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()]
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
        out = out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return out

    def get_history(
        self, code: str, days: int = 500, bars: int = None,
        use_cache: bool = True, update_db: bool = True,
        instrument: dict = None, interval: str = "1d",
        max_age_hours: float = None,
    ) -> pd.DataFrame:
        from core.instruments import parse_instrument, instrument_key

        inst = instrument or parse_instrument(code)
        key = instrument_key(inst)
        interval = str(interval).lower()

        offline = str(os.environ.get("TRADING_OFFLINE", "0")).lower() in ("1", "true", "yes")

        if bars is not None:
            count = max(1, int(bars))
            fetch_days = bars_to_days(count, interval)
        else:
            count = max(1, int(days))
            max_days = INTERVAL_MAX_DAYS.get(interval, 10000)
            fetch_days = min(count, max_days)

        if max_age_hours is not None:
            ttl = float(max_age_hours)
        else:
            ttl = float(CONFIG.data.cache_ttl_hours) if interval == "1d" \
                else min(float(CONFIG.data.cache_ttl_hours), 1.0 / 120.0)

        cache_key = f"history:{key}:{interval}:{count}"

        if use_cache:
            cached_df = self._cache.get(cache_key, ttl)
            if isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
                cached_df = self._clean_dataframe(cached_df)
                if len(cached_df) >= min(count, 100):
                    return cached_df.tail(count)

        # Intraday CN
        if interval != "1d" and inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            db_df = pd.DataFrame()
            try:
                code6 = str(inst["symbol"]).zfill(6)
                db_df = self._clean_dataframe(
                    self._db.get_intraday_bars(code6, interval=interval, limit=count))
            except Exception:
                pass

            if not offline:
                online_df = self._clean_dataframe(
                    self._fetch_from_sources_instrument(inst, days=fetch_days, interval=interval))
                merged = self._clean_dataframe(pd.concat([db_df, online_df], axis=0)) \
                    if (not db_df.empty or not online_df.empty) else pd.DataFrame()
            else:
                merged = db_df

            if merged.empty:
                return pd.DataFrame()
            out = merged.tail(count)
            self._cache.set(cache_key, out)
            try:
                self._db.upsert_intraday_bars(str(inst["symbol"]).zfill(6), interval, out)
            except Exception:
                pass
            return out

        # Daily CN
        if interval == "1d" and inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            db_df = self._clean_dataframe(self._db.get_bars(inst["symbol"], limit=count))
            if len(db_df) >= count:
                out = db_df.tail(count)
                self._cache.set(cache_key, out)
                return out
            if offline:
                return db_df.tail(count) if not db_df.empty else pd.DataFrame()

            online_df = self._clean_dataframe(
                self._fetch_from_sources_instrument(inst, days=fetch_days, interval="1d"))
            merged = self._clean_dataframe(pd.concat([db_df, online_df], axis=0)) \
                if (not db_df.empty or not online_df.empty) else pd.DataFrame()
            if merged.empty:
                return pd.DataFrame()
            out = merged.tail(count)
            self._cache.set(cache_key, out)
            if update_db:
                try:
                    self._db.upsert_bars(inst["symbol"], out)
                except Exception:
                    pass
            return out

        # Non-CN
        if offline:
            return pd.DataFrame()
        df = self._clean_dataframe(
            self._fetch_from_sources_instrument(inst, days=fetch_days, interval=interval))
        if df.empty:
            return pd.DataFrame()
        out = df.tail(count)
        self._cache.set(cache_key, out)
        return out

    def _fetch_from_sources(self, code: str, days: int) -> pd.DataFrame:
        with self._rate_limiter:
            for source in self._get_active_sources():
                try:
                    self._rate_limit(source.name)
                    df = source.get_history(code, days)
                    if not df.empty and len(df) >= min(days // 2, 50):
                        return df
                except Exception as e:
                    log.warning(f"{source.name} failed for {code}: {e}")
        log.error(f"All sources failed for {code}")
        return pd.DataFrame()

    def get_realtime(self, code: str, instrument: dict = None) -> Optional[Quote]:
        from core.instruments import parse_instrument
        inst = instrument or parse_instrument(code)

        offline = str(os.environ.get("TRADING_OFFLINE", "0")).lower() in ("1", "true", "yes")
        if offline:
            return None

        if not hasattr(self, "_rt_cache_lock"):
            self._rt_cache_lock = threading.RLock()

        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            code6 = str(inst.get("symbol") or "").zfill(6)
            if code6:
                now = time.time()

                with self._rt_cache_lock:
                    rec = self._rt_single_microcache.get(code6)
                    if rec and (now - float(rec["ts"])) < 0.25:
                        return rec["q"]

                try:
                    out = self.get_realtime_batch([code6])
                    q = out.get(code6)
                    if q and q.price > 0:
                        with self._rt_cache_lock:
                            self._rt_single_microcache[code6] = {"ts": now, "q": q}
                        return q
                except Exception:
                    pass

                with self._last_good_lock:
                    q = self._last_good_quotes.get(code6)
                    if q and q.price > 0:
                        age = (datetime.now() - (q.timestamp or datetime.now())).total_seconds()
                        if age <= 3.0:
                            return q

        candidates: List[Quote] = []
        with self._rate_limiter:
            for source in self._get_active_sources():
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

        try:
            if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
                code6 = str(inst.get("symbol") or "").zfill(6)
                with self._last_good_lock:
                    self._last_good_quotes[code6] = best
        except Exception:
            pass

        return best

    def get_multiple_parallel(
        self, codes: List[str], days: int = 500,
        callback: Callable[[str, int, int], None] = None,
        max_workers: int = None, interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        total = len(codes)
        completed = 0
        lock = threading.Lock()

        for source in self._all_sources:
            source.status.consecutive_errors = 0
            source.status.disabled_until = None

        def fetch_one(code: str) -> Tuple[str, pd.DataFrame]:
            try:
                df = self.get_history(code, days, interval=interval)
                return code, df
            except Exception as e:
                log.debug(f"Failed to fetch {code}: {e}")
                return code, pd.DataFrame()

        if interval in ("1m", "5m", "15m", "30m"):
            workers = min(max_workers or 2, 2)
        else:
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
        for source in self._get_active_sources():
            if source.name == "akshare":
                try:
                    df = source.get_all_stocks()
                    if not df.empty:
                        return df
                except Exception as e:
                    log.warning(f"Failed to get stock list: {e}")
        return pd.DataFrame()

    def get_source_status(self) -> List[DataSourceStatus]:
        return [s.status for s in self._all_sources]

    def reset_sources(self):
        from core.network import invalidate_network_cache
        invalidate_network_cache()

        for source in self._all_sources:
            with source._lock:
                source.status.consecutive_errors = 0
                source.status.disabled_until = None
                source.status.available = True
        log.info("All data sources reset, network cache invalidated")


# =============================================================================
# Thread-safe singleton (Issue 16: double-checked locking)
# =============================================================================

_fetcher: Optional[DataFetcher] = None
_fetcher_lock = threading.Lock()


def get_fetcher() -> DataFetcher:
    global _fetcher
    if _fetcher is None:
        with _fetcher_lock:
            if _fetcher is None:
                _fetcher = DataFetcher()
    return _fetcher