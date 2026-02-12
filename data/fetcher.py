# data/fetcher.py
import math
import os
import socket
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

from config.settings import CONFIG
from data.cache import get_cache
from data.database import get_database
from core.exceptions import DataFetchError, DataSourceUnavailableError
from utils.logger import get_logger
from utils.helpers import to_float, to_int

log = get_logger(__name__)

# Maximum calendar days each interval can fetch (API limits)
INTERVAL_MAX_DAYS: Dict[str, int] = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "1h": 730,
    "1d": 10_000,
    "1wk": 10_000,
    "1mo": 10_000,
}

BARS_PER_DAY: Dict[str, float] = {
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

# Intraday intervals that need tighter rate-limiting
_INTRADAY_INTERVALS = frozenset({"1m", "2m", "5m", "15m", "30m"})

# Micro-cache TTL in seconds
_MICRO_CACHE_TTL: float = 0.25

# Maximum staleness (seconds) for last-good quote fallback
_LAST_GOOD_MAX_AGE: float = 3.0

_TENCENT_CHUNK_SIZE: int = 120

# Default socket timeout for AkShare calls (seconds)
_AKSHARE_SOCKET_TIMEOUT: int = 15

# SpotCache default TTL (seconds)
_SPOT_CACHE_TTL: float = 30.0

def _is_offline() -> bool:
    """Check TRADING_OFFLINE environment variable."""
    return str(os.environ.get("TRADING_OFFLINE", "0")).lower() in ("1", "true", "yes")

def bars_to_days(bars: int, interval: str) -> int:
    """Convert bar count to calendar days needed, respecting API limits."""
    interval = str(interval).lower()
    bpd = BARS_PER_DAY.get(interval, 1.0)
    if bpd <= 0:
        bpd = 1.0
    trading_days = max(1, int(math.ceil(bars / bpd)))
    # ~1.5x multiplier converts trading days → calendar days, +2 for safety
    calendar_days = int(trading_days * 1.5) + 2
    max_days = INTERVAL_MAX_DAYS.get(interval, 10_000)
    return min(calendar_days, max_days)

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error: Optional[Exception] = None
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_error = exc
                    if attempt < max_attempts - 1:
                        log.debug(
                            f"Retry {attempt + 1}/{max_attempts} for "
                            f"{func.__name__}: {exc}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
            raise last_error  # type: ignore[misc]
        return wrapper
    return decorator

@dataclass
class Quote:
    """Real-time quote for a single instrument."""
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
    timestamp: Optional[datetime] = None
    source: str = ""
    is_delayed: bool = True
    latency_ms: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class DataSourceStatus:
    """Health / telemetry for a single data source."""
    name: str
    available: bool = True
    last_success: Optional[datetime] = None
    last_error: Optional[str] = None
    success_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    avg_latency_ms: float = 0.0
    disabled_until: Optional[datetime] = None

class DataSource:
    """Abstract data source with error tracking and circuit-breaker."""

    name: str = "base"
    priority: int = 0
    needs_china_direct: bool = False
    needs_vpn: bool = False

    # Circuit-breaker thresholds
    _CB_ERROR_THRESHOLD: int = 8
    _CB_MIN_COOLDOWN: int = 30
    _CB_MAX_COOLDOWN: int = 120
    _CB_COOLDOWN_INCREMENT: int = 3

    def __init__(self):
        self.status = DataSourceStatus(name=self.name)
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            )
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
                # Cooldown expired → re-enable
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

    def _record_success(self, latency_ms: float = 0.0) -> None:
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
                self.status.avg_latency_ms = float(np.mean(self._latencies))

    def _record_error(self, error: str) -> None:
        with self._lock:
            self.status.last_error = error
            self.status.error_count += 1
            self.status.consecutive_errors += 1
            if self.status.consecutive_errors >= self._CB_ERROR_THRESHOLD:
                cooldown = min(
                    self._CB_MIN_COOLDOWN
                    + self.status.consecutive_errors * self._CB_COOLDOWN_INCREMENT,
                    self._CB_MAX_COOLDOWN,
                )
                self.status.disabled_until = (
                    datetime.now() + timedelta(seconds=cooldown)
                )
                log.warning(
                    f"Data source {self.name} disabled for {cooldown}s "
                    f"({self.status.consecutive_errors} consecutive errors)"
                )

    def get_history(self, code: str, days: int) -> pd.DataFrame:
        raise NotImplementedError

    def get_history_instrument(
        self, inst: dict, days: int, interval: str = "1d"
    ) -> pd.DataFrame:
        raise NotImplementedError

    def get_realtime(self, code: str) -> Optional[Quote]:
        return None

class SpotCache:
    """Thread-safe cached A-share spot data with TTL."""

    def __init__(self, ttl_seconds: float = _SPOT_CACHE_TTL):
        self._cache: Optional[pd.DataFrame] = None
        self._cache_time: float = 0.0
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
        """Return cached spot DataFrame, refreshing if stale."""
        now = time.time()

        with self._lock:
            if (
                not force_refresh
                and self._cache is not None
                and (now - self._cache_time) < self._ttl
            ):
                return self._cache
            stale = self._cache  # keep stale copy as fallback

        if self._ak is None:
            return stale

        from core.network import get_network_env
        env = get_network_env()
        if not env.eastmoney_ok:
            return stale

        with self._rate_lock:
            # Re-check after acquiring rate lock (another thread may have refreshed)
            with self._lock:
                if (
                    not force_refresh
                    and self._cache is not None
                    and (time.time() - self._cache_time) < self._ttl
                ):
                    return self._cache

            try:
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(8)
                try:
                    fresh = self._ak.stock_zh_a_spot_em()
                finally:
                    socket.setdefaulttimeout(old_timeout)

                with self._lock:
                    if fresh is not None and not fresh.empty:
                        self._cache = fresh
                        self._cache_time = time.time()
                    return self._cache

            except Exception:
                with self._lock:
                    return self._cache

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Look up a single stock from the cached spot snapshot."""
        symbol = str(symbol).strip()
        for prefix in ("sh", "sz", "SH", "SZ", "bj", "BJ"):
            if symbol.startswith(prefix):
                symbol = symbol[len(prefix):]
                break
        symbol = symbol.replace(".", "").replace("-", "").zfill(6)

        df = self.get()
        if df is None or df.empty:
            return None

        try:
            code_col = (
                df["代码"]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .str.zfill(6)
            )
            row = df[code_col == symbol]
            if row.empty:
                return None
            r = row.iloc[0]
        except Exception:
            return None

        return {
            "code": symbol,
            "name": str(r.get("名称", "") or ""),
            "price": to_float(r.get("最新价", 0)),
            "open": to_float(r.get("今开", 0)),
            "high": to_float(r.get("最高", 0)),
            "low": to_float(r.get("最低", 0)),
            "close": to_float(r.get("昨收", 0)),
            "volume": to_int(r.get("成交量", 0)),
            "amount": to_float(r.get("成交额", 0)),
            "change": to_float(r.get("涨跌额", 0)),
            "change_pct": to_float(r.get("涨跌幅", 0)),
        }

_spot_cache: Optional[SpotCache] = None
_spot_cache_lock = threading.Lock()

def get_spot_cache() -> SpotCache:
    """Module-level singleton for SpotCache."""
    global _spot_cache
    if _spot_cache is None:
        with _spot_cache_lock:
            if _spot_cache is None:
                _spot_cache = SpotCache()
    return _spot_cache

class AkShareSource(DataSource):
    """AkShare data source — works ONLY on China direct IP."""

    name = "akshare"
    priority = 1
    needs_china_direct = True

    _AKSHARE_PERIOD_MAP = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}
    _AKSHARE_MIN_MAP = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60"}

    _COLUMN_MAP = {
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount", "涨跌幅": "change_pct", "换手率": "turnover",
    }

    _INTRADAY_COL_MAPS = [
        {
            "时间": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount",
        },
        {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount",
        },
    ]

    def __init__(self):
        super().__init__()
        self._ak = None
        self._spot_cache: Optional[SpotCache] = None
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

    def _get_effective_timeout(self) -> int:
        """Short timeout when eastmoney is known to be blocked."""
        from core.network import get_network_env
        env = get_network_env()
        if not env.eastmoney_ok:
            return 5  # Fast fail on VPN
        return _AKSHARE_SOCKET_TIMEOUT  # Normal timeout on China direct

    @retry(max_attempts=2, delay=1.0, backoff=2.0)
    def get_history(self, code: str, days: int) -> pd.DataFrame:
        if not self._ak or not self.is_available():
            raise DataSourceUnavailableError("AkShare not available")

        start_t = time.time()
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(self._get_effective_timeout())  # ← CHANGED
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (
                datetime.now() - timedelta(days=int(days * 1.5))
            ).strftime("%Y%m%d")
            df = self._ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_date, end_date=end_date, adjust="qfq",
            )
        finally:
            socket.setdefaulttimeout(old_timeout)

        if df is None or df.empty:
            raise DataFetchError(f"No data for {code}")

        df = self._normalize_daily(df)
        latency = (time.time() - start_t) * 1000
        self._record_success(latency)
        return df.tail(days)

    def get_realtime(self, code: str) -> Optional[Quote]:
        if not self._ak or not self.is_available():
            return None
        try:
            data = self._get_spot_cache().get_quote(code)
            if data is None or data["price"] <= 0:
                return None
            return Quote(
                code=code, name=data["name"], price=data["price"],
                open=data["open"], high=data["high"], low=data["low"],
                close=data["close"], volume=data["volume"],
                amount=data["amount"], change=data["change"],
                change_pct=data["change_pct"], source=self.name,
            )
        except Exception as exc:
            self._record_error(str(exc))
            return None

    def get_history_instrument(
        self, inst: dict, days: int, interval: str = "1d"
    ) -> pd.DataFrame:
        if not self._ak or not self.is_available():
            return pd.DataFrame()
        if inst.get("market") != "CN" or inst.get("asset") != "EQUITY":
            return pd.DataFrame()

        start_t = time.time()
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(self._get_effective_timeout())  # ← CHANGED
        try:
            if interval in self._AKSHARE_MIN_MAP:
                time.sleep(0.5)
                df = self._ak.stock_zh_a_hist_min_em(
                    symbol=str(inst["symbol"]).zfill(6),
                    period=self._AKSHARE_MIN_MAP[interval],
                    adjust="qfq",
                )
                if df is None or df.empty:
                    return pd.DataFrame()
                df = self._normalize_intraday(df)
                latency = (time.time() - start_t) * 1000
                self._record_success(latency)
                return df

            period = self._AKSHARE_PERIOD_MAP.get(interval, "daily")
            end_date = datetime.now().strftime("%Y%m%d")
            max_cal_days = INTERVAL_MAX_DAYS.get(interval, 10_000)
            cal_days = min(int(days * 2.2), max_cal_days)
            start_date = (
                datetime.now() - timedelta(days=cal_days)
            ).strftime("%Y%m%d")

            df = self._ak.stock_zh_a_hist(
                symbol=inst["symbol"], period=period,
                start_date=start_date, end_date=end_date, adjust="qfq",
            )
        finally:
            socket.setdefaulttimeout(old_timeout)

        if df is None or df.empty:
            return pd.DataFrame()

        df = self._normalize_daily(df)
        latency = (time.time() - start_t) * 1000
        self._record_success(latency)
        return df.tail(days)

    def get_all_stocks(self) -> pd.DataFrame:
        if not self._ak or not self.is_available():
            return pd.DataFrame()
        try:
            return self._ak.stock_zh_a_spot_em()
        except Exception:
            return pd.DataFrame()

    def _normalize_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=self._COLUMN_MAP)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        df = df[df["volume"] > 0]
        df = df[df["high"] >= df["low"]]
        return df

    def _normalize_intraday(self, df: pd.DataFrame) -> pd.DataFrame:
        for cmap in self._INTRADAY_COL_MAPS:
            if set(cmap.keys()).issubset(set(df.columns)):
                df = df.rename(columns=cmap)
                break
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        for c in ("open", "high", "low", "close", "volume", "amount"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"])
        if "volume" in df.columns:
            df = df[df["volume"].fillna(0) >= 0]
        return df

class YahooSource(DataSource):
    """Yahoo Finance — works ONLY through VPN (foreign IP)."""

    name = "yahoo"
    priority = 1
    needs_vpn = True
    _CB_ERROR_THRESHOLD = 20
    _CB_MIN_COOLDOWN = 20
    _CB_MAX_COOLDOWN = 90

    _SUFFIX_MAP = {"6": ".SS", "0": ".SZ", "3": ".SZ"}
    _SUPPORTED_PREFIXES = ("0", "3", "6")

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

    def is_suitable_for_network(self) -> bool:
        """Yahoo should be tried if it's reachable."""
        from core.network import get_network_env
        env = get_network_env()
        # Prefer Yahoo only when VPN/foreign routing is active.
        return bool(env.is_vpn_active) or (
            bool(getattr(env, "yahoo_ok", False)) and not env.is_china_direct
        )

    def _record_error(self, error: str) -> None:
        # Avoid tripping circuit-breaker too fast on expected Yahoo no-data cases.
        msg = str(error).lower()
        if "no data" in msg or "returned empty" in msg:
            with self._lock:
                self.status.last_error = str(error)
                self.status.error_count += 1
            return
        super()._record_error(error)

    def _to_yahoo_symbol(self, code: str) -> str:
        code = str(code).zfill(6)
        if not code or code[0] not in self._SUPPORTED_PREFIXES:
            return ""
        suffix = self._SUFFIX_MAP.get(code[0], ".SS")
        return f"{code}{suffix}"

    @retry(max_attempts=2, delay=1.0, backoff=2.0)
    def get_history(self, code: str, days: int) -> pd.DataFrame:
        if not self._yf or not self.is_available():
            raise DataSourceUnavailableError("Yahoo Finance not available")

        start_t = time.time()
        symbol = self._to_yahoo_symbol(code)
        ticker = self._yf.Ticker(symbol)
        end = datetime.now()
        start_date = end - timedelta(days=int(days * 1.5))
        df = ticker.history(start=start_date, end=end)

        if df is None or df.empty:
            raise DataFetchError(f"No data from Yahoo for {code}")

        df = self._normalize(df)
        latency = (time.time() - start_t) * 1000
        self._record_success(latency)
        return df.tail(days)

    def get_history_instrument(
        self, inst: dict, days: int, interval: str = "1d"
    ) -> pd.DataFrame:
        if not self._yf or not self.is_available():
            return pd.DataFrame()

        start_t = time.time()
        try:
            yahoo_symbol = self._resolve_symbol(inst)
            if not yahoo_symbol:
                return pd.DataFrame()

            ticker = self._yf.Ticker(yahoo_symbol)
            max_days = INTERVAL_MAX_DAYS.get(interval, 10_000)
            capped_days = min(int(days), max_days)

            yahoo_interval = "1h" if interval == "60m" else interval

            if interval in ("1m", "2m", "5m", "15m", "30m", "60m", "1h"):
                period_str = f"{capped_days}d"
                log.debug(
                    f"Yahoo intraday: {yahoo_symbol} "
                    f"interval={yahoo_interval} period={period_str}"
                )
                df = ticker.history(period=period_str, interval=yahoo_interval)
            else:
                end = datetime.now()
                start_date = end - timedelta(days=capped_days)
                df = ticker.history(
                    start=start_date, end=end, interval=yahoo_interval
                )

            if df is None or df.empty:
                log.debug(f"Yahoo returned empty for {yahoo_symbol} ({interval})")
                return pd.DataFrame()

            df = self._normalize(df)
            latency = (time.time() - start_t) * 1000
            self._record_success(latency)
            log.debug(f"Yahoo OK: {yahoo_symbol} ({interval}): {len(df)} bars")
            # For intraday, `days` is calendar days, not bar count.
            # Trimming by days here can collapse data to a handful of rows
            # and make downstream min-bar checks fail.
            return df

        except Exception as exc:
            self._record_error(str(exc))
            log.debug(
                f"Yahoo failed for {inst.get('symbol')} ({interval}): {exc}"
            )
            return pd.DataFrame()

    def get_realtime(self, code: str) -> Optional[Quote]:
        if not self._yf or not self.is_available():
            return None
        try:
            symbol = self._to_yahoo_symbol(code)
            ticker = self._yf.Ticker(symbol)
            info = ticker.info
            if not info or "regularMarketPrice" not in info:
                return None
            return Quote(
                code=code,
                name=info.get("shortName", ""),
                price=float(info.get("regularMarketPrice", 0)),
                open=float(info.get("regularMarketOpen", 0)),
                high=float(info.get("regularMarketDayHigh", 0)),
                low=float(info.get("regularMarketDayLow", 0)),
                close=float(info.get("previousClose", 0)),
                volume=int(info.get("regularMarketVolume", 0)),
                source=self.name,
            )
        except Exception as exc:
            self._record_error(str(exc))
            return None

    def _resolve_symbol(self, inst: dict) -> Optional[str]:
        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            code6 = str(inst.get("symbol", "")).zfill(6)
            if not code6 or code6[0] not in self._SUPPORTED_PREFIXES:
                return None
            return self._to_yahoo_symbol(code6)
        return inst.get("yahoo") or inst.get("symbol") or None

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df.index.name = "date"
        keep = [c for c in ("open", "high", "low", "close", "volume")
                if c in df.columns]
        df = df[keep].copy()
        if "close" in df.columns and "volume" in df.columns:
            df["amount"] = df["close"] * df["volume"]
        df = df.dropna()
        if "volume" in df.columns:
            df = df[df["volume"] > 0]
        return df

class TencentQuoteSource(DataSource):
    """Tencent quotes — works from ANY IP (China or foreign)."""

    name = "tencent"
    priority = 0
    needs_china_direct = False
    needs_vpn = False

    def get_realtime_batch(self, codes: List[str]) -> Dict[str, Quote]:
        if not self.is_available():
            return {}

        from core.constants import get_exchange

        vendor_symbols: List[str] = []
        vendor_to_code: Dict[str, str] = {}
        for c in codes:
            code6 = str(c).zfill(6)
            ex = get_exchange(code6)
            prefix_map = {"SSE": "sh", "SZSE": "sz", "BSE": "bj"}
            prefix = prefix_map.get(ex)
            if prefix is None:
                continue
            sym = f"{prefix}{code6}"
            vendor_symbols.append(sym)
            vendor_to_code[sym] = code6

        if not vendor_symbols:
            return {}

        out: Dict[str, Quote] = {}
        start_all = time.time()

        try:
            for i in range(0, len(vendor_symbols), _TENCENT_CHUNK_SIZE):
                chunk = vendor_symbols[i : i + _TENCENT_CHUNK_SIZE]
                url = "https://qt.gtimg.cn/q=" + ",".join(chunk)
                resp = self._session.get(url, timeout=5)

                for line in resp.text.splitlines():
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
                        chg_pct = (
                            (chg / prev_close * 100) if prev_close > 0 else 0.0
                        )
                        out[code6] = Quote(
                            code=code6, name=name, price=price,
                            open=open_px, high=price, low=price,
                            close=prev_close, volume=volume, amount=0.0,
                            change=chg, change_pct=chg_pct,
                            source=self.name, is_delayed=False,
                            latency_ms=0.0,
                        )
                    except Exception:
                        continue

            latency = (time.time() - start_all) * 1000
            self._record_success(latency)
            for q in out.values():
                q.latency_ms = latency
            return out

        except Exception as exc:
            self._record_error(str(exc))
            return {}

    def get_realtime(self, code: str) -> Optional[Quote]:
        res = self.get_realtime_batch([code])
        return res.get(str(code).zfill(6))

    def get_history(self, code: str, days: int) -> pd.DataFrame:
        return pd.DataFrame()

    def get_history_instrument(
        self, inst: dict, days: int, interval: str = "1d"
    ) -> pd.DataFrame:
        return pd.DataFrame()

class DataFetcher:
    """
    High-performance data fetcher with automatic network-aware source
    selection, local DB caching, and multi-source fallback.
    """

    def __init__(self):
        self._all_sources: List[DataSource] = []
        self._cache = get_cache()
        self._db = get_database()
        self._rate_limiter = threading.Semaphore(CONFIG.data.parallel_downloads)
        self._request_times: Dict[str, float] = {}
        self._min_interval: float = 0.5
        self._intraday_interval: float = 1.2

        self._last_good_quotes: Dict[str, Quote] = {}
        self._last_good_lock = threading.RLock()

        # Micro-caches — initialized here, not lazily
        self._rt_cache_lock = threading.RLock()
        self._rt_batch_microcache: Dict[str, object] = {
            "ts": 0.0, "key": None, "data": {},
        }
        self._rt_single_microcache: Dict[str, Dict[str, object]] = {}

        self._rate_lock = threading.Lock()
        self._last_network_mode: Optional[bool] = None
        self._init_sources()

    def _init_sources(self) -> None:
        self._all_sources = []
        self._init_local_db_source()

        for source_cls in (AkShareSource, TencentQuoteSource, YahooSource):
            try:
                source = source_cls()
                if source.status.available:
                    self._all_sources.append(source)
                    log.info(
                        f"Data source {source.name} initialized "
                        f"(china_direct={source.needs_china_direct}, "
                        f"vpn={source.needs_vpn})"
                    )
            except Exception as exc:
                log.warning(f"Failed to init {source_cls.__name__}: {exc}")

        if not self._all_sources:
            log.error("No data sources available!")

    def _init_local_db_source(self) -> None:
        """Create and register the local database source."""
        try:
            db = self._db

            class LocalDatabaseSource(DataSource):
                name = "localdb"
                priority = -1
                needs_china_direct = False
                needs_vpn = False

                def __init__(self, db_ref):
                    super().__init__()
                    self._db = db_ref

                def get_history(self, code: str, days: int) -> pd.DataFrame:
                    return self._db.get_bars(str(code).zfill(6), limit=int(days))

                def get_history_instrument(
                    self, inst: dict, days: int, interval: str = "1d"
                ) -> pd.DataFrame:
                    sym = str(inst.get("symbol") or "").zfill(6)
                    if not sym:
                        return pd.DataFrame()
                    if interval == "1d":
                        return self._db.get_bars(sym, limit=int(days))
                    return self._db.get_intraday_bars(
                        sym, interval=interval, limit=int(days)
                    )

                def get_realtime(self, code: str) -> Optional[Quote]:
                    return None

            self._all_sources.append(LocalDatabaseSource(db))
            log.info("Data source localdb initialized")

        except Exception as exc:
            log.warning(f"Failed to init localdb source: {exc}")

    @property
    def _sources(self) -> List[DataSource]:
        """Backward-compatible alias — always reflects current _all_sources."""
        return self._all_sources

    def _get_active_sources(self) -> List[DataSource]:
        """
        Get sources prioritized by current network environment.

        FIX: Network-suitable sources first, others as fallback.
        """
        from core.network import get_network_env
        env = get_network_env()

        # If user toggles VPN/China-direct, clear source cooldowns and pacing state.
        if self._last_network_mode is None:
            self._last_network_mode = bool(env.is_china_direct)
        elif bool(env.is_china_direct) != self._last_network_mode:
            self._last_network_mode = bool(env.is_china_direct)
            for s in self._all_sources:
                with s._lock:
                    s.status.consecutive_errors = 0
                    s.status.disabled_until = None
                    s.status.available = True
            with self._rate_lock:
                self._request_times.clear()
            log.info(
                "Network mode changed; data source cooldowns reset "
                f"({'CHINA_DIRECT' if env.is_china_direct else 'VPN_FOREIGN'})"
            )

        active = [s for s in self._all_sources if s.is_available()]
        ranked = sorted(
            active,
            key=lambda s: (-self._source_health_score(s, env), s.priority),
        )
        return ranked

    def _source_health_score(self, source: DataSource, env) -> float:
        """
        Score a source by network suitability + recent health.
        Higher score means earlier selection.
        """
        score = 0.0

        # Static preference by network mode
        if source.name == "localdb":
            score += 120.0
        elif env.is_china_direct:
            if source.name == "akshare":
                score += 90.0
            elif source.name == "tencent":
                score += 55.0
            elif source.name == "yahoo":
                score += 10.0
        else:
            if source.name == "yahoo":
                score += 90.0
            elif source.name == "tencent":
                score += 60.0
            elif source.name == "akshare":
                score += 8.0

        # Network suitability
        try:
            if source.is_suitable_for_network():
                score += 15.0
            else:
                score -= 40.0
        except Exception:
            score -= 5.0

        # Runtime health telemetry
        st = source.status
        attempts = max(1, int(st.success_count + st.error_count))
        success_rate = float(st.success_count) / attempts
        score += 30.0 * success_rate

        if st.avg_latency_ms > 0:
            # Penalize slower sources (cap to avoid over-penalizing).
            score -= min(25.0, st.avg_latency_ms / 200.0)

        # Penalize frequent failures and cooldown state.
        score -= min(20.0, float(st.consecutive_errors) * 1.5)
        if st.disabled_until and datetime.now() < st.disabled_until:
            score -= 50.0

        return score

    def _rate_limit(self, source: str, interval: str = "1d") -> None:
        with self._rate_lock:
            now = time.time()
            last = self._request_times.get(source, 0.0)
            if source == "yahoo":
                min_wait = 2.2 if interval in _INTRADAY_INTERVALS else 1.4
            else:
                min_wait = (
                    self._intraday_interval
                    if interval in _INTRADAY_INTERVALS
                    else self._min_interval
                )
            wait = min_wait - (now - last)
            if wait > 0:
                time.sleep(wait)
            self._request_times[source] = time.time()

    def _db_is_fresh_enough(
        self, code6: str, max_lag_days: int = 3
    ) -> bool:
        """Check whether local DB data is recent enough to skip online fetch."""
        try:
            last = self._db.get_last_date(code6)
            if not last:
                return False
            from core.constants import is_trading_day

            today = datetime.now().date()
            lag = 0
            d = last
            while d < today and lag <= max_lag_days:
                d += timedelta(days=1)
                if is_trading_day(d):
                    lag += 1
            return lag <= max_lag_days
        except Exception:
            return False

    def get_realtime_batch(self, codes: List[str]) -> Dict[str, Quote]:
        """Fetch real-time quotes for multiple codes in one batch."""
        cleaned = list(dict.fromkeys(
            c for c in (self.clean_code(c) for c in codes) if c
        ))
        if not cleaned:
            return {}
        if _is_offline():
            return {}

        now = time.time()
        key = ",".join(cleaned)

        # Micro-cache read
        with self._rt_cache_lock:
            mc = self._rt_batch_microcache
            if (
                mc["key"] == key
                and (now - float(mc["ts"])) < _MICRO_CACHE_TTL
            ):
                data = mc["data"]
                if isinstance(data, dict) and data:
                    return dict(data)

        result: Dict[str, Quote] = {}

        # Try batch-capable sources first
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
            self._fill_from_spot_cache(missing, result)

        # Last-good fallback
        if not result:
            result = self._fallback_last_good(cleaned)

        # Update last-good store
        if result:
            with self._last_good_lock:
                for c, q in result.items():
                    if q and q.price > 0:
                        self._last_good_quotes[c] = q

        # Micro-cache write
        with self._rt_cache_lock:
            self._rt_batch_microcache["ts"] = now
            self._rt_batch_microcache["key"] = key
            self._rt_batch_microcache["data"] = dict(result)

        return result

    def _fill_from_spot_cache(
        self, missing: List[str], result: Dict[str, Quote]
    ) -> None:
        """Attempt to fill missing quotes from EastMoney spot cache."""
        try:
            cache = get_spot_cache()
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

    def _fallback_last_good(self, codes: List[str]) -> Dict[str, Quote]:
        """Return last-good quotes if they are recent enough."""
        result: Dict[str, Quote] = {}
        with self._last_good_lock:
            for c in codes:
                q = self._last_good_quotes.get(c)
                if q and q.price > 0:
                    age = (
                        datetime.now() - (q.timestamp or datetime.now())
                    ).total_seconds()
                    if age <= _LAST_GOOD_MAX_AGE:
                        result[c] = q
        return result

    def _fetch_from_sources_instrument(
        self, inst: dict, days: int, interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch from active sources with smart fallback.

        FIX: Tries ALL available sources, not just network-preferred ones.
        """
        sources = self._get_active_sources()

        if not sources:
            # No "active" sources — try ALL sources anyway
            log.warning(
                f"No active sources for {inst.get('symbol')} ({interval}), "
                f"trying all sources as fallback"
            )
            sources = [s for s in self._all_sources if s.name != "localdb"]

        if not sources:
            log.warning(f"No sources at all for {inst.get('symbol')} ({interval})")
            return pd.DataFrame()

        log.debug(
            f"Sources for {inst.get('symbol')} ({interval}): "
            f"{[s.name for s in sources]}"
        )

        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            from core.network import get_network_env
            env = get_network_env()

            if env.is_china_direct:
                preferred = "akshare"
            elif env.yahoo_ok:
                preferred = "yahoo"
            else:
                preferred = "akshare"  # Try anyway

            sources.sort(key=lambda s: (
                0 if s.name == preferred else
                1 if s.name == "localdb" else 2,
                s.priority,
            ))

        with self._rate_limiter:
            errors = []
            for source in sources:
                try:
                    self._rate_limit(source.name, interval)
                    df = self._try_source_instrument(
                        source, inst, days, interval
                    )
                    if df is not None and not df.empty:
                        min_required = min(days // 4, 30)
                        if len(df) >= min_required:
                            log.debug(
                                f"Got {len(df)} bars from {source.name} "
                                f"for {inst.get('symbol')} ({interval})"
                            )
                            return df
                        log.debug(
                            f"{source.name} returned {len(df)} bars for "
                            f"{inst.get('symbol')} ({interval}), "
                            f"need >= {min_required}"
                        )
                    else:
                        log.debug(
                            f"{source.name} returned empty for "
                            f"{inst.get('symbol')} ({interval})"
                        )
                except Exception as exc:
                    errors.append(f"{source.name}: {exc}")
                    log.debug(
                        f"{source.name} failed for "
                        f"{inst.get('symbol')} ({interval}): {exc}"
                    )
                    continue

        if errors:
            log.warning(
                f"All sources failed for {inst.get('symbol')} ({interval}): "
                f"{'; '.join(errors[:3])}"
            )
        else:
            log.warning(f"All sources failed for {inst.get('symbol')} ({interval})")
        return pd.DataFrame()

    @staticmethod
    def _try_source_instrument(
        source: DataSource, inst: dict, days: int, interval: str
    ) -> pd.DataFrame:
        """Try get_history_instrument, fall back to get_history for CN equity."""
        fn = getattr(source, "get_history_instrument", None)
        if callable(fn):
            return fn(inst, days=days, interval=interval)
        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            return source.get_history(inst["symbol"], days)
        return pd.DataFrame()

    @staticmethod
    def clean_code(code: str) -> str:
        """Normalize a stock code to bare 6-digit form."""
        if code is None:
            return ""
        s = str(code).strip()
        if not s:
            return ""
        s = s.replace(" ", "").replace("-", "").replace("_", "")

        prefixes = (
            "sh.", "sz.", "bj.", "SH.", "SZ.", "BJ.",
            "sh", "sz", "bj", "SH", "SZ", "BJ",
        )
        for p in prefixes:
            if s.startswith(p):
                s = s[len(p):]
                break

        suffixes = (".SS", ".SZ", ".BJ", ".ss", ".sz", ".bj")
        for suf in suffixes:
            if s.endswith(suf):
                s = s[: -len(suf)]
                break

        digits = "".join(ch for ch in s if ch.isdigit())
        return digits.zfill(6) if digits else ""

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize and validate an OHLCV dataframe."""
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

        if (
            "amount" not in out.columns
            and "close" in out.columns
            and "volume" in out.columns
        ):
            out["amount"] = out["close"] * out["volume"]

        out = out.replace([np.inf, -np.inf], np.nan)
        ohlc_cols = [c for c in ("open", "high", "low", "close") if c in out.columns]
        if ohlc_cols:
            out[ohlc_cols] = out[ohlc_cols].ffill()
        out = out.fillna(0)

        return out

    def get_history(
        self,
        code: str,
        days: int = 500,
        bars: Optional[int] = None,
        use_cache: bool = True,
        update_db: bool = True,
        instrument: Optional[dict] = None,
        interval: str = "1d",
        max_age_hours: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Unified history fetcher.

        Priority: memory cache → local DB → online sources.
        """
        from core.instruments import parse_instrument, instrument_key

        inst = instrument or parse_instrument(code)
        key = instrument_key(inst)
        interval = str(interval).lower()
        offline = _is_offline()

        count = max(1, int(bars if bars is not None else days))
        max_days = INTERVAL_MAX_DAYS.get(interval, 10_000)
        fetch_days = min(bars_to_days(count, interval), max_days)

        if max_age_hours is not None:
            ttl = float(max_age_hours)
        elif interval == "1d":
            ttl = float(CONFIG.data.cache_ttl_hours)
        else:
            ttl = min(float(CONFIG.data.cache_ttl_hours), 1.0 / 120.0)

        cache_key = f"history:{key}:{interval}:{count}"

        if use_cache:
            cached_df = self._cache.get(cache_key, ttl)
            if isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
                cached_df = self._clean_dataframe(cached_df)
                if len(cached_df) >= min(count, 100):
                    return cached_df.tail(count)

        is_cn_equity = (
            inst.get("market") == "CN" and inst.get("asset") == "EQUITY"
        )

        if is_cn_equity and interval != "1d":
            return self._get_history_cn_intraday(
                inst, count, fetch_days, interval, cache_key, offline
            )

        if is_cn_equity and interval == "1d":
            return self._get_history_cn_daily(
                inst, count, fetch_days, cache_key, offline, update_db
            )

        # Non-CN instrument
        if offline:
            return pd.DataFrame()
        df = self._clean_dataframe(
            self._fetch_from_sources_instrument(
                inst, days=fetch_days, interval=interval
            )
        )
        if df.empty:
            return pd.DataFrame()
        out = df.tail(count)
        self._cache.set(cache_key, out)
        return out

    def _get_history_cn_intraday(
        self,
        inst: dict,
        count: int,
        fetch_days: int,
        interval: str,
        cache_key: str,
        offline: bool,
    ) -> pd.DataFrame:
        """Handle CN equity intraday intervals."""
        code6 = str(inst["symbol"]).zfill(6)
        db_df = pd.DataFrame()
        try:
            db_df = self._clean_dataframe(
                self._db.get_intraday_bars(code6, interval=interval, limit=count)
            )
        except Exception:
            pass

        online_df = pd.DataFrame()
        if not offline:
            online_df = self._clean_dataframe(
                self._fetch_from_sources_instrument(
                    inst, days=fetch_days, interval=interval
                )
            )

        merged = self._merge_parts(db_df, online_df) if not offline else db_df

        if merged.empty:
            return pd.DataFrame()

        out = self._cache_tail(cache_key, merged, count)
        try:
            self._db.upsert_intraday_bars(code6, interval, out)
        except Exception:
            pass
        return out

    def _get_history_cn_daily(
        self,
        inst: dict,
        count: int,
        fetch_days: int,
        cache_key: str,
        offline: bool,
        update_db: bool,
    ) -> pd.DataFrame:
        """Handle CN equity daily interval."""
        code6 = str(inst["symbol"]).zfill(6)
        db_df = self._clean_dataframe(
            self._db.get_bars(inst["symbol"], limit=count)
        )

        if (
            len(db_df) >= count
            and self._db_is_fresh_enough(code6, max_lag_days=3)
        ):
            return self._cache_tail(cache_key, db_df, count)

        if offline:
            return db_df.tail(count) if not db_df.empty else pd.DataFrame()

        online_df = self._clean_dataframe(
            self._fetch_from_sources_instrument(
                inst, days=fetch_days, interval="1d"
            )
        )
        merged = self._merge_parts(db_df, online_df)
        if merged.empty:
            return pd.DataFrame()

        out = self._cache_tail(cache_key, merged, count)
        if update_db:
            try:
                self._db.upsert_bars(inst["symbol"], out)
            except Exception:
                pass
        return out

    def _merge_parts(self, *dfs: pd.DataFrame) -> pd.DataFrame:
        """Merge and clean non-empty dataframes."""
        parts = [p for p in dfs if isinstance(p, pd.DataFrame) and not p.empty]
        if not parts:
            return pd.DataFrame()
        return self._clean_dataframe(pd.concat(parts, axis=0))

    def _cache_tail(self, cache_key: str, df: pd.DataFrame, count: int) -> pd.DataFrame:
        out = df.tail(count)
        self._cache.set(cache_key, out)
        return out

    def get_realtime(
        self, code: str, instrument: Optional[dict] = None
    ) -> Optional[Quote]:
        """Get real-time quote for a single instrument."""
        from core.instruments import parse_instrument
        inst = instrument or parse_instrument(code)

        if _is_offline():
            return None

        # Fast path: CN equity via batch
        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            code6 = str(inst.get("symbol") or "").zfill(6)
            if code6:
                return self._get_realtime_cn(code6)

        # Generic path: try each source
        return self._get_realtime_generic(inst)

    def _get_realtime_cn(self, code6: str) -> Optional[Quote]:
        """Optimized CN equity real-time via batch + micro-cache."""
        now = time.time()

        # Micro-cache check
        with self._rt_cache_lock:
            rec = self._rt_single_microcache.get(code6)
            if rec and (now - float(rec["ts"])) < _MICRO_CACHE_TTL:
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

        # Last-good fallback
        with self._last_good_lock:
            q = self._last_good_quotes.get(code6)
            if q and q.price > 0:
                age = (
                    datetime.now() - (q.timestamp or datetime.now())
                ).total_seconds()
                if age <= _LAST_GOOD_MAX_AGE:
                    return q
        return None

    def _get_realtime_generic(self, inst: dict) -> Optional[Quote]:
        """Fetch real-time quote from all sources, pick best."""
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

        # Median-filter outliers, then pick lowest-latency non-delayed
        prices = np.array([c.price for c in candidates], dtype=float)
        med = float(np.median(prices))
        good = [
            c for c in candidates
            if abs(c.price - med) / max(med, 1e-8) < 0.01
        ]
        pool = good if good else candidates
        pool.sort(key=lambda q: (q.is_delayed, q.latency_ms))
        best = pool[0]

        # Update last-good
        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            code6 = str(inst.get("symbol") or "").zfill(6)
            with self._last_good_lock:
                self._last_good_quotes[code6] = best

        return best

    def get_multiple_parallel(
        self,
        codes: List[str],
        days: int = 500,
        callback: Optional[Callable[[str, int, int], None]] = None,
        max_workers: Optional[int] = None,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch history for multiple codes in parallel."""
        results: Dict[str, pd.DataFrame] = {}
        total = len(codes)
        completed = 0
        lock = threading.Lock()

        # Reset circuit-breakers before bulk fetch
        for source in self._all_sources:
            with source._lock:
                source.status.consecutive_errors = 0
                source.status.disabled_until = None

        def fetch_one(code: str) -> Tuple[str, pd.DataFrame]:
            try:
                df = self.get_history(code, days, interval=interval)
                return code, df
            except Exception as exc:
                log.debug(f"Failed to fetch {code}: {exc}")
                return code, pd.DataFrame()

        if interval in _INTRADAY_INTERVALS:
            workers = min(max_workers or 2, 2)
        else:
            workers = min(max_workers or 5, 5)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch_one, c): c for c in codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    code, df = future.result(timeout=120)
                    if (
                        not df.empty
                        and len(df) >= CONFIG.data.min_history_days
                    ):
                        results[code] = df
                except Exception as exc:
                    log.warning(f"Failed to fetch {code}: {exc}")
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
                except Exception as exc:
                    log.warning(f"Failed to get stock list: {exc}")
        return pd.DataFrame()

    def get_source_status(self) -> List[DataSourceStatus]:
        return [s.status for s in self._all_sources]

    def reset_sources(self) -> None:
        from core.network import invalidate_network_cache
        invalidate_network_cache()

        for source in self._all_sources:
            with source._lock:
                source.status.consecutive_errors = 0
                source.status.disabled_until = None
                source.status.available = True
        log.info("All data sources reset, network cache invalidated")

_fetcher: Optional[DataFetcher] = None
_fetcher_lock = threading.Lock()

def get_fetcher() -> DataFetcher:
    """Double-checked locking singleton for DataFetcher."""
    global _fetcher
    if _fetcher is None:
        with _fetcher_lock:
            if _fetcher is None:
                _fetcher = DataFetcher()
    return _fetcher
