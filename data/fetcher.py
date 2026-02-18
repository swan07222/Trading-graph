# data/fetcher.py
import json
import math
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import wraps

import numpy as np
import pandas as pd
import requests

from config.settings import CONFIG
from core.exceptions import DataFetchError, DataSourceUnavailableError
from data.cache import get_cache
from data.database import get_database
from data.session_cache import get_session_bar_cache
from utils.helpers import to_float, to_int
from utils.logger import get_logger

log = get_logger(__name__)

# Maximum calendar days each interval can fetch (API limits)
INTERVAL_MAX_DAYS: dict[str, int] = {
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

BARS_PER_DAY: dict[str, float] = {
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
_INTRADAY_INTERVALS = frozenset({"1m", "2m", "5m", "15m", "30m", "60m", "1h"})

# Micro-cache TTL in seconds
_MICRO_CACHE_TTL: float = 0.25

# Maximum staleness (seconds) for last-good quote fallback
_LAST_GOOD_MAX_AGE: float = 12.0

_TENCENT_CHUNK_SIZE: int = 80  # reduced chunk size for reliability

# Default socket timeout for AkShare calls (seconds)
_AKSHARE_SOCKET_TIMEOUT: int = 15

# SpotCache default TTL (seconds)
_SPOT_CACHE_TTL: float = 30.0

# China A-share daily price limit (10% for normal, 20% for ST)
_CN_DAILY_LIMIT: float = 0.205

# Intraday caps used by _clean_dataframe sanitization.
# (body_cap, span_cap, wick_cap, jump_cap) per interval
_INTRADAY_CAPS: dict[str, tuple[float, float, float, float]] = {
    "1m":  (0.010, 0.030, 0.010, 0.08),
    "2m":  (0.012, 0.036, 0.012, 0.10),
    "5m":  (0.016, 0.050, 0.016, 0.12),
    "15m": (0.024, 0.070, 0.022, 0.14),
    "30m": (0.032, 0.090, 0.028, 0.16),
    "60m": (0.045, 0.120, 0.038, 0.18),
    "1h":  (0.045, 0.120, 0.038, 0.18),
}


def _run_with_timeout(
    task: Callable[[], object],
    timeout_s: float,
) -> object | None:
    """
    Run a callable with a timeout without mutating process-global socket defaults.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(task)
    try:
        return future.result(timeout=max(0.1, float(timeout_s)))
    except FuturesTimeout:
        return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


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
    # 1.8x multiplier converts trading days → calendar days
    # (accounts for weekends, holidays), +3 for safety buffer
    calendar_days = int(trading_days * 1.8) + 3
    max_days = INTERVAL_MAX_DAYS.get(interval, 10_000)
    return min(calendar_days, max_days)


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error: Exception | None = None
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
    timestamp: datetime | None = None
    source: str = ""
    is_delayed: bool = True
    latency_ms: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            try:
                from zoneinfo import ZoneInfo
                self.timestamp = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
            except Exception:
                self.timestamp = datetime.now(tz=timezone.utc)


@dataclass
class DataSourceStatus:
    """Health / telemetry for a single data source."""
    name: str
    available: bool = True
    last_success: datetime | None = None
    last_error: str | None = None
    success_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    avg_latency_ms: float = 0.0
    disabled_until: datetime | None = None


class DataSource:
    """Abstract data source with error tracking and circuit-breaker."""

    name: str = "base"
    priority: int = 0
    needs_china_direct: bool = False
    needs_vpn: bool = False

    # Circuit-breaker thresholds — raised to avoid premature disabling
    _CB_ERROR_THRESHOLD: int = 12
    _CB_MIN_COOLDOWN: int = 20
    _CB_MAX_COOLDOWN: int = 90
    _CB_COOLDOWN_INCREMENT: int = 2
    _CB_RECENT_SUCCESS_DECAY_SEC: float = 30.0
    _CB_HALF_OPEN_PROBE_INTERVAL: float = 8.0
    _CB_DISABLE_WARN_MIN_GAP_SEC: float = 8.0

    def __init__(self):
        self.status = DataSourceStatus(name=self.name)
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })
        self._latencies: list[float] = []
        self._lock = threading.Lock()
        self._next_half_open_probe_ts: float = 0.0
        self._last_disable_warn_ts: float = 0.0

    def is_available(self) -> bool:
        with self._lock:
            if not self.status.available:
                return False
            if self.status.disabled_until:
                if datetime.now() < self.status.disabled_until:
                    now_probe = time.monotonic()
                    if now_probe >= float(self._next_half_open_probe_ts):
                        self._next_half_open_probe_ts = (
                            now_probe + float(self._CB_HALF_OPEN_PROBE_INTERVAL)
                        )
                        return True
                    return False
                # Cooldown expired → re-enable
                self.status.disabled_until = None
                self.status.consecutive_errors = 0
                self._next_half_open_probe_ts = 0.0
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
            self._next_half_open_probe_ts = 0.0
            if latency_ms > 0:
                self._latencies.append(latency_ms)
                if len(self._latencies) > 100:
                    self._latencies.pop(0)
                self.status.avg_latency_ms = float(np.mean(self._latencies))

    def _record_error(self, error: str) -> None:
        with self._lock:
            now_dt = datetime.now()
            self.status.last_error = error
            self.status.error_count += 1
            # Dampen error streak if source was recently healthy
            if (
                self.status.last_success is not None
                and self.status.consecutive_errors > 0
            ):
                try:
                    age = (now_dt - self.status.last_success).total_seconds()
                except Exception:
                    age = float("inf")
                if age <= float(self._CB_RECENT_SUCCESS_DECAY_SEC):
                    self.status.consecutive_errors = max(
                        0, int(self.status.consecutive_errors) - 1
                    )
            self.status.consecutive_errors += 1
            if self.status.consecutive_errors >= self._CB_ERROR_THRESHOLD:
                overflow = max(
                    0,
                    int(self.status.consecutive_errors)
                    - int(self._CB_ERROR_THRESHOLD),
                )
                cooldown = min(
                    self._CB_MIN_COOLDOWN
                    + (overflow * self._CB_COOLDOWN_INCREMENT),
                    self._CB_MAX_COOLDOWN,
                )
                self.status.disabled_until = (
                    now_dt + timedelta(seconds=cooldown)
                )
                probe_after = max(
                    2.0,
                    min(
                        float(self._CB_HALF_OPEN_PROBE_INTERVAL),
                        float(cooldown) * 0.30,
                    ),
                )
                self._next_half_open_probe_ts = (
                    time.monotonic() + float(probe_after)
                )
                now_warn = time.monotonic()
                if (
                    now_warn - float(self._last_disable_warn_ts)
                    >= float(self._CB_DISABLE_WARN_MIN_GAP_SEC)
                ):
                    self._last_disable_warn_ts = now_warn
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

    def get_realtime(self, code: str) -> Quote | None:
        return None


class SpotCache:
    """Thread-safe cached A-share spot data with TTL."""

    # Robust column name mapping (handles API version differences)
    _FIELD_MAP = {
        "price":      ("最新价", "现价", "price"),
        "open":       ("今开", "开盘价", "open"),
        "high":       ("最高", "最高价", "high"),
        "low":        ("最低", "最低价", "low"),
        "close":      ("昨收", "昨收价", "prev_close", "close"),
        "volume":     ("成交量", "volume"),
        "amount":     ("成交额", "amount"),
        "change":     ("涨跌额", "change"),
        "change_pct": ("涨跌幅", "change_pct"),
        "name":       ("名称", "股票名称", "name"),
    }

    def __init__(self, ttl_seconds: float = _SPOT_CACHE_TTL):
        self._cache: pd.DataFrame | None = None
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

    def _get_field(self, row: pd.Series, field: str) -> object:
        """Safely get a field from a row trying multiple column name variants."""
        for col in self._FIELD_MAP.get(field, (field,)):
            val = row.get(col)
            if val is not None and str(val) not in ("", "nan", "None", "-"):
                return val
        return None

    def get(self, force_refresh: bool = False) -> pd.DataFrame | None:
        """Return cached spot DataFrame, refreshing if stale."""
        now = time.time()
        with self._lock:
            if (
                not force_refresh
                and self._cache is not None
                and (now - self._cache_time) < self._ttl
            ):
                return self._cache
            stale = self._cache

        if self._ak is None:
            return stale

        from core.network import get_network_env
        env = get_network_env()
        if not env.eastmoney_ok:
            return stale

        with self._rate_lock:
            # Re-check after acquiring rate lock
            with self._lock:
                if (
                    not force_refresh
                    and self._cache is not None
                    and (time.time() - self._cache_time) < self._ttl
                ):
                    return self._cache

            try:
                timeout_s = 10.0
                fresh = _run_with_timeout(
                    lambda: self._ak.stock_zh_a_spot_em(),
                    timeout_s,
                )
                if fresh is None:
                    log.debug("SpotCache refresh timed out after %.1fs", timeout_s)

                with self._lock:
                    if isinstance(fresh, pd.DataFrame) and not fresh.empty:
                        self._cache = fresh
                        self._cache_time = time.time()
                        log.debug(
                            "SpotCache refreshed: %d rows, cols=%s",
                            len(fresh),
                            list(fresh.columns[:8]),
                        )
                    return self._cache

            except Exception as exc:
                log.debug("SpotCache refresh failed: %s", exc)
                with self._lock:
                    return self._cache

    def get_quote(self, symbol: str) -> dict | None:
        """Look up a single stock from the cached spot snapshot."""
        symbol = str(symbol).strip()
        # Strip common exchange prefixes
        for prefix in ("sh", "sz", "SH", "SZ", "bj", "BJ"):
            if symbol.upper().startswith(prefix.upper()) and len(symbol) > len(prefix):
                candidate = symbol[len(prefix):]
                if candidate.isdigit():
                    symbol = candidate
                    break
        # Strip dot-suffixes like .SZ .SS
        if "." in symbol:
            symbol = symbol.split(".")[0]
        symbol = symbol.strip().zfill(6)

        df = self.get()
        if df is None or df.empty:
            return None

        try:
            # Find code column robustly
            code_col_name = None
            for candidate in ("代码", "股票代码", "code", "symbol"):
                if candidate in df.columns:
                    code_col_name = candidate
                    break
            if code_col_name is None:
                log.debug("SpotCache: no code column found in %s", list(df.columns))
                return None

            code_col = (
                df[code_col_name]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .str.zfill(6)
            )
            row_mask = code_col == symbol
            if not row_mask.any():
                return None
            r = df[row_mask].iloc[0]
        except Exception as exc:
            log.debug("SpotCache lookup error for %s: %s", symbol, exc)
            return None

        try:
            price = to_float(self._get_field(r, "price"))
            if price is None or price <= 0:
                return None
            return {
                "code":       symbol,
                "name":       str(self._get_field(r, "name") or ""),
                "price":      price,
                "open":       to_float(self._get_field(r, "open") or 0),
                "high":       to_float(self._get_field(r, "high") or 0),
                "low":        to_float(self._get_field(r, "low") or 0),
                "close":      to_float(self._get_field(r, "close") or 0),
                "volume":     to_int(self._get_field(r, "volume") or 0),
                "amount":     to_float(self._get_field(r, "amount") or 0),
                "change":     to_float(self._get_field(r, "change") or 0),
                "change_pct": to_float(self._get_field(r, "change_pct") or 0),
            }
        except Exception as exc:
            log.debug("SpotCache field extraction error for %s: %s", symbol, exc)
            return None


_spot_cache: SpotCache | None = None
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
    _AKSHARE_MIN_MAP = {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60"
    }

    _COLUMN_MAP = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "change_pct",
        "换手率": "turnover",
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
        self._spot_cache: SpotCache | None = None
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

    def is_suitable_for_network(self) -> bool:
        from core.network import get_network_env
        env = get_network_env()
        if not bool(getattr(env, "eastmoney_ok", False)):
            return False
        return bool(env.is_china_direct)

    def _get_spot_cache(self) -> SpotCache:
        if self._spot_cache is None:
            self._spot_cache = get_spot_cache()
        return self._spot_cache

    def _get_effective_timeout(self) -> int:
        from core.network import get_network_env
        env = get_network_env()
        if not env.eastmoney_ok:
            return 5
        return _AKSHARE_SOCKET_TIMEOUT

    @retry(max_attempts=2, delay=1.0, backoff=2.0)
    def get_history(self, code: str, days: int) -> pd.DataFrame:
        if not self._ak or not self.is_available():
            raise DataSourceUnavailableError("AkShare not available")

        start_t = time.time()
        timeout_s = float(self._get_effective_timeout())
        end_date = datetime.now().strftime("%Y%m%d")
        # Fetch extra days to account for weekends/holidays
        start_date = (
            datetime.now() - timedelta(days=int(days * 2.0) + 10)
        ).strftime("%Y%m%d")
        df = _run_with_timeout(
            lambda: self._ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            ),
            timeout_s,
        )

        if df is None:
            raise DataFetchError(f"AkShare timeout for {code}")
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise DataFetchError(f"No data for {code}")

        df = self._normalize_daily(df)
        latency = (time.time() - start_t) * 1000
        self._record_success(latency)
        log.debug("AkShare daily %s: %d bars", code, len(df))
        return df.tail(days)

    def get_realtime(self, code: str) -> Quote | None:
        if not self._ak or not self.is_available():
            return None
        try:
            data = self._get_spot_cache().get_quote(code)
            if data is None or not data.get("price") or data["price"] <= 0:
                return None
            return Quote(
                code=code, name=data["name"], price=data["price"],
                open=data["open"], high=data["high"], low=data["low"],
                close=data["close"], volume=data["volume"],
                amount=data["amount"], change=data["change"],
                change_pct=data["change_pct"], source=self.name,
                is_delayed=False,
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
        timeout_s = float(self._get_effective_timeout())
        symbol = str(inst["symbol"]).zfill(6)

        if interval in self._AKSHARE_MIN_MAP:
            time.sleep(0.3)
            df = _run_with_timeout(
                lambda: self._ak.stock_zh_a_hist_min_em(
                    symbol=symbol,
                    period=self._AKSHARE_MIN_MAP[interval],
                    adjust="qfq",
                ),
                timeout_s,
            )
            if not isinstance(df, pd.DataFrame) or df.empty:
                return pd.DataFrame()
            df = self._normalize_intraday(df)
            if df.empty:
                return pd.DataFrame()
            latency = (time.time() - start_t) * 1000
            self._record_success(latency)
            log.debug(
                "AkShare intraday %s (%s): %d bars",
                symbol, interval, len(df)
            )
            return df

        period = self._AKSHARE_PERIOD_MAP.get(interval, "daily")
        end_date = datetime.now().strftime("%Y%m%d")
        max_cal_days = INTERVAL_MAX_DAYS.get(interval, 10_000)
        # Fetch 2.5x requested days to ensure enough bars after holiday filtering
        cal_days = min(int(days * 2.5) + 10, max_cal_days)
        start_date = (
            datetime.now() - timedelta(days=cal_days)
        ).strftime("%Y%m%d")

        df = _run_with_timeout(
            lambda: self._ak.stock_zh_a_hist(
                symbol=symbol, period=period,
                start_date=start_date, end_date=end_date, adjust="qfq",
            ),
            timeout_s,
        )

        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()

        df = self._normalize_daily(df)
        latency = (time.time() - start_t) * 1000
        self._record_success(latency)
        log.debug(
            "AkShare %s (%s): %d bars",
            inst.get("symbol"), interval, len(df)
        )
        return df.tail(days)

    def get_all_stocks(self) -> pd.DataFrame:
        if not self._ak or not self.is_available():
            return pd.DataFrame()
        try:
            return self._ak.stock_zh_a_spot_em()
        except Exception:
            return pd.DataFrame()

    def _normalize_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.rename(columns=self._COLUMN_MAP)
        if "date" not in df.columns:
            # Try to find date column
            for col in df.columns:
                if "日" in col or "date" in col.lower() or "time" in col.lower():
                    df = df.rename(columns={col: "date"})
                    break
        if "date" not in df.columns:
            log.warning("AkShare: no date column found, cols=%s", list(df.columns))
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date").sort_index()

        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]
        if "volume" in df.columns:
            df = df[df["volume"].fillna(0) >= 0]
        if "high" in df.columns and "low" in df.columns:
            df = df[df["high"] >= df["low"]]

        # Validate OHLC consistency
        if all(c in df.columns for c in ("open", "high", "low", "close")):
            df["high"] = df[["open", "high", "close"]].max(axis=1)
            df["low"] = df[["open", "low", "close"]].min(axis=1)

        return df

    def _normalize_intraday(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
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
        df = df[df["close"] > 0]

        if "volume" in df.columns:
            df = df[df["volume"].fillna(0) >= 0]

        # Validate and fix OHLC
        if all(c in df.columns for c in ("open", "high", "low", "close")):
            df["open"] = df["open"].where(df["open"] > 0, df["close"])
            df["high"] = df[["open", "high", "close"]].max(axis=1)
            df["low"] = df[["open", "low", "close"]].min(axis=1)

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
        from core.network import get_network_env
        env = get_network_env()
        return bool(env.is_vpn_active) or (
            bool(getattr(env, "yahoo_ok", False)) and not env.is_china_direct
        )

    def _record_error(self, error: str) -> None:
        msg = str(error).lower()
        # Don't trip circuit-breaker for expected no-data responses
        if any(k in msg for k in ("no data", "returned empty", "period=", "no timezone")):
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
        if not symbol:
            raise DataFetchError(f"Cannot map {code} to Yahoo symbol")

        ticker = self._yf.Ticker(symbol)
        end = datetime.now()
        start_date = end - timedelta(days=int(days * 2.0) + 10)
        df = ticker.history(start=start_date, end=end, auto_adjust=True)

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
                df = ticker.history(
                    period=period_str,
                    interval=yahoo_interval,
                    auto_adjust=True,
                )
            else:
                end = datetime.now()
                start_date = end - timedelta(days=int(capped_days * 2.0) + 10)
                df = ticker.history(
                    start=start_date,
                    end=end,
                    interval=yahoo_interval,
                    auto_adjust=True,
                )

            if df is None or df.empty:
                log.debug(
                    "Yahoo returned empty for %s (%s)",
                    yahoo_symbol, interval
                )
                return pd.DataFrame()

            df = self._normalize(df)
            if df.empty:
                return pd.DataFrame()

            latency = (time.time() - start_t) * 1000
            self._record_success(latency)
            log.debug(
                "Yahoo OK: %s (%s): %d bars",
                yahoo_symbol, interval, len(df)
            )
            return df

        except Exception as exc:
            self._record_error(str(exc))
            log.debug(
                "Yahoo failed for %s (%s): %s",
                inst.get("symbol"), interval, exc
            )
            return pd.DataFrame()

    def get_realtime(self, code: str) -> Quote | None:
        if not self._yf or not self.is_available():
            return None
        try:
            symbol = self._to_yahoo_symbol(code)
            if not symbol:
                return None
            ticker = self._yf.Ticker(symbol)
            info = ticker.info
            if not info or "regularMarketPrice" not in info:
                return None
            price = float(info.get("regularMarketPrice") or 0)
            if price <= 0:
                return None
            return Quote(
                code=code,
                name=info.get("shortName", ""),
                price=price,
                open=float(info.get("regularMarketOpen") or 0),
                high=float(info.get("regularMarketDayHigh") or 0),
                low=float(info.get("regularMarketDayLow") or 0),
                close=float(info.get("previousClose") or 0),
                volume=int(info.get("regularMarketVolume") or 0),
                source=self.name,
                is_delayed=False,
            )
        except Exception as exc:
            self._record_error(str(exc))
            return None

    def _resolve_symbol(self, inst: dict) -> str | None:
        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            code6 = str(inst.get("symbol", "")).zfill(6)
            if not code6 or code6[0] not in self._SUPPORTED_PREFIXES:
                return None
            return self._to_yahoo_symbol(code6)
        return inst.get("yahoo") or inst.get("symbol") or None

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
            "Dividends": "dividends", "Stock Splits": "splits",
        })
        df.index.name = "date"

        # Drop metadata columns
        keep = [c for c in ("open", "high", "low", "close", "volume")
                if c in df.columns]
        df = df[keep].copy()

        for col in keep:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]

        if "volume" in df.columns:
            df = df[df["volume"].fillna(0) >= 0]

        if "close" in df.columns and "volume" in df.columns:
            df["amount"] = df["close"] * df["volume"]

        # Fix OHLC consistency
        if all(c in df.columns for c in ("open", "high", "low", "close")):
            df["open"] = df["open"].where(df["open"] > 0, df["close"])
            df["high"] = df[["open", "high", "close"]].max(axis=1)
            df["low"] = df[["open", "low", "close"]].min(axis=1)

        return df


class TencentQuoteSource(DataSource):
    """Tencent quotes — works from ANY IP (China or foreign)."""

    name = "tencent"
    priority = 0
    needs_china_direct = False
    needs_vpn = False
    _CB_ERROR_THRESHOLD = 10
    _CB_MIN_COOLDOWN = 18
    _CB_MAX_COOLDOWN = 75
    _CB_COOLDOWN_INCREMENT = 2
    _CB_HALF_OPEN_PROBE_INTERVAL = 6.0

    def get_realtime_batch(self, codes: list[str]) -> dict[str, Quote]:
        if not self.is_available():
            return {}

        from core.constants import get_exchange

        vendor_symbols: list[str] = []
        vendor_to_code: dict[str, str] = {}
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

        out: dict[str, Quote] = {}
        start_all = time.time()

        try:
            for i in range(0, len(vendor_symbols), _TENCENT_CHUNK_SIZE):
                chunk = vendor_symbols[i: i + _TENCENT_CHUNK_SIZE]
                url = "https://qt.gtimg.cn/q=" + ",".join(chunk)
                resp = self._session.get(url, timeout=6)
                resp.encoding = "gbk"  # Tencent returns GBK encoded content

                for line in resp.text.splitlines():
                    if "~" not in line or "=" not in line:
                        continue
                    try:
                        left, right = line.split("=", 1)
                        vendor_sym = left.strip().replace("v_", "")
                        payload = right.strip().strip('";')
                        if not payload or payload == "":
                            continue
                        parts = payload.split("~")
                        if len(parts) < 32:
                            continue

                        code6 = vendor_to_code.get(vendor_sym)
                        if not code6:
                            continue

                        name = str(parts[1]) if parts[1] else ""
                        price_str = parts[3].strip()
                        if not price_str:
                            continue
                        price = float(price_str)
                        if price <= 0:
                            continue

                        prev_close = float(parts[4] or 0)
                        open_px   = float(parts[5] or 0)
                        # parts[6] = volume in lots (手), multiply by 100 for shares
                        volume    = int(float(parts[6] or 0) * 100)
                        # parts[37] = amount in CNY (元)
                        amount    = float(parts[37] or 0) if len(parts) > 37 else 0.0
                        # parts[33] = high, parts[34] = low for the day
                        high_px   = float(parts[33] or price) if len(parts) > 33 else price
                        low_px    = float(parts[34] or price) if len(parts) > 34 else price
                        # parts[30] = bid1, parts[32] = ask1
                        bid_px    = float(parts[9] or 0)   if len(parts) > 9  else 0.0
                        ask_px    = float(parts[19] or 0)  if len(parts) > 19 else 0.0

                        # Validate price bounds (sanity check)
                        if prev_close > 0:
                            ratio = price / prev_close
                            if ratio > 1.25 or ratio < 0.75:
                                # Likely bad data; skip
                                log.debug(
                                    "Tencent: suspicious price for %s: "
                                    "price=%.2f prev_close=%.2f",
                                    code6, price, prev_close
                                )
                                continue

                        # Fix OHLC bounds
                        open_px = open_px if open_px > 0 else price
                        high_px = max(high_px, open_px, price)
                        low_px  = min(low_px,  open_px, price)
                        if low_px <= 0:
                            low_px = price

                        chg = price - prev_close if prev_close > 0 else 0.0
                        chg_pct = (chg / prev_close * 100) if prev_close > 0 else 0.0

                        out[code6] = Quote(
                            code=code6, name=name,
                            price=price,
                            open=open_px, high=high_px, low=low_px,
                            close=prev_close,
                            volume=volume, amount=amount,
                            change=chg, change_pct=chg_pct,
                            bid=bid_px, ask=ask_px,
                            source=self.name, is_delayed=False,
                            latency_ms=0.0,
                        )
                    except Exception as exc:
                        log.debug("Tencent parse error line: %s", exc)
                        continue

            latency = (time.time() - start_all) * 1000
            self._record_success(latency)
            for q in out.values():
                q.latency_ms = latency
            log.debug(
                "Tencent batch: %d/%d quotes fetched in %.0fms",
                len(out), len(codes), latency
            )
            return out

        except Exception as exc:
            self._record_error(str(exc))
            log.debug("Tencent batch failed: %s", exc)
            return {}

    def get_realtime(self, code: str) -> Quote | None:
        res = self.get_realtime_batch([code])
        return res.get(str(code).zfill(6))

    def get_history(self, code: str, days: int) -> pd.DataFrame:
        inst = {
            "market": "CN", "asset": "EQUITY",
            "symbol": str(code).zfill(6)
        }
        return self.get_history_instrument(inst, days=days, interval="1d")

    def get_history_instrument(
        self, inst: dict, days: int, interval: str = "1d"
    ) -> pd.DataFrame:
        if inst.get("market") != "CN" or inst.get("asset") != "EQUITY":
            return pd.DataFrame()
        if str(interval).lower() != "1d":
            return pd.DataFrame()

        code6 = str(inst.get("symbol") or "").zfill(6)
        if not code6.isdigit() or len(code6) != 6:
            return pd.DataFrame()

        from core.constants import get_exchange
        ex = get_exchange(code6)
        prefix = {"SSE": "sh", "SZSE": "sz", "BSE": "bj"}.get(ex)
        if not prefix:
            return pd.DataFrame()

        vendor_symbol = f"{prefix}{code6}"
        start_t = time.time()
        try:
            # Request more bars than needed to account for gaps
            fetch_count = max(100, int(days) + 60)
            url = (
                "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                f"?param={vendor_symbol},day,,,{fetch_count},qfq"
            )
            resp = self._session.get(url, timeout=10)
            resp.encoding = "utf-8"
            if resp.status_code != 200:
                self._record_error(f"HTTP {resp.status_code}")
                return pd.DataFrame()
            payload = resp.text
            if not payload:
                return pd.DataFrame()
            data = self._parse_daily_kline(payload, vendor_symbol)
            if data.empty:
                return pd.DataFrame()
            latency = (time.time() - start_t) * 1000.0
            self._record_success(latency)
            log.debug(
                "Tencent daily %s: %d bars in %.0fms",
                code6, len(data), latency
            )
            return data.tail(max(1, int(days)))
        except Exception as exc:
            self._record_error(str(exc))
            log.debug("Tencent history failed for %s: %s", code6, exc)
            return pd.DataFrame()

    @staticmethod
    def _parse_daily_kline(payload_text: str, vendor_symbol: str) -> pd.DataFrame:
        """Parse Tencent qfq daily K-line JSON response."""
        text = str(payload_text or "").strip()
        if not text:
            return pd.DataFrame()

        payload = None
        try:
            payload = json.loads(text)
        except Exception:
            # Handle JSONP wrapper
            left = text.find("{")
            right = text.rfind("}")
            if left < 0 or right <= left:
                return pd.DataFrame()
            try:
                payload = json.loads(text[left: right + 1])
            except Exception:
                return pd.DataFrame()

        if not isinstance(payload, dict):
            return pd.DataFrame()

        data_root = payload.get("data")
        if not isinstance(data_root, dict):
            return pd.DataFrame()

        item = data_root.get(vendor_symbol)
        if not isinstance(item, dict):
            return pd.DataFrame()

        # Try multiple key names Tencent uses
        rows = (
            item.get("qfqday")
            or item.get("day")
            or item.get("hfqday")
            or []
        )
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame()

        out_rows = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 6:
                continue
            try:
                # Tencent format: [date, open, close, high, low, volume, ...]
                date_str = str(row[0]).strip()
                date = pd.to_datetime(date_str, errors="coerce")
                if pd.isna(date):
                    continue

                open_px  = float(row[1] or 0)
                close_px = float(row[2] or 0)
                high_px  = float(row[3] or 0)
                low_px   = float(row[4] or 0)
                # Volume in Tencent is in shares (not lots)
                vol      = float(row[5] or 0)

                if close_px <= 0:
                    continue

                # Fix OHLC
                open_px = open_px if open_px > 0 else close_px
                high_px = max(high_px, open_px, close_px)
                low_px  = min(low_px,  open_px, close_px)
                if low_px <= 0:
                    low_px = close_px

                if high_px < low_px:
                    continue

                # Amount: Tencent sometimes provides index 6
                amount = 0.0
                if len(row) > 6:
                    try:
                        amount = float(row[6] or 0)
                    except Exception:
                        amount = close_px * max(0.0, vol)
                else:
                    amount = close_px * max(0.0, vol)

                out_rows.append({
                    "date":   date,
                    "open":   open_px,
                    "high":   high_px,
                    "low":    low_px,
                    "close":  close_px,
                    "volume": max(0.0, vol),
                    "amount": max(0.0, amount),
                })
            except Exception:
                continue

        if not out_rows:
            return pd.DataFrame()

        df = (
            pd.DataFrame(out_rows)
            .dropna(subset=["date"])
            .sort_values("date")
            .set_index("date")
        )
        return df


class DataFetcher:
    """
    High-performance data fetcher with automatic network-aware source
    selection, local DB caching, and multi-source fallback.
    """

    def __init__(self):
        self._all_sources: list[DataSource] = []
        self._cache = get_cache()
        self._db = get_database()
        self._rate_limiter = threading.Semaphore(CONFIG.data.parallel_downloads)
        self._request_times: dict[str, float] = {}
        self._min_interval: float = 0.5
        self._intraday_interval: float = 1.2

        self._last_good_quotes: dict[str, Quote] = {}
        self._last_good_lock = threading.RLock()

        # Micro-caches
        self._rt_cache_lock = threading.RLock()
        self._rt_batch_microcache: dict[str, object] = {
            "ts": 0.0, "key": None, "data": {},
        }
        self._rt_single_microcache: dict[str, dict[str, object]] = {}

        self._rate_lock = threading.Lock()
        self._last_source_fail_warn_ts: dict[str, float] = {}
        self._source_fail_warn_cooldown_s: float = 45.0
        self._last_source_fail_warn_global_ts: float = 0.0
        self._source_fail_warn_global_cooldown_s: float = 8.0
        self._last_network_mode: tuple[bool, bool, bool] | None = None
        self._last_network_force_refresh_ts: float = 0.0
        self._network_force_refresh_cooldown_s: float = 20.0
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
                        "Data source %s initialized "
                        "(china_direct=%s, vpn=%s)",
                        source.name,
                        source.needs_china_direct,
                        source.needs_vpn,
                    )
            except Exception as exc:
                log.warning("Failed to init %s: %s", source_cls.__name__, exc)

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

                def get_realtime(self, code: str) -> Quote | None:
                    return None

            self._all_sources.append(LocalDatabaseSource(db))
            log.info("Data source localdb initialized")

        except Exception as exc:
            log.warning("Failed to init localdb source: %s", exc)

    @property
    def _sources(self) -> list[DataSource]:
        """Backward-compatible alias."""
        return self._all_sources

    def _get_active_sources(self) -> list[DataSource]:
        """Get sources prioritized by current network environment."""
        from core.network import get_network_env
        env = get_network_env()

        net_sig = (
            bool(env.is_china_direct),
            bool(getattr(env, "eastmoney_ok", False)),
            bool(getattr(env, "yahoo_ok", False)),
        )
        if self._last_network_mode is None:
            self._last_network_mode = net_sig
        elif net_sig != self._last_network_mode:
            self._last_network_mode = net_sig
            for s in self._all_sources:
                with s._lock:
                    s.status.consecutive_errors = 0
                    s.status.disabled_until = None
                    s.status.available = True
            with self._rate_lock:
                self._request_times.clear()
            log.info(
                "Network mode changed → cooldowns reset "
                "(%s)",
                "CHINA_DIRECT" if env.is_china_direct else "VPN_FOREIGN",
            )

        active = [s for s in self._all_sources if s.is_available()]
        ranked = sorted(
            active,
            key=lambda s: (-self._source_health_score(s, env), s.priority),
        )
        return ranked

    def _source_health_score(self, source: DataSource, env) -> float:
        """Score a source by network suitability + recent health."""
        score = 0.0

        if source.name == "localdb":
            score += 120.0
        elif env.is_china_direct:
            eastmoney_ok = bool(getattr(env, "eastmoney_ok", False))
            if source.name == "akshare":
                score += 90.0 if eastmoney_ok else 8.0
            elif source.name == "tencent":
                score += 55.0 if eastmoney_ok else 88.0
            elif source.name == "yahoo":
                score += 10.0
        else:
            if source.name == "yahoo":
                score += 90.0
            elif source.name == "tencent":
                score += 60.0
            elif source.name == "akshare":
                score += 8.0

        try:
            if source.is_suitable_for_network():
                score += 15.0
            else:
                score -= 40.0
        except Exception:
            score -= 5.0

        st = source.status
        attempts = max(1, int(st.success_count + st.error_count))
        success_rate = float(st.success_count) / attempts
        score += 30.0 * success_rate

        if st.avg_latency_ms > 0:
            score -= min(25.0, st.avg_latency_ms / 200.0)

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

    def get_realtime_batch(self, codes: list[str]) -> dict[str, Quote]:
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

        result: dict[str, Quote] = {}

        # Batch-capable sources first
        sources = self._get_active_sources()
        for source in sources:
            fn = getattr(source, "get_realtime_batch", None)
            if not callable(fn):
                continue
            remaining = [c for c in cleaned if c not in result]
            if not remaining:
                break
            try:
                out = fn(remaining)
                if isinstance(out, dict):
                    for code, q in out.items():
                        code6 = self.clean_code(code)
                        if not code6 or code6 not in remaining:
                            continue
                        if q and float(getattr(q, "price", 0.0) or 0.0) > 0:
                            result[code6] = q
            except Exception as exc:
                log.debug("Batch quote source %s failed: %s", source.name, exc)
                continue

        # SpotCache fill for missing
        missing = [c for c in cleaned if c not in result]
        if missing:
            self._fill_from_spot_cache(missing, result)

        # Per-symbol fallback for remaining (only sources WITHOUT batch method)
        missing = [c for c in cleaned if c not in result]
        if missing:
            self._fill_from_single_source_quotes(missing, result, sources)

        # Force network refresh and retry once if still missing
        missing = [c for c in cleaned if c not in result]
        if missing and self._maybe_force_network_refresh():
            retry_sources = self._get_active_sources()
            for source in retry_sources:
                fn = getattr(source, "get_realtime_batch", None)
                if not callable(fn):
                    continue
                remaining = [c for c in cleaned if c not in result]
                if not remaining:
                    break
                try:
                    out = fn(remaining)
                    if isinstance(out, dict):
                        for code, q in out.items():
                            code6 = self.clean_code(code)
                            if not code6 or code6 not in remaining:
                                continue
                            if q and float(getattr(q, "price", 0.0) or 0.0) > 0:
                                result[code6] = q
                except Exception:
                    continue
            missing = [c for c in cleaned if c not in result]
            if missing:
                self._fill_from_spot_cache(missing, result)
                self._fill_from_single_source_quotes(
                    [c for c in cleaned if c not in result],
                    result,
                    retry_sources,
                )

        # Last-good fallback
        missing = [c for c in cleaned if c not in result]
        if missing:
            last_good = self._fallback_last_good(missing)
            for code, quote in last_good.items():
                if code not in result:
                    result[code] = quote

        # DB last-close fallback
        missing = [c for c in cleaned if c not in result]
        if missing:
            last_close = self._fallback_last_close_from_db(missing)
            for code, quote in last_close.items():
                if code not in result:
                    result[code] = quote

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
        self, missing: list[str], result: dict[str, Quote]
    ) -> None:
        """Attempt to fill missing quotes from EastMoney spot cache."""
        try:
            cache = get_spot_cache()
            for c in missing:
                if c in result:
                    continue
                q = cache.get_quote(c)
                if q and q.get("price", 0) and q["price"] > 0:
                    result[c] = Quote(
                        code=c,
                        name=q.get("name", ""),
                        price=float(q["price"]),
                        open=float(q.get("open") or 0),
                        high=float(q.get("high") or 0),
                        low=float(q.get("low") or 0),
                        close=float(q.get("close") or 0),
                        volume=int(q.get("volume") or 0),
                        amount=float(q.get("amount") or 0),
                        change=float(q.get("change") or 0),
                        change_pct=float(q.get("change_pct") or 0),
                        source="spot_cache",
                        is_delayed=False,
                        latency_ms=0.0,
                    )
        except Exception as exc:
            log.debug(
                "Spot-cache quote fill failed (symbols=%d): %s",
                len(missing), exc
            )

    def _fill_from_single_source_quotes(
        self,
        missing: list[str],
        result: dict[str, Quote],
        sources: list[DataSource],
    ) -> None:
        """
        Fill missing symbols using per-symbol source APIs.
        Only uses sources that do NOT have a batch method (to avoid double-calling).
        """
        if not missing:
            return
        remaining = list(dict.fromkeys(
            self.clean_code(c) for c in missing if c
        ))
        if not remaining:
            return

        for source in sources:
            if not remaining:
                break
            # FIXED: skip sources that HAVE batch (already tried above)
            # Only use sources that only have per-symbol get_realtime
            fn = getattr(source, "get_realtime_batch", None)
            if callable(fn):
                continue  # already tried via batch path

            next_remaining: list[str] = []
            for code6 in remaining:
                if code6 in result:
                    continue
                try:
                    q = source.get_realtime(code6)
                    if q and float(getattr(q, "price", 0.0) or 0.0) > 0:
                        result[code6] = q
                    else:
                        next_remaining.append(code6)
                except Exception:
                    next_remaining.append(code6)
            remaining = next_remaining

    def _fallback_last_good(self, codes: list[str]) -> dict[str, Quote]:
        """Return last-good quotes if they are recent enough."""
        result: dict[str, Quote] = {}
        with self._last_good_lock:
            for c in codes:
                q = self._last_good_quotes.get(c)
                if q and q.price > 0:
                    age = self._quote_age_seconds(q)
                    if age <= _LAST_GOOD_MAX_AGE:
                        result[c] = self._mark_quote_as_delayed(q)
        return result

    @staticmethod
    def _mark_quote_as_delayed(q: Quote) -> Quote:
        """Clone quote for fallback use and mark as delayed."""
        try:
            src = str(getattr(q, "source", "") or "")
            return replace(
                q,
                source=src if src else "last_good",
                is_delayed=True,
                latency_ms=max(float(getattr(q, "latency_ms", 0.0) or 0.0), 1.0),
            )
        except Exception:
            return q

    @staticmethod
    def _quote_age_seconds(q: Quote | None) -> float:
        """Compute quote age robustly for naive and timezone-aware timestamps."""
        if q is None:
            return float("inf")
        ts = getattr(q, "timestamp", None)
        if ts is None:
            return float("inf")
        try:
            if getattr(ts, "tzinfo", None) is not None:
                now = datetime.now(tz=ts.tzinfo)
            else:
                now = datetime.now()
            return max(0.0, float((now - ts).total_seconds()))
        except Exception:
            return float("inf")

    def _fallback_last_close_from_db(
        self, codes: list[str]
    ) -> dict[str, Quote]:
        """Fallback quote from local DB (last close)."""
        out: dict[str, Quote] = {}
        for code in codes:
            code6 = self.clean_code(code)
            if not code6:
                continue
            try:
                df = self._db.get_bars(code6, limit=1)
                if df is None or df.empty:
                    continue
                row = df.iloc[-1]
                px = float(row.get("close", 0.0) or 0.0)
                if px <= 0:
                    continue
                ts = None
                try:
                    ts = df.index[-1].to_pydatetime()
                except Exception:
                    ts = datetime.now()
                out[code6] = Quote(
                    code=code6, name="",
                    price=px,
                    open=float(row.get("open", px) or px),
                    high=float(row.get("high", px) or px),
                    low=float(row.get("low", px) or px),
                    close=px,
                    volume=int(row.get("volume", 0) or 0),
                    amount=float(row.get("amount", 0.0) or 0.0),
                    change=0.0, change_pct=0.0,
                    source="localdb_last_close",
                    is_delayed=True, latency_ms=0.0,
                    timestamp=ts,
                )
            except Exception:
                continue
        return out

    def _maybe_force_network_refresh(self) -> bool:
        """Force network redetection at most once per cooldown window."""
        now = time.time()
        if (
            now - float(self._last_network_force_refresh_ts)
            < self._network_force_refresh_cooldown_s
        ):
            return False
        self._last_network_force_refresh_ts = now
        try:
            from core.network import get_network_env
            _ = get_network_env(force_refresh=True)
            return True
        except Exception:
            return False

    def _fetch_from_sources_instrument(
        self,
        inst: dict,
        days: int,
        interval: str = "1d",
        include_localdb: bool = True,
    ) -> pd.DataFrame:
        """Fetch from active sources with smart fallback."""
        sources = self._get_active_sources()
        if not include_localdb:
            sources = [
                s for s in sources if str(getattr(s, "name", "")) != "localdb"
            ]

        if not sources:
            log.warning(
                "No active sources for %s (%s), trying all as fallback",
                inst.get("symbol"), interval,
            )
            sources = [s for s in self._all_sources if s.name != "localdb"]

        if not sources:
            log.warning("No sources at all for %s (%s)", inst.get("symbol"), interval)
            return pd.DataFrame()

        # For CN equity, sort by preferred source for current network
        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            from core.network import get_network_env
            env = get_network_env()
            if env.is_china_direct and bool(getattr(env, "eastmoney_ok", False)):
                preferred = "akshare"
            elif bool(getattr(env, "tencent_ok", False)):
                preferred = "tencent"
            elif env.yahoo_ok:
                preferred = "yahoo"
            else:
                preferred = "localdb"

            sources.sort(key=lambda s: (
                0 if s.name == preferred else
                1 if s.name == "localdb" else 2,
                s.priority,
            ))

        log.debug(
            "Sources for %s (%s): %s",
            inst.get("symbol"), interval,
            [s.name for s in sources],
        )

        with self._rate_limiter:
            errors: list[str] = []
            iv_norm = self._normalize_interval_token(interval)
            is_intraday = iv_norm not in {"1d", "1wk", "1mo"}
            collected: list[dict] = []

            for src_rank, source in enumerate(sources):
                try:
                    self._rate_limit(source.name, interval)
                    df = self._try_source_instrument(
                        source, inst, days, interval
                    )
                    if df is None or df.empty:
                        log.debug(
                            "%s returned empty for %s (%s)",
                            source.name, inst.get("symbol"), interval,
                        )
                        continue

                    df = self._clean_dataframe(df, interval=interval)
                    if df.empty:
                        log.debug(
                            "%s returned unusable rows for %s (%s)",
                            source.name, inst.get("symbol"), interval,
                        )
                        continue

                    quality = (
                        self._intraday_frame_quality(df, interval)
                        if is_intraday
                        else {
                            "score": 1.0,
                            "rows": float(len(df)),
                            "stale_ratio": 0.0,
                            "doji_ratio": 0.0,
                            "zero_vol_ratio": 0.0,
                            "extreme_ratio": 0.0,
                            "suspect": False,
                        }
                    )

                    min_required = max(5, min(days // 8, 20))
                    row_count = len(df)
                    log.debug(
                        "%s: %d bars for %s (%s) [score=%.3f]",
                        source.name, row_count, inst.get("symbol"),
                        interval, float(quality.get("score", 0.0)),
                    )

                    collected.append({
                        "source":  source.name,
                        "rank":    int(src_rank),
                        "df":      df,
                        "quality": quality,
                        "rows":    row_count,
                    })

                    # Stop early if we have enough good data
                    if row_count >= min_required and float(quality.get("score", 0.0)) >= 0.50:
                        if not is_intraday:
                            break  # Daily: first good source wins
                        # Intraday: keep collecting to pick best quality

                except Exception as exc:
                    errors.append(f"{source.name}: {exc}")
                    log.debug(
                        "%s failed for %s (%s): %s",
                        source.name, inst.get("symbol"), interval, exc,
                    )
                    continue

        if not collected:
            if errors:
                severe = [e for e in errors if not self._is_expected_no_data_error(e)]
                symbol = str(inst.get("symbol") or "")
                if severe and self._should_emit_source_fail_warning(symbol, interval):
                    log.warning(
                        "All sources failed for %s (%s): %s",
                        symbol, interval, "; ".join(severe[:3]),
                    )
                else:
                    log.debug(
                        "No usable history for %s (%s)",
                        symbol, interval,
                    )
            return pd.DataFrame()

        # Pick best result
        try:
            if is_intraday:
                best = max(
                    collected,
                    key=lambda item: (
                        float(dict(item.get("quality") or {}).get("score", 0.0)),
                        int(item.get("rows", 0)),
                        -int(item.get("rank", 0)),
                    ),
                )
                best_df = best["df"]
                best_q = dict(best.get("quality") or {})
                best_score = float(best_q.get("score", 0.0))
                best_source = str(best.get("source", "unknown"))

                # Opportunistically extend from alternatives
                bpd = float(BARS_PER_DAY.get(iv_norm, 1.0))
                target_rows = int(max(120, min(float(days) * bpd, 2200.0)))
                if len(best_df) < target_rows and len(collected) > 1:
                    candidates = sorted(
                        [c for c in collected if c is not best],
                        key=lambda item: (
                            float(dict(item.get("quality") or {}).get("score", 0.0)),
                            int(item.get("rows", 0)),
                            -int(item.get("rank", 0)),
                        ),
                        reverse=True,
                    )
                    out = best_df.copy()
                    for item in candidates:
                        q = dict(item.get("quality") or {})
                        if float(q.get("score", 0.0)) < 0.25:
                            continue
                        if bool(q.get("suspect", False)) and float(q.get("score", 0.0)) < best_score:
                            continue
                        df_alt = item["df"]
                        if df_alt.empty:
                            continue
                        extra = df_alt.loc[~df_alt.index.isin(out.index)]
                        if extra.empty:
                            continue
                        combined = pd.concat([extra, out], axis=0)
                        out = self._clean_dataframe(combined, interval=interval)
                        if len(out) >= target_rows:
                            break
                    best_df = out

                log.debug(
                    "Selected %s for %s (%s): score=%.3f rows=%d",
                    best_source, inst.get("symbol"), interval,
                    best_score, len(best_df),
                )
                return best_df

            # Daily: merge all sources, deduplicate keeping most recent fetch
            parts = [item["df"] for item in collected if not item["df"].empty]
            if not parts:
                return pd.DataFrame()

            # Merge: later sources fill gaps only (primary source rows win)
            # Sort collected by score descending so best source is last (wins in dedup)
            collected_by_score = sorted(
                collected,
                key=lambda item: float(dict(item.get("quality") or {}).get("score", 0.0)),
            )
            merged_parts = [item["df"] for item in collected_by_score if not item["df"].empty]
            merged = self._clean_dataframe(
                pd.concat(merged_parts, axis=0),
                interval=interval,
            )
            return merged

        except Exception as exc:
            log.debug(
                "Failed to select/merge history for %s: %s",
                inst.get("symbol"), exc,
            )
            # Return the first collected result as safe fallback
            if collected:
                return collected[0]["df"]
            return pd.DataFrame()

    @staticmethod
    def _is_expected_no_data_error(err_msg: str) -> bool:
        msg = str(err_msg or "").lower()
        expected = (
            "no data", "returned empty", "empty dataframe",
            "not found", "404", "no history", "symbol not found",
        )
        return any(k in msg for k in expected)

    def _should_emit_source_fail_warning(
        self, symbol: str, interval: str
    ) -> bool:
        key = f"{symbol}:{interval}"
        now = time.time()
        if (
            now - float(self._last_source_fail_warn_global_ts)
            < self._source_fail_warn_global_cooldown_s
        ):
            return False
        last = float(self._last_source_fail_warn_ts.get(key, 0.0))
        if (now - last) < self._source_fail_warn_cooldown_s:
            return False
        self._last_source_fail_warn_ts[key] = now
        self._last_source_fail_warn_global_ts = now
        return True

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
            if s.startswith(p) and len(s) > len(p):
                candidate = s[len(p):]
                if candidate.replace(".", "").isdigit():
                    s = candidate
                    break

        suffixes = (".SS", ".SZ", ".BJ", ".ss", ".sz", ".bj")
        for suf in suffixes:
            if s.endswith(suf):
                s = s[: -len(suf)]
                break

        digits = "".join(ch for ch in s if ch.isdigit())
        return digits.zfill(6) if digits else ""

    @staticmethod
    def _normalize_interval_token(interval: str | None) -> str:
        iv = str(interval or "1d").strip().lower()
        aliases = {
            "1h":    "60m",
            "60min": "60m",
            "daily": "1d",
            "day":   "1d",
            "1day":  "1d",
            "1440m": "1d",
        }
        return aliases.get(iv, iv)

    @staticmethod
    def _interval_seconds(interval: str | None) -> int:
        iv = str(interval or "1d").strip().lower()
        aliases = {
            "1h":    "60m",
            "60min": "60m",
            "daily": "1d",
            "day":   "1d",
            "1day":  "1d",
            "1440m": "1d",
        }
        iv = aliases.get(iv, iv)
        mapping = {
            "1m":  60,
            "2m":  120,
            "5m":  300,
            "15m": 900,
            "30m": 1800,
            "60m": 3600,
            "1d":  86400,
            "1wk": 86400 * 7,
            "1mo": 86400 * 30,
        }
        if iv in mapping:
            return int(mapping[iv])
        try:
            if iv.endswith("m"):
                return max(1, int(float(iv[:-1]) * 60))
            if iv.endswith("s"):
                return max(1, int(float(iv[:-1])))
            if iv.endswith("h"):
                return max(1, int(float(iv[:-1]) * 3600))
        except Exception as exc:
            log.debug("Invalid interval token (%s): %s", iv, exc)
        return 60

    @staticmethod
    def _now_shanghai_naive() -> datetime:
        """Return current Asia/Shanghai wall time as a naive datetime."""
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(tz=ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
        except Exception:
            return datetime.utcnow()

    @staticmethod
    def _intraday_quality_caps(
        interval: str | None,
    ) -> tuple[float, float, float, float]:
        """
        Return (body_cap, span_cap, wick_cap, jump_cap) for intraday cleanup.

        Values are deliberately generous to avoid corrupting legitimate price
        moves (China A-shares can move ±10% intraday; ST stocks ±5%).
        Only truly malformed bars are removed.
        """
        iv = DataFetcher._normalize_interval_token(interval)
        caps = _INTRADAY_CAPS.get(iv)
        if caps:
            return caps
        # Default: conservative daily-like caps
        return 0.15, 0.22, 0.16, 0.22

    @classmethod
    def _intraday_frame_quality(
        cls,
        df: pd.DataFrame,
        interval: str,
    ) -> dict[str, float | bool]:
        """
        Score intraday frame quality.
        Higher score = cleaner and more usable bars.
        """
        if df is None or df.empty:
            return {
                "score": 0.0, "rows": 0.0,
                "stale_ratio": 1.0, "doji_ratio": 1.0,
                "zero_vol_ratio": 1.0, "extreme_ratio": 1.0,
                "suspect": True,
            }

        out = cls._clean_dataframe(df, interval=interval)
        if out.empty:
            return {
                "score": 0.0, "rows": 0.0,
                "stale_ratio": 1.0, "doji_ratio": 1.0,
                "zero_vol_ratio": 1.0, "extreme_ratio": 1.0,
                "suspect": True,
            }

        body_cap, span_cap, wick_cap, _ = cls._intraday_quality_caps(interval)
        close_safe = out["close"].clip(lower=1e-8)
        rows_n = float(len(out))

        body = (out["open"] - out["close"]).abs() / close_safe
        span = (out["high"] - out["low"]).abs() / close_safe
        oc_top = out[["open", "close"]].max(axis=1)
        oc_bot = out[["open", "close"]].min(axis=1)
        upper_wick = (out["high"] - oc_top).clip(lower=0.0) / close_safe
        lower_wick = (oc_bot - out["low"]).clip(lower=0.0) / close_safe

        vol = (
            out["volume"] if "volume" in out.columns
            else pd.Series(0.0, index=out.index)
        ).fillna(0)

        zero_vol = vol <= 0

        # Stale detection: same close, flat OHLC, zero volume
        same_close = out["close"].diff().abs() <= (close_safe * 1e-6)
        flat_body  = body <= 1e-6
        flat_span  = span <= 2e-6
        stale_flat = same_close & flat_body & flat_span & zero_vol

        # Doji: near-zero body relative to span
        doji_ratio      = float((body <= (span.clip(lower=1e-8) * 0.05)).mean())
        stale_ratio     = float(stale_flat.mean())
        zero_vol_ratio  = float(zero_vol.mean())

        # Extreme: bars with body/span/wick far above expected cap
        extreme_mask = (
            (body > float(body_cap) * 2.0)
            | (span > float(span_cap) * 2.0)
            | (upper_wick > float(wick_cap) * 2.0)
            | (lower_wick > float(wick_cap) * 2.0)
        )
        extreme_ratio = float(extreme_mask.mean())

        # Score: depth dominates; penalize quality issues
        depth_score = min(1.0, rows_n / 600.0)
        score = (
            (0.50 * depth_score)
            + (0.25 * (1.0 - min(1.0, stale_ratio * 2.0)))
            + (0.15 * (1.0 - min(1.0, zero_vol_ratio)))
            + (0.10 * (1.0 - min(1.0, extreme_ratio * 3.0)))
        )
        if doji_ratio > 0.95:
            score -= float((doji_ratio - 0.95) * 1.5)
        score = float(max(0.0, min(1.0, score)))

        suspect = bool(
            (rows_n < 40)
            or (stale_ratio >= 0.60)
            or (extreme_ratio >= 0.15)
            or (doji_ratio >= 0.98 and zero_vol_ratio >= 0.85)
        )
        return {
            "score":          score,
            "rows":           rows_n,
            "stale_ratio":    float(stale_ratio),
            "doji_ratio":     float(doji_ratio),
            "zero_vol_ratio": float(zero_vol_ratio),
            "extreme_ratio":  float(extreme_ratio),
            "suspect":        suspect,
        }

    @classmethod
    def _to_shanghai_naive_ts(cls, value: object) -> pd.Timestamp:
        """
        Parse one timestamp-like value → Asia/Shanghai naive time.
        Returns NaT on failure.
        """
        if value is None:
            return pd.NaT

        try:
            if isinstance(value, (int, float, np.integer, np.floating)):
                v = float(value)
                if not np.isfinite(v) or abs(v) < 1e9:
                    return pd.NaT
                if abs(v) >= 1e11:
                    v /= 1000.0
                ts = pd.to_datetime(v, unit="s", errors="coerce", utc=True)
            else:
                text = str(value).strip()
                if not text:
                    return pd.NaT
                if text.isdigit():
                    num = float(text)
                    if abs(num) < 1e9:
                        return pd.NaT
                    if abs(num) >= 1e11:
                        num /= 1000.0
                    ts = pd.to_datetime(num, unit="s", errors="coerce", utc=True)
                else:
                    ts = pd.to_datetime(value, errors="coerce")
        except Exception:
            return pd.NaT

        if pd.isna(ts):
            return pd.NaT

        try:
            ts_obj = pd.Timestamp(ts)
        except Exception:
            return pd.NaT

        try:
            if ts_obj.tzinfo is not None:
                ts_obj = ts_obj.tz_convert("Asia/Shanghai").tz_localize(None)
        except Exception:
            try:
                ts_obj = ts_obj.tz_localize(None)
            except Exception:
                return pd.NaT
        return ts_obj

    @classmethod
    def _normalize_datetime_index(
        cls,
        idx: object,
    ) -> pd.DatetimeIndex | None:
        """
        Convert an index-like object to DatetimeIndex in Asia/Shanghai naive time.
        Returns None when conversion is unreliable.
        """
        if isinstance(idx, pd.DatetimeIndex):
            out = idx
            try:
                if out.tz is not None:
                    out = out.tz_convert("Asia/Shanghai").tz_localize(None)
            except Exception:
                try:
                    out = out.tz_localize(None)
                except Exception as exc:
                    log.debug("DatetimeIndex tz normalization failed: %s", exc)
            return pd.DatetimeIndex(out)

        values = list(idx) if idx is not None else []
        if not values:
            return None

        parsed = [cls._to_shanghai_naive_ts(v) for v in values]
        dt = pd.DatetimeIndex(parsed)
        valid_ratio = float(dt.notna().sum()) / float(max(1, len(dt)))
        if valid_ratio < 0.80:
            return None
        return dt

    @classmethod
    def _clean_dataframe(
        cls,
        df: pd.DataFrame,
        interval: str | None = None,
    ) -> pd.DataFrame:
        """
        Standardize and validate an OHLCV dataframe.

        Key fixes vs original:
        - Intraday caps are generous (won't corrupt legitimate ±10% moves)
        - Stale bar removal keeps every 10th bar to preserve continuity
        - Deduplication keeps LAST occurrence (newest fetch wins on merge)
        - Jump correction only clips bars that are truly impossible (>20%)
        """
        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()
        iv = cls._normalize_interval_token(interval)

        # ── 1. Index → DatetimeIndex ──────────────────────────────────────────
        norm_idx = cls._normalize_datetime_index(out.index)
        has_dt_index = norm_idx is not None
        if norm_idx is not None:
            out.index = norm_idx

        if not has_dt_index:
            parsed_dt = None
            for col in ("datetime", "timestamp", "date", "time"):
                if col not in out.columns:
                    continue
                dt = cls._normalize_datetime_index(out[col])
                if dt is None or len(dt) == 0:
                    continue
                if float(dt.notna().sum()) / float(len(dt)) >= 0.80:
                    parsed_dt = dt
                    break

            if parsed_dt is None:
                try:
                    idx_num = pd.to_numeric(
                        pd.Series(out.index, dtype=object), errors="coerce"
                    )
                    numeric_ratio = (
                        float(idx_num.notna().sum()) / float(len(idx_num))
                        if len(idx_num) > 0 else 0.0
                    )
                except Exception:
                    numeric_ratio = 0.0

                if numeric_ratio < 0.60:
                    dt = cls._normalize_datetime_index(out.index)
                    if dt is not None and len(dt) > 0:
                        if float(dt.notna().sum()) / float(len(dt)) >= 0.80:
                            parsed_dt = dt

            if parsed_dt is not None:
                out.index = parsed_dt
                has_dt_index = isinstance(out.index, pd.DatetimeIndex)
            else:
                if iv not in {"1d", "1wk", "1mo"} and len(out) > 0:
                    step = int(max(1, cls._interval_seconds(iv)))
                    end = cls._now_shanghai_naive()
                    out.index = pd.date_range(
                        end=end, periods=len(out), freq=f"{step}s"
                    )
                    has_dt_index = True
                else:
                    out = out.reset_index(drop=True)

        # ── 2. Deduplicate & sort ─────────────────────────────────────────────
        if has_dt_index:
            out = out[~out.index.isna()]
            # keep="last" means newest fetch wins when merging multiple sources
            out = out[~out.index.duplicated(keep="last")].sort_index()

        # ── 3. Numeric coercion ───────────────────────────────────────────────
        for c in ("open", "high", "low", "close", "volume", "amount"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        # ── 4. Basic close validity ───────────────────────────────────────────
        if "close" not in out.columns:
            return pd.DataFrame()
        out = out.dropna(subset=["close"])
        out = out[out["close"] > 0]
        if out.empty:
            return pd.DataFrame()

        # ── 5. Repair open=0 ─────────────────────────────────────────────────
        if "open" not in out.columns:
            out["open"] = out["close"]
        out["open"] = pd.to_numeric(out["open"], errors="coerce").fillna(0.0)
        out["open"] = out["open"].where(out["open"] > 0, out["close"])

        # ── 6. Repair high/low ────────────────────────────────────────────────
        if "high" not in out.columns:
            out["high"] = out[["open", "close"]].max(axis=1)
        else:
            out["high"] = pd.to_numeric(out["high"], errors="coerce")

        if "low" not in out.columns:
            out["low"] = out[["open", "close"]].min(axis=1)
        else:
            out["low"] = pd.to_numeric(out["low"], errors="coerce")

        # Ensure OHLC consistency
        out["high"] = pd.concat(
            [out["high"], out["open"], out["close"]], axis=1
        ).max(axis=1)
        out["low"] = pd.concat(
            [out["low"], out["open"], out["close"]], axis=1
        ).min(axis=1)

        # ── 7. Intraday-specific cleanup ──────────────────────────────────────
        is_intraday = iv not in {"1d", "1wk", "1mo"}
        if is_intraday:
            body_cap, span_cap, wick_cap, jump_cap = cls._intraday_quality_caps(iv)
            close_safe = out["close"].clip(lower=1e-8)

            # 7a. Fix wildly wrong open (body > cap × 3 means provider glitch)
            body_ratio = (out["open"] - out["close"]).abs() / close_safe
            bad_open = body_ratio > (float(body_cap) * 3.0)
            if bad_open.any():
                out.loc[bad_open, "open"] = out.loc[bad_open, "close"]

            # Re-compute after open fix
            oc_top = out[["open", "close"]].max(axis=1)
            oc_bot = out[["open", "close"]].min(axis=1)
            out["high"] = pd.concat([out["high"], oc_top], axis=1).max(axis=1)
            out["low"]  = pd.concat([out["low"],  oc_bot], axis=1).min(axis=1)

            # 7b. Fix extreme wicks/spans (cap × 3)
            upper_wick = (out["high"] - oc_top).clip(lower=0.0) / close_safe
            lower_wick = (oc_bot - out["low"]).clip(lower=0.0) / close_safe
            span_ratio = (out["high"] - out["low"]).abs() / close_safe
            bad_shape = (
                (span_ratio  > float(span_cap)  * 3.0)
                | (upper_wick > float(wick_cap)  * 3.0)
                | (lower_wick > float(wick_cap)  * 3.0)
            )
            if bad_shape.any():
                wick_allow = close_safe * float(wick_cap)
                out.loc[bad_shape, "high"] = (oc_top + wick_allow)[bad_shape]
                out.loc[bad_shape, "low"]  = (oc_bot - wick_allow)[bad_shape]
                out["high"] = pd.concat([out["high"], oc_top], axis=1).max(axis=1)
                out["low"]  = pd.concat([out["low"],  oc_bot], axis=1).min(axis=1)

            # 7c. Fix impossible inter-bar jumps (only intra-day, skip day boundary)
            prev_close = out["close"].shift(1)
            prev_safe  = prev_close.where(prev_close > 0, np.nan)
            jump_ratio = (out["close"] / prev_safe - 1.0).abs()
            # China limit ≈ 10% (20% for some instruments); use jump_cap from caps
            bad_jump = jump_ratio > float(jump_cap)
            if isinstance(out.index, pd.DatetimeIndex):
                day_change = (
                    pd.Series(out.index.normalize(), index=out.index)
                    .diff()
                    .ne(pd.Timedelta(0))
                )
                bad_jump = bad_jump & (~day_change.fillna(False))
            bad_jump = bad_jump.fillna(False)
            if bad_jump.any():
                prev_vals = prev_close[bad_jump].astype(float)
                curr_vals = out.loc[bad_jump, "close"].astype(float)
                signs  = np.where(curr_vals >= prev_vals, 1.0, -1.0)
                clipped = prev_vals * (1.0 + signs * float(jump_cap))
                out.loc[bad_jump, "close"] = clipped.values
                out.loc[bad_jump, "open"]  = prev_vals.values
                close_safe2 = out["close"].clip(lower=1e-8)
                oc_top2 = out[["open", "close"]].max(axis=1)
                oc_bot2 = out[["open", "close"]].min(axis=1)
                wick_allow2 = close_safe2 * float(wick_cap)
                out.loc[bad_jump, "high"] = np.minimum(
                    out.loc[bad_jump, "high"],
                    (oc_top2 + wick_allow2)[bad_jump],
                )
                out.loc[bad_jump, "low"] = np.maximum(
                    out.loc[bad_jump, "low"],
                    (oc_bot2 - wick_allow2)[bad_jump],
                )
                out["high"] = pd.concat([out["high"], oc_top2], axis=1).max(axis=1)
                out["low"]  = pd.concat([out["low"],  oc_bot2], axis=1).min(axis=1)

            # 7d. Remove consecutive stale bars (keep every 10th for continuity)
            vol_s = (
                out["volume"] if "volume" in out.columns
                else pd.Series(0.0, index=out.index)
            )
            close_safe3 = out["close"].clip(lower=1e-8)
            same_close3 = out["close"].diff().abs() <= (close_safe3 * 1e-6)
            flat_body3  = (out["open"] - out["close"]).abs() <= (close_safe3 * 1e-6)
            flat_span3  = (out["high"] - out["low"]).abs() <= (close_safe3 * 2e-6)
            stale_flat3 = same_close3 & flat_body3 & flat_span3 & (vol_s.fillna(0) <= 0)
            if stale_flat3.any():
                group_key  = stale_flat3.ne(stale_flat3.shift(fill_value=False)).cumsum()
                stale_pos  = stale_flat3.groupby(group_key).cumcount()
                drop_mask  = stale_flat3 & (stale_pos % 10 != 0)
                if drop_mask.any():
                    out = out.loc[~drop_mask]

        # ── 8. Volume ≥ 0 ────────────────────────────────────────────────────
        if "volume" in out.columns:
            out = out[out["volume"].fillna(0) >= 0]

        # ── 9. high ≥ low ────────────────────────────────────────────────────
        if "high" in out.columns and "low" in out.columns:
            out = out[out["high"].fillna(0) >= out["low"].fillna(0)]

        # ── 10. Derive amount if missing ─────────────────────────────────────
        if (
            "amount" not in out.columns
            and "close" in out.columns
            and "volume" in out.columns
        ):
            out["amount"] = out["close"] * out["volume"]

        # ── 11. Final cleanup ─────────────────────────────────────────────────
        out = out.replace([np.inf, -np.inf], np.nan)
        ohlc_cols = [c for c in ("open", "high", "low", "close") if c in out.columns]
        if ohlc_cols:
            out[ohlc_cols] = out[ohlc_cols].ffill().bfill()
        out = out.fillna(0)

        return out

    # ─────────────────────────────────────────────────────────────────────────
    # History orchestration (unchanged logic, minor robustness improvements)
    # ─────────────────────────────────────────────────────────────────────────

    def get_history(
        self,
        code: str,
        days: int = 500,
        bars: int | None = None,
        use_cache: bool = True,
        update_db: bool = True,
        instrument: dict | None = None,
        interval: str = "1d",
        max_age_hours: float | None = None,
        allow_online: bool = True,
        refresh_intraday_after_close: bool = False,
    ) -> pd.DataFrame:
        """Unified history fetcher. Priority: cache → local DB → online."""
        from core.instruments import instrument_key, parse_instrument

        inst = instrument or parse_instrument(code)
        key = instrument_key(inst)
        interval = self._normalize_interval_token(interval)
        offline = _is_offline() or (not bool(allow_online))
        force_exact_intraday = bool(
            refresh_intraday_after_close
            and self._should_refresh_intraday_exact(
                interval=interval,
                update_db=bool(update_db),
                allow_online=bool(allow_online),
            )
        )
        is_cn_equity = (
            inst.get("market") == "CN" and inst.get("asset") == "EQUITY"
        )

        count = self._resolve_requested_bar_count(
            days=days, bars=bars, interval=interval
        )
        max_days = INTERVAL_MAX_DAYS.get(interval, 10_000)
        fetch_days = min(bars_to_days(count, interval), max_days)

        if max_age_hours is not None:
            ttl = float(max_age_hours)
        elif interval == "1d":
            ttl = float(CONFIG.data.cache_ttl_hours)
        else:
            ttl = min(float(CONFIG.data.cache_ttl_hours), 1.0 / 120.0)

        cache_key = f"history:{key}:{interval}:{count}"
        stale_cached_df = pd.DataFrame()

        if use_cache and (not force_exact_intraday):
            cached_df = self._cache.get(cache_key, ttl)
            if isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
                cached_df = self._clean_dataframe(cached_df, interval=interval)
                if (
                    is_cn_equity
                    and self._normalize_interval_token(interval)
                    not in {"1d", "1wk", "1mo"}
                ):
                    cached_df = self._filter_cn_intraday_session(
                        cached_df, interval
                    )
                stale_cached_df = cached_df
                if len(cached_df) >= min(count, 100):
                    return cached_df.tail(count)
                if offline and len(cached_df) >= max(20, min(count, 80)):
                    return cached_df.tail(count)

        session_df = pd.DataFrame()
        if not force_exact_intraday:
            session_df = self._get_session_history(
                symbol=str(inst.get("symbol", code)),
                interval=interval,
                bars=count,
            )
        if (
            not force_exact_intraday
            and interval in _INTRADAY_INTERVALS
            and not session_df.empty
            and count <= 500
            and len(session_df) >= count
        ):
            return self._cache_tail(cache_key, session_df, count)

        if is_cn_equity and interval in _INTRADAY_INTERVALS:
            if force_exact_intraday:
                return self._get_history_cn_intraday_exact(
                    inst, count, fetch_days, interval, cache_key, offline,
                )
            persist_intraday_db = bool(update_db) and (
                not bool(CONFIG.is_market_open())
            )
            try:
                return self._get_history_cn_intraday(
                    inst, count, fetch_days, interval,
                    cache_key, offline, session_df,
                    persist_intraday_db=persist_intraday_db,
                )
            except TypeError:
                return self._get_history_cn_intraday(
                    inst, count, fetch_days, interval,
                    cache_key, offline, session_df,
                )

        if is_cn_equity and interval in {"1d", "1wk", "1mo"}:
            return self._get_history_cn_daily(
                inst, count, fetch_days, cache_key,
                offline, update_db, session_df, interval=interval,
            )

        # Non-CN instrument
        if offline:
            return (
                stale_cached_df.tail(count)
                if not stale_cached_df.empty
                else pd.DataFrame()
            )
        df = self._fetch_history_with_depth_retry(
            inst=inst,
            interval=interval,
            requested_count=count,
            base_fetch_days=fetch_days,
        )
        if df.empty:
            return pd.DataFrame()
        out = self._merge_parts(df, session_df, interval=interval).tail(count)
        self._cache.set(cache_key, out)
        return out

    def _should_refresh_intraday_exact(
        self,
        *,
        interval: str,
        update_db: bool,
        allow_online: bool,
    ) -> bool:
        iv = self._normalize_interval_token(interval)
        if iv in {"1d", "1wk", "1mo"}:
            return False
        if (not bool(update_db)) or (not bool(allow_online)):
            return False
        if _is_offline():
            return False
        return bool(self._is_post_close_or_preopen_window())

    @staticmethod
    def _is_post_close_or_preopen_window() -> bool:
        """True when outside regular A-share trading session."""
        try:
            from zoneinfo import ZoneInfo
            now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
        except Exception:
            now = datetime.now()

        if now.weekday() >= 5:
            return True

        try:
            from core.constants import is_trading_day
            if not is_trading_day(now.date()):
                return True
        except Exception as exc:
            log.debug("Trading-day calendar lookup failed: %s", exc)

        t = CONFIG.trading
        cur = now.time()
        morning   = t.market_open_am <= cur <= t.market_close_am
        afternoon = t.market_open_pm <= cur <= t.market_close_pm
        lunch     = t.market_close_am < cur < t.market_open_pm
        if morning or afternoon or lunch:
            return False
        return True

    @staticmethod
    def _resolve_requested_bar_count(
        days: int,
        bars: int | None,
        interval: str,
    ) -> int:
        if bars is not None:
            return max(1, int(bars))
        iv = str(interval or "1d").lower()
        if iv == "1d":
            return max(1, int(days))
        day_count = max(1, int(days))
        bpd = BARS_PER_DAY.get(iv, 1.0)
        if bpd <= 0:
            bpd = 1.0
        approx = int(math.ceil(day_count * bpd))
        max_bars = max(1, int(INTERVAL_MAX_DAYS.get(iv, 365) * bpd))
        return max(1, min(approx, max_bars))

    def _fetch_history_with_depth_retry(
        self,
        inst: dict,
        interval: str,
        requested_count: int,
        base_fetch_days: int,
    ) -> pd.DataFrame:
        """Fetch history with adaptive depth retries."""
        iv = str(interval or "1d").lower()
        max_days = int(INTERVAL_MAX_DAYS.get(iv, 10_000))
        base = max(1, int(base_fetch_days))
        candidates = [base, int(base * 2.0), int(base * 3.0)]

        tried: set[int] = set()
        best = pd.DataFrame()
        best_score = -1.0
        target = max(60, int(min(requested_count, 1200)))
        is_intraday = iv not in {"1d", "1wk", "1mo"}

        for days in candidates:
            d = max(1, min(int(days), max_days))
            if d in tried:
                continue
            tried.add(d)
            try:
                raw_df = self._fetch_from_sources_instrument(
                    inst, days=d, interval=iv,
                    include_localdb=not is_intraday,
                )
            except TypeError:
                raw_df = self._fetch_from_sources_instrument(
                    inst, days=d, interval=iv,
                )
            df = self._clean_dataframe(raw_df, interval=iv)
            if df.empty:
                continue

            if is_intraday:
                q = self._intraday_frame_quality(df, iv)
                score = float(q.get("score", 0.0))
                if (
                    score > best_score + 0.02
                    or (abs(score - best_score) <= 0.02 and len(df) > len(best))
                ):
                    best = df
                    best_score = score
            else:
                if len(df) > len(best):
                    best = df

            if len(best) >= target:
                if (not is_intraday) or (best_score >= 0.28):
                    break

        return best

    def _accept_online_intraday_snapshot(
        self,
        *,
        symbol: str,
        interval: str,
        online_df: pd.DataFrame,
        baseline_df: pd.DataFrame | None = None,
    ) -> bool:
        """Decide whether to trust an online intraday snapshot over baseline."""
        if online_df is None or online_df.empty:
            return False
        if baseline_df is None or baseline_df.empty:
            return True

        iv = self._normalize_interval_token(interval)
        oq = self._intraday_frame_quality(online_df, iv)
        bq = self._intraday_frame_quality(baseline_df, iv)
        online_score   = float(oq.get("score", 0.0))
        base_score     = float(bq.get("score", 0.0))
        online_suspect = bool(oq.get("suspect", False))

        online_fresher = False
        try:
            if (
                isinstance(online_df.index, pd.DatetimeIndex)
                and isinstance(baseline_df.index, pd.DatetimeIndex)
                and len(online_df.index) > 0
                and len(baseline_df.index) > 0
            ):
                step = int(max(1, self._interval_seconds(iv)))
                online_last = pd.Timestamp(online_df.index.max())
                base_last   = pd.Timestamp(baseline_df.index.max())
                online_fresher = bool(
                    online_last >= (base_last + pd.Timedelta(seconds=step))
                )
        except Exception:
            online_fresher = False

        reject = bool(
            (
                online_suspect
                and base_score >= (online_score + 0.08)
                and not online_fresher
            )
            or (
                float(oq.get("stale_ratio", 0.0)) >= 0.55
                and float(bq.get("stale_ratio", 0.0))
                    <= (float(oq.get("stale_ratio", 0.0)) - 0.20)
            )
            or (
                float(oq.get("rows", 0.0)) < max(40.0, float(bq.get("rows", 0.0)) * 0.20)
                and online_score < base_score
                and not online_fresher
            )
        )
        if reject:
            log.warning(
                "Rejected weak online snapshot for %s (%s): "
                "online score=%.3f stale=%.1f%% rows=%d; "
                "baseline score=%.3f rows=%d",
                str(symbol or ""), iv,
                online_score,
                float(oq.get("stale_ratio", 0.0)) * 100.0,
                int(oq.get("rows", 0.0)),
                base_score,
                int(bq.get("rows", 0.0)),
            )
            return False
        return True

    def _get_history_cn_intraday(
        self,
        inst: dict,
        count: int,
        fetch_days: int,
        interval: str,
        cache_key: str,
        offline: bool,
        session_df: pd.DataFrame | None = None,
        *,
        persist_intraday_db: bool = True,
    ) -> pd.DataFrame:
        """Handle CN equity intraday intervals."""
        code6 = str(inst["symbol"]).zfill(6)
        db_df = pd.DataFrame()
        db_limit = int(max(count * 3, count + 600))
        try:
            db_df = self._clean_dataframe(
                self._db.get_intraday_bars(
                    code6, interval=interval, limit=db_limit
                ),
                interval=interval,
            )
        except Exception as exc:
            log.warning(
                "Intraday DB read failed for %s (%s): %s",
                code6, interval, exc,
            )

        online_df = pd.DataFrame()
        if not offline:
            online_df = self._fetch_history_with_depth_retry(
                inst=inst, interval=interval,
                requested_count=count, base_fetch_days=fetch_days,
            )
            online_df = self._filter_cn_intraday_session(online_df, interval)

        baseline_df = self._merge_parts(db_df, session_df, interval=interval)
        if (
            not offline
            and not online_df.empty
            and not self._accept_online_intraday_snapshot(
                symbol=code6, interval=interval,
                online_df=online_df, baseline_df=baseline_df,
            )
        ):
            online_df = pd.DataFrame()

        if offline:
            merged = self._merge_parts(db_df, session_df, interval=interval)
        else:
            merged = self._merge_parts(
                db_df, session_df, online_df, interval=interval
            )
        merged = self._filter_cn_intraday_session(merged, interval)

        if merged.empty:
            return pd.DataFrame()

        out = self._cache_tail(cache_key, merged, count)
        if bool(persist_intraday_db):
            try:
                self._db.upsert_intraday_bars(code6, interval, out)
            except Exception as exc:
                log.warning(
                    "Intraday DB upsert failed for %s (%s): %s",
                    code6, interval, exc,
                )
        return out

    def _get_history_cn_intraday_exact(
        self,
        inst: dict,
        count: int,
        fetch_days: int,
        interval: str,
        cache_key: str,
        offline: bool,
    ) -> pd.DataFrame:
        """Post-close exact mode: prefer online bars, update DB."""
        code6 = str(inst["symbol"]).zfill(6)
        online_df = pd.DataFrame()
        if not offline:
            online_df = self._fetch_history_with_depth_retry(
                inst=inst, interval=interval,
                requested_count=count, base_fetch_days=fetch_days,
            )
            online_df = self._filter_cn_intraday_session(online_df, interval)

        db_df = pd.DataFrame()
        db_limit = int(max(count * 3, count + 600))
        try:
            db_df = self._clean_dataframe(
                self._db.get_intraday_bars(
                    code6, interval=interval, limit=db_limit
                ),
                interval=interval,
            )
            db_df = self._filter_cn_intraday_session(db_df, interval)
        except Exception as exc:
            log.warning(
                "Intraday exact DB read failed for %s (%s): %s",
                code6, interval, exc,
            )

        if (
            not offline
            and not online_df.empty
            and not self._accept_online_intraday_snapshot(
                symbol=code6, interval=interval,
                online_df=online_df, baseline_df=db_df,
            )
        ):
            online_df = pd.DataFrame()

        if online_df is None or online_df.empty:
            if db_df is None or db_df.empty:
                return pd.DataFrame()
            return self._cache_tail(cache_key, db_df, count)

        merged = self._merge_parts(db_df, online_df, interval=interval)
        merged = self._filter_cn_intraday_session(merged, interval)
        if merged.empty:
            return pd.DataFrame()

        out = self._cache_tail(cache_key, merged, count)
        try:
            self._db.upsert_intraday_bars(code6, interval, out)
        except Exception as exc:
            log.warning(
                "Intraday exact DB upsert failed for %s (%s): %s",
                code6, interval, exc,
            )
        return out

    def _get_history_cn_daily(
        self,
        inst: dict,
        count: int,
        fetch_days: int,
        cache_key: str,
        offline: bool,
        update_db: bool,
        session_df: pd.DataFrame | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Handle CN equity daily/weekly/monthly intervals."""
        iv = self._normalize_interval_token(interval)
        code6 = str(inst["symbol"]).zfill(6)
        db_limit = (
            int(max(count, fetch_days))
            if iv == "1d"
            else int(max(count * 8, fetch_days))
        )
        db_df = self._clean_dataframe(
            self._db.get_bars(inst["symbol"], limit=db_limit),
            interval="1d",
        )
        base_df = self._resample_daily_to_interval(
            self._merge_parts(db_df, session_df, interval="1d"),
            iv,
        )

        if (
            iv == "1d"
            and len(base_df) >= count
            and self._db_is_fresh_enough(code6, max_lag_days=3)
        ):
            return self._cache_tail(cache_key, base_df, count)

        if offline:
            return base_df.tail(count) if not base_df.empty else pd.DataFrame()

        online_df = self._fetch_history_with_depth_retry(
            inst=inst, interval=iv,
            requested_count=count, base_fetch_days=fetch_days,
        )
        merged = self._merge_parts(
            base_df, online_df, interval=iv
        )
        if merged.empty:
            return pd.DataFrame()

        out = self._cache_tail(cache_key, merged, count)
        if update_db and iv == "1d":
            try:
                self._db.upsert_bars(inst["symbol"], out)
            except Exception as exc:
                log.warning(
                    "Daily DB upsert failed for %s: %s",
                    str(inst.get("symbol", "")), exc,
                )
        return out

    @classmethod
    def _resample_daily_to_interval(
        cls,
        df: pd.DataFrame,
        interval: str,
    ) -> pd.DataFrame:
        """Resample daily OHLCV bars to weekly/monthly bars when requested."""
        iv = cls._normalize_interval_token(interval)
        if iv == "1d":
            return cls._clean_dataframe(df, interval="1d")
        if iv not in {"1wk", "1mo"}:
            return cls._clean_dataframe(df, interval=iv)

        daily = cls._clean_dataframe(df, interval="1d")
        if daily.empty or not isinstance(daily.index, pd.DatetimeIndex):
            return pd.DataFrame()

        rule = "W-FRI" if iv == "1wk" else "ME"
        agg: dict[str, str] = {}
        if "open" in daily.columns:
            agg["open"] = "first"
        if "high" in daily.columns:
            agg["high"] = "max"
        if "low" in daily.columns:
            agg["low"] = "min"
        if "close" in daily.columns:
            agg["close"] = "last"
        if "volume" in daily.columns:
            agg["volume"] = "sum"
        if "amount" in daily.columns:
            agg["amount"] = "sum"
        if not agg:
            return pd.DataFrame()

        resampled = daily.resample(rule).agg(agg)
        return cls._clean_dataframe(resampled, interval=iv)

    def _merge_parts(
        self,
        *dfs: pd.DataFrame,
        interval: str | None = None,
    ) -> pd.DataFrame:
        """Merge and deduplicate non-empty dataframes."""
        parts = [
            p for p in dfs
            if isinstance(p, pd.DataFrame) and not p.empty
        ]
        if not parts:
            return pd.DataFrame()
        if len(parts) == 1:
            return self._clean_dataframe(parts[0], interval=interval)
        return self._clean_dataframe(
            pd.concat(parts, axis=0),
            interval=interval,
        )

    @classmethod
    def _filter_cn_intraday_session(
        cls,
        df: pd.DataFrame,
        interval: str,
    ) -> pd.DataFrame:
        """Keep only regular CN A-share intraday session rows."""
        iv = cls._normalize_interval_token(interval)
        if iv in {"1d", "1wk", "1mo"}:
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()

        out = cls._clean_dataframe(df, interval=iv)
        if out.empty or not isinstance(out.index, pd.DatetimeIndex):
            return out

        idx  = out.index
        hhmm = (idx.hour * 100) + idx.minute
        in_morning   = (hhmm >= 930)  & (hhmm <= 1130)
        in_afternoon = (hhmm >= 1300) & (hhmm <= 1500)
        weekday      = idx.dayofweek < 5
        mask = weekday & (in_morning | in_afternoon)
        return out.loc[mask]

    def _cache_tail(
        self, cache_key: str, df: pd.DataFrame, count: int
    ) -> pd.DataFrame:
        out = df.tail(count)
        self._cache.set(cache_key, out)
        return out

    def _get_session_history(
        self, symbol: str, interval: str, bars: int
    ) -> pd.DataFrame:
        try:
            cache = get_session_bar_cache()
            return self._clean_dataframe(
                cache.read_history(
                    symbol=symbol, interval=interval, bars=bars
                ),
                interval=interval,
            )
        except Exception as exc:
            log.debug("Session cache lookup failed: %s", exc)
            return pd.DataFrame()

    def get_realtime(
        self, code: str, instrument: dict | None = None
    ) -> Quote | None:
        """Get real-time quote for a single instrument."""
        from core.instruments import parse_instrument
        inst = instrument or parse_instrument(code)

        if _is_offline():
            return None

        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            code6 = str(inst.get("symbol") or "").zfill(6)
            if code6:
                return self._get_realtime_cn(code6)

        return self._get_realtime_generic(inst)

    def _get_realtime_cn(self, code6: str) -> Quote | None:
        """Optimized CN equity real-time via batch + micro-cache."""
        now = time.time()
        with self._rt_cache_lock:
            rec = self._rt_single_microcache.get(code6)
            if rec and (now - float(rec["ts"])) < _MICRO_CACHE_TTL:
                return rec["q"]  # type: ignore[return-value]

        try:
            out = self.get_realtime_batch([code6])
            q = out.get(code6)
            if q and q.price > 0:
                with self._rt_cache_lock:
                    self._rt_single_microcache[code6] = {"ts": now, "q": q}
                return q
        except Exception as exc:
            log.debug("CN realtime batch fetch failed for %s: %s", code6, exc)

        with self._last_good_lock:
            q = self._last_good_quotes.get(code6)
            if q and q.price > 0:
                age = self._quote_age_seconds(q)
                if age <= _LAST_GOOD_MAX_AGE:
                    return self._mark_quote_as_delayed(q)
        return None

    def _get_realtime_generic(self, inst: dict) -> Quote | None:
        """Fetch real-time quote from all sources, pick best."""
        candidates: list[Quote] = []
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
        good = [
            c for c in candidates
            if abs(c.price - med) / max(med, 1e-8) < 0.02
        ]
        pool = good if good else candidates
        pool.sort(key=lambda q: (q.is_delayed, q.latency_ms))
        best = pool[0]

        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            code6 = str(inst.get("symbol") or "").zfill(6)
            with self._last_good_lock:
                self._last_good_quotes[code6] = best

        return best

    def get_multiple_parallel(
        self,
        codes: list[str],
        days: int = 500,
        callback: Callable[[str, int, int], None] | None = None,
        max_workers: int | None = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch history for multiple codes in parallel."""
        results: dict[str, pd.DataFrame] = {}
        total = len(codes)
        completed = 0
        lock = threading.Lock()

        for source in self._all_sources:
            with source._lock:
                source.status.consecutive_errors = 0
                source.status.disabled_until = None

        def fetch_one(code: str) -> tuple[str, pd.DataFrame]:
            try:
                df = self.get_history(code, days, interval=interval)
                return code, df
            except Exception as exc:
                log.debug("Failed to fetch %s: %s", code, exc)
                return code, pd.DataFrame()

        workers = min(max_workers or 2, 2) if interval in _INTRADAY_INTERVALS else min(max_workers or 5, 5)

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
                    log.warning("Failed to fetch %s: %s", code, exc)
                with lock:
                    completed += 1
                    if callback:
                        callback(code, completed, total)

        log.info("Parallel fetch: %d/%d successful", len(results), total)
        return results

    def get_all_stocks(self) -> pd.DataFrame:
        for source in self._get_active_sources():
            if source.name == "akshare":
                try:
                    df = source.get_all_stocks()
                    if not df.empty:
                        return df
                except Exception as exc:
                    log.warning("Failed to get stock list: %s", exc)
        return pd.DataFrame()

    def get_source_status(self) -> list[DataSourceStatus]:
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


_fetcher: DataFetcher | None = None
_fetcher_lock = threading.Lock()


def get_fetcher() -> DataFetcher:
    """Double-checked locking singleton for DataFetcher."""
    global _fetcher
    if _fetcher is None:
        with _fetcher_lock:
            if _fetcher is None:
                _fetcher = DataFetcher()
    return _fetcher
