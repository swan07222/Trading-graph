# data/fetcher_sources.py
import json
import math
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps

import numpy as np
import pandas as pd
import requests

from core.exceptions import DataFetchError, DataSourceUnavailableError
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


def _endpoint_candidates(
    env_key: str,
    defaults: tuple[str, ...] | list[str],
) -> list[str]:
    """Resolve endpoint templates from environment with safe fallback.

    Format: comma-separated URLs/templates. Empty/invalid values are ignored.
    """
    fallback = [str(v).strip() for v in defaults if str(v or "").strip()]
    raw = str(os.environ.get(str(env_key), "") or "").strip()
    if not raw:
        return fallback

    proto_hits = raw.count("http://") + raw.count("https://")
    if ";" in raw or "\n" in raw:
        pieces = raw.replace("\n", ";").split(";")
    elif proto_hits <= 1:
        # Single endpoint template may legitimately contain commas.
        pieces = [raw]
    else:
        pieces = raw.split(",")

    out: list[str] = []
    for piece in pieces:
        item = str(piece or "").strip()
        if not item:
            continue
        if not (item.startswith("http://") or item.startswith("https://")):
            continue
        out.append(item)
    return out or fallback


def _build_tencent_batch_url(template: str, symbols_csv: str) -> str:
    tpl = str(template or "").strip()
    if not tpl:
        return ""
    if "{symbols}" in tpl:
        try:
            return tpl.format(symbols=str(symbols_csv))
        except Exception:
            return ""
    if tpl.endswith("="):
        return f"{tpl}{symbols_csv}"
    if "q=" in tpl:
        return f"{tpl}{symbols_csv}"
    joiner = "&" if "?" in tpl else "?"
    return f"{tpl}{joiner}q={symbols_csv}"


def _build_tencent_daily_url(
    template: str,
    *,
    vendor_symbol: str,
    fetch_count: int,
) -> str:
    tpl = str(template or "").strip()
    if not tpl:
        return ""
    if ("{vendor_symbol}" in tpl) or ("{fetch_count}" in tpl):
        try:
            return tpl.format(
                vendor_symbol=str(vendor_symbol),
                fetch_count=int(fetch_count),
            )
        except Exception:
            return ""
    param = f"{vendor_symbol},day,,,{int(fetch_count)},qfq"
    if tpl.endswith("param="):
        return f"{tpl}{param}"
    if "param=" in tpl:
        return f"{tpl}{param}"
    joiner = "&" if "?" in tpl else "?"
    return f"{tpl}{joiner}param={param}"


def _run_with_timeout(
    task: Callable[[], object],
    timeout_s: float,
) -> object | None:
    """Run a callable with a timeout without mutating process-global socket defaults."""
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
    # 1.8x multiplier converts trading days -> calendar days
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

    # Circuit-breaker thresholds -> raised to avoid premature disabling
    _CB_ERROR_THRESHOLD: int = 12
    _CB_MIN_COOLDOWN: int = 20
    _CB_MAX_COOLDOWN: int = 90
    _CB_COOLDOWN_INCREMENT: int = 2
    _CB_RECENT_SUCCESS_DECAY_SEC: float = 30.0
    _CB_HALF_OPEN_PROBE_INTERVAL: float = 8.0
    _CB_DISABLE_WARN_MIN_GAP_SEC: float = 8.0

    def __init__(self) -> None:
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
                # Cooldown expired -> re-enable
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
        "price":      ("\u6700\u65b0\u4ef7", "\u73b0\u4ef7", "price"),
        "open":       ("\u4eca\u5f00", "\u5f00\u76d8\u4ef7", "open"),
        "high":       ("\u6700\u9ad8", "\u6700\u9ad8\u4ef7", "high"),
        "low":        ("\u6700\u4f4e", "\u6700\u4f4e\u4ef7", "low"),
        "close":      ("\u6628\u6536", "\u6628\u6536\u4ef7", "prev_close", "close"),
        "volume":     ("\u6210\u4ea4\u91cf", "volume"),
        "amount":     ("\u6210\u4ea4\u989d", "amount"),
        "change":     ("\u6da8\u8dcc\u989d", "change"),
        "change_pct": ("\u6da8\u8dcc\u5e45", "change_pct"),
        "name":       ("\u540d\u79f0", "\u80a1\u7968\u540d\u79f0", "name"),
    }
    def __init__(self, ttl_seconds: float = _SPOT_CACHE_TTL) -> None:
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
            for candidate in ("\u4ee3\u7801", "\u80a1\u7968\u4ee3\u7801", "code", "symbol"):
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
    """AkShare data source -> works ONLY on China direct IP."""

    name = "akshare"
    priority = 1
    needs_china_direct = True

    _AKSHARE_PERIOD_MAP = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}
    _AKSHARE_MIN_MAP = {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60"
    }
    _COLUMN_MAP = {
        "\u65e5\u671f": "date",
        "\u5f00\u76d8": "open",
        "\u6536\u76d8": "close",
        "\u6700\u9ad8": "high",
        "\u6700\u4f4e": "low",
        "\u6210\u4ea4\u91cf": "volume",
        "\u6210\u4ea4\u989d": "amount",
        "\u6da8\u8dcc\u5e45": "change_pct",
        "\u6362\u624b\u7387": "turnover",
    }
    _INTRADAY_COL_MAPS = [
        {
            "\u65f6\u95f4": "date", "\u5f00\u76d8": "open", "\u6536\u76d8": "close",
            "\u6700\u9ad8": "high", "\u6700\u4f4e": "low", "\u6210\u4ea4\u91cf": "volume",
            "\u6210\u4ea4\u989d": "amount",
        },
        {
            "\u65e5\u671f": "date", "\u5f00\u76d8": "open", "\u6536\u76d8": "close",
            "\u6700\u9ad8": "high", "\u6700\u4f4e": "low", "\u6210\u4ea4\u91cf": "volume",
            "\u6210\u4ea4\u989d": "amount",
        },
        {
            "time": "date", "open": "open", "close": "close",
            "high": "high", "low": "low", "volume": "volume",
            "amount": "amount",
        },
    ]
    def __init__(self) -> None:
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
                if "\u65e5" in col or "date" in col.lower() or "time" in col.lower():
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


from data.fetcher_tencent_source import (  # noqa: E402
    TencentQuoteSource as _TencentQuoteSource,
)

TencentQuoteSource = _TencentQuoteSource


class SinaHistorySource(DataSource):
    """Sina historical K-line source for CN equity (China direct IP)."""

    name = "sina"
    priority = 2
    needs_china_direct = True
    needs_vpn = False
    _CB_ERROR_THRESHOLD = 8
    _CB_MIN_COOLDOWN = 15
    _CB_MAX_COOLDOWN = 90

    _KLINE_SCALE_MAP = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "60m": "60",
        "1h": "60",
        # Sina uses 240-min bars as daily bars.
        "1d": "240",
    }
    _MAX_DATALEN = 1600
    _ENDPOINTS = (
        "https://quotes.sina.cn/cn/api/openapi.php/CN_MarketDataService.getKLineData",
        "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData",
    )
    _ENDPOINTS_ENV = "TRADING_SINA_KLINE_ENDPOINTS"

    def is_available(self) -> bool:
        if not self.is_suitable_for_network():
            return False
        return super().is_available()

    def is_suitable_for_network(self) -> bool:
        from core.network import get_network_env

        env = get_network_env()
        return bool(getattr(env, "is_china_direct", False))

    def get_history(self, code: str, days: int) -> pd.DataFrame:
        inst = {
            "market": "CN",
            "asset": "EQUITY",
            "symbol": str(code).zfill(6),
        }
        return self.get_history_instrument(inst, days=days, interval="1d")

    def get_history_instrument(
        self, inst: dict, days: int, interval: str = "1d"
    ) -> pd.DataFrame:
        if not self.is_available():
            return pd.DataFrame()
        if inst.get("market") != "CN" or str(inst.get("asset") or "").upper() != "EQUITY":
            return pd.DataFrame()

        iv_out = str(interval or "1d").lower()
        iv_req = "1m" if iv_out == "2m" else iv_out
        if iv_req in {"1wk", "1mo"}:
            iv_req = "1d"
        scale = self._KLINE_SCALE_MAP.get(iv_req)
        if not scale:
            return pd.DataFrame()

        symbol = self._to_sina_symbol(str(inst.get("symbol") or ""))
        if not symbol:
            return pd.DataFrame()

        target_rows = self._estimate_target_rows(days, iv_out)
        # Request extra rows for cleanup and weekend gaps.
        request_rows = int(min(self._MAX_DATALEN, max(120, target_rows + 80)))
        start_t = time.time()
        try:
            df = self._request_kline(symbol=symbol, scale=scale, datalen=request_rows)
            if df.empty:
                return pd.DataFrame()

            if iv_out == "2m":
                df = self._resample_to_2m(df)
            elif iv_out in {"1wk", "1mo"}:
                df = self._resample_daily(df, interval=iv_out)

            if df.empty:
                return pd.DataFrame()

            latency = (time.time() - start_t) * 1000.0
            self._record_success(latency)
            log.debug(
                "Sina history %s (%s): %d bars in %.0fms",
                symbol,
                iv_out,
                len(df),
                latency,
            )
            return df.tail(max(1, int(target_rows)))
        except Exception as exc:
            self._record_error(str(exc))
            log.debug("Sina history failed for %s (%s): %s", symbol, iv_out, exc)
            return pd.DataFrame()

    @staticmethod
    def _to_sina_symbol(code: str) -> str:
        code6 = "".join(ch for ch in str(code or "") if ch.isdigit()).zfill(6)
        if not code6 or len(code6) != 6:
            return ""
        try:
            from core.constants import get_exchange

            ex = str(get_exchange(code6) or "").upper()
        except Exception:
            ex = ""
        prefix = {"SSE": "sh", "SZSE": "sz", "BSE": "bj"}.get(ex)
        if prefix:
            return f"{prefix}{code6}"
        if code6.startswith(("5", "6", "9")):
            return f"sh{code6}"
        return f"sz{code6}"

    @staticmethod
    def _estimate_target_rows(days: int, interval: str) -> int:
        iv = str(interval or "1d").lower()
        # For 2m: Sina fetches 1m bars then resamples, so use 1m bpd
        lookup_iv = "1m" if iv == "2m" else iv
        bpd = float(BARS_PER_DAY.get(lookup_iv, 1.0) or 1.0)
        bpd = max(0.05, bpd)
        d = max(1, int(days or 1))
        rows = int(math.ceil(float(d) * bpd)) + 8
        return int(max(1, min(rows, 120_000)))

    def _request_kline(self, *, symbol: str, scale: str, datalen: int) -> pd.DataFrame:
        params = {
            "symbol": str(symbol),
            "scale": str(scale),
            "ma": "no",
            "datalen": str(max(1, min(int(datalen), self._MAX_DATALEN))),
        }

        errors: list[str] = []
        endpoints = _endpoint_candidates(self._ENDPOINTS_ENV, self._ENDPOINTS)
        for url in endpoints:
            try:
                resp = self._session.get(url, params=params, timeout=12)
                if int(resp.status_code) != 200:
                    errors.append(f"{url} HTTP {resp.status_code}")
                    continue
                frame = self._parse_kline_payload(resp.text)
                if frame is not None and not frame.empty:
                    return frame
                errors.append(f"{url} empty")
            except Exception as exc:
                errors.append(f"{url} {exc}")
                continue

        if errors:
            raise DataFetchError("; ".join(errors[:2]))
        return pd.DataFrame()

    def _parse_kline_payload(self, payload_text: str) -> pd.DataFrame:
        text = str(payload_text or "").strip()
        if not text:
            return pd.DataFrame()

        parsed = self._parse_json_like(text)
        if parsed is None:
            return pd.DataFrame()

        rows: list[dict] = []
        if isinstance(parsed, dict):
            result = parsed.get("result")
            if isinstance(result, dict):
                data = result.get("data")
                if isinstance(data, list):
                    rows = [r for r in data if isinstance(r, dict)]
            if not rows:
                data2 = parsed.get("data")
                if isinstance(data2, list):
                    rows = [r for r in data2 if isinstance(r, dict)]
        elif isinstance(parsed, list):
            rows = [r for r in parsed if isinstance(r, dict)]

        if not rows:
            return pd.DataFrame()
        return self._rows_to_frame(rows)

    @staticmethod
    def _parse_json_like(text: str) -> object | None:
        try:
            return json.loads(text)
        except Exception:
            pass

        # Handle JSONP wrappers or assignment wrappers.
        for left_char, right_char in (("{", "}"), ("[", "]")):
            left = text.find(left_char)
            right = text.rfind(right_char)
            if left < 0 or right <= left:
                continue
            frag = text[left : right + 1]
            try:
                return json.loads(frag)
            except Exception:
                continue
        return None

    @staticmethod
    def _rows_to_frame(rows: list[dict]) -> pd.DataFrame:
        out_rows: list[dict] = []
        for row in rows:
            try:
                ts_raw = (
                    row.get("day")
                    or row.get("date")
                    or row.get("time")
                    or row.get("d")
                    or ""
                )
                ts = pd.to_datetime(str(ts_raw), errors="coerce")
                if pd.isna(ts):
                    continue

                open_px = float(to_float(row.get("open") or row.get("o") or 0) or 0.0)
                high_px = float(to_float(row.get("high") or row.get("h") or 0) or 0.0)
                low_px = float(to_float(row.get("low") or row.get("l") or 0) or 0.0)
                close_px = float(to_float(row.get("close") or row.get("c") or 0) or 0.0)
                if close_px <= 0:
                    continue

                if open_px <= 0:
                    open_px = close_px
                if high_px <= 0:
                    high_px = max(open_px, close_px)
                if low_px <= 0:
                    low_px = min(open_px, close_px)
                high_px = max(high_px, open_px, close_px)
                low_px = min(low_px, open_px, close_px)
                if high_px < low_px:
                    continue

                vol = float(to_float(row.get("volume") or row.get("v") or 0) or 0.0)
                amount = float(
                    to_float(row.get("amount") or row.get("money") or row.get("tu") or 0)
                    or 0.0
                )
                if amount <= 0:
                    amount = close_px * max(0.0, vol)

                out_rows.append(
                    {
                        "date": ts,
                        "open": float(open_px),
                        "high": float(high_px),
                        "low": float(low_px),
                        "close": float(close_px),
                        "volume": max(0.0, vol),
                        "amount": max(0.0, amount),
                    }
                )
            except Exception:
                continue

        if not out_rows:
            return pd.DataFrame()
        df = pd.DataFrame(out_rows).set_index("date").sort_index()
        return df[~df.index.duplicated(keep="last")]

    @staticmethod
    def _resample_to_2m(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return pd.DataFrame()

        agg: dict[str, str] = {}
        if "open" in df.columns:
            agg["open"] = "first"
        if "high" in df.columns:
            agg["high"] = "max"
        if "low" in df.columns:
            agg["low"] = "min"
        if "close" in df.columns:
            agg["close"] = "last"
        if "volume" in df.columns:
            agg["volume"] = "sum"
        if "amount" in df.columns:
            agg["amount"] = "sum"
        if not agg:
            return pd.DataFrame()

        out = df.resample("2min").agg(agg)
        out = out.dropna(subset=["close"]) if "close" in out.columns else out
        return out[out["close"] > 0] if "close" in out.columns else out

    @staticmethod
    def _resample_daily(df: pd.DataFrame, interval: str) -> pd.DataFrame:
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return pd.DataFrame()

        rule = "W-FRI" if str(interval).lower() == "1wk" else "ME"
        agg: dict[str, str] = {}
        if "open" in df.columns:
            agg["open"] = "first"
        if "high" in df.columns:
            agg["high"] = "max"
        if "low" in df.columns:
            agg["low"] = "min"
        if "close" in df.columns:
            agg["close"] = "last"
        if "volume" in df.columns:
            agg["volume"] = "sum"
        if "amount" in df.columns:
            agg["amount"] = "sum"
        if not agg:
            return pd.DataFrame()

        out = df.resample(rule).agg(agg)
        out = out.dropna(subset=["close"]) if "close" in out.columns else out
        return out[out["close"] > 0] if "close" in out.columns else out


class YahooSource(DataSource):
    """Yahoo Finance -> works ONLY through VPN (foreign IP)."""

    name = "yahoo"
    priority = 1
    needs_vpn = True
    _CB_ERROR_THRESHOLD = 20
    _CB_MIN_COOLDOWN = 20
    _CB_MAX_COOLDOWN = 90

    _SUFFIX_MAP = {"6": ".SS", "0": ".SZ", "3": ".SZ", "4": ".BJ", "8": ".BJ"}
    _SUPPORTED_PREFIXES = ("0", "3", "4", "6", "8")

    def __init__(self) -> None:
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



