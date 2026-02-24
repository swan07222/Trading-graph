# data/fetcher.py
import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from config.runtime_env import env_flag as _read_env_flag
from config.runtime_env import env_text as _read_env_text
from config.settings import CONFIG
from core.symbols import clean_code as _clean_code
from core.symbols import validate_stock_code as _validate_stock_code
from data.cache import get_cache
from data.database import get_database
from data.fetcher_clean_ops import (
    _clean_dataframe as _clean_dataframe_impl,
)
from data.fetcher_clean_ops import (
    _normalize_datetime_index as _normalize_datetime_index_impl,
)
from data.fetcher_clean_ops import (
    _to_shanghai_naive_ts as _to_shanghai_naive_ts_impl,
)
from data.fetcher_frame_ops import (
    filter_cn_intraday_session as _filter_cn_intraday_session_impl,
)
from data.fetcher_frame_ops import (
    history_cache_store_rows as _history_cache_store_rows_impl,
)
from data.fetcher_frame_ops import (
    merge_parts as _merge_parts_impl,
)
from data.fetcher_frame_ops import (
    resample_daily_to_interval as _resample_daily_to_interval_impl,
)
from data.fetcher_history_flow_ops import (
    _accept_online_intraday_snapshot as _accept_online_intraday_snapshot_impl,
)
from data.fetcher_history_flow_ops import (
    _fetch_history_with_depth_retry as _fetch_history_with_depth_retry_impl,
)
from data.fetcher_history_flow_ops import (
    _get_history_cn_daily as _get_history_cn_daily_impl,
)
from data.fetcher_history_flow_ops import (
    _get_history_cn_intraday as _get_history_cn_intraday_impl,
)
from data.fetcher_history_flow_ops import (
    _get_history_cn_intraday_exact as _get_history_cn_intraday_exact_impl,
)
from data.fetcher_history_flow_ops import (
    _is_post_close_or_preopen_window as _is_post_close_or_preopen_window_impl,
)
from data.fetcher_history_flow_ops import (
    _resolve_requested_bar_count as _resolve_requested_bar_count_impl,
)
from data.fetcher_history_flow_ops import (
    _should_refresh_intraday_exact as _should_refresh_intraday_exact_impl,
)
from data.fetcher_history_flow_ops import (
    get_history as _get_history_impl,
)
from data.fetcher_history_ops import (
    _fetch_from_sources_instrument as _fetch_from_sources_instrument_impl,
)
from data.fetcher_instance_ops import (
    get_fetcher_instance as _get_fetcher_instance_impl,
)
from data.fetcher_instance_ops import (
    reset_fetcher_instances as _reset_fetcher_instances_impl,
)
from data.fetcher_quality_ops import (
    _cross_validate_bars as _cross_validate_bars_impl,
)
from data.fetcher_quality_ops import (
    _daily_consensus_quorum_meta as _daily_consensus_quorum_meta_impl,
)
from data.fetcher_quality_ops import (
    _daily_frame_quality as _daily_frame_quality_impl,
)
from data.fetcher_quality_ops import (
    _drop_stale_flat_bars as _drop_stale_flat_bars_impl,
)
from data.fetcher_quality_ops import (
    _history_quorum_allows_persist as _history_quorum_allows_persist_impl,
)
from data.fetcher_quality_ops import (
    _intraday_frame_quality as _intraday_frame_quality_impl,
)
from data.fetcher_quality_ops import (
    _intraday_quality_caps as _intraday_quality_caps_impl,
)
from data.fetcher_quality_ops import (
    _max_close_cluster_size as _max_close_cluster_size_impl,
)
from data.fetcher_quality_ops import (
    _merge_daily_by_consensus as _merge_daily_by_consensus_impl,
)
from data.fetcher_quote_ops import (
    _drop_stale_quotes as _drop_stale_quotes_wrapper_impl,
)
from data.fetcher_quote_ops import (
    _fallback_last_close_from_db as _fallback_last_close_from_db_impl,
)
from data.fetcher_quote_ops import (
    _fallback_last_good as _fallback_last_good_impl,
)
from data.fetcher_quote_ops import (
    _fill_from_batch_sources as _fill_from_batch_sources_impl,
)
from data.fetcher_quote_ops import (
    _fill_from_single_source_quotes as _fill_from_single_source_quotes_impl,
)
from data.fetcher_quote_ops import (
    _fill_from_spot_cache as _fill_from_spot_cache_wrapper_impl,
)
from data.fetcher_quote_ops import (
    _is_tencent_source as _is_tencent_source_impl,
)
from data.fetcher_quote_ops import (
    _mark_quote_as_delayed as _mark_quote_as_delayed_impl,
)
from data.fetcher_quote_ops import (
    _quote_age_seconds as _quote_age_seconds_impl,
)
from data.fetcher_quote_ops import (
    get_realtime_batch as _get_realtime_batch_impl,
)
from data.fetcher_reconcile_ops import (
    _clear_refresh_reconcile_pending as _clear_refresh_reconcile_pending_impl,
)
from data.fetcher_reconcile_ops import (
    _get_pending_reconcile_codes as _get_pending_reconcile_codes_impl,
)
from data.fetcher_reconcile_ops import (
    _get_pending_reconcile_entries as _get_pending_reconcile_entries_impl,
)
from data.fetcher_reconcile_ops import (
    _get_refresh_reconcile_lock as _get_refresh_reconcile_lock_impl,
)
from data.fetcher_reconcile_ops import (
    _get_refresh_reconcile_path as _get_refresh_reconcile_path_impl,
)
from data.fetcher_reconcile_ops import (
    _load_refresh_reconcile_queue as _load_refresh_reconcile_queue_impl,
)
from data.fetcher_reconcile_ops import (
    _mark_refresh_reconcile_pending as _mark_refresh_reconcile_pending_impl,
)
from data.fetcher_reconcile_ops import (
    _now_shanghai_naive as _now_shanghai_naive_impl,
)
from data.fetcher_reconcile_ops import (
    _reconcile_pending_cache_sync as _reconcile_pending_cache_sync_impl,
)
from data.fetcher_reconcile_ops import (
    _refresh_reconcile_key as _refresh_reconcile_key_impl,
)
from data.fetcher_reconcile_ops import (
    _save_refresh_reconcile_queue as _save_refresh_reconcile_queue_impl,
)
from data.fetcher_refresh_ops import (
    refresh_trained_stock_history as _refresh_trained_stock_history_impl,
)
from data.fetcher_registry import (
    FetcherRegistry,
    use_fetcher_registry,
)
from data.fetcher_source_ops import (
    create_local_database_source as _create_local_database_source,
)
from data.fetcher_source_ops import (
    normalize_source_name as _normalize_source_name_impl,
)
from data.fetcher_source_ops import (
    resolve_source_order as _resolve_source_order_impl,
)
from data.fetcher_source_ops import (
    source_health_score as _source_health_score_impl,
)
from data.fetcher_sources import (
    _INTRADAY_INTERVALS,
    _LAST_GOOD_MAX_AGE,
    _MICRO_CACHE_TTL,
    AkShareSource,
    BARS_PER_DAY,
    DataSource,
    DataSourceStatus,
    INTERVAL_MAX_DAYS,
    Quote,
    SinaHistorySource,
    TencentQuoteSource,
    YahooSource,
    _is_offline,
    bars_to_days,
    get_spot_cache,
)
from data.session_cache import get_session_bar_cache
from utils.logger import get_logger

log = get_logger(__name__)
__all__ = [
    "AkShareSource",
    "BARS_PER_DAY",
    "DataFetcher",
    "DataSource",
    "DataSourceStatus",
    "FetcherRegistry",
    "INTERVAL_MAX_DAYS",
    "Quote",
    "SinaHistorySource",
    "TencentQuoteSource",
    "YahooSource",
    "bars_to_days",
    "create_fetcher",
    "get_fetcher",
    "get_spot_cache",
    "reset_fetcher",
    "use_fetcher_registry",
]
_RECOVERABLE_FETCH_EXCEPTIONS = (
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)
_SOURCE_CLASSES: dict[str, type[DataSource]] = {
    "tencent": TencentQuoteSource,
    "akshare": AkShareSource,
    "sina": SinaHistorySource,
    "yahoo": YahooSource,
}
def _env_flag(name: str, default: str = "0") -> bool:
    return _read_env_flag(name, default)

class DataFetcher:
    """Network-aware fetcher with local cache/DB and fallback sources."""

    def __init__(self) -> None:
        self._all_sources: list[DataSource] = []
        self._cache = get_cache()
        self._db = get_database()
        self._is_live_mode = self._detect_live_mode()
        self._strict_errors = _env_flag("TRADING_STRICT_ERRORS", "0")
        self._strict_realtime_quotes = _env_flag(
            "TRADING_STRICT_REALTIME_QUOTES",
            "1" if self._is_live_mode else "0",
        )
        self._allow_last_close_fallback = _env_flag(
            "TRADING_ALLOW_LAST_CLOSE_FALLBACK",
            "0",
        )
        self._allow_stale_realtime_fallback = _env_flag(
            "TRADING_ALLOW_STALE_REALTIME_FALLBACK",
            "0",
        )
        self._realtime_quote_max_age_s = self._resolve_realtime_quote_max_age()
        self._last_good_max_age_s = self._resolve_last_good_max_age()
        self._source_order = self._resolve_source_order()
        self._rate_limiter = threading.Semaphore(CONFIG.data.parallel_downloads)
        self._request_times: dict[str, float] = {}
        self._min_interval: float = 0.5
        self._intraday_interval: float = 1.2
        self._last_good_quotes: dict[str, Quote] = {}
        self._last_good_lock = threading.RLock()
        # Micro-caches
        self._rt_cache_lock = threading.RLock()
        self._rt_batch_microcache: dict[str, Any] = {
            "ts": 0.0, "key": None, "data": {},
        }
        self._rt_single_microcache: dict[str, dict[str, Any]] = {}

        self._rate_lock = threading.Lock()
        self._last_source_fail_warn_ts: dict[str, float] = {}
        self._source_fail_warn_cooldown_s: float = 45.0
        self._last_source_fail_warn_global_ts: float = 0.0
        self._source_fail_warn_global_cooldown_s: float = 8.0
        self._last_network_mode: tuple[bool, bool, bool] | None = None
        self._last_network_force_refresh_ts: float = 0.0
        self._network_force_refresh_cooldown_s: float = 20.0
        self._refresh_reconcile_lock = threading.RLock()
        self._refresh_reconcile_path = Path(CONFIG.data_dir) / "refresh_reconcile_queue.json"
        self._init_sources()

    @staticmethod
    def _detect_live_mode() -> bool:
        mode = getattr(CONFIG, "trading_mode", "")
        raw = str(getattr(mode, "value", mode) or "").strip().lower()
        return raw == "live"

    @staticmethod
    def _resolve_last_good_max_age() -> float:
        default_age = float(_LAST_GOOD_MAX_AGE)
        risk_cfg = getattr(CONFIG, "risk", None)
        raw = getattr(risk_cfg, "quote_staleness_seconds", default_age)
        try:
            return max(1.0, min(default_age, float(raw)))
        except (TypeError, ValueError):
            return default_age

    @staticmethod
    def _resolve_realtime_quote_max_age() -> float:
        risk_cfg = getattr(CONFIG, "risk", None)
        raw = getattr(risk_cfg, "quote_staleness_seconds", 8.0)
        try:
            return max(1.0, min(120.0, float(raw)))
        except (TypeError, ValueError):
            return 8.0

    @staticmethod
    def _normalize_source_name(name: str) -> str:
        return _normalize_source_name_impl(name)

    def _resolve_source_order(self) -> list[str]:
        return _resolve_source_order_impl(
            raw_value=_read_env_text("TRADING_ENABLED_SOURCES", ""),
            source_classes=_SOURCE_CLASSES,
            default_order=("tencent", "akshare", "sina", "yahoo"),
            logger=log,
        )

    def _init_sources(self) -> None:
        self._all_sources = []
        self._init_local_db_source()

        # Runtime policy:
        # - market profile selects default provider order
        # - TRADING_ENABLED_SOURCES can override provider set/order
        for source_name in self._source_order:
            source_cls = _SOURCE_CLASSES.get(source_name)
            if source_cls is None:
                log.warning("Unknown data source skipped: %s", source_name)
                continue
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
            except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
                if self._strict_errors:
                    raise
                log.warning("Failed to init %s: %s", source_cls.__name__, exc)

        if not self._all_sources:
            log.error("No data sources available!")

    def _init_local_db_source(self) -> None:
        """Create and register the local database source."""
        try:
            self._all_sources.append(_create_local_database_source(self._db))
            log.info("Data source localdb initialized")

        except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
            if self._strict_errors:
                raise
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
                "Network mode changed -> cooldowns reset "
                "(%s)",
                "CHINA_DIRECT" if env.is_china_direct else "VPN_FOREIGN",
            )

        active: list[DataSource] = []
        for s in self._all_sources:
            try:
                if not s.is_available():
                    continue
                if not s.is_suitable_for_network():
                    continue
            except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
                log.debug(
                    "Source suitability check failed for %s: %s",
                    getattr(s, "name", "?"),
                    exc,
                )
                continue
            active.append(s)
        ranked = sorted(
            active,
            key=lambda s: (-self._source_health_score(s, env), s.priority),
        )
        return ranked

    def _source_health_score(self, source: DataSource, env: object) -> float:
        return _source_health_score_impl(source, env)

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
        except _RECOVERABLE_FETCH_EXCEPTIONS:
            return False
    @staticmethod
    def _is_tencent_source(source: object) -> bool:
        return _is_tencent_source_impl(source)

    def _fill_from_batch_sources(
        self,
        cleaned: list[str],
        result: dict[str, Quote],
        sources: list[DataSource],
    ) -> None:
        _fill_from_batch_sources_impl(self, cleaned, result, sources)

    def get_realtime_batch(self, codes: list[str]) -> dict[str, Quote]:
        return _get_realtime_batch_impl(self, codes)

    def _fill_from_spot_cache(
        self,
        missing: list[str],
        result: dict[str, Quote],
    ) -> None:
        _fill_from_spot_cache_wrapper_impl(
            self,
            missing,
            result,
            get_spot_cache_fn=get_spot_cache,
        )

    def _fill_from_single_source_quotes(
        self,
        missing: list[str],
        result: dict[str, Quote],
        sources: list[DataSource],
    ) -> None:
        _fill_from_single_source_quotes_impl(self, missing, result, sources)

    def _fallback_last_good(self, codes: list[str]) -> dict[str, Quote]:
        return _fallback_last_good_impl(self, codes)

    @staticmethod
    def _mark_quote_as_delayed(q: Quote) -> Quote:
        return _mark_quote_as_delayed_impl(q)

    @staticmethod
    def _quote_age_seconds(q: Quote | None) -> float:
        return _quote_age_seconds_impl(q)

    def _drop_stale_quotes(
        self,
        quotes: dict[str, Quote],
        *,
        context: str,
    ) -> dict[str, Quote]:
        return _drop_stale_quotes_wrapper_impl(self, quotes, context=context)

    def _fallback_last_close_from_db(
        self,
        codes: list[str],
    ) -> dict[str, Quote]:
        return _fallback_last_close_from_db_impl(self, codes)

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
        except _RECOVERABLE_FETCH_EXCEPTIONS:
            return False

    def _fetch_from_sources_instrument(
        self,
        inst: dict[str, Any],
        days: int,
        interval: str = "1d",
        include_localdb: bool = True,
        return_meta: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        return _fetch_from_sources_instrument_impl(
            self,
            inst,
            days,
            interval=interval,
            include_localdb=include_localdb,
            return_meta=return_meta,
        )

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
        source: DataSource, inst: dict[str, Any], days: int, interval: str
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
        """Normalize a stock code to bare 6-digit form.

        Delegates to core.symbols.clean_code for canonical implementation.
        """
        return _clean_code(code)

    @staticmethod
    def validate_stock_code(code: str) -> tuple[bool, str]:
        """Validate a stock code format.

        Delegates to core.symbols.validate_stock_code for canonical implementation.

        Returns:
            (is_valid, error_message) tuple.
            If valid, error_message is empty string.
        """
        return _validate_stock_code(code)

    @staticmethod
    def _normalize_interval_token(interval: str | None) -> str:
        iv = str(interval or "1m").strip().lower()
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
        iv = str(interval or "1m").strip().lower()
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
        except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
            log.debug("Invalid interval token (%s): %s", iv, exc)
        return 60

    @staticmethod
    def _now_shanghai_naive() -> datetime:
        return _now_shanghai_naive_impl()

    def _get_refresh_reconcile_lock(self) -> threading.RLock:
        return _get_refresh_reconcile_lock_impl(self)

    def _get_refresh_reconcile_path(self) -> Path:
        return _get_refresh_reconcile_path_impl(self)

    def _refresh_reconcile_key(self, code: str, interval: str) -> str:
        return _refresh_reconcile_key_impl(self, code, interval)

    def _load_refresh_reconcile_queue(self) -> dict[str, dict[str, object]]:
        return _load_refresh_reconcile_queue_impl(self)

    def _save_refresh_reconcile_queue(self, queue: dict[str, dict[str, object]]) -> None:
        _save_refresh_reconcile_queue_impl(self, queue)

    def _mark_refresh_reconcile_pending(
        self,
        queue: dict[str, dict[str, object]],
        code: str,
        interval: str,
        *,
        error_text: str,
    ) -> bool:
        return _mark_refresh_reconcile_pending_impl(
            self,
            queue,
            code,
            interval,
            error_text=error_text,
        )

    def _clear_refresh_reconcile_pending(
        self,
        queue: dict[str, dict[str, object]],
        code: str,
        interval: str,
    ) -> bool:
        return _clear_refresh_reconcile_pending_impl(
            self,
            queue,
            code,
            interval,
        )

    def get_pending_reconcile_entries(
        self,
        interval: str | None = None,
    ) -> dict[str, dict[str, object]]:
        return _get_pending_reconcile_entries_impl(self, interval=interval)

    def get_pending_reconcile_codes(
        self,
        interval: str | None = None,
    ) -> list[str]:
        return _get_pending_reconcile_codes_impl(self, interval=interval)

    def reconcile_pending_cache_sync(
        self,
        *,
        codes: list[str] | None = None,
        interval: str = "1m",
        db_limit: int | None = None,
    ) -> dict[str, object]:
        return _reconcile_pending_cache_sync_impl(
            self,
            codes=codes,
            interval=interval,
            db_limit=db_limit,
            get_session_bar_cache_fn=get_session_bar_cache,
        )

    @classmethod
    def _intraday_quality_caps(
        cls,
        interval: str | None,
    ) -> tuple[float, float, float, float]:
        return _intraday_quality_caps_impl(cls, interval)

    @classmethod
    def _intraday_frame_quality(
        cls,
        df: pd.DataFrame,
        interval: str,
    ) -> dict[str, float | bool]:
        return _intraday_frame_quality_impl(cls, df, interval)

    @classmethod
    def _daily_frame_quality(cls, df: pd.DataFrame) -> dict[str, float | bool]:
        return _daily_frame_quality_impl(cls, df)

    @staticmethod
    def _max_close_cluster_size(
        closes: list[float],
        tolerance_ratio: float,
    ) -> int:
        return _max_close_cluster_size_impl(closes, tolerance_ratio)

    @classmethod
    def _daily_consensus_quorum_meta(
        cls,
        collected: list[dict[str, Any]],
    ) -> dict[str, object]:
        return _daily_consensus_quorum_meta_impl(cls, collected)

    def _history_quorum_allows_persist(
        self,
        *,
        interval: str,
        symbol: str,
        meta: dict[str, object] | None,
    ) -> bool:
        return _history_quorum_allows_persist_impl(
            self,
            interval=interval,
            symbol=symbol,
            meta=meta,
        )

    @classmethod
    def _merge_daily_by_consensus(
        cls,
        collected: list[dict[str, Any]],
        *,
        interval: str = "1d",
    ) -> pd.DataFrame:
        return _merge_daily_by_consensus_impl(
            cls,
            collected,
            interval=interval,
        )

    @staticmethod
    def _drop_stale_flat_bars(df: pd.DataFrame) -> pd.DataFrame:
        return _drop_stale_flat_bars_impl(df)

    @classmethod
    def _cross_validate_bars(
        cls,
        best_df: pd.DataFrame,
        alternatives: list[pd.DataFrame],
        interval: str,
    ) -> pd.DataFrame:
        return _cross_validate_bars_impl(
            cls,
            best_df,
            alternatives,
            interval,
        )

    @classmethod
    def _to_shanghai_naive_ts(cls, value: object) -> pd.Timestamp:
        return _to_shanghai_naive_ts_impl(cls, value)

    @classmethod
    def _normalize_datetime_index(
        cls,
        idx: object,
    ) -> pd.DatetimeIndex | None:
        return _normalize_datetime_index_impl(cls, idx)

    @classmethod
    def _clean_dataframe(
        cls,
        df: pd.DataFrame,
        interval: str | None = None,
        *,
        preserve_truth: bool | None = None,
        aggressive_repairs: bool | None = None,
        allow_synthetic_index: bool | None = None,
    ) -> pd.DataFrame:
        return _clean_dataframe_impl(
            cls,
            df,
            interval=interval,
            preserve_truth=preserve_truth,
            aggressive_repairs=aggressive_repairs,
            allow_synthetic_index=allow_synthetic_index,
        )

    # ------------------------------------------------------------------
    # History orchestration (unchanged logic, minor robustness improvements)
    # ------------------------------------------------------------------

    def get_history(
        self,
        code: str,
        days: int = 500,
        bars: int | None = None,
        use_cache: bool = True,
        update_db: bool = True,
        instrument: dict[str, Any] | None = None,
        interval: str = "1d",
        max_age_hours: float | None = None,
        allow_online: bool = True,
        refresh_intraday_after_close: bool = False,
    ) -> pd.DataFrame:
        return _get_history_impl(
            self,
            code,
            days=days,
            bars=bars,
            use_cache=use_cache,
            update_db=update_db,
            instrument=instrument,
            interval=interval,
            max_age_hours=max_age_hours,
            allow_online=allow_online,
            refresh_intraday_after_close=refresh_intraday_after_close,
        )

    def _should_refresh_intraday_exact(
        self,
        *,
        interval: str,
        update_db: bool,
        allow_online: bool,
    ) -> bool:
        return _should_refresh_intraday_exact_impl(
            self,
            interval=interval,
            update_db=update_db,
            allow_online=allow_online,
        )

    @staticmethod
    def _is_post_close_or_preopen_window() -> bool:
        return _is_post_close_or_preopen_window_impl()

    @staticmethod
    def _resolve_requested_bar_count(
        days: int,
        bars: int | None,
        interval: str,
    ) -> int:
        return _resolve_requested_bar_count_impl(days, bars, interval)

    def _fetch_history_with_depth_retry(
        self,
        inst: dict[str, Any],
        interval: str,
        requested_count: int,
        base_fetch_days: int,
        return_meta: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        return _fetch_history_with_depth_retry_impl(
            self,
            inst,
            interval,
            requested_count,
            base_fetch_days,
            return_meta=return_meta,
        )

    def _accept_online_intraday_snapshot(
        self,
        *,
        symbol: str,
        interval: str,
        online_df: pd.DataFrame,
        baseline_df: pd.DataFrame | None = None,
    ) -> bool:
        return _accept_online_intraday_snapshot_impl(
            self,
            symbol=symbol,
            interval=interval,
            online_df=online_df,
            baseline_df=baseline_df,
        )

    def _get_history_cn_intraday(
        self,
        inst: dict[str, Any],
        count: int,
        fetch_days: int,
        interval: str,
        cache_key: str,
        offline: bool,
        session_df: pd.DataFrame | None = None,
        *,
        persist_intraday_db: bool = True,
    ) -> pd.DataFrame:
        return _get_history_cn_intraday_impl(
            self,
            inst,
            count,
            fetch_days,
            interval,
            cache_key,
            offline,
            session_df,
            persist_intraday_db=persist_intraday_db,
        )

    def _get_history_cn_intraday_exact(
        self,
        inst: dict[str, Any],
        count: int,
        fetch_days: int,
        interval: str,
        cache_key: str,
        offline: bool,
    ) -> pd.DataFrame:
        return _get_history_cn_intraday_exact_impl(
            self,
            inst,
            count,
            fetch_days,
            interval,
            cache_key,
            offline,
        )

    def _get_history_cn_daily(
        self,
        inst: dict[str, Any],
        count: int,
        fetch_days: int,
        cache_key: str,
        offline: bool,
        update_db: bool,
        session_df: pd.DataFrame | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        return _get_history_cn_daily_impl(
            self,
            inst,
            count,
            fetch_days,
            cache_key,
            offline,
            update_db,
            session_df,
            interval=interval,
        )

    def refresh_trained_stock_history(
        self,
        codes: list[str],
        *,
        interval: str = "1m",
        window_days: int = 2,
        allow_online: bool = True,
        sync_session_cache: bool = True,
        replace_realtime_after_close: bool = True,
    ) -> dict[str, object]:
        return _refresh_trained_stock_history_impl(
            self,
            codes,
            interval=interval,
            window_days=window_days,
            allow_online=allow_online,
            sync_session_cache=sync_session_cache,
            replace_realtime_after_close=replace_realtime_after_close,
            get_session_bar_cache_fn=get_session_bar_cache,
        )

    @classmethod
    def _resample_daily_to_interval(
        cls,
        df: pd.DataFrame,
        interval: str,
    ) -> pd.DataFrame:
        return _resample_daily_to_interval_impl(
            df=df,
            interval=interval,
            normalize_interval_token=cls._normalize_interval_token,
            clean_dataframe=cls._clean_dataframe,
        )

    def _merge_parts(
        self,
        *dfs: pd.DataFrame,
        interval: str | None = None,
    ) -> pd.DataFrame:
        return _merge_parts_impl(
            *dfs,
            interval=interval,
            clean_dataframe=self._clean_dataframe,
        )

    @classmethod
    def _filter_cn_intraday_session(
        cls,
        df: pd.DataFrame,
        interval: str,
    ) -> pd.DataFrame:
        return _filter_cn_intraday_session_impl(
            df=df,
            interval=interval,
            normalize_interval_token=cls._normalize_interval_token,
            clean_dataframe=cls._clean_dataframe,
        )

    @classmethod
    def _history_cache_store_rows(
        cls,
        interval: str | None,
        requested_rows: int,
    ) -> int:
        return _history_cache_store_rows_impl(
            interval=interval,
            requested_rows=requested_rows,
            normalize_interval_token=cls._normalize_interval_token,
        )

    def _cache_tail(
        self,
        cache_key: str,
        df: pd.DataFrame,
        count: int,
        *,
        interval: str | None = None,
    ) -> pd.DataFrame:
        out = df.tail(count).copy()
        keep_rows = min(
            len(df),
            self._history_cache_store_rows(interval, count),
        )
        cache_df = df.tail(max(1, int(keep_rows))).copy()
        self._cache.set(cache_key, cache_df)
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
        except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
            log.debug("Session cache lookup failed: %s", exc)
            return pd.DataFrame()

    def get_realtime(
        self, code: str, instrument: dict[str, Any] | None = None
    ) -> Quote | None:
        """Get real-time quote for a single instrument."""
        from core.instruments import parse_instrument

        # FIX: Validate stock code format
        is_valid, error_msg = self.validate_stock_code(code)
        if not is_valid:
            log.debug("Invalid stock code in get_realtime: %s", error_msg)
            return None

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
                cached_quote = rec.get("q")
                if isinstance(cached_quote, Quote):
                    return cached_quote

        try:
            out = self.get_realtime_batch([code6])
            q = out.get(code6)
            if q and q.price > 0:
                with self._rt_cache_lock:
                    self._rt_single_microcache[code6] = {"ts": now, "q": q}
                return q
        except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
            log.debug("CN realtime batch fetch failed for %s: %s", code6, exc)
            if bool(getattr(self, "_strict_errors", False)):
                raise

        if bool(getattr(self, "_strict_realtime_quotes", False)):
            return None
        with self._last_good_lock:
            q = self._last_good_quotes.get(code6)
            if q and q.price > 0:
                age = self._quote_age_seconds(q)
                if age <= float(getattr(self, "_last_good_max_age_s", _LAST_GOOD_MAX_AGE)):
                    return self._mark_quote_as_delayed(q)
        return None

    def _get_realtime_generic(self, inst: dict[str, Any]) -> Quote | None:
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
                except _RECOVERABLE_FETCH_EXCEPTIONS:
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
        filtered = self._drop_stale_quotes(
            {"best": best},
            context="_get_realtime_generic",
        )
        best = filtered.get("best")
        if best is None:
            return None

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
            except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
                log.debug("Failed to fetch %s: %s", code, exc)
                return code, pd.DataFrame()

        default_workers = 2 if interval in _INTRADAY_INTERVALS else 5
        cap_workers = 2 if interval in _INTRADAY_INTERVALS else 5
        requested_workers = default_workers if max_workers is None else int(max_workers)
        workers = max(1, min(requested_workers, cap_workers))
        requested_bars = self._resolve_requested_bar_count(
            days=days, bars=None, interval=interval
        )
        min_required_rows = max(1, min(int(CONFIG.data.min_history_days), int(requested_bars)))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch_one, c): c for c in codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    code, df = future.result(timeout=120)
                    if (
                        not df.empty
                        and len(df) >= min_required_rows
                    ):
                        results[code] = df
                except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
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
                except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
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


def create_fetcher() -> DataFetcher:
    """Create a new fetcher instance without touching singleton registry."""
    return DataFetcher()


def get_fetcher(
    *,
    instance: str | None = None,
    force_new: bool = False,
    registry: FetcherRegistry[object] | None = None,
) -> DataFetcher:
    """Get/create fetcher instance by key.

    Default scope is thread-local (`TRADING_FETCHER_SCOPE=thread`) to avoid
    cross-thread mutable-cache coupling. Set `TRADING_FETCHER_SCOPE=process`
    to restore legacy process-global singleton behavior.
    """
    inst = _get_fetcher_instance_impl(
        create=create_fetcher,
        disable_singletons=_env_flag("TRADING_DISABLE_SINGLETONS", "0"),
        instance=instance,
        force_new=force_new,
        registry=registry,
    )
    return cast(DataFetcher, inst)


def reset_fetcher(
    *,
    instance: str | None = None,
    registry: FetcherRegistry[object] | None = None,
) -> None:
    """Reset fetcher singleton(s); defaults to clearing all instances."""
    _reset_fetcher_instances_impl(instance=instance, registry=registry)
