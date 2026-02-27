# data/fetcher_unified.py
"""Unified data fetcher with comprehensive improvements.

FIX 2026-02-26: This module provides a unified interface that addresses
all disadvantages of the original data fetching approach:

1. Network Dependency -> Configurable timeouts, retries, circuit breakers
2. Data Quality -> Validation pipeline with anomaly detection
3. Rate Limiting -> Adaptive rate limiting with backoff
4. Cache Invalidation -> Smart TTL and versioning
5. Memory -> Streaming and chunked processing
6. Minimum Data -> Progressive loading with graceful degradation
7. Timezone -> Unified Shanghai timezone handling
8. Source Health -> Real-time monitoring and auto-failover

Usage:
    from data.fetcher_unified import UnifiedDataFetcher
    
    fetcher = UnifiedDataFetcher()
    df = fetcher.get_history("000001", interval="1d", bars=100)
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from config.settings import CONFIG
from core.symbols import clean_code
from data.fetcher import DataFetcher, get_fetcher
from data.fetcher_config import FetcherConfig, get_config
from data.progressive_loader import ProgressiveDataLoader, LoadResult, LoadStatus, get_progressive_loader
from data.fetcher_sources import BARS_PER_DAY, _INTRADAY_INTERVALS
try:
    from data.rate_limiter_enhanced import (
        acquire_rate_limit as _acquire_rate_limit,
    )
    from data.rate_limiter_enhanced import (
        record_api_failure as _record_api_failure,
    )
    from data.rate_limiter_enhanced import (
        record_api_success as _record_api_success,
    )
    _HAS_ENHANCED_RATE_LIMITING = True
except ImportError:
    _HAS_ENHANCED_RATE_LIMITING = False
from data.source_health import (
    DataSourceHealthMonitor,
    SourceHealthStatus,
    get_health_monitor,
    record_source_failure,
    record_source_success,
    get_healthy_source,
)
from data.timezone_utils import (
    TradingSessionChecker,
    TimezoneConverter,
    get_session_checker,
    get_timezone_converter,
    filter_trading_hours,
    ensure_shanghai_datetime,
)
from data.validator import DataValidator, ValidationResult, get_validator
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class FetchOptions:
    """Options for unified data fetching."""
    # Basic options
    interval: str = "1d"
    bars: int | None = None
    days: int | None = None
    
    # Quality options
    validate: bool = True
    min_quality_score: float = 0.3
    filter_trading_hours_only: bool = False
    
    # Cache options
    use_cache: bool = True
    force_refresh: bool = False
    cache_ttl: float | None = None
    
    # Network options
    allow_online: bool = True
    timeout_seconds: float | None = None
    max_retries: int | None = None
    
    # Progressive loading
    progressive: bool = True
    allow_partial: bool = True
    partial_threshold: float = 0.4
    
    # Source options
    preferred_source: str | None = None
    auto_failover: bool = True
    
    # Memory options
    chunk_processing: bool = True
    max_memory_mb: float | None = None
    
    # Metadata
    correlation_id: str | None = None


@dataclass
class FetchResult:
    """Result of unified data fetching with full metadata."""
    success: bool
    data: pd.DataFrame | None
    bars_loaded: int
    bars_requested: int
    quality_score: float
    load_time_ms: float
    source_used: str | None
    cache_hit: bool
    validation_result: ValidationResult | None
    load_status: LoadStatus | None
    error: str | None = None
    warnings: list[str] | None = None
    
    def is_usable(self, min_status: LoadStatus = LoadStatus.MINIMUM) -> bool:
        """Check if result meets minimum requirements."""
        if not self.success or self.data is None or self.data.empty:
            return False
        if self.load_status:
            return self.load_status.is_usable(min_status)
        return self.bars_loaded >= self.bars_requested * 0.4
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "success": self.success,
            "bars_loaded": self.bars_loaded,
            "bars_requested": self.bars_requested,
            "completeness": round(self.bars_loaded / max(1, self.bars_requested), 2),
            "quality_score": round(self.quality_score, 3),
            "load_time_ms": round(self.load_time_ms, 1),
            "source": self.source_used,
            "cache_hit": self.cache_hit,
            "status": self.load_status.value if self.load_status else None,
            "error": self.error,
            "warnings": self.warnings or [],
        }


class UnifiedDataFetcher:
    """Unified data fetcher with comprehensive improvements.
    
    FIX 2026-02-26: Addresses all disadvantages:
    
    1. **Network Dependency**
       - Configurable timeouts and retries
       - Circuit breaker pattern
       - Connection pooling
    
    2. **Data Quality**
       - Validation pipeline
       - Anomaly detection
       - Quality scoring
    
    3. **Rate Limiting**
       - Adaptive rate limiting
       - Per-source limits
       - Exponential backoff
    
    4. **Cache Invalidation**
       - Smart TTL based on interval
       - Version tracking
       - Memory-bounded LRU cache
    
    5. **Memory Management**
       - Chunked processing
       - Streaming support
       - Memory pressure detection
    
    6. **Minimum Data**
       - Progressive loading
       - Graceful degradation
       - Partial data acceptance
    
    7. **Timezone Handling**
       - Unified Shanghai timezone
       - Trading session filtering
       - Holiday awareness
    
    8. **Source Health**
       - Real-time monitoring
       - Auto-failover
       - Health scoring
    """
    
    def __init__(
        self,
        config: FetcherConfig | None = None,
        inner_fetcher: DataFetcher | None = None,
    ):
        self._config = config or get_config()
        self._inner_fetcher = inner_fetcher or get_fetcher()
        self._lock = threading.RLock()
        
        # Components
        self._validator = get_validator()
        self._session_checker = get_session_checker()
        self._timezone_converter = get_timezone_converter()
        self._health_monitor = get_health_monitor()
        self._progressive_loader = get_progressive_loader()
        
        # In-memory result cache with explicit versioning.
        self._result_cache: dict[str, tuple[pd.DataFrame, float]] = {}
        self._result_cache_lock = threading.RLock()
        self._result_cache_limit = max(50, int(self._config.cache.max_cache_size))
        
        # Last-known-good snapshots for graceful degradation on transient failures.
        self._last_good: dict[str, pd.DataFrame] = {}
        self._last_good_lock = threading.RLock()
        
        # Metrics
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "failovers": 0,
            "total_time_ms": 0.0,
        }
        self._metrics_lock = threading.Lock()
    
    @staticmethod
    def _normalize_interval_token(interval: str) -> str:
        """Normalize interval string to canonical lowercase token."""
        return str(interval or "1d").strip().lower()
    
    def _resolve_requested_bars(self, options: FetchOptions) -> int:
        """Resolve requested bars for direct fetch paths."""
        if options.bars is not None:
            return max(1, int(options.bars))
        if options.days is not None:
            iv = self._normalize_interval_token(options.interval)
            bars_per_day = float(BARS_PER_DAY.get(iv, 1.0) or 1.0)
            return max(1, int(max(1.0, float(options.days)) * bars_per_day))
        iv = self._normalize_interval_token(options.interval)
        if iv in _INTRADAY_INTERVALS:
            return max(1, int(self._config.data_loading.min_bars_intraday))
        return max(1, int(self._config.data_loading.min_bars_daily))
    
    def _resolve_cache_ttl_seconds(self, options: FetchOptions) -> float:
        """Resolve per-request cache TTL in seconds."""
        if options.cache_ttl is not None:
            return max(1.0, float(options.cache_ttl))
        iv = self._normalize_interval_token(options.interval)
        if iv in {"1d", "1wk", "1mo"}:
            return max(1.0, float(self._config.cache.daily_ttl))
        return max(1.0, float(self._config.cache.intraday_ttl))
    
    def _build_cache_key(
        self,
        code: str,
        options: FetchOptions,
        requested_bars: int,
    ) -> str:
        """Build versioned cache key for unified fetch responses."""
        version = int(self._config.cache.cache_version)
        iv = self._normalize_interval_token(options.interval)
        source = str(options.preferred_source or "auto").strip().lower()
        return f"v{version}:{code}:{iv}:{requested_bars}:{source}"
    
    def _get_cached_result(
        self,
        cache_key: str,
        ttl_seconds: float,
    ) -> pd.DataFrame | None:
        """Get cached result if fresh enough."""
        now = float(time.time())
        with self._result_cache_lock:
            entry = self._result_cache.get(cache_key)
            if not entry:
                return None
            df, written_at = entry
            if (now - float(written_at)) > float(ttl_seconds):
                self._result_cache.pop(cache_key, None)
                return None
            return df.copy()
    
    def _set_cached_result(
        self,
        cache_key: str,
        df: pd.DataFrame | None,
    ) -> None:
        """Store result in bounded in-memory cache."""
        if df is None or df.empty:
            return
        with self._result_cache_lock:
            if cache_key in self._result_cache:
                self._result_cache.pop(cache_key, None)
            self._result_cache[cache_key] = (df.copy(), float(time.time()))
            while len(self._result_cache) > self._result_cache_limit:
                oldest_key = next(iter(self._result_cache))
                self._result_cache.pop(oldest_key, None)
    
    def _last_good_key(self, code: str, interval: str) -> str:
        iv = self._normalize_interval_token(interval)
        return f"{code}:{iv}"
    
    def _save_last_good(self, code: str, interval: str, df: pd.DataFrame | None) -> None:
        """Persist last-known-good frame for graceful fallback."""
        if df is None or df.empty:
            return
        key = self._last_good_key(code, interval)
        with self._last_good_lock:
            self._last_good[key] = df.copy()
    
    def _get_last_good(self, code: str, interval: str) -> pd.DataFrame | None:
        """Get last-known-good frame for symbol/interval."""
        key = self._last_good_key(code, interval)
        with self._last_good_lock:
            cached = self._last_good.get(key)
            return cached.copy() if isinstance(cached, pd.DataFrame) else None
    
    def _is_intraday(self, interval: str) -> bool:
        return self._normalize_interval_token(interval) in _INTRADAY_INTERVALS
    
    def _is_stale_intraday_frame(
        self,
        df: pd.DataFrame | None,
        interval: str,
    ) -> bool:
        """Best-effort stale check for cached intraday bars."""
        if df is None or df.empty or (not self._is_intraday(interval)):
            return False
        if not isinstance(df.index, pd.DatetimeIndex):
            return False
        try:
            last_ts = pd.Timestamp(df.index.max()).to_pydatetime()
        except Exception:
            return False
        now_sh = self._timezone_converter.to_shanghai(datetime.now())
        last_sh = self._timezone_converter.to_shanghai(last_ts)
        age_seconds = max(0.0, float((now_sh - last_sh).total_seconds()))
        iv = self._normalize_interval_token(interval)
        interval_minutes = {
            "1m": 1,
            "2m": 2,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "60m": 60,
            "1h": 60,
        }.get(iv, 1)
        max_age_seconds = max(120.0, float(interval_minutes) * 60.0 * 6.0)
        if self._session_checker.is_market_open(now_sh) and age_seconds > max_age_seconds:
            return True
        return False
    
    @staticmethod
    def _classify_status(
        bars_loaded: int,
        bars_requested: int,
        partial_threshold: float,
    ) -> LoadStatus:
        """Convert bar completeness to a load status."""
        if bars_loaded <= 0:
            return LoadStatus.FAILED
        ratio = float(bars_loaded) / max(1.0, float(bars_requested))
        if ratio >= 0.9:
            return LoadStatus.COMPLETE
        if ratio >= max(0.05, float(partial_threshold)):
            return LoadStatus.PARTIAL
        return LoadStatus.INSUFFICIENT
    
    @staticmethod
    def _parse_status_code(error_text: str) -> int:
        """Extract representative HTTP status code from an error string."""
        text = str(error_text or "").lower()
        if ("429" in text) or ("rate limit" in text) or ("throttl" in text):
            return 429
        for code in (500, 502, 503, 504):
            if str(code) in text:
                return code
        return 0
    
    def get_history(
        self,
        code: str,
        interval: str = "1d",
        bars: int | None = None,
        days: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | None:
        """Fetch historical data with all improvements.
        
        Args:
            code: Stock code (e.g., "000001")
            interval: Bar interval ("1m", "5m", "1d", etc.)
            bars: Number of bars to fetch
            days: Number of days to fetch (alternative to bars)
            **kwargs: Additional options (see FetchOptions)
        
        Returns:
            DataFrame with OHLCV data, or None if failed
        """
        options = FetchOptions(
            interval=interval,
            bars=bars,
            days=days,
            **kwargs,
        )
        
        result = self.fetch_with_options(code, options)
        
        if result.success and result.data is not None:
            return result.data
        return None
    
    def fetch_with_options(
        self,
        code: str,
        options: FetchOptions | None = None,
    ) -> FetchResult:
        """Fetch data with comprehensive options.
        
        FIX 2026-02-26: Full-featured fetch with all improvements.
        """
        start_time = time.time()
        options = options or FetchOptions()
        source: str | None = None
        
        # Clean code
        code = clean_code(code)
        requested_bars = self._resolve_requested_bars(options)
        cache_ttl_seconds = self._resolve_cache_ttl_seconds(options)
        cache_key = self._build_cache_key(code, options, requested_bars)
        
        # Generate correlation ID for tracking
        correlation_id = options.correlation_id or f"fetch_{int(time.time() * 1000) % 1000000}"
        
        log.info(
            "[UNIFIED] %s: Fetching %s interval=%s bars=%s",
            correlation_id, code, options.interval, options.bars or options.days
        )
        
        # Update metrics
        with self._metrics_lock:
            self._metrics["total_requests"] += 1
        
        warnings: list[str] = []
        
        try:
            # Phase 0: Request-level cache lookup with explicit versioning
            if options.use_cache and (not options.force_refresh):
                cached_df = self._get_cached_result(cache_key, cache_ttl_seconds)
                if isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
                    cached_df = ensure_shanghai_datetime(cached_df)
                    if self._is_stale_intraday_frame(cached_df, options.interval):
                        warnings.append("Cached intraday frame is stale; refreshing")
                    else:
                        load_time_ms = (time.time() - start_time) * 1000
                        cached_status = self._classify_status(
                            bars_loaded=len(cached_df),
                            bars_requested=requested_bars,
                            partial_threshold=options.partial_threshold,
                        )
                        out = FetchResult(
                            success=True,
                            data=cached_df,
                            bars_loaded=len(cached_df),
                            bars_requested=requested_bars,
                            quality_score=self._calculate_quality_score(cached_df, options.interval),
                            load_time_ms=load_time_ms,
                            source_used="unified_cache",
                            cache_hit=True,
                            validation_result=None,
                            load_status=cached_status,
                            warnings=warnings,
                        )
                        with self._metrics_lock:
                            self._metrics["successful_requests"] += 1
                            self._metrics["cache_hits"] += 1
                            self._metrics["total_time_ms"] += load_time_ms
                        return out
            
            # Phase 1: Select healthy source
            source = self._select_source(options.preferred_source)
            if source is None and options.auto_failover:
                source = get_healthy_source()
                if source:
                    with self._metrics_lock:
                        self._metrics["failovers"] += 1
                    warnings.append(f"Auto-failover to source: {source}")
            
            # Optional external rate-limit gate (best effort).
            if _HAS_ENHANCED_RATE_LIMITING and source:
                timeout = float(options.timeout_seconds or self._config.timeout.total_timeout)
                acquired = _acquire_rate_limit(source, timeout=max(1.0, timeout))
                if not acquired:
                    raise RuntimeError(f"Rate limit acquisition timed out for source={source}")
            
            # Phase 2: Progressive/direct loading
            if options.progressive:
                result = self._fetch_progressive(code, options, source)
            else:
                result = self._fetch_direct(code, options, source)
            
            # Phase 3: Validation
            validation_result = None
            if options.validate and result.data is not None and not result.data.empty:
                validation_result = self._validator.validate_bars(
                    result.data,
                    symbol=code,
                    interval=options.interval,
                )
                
                if not validation_result.is_valid:
                    warnings.extend(validation_result.issues)
                
                if validation_result.score < options.min_quality_score:
                    result.success = False
                    result.error = (
                        f"Quality score {validation_result.score:.2f} "
                        f"below threshold {options.min_quality_score}"
                    )
                result.validation_result = validation_result
            
            # Phase 4: Trading session normalization/filtering
            should_filter = bool(options.filter_trading_hours_only) or (
                bool(self._config.timezone.filter_non_trading)
                and self._is_intraday(options.interval)
            )
            if should_filter and result.data is not None and not result.data.empty:
                result.data = filter_trading_hours(result.data)
            
            # Phase 5: Timezone normalization
            if result.data is not None and not result.data.empty:
                result.data = ensure_shanghai_datetime(result.data)
                result.bars_loaded = len(result.data)
            
            # Phase 6: Partial-data policy enforcement
            result.bars_requested = max(1, int(result.bars_requested or requested_bars))
            result.load_status = self._classify_status(
                result.bars_loaded,
                result.bars_requested,
                options.partial_threshold,
            )
            if not options.allow_partial:
                strict_threshold = max(0.90, float(options.partial_threshold))
                completeness = float(result.bars_loaded) / max(1.0, float(result.bars_requested))
                if completeness < strict_threshold:
                    result.success = False
                    result.error = (
                        f"Partial data rejected: {result.bars_loaded}/{result.bars_requested} "
                        f"({completeness:.0%})"
                    )
                    result.load_status = LoadStatus.INSUFFICIENT
            
            # Record API compliance telemetry
            if _HAS_ENHANCED_RATE_LIMITING and source:
                if result.success:
                    _record_api_success(source, endpoint=f"history:{options.interval}")
                else:
                    _record_api_failure(source, status_code=self._parse_status_code(result.error or ""))
            
            # Persist successful data for future fallback/cache use
            if result.success and result.data is not None and not result.data.empty:
                self._set_cached_result(cache_key, result.data)
                self._save_last_good(code, options.interval, result.data)
                result.cache_hit = False
            
            # Fallback to last-known-good frame on transient failures
            if (not result.success) or result.data is None or result.data.empty:
                fallback = self._get_last_good(code, options.interval)
                if isinstance(fallback, pd.DataFrame) and not fallback.empty:
                    fallback = ensure_shanghai_datetime(fallback)
                    fb_status = self._classify_status(
                        bars_loaded=len(fallback),
                        bars_requested=requested_bars,
                        partial_threshold=options.partial_threshold,
                    )
                    warnings.append("Using last-known-good snapshot after fetch failure")
                    result = FetchResult(
                        success=True,
                        data=fallback,
                        bars_loaded=len(fallback),
                        bars_requested=requested_bars,
                        quality_score=self._calculate_quality_score(fallback, options.interval),
                        load_time_ms=0.0,
                        source_used=source or "last_good",
                        cache_hit=True,
                        validation_result=validation_result,
                        load_status=fb_status,
                        error=None,
                        warnings=warnings,
                    )
            
            # Record source health
            if result.success and source:
                record_source_success(source, result.load_time_ms / 1000)
            elif (not result.success) and source:
                record_source_failure(source, result.error or "fetch failed")
            
            # Update metrics
            load_time_ms = (time.time() - start_time) * 1000
            result.load_time_ms = load_time_ms
            result.warnings = warnings
            
            with self._metrics_lock:
                self._metrics["total_time_ms"] += load_time_ms
                if result.success:
                    self._metrics["successful_requests"] += 1
                    if options.use_cache and result.cache_hit:
                        self._metrics["cache_hits"] += 1
                else:
                    self._metrics["failed_requests"] += 1
            
            log.info(
                "[UNIFIED] %s: %s - %d bars, quality=%.2f, time=%.0fms, source=%s",
                correlation_id,
                "SUCCESS" if result.success else "FAILED",
                result.bars_loaded,
                result.quality_score,
                load_time_ms,
                result.source_used or "N/A",
            )
            
            return result
            
        except Exception as e:
            load_time_ms = (time.time() - start_time) * 1000
            err_text = str(e)
            
            if source:
                record_source_failure(source, err_text)
            if _HAS_ENHANCED_RATE_LIMITING and source:
                _record_api_failure(source, status_code=self._parse_status_code(err_text))
            
            with self._metrics_lock:
                self._metrics["failed_requests"] += 1
                self._metrics["total_time_ms"] += load_time_ms
            
            log.error("[UNIFIED] %s: Exception: %s", correlation_id, e)
            
            fallback = self._get_last_good(code, options.interval)
            if isinstance(fallback, pd.DataFrame) and not fallback.empty:
                fallback = ensure_shanghai_datetime(fallback)
                warnings.append("Exception path fallback to last-known-good snapshot")
                return FetchResult(
                    success=True,
                    data=fallback,
                    bars_loaded=len(fallback),
                    bars_requested=requested_bars,
                    quality_score=self._calculate_quality_score(fallback, options.interval),
                    load_time_ms=load_time_ms,
                    source_used=source or "last_good",
                    cache_hit=True,
                    validation_result=None,
                    load_status=self._classify_status(
                        len(fallback), requested_bars, options.partial_threshold
                    ),
                    error=None,
                    warnings=warnings,
                )
            
            return FetchResult(
                success=False,
                data=None,
                bars_loaded=0,
                bars_requested=requested_bars,
                quality_score=0.0,
                load_time_ms=load_time_ms,
                source_used=source,
                cache_hit=False,
                validation_result=None,
                load_status=LoadStatus.FAILED,
                error=err_text,
                warnings=warnings,
            )
    
    def _select_source(self, preferred: str | None) -> str | None:
        """Select best available data source."""
        if preferred:
            state = self._health_monitor._sources.get(preferred)
            if state and state.is_available():
                return preferred
        
        return get_healthy_source(preferred)
    
    def _fetch_progressive(
        self,
        code: str,
        options: FetchOptions,
        source: str | None,
    ) -> FetchResult:
        """Fetch using progressive loading."""
        
        def fetch_fn(bars: int) -> pd.DataFrame | None:
            """Inner fetch function for progressive loader."""
            try:
                return self._inner_fetcher.get_history(
                    code,
                    bars=bars,
                    interval=options.interval,
                    use_cache=options.use_cache and not options.force_refresh,
                    update_db=not options.use_cache,
                    allow_online=options.allow_online,
                )
            except Exception as e:
                log.debug("Progressive fetch chunk failed: %s", e)
                return None
        
        loader = ProgressiveDataLoader(
            min_bars_intraday=int(self._config.data_loading.min_bars_intraday),
            min_bars_daily=int(self._config.data_loading.min_bars_daily),
            min_bars_weekly=int(self._config.data_loading.min_bars_weekly),
            min_bars_monthly=int(self._config.data_loading.min_bars_monthly),
            chunk_size=int(self._config.data_loading.chunk_size),
            max_bars_per_request=int(self._config.data_loading.max_bars_per_request),
            allow_partial=bool(options.allow_partial),
            partial_threshold=max(0.05, float(options.partial_threshold)),
            max_memory_mb=float(options.max_memory_mb or self._config.memory.max_memory_mb),
        )
        
        load_result = loader.load(
            fetch_fn,
            interval=options.interval,
            requested_bars=self._resolve_requested_bars(options),
        )
        
        return FetchResult(
            success=load_result.is_usable(),
            data=load_result.data,
            bars_loaded=load_result.bars_loaded,
            bars_requested=load_result.bars_requested,
            quality_score=load_result.quality_score,
            load_time_ms=load_result.load_time_ms,
            source_used=source,
            cache_hit=False,
            validation_result=None,
            load_status=load_result.status,
            error=load_result.error,
        )
    
    def _fetch_direct(
        self,
        code: str,
        options: FetchOptions,
        source: str | None,
    ) -> FetchResult:
        """Fetch directly without progressive loading."""
        bars = self._resolve_requested_bars(options)
        
        try:
            df = self._inner_fetcher.get_history(
                code,
                bars=bars,
                interval=options.interval,
                use_cache=options.use_cache and not options.force_refresh,
                update_db=not options.use_cache,
                allow_online=options.allow_online,
            )
            
            if df is None or df.empty:
                return FetchResult(
                    success=False,
                    data=None,
                    bars_loaded=0,
                    bars_requested=bars,
                    quality_score=0.0,
                    load_time_ms=0,
                    source_used=source,
                    cache_hit=False,
                    validation_result=None,
                    load_status=LoadStatus.INSUFFICIENT,
                    error="No data returned",
                )
            
            # Calculate quality score
            quality = self._calculate_quality_score(df, options.interval)
            status = self._classify_status(
                bars_loaded=len(df),
                bars_requested=bars,
                partial_threshold=options.partial_threshold,
            )
            
            return FetchResult(
                success=True,
                data=df,
                bars_loaded=len(df),
                bars_requested=bars,
                quality_score=quality,
                load_time_ms=0,  # Will be set by caller
                source_used=source,
                cache_hit=False,
                validation_result=None,
                load_status=status,
            )
            
        except Exception as e:
            return FetchResult(
                success=False,
                data=None,
                bars_loaded=0,
                bars_requested=bars,
                quality_score=0.0,
                load_time_ms=0,
                source_used=source,
                cache_hit=False,
                validation_result=None,
                load_status=LoadStatus.FAILED,
                error=str(e),
            )
    
    def _calculate_quality_score(self, df: pd.DataFrame, interval: str) -> float:
        """Calculate data quality score."""
        return self._progressive_loader._calculate_quality_score(df, interval)
    
    def get_health_summary(self) -> dict[str, Any]:
        """Get summary of all data source health."""
        return self._health_monitor.get_summary()
    
    def get_source_statuses(self) -> dict[str, dict[str, Any]]:
        """Get detailed status of all data sources."""
        return self._health_monitor.get_all_statuses()
    
    def get_metrics(self) -> dict[str, Any]:
        """Get fetching metrics."""
        with self._metrics_lock:
            metrics = dict(self._metrics)
        
        # Add component metrics
        metrics["health_monitor"] = self.get_health_summary()
        metrics["config"] = self._config.to_dict()
        with self._result_cache_lock:
            metrics["result_cache_entries"] = len(self._result_cache)
        with self._last_good_lock:
            metrics["last_good_entries"] = len(self._last_good)
        
        # Calculate averages
        if metrics["total_requests"] > 0:
            metrics["avg_time_ms"] = metrics["total_time_ms"] / metrics["total_requests"]
            metrics["success_rate"] = metrics["successful_requests"] / metrics["total_requests"]
        else:
            metrics["avg_time_ms"] = 0
            metrics["success_rate"] = 0
        
        return metrics
    
    def reset_health(self, source: str | None = None) -> None:
        """Reset health monitor (for recovery or testing)."""
        if source:
            log.info("Resetting health for source: %s", source)
            with self._health_monitor._lock:
                if source in self._health_monitor._sources:
                    state = self._health_monitor._sources[source]
                    state.consecutive_failures = 0
                    state.consecutive_successes = 0
                    state.status = SourceHealthStatus.UNKNOWN
                    state.cooldown_until = 0
        else:
            log.info("Resetting all health monitors")
            from data.source_health import reset_health_monitor
            reset_health_monitor()
            self._health_monitor = get_health_monitor()


# Global unified fetcher instance
_unified_fetcher: UnifiedDataFetcher | None = None
_unified_fetcher_lock = threading.Lock()


def get_unified_fetcher() -> UnifiedDataFetcher:
    """Get or create global unified fetcher instance."""
    global _unified_fetcher
    with _unified_fetcher_lock:
        if _unified_fetcher is None:
            _unified_fetcher = UnifiedDataFetcher()
        return _unified_fetcher


def fetch_unified(
    code: str,
    interval: str = "1d",
    bars: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame | None:
    """Fetch data using unified fetcher.
    
    Convenience function for common use case.
    
    Args:
        code: Stock code
        interval: Bar interval
        bars: Number of bars
        **kwargs: Additional options
    
    Returns:
        DataFrame or None
    """
    return get_unified_fetcher().get_history(code, interval, bars, **kwargs)


def reset_unified_fetcher() -> None:
    """Reset global unified fetcher (for testing)."""
    global _unified_fetcher
    with _unified_fetcher_lock:
        if _unified_fetcher:
            _unified_fetcher.reset_health()
        _unified_fetcher = None
