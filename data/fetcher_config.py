# data/fetcher_config.py
"""Unified configuration for data fetching operations.

Centralizes all configuration for data fetching with:
- Timeout and retry settings
- Connection pooling parameters
- Cache configuration
- Rate limiting settings
- Circuit breaker thresholds
- Quality validation thresholds

FIX 2026-02-26: Addresses disadvantages:
1. Network dependency - configurable timeouts and retries
2. Data quality - validation thresholds
3. Rate limiting - adaptive settings
4. Cache invalidation - TTL and versioning
5. Memory - size limits and streaming
"""

from dataclasses import dataclass, field
from typing import Any

from config.runtime_env import env_flag, env_float, env_int, env_text


@dataclass(frozen=True)
class TimeoutConfig:
    """HTTP timeout configuration."""
    connect_timeout: float = float(env_text("TRADING_CONNECT_TIMEOUT", "8.0"))
    read_timeout: float = float(env_text("TRADING_READ_TIMEOUT", "20.0"))
    total_timeout: float = float(env_text("TRADING_TOTAL_TIMEOUT", "60.0"))
    
    @property
    def tuple(self) -> tuple[float, float]:
        return (self.connect_timeout, self.read_timeout)


@dataclass(frozen=True)
class RetryConfig:
    """Retry configuration with exponential backoff."""
    max_retries: int = int(env_int("TRADING_MAX_RETRIES", "3"))
    backoff_base: float = float(env_text("TRADING_BACKOFF_BASE", "0.5"))
    backoff_max: float = float(env_text("TRADING_BACKOFF_MAX", "8.0"))
    backoff_exponential: bool = bool(env_flag("TRADING_BACKOFF_EXPONENTIAL", "1"))
    retry_on_timeout: bool = bool(env_flag("TRADING_RETRY_ON_TIMEOUT", "1"))
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504)


@dataclass(frozen=True)
class ConnectionPoolConfig:
    """Connection pooling configuration."""
    pool_size: int = int(env_int("TRADING_POOL_SIZE", "10"))
    pool_connections: int = int(env_int("TRADING_POOL_CONNECTIONS", "5"))
    max_retries: int = int(env_int("TRADING_POOL_MAX_RETRIES", "3"))
    pool_block: bool = bool(env_flag("TRADING_POOL_BLOCK", "0"))
    keep_alive: bool = bool(env_flag("TRADING_KEEP_ALIVE", "1"))


@dataclass(frozen=True)
class CacheConfig:
    """Cache configuration with TTL and memory limits."""
    # TTL settings
    default_ttl: float = float(env_text("TRADING_CACHE_TTL", "120.0"))
    intraday_ttl: float = float(env_text("TRADING_INTRADAY_CACHE_TTL", "30.0"))
    daily_ttl: float = float(env_text("TRADING_DAILY_CACHE_TTL", "300.0"))
    
    # Memory limits
    max_cache_size: int = int(env_int("TRADING_MAX_CACHE_SIZE", "500"))
    max_entry_rows: int = int(env_int("TRADING_MAX_CACHE_ENTRY_ROWS", "50000"))
    max_entry_memory_mb: float = float(env_text("TRADING_MAX_ENTRY_MEMORY_MB", "25.0"))
    
    # Eviction settings
    cleanup_interval: float = float(env_text("TRADING_CACHE_CLEANUP_INTERVAL", "60.0"))
    eviction_on_memory_pressure: bool = bool(env_flag("TRADING_EVICTION_ON_PRESSURE", "1"))
    
    # Versioning for cache invalidation
    cache_version: int = int(env_int("TRADING_CACHE_VERSION", "1"))
    version_check_enabled: bool = bool(env_flag("TRADING_CACHE_VERSION_CHECK", "1"))


@dataclass(frozen=True)
class RateLimitConfig:
    """Rate limiting configuration."""
    # Base rates (requests per second)
    default_rate: float = float(env_text("TRADING_RATE_LIMIT", "2.0"))
    akshare_rate: float = float(env_text("TRADING_AKSHARE_RATE", "1.0"))
    sina_rate: float = float(env_text("TRADING_SINA_RATE", "1.0"))
    eastmoney_rate: float = float(env_text("TRADING_EASTMONEY_RATE", "1.5"))
    
    # Burst handling
    burst_size: int = int(env_int("TRADING_BURST_SIZE", "5"))
    
    # Backoff settings
    backoff_base: float = float(env_text("TRADING_RATE_BACKOFF_BASE", "2.0"))
    backoff_max: float = float(env_text("TRADING_RATE_BACKOFF_MAX", "60.0"))
    
    # Error rate thresholds
    error_rate_window: float = float(env_text("TRADING_ERROR_RATE_WINDOW", "60.0"))
    error_rate_threshold: float = float(env_text("TRADING_ERROR_RATE_THRESHOLD", "0.5"))

    # Network-aware adjustments (China-only mode)
    china_direct_multiplier: float = float(env_text("TRADING_CHINA_MULTIPLIER", "0.8"))


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = int(env_int("TRADING_CB_FAILURE_THRESHOLD", "5"))
    recovery_timeout: float = float(env_text("TRADING_CB_RECOVERY_TIMEOUT", "30.0"))
    half_open_max_calls: int = int(env_int("TRADING_CB_HALF_OPEN_CALLS", "3"))
    success_threshold: int = int(env_int("TRADING_CB_SUCCESS_THRESHOLD", "2"))
    
    # Per-source thresholds
    source_thresholds: dict[str, int] = field(default_factory=lambda: {
        "akshare": 5,
        "sina": 5,
        "eastmoney": 5,
        "tencent": 3,
        "default": 5,
    })


@dataclass(frozen=True)
class QualityConfig:
    """Data quality validation thresholds."""
    # Price validation
    max_price_ratio: float = float(env_text("TRADING_MAX_PRICE_RATIO", "1.5"))
    min_price: float = float(env_text("TRADING_MIN_PRICE", "0.01"))
    max_single_bar_change: float = float(env_text("TRADING_MAX_BAR_CHANGE", "0.25"))
    max_wick_ratio: float = float(env_text("TRADING_MAX_WICK_RATIO", "0.15"))
    
    # Volume validation
    max_volume_spike: float = float(env_text("TRADING_VOLUME_SPIKE", "10.0"))
    min_volume_ratio: float = float(env_text("TRADING_MIN_VOLUME_RATIO", "0.01"))
    
    # Data completeness
    max_nan_ratio: float = float(env_text("TRADING_MAX_NAN_RATIO", "0.5"))
    max_price_gap: float = float(env_text("TRADING_MAX_PRICE_GAP", "0.30"))
    min_data_points: int = int(env_int("TRADING_MIN_DATA_POINTS", "5"))
    
    # Quality scores for acceptance
    min_quality_score_intraday: float = float(env_text("TRADING_MIN_QUALITY_INTRADAY", "0.28"))
    min_quality_score_daily: float = float(env_text("TRADING_MIN_QUALITY_DAILY", "0.15"))
    max_scale_ratio_deviation: float = float(env_text("TRADING_MAX_SCALE_DEVIATION", "2.2"))


@dataclass(frozen=True)
class DataLoadingConfig:
    """Progressive data loading configuration."""
    # Minimum data requirements
    min_bars_intraday: int = int(env_int("TRADING_MIN_BARS_INTRADAY", "480"))  # 2 days of 1min
    min_bars_daily: int = int(env_int("TRADING_MIN_BARS_DAILY", "14"))
    min_bars_weekly: int = int(env_int("TRADING_MIN_BARS_WEEKLY", "8"))
    min_bars_monthly: int = int(env_int("TRADING_MIN_BARS_MONTHLY", "6"))
    
    # Progressive loading
    initial_bars_ratio: float = float(env_text("TRADING_INITIAL_BARS_RATIO", "0.5"))
    max_bars_per_request: int = int(env_int("TRADING_MAX_BARS_PER_REQUEST", "2000"))
    chunk_size: int = int(env_int("TRADING_LOADING_CHUNK_SIZE", "500"))
    
    # Fallback settings
    allow_partial_data: bool = bool(env_flag("TRADING_ALLOW_PARTIAL", "1"))
    partial_data_threshold: float = float(env_text("TRADING_PARTIAL_THRESHOLD", "0.4"))


@dataclass(frozen=True)
class MemoryConfig:
    """Memory management configuration."""
    # DataFrame chunking
    chunk_processing: bool = bool(env_flag("TRADING_CHUNK_PROCESSING", "1"))
    chunk_size_rows: int = int(env_int("TRADING_CHUNK_SIZE", "1000"))
    
    # Memory limits
    max_memory_mb: float = float(env_text("TRADING_MAX_MEMORY_MB", "500.0"))
    gc_on_memory_pressure: bool = bool(env_flag("TRADING_GC_ON_PRESSURE", "1"))
    memory_pressure_threshold: float = float(env_text("TRADING_MEMORY_PRESSURE_THRESHOLD", "0.85"))
    
    # Streaming
    enable_streaming: bool = bool(env_flag("TRADING_ENABLE_STREAMING", "1"))
    streaming_buffer_size: int = int(env_int("TRADING_STREAMING_BUFFER", "100"))


@dataclass(frozen=True)
class TimezoneConfig:
    """Timezone and session handling configuration."""
    # Timezone
    default_timezone: str = "Asia/Shanghai"
    force_naive: bool = bool(env_flag("TRADING_FORCE_NAIVE", "1"))
    
    # Trading sessions (China A-share)
    morning_start: str = "09:30"
    morning_end: str = "11:30"
    afternoon_start: str = "13:00"
    afternoon_end: str = "15:00"
    
    # Session filtering
    filter_non_trading: bool = bool(env_flag("TRADING_FILTER_NON_TRADING", "1"))
    allow_after_hours: bool = bool(env_flag("TRADING_ALLOW_AFTER_HOURS", "0"))


@dataclass(frozen=True)
class DataSourceConfig:
    """Data source configuration with health monitoring."""
    # Source priorities
    primary_sources: list[str] = field(default_factory=lambda: ["akshare", "sina", "eastmoney"])
    fallback_sources: list[str] = field(default_factory=lambda: ["tencent", "yahoo"])
    
    # Health monitoring
    health_check_interval: float = float(env_text("TRADING_HEALTH_CHECK_INTERVAL", "30.0"))
    health_check_timeout: float = float(env_text("TRADING_HEALTH_CHECK_TIMEOUT", "5.0"))
    unhealthy_threshold: int = int(env_int("TRADING_UNHEALTHY_THRESHOLD", "3"))
    healthy_threshold: int = int(env_int("TRADING_HEALTHY_THRESHOLD", "2"))
    
    # Auto-failover
    auto_failover: bool = bool(env_flag("TRADING_AUTO_FAILOVER", "1"))
    failover_cooldown: float = float(env_text("TRADING_FAILOVER_COOLDOWN", "10.0"))


@dataclass(frozen=True)
class FetcherConfig:
    """Master configuration for all data fetching operations.
    
    FIX 2026-02-26: Centralized configuration addressing all disadvantages:
    1. Network dependency -> timeout/retry config
    2. Data quality -> quality config
    3. Rate limiting -> rate limit config
    4. Cache invalidation -> cache config with versioning
    5. Memory -> memory config with streaming
    6. Minimum data -> data loading config
    7. Circuit breaker -> circuit breaker config
    8. Timezone -> timezone config
    9. Source health -> data source config
    """
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    connection_pool: ConnectionPoolConfig = field(default_factory=ConnectionPoolConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    data_loading: DataLoadingConfig = field(default_factory=DataLoadingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    timezone: TimezoneConfig = field(default_factory=TimezoneConfig)
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    
    # Global settings
    debug_logging: bool = bool(env_flag("TRADING_FETCHER_DEBUG", "0"))
    metrics_enabled: bool = bool(env_flag("TRADING_FETCHER_METRICS", "1"))
    
    @classmethod
    def from_env(cls) -> "FetcherConfig":
        """Create config from environment variables."""
        return cls()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "timeout": {
                "connect": self.timeout.connect_timeout,
                "read": self.timeout.read_timeout,
                "total": self.timeout.total_timeout,
            },
            "retry": {
                "max_retries": self.retry.max_retries,
                "backoff_base": self.retry.backoff_base,
                "backoff_max": self.retry.backoff_max,
            },
            "cache": {
                "default_ttl": self.cache.default_ttl,
                "max_size": self.cache.max_cache_size,
                "version": self.cache.cache_version,
            },
            "rate_limit": {
                "default_rate": self.rate_limit.default_rate,
                "burst_size": self.rate_limit.burst_size,
            },
            "circuit_breaker": {
                "failure_threshold": self.circuit_breaker.failure_threshold,
                "recovery_timeout": self.circuit_breaker.recovery_timeout,
            },
            "quality": {
                "min_score_intraday": self.quality.min_quality_score_intraday,
                "min_score_daily": self.quality.min_quality_score_daily,
            },
            "memory": {
                "max_memory_mb": self.memory.max_memory_mb,
                "chunk_processing": self.memory.chunk_processing,
                "streaming": self.memory.enable_streaming,
            },
        }


# Global default config instance
_default_config: FetcherConfig | None = None


def get_config() -> FetcherConfig:
    """Get or create global config instance."""
    global _default_config
    if _default_config is None:
        _default_config = FetcherConfig.from_env()
    return _default_config


def reset_config() -> None:
    """Reset global config (for testing)."""
    global _default_config
    _default_config = None
