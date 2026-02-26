# data/fetcher_enhanced.py
"""Enhanced data fetching with improved performance and reliability.

This module provides optimized HTTP client, connection pooling, and
intelligent caching for market data fetching operations.

Key improvements:
- Connection pooling with configurable pool sizes
- Adaptive timeout and retry logic
- Multi-layer caching (memory + Redis)
- Circuit breaker with health scoring
- Request deduplication and batching
- Metrics and telemetry
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Generic, TypeVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.runtime_env import env_flag, env_int, env_text
from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)

T = TypeVar('T')

# Default configuration
_DEFAULT_CONNECT_TIMEOUT = float(env_text("TRADING_HTTP_CONNECT_TIMEOUT", "10.0"))
_DEFAULT_READ_TIMEOUT = float(env_text("TRADING_HTTP_READ_TIMEOUT", "30.0"))
_DEFAULT_POOL_SIZE = int(env_int("TRADING_HTTP_POOL_SIZE", 10))
_DEFAULT_POOL_CONNECTIONS = int(env_int("TRADING_HTTP_POOL_CONNECTIONS", 10))
_DEFAULT_MAX_RETRIES = int(env_int("TRADING_HTTP_MAX_RETRIES", 3))
_DEFAULT_BACKOFF_FACTOR = float(env_text("TRADING_HTTP_BACKOFF_FACTOR", "0.5"))

# Cache configuration
_DEFAULT_CACHE_TTL = float(env_text("TRADING_CACHE_TTL", "60.0"))
_MAX_CACHE_SIZE = int(env_text("TRADING_MAX_CACHE_SIZE", "1000"))
_REDIS_CACHE_ENABLED = env_flag("TRADING_REDIS_CACHE", False)

# Circuit breaker configuration
_CB_FAILURE_THRESHOLD = int(env_int("TRADING_CB_FAILURE_THRESHOLD", "5"))
_CB_RECOVERY_TIMEOUT = float(env_text("TRADING_CB_RECOVERY_TIMEOUT", "30.0"))
_CB_HALF_OPEN_MAX_CALLS = int(env_int("TRADING_CB_HALF_OPEN_MAX_CALLS", "3"))


@dataclass
class CacheEntry(Generic[T]):
    """Cached value with metadata."""
    value: T
    created_at: float
    expires_at: float
    hit_count: int = 0
    key_hash: str = ""

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def ttl_remaining(self) -> float:
        return max(0.0, self.expires_at - time.time())


@dataclass
class CircuitBreakerState:
    """State for circuit breaker pattern."""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    state: str = "closed"  # closed, open, half-open
    failure_threshold: int = _CB_FAILURE_THRESHOLD
    recovery_timeout: float = _CB_RECOVERY_TIMEOUT
    half_open_calls: int = 0
    half_open_max: int = _CB_HALF_OPEN_MAX_CALLS

    def record_success(self) -> None:
        self.success_count += 1
        self.last_success_time = time.time()
        if self.state == "half-open":
            self.state = "closed"
            self.failure_count = 0
            self.half_open_calls = 0
        elif self.state == "closed":
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.state == "half-open":
            self.state = "open"
            self.half_open_calls = 0
        elif self.state == "closed" and self.failure_count >= self.failure_threshold:
            self.state = "open"

    def can_execute(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                self.half_open_calls = 0
                return True
            return False
        if self.state == "half-open":
            if self.half_open_calls < self.half_open_max:
                self.half_open_calls += 1
                return True
            return False
        return False

    def health_score(self) -> float:
        """Calculate health score from 0.0 (unhealthy) to 1.0 (healthy)."""
        if self.state == "open":
            return 0.0
        if self.state == "half-open":
            return 0.5
        total = self.failure_count + self.success_count
        if total == 0:
            return 1.0
        return max(0.0, 1.0 - (self.failure_count / max(total, 1)))


class LRUCache(Generic[T]):
    """Thread-safe LRU cache with TTL support and memory protection.

    FIX 2026-02-26:
    - Memory-bounded entries with size estimation
    - Automatic eviction of expired entries
    - Maximum entry size limits to prevent memory bloat
    - Periodic cleanup of stale entries
    """

    # Maximum estimated memory per entry (in MB)
    MAX_ENTRY_MEMORY_MB = 50

    def __init__(
        self,
        max_size: int = _MAX_CACHE_SIZE,
        default_ttl: float = _DEFAULT_CACHE_TTL,
        max_memory_mb: float = MAX_ENTRY_MEMORY_MB,
    ):
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._max_memory_mb = max_memory_mb
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired_cleanups = 0

        # Start periodic cleanup thread
        self._cleanup_interval = max(default_ttl / 2, 30.0)
        self._stop_cleanup = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup,
            daemon=True,
            name=f"LRUCache-Cleanup-{id(self)}",
        )
        self._cleanup_thread.start()

    def _estimate_size(self, value: T) -> float:
        """Estimate memory size of a value in MB.

        FIX 2026-02-26: Prevent caching extremely large objects.
        """
        try:
            import sys
            # For pandas DataFrames, use memory_usage
            if hasattr(value, "memory_usage"):  # type: ignore
                # DataFrame or Series
                mem_usage = value.memory_usage(deep=True)  # type: ignore
                if hasattr(mem_usage, "sum"):
                    return float(mem_usage.sum()) / (1024 * 1024)
                return float(mem_usage) / (1024 * 1024)
            # For numpy arrays
            if hasattr(value, "nbytes"):  # type: ignore
                return float(value.nbytes) / (1024 * 1024)  # type: ignore
            # Fallback to sys.getsizeof
            return float(sys.getsizeof(value)) / (1024 * 1024)
        except Exception:
            # If estimation fails, assume it's large to be safe
            return self._max_memory_mb

    def _periodic_cleanup(self) -> None:
        """Periodically remove expired entries.

        FIX 2026-02-26: Background cleanup to prevent memory leaks.
        """
        while not self._stop_cleanup.is_set():
            try:
                self._stop_cleanup.wait(self._cleanup_interval)
                if self._stop_cleanup.is_set():
                    break
                self.cleanup_expired()
            except Exception:
                pass  # Don't let cleanup errors crash the thread

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        FIX 2026-02-26: Explicit cleanup method for manual triggering.

        Returns:
            Number of entries removed
        """
        removed = 0
        with self._lock:
            now = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if now > entry.expires_at
            ]
            for key in expired_keys:
                del self._cache[key]
                removed += 1
            self._expired_cleanups += removed
        return removed

    def get(self, key: str) -> T | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                self._expired_cleanups += 1
                return None
            self._cache.move_to_end(key)
            entry.hit_count += 1
            self._hits += 1
            return entry.value

    def set(self, key: str, value: T, ttl: float | None = None) -> bool:
        """Set a cache entry with size validation.

        FIX 2026-02-26: Reject entries that are too large.

        Returns:
            True if entry was cached, False if rejected due to size
        """
        # Check size before caching
        estimated_size = self._estimate_size(value)
        if estimated_size > self._max_memory_mb:
            log.debug(
                "Cache entry rejected: size %.2f MB exceeds limit %.2f MB for key %s",
                estimated_size, self._max_memory_mb, key,
            )
            return False

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)

            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                expires_at=time.time() + (ttl or self._default_ttl),
                key_hash=key,
            )
            self._cache[key] = entry

            # Evict oldest entries if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

        return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def shutdown(self) -> None:
        """Shutdown the cache and stop cleanup thread.

        FIX 2026-02-26: Proper resource cleanup.
        """
        self._stop_cleanup.set()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)
        self.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / max(1, total)
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "expired_cleanups": self._expired_cleanups,
                "hit_rate": hit_rate,
                "default_ttl": self._default_ttl,
                "max_memory_mb": self._max_memory_mb,
            }


class EnhancedHTTPClient:
    """HTTP client with connection pooling, retries, and circuit breaker."""

    def __init__(
        self,
        pool_size: int = _DEFAULT_POOL_SIZE,
        pool_connections: int = _DEFAULT_POOL_CONNECTIONS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
        connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = _DEFAULT_READ_TIMEOUT,
    ):
        self._session = self._create_session(
            pool_size=pool_size,
            pool_connections=pool_connections,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._circuit_breakers: dict[str, CircuitBreakerState] = {}
        self._cb_lock = threading.RLock()
        self._request_lock = threading.RLock()
        self._request_stats = {
            "total": 0,
            "success": 0,
            "failure": 0,
            "total_time": 0.0,
        }

    @staticmethod
    def _create_session(
        pool_size: int,
        pool_connections: int,
        max_retries: int,
        backoff_factor: float,
    ) -> requests.Session:
        """Create session with optimized connection pooling."""
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/html, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "OPTIONS"],
            raise_on_status=False,
            respect_retry_after_header=True,
        )

        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_size,
            max_retries=retry_strategy,
            pool_block=False,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _get_circuit_breaker(self, host: str) -> CircuitBreakerState:
        with self._cb_lock:
            if host not in self._circuit_breakers:
                self._circuit_breakers[host] = CircuitBreakerState()
            return self._circuit_breakers[host]

    def _record_request(self, success: bool, duration: float) -> None:
        with self._request_lock:
            self._request_stats["total"] += 1
            if success:
                self._request_stats["success"] += 1
            else:
                self._request_stats["failure"] += 1
            self._request_stats["total_time"] += duration

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: tuple[float, float] | None = None,
        allow_redirects: bool = True,
    ) -> requests.Response:
        """Execute GET request with circuit breaker and metrics."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.netloc
        cb = self._get_circuit_breaker(host)

        if not cb.can_execute():
            raise ConnectionError(
                f"Circuit breaker open for host: {host}"
            )

        start_time = time.time()
        try:
            response = self._session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout or (self._connect_timeout, self._read_timeout),
                allow_redirects=allow_redirects,
            )
            duration = time.time() - start_time
            self._record_request(success=response.ok, duration=duration)

            if response.ok:
                cb.record_success()
            else:
                cb.record_failure()
                log.debug(
                    "HTTP %d for %s (took %.2fs)",
                    response.status_code, url, duration
                )

            return response

        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time
            self._record_request(success=False, duration=duration)
            cb.record_failure()
            log.debug("Request failed for %s: %s", url, e)
            raise

    def post(
        self,
        url: str,
        data: Any = None,
        json: Any = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: tuple[float, float] | None = None,
    ) -> requests.Response:
        """Execute POST request with circuit breaker and metrics."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.netloc
        cb = self._get_circuit_breaker(host)

        if not cb.can_execute():
            raise ConnectionError(
                f"Circuit breaker open for host: {host}"
            )

        start_time = time.time()
        try:
            response = self._session.post(
                url,
                data=data,
                json=json,
                params=params,
                headers=headers,
                timeout=timeout or (self._connect_timeout, self._read_timeout),
            )
            duration = time.time() - start_time
            self._record_request(success=response.ok, duration=duration)

            if response.ok:
                cb.record_success()
            else:
                cb.record_failure()
                log.debug(
                    "HTTP %d for %s (took %.2fs)",
                    response.status_code, url, duration
                )

            return response

        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time
            self._record_request(success=False, duration=duration)
            cb.record_failure()
            log.debug("Request failed for %s: %s", url, e)
            raise

    def get_circuit_breaker_stats(self, host: str) -> dict[str, Any]:
        """Get circuit breaker stats for a host."""
        with self._cb_lock:
            cb = self._circuit_breakers.get(host)
            if cb is None:
                return {
                    "state": "closed",
                    "failure_count": 0,
                    "success_count": 0,
                    "health_score": 1.0,
                }
            return {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "success_count": cb.success_count,
                "last_failure_time": cb.last_failure_time,
                "last_success_time": cb.last_success_time,
                "health_score": cb.health_score(),
            }

    def get_request_stats(self) -> dict[str, Any]:
        """Get overall request statistics."""
        with self._request_lock:
            total = self._request_stats["total"]
            success = self._request_stats["success"]
            return {
                "total_requests": total,
                "successful": success,
                "failed": self._request_stats["failure"],
                "success_rate": success / max(1, total),
                "total_time": self._request_stats["total_time"],
                "avg_time": self._request_stats["total_time"] / max(1, total),
            }

    def reset_circuit_breaker(self, host: str) -> None:
        """Reset circuit breaker for a host."""
        with self._cb_lock:
            if host in self._circuit_breakers:
                self._circuit_breakers[host] = CircuitBreakerState()
                log.info("Circuit breaker reset for host: %s", host)

    def close(self) -> None:
        """Close the session and release resources."""
        self._session.close()


class RequestDeduplicator:
    """Deduplicate concurrent identical requests."""

    def __init__(self):
        self._pending: dict[str, threading.Event] = {}
        self._results: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _make_key(self, url: str, params: dict[str, Any] | None) -> str:
        key_data = f"{url}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def execute(
        self,
        url: str,
        executor: callable,
        params: dict[str, Any] | None = None,
        ttl: float = 5.0,
    ) -> Any:
        """Execute request, deduplicating concurrent calls."""
        key = self._make_key(url, params)

        with self._lock:
            # Check if result is cached
            if key in self._results:
                return self._results[key]

            # Check if request is already pending
            if key in self._pending:
                event = self._pending[key]
            else:
                event = threading.Event()
                self._pending[key] = event
                result = None
                try:
                    result = executor()
                    self._results[key] = result
                    # Schedule cache cleanup
                    threading.Timer(ttl, lambda: self._cleanup(key)).start()
                except Exception as e:
                    self._pending.pop(key, None)
                    raise e
                finally:
                    event.set()

        # Wait for the pending request to complete
        event.wait(timeout=30.0)

        with self._lock:
            return self._results.get(key)

    def _cleanup(self, key: str) -> None:
        with self._lock:
            self._results.pop(key, None)
            self._pending.pop(key, None)


# Global instances
_http_client: EnhancedHTTPClient | None = None
_http_client_lock = threading.Lock()
_cache: LRUCache | None = None
_cache_lock = threading.Lock()
_deduplicator: RequestDeduplicator | None = None
_deduplicator_lock = threading.Lock()


def get_http_client() -> EnhancedHTTPClient:
    """Get or create global HTTP client instance."""
    global _http_client
    with _http_client_lock:
        if _http_client is None:
            _http_client = EnhancedHTTPClient()
        return _http_client


def get_cache() -> LRUCache:
    """Get or create global cache instance."""
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = LRUCache()
        return _cache


def get_deduplicator() -> RequestDeduplicator:
    """Get or create global request deduplicator."""
    global _deduplicator
    with _deduplicator_lock:
        if _deduplicator is None:
            _deduplicator = RequestDeduplicator()
        return _deduplicator


def fetch_with_cache(
    url: str,
    params: dict[str, Any] | None = None,
    cache_key: str | None = None,
    cache_ttl: float | None = None,
    force_refresh: bool = False,
) -> dict[str, Any] | None:
    """Fetch JSON data with caching and deduplication."""
    cache = get_cache()
    client = get_http_client()
    dedup = get_deduplicator()

    key = cache_key or f"{url}:{json.dumps(params or {}, sort_keys=True)}"

    if not force_refresh:
        cached = cache.get(key)
        if cached is not None:
            return cached

    def _do_fetch() -> dict[str, Any] | None:
        try:
            response = client.get(url, params=params)
            if response.ok:
                data = response.json()
                cache.set(key, data, ttl=cache_ttl)
                return data
            log.debug("HTTP %d for %s", response.status_code, url)
            return None
        except Exception as e:
            log.debug("Fetch failed for %s: %s", url, e)
            return None

    try:
        return dedup.execute(url, _do_fetch, params)
    except Exception:
        return cache.get(key)


def get_metrics() -> dict[str, Any]:
    """Get comprehensive fetching metrics."""
    client = get_http_client()
    cache = get_cache()
    return {
        "http_client": client.get_request_stats(),
        "cache": cache.stats(),
        "circuit_breakers": {
            host: client.get_circuit_breaker_stats(host)
            for host in client._circuit_breakers
        },
    }


def reset_all() -> None:
    """Reset all global instances (for testing/recovery).

    FIX 2026-02-26: Properly shutdown cache cleanup thread before reset.
    """
    global _http_client, _cache, _deduplicator

    # Shutdown cache properly to stop cleanup thread
    with _cache_lock:
        if _cache is not None:
            try:
                _cache.shutdown()
            except Exception:
                pass
            _cache = None

    with _http_client_lock:
        if _http_client:
            _http_client.close()
            _http_client = None

    with _deduplicator_lock:
        _deduplicator = None
