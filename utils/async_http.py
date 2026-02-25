"""Modern async HTTP client with automatic retry, circuit breaker, and connection pooling.

This module provides production-grade async HTTP capabilities using aiohttp and httpx,
replacing the legacy synchronous requests-based approach.

Features:
    - Async I/O with aiohttp for high-concurrency scenarios
    - Automatic retry with exponential backoff
    - Circuit breaker pattern for fault tolerance
    - Connection pooling with keep-alive
    - Request/response logging
    - Rate limiting with token bucket algorithm
    - Proxy support (HTTP/SOCKS5)
    - China-optimized settings (extended timeouts, DNS optimization)

Example:
    >>> async with AsyncHttpClient() as client:
    ...     response = await client.get("https://api.example.com/data")
    ...     data = response.json()
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ParamSpec, TypeVar

import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from aiohttp_socks import ProxyConnector
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.runtime_env import env_text
from utils.logger import get_logger

log = get_logger()

P = ParamSpec("P")
R = TypeVar("R")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Failing, reject requests
    HALF_OPEN = auto()  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes before closing
    timeout: float = 30.0  # Seconds before half-open
    half_open_max_calls: int = 3  # Max calls in half-open state


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    calls_per_second: float = 10.0
    burst_size: int = 20


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    min_wait: float = 1.0
    max_wait: float = 60.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple[type[Exception], ...] = (
        aiohttp.ClientError,
        asyncio.TimeoutError,
    )


@dataclass
class HttpClientConfig:
    """Configuration for async HTTP client."""
    timeout: float = 30.0
    connect_timeout: float = 10.0
    sock_read_timeout: float = 20.0
    max_connections: int = 100
    max_connections_per_host: int = 10
    enable_ssl: bool = True
    proxy_url: str | None = None
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    user_agent: str = "TradingGraph/2.0"
    china_optimized: bool = False


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    async def call(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_try_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    log.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is OPEN")

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._config.half_open_max_calls:
                    raise CircuitBreakerOpenError("Circuit breaker HALF_OPEN call limit reached")
                self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception:
            await self._on_failure()
            raise

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self._success_count += 1
            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self._config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    log.info("Circuit breaker CLOSED after successful recovery")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self._failure_count += 1
            self._success_count = 0
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                log.warning("Circuit breaker OPEN after half-open failure")
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._config.failure_threshold:
                    self._state = CircuitState.OPEN
                    log.warning(f"Circuit breaker OPEN after {self._failure_count} failures")

    def _should_try_reset(self) -> bool:
        """Check if enough time has passed to try resetting."""
        if self._last_failure_time is None:
            return True
        return (time.time() - self._last_failure_time) >= self._config.timeout


class RateLimiter:
    """Token bucket rate limiter implementation."""

    def __init__(self, config: RateLimitConfig) -> None:
        self._config = config
        self._tokens = float(config.burst_size)
        self._last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            while True:
                now = time.time()
                elapsed = now - self._last_update
                self._tokens = min(
                    self._config.burst_size,
                    self._tokens + elapsed * self._config.calls_per_second
                )
                self._last_update = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                # Calculate wait time
                wait_time = (1.0 - self._tokens) / self._config.calls_per_second
                await asyncio.sleep(wait_time)


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class AsyncHttpClient:
    """Production-grade async HTTP client with enterprise features.

    This client provides:
        - Connection pooling with keep-alive
        - Automatic retry with exponential backoff
        - Circuit breaker for fault tolerance
        - Rate limiting
        - China-optimized settings
        - Proxy support
        - Comprehensive logging

    Example:
        >>> async with AsyncHttpClient() as client:
        ...     response = await client.get("https://api.example.com")
        ...     data = response.json()
    """

    def __init__(self, config: HttpClientConfig | None = None) -> None:
        """Initialize async HTTP client.

        Args:
            config: Client configuration. Uses defaults if not provided.
        """
        self._config = config or self._default_config()
        self._session: aiohttp.ClientSession | None = None
        self._circuit_breaker = CircuitBreaker(self._config.circuit_breaker)
        self._rate_limiter = RateLimiter(self._config.rate_limit)
        self._connector: TCPConnector | ProxyConnector | None = None

    def _default_config(self) -> HttpClientConfig:
        """Create default configuration with China optimizations."""
        china_mode = env_text("TRADING_CHINA_DIRECT", "0") == "1"
        vpn_mode = env_text("TRADING_VPN", "0") == "1"
        proxy_url = env_text("TRADING_PROXY_URL", "")

        config = HttpClientConfig(
            timeout=30.0,
            connect_timeout=10.0,
            china_optimized=china_mode or vpn_mode,
        )

        # China-optimized settings
        if config.china_optimized:
            config.timeout = 60.0
            config.connect_timeout = 30.0
            config.sock_read_timeout = 45.0
            log.info("China-optimized HTTP settings enabled")

        # Proxy configuration
        if proxy_url:
            config.proxy_url = proxy_url
            log.info(f"Using proxy: {proxy_url}")

        return config

    async def __aenter__(self) -> AsyncHttpClient:
        """Async context manager entry."""
        await self._create_session()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def _create_session(self) -> None:
        """Create aiohttp session with optimized settings."""
        if self._session is not None:
            return

        timeout = ClientTimeout(
            total=self._config.timeout,
            connect=self._config.connect_timeout,
            sock_read=self._config.sock_read_timeout,
        )

        # Create connector
        if self._config.proxy_url:
            self._connector = ProxyConnector.from_url(
                self._config.proxy_url,
                limit=self._config.max_connections,
                limit_per_host=self._config.max_connections_per_host,
                enable_cleanup_closed=True,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
        else:
            self._connector = TCPConnector(
                limit=self._config.max_connections,
                limit_per_host=self._config.max_connections_per_host,
                enable_cleanup_closed=True,
                ttl_dns_cache=300,
                use_dns_cache=True,
                # China-optimized DNS
                local_addr=("0.0.0.0", 0) if self._config.china_optimized else None,
            )

        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=timeout,
            headers={"User-Agent": self._config.user_agent},
        )

        log.info("Async HTTP session created")

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
        if self._connector:
            await self._connector.close()
            self._connector = None
        log.info("Async HTTP session closed")

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> aiohttp.ClientResponse:
        """Execute request with retry logic."""
        retry_config = self._config.retry

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(retry_config.max_attempts),
            wait=wait_exponential(
                multiplier=retry_config.exponential_base,
                min=retry_config.min_wait,
                max=retry_config.max_wait,
            ),
            retry=retry_if_exception_type(retry_config.retryable_exceptions),
            reraise=True,
        ):
            with attempt:
                if self._session is None:
                    await self._create_session()

                return await self._session.request(method, url, **kwargs)

        # Should never reach here due to reraise=True
        raise RuntimeError("Retry logic failed unexpectedly")

    async def request(
        self,
        method: str,
        url: str,
        use_circuit_breaker: bool = True,
        use_rate_limit: bool = True,
        **kwargs: Any,
    ) -> aiohttp.ClientResponse:
        """Execute HTTP request with full protection.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            use_circuit_breaker: Enable circuit breaker protection
            use_rate_limit: Enable rate limiting
            **kwargs: Additional arguments passed to aiohttp

        Returns:
            aiohttp.ClientResponse object

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            aiohttp.ClientError: On HTTP errors
        """
        # Rate limiting
        if use_rate_limit:
            await self._rate_limiter.acquire()

        # Circuit breaker
        if use_circuit_breaker:
            return await self._circuit_breaker.call(
                self._request_with_retry,
                method,
                url,
                **kwargs,
            )
        else:
            return await self._request_with_retry(method, url, **kwargs)

    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> aiohttp.ClientResponse:
        """Execute GET request."""
        return await self.request("GET", url, params=params, **kwargs)

    async def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        data: Any = None,
        **kwargs: Any,
    ) -> aiohttp.ClientResponse:
        """Execute POST request."""
        return await self.request("POST", url, json=json, data=data, **kwargs)

    async def fetch_all(
        self,
        urls: list[str],
        method: str = "GET",
        **kwargs: Any,
    ) -> list[aiohttp.ClientResponse]:
        """Fetch multiple URLs concurrently.

        Args:
            urls: List of URLs to fetch
            method: HTTP method to use
            **kwargs: Additional arguments for each request

        Returns:
            List of ClientResponse objects

        Example:
            >>> urls = ["https://api1.com", "https://api2.com"]
            >>> responses = await client.fetch_all(urls)
        """
        tasks = [self.request(method, url, **kwargs) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "circuit_breaker_state": self._circuit_breaker.state.name,
            "rate_limiter_tokens": self._rate_limiter._tokens,
            "session_active": self._session is not None,
        }


# Convenience function for simple use cases
async def fetch(
    url: str,
    method: str = "GET",
    timeout: float = 30.0,
    **kwargs: Any,
) -> aiohttp.ClientResponse:
    """Simple async HTTP fetch with sensible defaults.

    Args:
        url: URL to fetch
        method: HTTP method
        timeout: Request timeout in seconds
        **kwargs: Additional arguments

    Returns:
        ClientResponse object

    Example:
        >>> response = await fetch("https://api.example.com")
        >>> data = await response.json()
    """
    config = HttpClientConfig(timeout=timeout)
    async with AsyncHttpClient(config) as client:
        return await client.request(method, url, **kwargs)
