# data/rate_limiter.py
"""Adaptive rate limiter with intelligent backoff.

This module provides rate limiting with:
- Per-source rate limits
- Adaptive backoff based on error rates
- Token bucket algorithm for burst handling
- Network-aware rate adjustment
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config.runtime_env import env_float, env_int
from utils.logger import get_logger

log = get_logger(__name__)

# Default configuration
_DEFAULT_RATE_LIMIT = float(env_float("TRADING_RATE_LIMIT", "2.0"))
_DEFAULT_BURST_SIZE = int(env_int("TRADING_BURST_SIZE", "5"))
_DEFAULT_BACKOFF_BASE = float(env_float("TRADING_BACKOFF_BASE", "2.0"))
_DEFAULT_BACKOFF_MAX = float(env_float("TRADING_BACKOFF_MAX", "60.0"))
_ERROR_RATE_WINDOW = float(env_float("TRADING_ERROR_RATE_WINDOW", "60.0"))
_ERROR_RATE_THRESHOLD = float(env_float("TRADING_ERROR_RATE_THRESHOLD", "0.5"))


@dataclass
class SourceStats:
    """Statistics for a single source."""
    request_count: int = 0
    error_count: int = 0
    last_request_time: float = 0.0
    last_error_time: float = 0.0
    current_delay: float = 0.0
    tokens: float = 0.0
    last_token_update: float = field(default_factory=time.time)
    error_timestamps: list[float] = field(default_factory=list)

    def record_request(self) -> None:
        self.request_count += 1
        self.last_request_time = time.time()
        self._update_tokens()

    def record_error(self) -> None:
        self.error_count += 1
        self.last_error_time = time.time()
        self.error_timestamps.append(time.time())
        self._prune_error_timestamps()

    def _update_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_token_update
        self.tokens = min(
            _DEFAULT_BURST_SIZE,
            self.tokens + elapsed * _DEFAULT_RATE_LIMIT
        )
        self.last_token_update = now

    def _prune_error_timestamps(self) -> None:
        """Remove error timestamps outside the window."""
        now = time.time()
        cutoff = now - _ERROR_RATE_WINDOW
        self.error_timestamps = [
            ts for ts in self.error_timestamps if ts > cutoff
        ]

    def error_rate(self) -> float:
        """Calculate recent error rate."""
        self._prune_error_timestamps()
        now = time.time()
        recent_requests = sum(
            1 for _ in range(self.request_count)
            if self.last_request_time - _ERROR_RATE_WINDOW < now
        )
        if recent_requests == 0:
            return 0.0
        return len(self.error_timestamps) / max(1, recent_requests)


class AdaptiveRateLimiter:
    """Rate limiter with adaptive backoff based on error rates."""

    def __init__(
        self,
        rate_limit: float = _DEFAULT_RATE_LIMIT,
        burst_size: int = _DEFAULT_BURST_SIZE,
        backoff_base: float = _DEFAULT_BACKOFF_BASE,
        backoff_max: float = _DEFAULT_BACKOFF_MAX,
    ):
        self._rate_limit = rate_limit
        self._burst_size = burst_size
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._sources: dict[str, SourceStats] = defaultdict(SourceStats)
        self._lock = threading.RLock()
        self._global_stats = SourceStats()
        self._network_mode: str = "unknown"
        self._last_network_check: float = 0.0

    def _get_stats(self, source: str) -> SourceStats:
        with self._lock:
            if source not in self._sources:
                self._sources[source] = SourceStats(
                    tokens=float(self._burst_size)
                )
            return self._sources[source]

    def acquire(self, source: str, interval: str = "1d") -> float:
        """
        Acquire permission to make a request.
        Returns the wait time in seconds (0.0 if ready immediately).
        """
        stats = self._get_stats(source)
        now = time.time()

        with self._lock:
            # Update tokens
            stats._update_tokens()

            # Calculate base wait time from rate limit
            time_since_last = now - stats.last_request_time if stats.last_request_time > 0 else float('inf')
            min_interval = self._get_min_interval(source, interval)

            # Calculate adaptive backoff based on error rate
            error_rate = stats.error_rate()
            backoff_multiplier = self._calculate_backoff_multiplier(error_rate)

            # Apply backoff to minimum interval
            adjusted_interval = min_interval * backoff_multiplier

            # Calculate wait time
            wait_time = max(0.0, adjusted_interval - time_since_last)

            # Also consider token bucket
            if stats.tokens < 1.0:
                token_wait = (1.0 - stats.tokens) / self._rate_limit
                wait_time = max(wait_time, token_wait)

            return wait_time

    def record_success(self, source: str) -> None:
        """Record a successful request."""
        stats = self._get_stats(source)
        with self._lock:
            stats.record_request()
            # Reduce delay on success
            if stats.current_delay > 0:
                stats.current_delay = max(
                    0.0,
                    stats.current_delay * 0.9
                )
            self._global_stats.record_request()

    def record_error(self, source: str, error: Exception | None = None) -> None:
        """Record a failed request."""
        stats = self._get_stats(source)
        with self._lock:
            stats.record_error()
            # Increase delay on error
            stats.current_delay = min(
                self._backoff_max,
                stats.current_delay * self._backoff_base if stats.current_delay > 0 else self._backoff_base
            )
            self._global_stats.record_error()

            error_type = type(error).__name__ if error else "unknown"
            log.debug(
                "Rate limiter: source=%s error=%s backoff=%.1fs error_rate=%.1f%%",
                source, error_type, stats.current_delay, stats.error_rate() * 100
            )

    def _get_min_interval(self, source: str, interval: str) -> float:
        """Get minimum interval based on source and interval type."""
        # Base intervals
        base_intervals = {
            "tencent": 0.5,
            "akshare": 1.0,
            "sina": 1.0,
            "yahoo": 2.0,
            "eastmoney": 1.0,
            "default": 1.0,
        }

        # Intraday intervals need more conservative rates
        intraday_multipliers = {
            "1m": 1.5,
            "2m": 1.4,
            "5m": 1.3,
            "15m": 1.2,
            "30m": 1.1,
            "60m": 1.0,
            "1h": 1.0,
            "1d": 1.0,
            "1wk": 1.0,
            "1mo": 1.0,
        }

        base = base_intervals.get(source.lower(), self._rate_limit)
        multiplier = intraday_multipliers.get(interval.lower(), 1.0)

        # Network-aware adjustment
        network_multiplier = self._get_network_multiplier()

        return base * multiplier * network_multiplier

    def _get_network_multiplier(self) -> float:
        """Get rate adjustment based on network conditions."""
        now = time.time()
        if now - self._last_network_check < 30.0:
            # Use cached value
            if self._network_mode == "china_direct":
                return 0.8  # Faster in China
            elif self._network_mode == "vpn":
                return 1.5  # Slower through VPN
            return 1.0

        # Detect network mode
        try:
            from core.network import get_network_env
            env = get_network_env()
            self._network_mode = "china_direct" if env.is_china_direct else "vpn"
        except Exception:
            self._network_mode = "unknown"

        self._last_network_check = now

        if self._network_mode == "china_direct":
            return 0.8
        elif self._network_mode == "vpn":
            return 1.5
        return 1.0

    def _calculate_backoff_multiplier(self, error_rate: float) -> float:
        """Calculate backoff multiplier based on error rate."""
        if error_rate <= 0.0:
            return 1.0
        if error_rate >= _ERROR_RATE_THRESHOLD:
            # Exponential backoff when error rate exceeds threshold
            excess = error_rate - _ERROR_RATE_THRESHOLD
            return min(
                self._backoff_max,
                self._backoff_base ** (1.0 + excess * 10)
            )
        # Linear scaling below threshold
        return 1.0 + (error_rate / _ERROR_RATE_THRESHOLD)

    def get_stats(self, source: str | None = None) -> dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            if source:
                stats = self._get_stats(source)
                return {
                    "source": source,
                    "request_count": stats.request_count,
                    "error_count": stats.error_count,
                    "error_rate": stats.error_rate(),
                    "current_delay": stats.current_delay,
                    "tokens": stats.tokens,
                    "last_request": stats.last_request_time,
                    "last_error": stats.last_error_time,
                }
            else:
                return {
                    "global": {
                        "request_count": self._global_stats.request_count,
                        "error_count": self._global_stats.error_count,
                        "error_rate": self._global_stats.error_rate(),
                    },
                    "sources": {
                        name: {
                            "request_count": s.request_count,
                            "error_count": s.error_count,
                            "error_rate": s.error_rate(),
                            "current_delay": s.current_delay,
                            "tokens": s.tokens,
                        }
                        for name, s in self._sources.items()
                    },
                    "config": {
                        "rate_limit": self._rate_limit,
                        "burst_size": self._burst_size,
                        "backoff_base": self._backoff_base,
                        "backoff_max": self._backoff_max,
                    },
                }

    def reset(self, source: str | None = None) -> None:
        """Reset rate limiter state."""
        with self._lock:
            if source:
                if source in self._sources:
                    stats = self._sources[source]
                    stats.current_delay = 0.0
                    stats.tokens = float(self._burst_size)
                    stats.error_timestamps.clear()
                    log.info("Rate limiter reset for source: %s", source)
            else:
                self._sources.clear()
                self._global_stats = SourceStats(tokens=float(self._burst_size))
                log.info("Rate limiter reset for all sources")


# Global instance
_rate_limiter: AdaptiveRateLimiter | None = None
_rate_limiter_lock = threading.Lock()


def get_rate_limiter() -> AdaptiveRateLimiter:
    """Get or create global rate limiter instance."""
    global _rate_limiter
    with _rate_limiter_lock:
        if _rate_limiter is None:
            _rate_limiter = AdaptiveRateLimiter()
        return _rate_limiter


def reset_rate_limiter(source: str | None = None) -> None:
    """Reset global rate limiter."""
    limiter = get_rate_limiter()
    limiter.reset(source)


def get_rate_limiter_stats(source: str | None = None) -> dict[str, Any]:
    """Get rate limiter statistics."""
    return get_rate_limiter().get_stats(source)
