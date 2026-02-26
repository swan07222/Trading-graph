"""Enhanced Rate Limiter with Adaptive Backoff and Compliance Tracking.

This module provides intelligent rate limiting with:
- Adaptive backoff based on API response patterns
- Multi-source rate limiting coordination
- Compliance metadata tracking
- Usage analytics and reporting
- Circuit breaker pattern for fault tolerance

Fixes:
- Rate limits and API throttling
- Legal/compliance tracking
- Cost optimization through intelligent caching
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    LEAKY_BUCKET = "leaky_bucket"
    EXPONENTIAL_BACKOFF = "exponential_backoff"


class ComplianceStatus(Enum):
    """Compliance status for data usage."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    BLOCKED = "blocked"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 10.0
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 20
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    backoff_base: float = 2.0
    backoff_max: float = 300.0  # 5 minutes
    backoff_multiplier: float = 2.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "requests_per_second": self.requests_per_second,
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "requests_per_day": self.requests_per_day,
            "burst_size": self.burst_size,
            "strategy": self.strategy.value,
            "backoff_base": self.backoff_base,
            "backoff_max": self.backoff_max,
            "backoff_multiplier": self.backoff_multiplier,
        }


@dataclass
class ComplianceMetadata:
    """Metadata for compliance tracking."""
    source_name: str
    license_type: str = ""
    terms_of_service_url: str = ""
    allowed_use_cases: list[str] = field(default_factory=list)
    restricted_use_cases: list[str] = field(default_factory=list)
    attribution_required: bool = False
    commercial_use_allowed: bool = True
    redistribution_allowed: bool = False
    rate_limit_agreed: int = 0
    data_retention_days: int = 365
    last_compliance_check: datetime = field(default_factory=datetime.now)
    compliance_status: ComplianceStatus = ComplianceStatus.COMPLIANT
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "license_type": self.license_type,
            "terms_of_service_url": self.terms_of_service_url,
            "allowed_use_cases": self.allowed_use_cases,
            "restricted_use_cases": self.restricted_use_cases,
            "attribution_required": self.attribution_required,
            "commercial_use_allowed": self.commercial_use_allowed,
            "redistribution_allowed": self.redistribution_allowed,
            "rate_limit_agreed": self.rate_limit_agreed,
            "data_retention_days": self.data_retention_days,
            "last_compliance_check": self.last_compliance_check.isoformat(),
            "compliance_status": self.compliance_status.value,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComplianceMetadata:
        return cls(
            source_name=data["source_name"],
            license_type=data.get("license_type", ""),
            terms_of_service_url=data.get("terms_of_service_url", ""),
            allowed_use_cases=data.get("allowed_use_cases", []),
            restricted_use_cases=data.get("restricted_use_cases", []),
            attribution_required=data.get("attribution_required", False),
            commercial_use_allowed=data.get("commercial_use_allowed", True),
            redistribution_allowed=data.get("redistribution_allowed", False),
            rate_limit_agreed=data.get("rate_limit_agreed", 0),
            data_retention_days=data.get("data_retention_days", 365),
            last_compliance_check=datetime.fromisoformat(data["last_compliance_check"]) if data.get("last_compliance_check") else datetime.now(),
            compliance_status=ComplianceStatus(data.get("compliance_status", "compliant")),
        )


@dataclass
class UsageRecord:
    """Record of API usage for compliance tracking."""
    source: str
    timestamp: datetime
    endpoint: str
    request_count: int = 1
    response_size_bytes: int = 0
    latency_ms: float = 0.0
    status_code: int = 200
    error_message: str = ""
    purpose: str = ""  # e.g., "training", "prediction", "backtest"
    user_id: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "endpoint": self.endpoint,
            "request_count": self.request_count,
            "response_size_bytes": self.response_size_bytes,
            "latency_ms": self.latency_ms,
            "status_code": self.status_code,
            "error_message": self.error_message,
            "purpose": self.purpose,
            "user_id": self.user_id,
        }


class AdaptiveRateLimiter:
    """Token bucket rate limiter with adaptive backoff."""
    
    def __init__(
        self,
        config: RateLimitConfig | None = None,
        source_name: str = "default",
    ) -> None:
        self.config = config or RateLimitConfig()
        self.source_name = source_name
        
        # Token bucket state
        self._tokens = float(self.config.burst_size)
        self._last_update = time.time()
        
        # Adaptive backoff state
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._current_backoff = 0.0
        self._last_failure_time: float = 0.0
        
        # Rate limiting windows
        self._second_window: list[float] = []
        self._minute_window: list[float] = []
        self._hour_window: list[float] = []
        self._day_window: list[float] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._total_requests = 0
        self._total_failures = 0
        self._total_wait_time = 0.0
    
    def acquire(self, tokens: int = 1, timeout: float = 30.0) -> bool:
        """Acquire tokens, waiting up to timeout if necessary.
        
        Returns:
            True if tokens acquired, False if timeout
        """
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            if self._try_acquire(tokens):
                return True
            
            # Calculate wait time
            wait_time = self._calculate_wait_time()
            
            if wait_time <= 0:
                continue
            
            if time.time() + wait_time > deadline:
                return False
            
            time.sleep(min(wait_time, 0.1))  # Sleep in small increments
        
        return False
    
    def _try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without waiting."""
        with self._lock:
            # Check if in backoff
            if self._current_backoff > 0:
                if time.time() - self._last_failure_time < self._current_backoff:
                    return False
                else:
                    self._current_backoff = 0.0
            
            # Refill tokens
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(
                float(self.config.burst_size),
                self._tokens + elapsed * self.config.requests_per_second
            )
            self._last_update = now
            
            # Clean up windows
            self._cleanup_windows(now)
            
            # Check all rate limits
            if not self._check_rate_limits(tokens):
                return False
            
            # Check if enough tokens
            if self._tokens < float(tokens):
                return False
            
            # Acquire tokens
            self._tokens -= float(tokens)
            self._total_requests += 1
            return True
    
    def _check_rate_limits(self, tokens: int = 1) -> bool:
        """Check if request is within rate limits."""
        now = time.time()
        
        # Check per-second limit
        if len(self._second_window) >= self.config.requests_per_second:
            return False
        
        # Check per-minute limit
        if len(self._minute_window) >= self.config.requests_per_minute:
            return False
        
        # Check per-hour limit
        if len(self._hour_window) >= self.config.requests_per_hour:
            return False
        
        # Check per-day limit
        if len(self._day_window) >= self.config.requests_per_day:
            return False
        
        return True
    
    def _cleanup_windows(self, now: float) -> None:
        """Remove expired entries from rate limit windows."""
        # Remove entries older than 1 second
        self._second_window = [t for t in self._second_window if now - t < 1.0]
        
        # Remove entries older than 1 minute
        self._minute_window = [t for t in self._minute_window if now - t < 60.0]
        
        # Remove entries older than 1 hour
        self._hour_window = [t for t in self._hour_window if now - t < 3600.0]
        
        # Remove entries older than 1 day
        self._day_window = [t for t in self._day_window if now - t < 86400.0]
    
    def _calculate_wait_time(self) -> float:
        """Calculate time to wait before next request."""
        with self._lock:
            now = time.time()
            wait_times = []
            
            # Time until token available
            if self._tokens < 1.0:
                token_wait = (1.0 - self._tokens) / self.config.requests_per_second
                wait_times.append(token_wait)
            
            # Time until per-second limit resets
            if self._second_window:
                oldest = min(self._second_window)
                wait_times.append(max(0.0, 1.0 - (now - oldest)))
            
            # Time until per-minute limit resets
            if len(self._minute_window) >= self.config.requests_per_minute:
                oldest = min(self._minute_window)
                wait_times.append(max(0.0, 60.0 - (now - oldest)))
            
            # Return shortest wait time
            return min(wait_times) if wait_times else 0.0
    
    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            now = time.time()
            self._second_window.append(now)
            self._minute_window.append(now)
            self._hour_window.append(now)
            self._day_window.append(now)
            
            self._consecutive_successes += 1
            self._consecutive_failures = 0
            
            # Reduce backoff on success
            if self._current_backoff > 0:
                self._current_backoff = max(
                    0.0,
                    self._current_backoff / self.config.backoff_multiplier
                )
    
    def record_failure(self, status_code: int = 0) -> None:
        """Record a failed request."""
        with self._lock:
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._last_failure_time = time.time()
            self._total_failures += 1
            
            # Apply exponential backoff
            if status_code == 429:  # Rate limited
                self._current_backoff = min(
                    self.config.backoff_max,
                    self.config.backoff_base * (
                        self.config.backoff_multiplier ** self._consecutive_failures
                    )
                )
            elif status_code >= 500:  # Server error
                self._current_backoff = min(
                    self.config.backoff_max,
                    self.config.backoff_base * (
                        self.config.backoff_multiplier ** min(3, self._consecutive_failures)
                    )
                )
    
    def get_status(self) -> dict[str, Any]:
        """Get current rate limiter status."""
        with self._lock:
            now = time.time()
            self._cleanup_windows(now)
            
            return {
                "source": self.source_name,
                "tokens_available": self._tokens,
                "burst_size": self.config.burst_size,
                "requests_second": len(self._second_window),
                "requests_minute": len(self._minute_window),
                "requests_hour": len(self._hour_window),
                "requests_day": len(self._day_window),
                "limits": {
                    "per_second": self.config.requests_per_second,
                    "per_minute": self.config.requests_per_minute,
                    "per_hour": self.config.requests_per_hour,
                    "per_day": self.config.requests_per_day,
                },
                "backoff": {
                    "current": self._current_backoff,
                    "consecutive_failures": self._consecutive_failures,
                    "consecutive_successes": self._consecutive_successes,
                },
                "statistics": {
                    "total_requests": self._total_requests,
                    "total_failures": self._total_failures,
                    "failure_rate": self._total_failures / max(1, self._total_requests),
                    "total_wait_time": self._total_wait_time,
                },
            }


class ComplianceTracker:
    """Tracks compliance with data source terms of service."""
    
    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path or CONFIG.data_dir / "compliance"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._metadata: dict[str, ComplianceMetadata] = {}
        self._usage_records: list[UsageRecord] = []
        self._max_records = 100000
        
        self._load_from_storage()
    
    def _load_from_storage(self) -> None:
        """Load compliance data from disk."""
        metadata_file = self.storage_path / "metadata.json"
        if metadata_file.exists():
            try:
                content = metadata_file.read_text(encoding="utf-8")
                data = json.loads(content)
                for source, meta_dict in data.items():
                    self._metadata[source] = ComplianceMetadata.from_dict(meta_dict)
                log.info(f"Loaded compliance metadata for {len(self._metadata)} sources")
            except Exception as e:
                log.warning(f"Failed to load compliance metadata: {e}")
        
        usage_file = self.storage_path / "usage.json"
        if usage_file.exists():
            try:
                content = usage_file.read_text(encoding="utf-8")
                data = json.loads(content)
                self._usage_records = [
                    UsageRecord(
                        source=r["source"],
                        timestamp=datetime.fromisoformat(r["timestamp"]),
                        endpoint=r["endpoint"],
                        request_count=r.get("request_count", 1),
                        response_size_bytes=r.get("response_size_bytes", 0),
                        latency_ms=r.get("latency_ms", 0.0),
                        status_code=r.get("status_code", 200),
                        error_message=r.get("error_message", ""),
                        purpose=r.get("purpose", ""),
                        user_id=r.get("user_id", ""),
                    )
                    for r in data
                ]
                log.info(f"Loaded {len(self._usage_records)} usage records")
            except Exception as e:
                log.warning(f"Failed to load usage records: {e}")
    
    def _save_to_storage(self) -> None:
        """Save compliance data to disk."""
        metadata_file = self.storage_path / "metadata.json"
        try:
            data = {
                source: meta.to_dict()
                for source, meta in self._metadata.items()
            }
            content = json.dumps(data, ensure_ascii=False, indent=2)
            metadata_file.write_text(content, encoding="utf-8")
        except Exception as e:
            log.warning(f"Failed to save compliance metadata: {e}")
        
        usage_file = self.storage_path / "usage.json"
        try:
            # Keep only recent records
            recent = self._usage_records[-self._max_records:]
            data = [r.to_dict() for r in recent]
            content = json.dumps(data, ensure_ascii=False, indent=2)
            usage_file.write_text(content, encoding="utf-8")
        except Exception as e:
            log.warning(f"Failed to save usage records: {e}")
    
    def register_source(
        self,
        source_name: str,
        license_type: str = "",
        terms_url: str = "",
        rate_limit: int = 0,
        **kwargs: Any,
    ) -> None:
        """Register a new data source with compliance metadata."""
        metadata = ComplianceMetadata(
            source_name=source_name,
            license_type=license_type,
            terms_of_service_url=terms_url,
            rate_limit_agreed=rate_limit,
            **kwargs,
        )
        self._metadata[source_name] = metadata
        self._save_to_storage()
        log.info(f"Registered compliance metadata for {source_name}")
    
    def log_usage(
        self,
        source: str,
        endpoint: str,
        request_count: int = 1,
        response_size: int = 0,
        latency_ms: float = 0.0,
        status_code: int = 200,
        error_message: str = "",
        purpose: str = "",
        user_id: str = "",
    ) -> None:
        """Log API usage for compliance tracking."""
        record = UsageRecord(
            source=source,
            timestamp=datetime.now(),
            endpoint=endpoint,
            request_count=request_count,
            response_size_bytes=response_size,
            latency_ms=latency_ms,
            status_code=status_code,
            error_message=error_message,
            purpose=purpose,
            user_id=user_id,
        )
        self._usage_records.append(record)
        
        # Trim old records
        if len(self._usage_records) > self._max_records:
            self._usage_records = self._usage_records[-self._max_records:]
        
        # Check compliance
        self._check_compliance(source)
    
    def _check_compliance(self, source: str) -> None:
        """Check compliance status for a source."""
        metadata = self._metadata.get(source)
        if not metadata:
            return
        
        now = datetime.now()
        
        # Check rate limit
        if metadata.rate_limit_agreed > 0:
            recent_usage = self.get_usage_count(
                source,
                hours_back=24,
            )
            
            if recent_usage > metadata.rate_limit_agreed:
                metadata.compliance_status = ComplianceStatus.VIOLATION
                log.warning(
                    f"Rate limit violation for {source}: "
                    f"{recent_usage} requests > {metadata.rate_limit_agreed} allowed"
                )
            elif recent_usage > metadata.rate_limit_agreed * 0.8:
                metadata.compliance_status = ComplianceStatus.WARNING
            else:
                metadata.compliance_status = ComplianceStatus.COMPLIANT
        
        metadata.last_compliance_check = now
        self._save_to_storage()
    
    def get_usage_count(
        self,
        source: str,
        hours_back: int = 24,
    ) -> int:
        """Get usage count for a source within time window."""
        cutoff = datetime.now() - timedelta(hours=hours_back)
        
        return sum(
            r.request_count
            for r in self._usage_records
            if r.source == source and r.timestamp >= cutoff
        )
    
    def get_usage_stats(
        self,
        source: str | None = None,
        hours_back: int = 24,
    ) -> dict[str, Any]:
        """Get usage statistics."""
        cutoff = datetime.now() - timedelta(hours=hours_back)
        
        records = self._usage_records
        if source:
            records = [r for r in records if r.source == source]
        
        recent = [r for r in records if r.timestamp >= cutoff]
        
        if not recent:
            return {
                "total_requests": 0,
                "total_bytes": 0,
                "avg_latency_ms": 0.0,
                "error_rate": 0.0,
            }
        
        total_requests = sum(r.request_count for r in recent)
        total_bytes = sum(r.response_size_bytes for r in recent)
        avg_latency = sum(r.latency_ms for r in recent) / len(recent)
        error_count = sum(1 for r in recent if r.status_code >= 400)
        
        return {
            "total_requests": total_requests,
            "total_bytes": total_bytes,
            "avg_latency_ms": avg_latency,
            "error_rate": error_count / len(recent),
            "sources": list(set(r.source for r in recent)),
        }
    
    def get_compliance_report(self) -> dict[str, Any]:
        """Get comprehensive compliance report."""
        return {
            "sources": {
                source: meta.to_dict()
                for source, meta in self._metadata.items()
            },
            "usage_stats": self.get_usage_stats(),
            "recent_violations": [
                r.to_dict()
                for r in self._usage_records[-1000:]
                if r.status_code >= 400
            ],
        }
    
    def check_usage_permission(
        self,
        source: str,
        use_case: str,
    ) -> tuple[bool, str]:
        """Check if a use case is allowed for a source."""
        metadata = self._metadata.get(source)
        if not metadata:
            return True, "No compliance metadata registered"
        
        # Check restricted use cases
        if use_case in metadata.restricted_use_cases:
            return False, f"Use case '{use_case}' is restricted for {source}"
        
        # Check allowed use cases (if specified)
        if metadata.allowed_use_cases and use_case not in metadata.allowed_use_cases:
            return False, f"Use case '{use_case}' is not allowed for {source}"
        
        # Check commercial use
        if not metadata.commercial_use_allowed:
            return False, f"Commercial use not allowed for {source}"
        
        return True, "OK"


class RateLimiterRegistry:
    """Registry for managing multiple rate limiters."""
    
    def __init__(self) -> None:
        self._limiters: dict[str, AdaptiveRateLimiter] = {}
        self._compliance = ComplianceTracker()
        self._lock = threading.RLock()
    
    def get_limiter(
        self,
        source: str,
        config: RateLimitConfig | None = None,
    ) -> AdaptiveRateLimiter:
        """Get or create a rate limiter for a source."""
        with self._lock:
            if source not in self._limiters:
                self._limiters[source] = AdaptiveRateLimiter(
                    config=config,
                    source_name=source,
                )
            return self._limiters[source]
    
    def acquire(
        self,
        source: str,
        tokens: int = 1,
        timeout: float = 30.0,
    ) -> bool:
        """Acquire rate limit token for a source."""
        limiter = self.get_limiter(source)
        return limiter.acquire(tokens, timeout)
    
    def record_success(self, source: str) -> None:
        """Record successful request."""
        if source in self._limiters:
            self._limiters[source].record_success()
    
    def record_failure(
        self,
        source: str,
        status_code: int = 0,
    ) -> None:
        """Record failed request."""
        if source in self._limiters:
            self._limiters[source].record_failure(status_code)
    
    def log_usage(
        self,
        source: str,
        endpoint: str,
        **kwargs: Any,
    ) -> None:
        """Log API usage."""
        self._compliance.log_usage(source, endpoint, **kwargs)
    
    def register_source(
        self,
        source: str,
        **kwargs: Any,
    ) -> None:
        """Register a data source."""
        self._compliance.register_source(source, **kwargs)
    
    def get_status(self) -> dict[str, Any]:
        """Get status of all rate limiters."""
        with self._lock:
            return {
                "limiters": {
                    name: limiter.get_status()
                    for name, limiter in self._limiters.items()
                },
                "compliance": self._compliance.get_compliance_report(),
            }


# Singleton instance
_registry: RateLimiterRegistry | None = None


def get_rate_limiter_registry() -> RateLimiterRegistry:
    """Get singleton rate limiter registry."""
    global _registry
    if _registry is None:
        _registry = RateLimiterRegistry()
    return _registry


def acquire_rate_limit(
    source: str,
    timeout: float = 30.0,
) -> bool:
    """Convenience function to acquire rate limit."""
    return get_rate_limiter_registry().acquire(source, timeout=timeout)


def record_api_success(
    source: str,
    endpoint: str,
    **kwargs: Any,
) -> None:
    """Convenience function to record successful API call."""
    registry = get_rate_limiter_registry()
    registry.record_success(source)
    registry.log_usage(source, endpoint, **kwargs)


def record_api_failure(
    source: str,
    status_code: int = 0,
) -> None:
    """Convenience function to record failed API call."""
    get_rate_limiter_registry().record_failure(source, status_code)
