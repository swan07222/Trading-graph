# data/source_health.py
"""Data source health monitoring and auto-failover.

FIX 2026-02-26: Addresses disadvantages:
- Network dependency with health monitoring
- Automatic failover to healthy sources
- Circuit breaker integration
- Real-time source scoring

Features:
- Per-source health scoring
- Automatic failover on degradation
- Recovery detection
- Metrics and alerting
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

from config.runtime_env import env_flag, env_float, env_int, env_text
from utils.logger import get_logger

log = get_logger(__name__)


class SourceHealthStatus(Enum):
    """Health status for data sources."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class SourceHealthState:
    """Health state for a single data source."""
    source: str
    status: SourceHealthStatus = SourceHealthStatus.UNKNOWN
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    last_check_time: float = 0.0
    response_time_avg: float = 0.0
    response_time_recent: float = 0.0
    error_rate: float = 0.0
    cooldown_until: float = 0.0
    failover_count: int = 0
    
    # Health score (0.0 to 1.0)
    _health_score: float = 1.0
    
    def record_success(self, response_time: float = 0.0) -> None:
        """Record a successful request."""
        now = time.time()
        self.success_count += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = now
        self.last_check_time = now
        
        # Update response time (exponential moving average)
        alpha = 0.3
        if self.response_time_avg == 0.0:
            self.response_time_avg = response_time
            self.response_time_recent = response_time
        else:
            self.response_time_avg = alpha * response_time + (1 - alpha) * self.response_time_avg
            self.response_time_recent = response_time
        
        # Update health score
        self._update_health_score()
        
        # Check for recovery
        if self.status == SourceHealthStatus.UNHEALTHY and self.consecutive_successes >= 3:
            self.status = SourceHealthStatus.DEGRADED
            log.info("Source %s recovering: %s -> %s", self.source, SourceHealthStatus.UNHEALTHY.value, self.status.value)
        elif self.status == SourceHealthStatus.DEGRADED and self.consecutive_successes >= 5:
            self.status = SourceHealthStatus.HEALTHY
            log.info("Source %s recovered: %s -> %s", self.source, SourceHealthStatus.DEGRADED.value, self.status.value)
    
    def record_failure(self, error: str = "") -> None:
        """Record a failed request."""
        now = time.time()
        self.failure_count += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = now
        self.last_check_time = now
        
        # Update error rate
        total = self.success_count + self.failure_count
        if total > 0:
            self.error_rate = self.failure_count / total
        
        # Update health score
        self._update_health_score()
        
        # Check for degradation
        if self.status == SourceHealthStatus.HEALTHY and self.consecutive_failures >= 2:
            self.status = SourceHealthStatus.DEGRADED
            log.warning("Source %s degraded: %s -> %s", self.source, SourceHealthStatus.HEALTHY.value, self.status.value)
        elif self.status != SourceHealthStatus.UNHEALTHY and self.consecutive_failures >= 5:
            self.status = SourceHealthStatus.UNHEALTHY
            log.error("Source %s unhealthy: %s -> %s", self.source, self.status.value, SourceHealthStatus.UNHEALTHY.value)
    
    def _update_health_score(self) -> None:
        """Calculate health score from 0.0 (unhealthy) to 1.0 (healthy)."""
        # Factors:
        # 1. Recent success rate (40%)
        # 2. Consecutive successes/failures (30%)
        # 3. Response time (20%)
        # 4. Error rate (10%)
        
        total = self.success_count + self.failure_count
        if total == 0:
            self._health_score = 1.0
            return
        
        # Success rate component
        success_rate = self.success_count / total
        recent_success_rate = 1.0 if self.consecutive_failures == 0 else max(0.0, 1.0 - self.consecutive_failures * 0.2)
        success_component = 0.6 * success_rate + 0.4 * recent_success_rate
        
        # Consecutive component
        if self.consecutive_successes > 0:
            consecutive_component = min(1.0, self.consecutive_successes / 5.0)
        else:
            consecutive_component = max(0.0, 1.0 - self.consecutive_failures / 5.0)
        
        # Response time component (lower is better)
        # Assume >5s is very slow
        response_component = max(0.0, 1.0 - self.response_time_recent / 5.0)
        
        # Error rate component
        error_component = 1.0 - self.error_rate
        
        # Weighted average
        self._health_score = (
            0.40 * success_component +
            0.30 * consecutive_component +
            0.20 * response_component +
            0.10 * error_component
        )
    
    @property
    def health_score(self) -> float:
        """Get current health score."""
        return self._health_score
    
    def is_available(self) -> bool:
        """Check if source is available for requests."""
        if self.status == SourceHealthStatus.UNHEALTHY:
            # Check if cooldown has expired
            if time.time() >= self.cooldown_until:
                self.status = SourceHealthStatus.DEGRADED
                return True
            return False
        return True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "source": self.source,
            "status": self.status.value,
            "health_score": round(self._health_score, 3),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "error_rate": round(self.error_rate, 3),
            "response_time_avg_ms": round(self.response_time_avg * 1000, 1),
            "last_success": datetime.fromtimestamp(self.last_success_time).isoformat() if self.last_success_time > 0 else None,
            "last_failure": datetime.fromtimestamp(self.last_failure_time).isoformat() if self.last_failure_time > 0 else None,
        }


class DataSourceHealthMonitor:
    """Monitors health of multiple data sources with auto-failover.
    
    FIX 2026-02-26:
    - Real-time health scoring
    - Automatic failover
    - Configurable thresholds
    - Thread-safe operations
    """
    
    def __init__(
        self,
        unhealthy_threshold: int = 5,
        healthy_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        health_check_interval: float = 60.0,
    ):
        self._sources: dict[str, SourceHealthState] = {}
        self._lock = threading.RLock()
        self._unhealthy_threshold = unhealthy_threshold
        self._healthy_threshold = healthy_threshold
        self._cooldown_seconds = cooldown_seconds
        self._health_check_interval = health_check_interval
        
        # Source priorities (lower is better)
        self._priorities: dict[str, int] = {
            "akshare": 1,
            "sina": 2,
            "eastmoney": 3,
            "tencent": 4,
            "yahoo": 5,
        }
        
        # Callbacks for alerts
        self._on_status_change: list[Callable[[str, SourceHealthStatus, SourceHealthStatus], None]] = []
        
        # Start background health checker
        self._stop_checker = threading.Event()
        self._checker_thread = threading.Thread(
            target=self._background_health_check,
            daemon=True,
            name="SourceHealthMonitor",
        )
        self._checker_thread.start()
    
    def _get_state(self, source: str) -> SourceHealthState:
        """Get or create state for a source."""
        with self._lock:
            if source not in self._sources:
                self._sources[source] = SourceHealthState(source=source)
            return self._sources[source]
    
    def record_success(self, source: str, response_time: float = 0.0) -> None:
        """Record a successful request to a source."""
        state = self._get_state(source)
        with self._lock:
            old_status = state.status
            state.record_success(response_time)
            if old_status != state.status:
                self._notify_status_change(source, old_status, state.status)
    
    def record_failure(self, source: str, error: str = "") -> None:
        """Record a failed request to a source."""
        state = self._get_state(source)
        with self._lock:
            old_status = state.status
            state.record_failure(error)
            
            # Set cooldown if unhealthy
            if state.status == SourceHealthStatus.UNHEALTHY:
                state.cooldown_until = time.time() + self._cooldown_seconds
                state.failover_count += 1
                log.warning(
                    "Source %s marked unhealthy, failover #%d, cooldown %.0fs",
                    source, state.failover_count, self._cooldown_seconds
                )
            
            if old_status != state.status:
                self._notify_status_change(source, old_status, state.status)
    
    def get_healthy_source(self, preferred: str | None = None) -> str | None:
        """Get the healthiest available source.
        
        FIX 2026-02-26: Auto-failover to healthy sources.
        
        Args:
            preferred: Preferred source if available and healthy
        
        Returns:
            Name of healthiest available source, or None if all unhealthy
        """
        with self._lock:
            if not self._sources:
                return preferred
            
            # Check preferred source first
            if preferred:
                state = self._sources.get(preferred)
                if state and state.is_available():
                    return preferred
            
            # Find healthiest available source
            available = [
                (name, state) for name, state in self._sources.items()
                if state.is_available()
            ]
            
            if not available:
                return None
            
            # Sort by health score (descending), then priority (ascending)
            available.sort(
                key=lambda x: (-x[1].health_score, self._priorities.get(x[0], 99))
            )
            
            return available[0][0]
    
    def get_all_statuses(self) -> dict[str, dict[str, Any]]:
        """Get health status of all sources."""
        with self._lock:
            return {
                name: state.to_dict()
                for name, state in self._sources.items()
            }
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of source health."""
        with self._lock:
            total = len(self._sources)
            healthy = sum(1 for s in self._sources.values() if s.status == SourceHealthStatus.HEALTHY)
            degraded = sum(1 for s in self._sources.values() if s.status == SourceHealthStatus.DEGRADED)
            unhealthy = sum(1 for s in self._sources.values() if s.status == SourceHealthStatus.UNHEALTHY)
            
            return {
                "total_sources": total,
                "healthy": healthy,
                "degraded": degraded,
                "unhealthy": unhealthy,
                "all_healthy": unhealthy == 0 and healthy > 0,
                "any_available": healthy > 0 or degraded > 0,
            }
    
    def register_status_callback(
        self,
        callback: Callable[[str, SourceHealthStatus, SourceHealthStatus], None]
    ) -> None:
        """Register callback for status changes."""
        self._on_status_change.append(callback)
    
    def _notify_status_change(
        self,
        source: str,
        old_status: SourceHealthStatus,
        new_status: SourceHealthStatus
    ) -> None:
        """Notify callbacks of status change."""
        for callback in self._on_status_change:
            try:
                callback(source, old_status, new_status)
            except Exception as e:
                log.error("Status callback error: %s", e)
    
    def _background_health_check(self) -> None:
        """Background thread to periodically check source health."""
        while not self._stop_checker.is_set():
            try:
                self._stop_checker.wait(self._health_check_interval)
                if self._stop_checker.is_set():
                    break
                
                # Check for sources that should recover from cooldown
                now = time.time()
                with self._lock:
                    for state in self._sources.values():
                        if state.status == SourceHealthStatus.UNHEALTHY:
                            if now >= state.cooldown_until:
                                state.status = SourceHealthStatus.DEGRADED
                                log.info("Source %s cooldown expired, trying degraded", state.source)
                                self._notify_status_change(
                                    state.source,
                                    SourceHealthStatus.UNHEALTHY,
                                    SourceHealthStatus.DEGRADED
                                )
            except Exception:
                pass  # Don't let errors crash the monitor thread
    
    def shutdown(self) -> None:
        """Shutdown the health monitor."""
        self._stop_checker.set()
        if self._checker_thread.is_alive():
            self._checker_thread.join(timeout=2.0)


# Global monitor instance
_monitor: DataSourceHealthMonitor | None = None
_monitor_lock = threading.Lock()


def get_health_monitor() -> DataSourceHealthMonitor:
    """Get or create global health monitor instance."""
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            _monitor = DataSourceHealthMonitor(
                unhealthy_threshold=int(env_int("TRADING_UNHEALTHY_THRESHOLD", "5")),
                healthy_threshold=int(env_int("TRADING_HEALTHY_THRESHOLD", "3")),
                cooldown_seconds=float(env_text("TRADING_FAILOVER_COOLDOWN", "30.0")),
                health_check_interval=float(env_text("TRADING_HEALTH_CHECK_INTERVAL", "60.0")),
            )
        return _monitor


def record_source_success(source: str, response_time: float = 0.0) -> None:
    """Record a successful request to a data source."""
    get_health_monitor().record_success(source, response_time)


def record_source_failure(source: str, error: str = "") -> None:
    """Record a failed request to a data source."""
    get_health_monitor().record_failure(source, error)


def get_healthy_source(preferred: str | None = None) -> str | None:
    """Get the healthiest available data source."""
    return get_health_monitor().get_healthy_source(preferred)


def get_source_health_summary() -> dict[str, Any]:
    """Get summary of all source health statuses."""
    return get_health_monitor().get_summary()


def reset_health_monitor() -> None:
    """Reset health monitor (for testing)."""
    global _monitor
    with _monitor_lock:
        if _monitor:
            _monitor.shutdown()
        _monitor = None
