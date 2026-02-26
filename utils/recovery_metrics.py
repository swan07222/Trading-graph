"""Recovery Metrics and Monitoring.

Provides metrics collection and monitoring for recovery operations
across the trading system.

Features:
- Recovery operation metrics
- Success/failure tracking
- Performance monitoring
- Health status dashboard
- Alerting support

Usage:
    from utils.recovery_metrics import RecoveryMetrics
    
    metrics = RecoveryMetrics()
    
    # Record operation
    metrics.record_operation(
        operation="fetch_data",
        success=True,
        duration_seconds=1.5,
        attempts=2,
    )
    
    # Get metrics
    summary = metrics.get_summary()
    health = metrics.get_health()
    
    # Export for monitoring
    export = metrics.export_metrics()
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from config.settings import CONFIG
from utils.atomic_io import atomic_write_json, read_json
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class OperationRecord:
    """Record of a single operation."""
    operation: str
    timestamp: str
    success: bool
    duration_seconds: float
    attempts: int = 1
    error_type: str | None = None
    error_message: str | None = None
    recovery_strategy: str | None = None
    fallback_used: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "timestamp": self.timestamp,
            "success": self.success,
            "duration_seconds": round(self.duration_seconds, 3),
            "attempts": self.attempts,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "recovery_strategy": self.recovery_strategy,
            "fallback_used": self.fallback_used,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OperationRecord":
        return cls(
            operation=data.get("operation", ""),
            timestamp=data.get("timestamp", ""),
            success=data.get("success", False),
            duration_seconds=data.get("duration_seconds", 0.0),
            attempts=data.get("attempts", 1),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            recovery_strategy=data.get("recovery_strategy"),
            fallback_used=data.get("fallback_used", False),
        )


@dataclass
class OperationMetrics:
    """Aggregated metrics for an operation."""
    operation: str
    total_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_attempts: int = 0
    total_duration_seconds: float = 0.0
    fallback_count: int = 0
    
    # Recent history for trend analysis
    recent_results: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 1.0
        return self.success_count / self.total_count
    
    @property
    def avg_attempts(self) -> float:
        if self.total_count == 0:
            return 1.0
        return self.total_attempts / self.total_count
    
    @property
    def avg_duration_seconds(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.total_duration_seconds / self.total_count
    
    @property
    def fallback_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.fallback_count / self.total_count
    
    @property
    def recent_success_rate(self) -> float:
        if not self.recent_results:
            return 1.0
        recent_successes = sum(1 for r in self.recent_results if r)
        return recent_successes / len(self.recent_results)
    
    def record(self, success: bool, attempts: int, duration: float, fallback: bool = False) -> None:
        """Record a new operation result."""
        self.total_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.total_attempts += attempts
        self.total_duration_seconds += duration
        if fallback:
            self.fallback_count += 1
        
        self.recent_results.append(success)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "total_count": self.total_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 4),
            "avg_attempts": round(self.avg_attempts, 2),
            "avg_duration_seconds": round(self.avg_duration_seconds, 3),
            "fallback_rate": round(self.fallback_rate, 4),
            "recent_success_rate": round(self.recent_success_rate, 4),
        }


@dataclass
class RecoveryHealth:
    """Overall recovery health status."""
    status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    success_rate_24h: float = 1.0
    avg_recovery_time_seconds: float = 0.0
    consecutive_failures: int = 0
    total_operations_24h: int = 0
    alerts: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "success_rate_24h": round(self.success_rate_24h, 4),
            "avg_recovery_time_seconds": round(self.avg_recovery_time_seconds, 2),
            "consecutive_failures": self.consecutive_failures,
            "total_operations_24h": self.total_operations_24h,
            "alerts": self.alerts,
            "recommendations": self.recommendations,
        }


class RecoveryMetrics:
    """Recovery metrics collector and monitor.
    
    Usage:
        metrics = RecoveryMetrics()
        
        # Record operations
        metrics.record_operation("fetch_data", success=True, duration=1.5)
        metrics.record_operation("train_model", success=False, duration=10.0, 
                                attempts=3, error="timeout")
        
        # Get metrics
        summary = metrics.get_summary()
        health = metrics.get_health()
    """
    
    def __init__(
        self,
        metrics_dir: Path | str | None = None,
        retention_hours: int = 24,
        max_records: int = 10000,
    ) -> None:
        """Initialize recovery metrics.
        
        Args:
            metrics_dir: Directory for metrics files
            retention_hours: Hours to retain detailed records
            max_records: Maximum records to keep in memory
        """
        self.metrics_dir = Path(metrics_dir) if metrics_dir else CONFIG.log_dir / "recovery"
        self.retention_hours = retention_hours
        self.max_records = max_records
        
        self._lock = threading.Lock()
        self._operations: dict[str, OperationMetrics] = {}
        self._recent_records: deque[OperationRecord] = deque(maxlen=max_records)
        self._consecutive_failures = 0
        self._start_time = datetime.now()
        
        self._init_directory()
        self._load_metrics()
    
    def _init_directory(self) -> None:
        """Initialize metrics directory."""
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_metrics(self) -> None:
        """Load persisted metrics."""
        summary_path = self.metrics_dir / "metrics_summary.json"
        if summary_path.exists():
            try:
                data = read_json(summary_path)
                self._consecutive_failures = data.get("consecutive_failures", 0)
                log.debug("Loaded recovery metrics summary")
            except Exception as e:
                log.debug("Failed to load metrics summary: %s", e)
    
    def _save_metrics(self) -> None:
        """Save metrics to disk."""
        summary_path = self.metrics_dir / "metrics_summary.json"
        try:
            summary = {
                "updated_at": datetime.now().isoformat(),
                "consecutive_failures": self._consecutive_failures,
                "total_operations": sum(op.total_count for op in self._operations.values()),
                "overall_success_rate": self._calculate_overall_success_rate(),
            }
            atomic_write_json(summary_path, summary)
        except Exception as e:
            log.warning("Failed to save metrics summary: %s", e)
    
    def record_operation(
        self,
        operation: str,
        success: bool,
        duration_seconds: float,
        attempts: int = 1,
        error_type: str | None = None,
        error_message: str | None = None,
        recovery_strategy: str | None = None,
        fallback_used: bool = False,
    ) -> None:
        """Record an operation result.
        
        Args:
            operation: Operation name
            success: Whether operation succeeded
            duration_seconds: Operation duration
            attempts: Number of attempts made
            error_type: Type of error if failed
            error_message: Error message if failed
            recovery_strategy: Recovery strategy used
            fallback_used: Whether fallback was used
        """
        with self._lock:
            # Create record
            record = OperationRecord(
                operation=operation,
                timestamp=datetime.now().isoformat(),
                success=success,
                duration_seconds=duration_seconds,
                attempts=attempts,
                error_type=error_type,
                error_message=error_message,
                recovery_strategy=recovery_strategy,
                fallback_used=fallback_used,
            )
            
            # Add to recent records
            self._recent_records.append(record)
            
            # Update operation metrics
            if operation not in self._operations:
                self._operations[operation] = OperationMetrics(operation=operation)
            
            self._operations[operation].record(
                success=success,
                attempts=attempts,
                duration=duration_seconds,
                fallback=fallback_used,
            )
            
            # Update consecutive failures
            if success:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
            
            # Save metrics
            self._save_metrics()
    
    def get_operation_metrics(self, operation: str) -> OperationMetrics | None:
        """Get metrics for a specific operation."""
        return self._operations.get(operation)
    
    def get_all_operations(self) -> list[OperationMetrics]:
        """Get metrics for all operations."""
        with self._lock:
            return list(self._operations.values())
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate."""
        total = sum(op.total_count for op in self._operations.values())
        if total == 0:
            return 1.0
        successes = sum(op.success_count for op in self._operations.values())
        return successes / total
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of all recovery metrics."""
        with self._lock:
            operations = [op.to_dict() for op in self._operations.values()]
            operations.sort(key=lambda x: x["total_count"], reverse=True)
            
            return {
                "generated_at": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
                "total_operations": sum(op.total_count for op in self._operations.values()),
                "overall_success_rate": round(self._calculate_overall_success_rate(), 4),
                "consecutive_failures": self._consecutive_failures,
                "operations": operations,
                "recent_records_count": len(self._recent_records),
            }
    
    def get_health(self) -> RecoveryHealth:
        """Get current recovery health status."""
        with self._lock:
            health = RecoveryHealth()
            
            # Calculate 24h metrics
            cutoff = datetime.now() - timedelta(hours=24)
            recent_ops = [
                r for r in self._recent_records
                if datetime.fromisoformat(r.timestamp) > cutoff
            ]
            
            health.total_operations_24h = len(recent_ops)
            
            if recent_ops:
                successes = sum(1 for r in recent_ops if r.success)
                health.success_rate_24h = successes / len(recent_ops)
                
                # Average recovery time (for operations with multiple attempts)
                recovery_ops = [r for r in recent_ops if r.attempts > 1]
                if recovery_ops:
                    health.avg_recovery_time_seconds = (
                        sum(r.duration_seconds for r in recovery_ops) / len(recovery_ops)
                    )
            
            health.consecutive_failures = self._consecutive_failures
            
            # Determine status
            if health.consecutive_failures >= 5 or health.success_rate_24h < 0.5:
                health.status = "unhealthy"
            elif health.consecutive_failures >= 3 or health.success_rate_24h < 0.8:
                health.status = "degraded"
            else:
                health.status = "healthy"
            
            # Generate alerts
            if health.consecutive_failures >= 5:
                health.alerts.append(f"High consecutive failures: {health.consecutive_failures}")
            if health.success_rate_24h < 0.8:
                health.alerts.append(f"Low success rate (24h): {health.success_rate_24h:.1%}")
            if health.avg_recovery_time_seconds > 10:
                health.alerts.append(
                    f"Slow recovery time: {health.avg_recovery_time_seconds:.1f}s"
                )
            
            # Generate recommendations
            if health.status == "unhealthy":
                health.recommendations.append(
                    "Consider investigating failing operations and increasing retry limits"
                )
            if health.fallback_rate > 0.3:
                health.recommendations.append(
                    "High fallback usage detected - review primary operation reliability"
                )
            
            return health
    
    def get_trends(self, hours: int = 24) -> dict[str, Any]:
        """Get recovery trends over time."""
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            
            # Group by hour
            hourly: dict[str, dict] = defaultdict(lambda: {"total": 0, "success": 0})
            
            for record in self._recent_records:
                ts = datetime.fromisoformat(record.timestamp)
                if ts < cutoff:
                    continue
                
                hour_key = ts.strftime("%Y-%m-%d %H:00")
                hourly[hour_key]["total"] += 1
                if record.success:
                    hourly[hour_key]["success"] += 1
            
            # Calculate hourly success rates
            trends = []
            for hour_key in sorted(hourly.keys()):
                data = hourly[hour_key]
                trends.append({
                    "hour": hour_key,
                    "total": data["total"],
                    "success": data["success"],
                    "success_rate": round(data["success"] / data["total"], 4) if data["total"] > 0 else 1.0,
                })
            
            return {
                "period_hours": hours,
                "trends": trends,
            }
    
    def get_top_failures(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get top failing operations."""
        with self._lock:
            failures = []
            for op_metrics in self._operations.values():
                if op_metrics.failure_count > 0:
                    failures.append({
                        "operation": op_metrics.operation,
                        "failure_count": op_metrics.failure_count,
                        "failure_rate": round(1 - op_metrics.success_rate, 4),
                        "avg_attempts": round(op_metrics.avg_attempts, 2),
                    })
            
            failures.sort(key=lambda x: x["failure_count"], reverse=True)
            return failures[:limit]
    
    def export_metrics(self, output_path: Path | str | None = None) -> dict[str, Any]:
        """Export all metrics.
        
        Args:
            output_path: Optional path to save metrics
        
        Returns:
            Exported metrics dictionary
        """
        with self._lock:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "summary": self.get_summary(),
                "health": self.get_health().to_dict(),
                "trends_24h": self.get_trends(hours=24),
                "top_failures": self.get_top_failures(limit=10),
                "operations": [op.to_dict() for op in self._operations.values()],
            }
            
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    atomic_write_json(output_path, export_data)
                    log.info("Recovery metrics exported to %s", output_path)
                except Exception as e:
                    log.warning("Failed to export metrics: %s", e)
            
            return export_data
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._operations.clear()
            self._recent_records.clear()
            self._consecutive_failures = 0
            self._start_time = datetime.now()
            log.info("Recovery metrics reset")
    
    def cleanup_old_records(self) -> int:
        """Clean up old records.
        
        Returns:
            Number of records cleaned up
        """
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=self.retention_hours)
            original_count = len(self._recent_records)
            
            # Create new deque with only recent records
            self._recent_records = deque(
                (r for r in self._recent_records 
                 if datetime.fromisoformat(r.timestamp) > cutoff),
                maxlen=self.max_records,
            )
            
            cleaned = original_count - len(self._recent_records)
            if cleaned > 0:
                log.info("Cleaned up %d old recovery records", cleaned)
            
            return cleaned


# Global instance
_recovery_metrics: RecoveryMetrics | None = None
_metrics_lock = threading.Lock()


def get_recovery_metrics() -> RecoveryMetrics:
    """Get or create the global recovery metrics instance."""
    global _recovery_metrics
    
    if _recovery_metrics is None:
        with _metrics_lock:
            if _recovery_metrics is None:
                _recovery_metrics = RecoveryMetrics()
    
    return _recovery_metrics


def reset_recovery_metrics() -> None:
    """Reset the global recovery metrics instance (for testing)."""
    global _recovery_metrics
    with _metrics_lock:
        _recovery_metrics = None


def record_recovery(
    operation: str,
    success: bool,
    duration_seconds: float,
    attempts: int = 1,
    error_type: str | None = None,
    error_message: str | None = None,
    fallback_used: bool = False,
) -> None:
    """Convenience function to record a recovery operation.
    
    Args:
        operation: Operation name
        success: Whether operation succeeded
        duration_seconds: Operation duration
        attempts: Number of attempts
        error_type: Error type if failed
        error_message: Error message if failed
        fallback_used: Whether fallback was used
    """
    metrics = get_recovery_metrics()
    metrics.record_operation(
        operation=operation,
        success=success,
        duration_seconds=duration_seconds,
        attempts=attempts,
        error_type=error_type,
        error_message=error_message,
        fallback_used=fallback_used,
    )
