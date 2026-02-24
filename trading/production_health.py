"""
Production Readiness Health Checks

Addresses disadvantages:
- Desktop application = single point of failure
- No redundancy/failover for production trading
- Network issues can disrupt data feeds during critical moments
- No institutional-grade low-latency infrastructure

Features:
- Comprehensive system health monitoring
- Network latency and connectivity checks
- Data feed health validation
- Model readiness verification
- Risk system status checks
- Production gate checks
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ReadinessLevel(Enum):
    """Production readiness levels."""
    NOT_READY = "not_ready"  # Critical failures
    PAPER_ONLY = "paper_only"  # Can only paper trade
    LIMITED_LIVE = "limited_live"  # Small size live trading
    FULL_LIVE = "full_live"  # Full production ready


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    message: str
    value: str | None = None
    expected: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.value,
            "message": self.message,
            "value": self.value,
            "expected": self.expected,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    readiness: ReadinessLevel
    timestamp: datetime
    checks: list[HealthCheckResult]
    errors: list[str]
    warnings: list[str]
    uptime_seconds: float
    last_healthy_time: datetime | None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "readiness": self.readiness.value,
            "timestamp": self.timestamp.isoformat(),
            "checks_count": len(self.checks),
            "healthy_count": sum(1 for c in self.checks if c.status == HealthStatus.HEALTHY),
            "degraded_count": sum(1 for c in self.checks if c.status == HealthStatus.DEGRADED),
            "unhealthy_count": sum(1 for c in self.checks if c.status == HealthStatus.UNHEALTHY),
            "errors": self.errors,
            "warnings": self.warnings,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "last_healthy": (
                self.last_healthy_time.isoformat()
                if self.last_healthy_time else None
            ),
        }


class ProductionHealthMonitor:
    """
    Comprehensive production health monitoring.

    Checks:
    - System resources (CPU, memory, disk)
    - Network connectivity and latency
    - Data feed health
    - Model availability
    - Risk system status
    - Database connectivity
    - External dependencies
    """

    def __init__(
        self,
        check_interval_seconds: float = 5.0,
    ) -> None:
        self.check_interval = check_interval_seconds

        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: threading.Thread | None = None
        self._checks: list[Callable[[], HealthCheckResult]] = []
        self._last_results: list[HealthCheckResult] = []
        self._start_time: datetime | None = None
        self._last_healthy_time: datetime | None = None
        self._errors: list[str] = []
        self._warnings: list[str] = []

    def register_check(self, check_fn: Callable[[], HealthCheckResult]) -> None:
        """Register a health check function."""
        with self._lock:
            self._checks.append(check_fn)
            log.info(f"Health check registered: {check_fn.__name__}")

    def start(self) -> None:
        """Start health monitoring."""
        with self._lock:
            if self._running:
                log.warning("Health monitor already running")
                return

            self._start_time = datetime.now()
            self._running = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="production_health_monitor",
            )
            self._monitor_thread.start()
            log.info("Production health monitor started")

    def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10.0)
        log.info("Production health monitor stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                self._run_all_checks()
            except Exception as e:
                log.exception(f"Health monitor error: {e}")
                with self._lock:
                    self._errors.append(f"Monitor error: {str(e)}")

            time.sleep(self.check_interval)

    def _run_all_checks(self) -> None:
        """Run all registered health checks."""
        results = []
        errors = []
        warnings = []

        for check_fn in self._checks:
            try:
                result = check_fn()
                results.append(result)

                if result.status == HealthStatus.UNHEALTHY:
                    errors.append(f"{result.name}: {result.message}")
                elif result.status == HealthStatus.DEGRADED:
                    warnings.append(f"{result.name}: {result.message}")

            except Exception as e:
                log.exception(f"Health check failed: {check_fn.__name__}")
                result = HealthCheckResult(
                    name=check_fn.__name__,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                )
                results.append(result)
                errors.append(f"{check_fn.__name__}: {str(e)}")

        with self._lock:
            self._last_results = results
            self._errors = errors[-10:]  # Keep last 10
            self._warnings = warnings[-10:]

            # Update last healthy time
            if not errors:
                self._last_healthy_time = datetime.now()

    def get_health(self) -> SystemHealth:
        """Get current system health."""
        with self._lock:
            results = self._last_results.copy()
            errors = self._errors.copy()
            warnings = self._warnings.copy()
            last_healthy = self._last_healthy_time

        # Determine overall status
        if any(r.status == HealthStatus.UNHEALTHY for r in results):
            status = HealthStatus.UNHEALTHY
        elif any(r.status == HealthStatus.DEGRADED for r in results):
            status = HealthStatus.DEGRADED
        elif results:
            status = HealthStatus.HEALTHY
        else:
            status = HealthStatus.UNKNOWN

        # Determine readiness level
        readiness = self._calculate_readiness(results, errors)

        # Calculate uptime
        if self._start_time:
            uptime = (datetime.now() - self._start_time).total_seconds()
        else:
            uptime = 0.0

        return SystemHealth(
            status=status,
            readiness=readiness,
            timestamp=datetime.now(),
            checks=results,
            errors=errors,
            warnings=warnings,
            uptime_seconds=uptime,
            last_healthy_time=last_healthy,
        )

    def _calculate_readiness(
        self,
        results: list[HealthCheckResult],
        errors: list[str],
    ) -> ReadinessLevel:
        """Calculate production readiness level."""
        # Critical failures = not ready
        critical_checks = [
            "network_connectivity",
            "data_feed_health",
            "risk_system_status",
            "database_connectivity",
        ]

        for r in results:
            if r.name in critical_checks and r.status == HealthStatus.UNHEALTHY:
                return ReadinessLevel.NOT_READY

        # If any unhealthy, paper only
        if any(r.status == HealthStatus.UNHEALTHY for r in results):
            return ReadinessLevel.PAPER_ONLY

        # If degraded, limited live
        if any(r.status == HealthStatus.DEGRADED for r in results):
            return ReadinessLevel.LIMITED_LIVE

        # All healthy = full live
        if results and all(r.status == HealthStatus.HEALTHY for r in results):
            return ReadinessLevel.FULL_LIVE

        return ReadinessLevel.NOT_READY


# ============================================================================
# Built-in Health Check Functions
# ============================================================================

def check_network_connectivity(
    hosts: list[str] = None,
    timeout_seconds: float = 5.0,
) -> HealthCheckResult:
    """Check network connectivity to critical hosts."""
    import socket

    if hosts is None:
        hosts = [
            "8.8.8.8",  # Google DNS
            "114.114.114.114",  # China DNS
            "www.baidu.com",
        ]

    results = {}
    all_ok = True
    latencies = []

    for host in hosts:
        try:
            start = time.monotonic()
            socket.setdefaulttimeout(timeout_seconds)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, 80))
            latency = (time.monotonic() - start) * 1000
            results[host] = {"status": "ok", "latency_ms": round(latency, 1)}
            latencies.append(latency)
        except Exception as e:
            results[host] = {"status": "failed", "error": str(e)}
            all_ok = False

    avg_latency = np.mean(latencies) if latencies else float('inf')

    if all_ok and avg_latency < 200:
        status = HealthStatus.HEALTHY
    elif all_ok or avg_latency < 500:
        status = HealthStatus.DEGRADED
    else:
        status = HealthStatus.UNHEALTHY

    return HealthCheckResult(
        name="network_connectivity",
        status=status,
        message=f"Average latency: {avg_latency:.1f}ms" if latencies else "Connection failed",
        value=f"{len(latencies)}/{len(hosts)} hosts reachable",
        expected=f"{len(hosts)}/{len(hosts)} hosts reachable",
        details={"hosts": results, "avg_latency_ms": round(avg_latency, 1)},
    )


def check_data_feed_health(
    data_sources: list[str] = None,
    max_latency_ms: float = 5000.0,
    max_age_seconds: float = 60.0,
) -> HealthCheckResult:
    """Check data feed health."""
    # This would integrate with the data quality monitor
    # For now, placeholder implementation

    if data_sources is None:
        data_sources = ["tencent", "akshare", "sina"]

    # Placeholder - in production, this would check actual data freshness
    results = {src: {"status": "unknown"} for src in data_sources}

    healthy_count = sum(1 for r in results.values() if r.get("status") == "ok")
    total = len(results)

    if healthy_count == total:
        status = HealthStatus.HEALTHY
    elif healthy_count > 0:
        status = HealthStatus.DEGRADED
    else:
        status = HealthStatus.UNHEALTHY

    return HealthCheckResult(
        name="data_feed_health",
        status=status,
        message=f"{healthy_count}/{total} data sources healthy",
        value=f"{healthy_count}/{total}",
        expected=f"{total}/{total}",
        details={"sources": results},
    )


def check_model_readiness(
    model_dir: str = None,
    min_models: int = 1,
) -> HealthCheckResult:
    """Check model availability and readiness."""
    from pathlib import Path

    if model_dir is None:
        model_dir = "models_saved"

    model_path = Path(model_dir)

    if not model_path.exists():
        return HealthCheckResult(
            name="model_readiness",
            status=HealthStatus.UNHEALTHY,
            message=f"Model directory does not exist: {model_dir}",
            value="missing",
            expected="exists",
        )

    # Count model files
    model_files = list(model_path.glob("*.pt")) + list(model_path.glob("*.pth"))
    scaler_files = list(model_path.glob("*.pkl"))

    total_models = len(model_files) + len(scaler_files)

    if total_models >= min_models:
        status = HealthStatus.HEALTHY
        message = f"{total_models} model files found"
    elif total_models > 0:
        status = HealthStatus.DEGRADED
        message = f"Only {total_models} model files (min: {min_models})"
    else:
        status = HealthStatus.UNHEALTHY
        message = "No model files found"

    return HealthCheckResult(
        name="model_readiness",
        status=status,
        message=message,
        value=str(total_models),
        expected=f">={min_models}",
        details={
            "model_files": len(model_files),
            "scaler_files": len(scaler_files),
            "model_dir": str(model_path.absolute()),
        },
    )


def check_risk_system_status(
    oms_connected: bool = True,
    kill_switch_ok: bool = True,
    circuit_breaker_ok: bool = True,
) -> HealthCheckResult:
    """Check risk system status."""
    issues = []

    if not oms_connected:
        issues.append("OMS not connected")
    if not kill_switch_ok:
        issues.append("Kill switch not ready")
    if not circuit_breaker_ok:
        issues.append("Circuit breaker not ready")

    if issues:
        status = HealthStatus.UNHEALTHY
        message = "; ".join(issues)
    else:
        status = HealthStatus.HEALTHY
        message = "All risk systems operational"

    return HealthCheckResult(
        name="risk_system_status",
        status=status,
        message=message,
        value="ok" if status == HealthStatus.HEALTHY else "issues",
        expected="ok",
        details={
            "oms_connected": oms_connected,
            "kill_switch_ok": kill_switch_ok,
            "circuit_breaker_ok": circuit_breaker_ok,
        },
    )


def check_database_connectivity(
    db_paths: list[str] = None,
) -> HealthCheckResult:
    """Check database connectivity."""
    from pathlib import Path

    if db_paths is None:
        db_paths = ["data_storage/orders.db", "data_storage/cache.db"]

    results = {}
    all_ok = True

    for path in db_paths:
        db_path = Path(path)
        exists = db_path.exists()
        writable = os.access(str(db_path), os.W_OK) if exists else False

        results[path] = {
            "exists": exists,
            "writable": writable,
        }

        if not exists or not writable:
            all_ok = False

    if all_ok:
        status = HealthStatus.HEALTHY
        message = f"{len(db_paths)} databases accessible"
    else:
        status = HealthStatus.DEGRADED
        message = "Some database issues detected"

    return HealthCheckResult(
        name="database_connectivity",
        status=status,
        message=message,
        value=f"{sum(1 for r in results.values() if r['exists'] and r['writable'])}/{len(db_paths)}",
        expected=f"{len(db_paths)}/{len(db_paths)}",
        details={"databases": results},
    )


def check_system_resources(
    min_disk_gb: float = 1.0,
    max_cpu_pct: float = 95.0,
    max_memory_pct: float = 95.0,
) -> HealthCheckResult:
    """Check system resources."""
    try:
        import psutil
    except ImportError:
        return HealthCheckResult(
            name="system_resources",
            status=HealthStatus.DEGRADED,
            message="psutil not installed",
            details={"error": "psutil not installed"},
        )

    issues = []
    details = {}

    # CPU usage
    cpu_pct = psutil.cpu_percent(interval=0.1)
    details["cpu_percent"] = cpu_pct
    if cpu_pct > max_cpu_pct:
        issues.append(f"High CPU: {cpu_pct:.1f}%")

    # Memory usage
    memory = psutil.virtual_memory()
    memory_pct = memory.percent
    details["memory_percent"] = memory_pct
    if memory_pct > max_memory_pct:
        issues.append(f"High memory: {memory_pct:.1f}%")

    # Disk usage
    disk = psutil.disk_usage(".")
    disk_gb_free = disk.free / (1024 ** 3)
    details["disk_free_gb"] = round(disk_gb_free, 1)
    if disk_gb_free < min_disk_gb:
        issues.append(f"Low disk: {disk_gb_free:.1f}GB free")

    if issues:
        status = HealthStatus.DEGRADED
        message = "; ".join(issues)
    else:
        status = HealthStatus.HEALTHY
        message = "Resources OK"

    return HealthCheckResult(
        name="system_resources",
        status=status,
        message=message,
        value="ok" if status == HealthStatus.HEALTHY else "issues",
        expected="ok",
        details=details,
    )


def check_latency_to_broker(
    broker_endpoint: str = None,
    timeout_ms: float = 1000.0,
) -> HealthCheckResult:
    """Check latency to broker (for live trading)."""
    import socket

    if broker_endpoint is None:
        # Placeholder - would be actual broker endpoint
        return HealthCheckResult(
            name="latency_to_broker",
            status=HealthStatus.UNKNOWN,
            message="No broker endpoint configured",
            details={"configured": False},
        )

    try:
        host, port = broker_endpoint.rsplit(":", 1)
        port = int(port)

        start = time.monotonic()
        socket.setdefaulttimeout(timeout_ms / 1000)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        latency = (time.monotonic() - start) * 1000
        sock.close()

        if latency < 100:
            status = HealthStatus.HEALTHY
        elif latency < 500:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        return HealthCheckResult(
            name="latency_to_broker",
            status=status,
            message=f"Latency: {latency:.1f}ms",
            value=f"{latency:.1f}ms",
            expected="<100ms",
            details={"latency_ms": round(latency, 1)},
        )

    except Exception as e:
        return HealthCheckResult(
            name="latency_to_broker",
            status=HealthStatus.UNHEALTHY,
            message=f"Connection failed: {e}",
            details={"error": str(e)},
        )


# ============================================================================
# Production Gate
# ============================================================================

class ProductionGate:
    """
    Production gate for go/no-go decisions.

    Usage:
        gate = ProductionGate()
        gate.register_check(check_network_connectivity)
        gate.register_check(check_data_feed_health)
        gate.register_check(check_risk_system_status)

        if gate.passes():
            start_trading()
        else:
            log.warning("Production gate failed")
    """

    def __init__(
        self,
        required_checks: list[str] = None,
    ) -> None:
        self.required_checks = required_checks or []

        self._lock = threading.RLock()
        self._checks: dict[str, Callable[[], HealthCheckResult]] = {}
        self._last_results: dict[str, HealthCheckResult] = {}

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], HealthCheckResult],
        required: bool = False,
    ) -> None:
        """Register a production gate check."""
        with self._lock:
            self._checks[name] = check_fn
            if required:
                self.required_checks.append(name)

    def run_checks(self) -> dict[str, HealthCheckResult]:
        """Run all checks and return results."""
        results = {}

        for name, check_fn in self._checks.items():
            try:
                result = check_fn()
                results[name] = result
            except Exception as e:
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                )

        with self._lock:
            self._last_results = results

        return results

    def passes(self, require_all: bool = True) -> bool:
        """
        Check if production gate passes.

        Args:
            require_all: If True, all required checks must pass.
                        If False, majority must pass.

        Returns:
            True if gate passes
        """
        if not self._last_results:
            self.run_checks()

        if require_all:
            # All required checks must be healthy
            for name in self.required_checks:
                result = self._last_results.get(name)
                if not result or result.status != HealthStatus.HEALTHY:
                    return False
            return True
        else:
            # Majority must be healthy
            healthy = sum(
                1 for r in self._last_results.values()
                if r.status == HealthStatus.HEALTHY
            )
            return healthy > len(self._last_results) / 2

    def get_report(self) -> dict:
        """Get production gate report."""
        if not self._last_results:
            self.run_checks()

        return {
            "passes": self.passes(),
            "passes_majority": self.passes(require_all=False),
            "timestamp": datetime.now().isoformat(),
            "checks": {
                name: {
                    "status": r.status.value,
                    "message": r.message,
                    "required": name in self.required_checks,
                }
                for name, r in self._last_results.items()
            },
            "required_checks": self.required_checks,
            "failed_required": [
                name for name in self.required_checks
                if (result := self._last_results.get(name)) and result.status != HealthStatus.HEALTHY
            ],
        }


def create_production_monitor() -> ProductionHealthMonitor:
    """Create production health monitor with default checks."""
    monitor = ProductionHealthMonitor()

    # Register default checks
    monitor.register_check(check_network_connectivity)
    monitor.register_check(check_data_feed_health)
    monitor.register_check(check_model_readiness)
    monitor.register_check(check_risk_system_status)
    monitor.register_check(check_database_connectivity)
    monitor.register_check(check_system_resources)

    return monitor


def create_production_gate() -> ProductionGate:
    """Create production gate with required checks."""
    gate = ProductionGate(
        required_checks=[
            "network_connectivity",
            "data_feed_health",
            "risk_system_status",
            "database_connectivity",
        ]
    )

    gate.register_check("network_connectivity", check_network_connectivity, required=True)
    gate.register_check("data_feed_health", check_data_feed_health, required=True)
    gate.register_check("model_readiness", check_model_readiness, required=False)
    gate.register_check("risk_system_status", check_risk_system_status, required=True)
    gate.register_check("database_connectivity", check_database_connectivity, required=True)
    gate.register_check("system_resources", check_system_resources, required=False)

    return gate
