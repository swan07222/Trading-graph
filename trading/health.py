# trading/health.py
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import os
from pathlib import Path

from config import CONFIG
from core.events import EVENT_BUS, Event, EventType
from utils.logger import get_logger

log = get_logger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

class ComponentType(Enum):
    DATABASE = "database"
    BROKER = "broker"
    DATA_FEED = "data_feed"
    MODEL = "model"
    RISK_MANAGER = "risk_manager"
    OMS = "oms"
    NETWORK = "network"

@dataclass
class ComponentHealth:
    """Health status of a component"""
    component: ComponentType
    status: HealthStatus = HealthStatus.HEALTHY
    last_check: datetime = None
    last_success: datetime = None
    last_error: str = ""
    error_count: int = 0
    latency_ms: float = 0.0
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.last_check:
            self.last_check = datetime.now()

@dataclass
class SystemHealth:
    """Overall system health"""
    status: HealthStatus = HealthStatus.HEALTHY
    components: dict[str, ComponentHealth] = field(default_factory=dict)

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0

    can_trade: bool = True
    degraded_mode: bool = False

    last_quote_time: datetime = None
    quote_delay_seconds: float = 0.0

    start_time: datetime = None
    uptime_seconds: float = 0.0

    recent_errors: list[str] = field(default_factory=list)
    slo_pass: bool = True
    slo_violations: list[str] = field(default_factory=list)

    timestamp: datetime = None

    def to_dict(self) -> dict:
        return {
            'status': self.status.value,
            'can_trade': self.can_trade,
            'degraded_mode': self.degraded_mode,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'disk_percent': self.disk_percent,
            'quote_delay_seconds': self.quote_delay_seconds,
            'uptime_seconds': self.uptime_seconds,
            'components': {
                name: {
                    'status': comp.status.value,
                    'latency_ms': comp.latency_ms,
                    'error_count': comp.error_count,
                    'last_error': comp.last_error,
                }
                for name, comp in self.components.items()
            },
            'recent_errors': self.recent_errors[-10:],
            'slo_pass': self.slo_pass,
            'slo_violations': self.slo_violations[-20:],
            'timestamp': (
                self.timestamp.isoformat() if self.timestamp else None
            ),
        }

def _get_disk_percent() -> float:
    """Get disk usage percent, cross-platform safe."""
    if not HAS_PSUTIL:
        return 0.0
    try:
        base = str(getattr(CONFIG, "base_dir", os.path.abspath(os.sep)))
        root = Path(base).anchor or os.path.abspath(os.sep)
        return psutil.disk_usage(root).percent
    except Exception:
        try:
            return psutil.disk_usage(os.path.abspath(os.sep)).percent
        except Exception:
            return 0.0

class HealthMonitor:
    """Comprehensive health monitoring system.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._start_time = datetime.now()
        self._components: dict[ComponentType, ComponentHealth] = {}
        self._last_quote_time: datetime | None = None
        self._recent_errors: list[str] = []
        self._max_errors = 100
        self._broker = None
        self._prev_degraded = False

        self._thresholds = {
            'cpu_warning': 80,
            'cpu_critical': 95,
            'memory_warning': 80,
            'memory_critical': 95,
            'disk_warning': 85,
            'disk_critical': 95,
            'quote_delay_warning': 30,
            'quote_delay_critical': 60,
            'error_count_warning': 5,
            'error_count_critical': 10,
        }
        self._slo = {
            "max_quote_delay_seconds": 20.0,
            "max_cpu_percent": 90.0,
            "max_memory_percent": 92.0,
            "max_component_unhealthy": 0,
        }

        self._on_status_change: list[Callable] = []
        self._on_degraded: list[Callable] = []
        self._last_status: HealthStatus | None = None

        self._init_components()

        EVENT_BUS.subscribe(EventType.ERROR, self._on_error_event)
        EVENT_BUS.subscribe(EventType.TICK, self._on_tick_event)

    def _init_components(self):
        for comp_type in ComponentType:
            self._components[comp_type] = ComponentHealth(component=comp_type)

    def start(self):
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self._thread.start()
        log.info("Health monitor started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Health monitor stopped")

    def _monitor_loop(self):
        while self._running:
            try:
                self._run_checks()
            except Exception as e:
                log.error(f"Health check error: {e}")

            time.sleep(10)

    def attach_broker(self, broker):
        with self._lock:
            self._broker = broker

    def _run_checks(self):
        """Run all health checks."""
        with self._lock:
            cpu = psutil.cpu_percent(interval=1) if HAS_PSUTIL else 0.0
            memory = (
                psutil.virtual_memory().percent if HAS_PSUTIL else 0.0
            )
            disk = _get_disk_percent()

            self._check_database()
            self._check_broker()
            self._check_data_feed()
            self._check_model()

            health = self._calculate_overall_health(cpu, memory, disk)
            self._check_status_change(health)

    def _check_database(self):
        """Check database health.

        FIX: Tests the market data database directly instead of importing
        OMS (which creates a circular dependency risk and tests the wrong
        abstraction layer).
        """
        comp = self._components[ComponentType.DATABASE]
        start = time.time()

        try:
            from data.database import get_database

            db = get_database()
            _ = db.get_data_stats()

            comp.status = HealthStatus.HEALTHY
            comp.latency_ms = (time.time() - start) * 1000
            comp.last_success = datetime.now()
            comp.error_count = 0
            comp.last_error = ""

        except Exception as e:
            comp.status = HealthStatus.UNHEALTHY
            comp.last_error = str(e)
            comp.error_count += 1

        comp.last_check = datetime.now()

    def _check_broker(self):
        """Check broker connection health."""
        comp = self._components[ComponentType.BROKER]
        start = time.time()

        with self._lock:
            broker = self._broker

        if broker is None:
            comp.status = HealthStatus.DEGRADED
            comp.last_error = "No broker attached"
            comp.last_check = datetime.now()
            return

        try:
            if not broker.is_connected:
                comp.status = HealthStatus.UNHEALTHY
                comp.last_error = "Broker disconnected"
                comp.error_count += 1
            else:
                try:
                    account = broker.get_account()
                    if account and (
                        account.cash > 0
                        or account.equity > 0
                        or len(account.positions) > 0
                    ):
                        comp.status = HealthStatus.HEALTHY
                        comp.latency_ms = (time.time() - start) * 1000
                        comp.last_success = datetime.now()
                        comp.error_count = 0
                        comp.last_error = ""
                    else:
                        comp.status = HealthStatus.DEGRADED
                        comp.last_error = "Broker returned empty account"
                except Exception as e:
                    comp.status = HealthStatus.UNHEALTHY
                    comp.last_error = f"Broker query failed: {e}"
                    comp.error_count += 1

        except Exception as e:
            comp.status = HealthStatus.UNHEALTHY
            comp.last_error = str(e)
            comp.error_count += 1

        comp.last_check = datetime.now()

    def _check_data_feed(self):
        comp = self._components[ComponentType.DATA_FEED]

        if self._last_quote_time:
            delay = (datetime.now() - self._last_quote_time).total_seconds()

            if delay > self._thresholds['quote_delay_critical']:
                comp.status = HealthStatus.CRITICAL
                comp.last_error = f"No quotes for {delay:.0f}s"
            elif delay > self._thresholds['quote_delay_warning']:
                comp.status = HealthStatus.DEGRADED
                comp.last_error = f"Quote delay: {delay:.0f}s"
            else:
                comp.status = HealthStatus.HEALTHY
                comp.last_error = ""

            comp.details['quote_delay'] = delay
        else:
            comp.status = HealthStatus.DEGRADED
            comp.last_error = "No quotes received yet"

        comp.last_check = datetime.now()

    def _check_model(self):
        """Check ML model health - accept legacy OR interval/horizon models."""
        comp = self._components[ComponentType.MODEL]

        try:
            model_dir = CONFIG.MODEL_DIR

            legacy_model = model_dir / "ensemble.pt"
            legacy_scaler = model_dir / "scaler.pkl"

            has_new_model = list(model_dir.glob("ensemble_*.pt"))
            has_new_scaler = list(model_dir.glob("scaler_*.pkl"))

            has_model = legacy_model.exists() or bool(has_new_model)
            has_scaler = legacy_scaler.exists() or bool(has_new_scaler)

            matched_pairs: set[str] = set()
            if legacy_model.exists() and legacy_scaler.exists():
                matched_pairs.add("legacy")

            for ens in has_new_model:
                stem = str(ens.stem)
                if not stem.startswith("ensemble_"):
                    continue
                suffix = stem[len("ensemble_"):]
                if not suffix:
                    continue
                sc = model_dir / f"scaler_{suffix}.pkl"
                if sc.exists():
                    matched_pairs.add(suffix)

            if matched_pairs:
                comp.status = HealthStatus.HEALTHY
                comp.last_success = datetime.now()
                comp.last_error = ""
                comp.details["matched_model_scaler_pairs"] = sorted(matched_pairs)
            else:
                comp.status = HealthStatus.DEGRADED
                comp.details["matched_model_scaler_pairs"] = []
                if not has_model and not has_scaler:
                    comp.last_error = "Missing: model, scaler"
                elif not has_model:
                    comp.last_error = "Missing: model"
                elif not has_scaler:
                    comp.last_error = "Missing: scaler"
                else:
                    comp.last_error = (
                        "Model/scaler files exist but no compatible pair found "
                        "(expected ensemble_<interval>_<horizon>.pt + "
                        "scaler_<interval>_<horizon>.pkl)"
                    )

        except Exception as e:
            comp.status = HealthStatus.UNHEALTHY
            comp.last_error = str(e)

        comp.last_check = datetime.now()

    def _calculate_overall_health(
        self,
        cpu: float,
        memory: float,
        disk: float,
    ) -> SystemHealth:
        health = SystemHealth(
            cpu_percent=cpu,
            memory_percent=memory,
            disk_percent=disk,
            start_time=self._start_time,
            uptime_seconds=(
                datetime.now() - self._start_time
            ).total_seconds(),
            last_quote_time=self._last_quote_time,
            recent_errors=self._recent_errors.copy(),
            timestamp=datetime.now(),
        )

        if self._last_quote_time:
            health.quote_delay_seconds = (
                datetime.now() - self._last_quote_time
            ).total_seconds()

        health.components = {
            comp_type.value: comp
            for comp_type, comp in self._components.items()
        }

        statuses = [comp.status for comp in self._components.values()]

        if HealthStatus.CRITICAL in statuses:
            health.status = HealthStatus.CRITICAL
            health.can_trade = False
        elif HealthStatus.UNHEALTHY in statuses:
            health.status = HealthStatus.UNHEALTHY
            health.can_trade = False
        elif HealthStatus.DEGRADED in statuses:
            health.status = HealthStatus.DEGRADED
            health.degraded_mode = True
            health.can_trade = True
        else:
            health.status = HealthStatus.HEALTHY

        if (
            cpu > self._thresholds['cpu_critical']
            or memory > self._thresholds['memory_critical']
        ):
            health.status = HealthStatus.CRITICAL
            health.can_trade = False
        elif (
            cpu > self._thresholds['cpu_warning']
            or memory > self._thresholds['memory_warning']
        ):
            if health.status == HealthStatus.HEALTHY:
                health.status = HealthStatus.DEGRADED
                health.degraded_mode = True

        health.slo_pass, health.slo_violations = self._evaluate_slos(health)
        return health

    def _evaluate_slos(self, health: SystemHealth):
        """Evaluate lightweight operational SLOs for runbook-style monitoring."""
        violations: list[str] = []

        if health.quote_delay_seconds > float(self._slo["max_quote_delay_seconds"]):
            violations.append(
                f"quote_delay>{self._slo['max_quote_delay_seconds']}s "
                f"({health.quote_delay_seconds:.1f}s)"
            )

        if health.cpu_percent > float(self._slo["max_cpu_percent"]):
            violations.append(
                f"cpu>{self._slo['max_cpu_percent']}% "
                f"({health.cpu_percent:.1f}%)"
            )

        if health.memory_percent > float(self._slo["max_memory_percent"]):
            violations.append(
                f"memory>{self._slo['max_memory_percent']}% "
                f"({health.memory_percent:.1f}%)"
            )

        unhealthy = sum(
            1 for c in health.components.values()
            if c.status in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL)
        )
        if unhealthy > int(self._slo["max_component_unhealthy"]):
            violations.append(
                f"unhealthy_components>{self._slo['max_component_unhealthy']} "
                f"({unhealthy})"
            )

        return len(violations) == 0, violations

    def _check_status_change(self, health: SystemHealth):
        if self._last_status is None or self._last_status != health.status:
            self._last_status = health.status
            for callback in self._on_status_change:
                try:
                    callback(health)
                except Exception as e:
                    log.error(f"Status change callback error: {e}")

        if health.degraded_mode and not self._prev_degraded:
            log.warning("System entering degraded mode")
            for callback in self._on_degraded:
                try:
                    callback(health)
                except Exception as e:
                    log.error(f"Degraded callback error: {e}")

        self._prev_degraded = health.degraded_mode

    def _on_error_event(self, event: Event):
        with self._lock:
            error_msg = (
                f"{datetime.now().isoformat()}: "
                f"{event.data.get('error', 'Unknown error')}"
            )
            self._recent_errors.append(error_msg)

            if len(self._recent_errors) > self._max_errors:
                self._recent_errors = self._recent_errors[-self._max_errors:]

    def _on_tick_event(self, event: Event):
        with self._lock:
            self._last_quote_time = datetime.now()

    def report_component_health(
        self,
        component: ComponentType,
        status: HealthStatus,
        latency_ms: float = 0,
        error: str = "",
    ):
        with self._lock:
            comp = self._components.get(component)
            if comp:
                comp.status = status
                comp.latency_ms = latency_ms
                comp.last_check = datetime.now()

                if status == HealthStatus.HEALTHY:
                    comp.last_success = datetime.now()
                    comp.error_count = 0
                    comp.last_error = ""
                else:
                    comp.last_error = error
                    comp.error_count += 1

    def get_health(self) -> SystemHealth:
        with self._lock:
            cpu = psutil.cpu_percent() if HAS_PSUTIL else 0.0
            memory = (
                psutil.virtual_memory().percent if HAS_PSUTIL else 0.0
            )
            disk = _get_disk_percent()

            return self._calculate_overall_health(cpu, memory, disk)

    def get_health_json(self) -> str:
        return json.dumps(self.get_health().to_dict(), indent=2)

    def on_degraded(self, callback: Callable):
        self._on_degraded.append(callback)

    def on_status_change(self, callback: Callable):
        self._on_status_change.append(callback)

# FIX: Module-level lock instead of globals() pattern
_health_monitor: HealthMonitor | None = None
_health_lock = threading.Lock()

def get_health_monitor() -> HealthMonitor:
    global _health_monitor
    if _health_monitor is None:
        with _health_lock:
            if _health_monitor is None:
                _health_monitor = HealthMonitor()
    return _health_monitor
