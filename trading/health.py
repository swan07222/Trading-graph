# trading/health.py
"""
Health Monitoring System

FIXES APPLIED:
- Module-level lock instead of globals() pattern
- _check_database tests market database directly instead of importing OMS
- Cross-platform disk usage handling
"""
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from config import CONFIG
from core.types import SystemStatus
from core.events import EVENT_BUS, EventType, Event
from utils.logger import get_logger
from pathlib import Path
import os

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
    details: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.last_check:
            self.last_check = datetime.now()


@dataclass
class SystemHealth:
    """Overall system health"""
    status: HealthStatus = HealthStatus.HEALTHY
    components: Dict[str, ComponentHealth] = field(default_factory=dict)

    # System metrics
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0

    # Trading metrics
    can_trade: bool = True
    degraded_mode: bool = False

    # Data freshness
    last_quote_time: datetime = None
    quote_delay_seconds: float = 0.0

    # Uptime
    start_time: datetime = None
    uptime_seconds: float = 0.0

    # Errors
    recent_errors: List[str] = field(default_factory=list)

    timestamp: datetime = None

    def to_dict(self) -> Dict:
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
    """
    Comprehensive health monitoring system.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # State
        self._start_time = datetime.now()
        self._components: Dict[ComponentType, ComponentHealth] = {}
        self._last_quote_time: Optional[datetime] = None
        self._recent_errors: List[str] = []
        self._max_errors = 100
        self._broker = None
        self._prev_degraded = False

        # Thresholds
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

        # Callbacks
        self._on_status_change: List[Callable] = []
        self._on_degraded: List[Callable] = []

        # Initialize components
        self._init_components()

        # Subscribe to events
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
        """
        Check database health.

        FIX: Tests the market data database directly instead of importing
        OMS (which creates a circular dependency risk and tests the wrong
        abstraction layer).
        """
        comp = self._components[ComponentType.DATABASE]
        start = time.time()

        try:
            from data.database import get_database

            db = get_database()
            # Simple query to check connectivity
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
                        or len(account.positions) >= 0
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

            has_new_model = any(model_dir.glob("ensemble_*.pt"))
            has_new_scaler = any(model_dir.glob("scaler_*.pkl"))

            has_model = legacy_model.exists() or has_new_model
            has_scaler = legacy_scaler.exists() or has_new_scaler

            if has_model and has_scaler:
                comp.status = HealthStatus.HEALTHY
                comp.last_success = datetime.now()
                comp.last_error = ""
            else:
                comp.status = HealthStatus.DEGRADED
                missing = []
                if not has_model:
                    missing.append("model")
                if not has_scaler:
                    missing.append("scaler")
                comp.last_error = f"Missing: {', '.join(missing)}"

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

        return health

    def _check_status_change(self, health: SystemHealth):
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


# FIX: Module-level lock instead of globals() pattern
_health_monitor: Optional[HealthMonitor] = None
_health_lock = threading.Lock()


def get_health_monitor() -> HealthMonitor:
    global _health_monitor
    if _health_monitor is None:
        with _health_lock:
            if _health_monitor is None:
                _health_monitor = HealthMonitor()
    return _health_monitor