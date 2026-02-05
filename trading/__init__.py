# trading/__init__.py
"""
Trading Package - Production Grade
"""
from .broker import (
    BrokerInterface,
    SimulatorBroker,
    THSBroker,
    create_broker,
)
from .oms import (
    Order,
    Fill,
    Position,
    Account,
    OrderManagementSystem,
    get_oms,
)
from .risk import (
    RiskManager,
    get_risk_manager,
)
from .kill_switch import (
    KillSwitch,
    CircuitBreakerType,
    get_kill_switch,
)
from .health import (
    HealthMonitor,
    HealthStatus,
    get_health_monitor,
)
from .alerts import (
    AlertManager,
    Alert,
    AlertPriority,
    AlertCategory,
    get_alert_manager,
)
from .portfolio import Portfolio
from .signals import SignalGenerator
from .executor import ExecutionEngine

__all__ = [
    # Broker
    'BrokerInterface',
    'SimulatorBroker',
    'THSBroker',
    'create_broker',
    # OMS
    'Order',
    'Fill',
    'Position',
    'Account',
    'OrderManagementSystem',
    'get_oms',
    # Risk
    'RiskManager',
    'get_risk_manager',
    # Kill Switch
    'KillSwitch',
    'CircuitBreakerType',
    'get_kill_switch',
    # Health
    'HealthMonitor',
    'HealthStatus',
    'get_health_monitor',
    # Alerts
    'AlertManager',
    'Alert',
    'AlertPriority',
    'AlertCategory',
    'get_alert_manager',
    # Other
    'Portfolio',
    'SignalGenerator',
    'ExecutionEngine',
]