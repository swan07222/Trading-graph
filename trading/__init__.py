# trading/__init__.py
def __getattr__(name: str):
    """
    Lazy import dispatcher.

    Allows `from trading import SimulatorBroker` etc. to work
    without eagerly importing every submodule at package load time.
    This prevents a broken dependency in one submodule (e.g., executor
    needs data.fetcher which needs network) from making the entire
    trading package unimportable.
    """
    _BROKER = {
        'BrokerInterface', 'SimulatorBroker', 'THSBroker', 'create_broker',
    }
    _OMS = {
        'Order', 'Fill', 'Position', 'Account',
        'OrderManagementSystem', 'get_oms',
    }
    _RISK = {'RiskManager', 'get_risk_manager'}
    _KILL = {'KillSwitch', 'CircuitBreakerType', 'get_kill_switch'}
    _HEALTH = {'HealthMonitor', 'HealthStatus', 'get_health_monitor'}
    _ALERTS = {
        'AlertManager', 'Alert', 'AlertPriority', 'AlertCategory',
        'get_alert_manager',
    }
    _PORTFOLIO = {'Portfolio'}
    _SIGNALS = {'SignalGenerator'}
    _EXECUTOR = {'ExecutionEngine'}

    if name in _BROKER:
        from .broker import (
            BrokerInterface, SimulatorBroker, THSBroker, create_broker,
        )
        return locals()[name]

    if name in _OMS:
        from .oms import (
            Order, Fill, Position, Account,
            OrderManagementSystem, get_oms,
        )
        return locals()[name]

    if name in _RISK:
        from .risk import RiskManager, get_risk_manager
        return locals()[name]

    if name in _KILL:
        from .kill_switch import KillSwitch, CircuitBreakerType, get_kill_switch
        return locals()[name]

    if name in _HEALTH:
        from .health import HealthMonitor, HealthStatus, get_health_monitor
        return locals()[name]

    if name in _ALERTS:
        from .alerts import (
            AlertManager, Alert, AlertPriority, AlertCategory,
            get_alert_manager,
        )
        return locals()[name]

    if name in _PORTFOLIO:
        from .portfolio import Portfolio
        return Portfolio

    if name in _SIGNALS:
        from .signals import SignalGenerator
        return SignalGenerator

    if name in _EXECUTOR:
        from .executor import ExecutionEngine
        return ExecutionEngine

    raise AttributeError(f"module 'trading' has no attribute {name!r}")

__all__ = [
    'BrokerInterface',
    'SimulatorBroker',
    'THSBroker',
    'create_broker',
    'Order',
    'Fill',
    'Position',
    'Account',
    'OrderManagementSystem',
    'get_oms',
    'RiskManager',
    'get_risk_manager',
    'KillSwitch',
    'CircuitBreakerType',
    'get_kill_switch',
    'HealthMonitor',
    'HealthStatus',
    'get_health_monitor',
    'AlertManager',
    'Alert',
    'AlertPriority',
    'AlertCategory',
    'get_alert_manager',
    'Portfolio',
    'SignalGenerator',
    'ExecutionEngine',
]