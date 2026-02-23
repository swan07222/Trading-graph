# trading/__init__.py
def __getattr__(name: str):
    """Lazy import dispatcher.

    Allows `from trading import SimulatorBroker` etc. to work
    without eagerly importing every submodule at package load time.
    This prevents a broken dependency in one submodule (e.g., executor
    needs data.fetcher which needs network) from making the entire
    trading package unimportable.
    """
    _BROKER = {
        'BrokerInterface', 'SimulatorBroker', 'THSBroker', 'MultiVenueBroker', 'create_broker',
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
        from . import broker as _broker
        return getattr(_broker, name)

    if name in _OMS:
        from . import oms as _oms
        return getattr(_oms, name)

    if name in _RISK:
        from . import risk as _risk
        return getattr(_risk, name)

    if name in _KILL:
        from . import kill_switch as _kill_switch
        return getattr(_kill_switch, name)

    if name in _HEALTH:
        from . import health as _health
        return getattr(_health, name)

    if name in _ALERTS:
        from . import alerts as _alerts
        return getattr(_alerts, name)

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
    'MultiVenueBroker',
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
