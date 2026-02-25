from types import SimpleNamespace

import pytest

try:
    from trading.health import ComponentType, HealthMonitor, HealthStatus

    _EXECUTION_STACK_AVAILABLE = True
except ImportError:
    _EXECUTION_STACK_AVAILABLE = False
    ComponentType = HealthMonitor = HealthStatus = object  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    not _EXECUTION_STACK_AVAILABLE,
    reason="Execution stack modules are removed in analysis-only build.",
)


def test_broker_health_empty_account_is_degraded() -> None:
    class _Broker:
        is_connected = True

        def get_account(self):
            return SimpleNamespace(cash=0.0, equity=0.0, positions={})

    monitor = HealthMonitor()
    monitor.attach_broker(_Broker())

    monitor._check_broker()
    comp = monitor._components[ComponentType.BROKER]

    assert comp.status == HealthStatus.DEGRADED
    assert "empty account" in comp.last_error.lower()


def test_broker_health_positive_cash_is_healthy() -> None:
    class _Broker:
        is_connected = True

        def get_account(self):
            return SimpleNamespace(cash=1000.0, equity=1000.0, positions={})

    monitor = HealthMonitor()
    monitor.attach_broker(_Broker())

    monitor._check_broker()
    comp = monitor._components[ComponentType.BROKER]

    assert comp.status == HealthStatus.HEALTHY
    assert comp.last_error == ""
