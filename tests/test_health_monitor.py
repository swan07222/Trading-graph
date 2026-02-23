from types import SimpleNamespace

from trading.health import ComponentType, HealthMonitor, HealthStatus


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
