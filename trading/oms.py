from __future__ import annotations

from threading import RLock

from core.types import Account, Fill, Order, Position

_DISABLED_REASON = (
    "OMS has been removed from this build; order execution and persistence are disabled."
)


class OrderManagementSystem:
    """Compatibility shim with OMS logic removed."""

    def __init__(
        self,
        initial_capital: float = 0.0,
        db_path: str | None = None,
        **kwargs,
    ) -> None:
        _ = (db_path, kwargs)
        cash = float(initial_capital or 0.0)
        self._account = Account(
            cash=cash,
            available=cash,
            initial_capital=cash,
            daily_start_equity=cash,
            peak_equity=cash,
        )

    def start(self) -> bool:
        return False

    def stop(self) -> None:
        return

    def close(self) -> None:
        return

    def get_account(self) -> Account:
        return self._account

    def get_positions(self) -> dict[str, Position]:
        return {}

    def get_position(self, symbol: str) -> Position | None:
        _ = symbol
        return None

    def get_orders(self, symbol: str | None = None) -> list[Order]:
        _ = symbol
        return []

    def get_active_orders(self) -> list[Order]:
        return []

    def get_order(self, order_id: str) -> Order | None:
        _ = order_id
        return None

    def get_order_by_broker_id(self, broker_id: str) -> Order | None:
        _ = broker_id
        return None

    def get_fills(self, order_id: str | None = None) -> list[Fill]:
        _ = order_id
        return []

    def get_order_timeline(self, order_id: str) -> list[dict[str, str]]:
        _ = order_id
        return []

    def submit_order(self, order: Order) -> Order:
        _ = order
        raise RuntimeError(_DISABLED_REASON)

    def update_order_status(self, *args, **kwargs) -> bool:
        _ = (args, kwargs)
        return False

    def process_fill(self, order: Order, fill: Fill) -> None:
        _ = (order, fill)
        raise RuntimeError(_DISABLED_REASON)

    def reconcile(self, *args, **kwargs) -> list[dict[str, str]]:
        _ = (args, kwargs)
        return []

    def force_sync_from_broker(self, *args, **kwargs) -> list[dict[str, str]]:
        _ = (args, kwargs)
        return []


_oms_instance: OrderManagementSystem | None = None
_oms_lock = RLock()


def create_oms(
    initial_capital: float = 0.0,
    db_path: str | None = None,
    **kwargs,
) -> OrderManagementSystem:
    return OrderManagementSystem(
        initial_capital=initial_capital,
        db_path=db_path,
        **kwargs,
    )


def get_oms(
    initial_capital: float = 0.0,
    db_path: str | None = None,
    **kwargs,
) -> OrderManagementSystem:
    global _oms_instance
    if _oms_instance is None:
        with _oms_lock:
            if _oms_instance is None:
                _oms_instance = create_oms(
                    initial_capital=initial_capital,
                    db_path=db_path,
                    **kwargs,
                )
    return _oms_instance


def reset_oms() -> None:
    global _oms_instance
    with _oms_lock:
        _oms_instance = None


__all__ = [
    "OrderManagementSystem",
    "create_oms",
    "get_oms",
    "reset_oms",
]
