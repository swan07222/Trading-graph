from __future__ import annotations

from dataclasses import replace
from threading import RLock

from core.types import Account, OrderSide, RiskLevel, RiskMetrics

_DISABLED_REASON = (
    "Risk management has been removed from this build; trading execution is disabled."
)


class RiskManager:
    """Compatibility shim with risk-management logic removed."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._metrics = RiskMetrics(
            risk_level=RiskLevel.HIGH,
            can_trade=False,
            kill_switch_active=True,
            warnings=[_DISABLED_REASON],
        )

    def initialize(self, account: Account | None = None) -> None:
        self.update(account)

    def update(self, account: Account | None = None) -> None:
        with self._lock:
            if account is not None:
                self._metrics = replace(
                    self._metrics,
                    equity=float(getattr(account, "equity", 0.0) or 0.0),
                    cash=float(getattr(account, "cash", 0.0) or 0.0),
                    positions_value=float(getattr(account, "positions_value", 0.0) or 0.0),
                    position_count=int(len(getattr(account, "positions", {}) or {})),
                )

    def get_metrics(self) -> RiskMetrics:
        with self._lock:
            return self._metrics

    def check_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
    ) -> tuple[bool, str]:
        _ = (symbol, side, quantity, price)
        return False, _DISABLED_REASON

    def can_submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
    ) -> tuple[bool, str]:
        return self.check_order(symbol, side, quantity, price)

    def record_order_submission(self) -> None:
        return

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        confidence: float = 0.0,
    ) -> int:
        _ = (symbol, entry_price, stop_loss, confidence)
        return 0


_risk_manager: RiskManager | None = None
_risk_lock = RLock()


def get_risk_manager() -> RiskManager:
    global _risk_manager
    if _risk_manager is None:
        with _risk_lock:
            if _risk_manager is None:
                _risk_manager = RiskManager()
    return _risk_manager


def reset_risk_manager() -> None:
    global _risk_manager
    with _risk_lock:
        _risk_manager = None


__all__ = ["RiskManager", "get_risk_manager", "reset_risk_manager"]
