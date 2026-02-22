from __future__ import annotations

from core.types import (
    Order,
    OrderStatus,
)
from trading.executor_error_policy import SOFT_FAIL_EXCEPTIONS
from trading.executor_policy_ops import _should_escalate_runtime_exception
from utils.logger import get_logger

log = get_logger(__name__)
_SOFT_FAIL_EXCEPTIONS = SOFT_FAIL_EXCEPTIONS

try:
    from utils.metrics_http import register_snapshot_provider, unregister_snapshot_provider
except (ImportError, OSError):  # pragma: no cover - optional runtime integration
    register_snapshot_provider = None
    unregister_snapshot_provider = None

def _reconciliation_loop(self) -> None:
    """Periodic reconciliation."""
    from trading.oms import get_oms

    oms = get_oms()

    while self._running:
        self._heartbeat("recon")
        try:
            if self._wait_or_stop(300.0):
                break
            if not self.broker.is_connected:
                continue

            broker_account = self.broker.get_account()
            broker_positions = self.broker.get_positions()
            discrepancies = oms.reconcile(
                broker_positions, broker_account.cash
            )

            if (
                abs(discrepancies.get('cash_diff', 0.0)) > 1.0
                or discrepancies.get('position_diffs')
                or discrepancies.get('missing_positions')
                or discrepancies.get('extra_positions')
            ):
                self._alert_manager.risk_alert(
                    "Reconciliation Discrepancy",
                    f"Cash diff: {discrepancies.get('cash_diff', 0):.2f}",
                    discrepancies,
                )
        except _SOFT_FAIL_EXCEPTIONS as e:
            if _should_escalate_runtime_exception(self, e):
                log.exception("Reconciliation fatal exception")
                raise
            log.error(f"Reconciliation error: {e}")


def _submit_with_retry(self, order: Order, attempts: int = 3) -> Order:
    """
    Retry broker.submit_order for transient failures.
    Does NOT retry validation failures (broker REJECTED).
    """
    total_attempts = max(1, int(attempts))
    delay = 0.5
    last_exc: BaseException | None = None
    for try_idx in range(total_attempts):
        try:
            result = self.broker.submit_order(order)
            if getattr(result, "status", None) == OrderStatus.REJECTED:
                return result
            return result
        except _SOFT_FAIL_EXCEPTIONS as e:
            if _should_escalate_runtime_exception(self, e):
                raise
            last_exc = e
            log.warning(
                "Order submit transient failure (%s/%s) order_id=%s symbol=%s: %s",
                int(try_idx + 1),
                int(total_attempts),
                str(getattr(order, "id", "") or "unknown"),
                str(getattr(order, "symbol", "") or "unknown"),
                e,
            )
            if self._wait_or_stop(delay):
                break
            delay = min(delay * 2.0, 5.0)

    details = (
        "submit_order failed after "
        f"{total_attempts} attempts "
        f"(order_id={getattr(order, 'id', 'unknown')} "
        f"symbol={getattr(order, 'symbol', 'unknown')})"
    )
    if last_exc is not None:
        raise RuntimeError(details) from last_exc
    raise RuntimeError(details)
