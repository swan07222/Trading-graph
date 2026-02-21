from __future__ import annotations

import os
import queue
import socket
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config import CONFIG, TradingMode
from core.types import (
    Account,
    AutoTradeMode,
    AutoTradeState,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TradeSignal,
)
from trading.health import ComponentType, HealthStatus
from trading.runtime_lease import RuntimeLeaseClient, create_runtime_lease_client
from utils.atomic_io import atomic_write_json, read_json
from utils.logger import get_logger
from utils.metrics import inc_counter, observe, set_gauge
from utils.security import get_access_control, get_audit_log

log = get_logger(__name__)
_SOFT_FAIL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    OverflowError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    ZeroDivisionError,
    queue.Empty,
    queue.Full,
)

try:
    from utils.metrics_http import register_snapshot_provider, unregister_snapshot_provider
except (ImportError, OSError):  # pragma: no cover - optional runtime integration
    register_snapshot_provider = None
    unregister_snapshot_provider = None

def _reconciliation_loop(self):
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
            log.error(f"Reconciliation error: {e}")


def _submit_with_retry(self, order: Order, attempts: int = 3) -> Order:
    """
    Retry broker.submit_order for transient failures.
    Does NOT retry validation failures (broker REJECTED).
    """
    delay = 0.5
    last_exc = None
    for _i in range(int(attempts)):
        try:
            result = self.broker.submit_order(order)
            if getattr(result, "status", None) == OrderStatus.REJECTED:
                return result
            return result
        except _SOFT_FAIL_EXCEPTIONS as e:
            last_exc = e
            if self._wait_or_stop(delay):
                break
            delay = min(delay * 2.0, 5.0)

    raise last_exc if last_exc else RuntimeError("submit_order failed")
