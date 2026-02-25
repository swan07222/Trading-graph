# trading/oms.py
import sqlite3
import threading
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

from config.runtime_env import env_flag
from config.settings import CONFIG
from core.events import EVENT_BUS, EventType, OrderEvent
from core.exceptions import (
    InsufficientFundsError,
    InsufficientPositionError,
    OrderError,
    OrderValidationError,
)
from core.types import Account, Fill, Order, OrderSide, OrderStatus, Position
from trading.oms_db import OrderDatabase
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS
from utils.security import get_audit_log

log = get_logger(__name__)

_OMS_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS

class OrderStateMachine:
    """Valid order state transitions.
    Prevents invalid state changes.
    
    FIX #12: Added CANCELLED transition from PENDING to allow
    order cancellation before submission.
    """

    VALID_TRANSITIONS = {
        OrderStatus.PENDING: [
            OrderStatus.SUBMITTED, OrderStatus.REJECTED, OrderStatus.CANCELLED
        ],
        OrderStatus.SUBMITTED: [
            OrderStatus.ACCEPTED, OrderStatus.PARTIAL, OrderStatus.FILLED,
            OrderStatus.REJECTED, OrderStatus.CANCELLED
        ],
        OrderStatus.ACCEPTED: [
            OrderStatus.PARTIAL, OrderStatus.FILLED,
            OrderStatus.CANCELLED, OrderStatus.EXPIRED
        ],
        OrderStatus.PARTIAL: [
            OrderStatus.PARTIAL, OrderStatus.FILLED, OrderStatus.CANCELLED
        ],
        OrderStatus.FILLED: [],
        OrderStatus.CANCELLED: [],
        OrderStatus.REJECTED: [],
        OrderStatus.EXPIRED: [],
    }

    @classmethod
    def can_transition(cls, from_status: OrderStatus, to_status: OrderStatus) -> bool:
        return to_status in cls.VALID_TRANSITIONS.get(from_status, [])

    @classmethod
    def validate_transition(cls, order: Order, new_status: OrderStatus) -> None:
        if not cls.can_transition(order.status, new_status):
            raise OrderError(
                f"Invalid state transition: {order.status.value} -> {new_status.value}"
            )

class OrderManagementSystem:
    """Production Order Management System.

    Features:
    - SQLite persistence with atomic transactions
    - Crash recovery with reservation reconstruction
    - Order state machine
    - T+1 settlement
    - Reconciliation
    - Idempotent fill processing
    """

    def __init__(
        self,
        initial_capital: float = None,
        db_path: Path = None
    ) -> None:
        self._lock = threading.RLock()
        self._db = OrderDatabase(db_path=db_path)
        self._audit = get_audit_log()

        self._on_order_update: list[Callable] = []
        self._on_fill: list[Callable] = []

        self._account = self._recover_or_init(initial_capital)
        self._reconstruct_reservations()
        self._process_settlement()

        log.info(f"OMS initialized: equity=CNY {self._account.equity:,.2f}")

    def _recover_or_init(self, initial_capital: float = None) -> Account:
        """Recover from database or initialize new account."""
        account = self._db.load_account_state()

        if account:
            log.info("Recovered account state from database")
            return account

        capital = initial_capital or getattr(CONFIG, 'capital', 100000.0)
        account = Account(
            cash=capital,
            available=capital,
            initial_capital=capital,
            peak_equity=capital,
            daily_start_equity=capital,
            daily_start_date=date.today()
        )

        self._db.save_account_state(account)
        return account

    def _reconstruct_reservations(self) -> None:
        """Reconstruct cash/share reservations from active orders.
        Called on recovery to fix available cash and frozen shares.
        """
        active_orders = self._db.load_active_orders()

        # Start from a clean baseline to clear stale frozen quantities.
        for pos in self._account.positions.values():
            pos.frozen_qty = 0
            pos.available_qty = max(0, int(pos.quantity or 0))

        if not active_orders:
            self._enforce_invariants()
            return

        total_cash_reserved = 0.0
        frozen_by_symbol: dict[str, int] = {}

        for order in active_orders:
            if order.side == OrderSide.BUY:
                tags = order.tags or {}
                remaining = float(tags.get("reserved_cash_remaining", 0.0))
                total_cash_reserved += remaining
            elif order.side == OrderSide.SELL:
                unfilled = max(0, order.quantity - order.filled_qty)
                if unfilled > 0:
                    symbol = order.symbol
                    frozen_by_symbol[symbol] = (
                        frozen_by_symbol.get(symbol, 0) + unfilled
                    )

        self._account.available = max(
            0.0, self._account.cash - total_cash_reserved
        )
        self._account.frozen = total_cash_reserved

        for symbol, frozen_qty in frozen_by_symbol.items():
            pos = self._account.positions.get(symbol)
            if pos:
                pos.frozen_qty = min(frozen_qty, pos.quantity)
                pos.available_qty = max(
                    0, pos.quantity - pos.frozen_qty
                )

        self._enforce_invariants()

        if active_orders:
            log.info(
                f"Reconstructed reservations: "
                f"{len(active_orders)} active orders, "
                f"CNY {total_cash_reserved:,.2f} cash reserved, "
                f"{len(frozen_by_symbol)} symbols with frozen shares"
            )

    def _process_settlement(self) -> None:
        """Process T+1 settlement on startup."""
        settled = self._db.process_t1_settlement()
        if settled:
            log.info(
                f"T+1 settlement: {len(settled)} positions now available"
            )
            self._account.positions = self._db.load_positions()

    def _enforce_position_invariants(self, pos: Position) -> None:
        """Keep (available_qty, frozen_qty, quantity) consistent."""
        pos.quantity = max(0, int(pos.quantity or 0))
        pos.available_qty = max(0, int(pos.available_qty or 0))
        pos.frozen_qty = max(0, int(pos.frozen_qty or 0))

        total = pos.available_qty + pos.frozen_qty
        if total > pos.quantity:
            overflow = total - pos.quantity
            reduce_frozen = min(overflow, pos.frozen_qty)
            pos.frozen_qty -= reduce_frozen
            overflow -= reduce_frozen
            if overflow > 0:
                pos.available_qty = max(0, pos.available_qty - overflow)

        pos.available_qty = min(pos.available_qty, pos.quantity)

    def _enforce_invariants(self) -> None:
        """Ensure account + positions invariants are maintained."""
        cash = float(self._account.cash or 0.0)
        if abs(cash) < 1e-9:
            cash = 0.0
        self._account.cash = cash

        frozen = max(0.0, float(self._account.frozen or 0.0))
        self._account.frozen = frozen

        available = max(0.0, float(self._account.available or 0.0))
        max_available = max(0.0, cash - self._account.frozen)
        self._account.available = min(available, max_available)

        for pos in list(self._account.positions.values()):
            self._enforce_position_invariants(pos)

    def _check_new_day(self) -> None:
        """Check and handle new trading day."""
        today = date.today()
        if self._account.daily_start_date != today:
            settled = self._db.process_t1_settlement()
            if settled:
                self._account.positions = self._db.load_positions()

            self._account.daily_start_equity = self._account.equity
            self._account.daily_start_date = today
            self._db.save_account_state(self._account)
            log.info("New trading day initialized")

    @staticmethod
    def _sanitize_status_progress(
        *,
        order: Order,
        filled_qty: int | None,
        avg_price: float | None,
    ) -> tuple[int | None, float | None]:
        safe_filled: int | None = None
        if filled_qty is not None:
            parsed = max(0, int(filled_qty))
            safe_filled = min(parsed, max(0, int(order.quantity)))
        safe_avg: float | None = None
        if avg_price is not None:
            safe_avg = max(0.0, float(avg_price))
        return safe_filled, safe_avg

    def get_active_orders(self) -> list[Order]:
        """Get all active (non-terminal) orders."""
        return self._db.load_active_orders()

    def submit_order(self, order: Order) -> Order:
        """Submit order with validation and reservation.

        FIX C8: Added NaN/Inf validation to prevent invalid orders.
        FIX #4: Added maximum quantity/price limits to prevent overflow.
        """
        import math

        with self._lock:
            self._check_new_day()

            # FIX C8: Validate quantity is positive and finite (not NaN/Inf)
            if order.quantity <= 0 or not math.isfinite(order.quantity):
                raise OrderValidationError(
                    "Quantity must be positive and finite (not NaN or Inf)"
                )

            # FIX #4: Add maximum quantity limit to prevent integer overflow
            MAX_QUANTITY = 10**9  # 1 billion shares
            if order.quantity > MAX_QUANTITY:
                raise OrderValidationError(
                    f"Quantity {order.quantity} exceeds maximum allowed ({MAX_QUANTITY})"
                )

            # FIX C8: Validate price is positive and finite (not NaN/Inf)
            if order.price is None or float(order.price) <= 0 or not math.isfinite(float(order.price)):
                raise OrderValidationError(
                    "Order price must be positive and finite (not NaN or Inf)"
                )

            # FIX #4: Add maximum price limit to prevent overflow
            MAX_PRICE = 10**6  # 1 million per share
            if float(order.price) > MAX_PRICE:
                raise OrderValidationError(
                    f"Price {order.price} exceeds maximum allowed (CNY {MAX_PRICE})"
                )

            # FIX #11: Add maximum order value to prevent integer overflow in calculations
            MAX_ORDER_VALUE = 10**9  # 1 billion CNY total order value
            est_order_value = float(order.quantity) * float(order.price)
            if est_order_value > MAX_ORDER_VALUE:
                raise OrderValidationError(
                    f"Order value CNY {est_order_value:,.0f} exceeds maximum allowed (CNY {MAX_ORDER_VALUE:,})"
                )

            from core.constants import get_lot_size
            lot = get_lot_size(order.symbol)
            if order.quantity % lot != 0:
                raise OrderValidationError(
                    f"Quantity must be multiple of {lot} for {order.symbol}"
                )

            if order.side == OrderSide.BUY:
                self._validate_buy_order(order)
            else:
                self._validate_sell_order(order)

            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()
            order.updated_at = datetime.now()

            with self._db.transaction() as conn:
                self._db.save_order(order, conn)
                self._db.save_account_state(self._account, conn)
                self._db.save_order_event(
                    order_id=order.id,
                    event_type="submitted",
                    old_status=None,
                    new_status=order.status.value,
                    filled_qty=order.filled_qty,
                    avg_price=order.avg_price,
                    message=order.message or "",
                    payload={
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "quantity": int(order.quantity),
                        "price": float(order.price or 0.0),
                    },
                    conn=conn,
                )

                if order.side == OrderSide.SELL:
                    pos = self._account.positions.get(order.symbol)
                    if pos:
                        self._db.save_position(pos, conn)

            self._audit.log_order(
                order_id=order.id,
                code=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                price=order.price,
                status=order.status.value
            )
            self._notify_order_update(order)

            log.info(
                f"Order submitted: {order.id} {order.side.value} "
                f"{order.quantity} {order.symbol} @ CNY {order.price:.2f}"
            )
            return order

    def _validate_buy_order(self, order: Order) -> None:
        """Validate buy order and reserve cash."""
        if order.price <= 0:
            raise OrderValidationError("BUY reservation requires price > 0")

        trading_cfg = getattr(CONFIG, "trading", None)
        slip = float(
            getattr(trading_cfg, "slippage", 0.001)
            if trading_cfg else 0.001
        )
        comm_rate = float(
            getattr(trading_cfg, "commission", 0.00025)
            if trading_cfg else 0.00025
        )
        comm_min = 5.0

        est_price = float(order.price) * (1.0 + slip)
        est_value = float(order.quantity) * est_price
        est_commission = max(est_value * comm_rate, comm_min)
        reserved_total = est_value + est_commission

        order.tags = order.tags or {}
        order.tags.update({
            "reserved_price": float(order.price),
            "reserved_slip": slip,
            "reserved_commission_rate": comm_rate,
            "reserved_commission_min": comm_min,
            "reserved_cash_total": float(reserved_total),
            "reserved_cash_remaining": float(reserved_total),
        })

        if reserved_total > float(self._account.available):
            raise InsufficientFundsError(
                f"Insufficient funds: need CNY {reserved_total:,.2f}, "
                f"have CNY {self._account.available:,.2f}"
            )

        # Concentration limits are enforced in the risk manager/broker layer.
        # OMS validation only handles reservation and accounting consistency.

        self._account.available -= float(reserved_total)
        self._account.frozen += float(reserved_total)
        self._enforce_invariants()

    def _validate_sell_order(self, order: Order) -> None:
        """Validate sell order and freeze shares."""
        position = self._account.positions.get(order.symbol)

        if not position:
            raise InsufficientPositionError(
                f"No position in {order.symbol}"
            )

        if order.quantity > position.available_qty:
            raise InsufficientPositionError(
                f"Insufficient shares: have {position.available_qty}, "
                f"need {order.quantity}"
            )

        position.available_qty -= order.quantity
        position.frozen_qty += order.quantity

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def update_order_status(
        self,
        order_id: str,
        new_status: OrderStatus,
        message: str = "",
        broker_id: str = None,
        filled_qty: int = None,
        avg_price: float = None
    ) -> Order | None:
        """Update order status with state machine validation.
        Idempotent: same-status updates just refresh metadata.
        """
        with self._lock:
            order = self._db.load_order(order_id)
            if not order:
                log.warning(
                    f"Order not found for status update: {order_id}"
                )
                return None

            old_status = order.status
            safe_filled_qty, safe_avg_price = self._sanitize_status_progress(
                order=order,
                filled_qty=filled_qty,
                avg_price=avg_price,
            )

            # Idempotent: same status just updates metadata
            if new_status == old_status:
                if broker_id:
                    order.broker_id = broker_id
                if message:
                    order.message = message
                if safe_filled_qty is not None:
                    order.filled_qty = safe_filled_qty
                if safe_avg_price is not None:
                    order.avg_price = safe_avg_price
                order.updated_at = datetime.now()
                with self._db.transaction() as conn:
                    self._db.save_order(order, conn)
                    self._db.save_order_event(
                        order_id=order.id,
                        event_type="status_refresh",
                        old_status=old_status.value,
                        new_status=new_status.value,
                        filled_qty=order.filled_qty,
                        avg_price=order.avg_price,
                        message=message or "",
                        payload={"broker_id": broker_id or order.broker_id or ""},
                        conn=conn,
                    )
                self._notify_order_update(order)
                return order

            if not OrderStateMachine.can_transition(old_status, new_status):
                log.warning(
                    f"Invalid state transition for {order_id}: "
                    f"{old_status.value} -> {new_status.value}"
                )
                return order

            order.status = new_status
            order.message = message or order.message
            order.updated_at = datetime.now()

            if broker_id:
                order.broker_id = broker_id
            if safe_filled_qty is not None:
                order.filled_qty = safe_filled_qty
            if safe_avg_price is not None:
                order.avg_price = safe_avg_price

            if new_status == OrderStatus.FILLED:
                order.filled_qty = int(order.quantity)
                if order.avg_price <= 0 and float(order.price or 0.0) > 0.0:
                    order.avg_price = float(order.price)

            if new_status in (
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
                OrderStatus.EXPIRED
            ):
                if new_status == OrderStatus.CANCELLED:
                    order.cancelled_at = datetime.now()
                self._release_reserved(order)
            elif new_status == OrderStatus.FILLED:
                order.filled_at = datetime.now()

            with self._db.transaction() as conn:
                self._db.save_order(order, conn)
                self._db.save_account_state(self._account, conn)
                self._db.save_order_event(
                    order_id=order.id,
                    event_type="status_transition",
                    old_status=old_status.value,
                    new_status=new_status.value,
                    filled_qty=order.filled_qty,
                    avg_price=order.avg_price,
                    message=order.message or "",
                    payload={"broker_id": broker_id or order.broker_id or ""},
                    conn=conn,
                )

                if order.side == OrderSide.SELL and new_status in (
                    OrderStatus.CANCELLED, OrderStatus.REJECTED,
                    OrderStatus.EXPIRED
                ):
                    pos = self._account.positions.get(order.symbol)
                    if pos:
                        self._db.save_position(pos, conn)

            self._audit.log_order(
                order_id=order.id,
                code=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                price=order.price,
                status=new_status.value
            )
            self._notify_order_update(order)

            log.info(
                f"Order {order_id}: "
                f"{old_status.value} -> {new_status.value}"
            )
            return order

    def get_order_by_broker_id(
        self, broker_id: str
    ) -> Order | None:
        return self._db.load_order_by_broker_id(broker_id)

    def _release_reserved(self, order: Order) -> None:
        """Release reserved funds/shares for terminal orders."""
        if order.side == OrderSide.BUY:
            tags = order.tags or {}

            if "reserved_cash_remaining" in tags:
                reserved_rem = float(
                    tags.get("reserved_cash_remaining", 0.0) or 0.0
                )
            else:
                reserved_total = float(
                    tags.get("reserved_cash_total", 0.0) or 0.0
                )
                actual_used = (
                    float(order.filled_qty) * float(order.avg_price)
                    + float(order.commission or 0.0)
                )
                reserved_rem = max(0.0, reserved_total - actual_used)

            if reserved_rem > 0:
                self._account.available += reserved_rem
                self._account.frozen = max(
                    0.0, self._account.frozen - reserved_rem
                )

            tags["reserved_cash_remaining"] = 0.0
            order.tags = tags
            self._enforce_invariants()
            return

        # SELL: release frozen shares
        position = self._account.positions.get(order.symbol)
        if position:
            remaining_qty = max(
                0, int(order.quantity) - int(order.filled_qty)
            )
            to_release = min(remaining_qty, int(position.frozen_qty))
            position.frozen_qty = max(
                0, int(position.frozen_qty) - to_release
            )
            position.available_qty = min(
                int(position.available_qty) + to_release,
                int(position.quantity)
            )
            self._enforce_invariants()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _validate_fill_against_order(self, order: Order, fill: Fill) -> None:
        if not fill.order_id:
            fill.order_id = order.id
        if str(fill.order_id) != str(order.id):
            raise OrderValidationError(
                f"Fill order_id mismatch: {fill.order_id} != {order.id}"
            )

        if not fill.symbol:
            fill.symbol = order.symbol
        if str(fill.symbol) != str(order.symbol):
            raise OrderValidationError(
                f"Fill symbol mismatch: {fill.symbol} != {order.symbol}"
            )

        if fill.side != order.side:
            raise OrderValidationError(
                f"Fill side mismatch for {order.id}: "
                f"{fill.side.value} != {order.side.value}"
            )

        qty = int(fill.quantity)
        if qty <= 0:
            raise OrderValidationError("Fill quantity must be positive")

        price = float(fill.price)
        if price <= 0:
            raise OrderValidationError("Fill price must be positive")

        remaining_qty = max(0, int(order.quantity) - int(order.filled_qty))
        if qty > remaining_qty:
            raise OrderValidationError(
                f"Fill quantity exceeds remaining for {order.id}: "
                f"fill={qty}, remaining={remaining_qty}"
            )

        fill.quantity = qty
        fill.price = price

    def process_fill(self, order: Order, fill: Fill) -> None:
        """Process order fill — IDEMPOTENT.
        All mutations happen inside the transaction block.

        Thread Safety:
        - Acquires self._lock (RLock) for entire operation
        - Transaction is atomic with rollback on failure
        - Fill deduplication via database unique constraint
        """
        with self._lock:
            self._check_new_day()

            persisted = self._db.load_order(order.id)
            if persisted is None:
                raise OrderError(
                    f"Order not found for fill processing: {order.id}"
                )
            order = persisted

            if not fill.order_id:
                fill.order_id = order.id
            if not fill.symbol:
                fill.symbol = order.symbol

            # Idempotent: check if fill already exists
            if self._db.fill_exists(fill):
                log.debug("Fill already processed, skipping")
                return

            self._validate_fill_against_order(order, fill)

            fill_date = (
                fill.timestamp.date() if fill.timestamp else date.today()
            )

            new_filled_qty = order.filled_qty + int(fill.quantity)
            new_commission = (
                order.commission
                + float(fill.commission)
                + float(fill.stamp_tax)
            )

            # Average price: compute with the values before mutation
            if new_filled_qty > 0:
                if order.filled_qty > 0:
                    prev_value = order.filled_qty * order.avg_price
                    new_value = fill.quantity * fill.price
                    new_avg_price = (
                        (prev_value + new_value) / new_filled_qty
                    )
                else:
                    new_avg_price = fill.price
            else:
                new_avg_price = 0.0

            if new_filled_qty >= order.quantity:
                new_status = OrderStatus.FILLED
            else:
                new_status = OrderStatus.PARTIAL

            # Begin atomic transaction 鈥?all mutations inside
            with self._db.transaction() as conn:
                if not self._db.save_fill(fill, conn):
                    log.debug(
                        f"Fill duplicate detected during save: {fill.id}"
                    )
                    return

                order.filled_qty = new_filled_qty
                order.commission = new_commission
                order.avg_price = new_avg_price
                order.status = new_status
                order.updated_at = datetime.now()

                if new_status == OrderStatus.FILLED:
                    order.filled_at = datetime.now()

                if order.side == OrderSide.BUY:
                    tags = order.tags or {}
                    reserved_rem = float(
                        tags.get("reserved_cash_remaining", 0.0)
                    )
                    actual_cost = (
                        float(fill.quantity) * float(fill.price)
                        + float(fill.commission)
                        + float(fill.stamp_tax)
                    )

                    if actual_cost <= reserved_rem:
                        tags["reserved_cash_remaining"] = (
                            reserved_rem - actual_cost
                        )
                        self._account.frozen = max(
                            0.0, self._account.frozen - actual_cost
                        )
                    else:
                        over = actual_cost - reserved_rem
                        tags["reserved_cash_remaining"] = 0.0
                        self._account.frozen = max(
                            0.0,
                            self._account.frozen - reserved_rem
                        )
                        self._account.available = max(
                            0.0, self._account.available - over
                        )
                        tags["reserved_cash_overrun"] = float(
                            tags.get("reserved_cash_overrun", 0.0)
                        ) + over

                    order.tags = tags

                self._update_account_on_fill(
                    order, fill, fill_date, conn
                )

                if (
                    order.status == OrderStatus.FILLED
                    and order.side == OrderSide.BUY
                ):
                    tags = order.tags or {}
                    left = float(
                        tags.get("reserved_cash_remaining", 0.0)
                    )
                    if left > 0:
                        self._account.available += left
                        self._account.frozen = max(
                            0.0, self._account.frozen - left
                        )
                        tags["reserved_cash_remaining"] = 0.0
                        order.tags = tags

                self._enforce_invariants()

                self._db.save_order(order, conn)
                self._db.save_account_state(self._account, conn)
                self._db.save_order_event(
                    order_id=order.id,
                    event_type="fill",
                    old_status=None,
                    new_status=order.status.value,
                    filled_qty=order.filled_qty,
                    avg_price=order.avg_price,
                    message=f"fill:{fill.id}",
                    payload={
                        "fill_id": fill.id,
                        "broker_fill_id": fill.broker_fill_id,
                        "qty": int(fill.quantity),
                        "price": float(fill.price),
                        "commission": float(fill.commission or 0.0),
                        "stamp_tax": float(fill.stamp_tax or 0.0),
                        "timestamp": (
                            fill.timestamp.isoformat()
                            if fill.timestamp is not None
                            else ""
                        ),
                    },
                    conn=conn,
                )

            # Outside transaction: audit, callbacks, events
            self._audit.log_trade(
                order_id=order.id,
                code=order.symbol,
                side=order.side.value,
                quantity=fill.quantity,
                price=fill.price,
                commission=fill.commission
            )

            self._notify_order_update(order)
            self._notify_fill(fill)

            event_type = (
                EventType.ORDER_FILLED
                if order.status == OrderStatus.FILLED
                else EventType.ORDER_PARTIALLY_FILLED
            )
            EVENT_BUS.publish(OrderEvent(
                type=event_type,
                order_id=order.id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=fill.quantity,
                price=fill.price,
                filled_qty=order.filled_qty,
                filled_price=order.avg_price
            ))

            log.info(
                f"Fill: {order.side.value.upper()} {fill.quantity} "
                f"{order.symbol} @ CNY {fill.price:.2f} "
                f"(commission: CNY {fill.commission:.2f})"
            )

    def _update_account_on_fill(
        self,
        order: Order,
        fill: Fill,
        fill_date: date,
        conn: sqlite3.Connection
    ) -> None:
        """Update account and positions after fill."""
        if order.side == OrderSide.BUY:
            self._apply_buy_fill(order, fill, fill_date, conn)
        else:
            self._apply_sell_fill(order, fill, conn)

        if self._account.equity > self._account.peak_equity:
            self._account.peak_equity = self._account.equity

    def _apply_buy_fill(
        self,
        order: Order,
        fill: Fill,
        fill_date: date,
        conn: sqlite3.Connection
    ) -> None:
        """Apply a BUY fill to account + positions."""
        qty = int(fill.quantity)
        px = float(fill.price)
        fees = float(fill.commission) + float(fill.stamp_tax)
        trade_value = qty * px
        total_cost = trade_value + fees

        self._account.cash -= total_cost
        if self._account.cash < -0.01:
            log.warning(
                f"Cash went negative after buy fill: "
                f"CNY {self._account.cash:,.2f} "
                f"(cost CNY {total_cost:,.2f})"
            )
        self._account.commission_paid += fees

        pos = self._account.positions.get(order.symbol)
        if pos is None:
            pos = Position(
                symbol=order.symbol,
                name=order.name,
                quantity=0,
                available_qty=0,
                frozen_qty=0,
                avg_cost=0.0,
                current_price=px,
                realized_pnl=0.0,
                commission_paid=0.0,
                opened_at=datetime.now(),
            )
            self._account.positions[order.symbol] = pos

        old_qty = int(pos.quantity)
        new_qty = old_qty + qty
        if new_qty > 0:
            pos.avg_cost = (
                (pos.avg_cost * old_qty + px * qty) / new_qty
            )
        pos.quantity = new_qty
        pos.current_price = px
        pos.commission_paid += fees
        pos.last_updated = datetime.now()

        # T+1: shares available next trading day
        self._db.save_t1_pending(order.symbol, qty, fill_date, conn)

        self._db.save_position(pos, conn)

    def _apply_sell_fill(
        self, order: Order, fill: Fill, conn: sqlite3.Connection
    ) -> None:
        """Apply a SELL fill to account + positions."""
        qty = int(fill.quantity)
        px = float(fill.price)
        fees = float(fill.commission) + float(fill.stamp_tax)

        trade_value = qty * px
        proceeds = trade_value - fees

        pos = self._account.positions.get(order.symbol)
        if pos is None:
            raise OrderError(
                f"SELL fill for missing position: {order.symbol} "
                f"(order_id={order.id})"
            )

        self._account.cash += proceeds
        self._account.available += proceeds
        self._account.commission_paid += fees

        pos.frozen_qty = max(0, int(pos.frozen_qty) - qty)
        pos.quantity = max(0, int(pos.quantity) - qty)
        pos.current_price = px
        pos.commission_paid += fees

        gross = (px - float(pos.avg_cost)) * qty
        realized = gross - fees
        pos.realized_pnl += realized
        self._account.realized_pnl += realized

        pos.last_updated = datetime.now()

        if pos.quantity <= 0:
            self._account.positions.pop(order.symbol, None)
            self._db.delete_position(order.symbol, conn)
            self._db.clear_t1_pending(order.symbol, conn=conn)
        else:
            self._enforce_position_invariants(pos)
            self._db.save_position(pos, conn)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        updated = self.update_order_status(
            order_id, OrderStatus.CANCELLED
        )
        return bool(updated and updated.status == OrderStatus.CANCELLED)

    def get_order(self, order_id: str) -> Order | None:
        return self._db.load_order(order_id)

    def get_orders(self, symbol: str = None) -> list[Order]:
        if symbol:
            return self._db.load_orders_by_symbol(symbol)
        return self._db.load_active_orders()

    def get_fills(self, order_id: str = None) -> list[Fill]:
        return self._db.load_fills(order_id)

    def get_order_timeline(self, order_id: str, limit: int = 200) -> list[dict]:
        """Return order lifecycle events in ascending time order.
        Useful for UI/ops troubleshooting of OMS state transitions.
        """
        return self._db.load_order_events(order_id, limit=limit)

    def get_position(self, symbol: str) -> Position | None:
        return self._account.positions.get(symbol)

    def get_positions(self) -> dict[str, Position]:
        return self._account.positions.copy()

    def get_account(self) -> Account:
        return self._account

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update position prices and persist."""
        with self._lock:
            updated = False
            for symbol, price in prices.items():
                if symbol in self._account.positions:
                    self._account.positions[symbol].update_price(price)
                    updated = True

            if updated:
                self._account.last_updated = datetime.now()

                if self._account.equity > self._account.peak_equity:
                    self._account.peak_equity = self._account.equity

                with self._db.transaction() as conn:
                    for symbol in self._account.positions:
                        self._db.save_position(
                            self._account.positions[symbol], conn
                        )
                    self._db.save_account_state(self._account, conn)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def on_order_update(self, callback: Callable) -> None:
        self._on_order_update.append(callback)

    def on_fill(self, callback: Callable) -> None:
        self._on_fill.append(callback)

    def _notify_order_update(self, order: Order) -> None:
        for callback in self._on_order_update:
            try:
                callback(order)
            except _OMS_RECOVERABLE_EXCEPTIONS:
                name = getattr(callback, "__qualname__", repr(callback))
                log.exception(
                    "Order callback failed (callback=%s, order_id=%s)",
                    name,
                    getattr(order, "id", ""),
                )

    def _notify_fill(self, fill: Fill) -> None:
        for callback in self._on_fill:
            try:
                callback(fill)
            except _OMS_RECOVERABLE_EXCEPTIONS:
                name = getattr(callback, "__qualname__", repr(callback))
                log.exception(
                    "Fill callback failed (callback=%s, fill_id=%s)",
                    name,
                    getattr(fill, "id", ""),
                )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def reconcile(
        self,
        broker_positions: dict[str, Position],
        broker_cash: float
    ) -> dict:
        """Reconcile OMS state with broker. Returns discrepancies."""
        with self._lock:
            discrepancies = {
                'cash_diff': broker_cash - self._account.cash,
                'position_diffs': [],
                'missing_positions': [],
                'extra_positions': []
            }

            our_symbols = set(self._account.positions.keys())
            broker_symbols = set(broker_positions.keys())

            for symbol in broker_symbols - our_symbols:
                discrepancies['missing_positions'].append({
                    'symbol': symbol,
                    'broker_qty': broker_positions[symbol].quantity
                })

            for symbol in our_symbols - broker_symbols:
                discrepancies['extra_positions'].append({
                    'symbol': symbol,
                    'our_qty': self._account.positions[symbol].quantity
                })

            for symbol in our_symbols & broker_symbols:
                our_qty = self._account.positions[symbol].quantity
                broker_qty = broker_positions[symbol].quantity

                if our_qty != broker_qty:
                    discrepancies['position_diffs'].append({
                        'symbol': symbol,
                        'our_qty': our_qty,
                        'broker_qty': broker_qty,
                        'diff': broker_qty - our_qty
                    })

            has_discrepancy = any([
                abs(discrepancies['cash_diff']) > 1,
                discrepancies['position_diffs'],
                discrepancies['missing_positions'],
                discrepancies['extra_positions']
            ])

            if has_discrepancy:
                log.warning(
                    f"Reconciliation discrepancies: {discrepancies}"
                )
                self._audit.log_risk_event(
                    'reconciliation', discrepancies
                )

            return discrepancies

    def force_sync_from_broker(
        self,
        broker_positions: dict[str, Position],
        broker_cash: float,
        broker_available: float = None
    ) -> None:
        """Force sync OMS state from broker.
        WARNING: This overwrites OMS state.
        Clears T+1 pending for removed positions.
        """
        with self._lock:
            log.warning("Force syncing OMS state from broker")

            old_symbols = set(self._account.positions.keys())
            new_symbols = set(broker_positions.keys())

            self._account.cash = broker_cash
            self._account.available = (
                broker_available
                if broker_available is not None
                else broker_cash
            )
            self._account.frozen = max(
                0.0, broker_cash - self._account.available
            )
            self._account.positions = broker_positions.copy()

            self._enforce_invariants()

            with self._db.transaction() as conn:
                conn.execute("DELETE FROM positions")
                for pos in self._account.positions.values():
                    self._db.save_position(pos, conn)

                # Clear T+1 for removed symbols
                removed = old_symbols - new_symbols
                for symbol in removed:
                    self._db.clear_t1_pending(symbol, conn=conn)

                self._db.save_account_state(self._account, conn)

            self._audit.log_risk_event('force_sync', {
                'cash': broker_cash,
                'positions': len(broker_positions),
                'removed_symbols': list(removed) if removed else []
            })

            log.info(
                f"Synced from broker: cash=CNY {broker_cash:,.2f}, "
                f"positions={len(broker_positions)}"
            )

    def close(self) -> None:
        """Cleanup resources."""
        try:
            self._db.close_connection()
        except _OMS_RECOVERABLE_EXCEPTIONS as e:
            log.warning(f"Error closing OMS: {e}")

_oms_instances: dict[str, OrderManagementSystem] = {}
_oms_lock = threading.Lock()

def _singletons_disabled() -> bool:
    """Allow callers to opt out of process-global OMS singleton behavior."""
    return env_flag("TRADING_DISABLE_SINGLETONS", "0")

def _resolve_oms_instance_key(
    *,
    db_path: Path | None = None,
    instance: str | None = None,
) -> str:
    key = str(instance or "").strip()
    if key:
        return key
    if db_path is not None:
        try:
            return str(Path(db_path).expanduser().resolve())
        except _OMS_RECOVERABLE_EXCEPTIONS:
            return str(Path(db_path))
    return "default"

def create_oms(
    initial_capital: float = None,
    db_path: Path = None,
) -> OrderManagementSystem:
    """Create a new OMS instance without touching singleton registry."""
    return OrderManagementSystem(
        initial_capital=initial_capital,
        db_path=db_path,
    )

def get_oms(
    initial_capital: float = None,
    db_path: Path = None,
    *,
    instance: str | None = None,
    force_new: bool = False,
) -> OrderManagementSystem:
    """Get or create OMS instance (default singleton unless overridden)."""
    if force_new or _singletons_disabled():
        return create_oms(initial_capital=initial_capital, db_path=db_path)

    key = _resolve_oms_instance_key(db_path=db_path, instance=instance)
    inst = _oms_instances.get(key)
    if inst is not None:
        return inst

    with _oms_lock:
        inst = _oms_instances.get(key)
        if inst is None:
            inst = create_oms(
                initial_capital=initial_capital,
                db_path=db_path,
            )
            _oms_instances[key] = inst
        return inst

def reset_oms(
    *,
    instance: str | None = None,
    db_path: Path = None,
) -> None:
    """Reset OMS singleton(s); defaults to clearing all registered instances."""
    with _oms_lock:
        if instance is None and db_path is None:
            to_close = list(set(_oms_instances.values()))
            _oms_instances.clear()
            for inst in to_close:
                try:
                    inst.close()
                except _OMS_RECOVERABLE_EXCEPTIONS as exc:
                    log.warning("OMS close failed during reset: %s", exc)
            return

        key = _resolve_oms_instance_key(db_path=db_path, instance=instance)
        inst = _oms_instances.pop(key, None)
        if inst is not None:
            try:
                inst.close()
            except _OMS_RECOVERABLE_EXCEPTIONS as exc:
                log.warning("OMS close failed during reset for %s: %s", key, exc)
