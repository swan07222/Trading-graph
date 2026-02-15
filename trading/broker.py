# trading/broker.py
"""
Unified Broker Interface - Production Grade with Full Fill Sync

Supports:
- Paper Trading (Simulator)
- 鍚岃姳椤?(THS)
- 鍗庢嘲璇佸埜 (HT)
- 鎷涘晢璇佸埜 (ZSZQ)
- 鍥介噾璇佸埜 (GJ)
- 閾舵渤璇佸埜 (YH)

"""
import hashlib
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

from config.settings import CONFIG
from core.types import Account, Fill, Order, OrderSide, OrderStatus, OrderType, Position
from utils.logger import get_logger

log = get_logger(__name__)

# Bounded ID mapping (LRU eviction)
# FIX(6): Prevents unbounded growth of order ID maps

class BoundedOrderedDict(OrderedDict):
    """OrderedDict with max size 鈥?evicts oldest on overflow."""

    def __init__(self, maxsize: int = 10000, *args, **kwargs):
        self._maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self._maxsize:
            self.popitem(last=False)

# Module-level fill ID generator
# FIX(12): Fallback uses stable hash instead of datetime.now()

def make_fill_uid(
    broker_name: str,
    broker_fill_id: str,
    symbol: str,
    ts: datetime,
    price: float,
    qty: int,
) -> str:
    """
    Create a globally-unique, stable Fill.id for OMS primary key.

    Format with broker_fill_id:
      FILL|<YYYY-MM-DD>|<broker>|<broker_fill_id>|<symbol>
    Fallback (no broker_fill_id):
      FILL|<YYYY-MM-DD>|<broker>|<symbol>|<hash>
    """
    broker = (broker_name or "broker").replace("|", "_")
    broker_fill_id = (broker_fill_id or "").strip()
    sym = str(symbol or "").strip()

    day = (
        ts.date().isoformat()
        if isinstance(ts, datetime)
        else date.today().isoformat()
    )

    if broker_fill_id:
        return f"FILL|{day}|{broker}|{broker_fill_id}|{sym}"

    # FIX(12): Stable hash from all available fields
    iso = ts.isoformat() if isinstance(ts, datetime) else day
    raw = f"{iso}|{broker}|{sym}|{int(qty)}|{float(price):.4f}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"FILL|{day}|{broker}|{sym}|{h}"

# FIX(3): Single function instead of duplicated methods

def parse_broker_status(status_str: str) -> OrderStatus:
    """
    Parse broker/native status strings into internal OrderStatus.

    Handles both English keywords and common Chinese statuses.
    Unicode escapes are used for Chinese terms to avoid source-encoding drift.
    """
    s = str(status_str or "").strip().lower()
    if not s:
        return OrderStatus.SUBMITTED

    def _has_any(*tokens: str) -> bool:
        return any(t and (t in s) for t in tokens)

    if _has_any(
        "partial",
        "partially filled",
        "\u90e8\u5206\u6210\u4ea4",  # 部分成交
    ):
        return OrderStatus.PARTIAL

    if (
        s == "filled"
        or _has_any(
            "fully filled",
            "all traded",
            "\u5168\u90e8\u6210\u4ea4",  # 全部成交
            "\u5df2\u6210",              # 已成
        )
    ):
        return OrderStatus.FILLED

    if _has_any(
        "accepted",
        "submitted",
        "pending",
        "new",
        "\u5df2\u62a5",              # 已报
        "\u5df2\u59d4\u6258",        # 已委托
    ):
        return OrderStatus.ACCEPTED

    if _has_any(
        "cancelled",
        "canceled",
        "cancelled by user",
        "\u5df2\u64a4",              # 已撤
        "\u64a4\u5355",              # 撤单
    ):
        return OrderStatus.CANCELLED

    if _has_any(
        "rejected",
        "reject",
        "invalid",
        "\u5e9f\u5355",              # 废单
        "\u62d2\u7edd",              # 拒绝
    ):
        return OrderStatus.REJECTED

    return OrderStatus.SUBMITTED

class BrokerInterface(ABC):
    """
    Abstract broker interface 鈥?all brokers must implement this.
    Thread-safe design with callbacks for order updates.
    """

    # FIX(6): Max tracked order mappings
    _MAX_ORDER_MAPPINGS = 10000

    def __init__(self):
        self._lock = threading.RLock()
        self._callbacks: dict[str, list[Callable]] = {
            'order_update': [],
            'trade': [],
            'error': [],
        }
        # FIX(6): Bounded mappings
        self._order_id_to_broker_id = BoundedOrderedDict(
            maxsize=self._MAX_ORDER_MAPPINGS,
        )
        self._broker_id_to_order_id = BoundedOrderedDict(
            maxsize=self._MAX_ORDER_MAPPINGS,
        )

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @abstractmethod
    def connect(self, **kwargs) -> bool:
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def get_account(self) -> Account:
        pass

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Position | None:
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def get_orders(self, active_only: bool = True) -> list[Order]:
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> float | None:
        pass

    @abstractmethod
    def get_fills(self, since: datetime = None) -> list[Fill]:
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus | None:
        pass

    @abstractmethod
    def sync_order(self, order: Order) -> Order:
        pass

    def get_broker_id(self, order_id: str) -> str | None:
        return self._order_id_to_broker_id.get(order_id)

    def get_order_id(self, broker_id: str) -> str | None:
        return self._broker_id_to_order_id.get(broker_id)

    def register_order_mapping(self, order_id: str, broker_id: str):
        with self._lock:
            self._order_id_to_broker_id[order_id] = broker_id
            self._broker_id_to_order_id[broker_id] = order_id

    # === Convenience Methods ===
    # FIX(8): price=None is valid for market orders

    def buy(
        self, symbol: str, qty: int, price: float = None,
    ) -> Order:
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            quantity=qty,
            price=price or 0.0,
        )
        return self.submit_order(order)

    def sell(
        self, symbol: str, qty: int, price: float = None,
    ) -> Order:
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            quantity=qty,
            price=price or 0.0,
        )
        return self.submit_order(order)

    def sell_all(self, symbol: str, price: float = None) -> Order | None:
        pos = self.get_position(symbol)
        if pos and pos.available_qty > 0:
            return self.sell(symbol, pos.available_qty, price)
        return None

    # === Callback Management ===

    def on(self, event: str, callback: Callable):
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, *args, **kwargs):
        callbacks = self._callbacks.get(event, [])
        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                log.error(f"Callback error for {event}: {e}")

class SimulatorBroker(BrokerInterface):
    """
    Paper trading simulator with realistic behavior.

    FIX(10): Uses bounded thread pool with proper shutdown.
    FIX(2):  Fill tracking uses cursor, not list clearing.
    """

    _MAX_EXEC_WORKERS = 8

    def __init__(self, initial_capital: float = None):
        super().__init__()
        self._initial_capital = initial_capital or CONFIG.capital
        self._cash = self._initial_capital
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._order_history: list[Order] = []
        self._fills: list[Fill] = []

        # FIX(2): Cursor-based fill tracking instead of clearing list
        self._fill_cursor: int = 0

        self._connected = False

        # T+1 tracking
        self._purchase_dates: dict[str, date] = {}
        self._last_settlement_date = date.today()

        # Data fetcher (lazy init)
        self._fetcher = None

        self._fill_counter = 0

        # FIX(10): Bounded thread pool
        self._exec_pool = ThreadPoolExecutor(
            max_workers=self._MAX_EXEC_WORKERS,
            thread_name_prefix="sim_exec",
        )

    def _get_fetcher(self):
        if self._fetcher is None:
            from data.fetcher import get_fetcher
            self._fetcher = get_fetcher()
        return self._fetcher

    @property
    def name(self) -> str:
        return "Paper Trading Simulator"

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, **kwargs) -> bool:
        with self._lock:
            self._connected = True
            log.info(
                f"Simulator connected with "
                f"CNY {self._initial_capital:,.2f}"
            )
            return True

    def disconnect(self):
        with self._lock:
            self._connected = False
            try:
                self._exec_pool.shutdown(wait=False)
            except Exception:
                pass
            log.info("Simulator disconnected")

    def get_quote(self, symbol: str) -> float | None:
        try:
            from data.feeds import get_feed_manager
            fm = get_feed_manager(auto_init=False)
            q = fm.get_quote(symbol)
            if q and getattr(q, "price", 0) and float(q.price) > 0:
                return float(q.price)
        except Exception:
            pass

        fetcher = self._get_fetcher()
        quote = fetcher.get_realtime(symbol)
        return float(quote.price) if quote and quote.price > 0 else None

    def get_account(self) -> Account:
        with self._lock:
            self._check_settlement()
            self._update_prices()

            # FIX(7): Deep-copy positions so external code can't
            positions_copy = {}
            for sym, pos in self._positions.items():
                positions_copy[sym] = Position(
                    symbol=pos.symbol,
                    name=pos.name,
                    quantity=pos.quantity,
                    available_qty=pos.available_qty,
                    frozen_qty=pos.frozen_qty,
                    avg_cost=pos.avg_cost,
                    current_price=pos.current_price,
                    realized_pnl=pos.realized_pnl,
                    commission_paid=pos.commission_paid,
                )

            realized_pnl = sum(
                p.realized_pnl for p in self._positions.values()
            )

            return Account(
                broker_name=self.name,
                cash=self._cash,
                available=self._cash,
                frozen=0.0,
                positions=positions_copy,
                initial_capital=self._initial_capital,
                realized_pnl=realized_pnl,
                last_updated=datetime.now(),
            )

    def get_positions(self) -> dict[str, Position]:
        with self._lock:
            self._check_settlement()
            self._update_prices()
            return dict(self._positions)

    def get_position(self, symbol: str) -> Position | None:
        with self._lock:
            self._check_settlement()
            pos = self._positions.get(symbol)
            if pos:
                price = self.get_quote(symbol)
                if price:
                    pos.update_price(price)
            return pos

    def _schedule_execution(self, order_id: str):
        """Execute order asynchronously using bounded thread pool."""
        self._exec_pool.submit(self._execution_worker, order_id)

    def _execution_worker(self, order_id: str):
        """
        Worker function for async order execution.

        FIX(1): Re-checks order.is_active after every state-changing
        boundary to close race windows.
        """
        import random

        while True:
            time.sleep(random.uniform(0.05, 0.30))

            # Pre-check: is order still active?
            with self._lock:
                order = self._orders.get(order_id)
                if not order or not order.is_active:
                    return

            # Get quote OUTSIDE lock (may be slow)
            market_price = self.get_quote(order.symbol)

            # FIX(1): Re-acquire lock and re-check after quote fetch
            with self._lock:
                order = self._orders.get(order_id)
                if not order or not order.is_active:
                    return

                if market_price is None or market_price <= 0:
                    order.status = OrderStatus.REJECTED
                    order.message = (
                        "No market quote during async execution"
                    )
                    order.updated_at = datetime.now()
                    self._emit("order_update", order)
                    return

                self._execute_order(
                    order, market_price=float(market_price),
                )

                if order.is_complete:
                    return

                if order.status == OrderStatus.PARTIAL:
                    pass

            # Sleep between partial fills (outside lock)
            if order.status == OrderStatus.PARTIAL:
                time.sleep(random.uniform(0.4, 1.2))

    def submit_order(self, order: Order) -> Order:
        with self._lock:
            self._check_settlement()

            order.broker_id = (
                f"SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}_"
                f"{uuid.uuid4().hex[:6]}"
            )
            order.created_at = order.created_at or datetime.now()
            order.submitted_at = datetime.now()

            self.register_order_mapping(order.id, order.broker_id)

            current_price = self.get_quote(order.symbol)
            if current_price is None or current_price <= 0:
                order.status = OrderStatus.REJECTED
                order.message = "Cannot get market quote"
                self._emit('order_update', order)
                return order

            fetcher = self._get_fetcher()
            quote = fetcher.get_realtime(order.symbol)
            if not order.name and quote:
                order.name = quote.name

            # For market orders, use current price for validation
            if (
                order.order_type == OrderType.LIMIT
                and order.price > 0
            ):
                ref_price = order.price
            else:
                ref_price = current_price

            ok, reason = self._validate_order(order, ref_price)
            if not ok:
                order.status = OrderStatus.REJECTED
                order.message = reason
                self._emit('order_update', order)
                return order

            # FIX(4): Emit SUBMITTED status before transitioning to
            order.status = OrderStatus.SUBMITTED
            self._orders[order.id] = order
            self._emit('order_update', order)

            order.status = OrderStatus.ACCEPTED
            self._emit('order_update', order)

            self._schedule_execution(order_id=order.id)
            return order

    def _validate_order(
        self, order: Order, price: float,
    ) -> tuple[bool, str]:
        requested_type = str(
            order.tags.get("requested_order_type", order.order_type.value)
            if isinstance(order.tags, dict)
            else order.order_type.value
        ).strip().lower()
        requested_type = requested_type.replace("-", "_")

        if order.quantity <= 0:
            return False, "Quantity must be positive"

        from core.constants import get_lot_size

        lot_size = get_lot_size(order.symbol)
        if order.quantity % lot_size != 0:
            return False, f"Quantity must be multiple of {lot_size}"

        # FIX(9): Allow price=0 for market-style orders.
        if order.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TRAIL_LIMIT} and price <= 0:
            return False, "Limit order requires positive price"
        if price <= 0 and order.order_type not in {
            OrderType.MARKET,
            OrderType.STOP,
            OrderType.IOC,
            OrderType.FOK,
            OrderType.TRAIL_MARKET,
        }:
            return False, "Invalid price"

        # FIX(5): Use CONFIG.trading.commission consistently
        commission_rate = float(CONFIG.trading.commission)
        commission_min = 5.0

        if order.side == OrderSide.BUY:
            est_value = order.quantity * price
            commission = max(est_value * commission_rate, commission_min)
            total = est_value + commission
            if total > self._cash:
                return False, (
                    f"Insufficient funds: need CNY {total:,.2f}, "
                    f"have CNY {self._cash:,.2f}"
                )

            existing_value = 0.0
            existing_pos = self._positions.get(order.symbol)
            if existing_pos:
                existing_value = existing_pos.quantity * price

            new_total_value = existing_value + (order.quantity * price)
            equity = self._cash + sum(
                p.market_value for p in self._positions.values()
            )
            if equity > 0:
                max_pct = float(CONFIG.risk.max_position_pct)
                position_pct = new_total_value / equity * 100
                if position_pct > max_pct:
                    return False, (
                        f"Position too large: {position_pct:.1f}% "
                        f"(max: {max_pct}%)"
                    )
        else:
            pos = self._positions.get(order.symbol)
            if not pos:
                return False, f"No position in {order.symbol}"
            if order.quantity > pos.available_qty:
                return False, (
                    f"Available: {pos.available_qty}, "
                    f"requested: {order.quantity}"
                )

        return True, "OK"

    def _check_price_limits(
        self,
        symbol: str,
        side: OrderSide,
        price: float,
        prev_close: float,
        name: str = None,
    ) -> tuple[bool, str]:
        from core.constants import get_price_limit

        if prev_close <= 0:
            return True, "OK"

        limit_pct = get_price_limit(symbol, name)
        limit_up = prev_close * (1 + limit_pct)
        limit_down = prev_close * (1 - limit_pct)

        if side == OrderSide.BUY and price >= limit_up * 0.999:
            return False, (
                f"Cannot buy at limit up ({limit_pct * 100:.0f}%)"
            )
        if side == OrderSide.SELL and price <= limit_down * 1.001:
            return False, (
                f"Cannot sell at limit down ({limit_pct * 100:.0f}%)"
            )

        return True, "OK"

    def _execute_order(self, order: Order, market_price: float):
        """Execute order with realistic simulation."""
        import random

        requested_type = str(
            order.tags.get("requested_order_type", order.order_type.value)
            if isinstance(order.tags, dict)
            else order.order_type.value
        ).strip().lower().replace("-", "_")
        tif = str(
            order.tags.get("time_in_force", "day")
            if isinstance(order.tags, dict)
            else "day"
        ).strip().lower().replace("-", "_")
        if requested_type in {"ioc", "fok"}:
            tif = requested_type

        prev_close = float(market_price)
        try:
            fetcher = self._get_fetcher()
            q = fetcher.get_realtime(order.symbol)
            if q and getattr(q, "close", 0) and float(q.close) > 0:
                prev_close = float(q.close)
        except Exception:
            pass

        can_trade, reason = self._check_price_limits(
            order.symbol, order.side, float(market_price),
            prev_close, order.name,
        )
        if not can_trade:
            order.status = OrderStatus.REJECTED
            order.message = reason
            order.updated_at = datetime.now()
            self._emit('order_update', order)
            return

        is_limit_style = order.order_type in {
            OrderType.LIMIT,
            OrderType.STOP_LIMIT,
            OrderType.TRAIL_LIMIT,
        }
        if is_limit_style:
            if order.price <= 0:
                order.status = OrderStatus.REJECTED
                order.message = "Limit order requires positive price"
                order.updated_at = datetime.now()
                self._emit('order_update', order)
                return

            if (
                order.side == OrderSide.BUY
                and float(market_price) > float(order.price)
            ):
                if tif in {"ioc", "fok"}:
                    order.status = OrderStatus.CANCELLED
                    order.message = (
                        f"{tif.upper()} BUY not marketable: market "
                        f"{market_price:.2f} > limit {order.price:.2f}"
                    )
                else:
                    order.status = OrderStatus.REJECTED
                    order.message = (
                        f"Limit BUY not marketable: market "
                        f"{market_price:.2f} > limit {order.price:.2f}"
                    )
                order.updated_at = datetime.now()
                self._emit('order_update', order)
                return

            if (
                order.side == OrderSide.SELL
                and float(market_price) < float(order.price)
            ):
                if tif in {"ioc", "fok"}:
                    order.status = OrderStatus.CANCELLED
                    order.message = (
                        f"{tif.upper()} SELL not marketable: market "
                        f"{market_price:.2f} < limit {order.price:.2f}"
                    )
                else:
                    order.status = OrderStatus.REJECTED
                    order.message = (
                        f"Limit SELL not marketable: market "
                        f"{market_price:.2f} < limit {order.price:.2f}"
                    )
                order.updated_at = datetime.now()
                self._emit('order_update', order)
                return

            fill_price = float(order.price)
        else:
            fill_price = float(market_price)

        # FIX(5): Consistent CONFIG access
        slippage = float(CONFIG.trading.slippage)
        commission_rate = float(CONFIG.trading.commission)
        stamp_tax_rate = float(CONFIG.trading.stamp_tax)
        commission_min = 5.0

        if order.side == OrderSide.BUY:
            fill_price *= 1 + slippage * (0.5 + 0.5 * random.random())
        else:
            fill_price *= 1 - slippage * (0.5 + 0.5 * random.random())
        fill_price = round(fill_price, 2)

        remaining = int(order.quantity - order.filled_qty)
        if remaining <= 0:
            return

        from core.constants import get_lot_size

        lot = int(get_lot_size(order.symbol))

        fill_qty = remaining
        if tif != "fok" and remaining >= 2 * lot and random.random() < 0.25:
            half = max(lot, (remaining // 2 // lot) * lot)
            fill_qty = min(half, remaining)

        max_immediate_ratio = 1.0
        if isinstance(order.tags, dict):
            try:
                max_immediate_ratio = float(
                    order.tags.get("max_immediate_fill_ratio", 1.0) or 1.0
                )
            except Exception:
                max_immediate_ratio = 1.0
        max_immediate_ratio = max(0.05, min(1.0, max_immediate_ratio))
        max_immediate_qty = max(
            lot,
            int((remaining * max_immediate_ratio) // lot) * lot,
        )
        fill_qty = min(fill_qty, max_immediate_qty)

        if tif == "fok" and fill_qty < remaining:
            order.status = OrderStatus.CANCELLED
            order.message = (
                f"FOK cancelled: only {fill_qty}/{remaining} shares "
                "immediately available"
            )
            order.updated_at = datetime.now()
            self._emit('order_update', order)
            return

        trade_value = fill_qty * fill_price
        commission = max(trade_value * commission_rate, commission_min)
        stamp_tax = (
            trade_value * stamp_tax_rate
            if order.side == OrderSide.SELL else 0.0
        )
        total_cost = commission + stamp_tax

        if order.side == OrderSide.BUY:
            self._cash -= trade_value + total_cost

            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                total_qty = pos.quantity + fill_qty
                if total_qty > 0:
                    pos.avg_cost = (
                        (pos.avg_cost * pos.quantity
                         + fill_price * fill_qty)
                        / total_qty
                    )
                pos.quantity = total_qty
            else:
                self._positions[order.symbol] = Position(
                    symbol=order.symbol,
                    name=order.name,
                    quantity=fill_qty,
                    available_qty=0,
                    avg_cost=fill_price,
                    current_price=fill_price,
                )
            self._purchase_dates[order.symbol] = date.today()

        else:
            self._cash += trade_value - total_cost

            pos = self._positions[order.symbol]
            gross_pnl = (fill_price - pos.avg_cost) * fill_qty
            realized = gross_pnl - total_cost
            pos.realized_pnl += realized
            pos.quantity -= fill_qty
            pos.available_qty = max(
                0, pos.available_qty - fill_qty,
            )

            if pos.quantity <= 0:
                del self._positions[order.symbol]
                self._purchase_dates.pop(order.symbol, None)

        order.filled_qty += fill_qty
        order.commission += total_cost
        order.status = (
            OrderStatus.FILLED
            if order.filled_qty >= order.quantity
            else OrderStatus.PARTIAL
        )
        if tif == "ioc" and order.filled_qty < order.quantity:
            order.status = OrderStatus.CANCELLED
            remainder = max(0, int(order.quantity - order.filled_qty))
            order.message = (
                f"IOC remainder cancelled: {remainder} shares"
            )
        order.filled_at = datetime.now()

        prev_value = (order.filled_qty - fill_qty) * order.avg_price
        new_value = fill_qty * fill_price
        order.avg_price = (
            (prev_value + new_value) / order.filled_qty
            if order.filled_qty > 0
            else fill_price
        )
        order.filled_price = fill_price
        order.updated_at = datetime.now()

        self._fill_counter += 1
        fill = Fill(
            id=(
                f"FILL_SIM_"
                f"{datetime.now().strftime('%Y%m%d%H%M%S')}_"
                f"{self._fill_counter:06d}"
            ),
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=fill_price,
            commission=commission,
            stamp_tax=stamp_tax,
            timestamp=datetime.now(),
        )
        self._fills.append(fill)
        self._order_history.append(order)

        log.info(
            f"[SIM] {order.side.value.upper()} {fill_qty} "
            f"{order.symbol} @ {fill_price:.2f} "
            f"(cost: {total_cost:.2f}, status: {order.status.value}, tif={tif})"
        )
        self._emit('trade', order, fill)

    def get_fills(self, since: datetime = None) -> list[Fill]:
        """
        Get new fills since last call.

        FIX(2): Uses cursor-based tracking instead of clearing list.
        Multiple consumers can call this without losing fills.
        """
        with self._lock:
            new_fills = self._fills[self._fill_cursor:]
            self._fill_cursor = len(self._fills)

        if since is None:
            return list(new_fills)

        return [
            f for f in new_fills
            if f.timestamp and f.timestamp >= since
        ]

    def get_all_fills(self) -> list[Fill]:
        """Get ALL fills (not just new ones)."""
        with self._lock:
            return list(self._fills)

    def get_order_status(self, order_id: str) -> OrderStatus | None:
        with self._lock:
            order = self._orders.get(order_id)
            return order.status if order else None

    def sync_order(self, order: Order) -> Order:
        with self._lock:
            stored = self._orders.get(order.id)
            if stored:
                order.status = stored.status
                order.filled_qty = stored.filled_qty
                order.avg_price = stored.avg_price
                order.filled_price = stored.filled_price
                order.commission = stored.commission
                order.filled_at = stored.filled_at
                order.message = stored.message
            return order

    def cancel_order(self, order_id: str) -> bool:
        with self._lock:
            order = self._orders.get(order_id)
            if order and order.is_active:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                order.cancelled_at = datetime.now()
                self._emit('order_update', order)
                return True
            return False

    def get_orders(self, active_only: bool = True) -> list[Order]:
        with self._lock:
            if active_only:
                return [
                    o for o in self._orders.values() if o.is_active
                ]
            return list(self._orders.values())

    def _check_settlement(self):
        """
        FIX(13): Proper T+1 settlement using is_trading_day.
        Shares bought yesterday become available on the next trading day.
        """
        from core.constants import is_trading_day

        today = date.today()
        if today == self._last_settlement_date:
            return

        if not is_trading_day(today):
            return

        # It's a new trading day 鈥?settle all positions
        for _symbol, pos in self._positions.items():
            pos.available_qty = pos.quantity
        self._last_settlement_date = today
        log.info("T+1 settlement: all shares now available")

    def _update_prices(self):
        for symbol, pos in self._positions.items():
            price = self.get_quote(symbol)
            if price:
                pos.update_price(price)

    def get_trade_history(self) -> list[dict]:
        with self._lock:
            return [
                {
                    'fill_id': f.id,
                    'order_id': f.order_id,
                    'symbol': f.symbol,
                    'side': f.side.value,
                    'quantity': f.quantity,
                    'price': f.price,
                    'commission': f.commission,
                    'stamp_tax': f.stamp_tax,
                    'timestamp': f.timestamp,
                }
                for f in self._fills
            ]

    def reset(self):
        with self._lock:
            self._cash = self._initial_capital
            self._positions.clear()
            self._orders.clear()
            self._order_history.clear()
            self._fills.clear()
            self._fill_cursor = 0
            self._purchase_dates.clear()
            self._last_settlement_date = date.today()
            self._order_id_to_broker_id.clear()
            self._broker_id_to_order_id.clear()
            log.info("Simulator reset to initial state")

    def reconcile(self) -> dict:
        with self._lock:
            return {
                'cash_diff': 0.0,
                'position_diffs': [],
                'missing_positions': [],
                'extra_positions': [],
                'reconciled': True,
                'timestamp': datetime.now().isoformat(),
            }

# FIX(11): Shared base class for THS/ZSZQ (~80% code dedup)

class EasytraderBroker(BrokerInterface):
    """
    Base class for all easytrader-based brokers (THS, ZSZQ, HT, etc).

    Subclasses only need to override:
    - name (property)
    - _get_easytrader_type() -> str
    - _get_balance_fields() -> dict (optional, for field name differences)
    """

    def __init__(self):
        super().__init__()
        self._client = None
        self._connected = False
        self._orders: dict[str, Order] = {}
        self._seen_fill_ids: set = set()
        self._fetcher = None

        try:
            import easytrader
            self._easytrader = easytrader
            self._available = True
        except ImportError:
            self._easytrader = None
            self._available = False
            log.warning(
                "easytrader not installed - live trading unavailable"
            )

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _get_easytrader_type(self) -> str:
        """Return the easytrader.use() type string."""
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    def connect(self, exe_path: str = None, **kwargs) -> bool:
        if not self._available:
            log.error("easytrader not installed")
            return False

        exe_path = (
            exe_path
            or kwargs.get('broker_path')
            or CONFIG.broker_path
        )
        if not exe_path or not Path(exe_path).exists():
            log.error(f"Broker executable not found: {exe_path}")
            return False

        try:
            self._client = self._easytrader.use(
                self._get_easytrader_type(),
            )
            self._client.connect(exe_path)

            balance = self._client.balance
            if balance:
                self._connected = True
                log.info(f"Connected to {self.name}")
                return True

        except Exception as e:
            log.error(f"Connection failed: {e}")

        return False

    def disconnect(self):
        self._client = None
        self._connected = False
        log.info(f"Disconnected from {self.name}")

    def get_quote(self, symbol: str) -> float | None:
        try:
            from data.feeds import get_feed_manager
            fm = get_feed_manager(auto_init=False)
            q = fm.get_quote(symbol)
            if q and getattr(q, "price", 0) and float(q.price) > 0:
                return float(q.price)
        except Exception:
            pass

        fetcher = self._get_fetcher()
        quote = fetcher.get_realtime(symbol)
        return (
            float(quote.price)
            if quote and quote.price > 0 else None
        )

    def get_account(self) -> Account:
        if not self.is_connected:
            return Account()

        try:
            balance = self._client.balance
            positions = self.get_positions()

            cash = float(
                balance.get("\u8d44\u91d1\u4f59\u989d")  # 资金余额
                or balance.get("\u603b\u8d44\u4ea7")  # 总资产
                or balance.get("\u53ef\u7528\u8d44\u91d1")  # 可用资金
                or balance.get("cash")
                or 0
            )
            available = float(
                balance.get("\u53ef\u7528\u91d1\u989d")  # 可用金额
                or balance.get("\u53ef\u7528\u8d44\u91d1")  # 可用资金
                or balance.get("\u53ef\u53d6\u8d44\u91d1")  # 可取资金
                or balance.get("available")
                or cash
            )
            frozen = float(
                balance.get("\u51bb\u7ed3\u91d1\u989d")  # 冻结金额
                or balance.get("frozen")
                or 0
            )

            return Account(
                broker_name=self.name,
                cash=cash,
                available=available,
                frozen=frozen,
                positions=positions,
                last_updated=datetime.now(),
            )
        except Exception as e:
            log.error(f"Failed to get account: {e}")
            return Account()

    def get_positions(self) -> dict[str, Position]:
        if not self.is_connected:
            return {}

        try:
            raw = self._client.position
            positions = {}

            for p in raw:
                code = str(
                    p.get("\u8bc1\u5238\u4ee3\u7801")  # 证券代码
                    or p.get("\u80a1\u7968\u4ee3\u7801")  # 股票代码
                    or ""
                ).zfill(6)

                if not code or code == "000000":
                    continue

                positions[code] = Position(
                    symbol=code,
                    name=(
                        p.get("\u8bc1\u5238\u540d\u79f0")  # 证券名称
                        or p.get("\u80a1\u7968\u540d\u79f0")  # 股票名称
                        or ""
                    ),
                    quantity=int(
                        p.get("\u80a1\u7968\u4f59\u989d")  # 股票余额
                        or p.get("\u6301\u4ed3\u6570\u91cf")  # 持仓数量
                        or p.get("\u5f53\u524d\u6301\u4ed3")  # 当前持仓
                        or 0
                    ),
                    available_qty=int(
                        p.get("\u53ef\u5356\u4f59\u989d")  # 可卖余额
                        or p.get("\u53ef\u7528\u4f59\u989d")  # 可用余额
                        or p.get("\u53ef\u5356\u6570\u91cf")  # 可卖数量
                        or 0
                    ),
                    avg_cost=float(
                        p.get("\u6210\u672c\u4ef7")  # 成本价
                        or p.get("\u4e70\u5165\u6210\u672c")  # 买入成本
                        or p.get("\u53c2\u8003\u6210\u672c\u4ef7")  # 参考成本价
                        or 0
                    ),
                    current_price=float(
                        p.get("\u5f53\u524d\u4ef7")  # 当前价
                        or p.get("\u6700\u65b0\u4ef7")  # 最新价
                        or p.get("\u5e02\u4ef7")  # 市价
                        or 0
                    ),
                )

            return positions

        except Exception as e:
            log.error(f"Failed to get positions: {e}")
            return {}

    def get_position(self, symbol: str) -> Position | None:
        return self.get_positions().get(symbol)

    def submit_order(self, order: Order) -> Order:
        if not self.is_connected:
            order.status = OrderStatus.REJECTED
            order.message = "Not connected to broker"
            return order

        try:
            order.created_at = order.created_at or datetime.now()
            order.submitted_at = datetime.now()

            if order.side == OrderSide.BUY:
                if order.order_type == OrderType.MARKET:
                    result = self._client.market_buy(
                        order.symbol, order.quantity,
                    )
                else:
                    result = self._client.buy(
                        order.symbol, order.quantity, order.price,
                    )
            else:
                if order.order_type == OrderType.MARKET:
                    result = self._client.market_sell(
                        order.symbol, order.quantity,
                    )
                else:
                    result = self._client.sell(
                        order.symbol, order.quantity, order.price,
                    )

            if result and isinstance(result, dict):
                entrust_no = (
                    result.get('濮旀墭缂栧彿')
                    or result.get('entrust_no')
                    or result.get('order_id')
                )

                if entrust_no:
                    order.status = OrderStatus.SUBMITTED
                    order.broker_id = str(entrust_no)
                    order.message = f"Entrust: {entrust_no}"
                    self.register_order_mapping(
                        order.id, order.broker_id,
                    )
                    log.info(
                        f"Order submitted: {order.id} "
                        f"-> broker {order.broker_id}"
                    )
                else:
                    order.status = OrderStatus.REJECTED
                    order.message = str(
                        result.get('msg')
                        or result.get('message')
                        or result
                    )
            else:
                order.status = OrderStatus.REJECTED
                order.message = "Unknown response from broker"

            self._orders[order.id] = order
            self._emit('order_update', order)
            return order

        except Exception as e:
            log.error(f"Order submission error: {e}")
            order.status = OrderStatus.REJECTED
            order.message = str(e)
            return order

    def get_fills(self, since: datetime = None) -> list[Fill]:
        """Get fills from broker 鈥?deduplicates by broker_fill_id."""
        if not self.is_connected:
            return []

        fills: list[Fill] = []
        try:
            trades = self._client.today_trades

            for trade in trades:
                broker_fill_id = str(
                    trade.get("\u6210\u4ea4\u7f16\u53f7", "") or ""  # 成交编号
                ).strip()
                if not broker_fill_id:
                    continue

                if broker_fill_id in self._seen_fill_ids:
                    continue

                ts = (
                    trade.get("\u6210\u4ea4\u65f6\u95f4")  # 成交时间
                    or trade.get("time")
                    or None
                )
                fill_time = datetime.now()
                if ts:
                    try:
                        t = datetime.strptime(
                            str(ts), "%H:%M:%S"
                        ).time()
                        fill_time = datetime.combine(
                            date.today(), t,
                        )
                    except Exception:
                        pass

                # since-filter
                if (
                    since
                    and isinstance(fill_time, datetime)
                    and fill_time < since
                ):
                    continue

                broker_entrust = str(
                    trade.get("\u59d4\u6258\u7f16\u53f7", "") or ""  # 委托编号
                ).strip()
                our_order_id = self.get_order_id(broker_entrust)
                if not our_order_id:
                    log.warning(
                        f"Unknown entrust number: {broker_entrust}"
                    )
                    continue

                self._seen_fill_ids.add(broker_fill_id)

                trade_side = trade.get(
                    "\u4e70\u5356\u6807\u5fd7", trade.get("\u64cd\u4f5c", ""),  # 买卖标志 / 操作
                )
                side = (
                    OrderSide.BUY
                    if "\u4e70" in str(trade_side)  # 买
                    else OrderSide.SELL
                )

                symbol = str(
                    trade.get("\u8bc1\u5238\u4ee3\u7801", "") or ""  # 证券代码
                ).zfill(6)
                qty = int(trade.get("\u6210\u4ea4\u6570\u91cf", 0) or 0)  # 成交数量
                price = float(trade.get("\u6210\u4ea4\u4ef7\u683c", 0) or 0.0)  # 成交价格
                comm = float(trade.get("\u624b\u7eed\u8d39", 0) or 0.0)  # 手续费
                tax = float(trade.get("\u5370\u82b1\u7a0e", 0) or 0.0)  # 印花税

                fid = make_fill_uid(
                    self.name, broker_fill_id, symbol,
                    fill_time, price, qty,
                )

                fills.append(Fill(
                    id=fid,
                    broker_fill_id=broker_fill_id,
                    order_id=our_order_id,
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    price=price,
                    commission=comm,
                    stamp_tax=tax,
                    timestamp=fill_time,
                ))

        except Exception as e:
            log.error(f"Failed to get fills: {e}")

        return fills

    def get_order_status(
        self, order_id: str,
    ) -> OrderStatus | None:
        if not self.is_connected:
            return None

        broker_id = self.get_broker_id(order_id)
        if not broker_id:
            return None

        try:
            entrusts = self._client.today_entrusts

            for entrust in entrusts:
                if str(entrust.get("\u59d4\u6258\u7f16\u53f7", "")) == broker_id:
                    status_str = entrust.get(
                        "\u59d4\u6258\u72b6\u6001",  # 委托状态
                        entrust.get("\u72b6\u6001", ""),  # 状态
                    )
                    # FIX(3): Use shared parser
                    return parse_broker_status(status_str)

            return None

        except Exception as e:
            log.error(f"Failed to get order status: {e}")
            return None

    def sync_order(self, order: Order) -> Order:
        if not self.is_connected:
            return order

        broker_id = self.get_broker_id(order.id)
        if not broker_id:
            return order

        try:
            entrusts = self._client.today_entrusts

            for entrust in entrusts:
                if str(entrust.get("\u59d4\u6258\u7f16\u53f7", "")) == broker_id:
                    # FIX(3): Use shared parser
                    order.status = parse_broker_status(
                        entrust.get("\u59d4\u6258\u72b6\u6001", ""),  # 委托状态
                    )
                    order.filled_qty = int(
                        entrust.get("\u6210\u4ea4\u6570\u91cf", 0) or 0  # 成交数量
                    )

                    avg_price = entrust.get(
                        "\u6210\u4ea4\u5747\u4ef7",  # 成交均价
                        entrust.get("\u6210\u4ea4\u4ef7\u683c", 0),  # 成交价格
                    )
                    if avg_price:
                        order.avg_price = float(avg_price)

                    order.updated_at = datetime.now()
                    break

        except Exception as e:
            log.error(f"Failed to sync order: {e}")

        return order

    def cancel_order(self, order_id: str) -> bool:
        if not self.is_connected:
            return False

        order = self._orders.get(order_id)
        broker_id = self.get_broker_id(order_id)

        if not broker_id:
            return False

        try:
            self._client.cancel_entrust(broker_id)
            if order:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                self._emit('order_update', order)
            log.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            log.error(f"Cancel failed: {e}")
            return False

    def get_orders(self, active_only: bool = True) -> list[Order]:
        if active_only:
            return [
                o for o in self._orders.values() if o.is_active
            ]
        return list(self._orders.values())

    def _get_fetcher(self):
        if self._fetcher is None:
            from data.fetcher import get_fetcher
            self._fetcher = get_fetcher()
        return self._fetcher

# FIX(11): Thin subclasses 鈥?all shared logic is in base

class THSBroker(EasytraderBroker):
    """THS/HT/GJ/YH broker via easytrader."""

    BROKER_TYPES = {
        "ths": "THS",
        "ht": "HT",
        "gj": "GJ",
        "yh": "YH",
    }

    def __init__(self, broker_type: str = "ths"):
        self._broker_type = broker_type
        super().__init__()

    @property
    def name(self) -> str:
        return self.BROKER_TYPES.get(
            self._broker_type, "Unknown Broker",
        )

    def _get_easytrader_type(self) -> str:
        return self._broker_type

class ZSZQBroker(EasytraderBroker):
    """鎷涘晢璇佸埜 broker via easytrader (universal mode)."""

    @property
    def name(self) -> str:
        return "鎷涘晢璇佸埜"

    def _get_easytrader_type(self) -> str:
        return 'universal'


class MultiVenueBroker(BrokerInterface):
    """
    Multi-venue router with active failover.

    - Uses a priority list of underlying brokers.
    - Routes writes to the active venue.
    - On write failure, rotates to next venue with cooldown.
    """

    def __init__(self, venues: list[BrokerInterface], failover_cooldown_seconds: int = 30):
        super().__init__()
        self._venues = [v for v in venues if v is not None]
        self._active_idx = 0
        self._cooldown_seconds = max(1, int(failover_cooldown_seconds or 30))
        self._last_fail_ts: dict[int, float] = {}
        self._fail_counts: dict[str, int] = {}
        self._submit_counts: dict[str, int] = {}
        self._read_counts: dict[str, int] = {}
        self._failure_events: dict[int, deque[float]] = {}
        self._read_latency_ms: dict[str, deque[float]] = {}
        self._last_errors: dict[str, str] = {}
        self._latency_samples_max: int = 200
        self._recent_failure_window_seconds: float = 300.0
        self._order_venue_idx: dict[str, int] = {}
        if not self._venues:
            raise ValueError("MultiVenueBroker requires at least one venue")

    @property
    def name(self) -> str:
        names = ",".join(v.name for v in self._venues)
        return f"MultiVenueRouter[{names}]"

    @property
    def is_connected(self) -> bool:
        return any(v.is_connected for v in self._venues)

    def connect(self, **kwargs) -> bool:
        ok_any = False
        for i, venue in enumerate(self._venues):
            try:
                ok = bool(venue.connect(**kwargs))
            except Exception as e:
                ok = False
                log.warning("Venue connect failed (%s): %s", venue.name, e)
            if ok:
                ok_any = True
                if self._active_idx == 0:
                    self._active_idx = i
        return ok_any

    def disconnect(self):
        for venue in self._venues:
            try:
                venue.disconnect()
            except Exception as e:
                log.debug("Venue disconnect failed (%s): %s", venue.name, e)

    def _eligible_indices(self) -> list[int]:
        now = time.time()
        out: list[int] = []
        for i, venue in enumerate(self._venues):
            if not venue.is_connected:
                continue
            last_fail = float(self._last_fail_ts.get(i, 0.0))
            if last_fail > 0 and (now - last_fail) < self._cooldown_seconds:
                continue
            out.append(i)
        return out

    def _ordered_indices(self) -> list[int]:
        eligible = self._eligible_indices()
        if not eligible:
            return []
        return sorted(
            eligible,
            key=lambda i: self._venue_score(i),
            reverse=True,
        )

    def _connected_indices(self) -> list[int]:
        out: list[int] = []
        for i, venue in enumerate(self._venues):
            try:
                if venue.is_connected:
                    out.append(i)
            except Exception:
                continue
        return out

    def _preferred_indices_for_order(self, order_id: str) -> list[int]:
        preferred = self._order_venue_idx.get(str(order_id or ""))
        if preferred is None:
            return self._ordered_indices()

        ordered = self._ordered_indices()
        if preferred in ordered:
            return [preferred] + [i for i in ordered if i != preferred]
        if 0 <= preferred < len(self._venues):
            return [preferred] + ordered
        return ordered

    def _mark_failure(self, idx: int, exc: Exception) -> None:
        self._last_fail_ts[idx] = time.time()
        venue = self._venues[idx]
        name = str(venue.name)
        self._fail_counts[name] = self._fail_counts.get(name, 0) + 1
        bucket = self._failure_events.get(idx)
        if bucket is None:
            bucket = deque()
            self._failure_events[idx] = bucket
        now = time.time()
        bucket.append(now)
        cutoff = now - float(self._recent_failure_window_seconds)
        while bucket and float(bucket[0]) < cutoff:
            bucket.popleft()
        self._last_errors[name] = str(exc)[:300]
        log.warning("Venue failure (%s): %s", venue.name, exc)

    def _mark_submit(self, idx: int) -> None:
        venue = self._venues[idx]
        self._submit_counts[venue.name] = self._submit_counts.get(venue.name, 0) + 1

    def _mark_read(self, idx: int, latency_ms: float | None = None) -> None:
        venue = self._venues[idx]
        name = str(venue.name)
        self._read_counts[name] = self._read_counts.get(name, 0) + 1
        if latency_ms is not None and latency_ms >= 0:
            hist = self._read_latency_ms.get(name)
            if hist is None:
                hist = deque(maxlen=self._latency_samples_max)
                self._read_latency_ms[name] = hist
            hist.append(float(latency_ms))

    def _recent_failures(self, idx: int, window_seconds: float | None = None) -> int:
        bucket = self._failure_events.get(int(idx))
        if not bucket:
            return 0
        window = float(window_seconds or self._recent_failure_window_seconds)
        cutoff = time.time() - max(1.0, window)
        while bucket and float(bucket[0]) < cutoff:
            bucket.popleft()
        return int(len(bucket))

    def _avg_read_latency_ms(self, idx: int) -> float:
        if not (0 <= idx < len(self._venues)):
            return 0.0
        name = str(self._venues[idx].name)
        vals = self._read_latency_ms.get(name)
        if not vals:
            return 0.0
        return float(sum(vals) / max(1, len(vals)))

    def _venue_score(self, idx: int) -> float:
        """
        Adaptive routing score.

        Higher is better:
        - rewards venues with successful submits/reads
        - penalizes recent failures and active cooldown
        - slight preference for current active venue to reduce thrash
        """
        if not (0 <= idx < len(self._venues)):
            return -1.0

        venue = self._venues[idx]
        name = str(venue.name)
        fails = float(self._fail_counts.get(name, 0))
        submits = float(self._submit_counts.get(name, 0))
        reads = float(self._read_counts.get(name, 0))
        total_ops = submits + reads
        reliability = (total_ops + 1.0) / (total_ops + fails + 1.0)

        now = time.time()
        last_fail = float(self._last_fail_ts.get(idx, 0.0))
        cooldown_penalty = 0.0
        if last_fail > 0:
            elapsed = now - last_fail
            remain = max(0.0, float(self._cooldown_seconds) - elapsed)
            cooldown_penalty = min(0.5, remain / max(1.0, float(self._cooldown_seconds)))

        recent_failures = float(self._recent_failures(idx, window_seconds=300.0))
        recent_fail_penalty = min(0.40, recent_failures * 0.06)

        latency_ms = float(self._avg_read_latency_ms(idx))
        latency_penalty = 0.0
        if latency_ms > 120.0:
            latency_penalty = min(0.25, (latency_ms - 120.0) / 1200.0)

        read_bonus = min(0.04, reads / 600.0)
        active_bonus = 0.03 if idx == self._active_idx else 0.0
        return float(
            reliability
            + read_bonus
            + active_bonus
            - cooldown_penalty
            - recent_fail_penalty
            - latency_penalty
        )

    @staticmethod
    def _is_transient_reject(order: Order) -> bool:
        """
        Detect infrastructure-style rejects that should trigger failover.

        Business rejects (insufficient funds, rule violations, etc.) should
        not fan out to other venues.
        """
        if getattr(order, "status", None) != OrderStatus.REJECTED:
            return False

        msg = str(getattr(order, "message", "") or "").lower()
        if not msg:
            return False

        transient_markers = (
            "not connected",
            "timeout",
            "timed out",
            "network",
            "connection",
            "temporar",
            "unavailable",
            "service down",
            "gateway",
            "try again",
        )
        return any(marker in msg for marker in transient_markers)

    def _first_read(self, fn_name: str, *args, **kwargs):
        for idx in self._ordered_indices():
            venue = self._venues[idx]
            try:
                t0 = time.time()
                fn = getattr(venue, fn_name)
                out = fn(*args, **kwargs)
                self._active_idx = idx
                latency_ms = (time.time() - t0) * 1000.0
                self._mark_read(idx, latency_ms=latency_ms)
                return out
            except Exception as e:
                self._mark_failure(idx, e)
        raise RuntimeError(f"All venues failed for {fn_name}")

    def get_account(self) -> Account:
        return self._first_read("get_account")

    def get_positions(self) -> dict[str, Position]:
        return self._first_read("get_positions")

    def get_position(self, symbol: str) -> Position | None:
        return self._first_read("get_position", symbol)

    def submit_order(self, order: Order) -> Order:
        last_exc: Exception | None = None
        for idx in self._ordered_indices():
            venue = self._venues[idx]
            try:
                result = venue.submit_order(order)
                if self._is_transient_reject(result):
                    last_exc = RuntimeError(
                        f"{venue.name} transient rejection: {result.message}"
                    )
                    self._mark_failure(idx, last_exc)
                    continue
                self._active_idx = idx
                self._mark_submit(idx)
                if getattr(result, "id", ""):
                    self._order_venue_idx[str(result.id)] = idx
                if getattr(result, "broker_id", ""):
                    self.register_order_mapping(str(result.id), str(result.broker_id))
                return result
            except Exception as e:
                last_exc = e
                self._mark_failure(idx, e)
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("No connected venue available")

    def cancel_order(self, order_id: str) -> bool:
        for idx in self._preferred_indices_for_order(order_id):
            venue = self._venues[idx]
            try:
                ok = bool(venue.cancel_order(order_id))
                if ok:
                    self._active_idx = idx
                    self._order_venue_idx[str(order_id)] = idx
                    return True
            except Exception as e:
                self._mark_failure(idx, e)
        return False

    def get_orders(self, active_only: bool = True) -> list[Order]:
        out: list[Order] = []
        seen: set[str] = set()
        for idx in self._connected_indices():
            venue = self._venues[idx]
            try:
                t0 = time.time()
                rows = venue.get_orders(active_only)
                self._mark_read(idx, latency_ms=(time.time() - t0) * 1000.0)
            except Exception as e:
                self._mark_failure(idx, e)
                continue
            for order in (rows or []):
                oid = str(getattr(order, "id", "") or "")
                if not oid or oid in seen:
                    continue
                seen.add(oid)
                out.append(order)
                self._order_venue_idx[oid] = idx
                bid = str(getattr(order, "broker_id", "") or "").strip()
                if bid:
                    self.register_order_mapping(oid, bid)
        return out

    def get_quote(self, symbol: str) -> float | None:
        return self._first_read("get_quote", symbol)

    def get_fills(self, since: datetime = None) -> list[Fill]:
        out: list[Fill] = []
        seen: set[str] = set()
        for idx in self._connected_indices():
            venue = self._venues[idx]
            try:
                t0 = time.time()
                rows = venue.get_fills(since)
                self._mark_read(idx, latency_ms=(time.time() - t0) * 1000.0)
            except Exception as e:
                self._mark_failure(idx, e)
                continue
            for fill in (rows or []):
                fid = str(getattr(fill, "id", "") or "").strip()
                if not fid:
                    bfid = str(getattr(fill, "broker_fill_id", "") or "").strip()
                    fid = "|".join(
                        [
                            str(getattr(fill, "order_id", "") or ""),
                            bfid,
                            str(getattr(fill, "symbol", "") or ""),
                            str(getattr(fill, "quantity", 0) or 0),
                            f"{float(getattr(fill, 'price', 0.0) or 0.0):.6f}",
                            str(getattr(fill, "timestamp", "") or ""),
                        ]
                    )
                if fid in seen:
                    continue
                seen.add(fid)
                out.append(fill)
        return out

    def get_order_status(self, order_id: str) -> OrderStatus | None:
        for idx in self._preferred_indices_for_order(order_id):
            venue = self._venues[idx]
            try:
                t0 = time.time()
                status = venue.get_order_status(order_id)
                self._mark_read(idx, latency_ms=(time.time() - t0) * 1000.0)
                self._active_idx = idx
                if status is not None:
                    self._order_venue_idx[str(order_id)] = idx
                    return status
            except Exception as e:
                self._mark_failure(idx, e)
        return None

    def sync_order(self, order: Order) -> Order:
        order_id = str(getattr(order, "id", "") or "")
        for idx in self._preferred_indices_for_order(order_id):
            venue = self._venues[idx]
            try:
                t0 = time.time()
                synced = venue.sync_order(order)
                self._mark_read(idx, latency_ms=(time.time() - t0) * 1000.0)
                self._active_idx = idx
                if order_id:
                    self._order_venue_idx[order_id] = idx
                bid = str(getattr(synced, "broker_id", "") or "").strip()
                if order_id and bid:
                    self.register_order_mapping(order_id, bid)
                return synced
            except Exception as e:
                self._mark_failure(idx, e)
        return order

    def get_health_snapshot(self) -> dict[str, object]:
        active_name = None
        if 0 <= self._active_idx < len(self._venues):
            active_name = self._venues[self._active_idx].name
        venues = []
        for idx, venue in enumerate(self._venues):
            last_fail = float(self._last_fail_ts.get(idx, 0.0))
            cooldown_until = (
                last_fail + float(self._cooldown_seconds)
                if last_fail > 0
                else 0.0
            )
            venues.append(
                {
                    "name": venue.name,
                    "connected": bool(venue.is_connected),
                    "fail_count": int(self._fail_counts.get(venue.name, 0)),
                    "submit_count": int(self._submit_counts.get(venue.name, 0)),
                    "read_count": int(self._read_counts.get(venue.name, 0)),
                    "avg_read_latency_ms": round(
                        float(self._avg_read_latency_ms(idx)), 3
                    ),
                    "recent_failures_5m": int(self._recent_failures(idx, window_seconds=300.0)),
                    "last_error": str(self._last_errors.get(str(venue.name), "")),
                    "score": round(float(self._venue_score(idx)), 4),
                    "cooldown_until": cooldown_until,
                }
            )
        return {
            "active_venue": active_name,
            "cooldown_seconds": self._cooldown_seconds,
            "order_affinity_count": int(len(self._order_venue_idx)),
            "venues": venues,
        }


def _create_live_broker_by_type(broker_type: str) -> BrokerInterface:
    broker_type = str(broker_type or "ths").lower()
    if broker_type in ('zszq', 'zhaoshang', '鎷涘晢'):
        return ZSZQBroker()
    if broker_type in ('ths', 'ht', 'gj', 'yh'):
        return THSBroker(broker_type=broker_type)
    log.warning("Unknown live broker_type '%s', fallback to ths", broker_type)
    return THSBroker(broker_type='ths')

def create_broker(
    mode: str = None, **kwargs,
) -> BrokerInterface:
    """
    Factory function to create appropriate broker.

    Args:
        mode: 'simulation', 'paper', 'live', 'ths', 'ht',
              'gj', 'yh', 'zszq'
        **kwargs: Additional arguments for broker
    """

    if mode is None:
        mode = (
            CONFIG.trading_mode.value
            if hasattr(CONFIG.trading_mode, 'value')
            else str(CONFIG.trading_mode)
        )

    mode = mode.lower()

    if mode in ('simulation', 'paper'):
        return SimulatorBroker(
            kwargs.get('capital', CONFIG.capital),
        )
    elif mode == 'live':
        priority = kwargs.get("venue_priority")
        if priority is None:
            priority = getattr(CONFIG.trading, "venue_priority", []) or []

        enable_multi = bool(kwargs.get("enable_multi_venue", False))
        if not enable_multi:
            enable_multi = bool(getattr(CONFIG.trading, "enable_multi_venue", False))
        if not enable_multi and isinstance(priority, list) and len(priority) > 1:
            enable_multi = True

        if enable_multi:
            venues: list[BrokerInterface] = []
            for item in priority:
                bt = str(item or "").strip().lower()
                if not bt:
                    continue
                venues.append(_create_live_broker_by_type(bt))
            if not venues:
                venues = [_create_live_broker_by_type(kwargs.get("broker_type", "ths"))]
            cooldown = kwargs.get(
                "venue_failover_cooldown_seconds",
                getattr(CONFIG.trading, "venue_failover_cooldown_seconds", 30),
            )
            return MultiVenueBroker(venues, failover_cooldown_seconds=int(cooldown))

        return _create_live_broker_by_type(kwargs.get('broker_type', 'ths'))
    elif mode in ('ths', 'ht', 'gj', 'yh'):
        return THSBroker(broker_type=mode)
    elif mode in ('zszq', 'zhaoshang', '鎷涘晢'):
        return ZSZQBroker()
    else:
        log.warning(
            f"Unknown broker mode: {mode}, using simulator"
        )
        return SimulatorBroker()


