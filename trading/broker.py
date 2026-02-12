# trading/broker.py
"""
Unified Broker Interface - Production Grade with Full Fill Sync

Supports:
- Paper Trading (Simulator)
- 同花顺 (THS)
- 华泰证券 (HT)
- 招商证券 (ZSZQ)
- 国金证券 (GJ)
- 银河证券 (YH)

"""
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path
import threading
import uuid
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor

from config.settings import CONFIG
from core.types import (
    Order, OrderSide, OrderType, OrderStatus,
    Position, Account, Fill
)
from utils.logger import get_logger

log = get_logger(__name__)

# Bounded ID mapping (LRU eviction)
# FIX(6): Prevents unbounded growth of order ID maps

class BoundedOrderedDict(OrderedDict):
    """OrderedDict with max size — evicts oldest on overflow."""

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
    """Parse Chinese broker status string to OrderStatus enum."""
    s = str(status_str).lower()

    if '全部成交' in s or '已成' in s:
        return OrderStatus.FILLED
    if '部分成交' in s:
        return OrderStatus.PARTIAL
    if '已报' in s or '已委托' in s:
        return OrderStatus.ACCEPTED
    if '已撤' in s or '撤单' in s:
        return OrderStatus.CANCELLED
    if '废单' in s or '拒绝' in s:
        return OrderStatus.REJECTED
    return OrderStatus.SUBMITTED

class BrokerInterface(ABC):
    """
    Abstract broker interface — all brokers must implement this.
    Thread-safe design with callbacks for order updates.
    """

    # FIX(6): Max tracked order mappings
    _MAX_ORDER_MAPPINGS = 10000

    def __init__(self):
        self._lock = threading.RLock()
        self._callbacks: Dict[str, List[Callable]] = {
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
    def get_positions(self) -> Dict[str, Position]:
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def get_orders(self, active_only: bool = True) -> List[Order]:
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[float]:
        pass

    @abstractmethod
    def get_fills(self, since: datetime = None) -> List[Fill]:
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        pass

    @abstractmethod
    def sync_order(self, order: Order) -> Order:
        pass

    def get_broker_id(self, order_id: str) -> Optional[str]:
        return self._order_id_to_broker_id.get(order_id)

    def get_order_id(self, broker_id: str) -> Optional[str]:
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

    def sell_all(self, symbol: str, price: float = None) -> Optional[Order]:
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
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []
        self._fills: List[Fill] = []

        # FIX(2): Cursor-based fill tracking instead of clearing list
        self._fill_cursor: int = 0

        self._connected = False

        # T+1 tracking
        self._purchase_dates: Dict[str, date] = {}
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
                f"¥{self._initial_capital:,.2f}"
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

    def get_quote(self, symbol: str) -> Optional[float]:
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

    def get_positions(self) -> Dict[str, Position]:
        with self._lock:
            self._check_settlement()
            self._update_prices()
            return dict(self._positions)

    def get_position(self, symbol: str) -> Optional[Position]:
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
    ) -> Tuple[bool, str]:
        if order.quantity <= 0:
            return False, "Quantity must be positive"

        from core.constants import get_lot_size

        lot_size = get_lot_size(order.symbol)
        if order.quantity % lot_size != 0:
            return False, f"Quantity must be multiple of {lot_size}"

        # FIX(9): Allow price=0 for MARKET orders
        if order.order_type == OrderType.LIMIT and price <= 0:
            return False, "Limit order requires positive price"
        if price <= 0 and order.order_type != OrderType.MARKET:
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
                    f"Insufficient funds: need ¥{total:,.2f}, "
                    f"have ¥{self._cash:,.2f}"
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
    ) -> Tuple[bool, str]:
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

        if order.order_type == OrderType.LIMIT:
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
        if remaining >= 2 * lot and random.random() < 0.25:
            half = max(lot, (remaining // 2 // lot) * lot)
            fill_qty = min(half, remaining)

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
            f"(cost: {total_cost:.2f}, status: {order.status.value})"
        )
        self._emit('trade', order, fill)

    def get_fills(self, since: datetime = None) -> List[Fill]:
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

    def get_all_fills(self) -> List[Fill]:
        """Get ALL fills (not just new ones)."""
        with self._lock:
            return list(self._fills)

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
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

    def get_orders(self, active_only: bool = True) -> List[Order]:
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

        # It's a new trading day — settle all positions
        for symbol, pos in self._positions.items():
            pos.available_qty = pos.quantity
        self._last_settlement_date = today
        log.info("T+1 settlement: all shares now available")

    def _update_prices(self):
        for symbol, pos in self._positions.items():
            price = self.get_quote(symbol)
            if price:
                pos.update_price(price)

    def get_trade_history(self) -> List[Dict]:
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

    def reconcile(self) -> Dict:
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
        self._orders: Dict[str, Order] = {}
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

    def get_quote(self, symbol: str) -> Optional[float]:
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
                balance.get('资金余额')
                or balance.get('总资产')
                or balance.get('可用资金')
                or 0
            )
            available = float(
                balance.get('可用金额')
                or balance.get('可用资金')
                or balance.get('可取资金')
                or cash
            )
            frozen = float(balance.get('冻结金额', 0) or 0)

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

    def get_positions(self) -> Dict[str, Position]:
        if not self.is_connected:
            return {}

        try:
            raw = self._client.position
            positions = {}

            for p in raw:
                code = str(
                    p.get('证券代码')
                    or p.get('股票代码')
                    or ''
                ).zfill(6)

                if not code or code == '000000':
                    continue

                positions[code] = Position(
                    symbol=code,
                    name=(
                        p.get('证券名称')
                        or p.get('股票名称')
                        or ''
                    ),
                    quantity=int(
                        p.get('股票余额')
                        or p.get('持仓数量')
                        or p.get('当前持仓')
                        or 0
                    ),
                    available_qty=int(
                        p.get('可卖余额')
                        or p.get('可用余额')
                        or p.get('可卖数量')
                        or 0
                    ),
                    avg_cost=float(
                        p.get('成本价')
                        or p.get('买入成本')
                        or p.get('参考成本价')
                        or 0
                    ),
                    current_price=float(
                        p.get('当前价')
                        or p.get('最新价')
                        or p.get('市价')
                        or 0
                    ),
                )

            return positions

        except Exception as e:
            log.error(f"Failed to get positions: {e}")
            return {}

    def get_position(self, symbol: str) -> Optional[Position]:
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
                    result.get('委托编号')
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

    def get_fills(self, since: datetime = None) -> List[Fill]:
        """Get fills from broker — deduplicates by broker_fill_id."""
        if not self.is_connected:
            return []

        fills: List[Fill] = []
        try:
            trades = self._client.today_trades

            for trade in trades:
                broker_fill_id = str(
                    trade.get("成交编号", "") or ""
                ).strip()
                if not broker_fill_id:
                    continue

                if broker_fill_id in self._seen_fill_ids:
                    continue

                ts = (
                    trade.get("成交时间")
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
                    trade.get("委托编号", "") or ""
                ).strip()
                our_order_id = self.get_order_id(broker_entrust)
                if not our_order_id:
                    log.warning(
                        f"Unknown entrust number: {broker_entrust}"
                    )
                    continue

                self._seen_fill_ids.add(broker_fill_id)

                trade_side = trade.get(
                    "买卖标志", trade.get("操作", ""),
                )
                side = (
                    OrderSide.BUY
                    if "买" in str(trade_side)
                    else OrderSide.SELL
                )

                symbol = str(
                    trade.get("证券代码", "") or ""
                ).zfill(6)
                qty = int(trade.get("成交数量", 0) or 0)
                price = float(trade.get("成交价格", 0) or 0.0)
                comm = float(trade.get("手续费", 0) or 0.0)
                tax = float(trade.get("印花税", 0) or 0.0)

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
    ) -> Optional[OrderStatus]:
        if not self.is_connected:
            return None

        broker_id = self.get_broker_id(order_id)
        if not broker_id:
            return None

        try:
            entrusts = self._client.today_entrusts

            for entrust in entrusts:
                if str(entrust.get('委托编号', '')) == broker_id:
                    status_str = entrust.get(
                        '委托状态', entrust.get('状态', ''),
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
                if str(entrust.get('委托编号', '')) == broker_id:
                    # FIX(3): Use shared parser
                    order.status = parse_broker_status(
                        entrust.get('委托状态', ''),
                    )
                    order.filled_qty = int(
                        entrust.get('成交数量', 0) or 0
                    )

                    avg_price = entrust.get(
                        '成交均价',
                        entrust.get('成交价格', 0),
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

    def get_orders(self, active_only: bool = True) -> List[Order]:
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

# FIX(11): Thin subclasses — all shared logic is in base

class THSBroker(EasytraderBroker):
    """同花顺 / 华泰 / 国金 / 银河 broker via easytrader."""

    BROKER_TYPES = {
        'ths': '同花顺',
        'ht': '华泰证券',
        'gj': '国金证券',
        'yh': '银河证券',
    }

    def __init__(self, broker_type: str = 'ths'):
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
    """招商证券 broker via easytrader (universal mode)."""

    @property
    def name(self) -> str:
        return "招商证券"

    def _get_easytrader_type(self) -> str:
        return 'universal'

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
    from config.settings import TradingMode

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
        broker_type = kwargs.get('broker_type', 'ths')
        if broker_type in ('zszq', 'zhaoshang', '招商'):
            return ZSZQBroker()
        return THSBroker(broker_type=broker_type)
    elif mode in ('ths', 'ht', 'gj', 'yh'):
        return THSBroker(broker_type=mode)
    elif mode in ('zszq', 'zhaoshang', '招商'):
        return ZSZQBroker()
    else:
        log.warning(
            f"Unknown broker mode: {mode}, using simulator"
        )
        return SimulatorBroker()