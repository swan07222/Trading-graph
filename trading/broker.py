# trading/broker.py
"""
Unified Broker Interface - Production Grade with Full Fill Sync

Supports:
- Paper Trading (Simulator)
- TongHuaShun (THS)
- HuaTai (HT)
- ZhaoShang (ZSZQ)
- GuoJin (GJ)
- YinHe (YH)

"""
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Any

from config.settings import CONFIG
from core.types import Account, Fill, Order, OrderSide, OrderStatus, OrderType, Position
from trading.broker_common import (
    BoundedOrderedDict,
    make_fill_uid,
    parse_broker_status,
)
from utils.logger import get_logger

log = get_logger(__name__)

class BrokerInterface(ABC):
    """
    Abstract broker interface - all brokers must implement this.
    Thread-safe design with callbacks for order updates.
    """

    # FIX(6): Max tracked order mappings
    _MAX_ORDER_MAPPINGS = 10000

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._callbacks: dict[str, list[Callable[..., object]]] = {
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
    def connect(self, **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> None:
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
    def get_fills(self, since: datetime | None = None) -> list[Fill]:
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

    def register_order_mapping(self, order_id: str, broker_id: str) -> None:
        with self._lock:
            self._order_id_to_broker_id[order_id] = broker_id
            self._broker_id_to_order_id[broker_id] = order_id

    # === Convenience Methods ===
    # FIX(8): price=None is valid for market orders

    def buy(
        self, symbol: str, qty: int, price: float | None = None,
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
        self, symbol: str, qty: int, price: float | None = None,
    ) -> Order:
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            quantity=qty,
            price=price or 0.0,
        )
        return self.submit_order(order)

    def sell_all(self, symbol: str, price: float | None = None) -> Order | None:
        pos = self.get_position(symbol)
        if pos and pos.available_qty > 0:
            return self.sell(symbol, pos.available_qty, price)
        return None

    # === Callback Management ===

    def on(self, event: str, callback: Callable[..., object]) -> None:
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
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

    def __init__(self, initial_capital: float | None = None) -> None:
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
        self._max_no_quote_retries = 20

        # Short-lived quote cache to smooth transient feed gaps.
        staleness = float(
            max(
                5.0,
                float(getattr(getattr(CONFIG, "risk", None), "quote_staleness_seconds", 5.0) or 5.0) * 2.0,
            )
        )
        self._quote_cache_max_age_s: float = min(30.0, staleness)
        self._quote_cache: dict[str, tuple[float, float]] = {}

        # FIX(10): Bounded thread pool
        self._exec_pool = ThreadPoolExecutor(
            max_workers=self._MAX_EXEC_WORKERS,
            thread_name_prefix="sim_exec",
        )

    def _get_fetcher(self) -> Any:
        if self._fetcher is None:
            from data.fetcher import get_fetcher
            self._fetcher = get_fetcher()
        return self._fetcher

    def _cache_quote(self, symbol: str, price: float) -> None:
        if price <= 0:
            return
        self._quote_cache[str(symbol)] = (float(price), float(time.time()))

    def _get_cached_quote(self, symbol: str) -> float | None:
        rec = self._quote_cache.get(str(symbol))
        if not rec:
            return None
        price, ts = rec
        if float(time.time()) - float(ts) > float(self._quote_cache_max_age_s):
            self._quote_cache.pop(str(symbol), None)
            return None
        if float(price) <= 0:
            return None
        return float(price)

    def _get_history_fallback_quote(self, symbol: str) -> float | None:
        fetcher = self._get_fetcher()
        for interval, bars in (("1m", 2), ("1d", 2)):
            try:
                try:
                    df = fetcher.get_history(
                        symbol,
                        interval=interval,
                        bars=int(bars),
                        use_cache=True,
                        update_db=False,
                        allow_online=False,
                    )
                except TypeError:
                    df = fetcher.get_history(
                        symbol,
                        interval=interval,
                        bars=int(bars),
                        use_cache=True,
                    )
            except Exception as e:
                log.debug(
                    "History fallback quote fetch failed for %s (%s): %s",
                    symbol,
                    interval,
                    e,
                )
                df = None
            if df is None or len(df) <= 0:
                continue
            try:
                close = float(df["close"].iloc[-1])
            except Exception:
                close = 0.0
            if close > 0:
                return close
        return None

    @staticmethod
    def _safe_tag_float(order: Order, key: str, default: float = 0.0) -> float:
        if not isinstance(order.tags, dict):
            return float(default)
        try:
            return float(order.tags.get(key, default) or default)
        except (TypeError, ValueError):
            return float(default)

    def _fallback_reference_price(self, order: Order) -> float:
        refs = [
            float(getattr(order, "price", 0.0) or 0.0),
            float(getattr(order, "stop_price", 0.0) or 0.0),
            self._safe_tag_float(order, "trigger_price", 0.0),
        ]
        for px in refs:
            if px > 0:
                return float(px)
        return 0.0

    @property
    def name(self) -> str:
        return "Paper Trading Simulator"

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, **kwargs: Any) -> bool:
        with self._lock:
            self._connected = True
            log.info(
                f"Simulator connected with "
                f"CNY {self._initial_capital:,.2f}"
            )
            return True

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False
            try:
                self._exec_pool.shutdown(wait=False)
            except Exception as e:
                log.debug("Simulator execution pool shutdown failed: %s", e)
            log.info("Simulator disconnected")

    def get_quote(self, symbol: str) -> float | None:
        symbol = str(symbol or "").strip()
        if not symbol:
            return None

        try:
            from data.feeds import get_feed_manager
            fm = get_feed_manager(auto_init=False)
            q = fm.get_quote(symbol)
            if q and getattr(q, "price", 0) and float(q.price) > 0:
                px = float(q.price)
                self._cache_quote(symbol, px)
                return px
        except Exception as e:
            log.debug("Feed-manager quote unavailable for %s: %s", symbol, e)

        try:
            fetcher = self._get_fetcher()
            quote = fetcher.get_realtime(symbol)
            if quote and getattr(quote, "price", 0) and float(quote.price) > 0:
                px = float(quote.price)
                self._cache_quote(symbol, px)
                return px
        except Exception as e:
            log.debug("Realtime quote fetch unavailable for %s: %s", symbol, e)

        cached = self._get_cached_quote(symbol)
        if cached is not None:
            return cached

        fallback = self._get_history_fallback_quote(symbol)
        if fallback is not None:
            self._cache_quote(symbol, fallback)
            return fallback
        return None

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

    def _schedule_execution(self, order_id: str) -> None:
        """Execute order asynchronously using bounded thread pool."""
        self._exec_pool.submit(self._execution_worker, order_id)

    def _execution_worker(self, order_id: str) -> None:
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
                    retries = int(self._safe_tag_float(order, "_no_quote_retries", 0.0)) + 1
                    if not isinstance(order.tags, dict):
                        order.tags = {}
                    order.tags["_no_quote_retries"] = retries
                    max_retries = int(
                        max(
                            3,
                            self._safe_tag_float(order, "max_no_quote_retries", float(self._max_no_quote_retries)),
                        )
                    )
                    if retries < max_retries:
                        if retries in {1, 3, 10}:
                            order.message = (
                                f"Waiting for market quote ({retries}/{max_retries})"
                            )
                            order.updated_at = datetime.now()
                            self._emit("order_update", order)
                        continue

                    order.status = OrderStatus.REJECTED
                    order.message = (
                        f"No market quote during async execution "
                        f"({retries}/{max_retries})"
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
                fallback_price = self._fallback_reference_price(order)
                if fallback_price > 0:
                    current_price = float(fallback_price)
                    if not isinstance(order.tags, dict):
                        order.tags = {}
                    order.tags["submitted_with_fallback_quote"] = True
                else:
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
        trail_limit_offset_pct = self._safe_tag_float(
            order, "trail_limit_offset_pct", 0.0
        )
        if (
            order.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT}
            and price <= 0
        ):
            return False, "Limit order requires positive price"
        if (
            order.order_type == OrderType.TRAIL_LIMIT
            and price <= 0
            and trail_limit_offset_pct <= 0
        ):
            return False, "Trailing limit requires price or trail_limit_offset_pct"
        if price <= 0 and order.order_type not in {
            OrderType.MARKET,
            OrderType.STOP,
            OrderType.IOC,
            OrderType.FOK,
            OrderType.TRAIL_MARKET,
        }:
            return False, "Invalid price"

        trigger_px = max(
            float(getattr(order, "stop_price", 0.0) or 0.0),
            self._safe_tag_float(order, "trigger_price", 0.0),
        )
        trailing_pct = self._safe_tag_float(order, "trailing_stop_pct", 0.0)
        if requested_type in {"stop", "stop_limit"} and trigger_px <= 0:
            trigger_px = float(order.price or 0.0)
            if trigger_px <= 0:
                return False, "Stop order requires trigger_price"
        if requested_type in {"trail_market", "trail_limit"} and trailing_pct <= 0:
            return False, "Trailing order requires trailing_stop_pct > 0"

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

    def _conditional_trigger_ready(
        self,
        order: Order,
        market_price: float,
        requested_type: str,
    ) -> tuple[bool, str]:
        if requested_type in {"stop", "stop_limit"}:
            trigger_px = max(
                float(getattr(order, "stop_price", 0.0) or 0.0),
                self._safe_tag_float(order, "trigger_price", 0.0),
                float(order.price or 0.0),
            )
            if trigger_px <= 0:
                return False, "Waiting for stop trigger configuration"
            if not isinstance(order.tags, dict):
                order.tags = {}
            order.tags["trigger_price"] = float(trigger_px)
            hit = (
                float(market_price) >= float(trigger_px)
                if order.side == OrderSide.BUY
                else float(market_price) <= float(trigger_px)
            )
            wait_msg = (
                f"Waiting stop trigger @ {trigger_px:.2f} "
                f"(mkt {float(market_price):.2f})"
            )
            return bool(hit), wait_msg

        if requested_type in {"trail_market", "trail_limit"}:
            trailing_pct = self._safe_tag_float(order, "trailing_stop_pct", 0.0)
            if trailing_pct <= 0:
                return False, "Waiting for trailing_stop_pct configuration"

            if not isinstance(order.tags, dict):
                order.tags = {}

            px = float(market_price)
            if order.side == OrderSide.SELL:
                anchor = float(order.tags.get("trail_anchor_price", px) or px)
                anchor = max(anchor, px)
                trigger_px = anchor * (1.0 - float(trailing_pct) / 100.0)
                hit = px <= trigger_px
            else:
                anchor = float(order.tags.get("trail_anchor_price", px) or px)
                anchor = min(anchor, px)
                trigger_px = anchor * (1.0 + float(trailing_pct) / 100.0)
                hit = px >= trigger_px

            order.tags["trail_anchor_price"] = float(anchor)
            order.tags["trigger_price"] = float(trigger_px)
            wait_msg = (
                f"Waiting trailing trigger @ {trigger_px:.2f} "
                f"(anchor {anchor:.2f}, mkt {px:.2f})"
            )
            return bool(hit), wait_msg

        return True, ""

    def _execute_order(self, order: Order, market_price: float) -> None:
        """Execute order with realistic simulation."""
        import random

        if not isinstance(order.tags, dict):
            order.tags = {}
        requested_type = str(
            order.tags.get("requested_order_type", order.order_type.value)
        ).strip().lower().replace("-", "_")
        tif = str(
            order.tags.get("time_in_force", "day")
        ).strip().lower().replace("-", "_")
        if requested_type in {"ioc", "fok"}:
            tif = requested_type

        if requested_type in {"stop", "stop_limit", "trail_market", "trail_limit"}:
            triggered, wait_msg = self._conditional_trigger_ready(
                order=order,
                market_price=float(market_price),
                requested_type=requested_type,
            )
            if not triggered:
                order.status = OrderStatus.ACCEPTED
                if order.message != wait_msg:
                    order.message = wait_msg
                    order.updated_at = datetime.now()
                    self._emit("order_update", order)
                return
            if not bool(order.tags.get("_conditional_triggered", False)):
                order.tags["_conditional_triggered"] = True
                order.tags["triggered_at"] = datetime.now().isoformat()
                order.message = (
                    f"Trigger fired ({requested_type}) @ "
                    f"{float(order.tags.get('trigger_price', market_price)):.2f}"
                )
                order.updated_at = datetime.now()
                self._emit("order_update", order)

        prev_close = float(market_price)
        try:
            fetcher = self._get_fetcher()
            q = fetcher.get_realtime(order.symbol)
            if q and getattr(q, "close", 0) and float(q.close) > 0:
                prev_close = float(q.close)
        except Exception as e:
            log.debug("Previous-close lookup failed for %s: %s", order.symbol, e)

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
        working_limit_price = float(order.price or 0.0)
        if is_limit_style:
            if order.order_type == OrderType.TRAIL_LIMIT and working_limit_price <= 0:
                offset_pct = self._safe_tag_float(order, "trail_limit_offset_pct", 0.0)
                if offset_pct <= 0:
                    order.status = OrderStatus.REJECTED
                    order.message = "Trailing limit requires price or trail_limit_offset_pct"
                    order.updated_at = datetime.now()
                    self._emit('order_update', order)
                    return
                if order.side == OrderSide.BUY:
                    working_limit_price = float(market_price) * (1.0 + offset_pct / 100.0)
                else:
                    working_limit_price = float(market_price) * (1.0 - offset_pct / 100.0)
                working_limit_price = max(0.01, working_limit_price)
                order.price = round(float(working_limit_price), 2)
            else:
                working_limit_price = float(order.price or 0.0)
            if working_limit_price <= 0:
                order.status = OrderStatus.REJECTED
                order.message = "Limit order requires positive price"
                order.updated_at = datetime.now()
                self._emit('order_update', order)
                return

            if (
                order.side == OrderSide.BUY
                and float(market_price) > float(working_limit_price)
            ):
                if tif in {"ioc", "fok"}:
                    order.status = OrderStatus.CANCELLED
                    order.message = (
                        f"{tif.upper()} BUY not marketable: market "
                        f"{market_price:.2f} > limit {working_limit_price:.2f}"
                    )
                else:
                    order.status = OrderStatus.ACCEPTED
                    order.message = (
                        f"Waiting limit BUY: market "
                        f"{market_price:.2f} > limit {working_limit_price:.2f}"
                    )
                order.updated_at = datetime.now()
                self._emit('order_update', order)
                return

            if (
                order.side == OrderSide.SELL
                and float(market_price) < float(working_limit_price)
            ):
                if tif in {"ioc", "fok"}:
                    order.status = OrderStatus.CANCELLED
                    order.message = (
                        f"{tif.upper()} SELL not marketable: market "
                        f"{market_price:.2f} < limit {working_limit_price:.2f}"
                    )
                else:
                    order.status = OrderStatus.ACCEPTED
                    order.message = (
                        f"Waiting limit SELL: market "
                        f"{market_price:.2f} < limit {working_limit_price:.2f}"
                    )
                order.updated_at = datetime.now()
                self._emit('order_update', order)
                return

            fill_price = float(working_limit_price)
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

    def get_fills(self, since: datetime | None = None) -> list[Fill]:
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

    def _check_settlement(self) -> None:
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

        # It's a new trading day - settle all positions
        for _symbol, pos in self._positions.items():
            pos.available_qty = pos.quantity
        self._last_settlement_date = today
        log.info("T+1 settlement: all shares now available")

    def _update_prices(self) -> None:
        for symbol, pos in self._positions.items():
            price = self.get_quote(symbol)
            if price:
                pos.update_price(price)

    def get_trade_history(self) -> list[dict[str, Any]]:
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

    def reset(self) -> None:
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
            self._quote_cache.clear()
            log.info("Simulator reset to initial state")

    def reconcile(self) -> dict[str, Any]:
        with self._lock:
            return {
                'cash_diff': 0.0,
                'position_diffs': [],
                'missing_positions': [],
                'extra_positions': [],
                'reconciled': True,
                'timestamp': datetime.now().isoformat(),
            }

def __getattr__(name: str) -> Any:
    """Lazy-export live broker classes without importing them at module load."""
    if name in {
        "EasytraderBroker",
        "THSBroker",
        "ZSZQBroker",
        "MultiVenueBroker",
    }:
        from . import broker_live as _broker_live

        return getattr(_broker_live, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def create_broker(
    mode: str | None = None,
    **kwargs: Any,
) -> BrokerInterface:
    """
    Factory function to create appropriate broker.

    Args:
        mode: 'simulation', 'paper', 'live', 'ths', 'ht',
              'gj', 'yh', 'zszq'
        **kwargs: Additional arguments for broker
    """
    from .broker_live import MultiVenueBroker, THSBroker, ZSZQBroker, _create_live_broker_by_type

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
    elif mode in ('zszq', 'zhaoshang'):
        return ZSZQBroker()
    else:
        log.warning(
            f"Unknown broker mode: {mode}, using simulator"
        )
        return SimulatorBroker()


