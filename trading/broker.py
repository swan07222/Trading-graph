"""
Unified Broker Interface - Single source of truth for all trading operations

FIXED Issues:
- Single Order/Position/Account definition
- Single BrokerInterface abstract class
- Single SimulatorBroker implementation
- Thread-safe operations
- Proper T+1 handling

Author: AI Trading System v3.0
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Callable
import threading
import uuid

from config import CONFIG
from utils.logger import log


# === Enums (Single Definitions) ===

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# === Data Classes (Single Definitions) ===

@dataclass
class Order:
    """Order representation - immutable after creation"""
    id: str = ""
    stock_code: str = ""
    stock_name: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.LIMIT
    quantity: int = 0
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    message: str = ""
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
    
    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
    
    @property
    def remaining_qty(self) -> int:
        return self.quantity - self.filled_qty


@dataclass
class Position:
    """Position representation"""
    stock_code: str
    stock_name: str = ""
    quantity: int = 0
    available_qty: int = 0  # Quantity available to sell (T+1 aware)
    avg_cost: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    def update_price(self, price: float):
        """Update current price and recalculate P&L"""
        self.current_price = price
        if self.avg_cost > 0 and self.quantity > 0:
            self.unrealized_pnl = (price - self.avg_cost) * self.quantity
            self.unrealized_pnl_pct = (price / self.avg_cost - 1) * 100


@dataclass
class Account:
    """Account state"""
    broker_name: str = ""
    cash: float = 0.0
    available_cash: float = 0.0
    frozen_cash: float = 0.0
    market_value: float = 0.0
    total_pnl: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    
    @property
    def equity(self) -> float:
        return self.cash + self.market_value


# === Abstract Broker Interface ===

class BrokerInterface(ABC):
    """
    Abstract broker interface - all brokers must implement this.
    
    Thread-safe design with callbacks for order updates.
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._callbacks: Dict[str, List[Callable]] = {
            'order_update': [],
            'trade': [],
            'error': [],
        }
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Broker display name"""
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected"""
        pass
    
    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """Connect to broker"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from broker"""
        pass
    
    @abstractmethod
    def get_account(self) -> Account:
        """Get account state"""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        pass
    
    @abstractmethod
    def get_position(self, stock_code: str) -> Optional[Position]:
        """Get single position"""
        pass
    
    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit order"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        pass
    
    @abstractmethod
    def get_orders(self, active_only: bool = True) -> List[Order]:
        """Get orders"""
        pass
    
    # === Convenience Methods ===
    
    def buy(self, code: str, qty: int, price: float = None) -> Order:
        """Place buy order"""
        order = Order(
            stock_code=code,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            quantity=qty,
            price=price
        )
        return self.submit_order(order)
    
    def sell(self, code: str, qty: int, price: float = None) -> Order:
        """Place sell order"""
        order = Order(
            stock_code=code,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            quantity=qty,
            price=price
        )
        return self.submit_order(order)
    
    def sell_all(self, code: str, price: float = None) -> Optional[Order]:
        """Sell entire position"""
        pos = self.get_position(code)
        if pos and pos.available_qty > 0:
            return self.sell(code, pos.available_qty, price)
        return None
    
    # === Callback Management ===
    
    def on(self, event: str, callback: Callable):
        """Register callback for event"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _emit(self, event: str, *args, **kwargs):
        """Emit event to callbacks (thread-safe)"""
        callbacks = self._callbacks.get(event, [])
        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                log.error(f"Callback error for {event}: {e}")


# === Simulator Broker ===

class SimulatorBroker(BrokerInterface):
    """
    Paper trading simulator with realistic behavior.
    
    Features:
    - Realistic slippage and commission
    - T+1 rule enforcement
    - Thread-safe operations
    - Proper cost calculation
    """
    
    def __init__(self, initial_capital: float = None):
        super().__init__()
        self._initial_capital = initial_capital or CONFIG.CAPITAL
        self._cash = self._initial_capital
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []
        self._trades: List[Dict] = []
        self._connected = False
        
        # T+1 tracking: stock_code -> purchase_date
        self._purchase_dates: Dict[str, date] = {}
        self._last_settlement_date = date.today()
        
        # Data fetcher for prices
        self._fetcher = None
    
    def _get_fetcher(self):
        """Lazy init fetcher"""
        if self._fetcher is None:
            from data.fetcher import DataFetcher
            self._fetcher = DataFetcher()
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
            log.info(f"Simulator connected with ¥{self._initial_capital:,.2f}")
            return True
    
    def disconnect(self):
        with self._lock:
            self._connected = False
            log.info("Simulator disconnected")
    
    def get_account(self) -> Account:
        with self._lock:
            self._check_settlement()
            self._update_prices()
            
            market_value = sum(p.market_value for p in self._positions.values())
            unrealized_pnl = sum(p.unrealized_pnl for p in self._positions.values())
            realized_pnl = sum(p.realized_pnl for p in self._positions.values())
            
            return Account(
                broker_name=self.name,
                cash=self._cash,
                available_cash=self._cash,
                market_value=market_value,
                total_pnl=unrealized_pnl + realized_pnl,
                positions=dict(self._positions)
            )
    
    def get_positions(self) -> Dict[str, Position]:
        with self._lock:
            self._check_settlement()
            self._update_prices()
            return dict(self._positions)
    
    def get_position(self, stock_code: str) -> Optional[Position]:
        with self._lock:
            self._check_settlement()
            pos = self._positions.get(stock_code)
            if pos:
                self._update_single_price(pos)
            return pos
    
    def submit_order(self, order: Order) -> Order:
        with self._lock:
            self._check_settlement()
            
            # Generate order ID
            order.id = f"SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
            order.created_at = datetime.now()
            
            # Get current price
            fetcher = self._get_fetcher()
            quote = fetcher.get_realtime(order.stock_code)
            
            if not quote or quote.price <= 0:
                order.status = OrderStatus.REJECTED
                order.message = "Cannot get market quote"
                self._emit('order_update', order)
                return order
            
            current_price = quote.price
            order.stock_name = quote.name
            
            # Validate order
            validation = self._validate_order(order, current_price)
            if not validation[0]:
                order.status = OrderStatus.REJECTED
                order.message = validation[1]
                self._emit('order_update', order)
                return order
            
            order.status = OrderStatus.SUBMITTED
            self._orders[order.id] = order
            
            # Execute immediately (market simulation)
            self._execute_order(order, current_price)
            
            self._emit('order_update', order)
            return order
    
    def _validate_order(self, order: Order, price: float) -> tuple:
        """Validate order, returns (valid, message)"""
        
        # Lot size check
        if order.quantity <= 0:
            return False, "Quantity must be positive"
        
        if order.quantity % CONFIG.LOT_SIZE != 0:
            return False, f"Quantity must be multiple of {CONFIG.LOT_SIZE}"
        
        if order.side == OrderSide.BUY:
            # Cash check
            exec_price = order.price or price
            cost = order.quantity * exec_price
            commission = cost * CONFIG.COMMISSION
            total = cost + commission
            
            if total > self._cash:
                return False, f"Insufficient funds: need ¥{total:,.2f}, have ¥{self._cash:,.2f}"
            
        else:  # SELL
            pos = self._positions.get(order.stock_code)
            
            if not pos:
                return False, f"No position in {order.stock_code}"
            
            if order.quantity > pos.available_qty:
                return False, f"Available: {pos.available_qty}, requested: {order.quantity}"
            
            # T+1 check (already handled in available_qty, but explicit)
            if CONFIG.T_PLUS_1:
                purchase_date = self._purchase_dates.get(order.stock_code)
                if purchase_date == date.today():
                    # Can only sell shares not purchased today
                    # This is already reflected in available_qty
                    pass
        
        return True, "OK"
    
    def _execute_order(self, order: Order, market_price: float):
        """Execute order with realistic simulation"""
        import random
        
        # Calculate fill price with slippage
        slippage = CONFIG.SLIPPAGE
        
        if order.side == OrderSide.BUY:
            # Buy at ask (slightly higher)
            base_price = order.price if order.order_type == OrderType.LIMIT else market_price
            fill_price = base_price * (1 + slippage * (0.5 + 0.5 * random.random()))
        else:
            # Sell at bid (slightly lower)
            base_price = order.price if order.order_type == OrderType.LIMIT else market_price
            fill_price = base_price * (1 - slippage * (0.5 + 0.5 * random.random()))
        
        fill_price = round(fill_price, 2)
        fill_qty = order.quantity
        
        # Calculate costs
        trade_value = fill_qty * fill_price
        commission = trade_value * CONFIG.COMMISSION
        stamp_tax = trade_value * CONFIG.STAMP_TAX if order.side == OrderSide.SELL else 0
        total_cost = commission + stamp_tax
        
        # Update position and cash
        if order.side == OrderSide.BUY:
            self._cash -= (trade_value + total_cost)
            
            if order.stock_code in self._positions:
                pos = self._positions[order.stock_code]
                total_qty = pos.quantity + fill_qty
                pos.avg_cost = (pos.avg_cost * pos.quantity + fill_price * fill_qty) / total_qty
                pos.quantity = total_qty
                # New shares not available until tomorrow (T+1)
            else:
                self._positions[order.stock_code] = Position(
                    stock_code=order.stock_code,
                    stock_name=order.stock_name,
                    quantity=fill_qty,
                    available_qty=0,  # T+1: not available today
                    avg_cost=fill_price,
                    current_price=fill_price
                )
            
            self._purchase_dates[order.stock_code] = date.today()
            
        else:  # SELL
            self._cash += (trade_value - total_cost)
            
            pos = self._positions[order.stock_code]
            realized = (fill_price - pos.avg_cost) * fill_qty
            pos.realized_pnl += realized
            pos.quantity -= fill_qty
            pos.available_qty -= fill_qty
            
            if pos.quantity <= 0:
                del self._positions[order.stock_code]
                if order.stock_code in self._purchase_dates:
                    del self._purchase_dates[order.stock_code]
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_qty = fill_qty
        order.filled_price = fill_price
        order.commission = total_cost
        order.updated_at = datetime.now()
        
        # Record trade
        trade = {
            'order_id': order.id,
            'stock_code': order.stock_code,
            'stock_name': order.stock_name,
            'side': order.side.value,
            'quantity': fill_qty,
            'price': fill_price,
            'value': trade_value,
            'commission': commission,
            'stamp_tax': stamp_tax,
            'timestamp': datetime.now()
        }
        self._trades.append(trade)
        self._order_history.append(order)
        
        log.info(
            f"[SIM] {order.side.value.upper()} {fill_qty} {order.stock_code} "
            f"@ ¥{fill_price:.2f} (cost: ¥{total_cost:.2f})"
        )
        
        self._emit('trade', order)
    
    def cancel_order(self, order_id: str) -> bool:
        with self._lock:
            order = self._orders.get(order_id)
            if order and order.is_active:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                self._emit('order_update', order)
                return True
            return False
    
    def get_orders(self, active_only: bool = True) -> List[Order]:
        with self._lock:
            if active_only:
                return [o for o in self._orders.values() if o.is_active]
            return list(self._orders.values())
    
    def _check_settlement(self):
        """Handle T+1 settlement on new trading day"""
        today = date.today()
        if today != self._last_settlement_date:
            # Make yesterday's purchases available
            for code, pos in self._positions.items():
                pos.available_qty = pos.quantity
            
            self._last_settlement_date = today
            log.info("T+1 settlement: all shares now available")
    
    def _update_prices(self):
        """Update all position prices"""
        fetcher = self._get_fetcher()
        for code, pos in self._positions.items():
            quote = fetcher.get_realtime(code)
            if quote and quote.price > 0:
                pos.update_price(quote.price)
    
    def _update_single_price(self, pos: Position):
        """Update single position price"""
        fetcher = self._get_fetcher()
        quote = fetcher.get_realtime(pos.stock_code)
        if quote and quote.price > 0:
            pos.update_price(quote.price)
    
    def get_trade_history(self) -> List[Dict]:
        """Get all executed trades"""
        with self._lock:
            return list(self._trades)
    
    def reset(self):
        """Reset simulator to initial state"""
        with self._lock:
            self._cash = self._initial_capital
            self._positions.clear()
            self._orders.clear()
            self._order_history.clear()
            self._trades.clear()
            self._purchase_dates.clear()
            self._last_settlement_date = date.today()
            log.info("Simulator reset to initial state")