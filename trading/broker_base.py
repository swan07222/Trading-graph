"""
Broker Interface - Abstract base for all brokers
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Callable


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order representation"""
    id: str = ""
    stock_code: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.LIMIT
    quantity: int = 0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    message: str = ""
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
    
    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]


@dataclass
class Position:
    """Position representation"""
    stock_code: str
    stock_name: str = ""
    quantity: int = 0
    available_qty: int = 0  # Available for selling (T+1)
    avg_cost: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    def update_price(self, price: float):
        self.current_price = price
        if self.avg_cost > 0 and self.quantity > 0:
            self.unrealized_pnl = (price - self.avg_cost) * self.quantity
            self.unrealized_pnl_pct = (price / self.avg_cost - 1) * 100


@dataclass
class Account:
    """Account state"""
    cash: float = 0.0
    available: float = 0.0
    frozen: float = 0.0
    market_value: float = 0.0
    total_pnl: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    
    @property
    def equity(self) -> float:
        return self.cash + self.market_value
    
    @property
    def position_ratio(self) -> float:
        if self.equity <= 0:
            return 0
        return self.market_value / self.equity * 100


class BrokerInterface(ABC):
    """Abstract broker interface"""
    
    # Callbacks
    on_order_update: Optional[Callable[[Order], None]] = None
    on_trade: Optional[Callable[[Order], None]] = None
    on_error: Optional[Callable[[str], None]] = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Broker name"""
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Connection status"""
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from broker"""
        pass
    
    @abstractmethod
    def get_account(self) -> Account:
        """Get account information"""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Get all positions"""
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
    def get_orders(self) -> List[Order]:
        """Get active orders"""
        pass
    
    def buy(self, code: str, qty: int, price: float = None) -> Order:
        """Convenience buy method"""
        order = Order(
            stock_code=code,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            quantity=qty,
            price=price
        )
        return self.submit_order(order)
    
    def sell(self, code: str, qty: int, price: float = None) -> Order:
        """Convenience sell method"""
        order = Order(
            stock_code=code,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            quantity=qty,
            price=price
        )
        return self.submit_order(order)