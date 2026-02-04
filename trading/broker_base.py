# trading/broker_base.py - SINGLE unified version

"""
Unified Broker Interface
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
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
    stock_name: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.LIMIT
    quantity: int = 0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    broker_order_id: str = ""
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
    available_qty: int = 0
    frozen_qty: int = 0
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
    broker: str = ""
    cash: float = 0.0
    available: float = 0.0
    frozen: float = 0.0
    market_value: float = 0.0
    total_pnl: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    
    @property
    def equity(self) -> float:
        return self.cash + self.market_value


class BrokerInterface(ABC):
    """Abstract broker interface - ONE interface for all brokers"""
    
    on_order_update: Optional[Callable[[Order], None]] = None
    on_trade: Optional[Callable[[Order], None]] = None
    on_error: Optional[Callable[[str], None]] = None
    
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
    def submit_order(self, order: Order) -> Order:
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass
    
    @abstractmethod
    def get_orders(self) -> List[Order]:
        pass
    
    def buy(self, code: str, qty: int, price: float = None) -> Order:
        order = Order(
            stock_code=code,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            quantity=qty,
            price=price
        )
        return self.submit_order(order)
    
    def sell(self, code: str, qty: int, price: float = None) -> Order:
        order = Order(
            stock_code=code,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            quantity=qty,
            price=price
        )
        return self.submit_order(order)