"""
Canonical Types - SINGLE SOURCE OF TRUTH for all trading types
All other modules MUST import from here
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid

# ============================================================
# Enums
# ============================================================

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SystemStatus(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    KILL_SWITCH = "kill_switch"


# ============================================================
# Order
# ============================================================

@dataclass
class Order:
    """Canonical Order representation"""
    id: str = ""
    client_id: str = ""
    broker_id: str = ""
    
    # Instrument
    symbol: str = ""
    name: str = ""
    
    # Order details
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.LIMIT
    quantity: int = 0
    price: float = 0.0
    stop_price: float = 0.0
    
    # Execution
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_price: float = 0.0
    avg_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    
    # Timestamps
    created_at: datetime = None
    submitted_at: datetime = None
    filled_at: datetime = None
    cancelled_at: datetime = None
    updated_at: datetime = None
    
    # Metadata
    message: str = ""
    strategy: str = ""
    signal_id: str = ""
    parent_id: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)
    
    # Risk
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    def __post_init__(self):
        if not self.id:
            self.id = f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now()
        if not self.updated_at:
            self.updated_at = datetime.now()
    
    @property
    def is_active(self) -> bool:
        return self.status in [
            OrderStatus.PENDING, 
            OrderStatus.SUBMITTED, 
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIAL
        ]
    
    @property
    def is_complete(self) -> bool:
        return self.status in [
            OrderStatus.FILLED, 
            OrderStatus.CANCELLED, 
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]
    
    @property
    def remaining_qty(self) -> int:
        return self.quantity - self.filled_qty
    
    @property
    def fill_ratio(self) -> float:
        return self.filled_qty / self.quantity if self.quantity > 0 else 0
    
    @property
    def notional_value(self) -> float:
        return self.quantity * self.price
    
    @property
    def filled_value(self) -> float:
        return self.filled_qty * self.avg_price
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'broker_id': self.broker_id,
            'symbol': self.symbol,
            'name': self.name,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status.value,
            'filled_qty': self.filled_qty,
            'avg_price': self.avg_price,
            'commission': self.commission,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'message': self.message,
            'strategy': self.strategy,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Order':
        order = cls()
        order.id = data.get('id', order.id)
        order.broker_id = data.get('broker_id', '')
        order.symbol = data.get('symbol', '')
        order.name = data.get('name', '')
        order.side = OrderSide(data.get('side', 'buy'))
        order.order_type = OrderType(data.get('order_type', 'limit'))
        order.quantity = data.get('quantity', 0)
        order.price = data.get('price', 0.0)
        order.status = OrderStatus(data.get('status', 'pending'))
        order.filled_qty = data.get('filled_qty', 0)
        order.avg_price = data.get('avg_price', 0.0)
        order.commission = data.get('commission', 0.0)
        order.message = data.get('message', '')
        order.strategy = data.get('strategy', '')
        
        if data.get('created_at'):
            order.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('filled_at'):
            order.filled_at = datetime.fromisoformat(data['filled_at'])
        
        return order


# ============================================================
# Fill
# ============================================================

@dataclass
class Fill:
    """Trade execution/fill"""
    id: str = ""
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    price: float = 0.0
    commission: float = 0.0
    stamp_tax: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if not self.id:
            self.id = f"FILL_{uuid.uuid4().hex[:12].upper()}"
        if not self.timestamp:
            self.timestamp = datetime.now()
    
    @property
    def value(self) -> float:
        return self.quantity * self.price
    
    @property
    def total_cost(self) -> float:
        return self.commission + self.stamp_tax
    
    @property
    def net_value(self) -> float:
        if self.side == OrderSide.BUY:
            return -(self.value + self.total_cost)
        else:
            return self.value - self.total_cost
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'stamp_tax': self.stamp_tax,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


# ============================================================
# Position
# ============================================================

@dataclass
class Position:
    """Position in a security"""
    symbol: str = ""
    name: str = ""
    
    # Quantities
    quantity: int = 0
    available_qty: int = 0
    frozen_qty: int = 0
    pending_buy: int = 0
    pending_sell: int = 0
    
    # Costs and prices
    avg_cost: float = 0.0
    current_price: float = 0.0
    
    # P&L
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    
    # Metadata
    opened_at: datetime = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now()
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_cost
    
    @property
    def unrealized_pnl(self) -> float:
        if self.quantity == 0 or self.avg_cost == 0:
            return 0
        return (self.current_price - self.avg_cost) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0
        return (self.unrealized_pnl / self.cost_basis) * 100
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def side(self) -> PositionSide:
        if self.quantity > 0:
            return PositionSide.LONG
        elif self.quantity < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT
    
    def update_price(self, price: float):
        """Update current price"""
        self.current_price = price
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'quantity': self.quantity,
            'available_qty': self.available_qty,
            'frozen_qty': self.frozen_qty,
            'avg_cost': self.avg_cost,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'realized_pnl': self.realized_pnl,
        }


# ============================================================
# Account
# ============================================================

@dataclass
class Account:
    """Trading account state"""
    broker_name: str = ""
    account_id: str = ""
    
    # Cash
    cash: float = 0.0
    available: float = 0.0
    frozen: float = 0.0
    
    # Positions
    positions: Dict[str, Position] = field(default_factory=dict)
    
    # P&L tracking
    initial_capital: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    
    # Daily tracking
    daily_start_equity: float = 0.0
    daily_start_date: date = None
    
    # Peak tracking
    peak_equity: float = 0.0
    
    # Metadata
    last_updated: datetime = None
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now()
        if not self.daily_start_date:
            self.daily_start_date = date.today()
    
    @property
    def positions_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())
    
    @property
    def market_value(self) -> float:
        """Alias for positions_value"""
        return self.positions_value
    
    @property
    def equity(self) -> float:
        return self.cash + self.positions_value
    
    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        return self.equity - self.initial_capital
    
    @property
    def total_pnl_pct(self) -> float:
        if self.initial_capital <= 0:
            return 0
        return (self.equity / self.initial_capital - 1) * 100
    
    @property
    def daily_pnl(self) -> float:
        return self.equity - self.daily_start_equity
    
    @property
    def daily_pnl_pct(self) -> float:
        if self.daily_start_equity <= 0:
            return 0
        return (self.equity / self.daily_start_equity - 1) * 100
    
    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0
        return self.peak_equity - self.equity
    
    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0
        return (self.peak_equity - self.equity) / self.peak_equity * 100
    
    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)
    
    def to_dict(self) -> Dict:
        return {
            'broker_name': self.broker_name,
            'equity': self.equity,
            'cash': self.cash,
            'available': self.available,
            'positions_value': self.positions_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl_pct,
            'drawdown_pct': self.drawdown_pct,
            'position_count': len(self.positions),
        }


# ============================================================
# Risk Metrics
# ============================================================

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    # Portfolio
    equity: float = 0.0
    cash: float = 0.0
    positions_value: float = 0.0
    
    # P&L
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_pnl: float = 0.0
    
    # Risk measures
    var_1d_95: float = 0.0
    var_1d_99: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    
    # Exposure
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    exposure_pct: float = 0.0
    
    # Concentration
    largest_position_pct: float = 0.0
    position_count: int = 0
    
    # Limits
    daily_loss_remaining_pct: float = 100.0
    position_limit_remaining: int = 10
    
    # Status
    risk_level: RiskLevel = RiskLevel.LOW
    can_trade: bool = True
    circuit_breaker_active: bool = False
    kill_switch_active: bool = False
    warnings: List[str] = field(default_factory=list)
    
    # Timestamps
    timestamp: datetime = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()


# ============================================================
# Trade Signal
# ============================================================

@dataclass
class TradeSignal:
    """Trade signal for execution"""
    id: str = ""
    symbol: str = ""
    name: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    price: float = 0.0
    
    # Risk levels
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Signal metadata
    confidence: float = 0.0
    strategy: str = ""
    reasons: List[str] = field(default_factory=list)
    
    # Timestamps
    generated_at: datetime = None
    expires_at: datetime = None
    
    def __post_init__(self):
        if not self.id:
            self.id = f"SIG_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
        if not self.generated_at:
            self.generated_at = datetime.now()