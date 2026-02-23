
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class Signal(Enum):
    """Trading signal for predictions."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

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
    IOC = "ioc"
    FOK = "fok"
    TRAIL_MARKET = "trail_market"
    TRAIL_LIMIT = "trail_limit"

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

class AutoTradeMode(Enum):
    """
    Trading mode that controls how signals are acted upon.

    MANUAL:    Signals displayed only; user must click Buy/Sell manually.
    AUTO:      Signals that meet all thresholds are executed automatically
               without user confirmation.
    SEMI_AUTO: Signals are queued and user gets a notification with
               one-click approve/reject (auto-reject after timeout).
    """
    MANUAL = "manual"
    AUTO = "auto"
    SEMI_AUTO = "semi_auto"

@dataclass
class Order:
    """Canonical Order representation"""
    id: str = ""
    client_id: str = ""
    broker_id: str = ""

    symbol: str = ""
    name: str = ""

    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.LIMIT
    quantity: int = 0
    price: float = 0.0
    stop_price: float = 0.0

    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_price: float = 0.0
    avg_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0

    created_at: datetime = None
    submitted_at: datetime = None
    filled_at: datetime = None
    cancelled_at: datetime = None
    updated_at: datetime = None

    message: str = ""
    strategy: str = ""
    signal_id: str = ""
    parent_id: str = ""
    tags: dict[str, Any] = field(default_factory=dict)

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

    def to_dict(self) -> dict:
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
    def from_dict(cls, data: dict) -> 'Order':
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
    broker_fill_id: str = ""

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

    def to_dict(self) -> dict:
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

@dataclass
class Position:
    """Position in a security"""
    symbol: str = ""
    name: str = ""

    quantity: int = 0
    available_qty: int = 0
    frozen_qty: int = 0
    pending_buy: int = 0
    pending_sell: int = 0

    avg_cost: float = 0.0
    current_price: float = 0.0

    # P&L
    realized_pnl: float = 0.0
    commission_paid: float = 0.0

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

    def to_dict(self) -> dict:
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

@dataclass
class Account:
    """Trading account state"""
    broker_name: str = ""
    account_id: str = ""

    cash: float = 0.0
    available: float = 0.0
    frozen: float = 0.0

    positions: dict[str, Position] = field(default_factory=dict)

    # P&L tracking
    initial_capital: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0

    daily_start_equity: float = 0.0
    daily_start_date: date = None

    peak_equity: float = 0.0

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

    def get_position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    def to_dict(self) -> dict:
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

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    equity: float = 0.0
    cash: float = 0.0
    positions_value: float = 0.0

    # P&L
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_pnl: float = 0.0

    var_1d_95: float = 0.0
    var_1d_99: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0

    long_exposure: float = 0.0
    short_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    exposure_pct: float = 0.0

    largest_position_pct: float = 0.0
    position_count: int = 0

    daily_loss_remaining_pct: float = 100.0
    position_limit_remaining: int = 10

    risk_level: RiskLevel = RiskLevel.LOW
    can_trade: bool = True
    circuit_breaker_active: bool = False
    kill_switch_active: bool = False
    warnings: list[str] = field(default_factory=list)

    timestamp: datetime = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()

@dataclass
class TradeSignal:
    """Trade signal for execution"""
    id: str = ""
    symbol: str = ""
    name: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    price: float = 0.0

    stop_loss: float = 0.0
    take_profit: float = 0.0

    confidence: float = 0.0
    strategy: str = ""
    reasons: list[str] = field(default_factory=list)

    generated_at: datetime = None
    expires_at: datetime = None

    # Auto-trade metadata
    auto_generated: bool = False
    auto_trade_action_id: str = ""
    approvals_count: int = 0
    approved_by: list[str] = field(default_factory=list)
    change_ticket: str = ""
    business_justification: str = ""
    order_type: str = "limit"
    time_in_force: str = "day"
    strict_time_in_force: bool = False
    trigger_price: float = 0.0
    trailing_stop_pct: float = 0.0
    trail_limit_offset_pct: float = 0.0
    oco_group: str = ""
    bracket: bool = False

    def __post_init__(self):
        if not self.id:
            self.id = f"SIG_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
        if not self.generated_at:
            self.generated_at = datetime.now()

# Auto-Trade Types

@dataclass
class AutoTradeAction:
    """
    Records a single auto-trade decision (executed, skipped, or pending).

    Used for:
    - Audit trail of all auto-trade decisions
    - UI display of recent actions
    - Post-session analysis
    """
    id: str = ""
    timestamp: datetime = None

    stock_code: str = ""
    stock_name: str = ""
    signal_type: str = ""       # e.g. "STRONG_BUY", "SELL"
    confidence: float = 0.0
    signal_strength: float = 0.0
    model_agreement: float = 0.0

    price: float = 0.0
    predicted_direction: str = ""  # "UP", "DOWN", "NEUTRAL"

    decision: str = ""  # "EXECUTED", "SKIPPED", "PENDING", "REJECTED", "EXPIRED"
    skip_reason: str = ""

    side: str = ""          # "buy" or "sell"
    quantity: int = 0
    order_id: str = ""
    signal_id: str = ""

    # Outcome (filled in later when fill arrives)
    fill_price: float = 0.0
    fill_quantity: int = 0
    realized_pnl: float = 0.0
    outcome: str = ""  # "WIN", "LOSS", "PENDING", ""

    def __post_init__(self):
        if not self.id:
            self.id = f"ATA_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
        if not self.timestamp:
            self.timestamp = datetime.now()

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'signal_type': self.signal_type,
            'confidence': self.confidence,
            'signal_strength': self.signal_strength,
            'price': self.price,
            'decision': self.decision,
            'skip_reason': self.skip_reason,
            'side': self.side,
            'quantity': self.quantity,
            'order_id': self.order_id,
            'fill_price': self.fill_price,
            'outcome': self.outcome,
        }

@dataclass
class AutoTradeState:
    """
    Runtime state of the auto-trader.

    Tracks:
    - Whether auto-trading is currently active
    - Current mode (MANUAL / AUTO / SEMI_AUTO)
    - Session statistics (trades today, wins, losses)
    - Recent actions for UI display
    - Cooldown timers per stock
    - Safety pause state
    """
    mode: AutoTradeMode = AutoTradeMode.MANUAL
    is_running: bool = False

    # Session counters (reset daily)
    trades_today: int = 0
    trades_per_stock_today: dict[str, int] = field(default_factory=dict)
    buys_today: int = 0
    sells_today: int = 0
    skipped_today: int = 0

    # P&L tracking for auto-trades only
    auto_trade_pnl: float = 0.0
    auto_trade_wins: int = 0
    auto_trade_losses: int = 0

    # Cooldowns: stock_code -> datetime when cooldown expires
    cooldowns: dict[str, datetime] = field(default_factory=dict)

    is_paused: bool = False
    pause_reason: str = ""
    pause_until: datetime = None

    # Recent actions (bounded list, newest first)
    recent_actions: list[AutoTradeAction] = field(default_factory=list)

    # Pending approvals (for SEMI_AUTO mode)
    pending_approvals: list[AutoTradeAction] = field(default_factory=list)

    last_scan_time: datetime = None
    last_trade_time: datetime = None

    session_start: datetime = None
    session_date: date = None

    consecutive_errors: int = 0
    last_error: str = ""
    last_error_time: datetime = None

    MAX_RECENT_ACTIONS: int = 200
    MAX_PENDING_APPROVALS: int = 20

    def __post_init__(self):
        if not self.session_start:
            self.session_start = datetime.now()
        if not self.session_date:
            self.session_date = date.today()

    def reset_daily(self):
        """Reset daily counters. Called at start of each trading day."""
        self.trades_today = 0
        self.trades_per_stock_today.clear()
        self.buys_today = 0
        self.sells_today = 0
        self.skipped_today = 0
        self.auto_trade_pnl = 0.0
        self.auto_trade_wins = 0
        self.auto_trade_losses = 0
        self.cooldowns.clear()
        self.consecutive_errors = 0
        self.last_error = ""
        self.is_paused = False
        self.pause_reason = ""
        self.pause_until = None
        self.session_date = date.today()
        self.session_start = datetime.now()

    def record_action(self, action: AutoTradeAction):
        """Add an action to recent history (bounded)."""
        self.recent_actions.insert(0, action)
        if len(self.recent_actions) > self.MAX_RECENT_ACTIONS:
            self.recent_actions = self.recent_actions[:self.MAX_RECENT_ACTIONS]

    def add_pending_approval(self, action: AutoTradeAction):
        """Add a pending approval for SEMI_AUTO mode."""
        self.pending_approvals.insert(0, action)
        if len(self.pending_approvals) > self.MAX_PENDING_APPROVALS:
            expired = self.pending_approvals[self.MAX_PENDING_APPROVALS:]
            for a in expired:
                a.decision = "EXPIRED"
                a.skip_reason = "Too many pending approvals"
                self.record_action(a)
            self.pending_approvals = self.pending_approvals[:self.MAX_PENDING_APPROVALS]

    def remove_pending(self, action_id: str) -> AutoTradeAction | None:
        """Remove and return a pending approval by ID."""
        for i, a in enumerate(self.pending_approvals):
            if a.id == action_id:
                return self.pending_approvals.pop(i)
        return None

    def is_on_cooldown(self, stock_code: str) -> bool:
        """Check if a stock is on cooldown."""
        expiry = self.cooldowns.get(stock_code)
        if expiry is None:
            return False
        if datetime.now() >= expiry:
            del self.cooldowns[stock_code]
            return False
        return True

    def set_cooldown(self, stock_code: str, seconds: int):
        """Set cooldown for a stock."""
        from datetime import timedelta
        self.cooldowns[stock_code] = datetime.now() + timedelta(seconds=seconds)

    def can_trade_stock(self, stock_code: str, max_per_stock: int) -> bool:
        """Check if we can trade this stock today."""
        count = self.trades_per_stock_today.get(stock_code, 0)
        return count < max_per_stock

    def record_trade(self, stock_code: str, side: str):
        """Record a trade execution."""
        self.trades_today += 1
        self.trades_per_stock_today[stock_code] = (
            self.trades_per_stock_today.get(stock_code, 0) + 1
        )
        if side == "buy":
            self.buys_today += 1
        else:
            self.sells_today += 1
        self.last_trade_time = datetime.now()
        self.consecutive_errors = 0

    def record_error(self, error: str):
        """Record an error."""
        self.consecutive_errors += 1
        self.last_error = error
        self.last_error_time = datetime.now()

    def record_skip(self):
        """Record a skipped signal."""
        self.skipped_today += 1

    @property
    def win_rate(self) -> float:
        """Win rate of auto-trades."""
        total = self.auto_trade_wins + self.auto_trade_losses
        if total == 0:
            return 0.0
        return self.auto_trade_wins / total

    @property
    def is_safety_paused(self) -> bool:
        """Check if auto-trading is paused for safety."""
        if not self.is_paused:
            return False
        if self.pause_until and datetime.now() >= self.pause_until:
            self.is_paused = False
            self.pause_reason = ""
            self.pause_until = None
            return False
        return True

    def to_dict(self) -> dict:
        return {
            'mode': self.mode.value,
            'is_running': self.is_running,
            'trades_today': self.trades_today,
            'buys_today': self.buys_today,
            'sells_today': self.sells_today,
            'skipped_today': self.skipped_today,
            'auto_trade_pnl': self.auto_trade_pnl,
            'win_rate': self.win_rate,
            'is_paused': self.is_safety_paused,
            'pause_reason': self.pause_reason,
            'consecutive_errors': self.consecutive_errors,
            'last_scan': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'last_trade': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'pending_approvals': len(self.pending_approvals),
            'recent_actions': len(self.recent_actions),
        }
