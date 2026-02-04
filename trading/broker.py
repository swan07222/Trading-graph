"""
Unified Broker Interface - SINGLE source of truth for all trading operations

This module contains:
- All enums (OrderSide, OrderType, OrderStatus)
- All dataclasses (Order, Position, Account)
- Abstract BrokerInterface
- SimulatorBroker implementation
- THSBroker implementation (via easytrader)

Author: AI Trading System v3.0
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path
import threading
import uuid
import time
import json

from config import CONFIG
from utils.logger import log


# ============================================================
# Enums - SINGLE DEFINITIONS
# ============================================================

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


# ============================================================
# Dataclasses - SINGLE DEFINITIONS
# ============================================================

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
    available_qty: int = 0
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
    """Account state - UNIFIED FIELD NAMES"""
    broker_name: str = ""
    cash: float = 0.0
    available: float = 0.0  # Available for trading
    frozen: float = 0.0
    market_value: float = 0.0
    total_pnl: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def equity(self) -> float:
        return self.cash + self.market_value


# ============================================================
# Abstract Broker Interface
# ============================================================

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
    
    @abstractmethod
    def get_quote(self, stock_code: str) -> Optional[float]:
        """Get current price for a stock"""
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


# ============================================================
# Simulator Broker
# ============================================================

class SimulatorBroker(BrokerInterface):
    """
    Paper trading simulator with realistic behavior.
    
    Features:
    - Realistic slippage and commission
    - T+1 rule enforcement
    - Thread-safe operations
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
        
        # T+1 tracking
        self._purchase_dates: Dict[str, date] = {}
        self._last_settlement_date = date.today()
        
        # Data fetcher (lazy init)
        self._fetcher = None
    
    def _get_fetcher(self):
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
    
    def get_quote(self, stock_code: str) -> Optional[float]:
        """Get current price"""
        fetcher = self._get_fetcher()
        quote = fetcher.get_realtime(stock_code)
        return quote.price if quote and quote.price > 0 else None
    
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
                available=self._cash,
                frozen=0.0,
                market_value=market_value,
                total_pnl=unrealized_pnl + realized_pnl,
                positions=dict(self._positions),
                timestamp=datetime.now()
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
                price = self.get_quote(stock_code)
                if price:
                    pos.update_price(price)
            return pos
    
    def submit_order(self, order: Order) -> Order:
        import random
        
        with self._lock:
            self._check_settlement()
            
            order.id = f"SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
            order.created_at = datetime.now()
            
            # Get current price
            current_price = self.get_quote(order.stock_code)
            
            if current_price is None or current_price <= 0:
                order.status = OrderStatus.REJECTED
                order.message = "Cannot get market quote"
                self._emit('order_update', order)
                return order
            
            # Get stock name
            fetcher = self._get_fetcher()
            quote = fetcher.get_realtime(order.stock_code)
            order.stock_name = quote.name if quote else order.stock_code
            
            # Validate order
            validation = self._validate_order(order, current_price)
            if not validation[0]:
                order.status = OrderStatus.REJECTED
                order.message = validation[1]
                self._emit('order_update', order)
                return order
            
            order.status = OrderStatus.SUBMITTED
            self._orders[order.id] = order
            
            # Execute immediately
            self._execute_order(order, current_price)
            
            self._emit('order_update', order)
            return order
    
    def _validate_order(self, order: Order, price: float) -> Tuple[bool, str]:
        """Validate order"""
        if order.quantity <= 0:
            return False, "Quantity must be positive"
        
        if order.quantity % CONFIG.LOT_SIZE != 0:
            return False, f"Quantity must be multiple of {CONFIG.LOT_SIZE}"
        
        if order.side == OrderSide.BUY:
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
        
        return True, "OK"
    
    def _execute_order(self, order: Order, market_price: float):
        """Execute order with realistic simulation"""
        import random
        
        slippage = CONFIG.SLIPPAGE
        
        if order.side == OrderSide.BUY:
            base_price = order.price if order.order_type == OrderType.LIMIT else market_price
            fill_price = base_price * (1 + slippage * (0.5 + 0.5 * random.random()))
        else:
            base_price = order.price if order.order_type == OrderType.LIMIT else market_price
            fill_price = base_price * (1 - slippage * (0.5 + 0.5 * random.random()))
        
        fill_price = round(fill_price, 2)
        fill_qty = order.quantity
        
        trade_value = fill_qty * fill_price
        commission = trade_value * CONFIG.COMMISSION
        stamp_tax = trade_value * CONFIG.STAMP_TAX if order.side == OrderSide.SELL else 0
        total_cost = commission + stamp_tax
        
        if order.side == OrderSide.BUY:
            self._cash -= (trade_value + total_cost)
            
            if order.stock_code in self._positions:
                pos = self._positions[order.stock_code]
                total_qty = pos.quantity + fill_qty
                pos.avg_cost = (pos.avg_cost * pos.quantity + fill_price * fill_qty) / total_qty
                pos.quantity = total_qty
            else:
                self._positions[order.stock_code] = Position(
                    stock_code=order.stock_code,
                    stock_name=order.stock_name,
                    quantity=fill_qty,
                    available_qty=0,  # T+1
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
        
        order.status = OrderStatus.FILLED
        order.filled_qty = fill_qty
        order.filled_price = fill_price
        order.commission = total_cost
        order.updated_at = datetime.now()
        
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
            for code, pos in self._positions.items():
                pos.available_qty = pos.quantity
            self._last_settlement_date = today
            log.info("T+1 settlement: all shares now available")
    
    def _update_prices(self):
        """Update all position prices"""
        for code, pos in self._positions.items():
            price = self.get_quote(code)
            if price:
                pos.update_price(price)
    
    def get_trade_history(self) -> List[Dict]:
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


# ============================================================
# THS Broker (Real Trading)
# ============================================================

class THSBroker(BrokerInterface):
    """
    TongHuaShun (同花顺) broker integration via easytrader.
    
    Also works with:
    - 华泰证券 (Huatai)
    - 国金证券 (Guojin)
    - 银河证券 (Yinhe)
    """
    
    BROKER_TYPES = {
        'ths': '同花顺',
        'ht': '华泰证券',
        'gj': '国金证券',
        'yh': '银河证券',
    }
    
    def __init__(self, broker_type: str = 'ths'):
        super().__init__()
        self._broker_type = broker_type
        self._client = None
        self._connected = False
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
        
        # Check easytrader availability
        try:
            import easytrader
            self._easytrader = easytrader
            self._available = True
        except ImportError:
            self._easytrader = None
            self._available = False
            log.warning("easytrader not installed - live trading unavailable")
    
    @property
    def name(self) -> str:
        return self.BROKER_TYPES.get(self._broker_type, "Unknown Broker")
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None
    
    def connect(self, exe_path: str = None, **kwargs) -> bool:
        if not self._available:
            log.error("easytrader not installed")
            return False
        
        exe_path = exe_path or CONFIG.BROKER_PATH
        if not exe_path or not Path(exe_path).exists():
            log.error(f"Broker executable not found: {exe_path}")
            return False
        
        try:
            self._client = self._easytrader.use(self._broker_type)
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
    
    def get_quote(self, stock_code: str) -> Optional[float]:
        """Get current price via data fetcher"""
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        quote = fetcher.get_realtime(stock_code)
        return quote.price if quote and quote.price > 0 else None
    
    def get_account(self) -> Account:
        if not self.is_connected:
            return Account()
        
        try:
            balance = self._client.balance
            positions = self.get_positions()
            
            market_value = sum(p.market_value for p in positions.values())
            
            return Account(
                broker_name=self.name,
                cash=float(balance.get('资金余额', balance.get('总资产', 0))),
                available=float(balance.get('可用金额', 0)),
                frozen=float(balance.get('冻结金额', 0)),
                market_value=market_value,
                total_pnl=float(balance.get('总盈亏', 0)),
                positions=positions,
                timestamp=datetime.now()
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
                code = str(p.get('证券代码', '')).zfill(6)
                
                positions[code] = Position(
                    stock_code=code,
                    stock_name=p.get('证券名称', ''),
                    quantity=int(p.get('股票余额', p.get('当前持仓', 0))),
                    available_qty=int(p.get('可卖余额', p.get('可用余额', 0))),
                    avg_cost=float(p.get('成本价', p.get('买入成本', 0))),
                    current_price=float(p.get('当前价', p.get('最新价', 0))),
                    unrealized_pnl=float(p.get('盈亏', p.get('浮动盈亏', 0))),
                    unrealized_pnl_pct=float(p.get('盈亏比例', 0)) * 100
                )
            
            return positions
        except Exception as e:
            log.error(f"Failed to get positions: {e}")
            return {}
    
    def get_position(self, stock_code: str) -> Optional[Position]:
        return self.get_positions().get(stock_code)
    
    def submit_order(self, order: Order) -> Order:
        if not self.is_connected:
            order.status = OrderStatus.REJECTED
            order.message = "Not connected"
            return order
        
        try:
            self._order_counter += 1
            order.id = f"THS_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._order_counter:04d}"
            order.created_at = datetime.now()
            
            if order.side == OrderSide.BUY:
                if order.order_type == OrderType.MARKET:
                    result = self._client.market_buy(order.stock_code, order.quantity)
                else:
                    result = self._client.buy(order.stock_code, order.quantity, order.price)
            else:
                if order.order_type == OrderType.MARKET:
                    result = self._client.market_sell(order.stock_code, order.quantity)
                else:
                    result = self._client.sell(order.stock_code, order.quantity, order.price)
            
            if result and isinstance(result, dict):
                if '委托编号' in result or 'entrust_no' in result:
                    order.status = OrderStatus.SUBMITTED
                    order.message = str(result.get('委托编号', result.get('entrust_no', '')))
                    log.info(f"Order submitted: {order.id}")
                else:
                    order.status = OrderStatus.REJECTED
                    order.message = str(result)
            else:
                order.status = OrderStatus.REJECTED
                order.message = "Unknown response"
            
            self._orders[order.id] = order
            self._emit('order_update', order)
            return order
            
        except Exception as e:
            log.error(f"Order error: {e}")
            order.status = OrderStatus.REJECTED
            order.message = str(e)
            return order
    
    def cancel_order(self, order_id: str) -> bool:
        if not self.is_connected:
            return False
        
        order = self._orders.get(order_id)
        if not order or not order.message:  # message contains broker order ID
            return False
        
        try:
            self._client.cancel_entrust(order.message)
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            self._emit('order_update', order)
            return True
        except Exception as e:
            log.error(f"Cancel failed: {e}")
            return False
    
    def get_orders(self, active_only: bool = True) -> List[Order]:
        if active_only:
            return [o for o in self._orders.values() if o.is_active]
        return list(self._orders.values())


# ============================================================
# Broker Factory
# ============================================================

def create_broker(mode: str = None, **kwargs) -> BrokerInterface:
    """
    Factory function to create appropriate broker.
    
    Args:
        mode: 'simulation', 'ths', 'ht', 'gj', 'yh'
        **kwargs: Additional arguments for broker
        
    Returns:
        BrokerInterface instance
    """
    mode = mode or CONFIG.TRADING_MODE.value
    
    if mode == 'simulation':
        return SimulatorBroker(kwargs.get('capital', CONFIG.CAPITAL))
    elif mode in ['ths', 'ht', 'gj', 'yh']:
        return THSBroker(broker_type=mode)
    else:
        log.warning(f"Unknown broker mode: {mode}, using simulator")
        return SimulatorBroker()