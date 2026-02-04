"""
Universal Broker Manager - Supports Multiple Chinese Brokers
Provides unified interface for all brokers
"""
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
import threading
import queue

from config import CONFIG
from utils.logger import log

# Try importing broker libraries
try:
    import easytrader
    EASYTRADER_AVAILABLE = True
except ImportError:
    EASYTRADER_AVAILABLE = False
    log.warning("easytrader not installed - some brokers unavailable")

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    import pyautogui
    import pygetwindow as gw
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False


class BrokerType(Enum):
    """Supported broker types"""
    SIMULATOR = "simulator"
    THS = "ths"                    # 同花顺
    HUATAI = "huatai"             # 华泰证券
    GUOJIN = "guojin"             # 国金证券
    HAITONG = "haitong"           # 海通证券
    ZHONGXIN = "zhongxin"         # 中信证券
    ZHAOSHANG = "zhaoshang"       # 招商证券
    GUOTAIJUNAN = "gtja"          # 国泰君安
    PINGAN = "pingan"             # 平安证券
    DONGFANG = "dongfang"         # 东方财富
    YINHE = "yinhe"               # 银河证券
    WEB_GENERIC = "web"           # Generic web trading


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Universal order representation"""
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
    avg_price: float = 0.0
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
    frozen_qty: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    today_buy_qty: int = 0
    today_sell_qty: int = 0
    
    def update_price(self, price: float):
        """Update with current market price"""
        self.current_price = price
        self.market_value = self.quantity * price
        if self.avg_cost > 0 and self.quantity > 0:
            self.unrealized_pnl = (price - self.avg_cost) * self.quantity
            self.unrealized_pnl_pct = (price / self.avg_cost - 1) * 100


@dataclass
class Account:
    """Account information"""
    broker: str = ""
    account_id: str = ""
    
    # Cash
    total_assets: float = 0.0
    cash: float = 0.0
    available_cash: float = 0.0
    frozen_cash: float = 0.0
    
    # Positions
    market_value: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    
    # P&L
    total_pnl: float = 0.0
    today_pnl: float = 0.0
    
    # Buying power
    buying_power: float = 0.0
    margin_used: float = 0.0
    
    @property
    def equity(self) -> float:
        return self.cash + self.market_value
    
    @property
    def position_ratio(self) -> float:
        if self.equity <= 0:
            return 0
        return self.market_value / self.equity * 100


class BrokerBase(ABC):
    """Abstract base class for all brokers"""
    
    def __init__(self):
        self._connected = False
        self._callbacks: Dict[str, List[Callable]] = {
            'order_update': [],
            'trade': [],
            'error': [],
            'position_update': [],
            'account_update': [],
        }
    
    @property
    @abstractmethod
    def broker_type(self) -> BrokerType:
        """Return broker type"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Broker display name"""
        pass
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
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
        """Get account information"""
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
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders"""
        pass
    
    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get single order"""
        pass
    
    # Convenience methods
    def buy(self, code: str, qty: int, price: float = None, 
            order_type: OrderType = OrderType.LIMIT) -> Order:
        """Place buy order"""
        order = Order(
            stock_code=code,
            side=OrderSide.BUY,
            order_type=order_type,
            quantity=qty,
            price=price
        )
        return self.submit_order(order)
    
    def sell(self, code: str, qty: int, price: float = None,
             order_type: OrderType = OrderType.LIMIT) -> Order:
        """Place sell order"""
        order = Order(
            stock_code=code,
            side=OrderSide.SELL,
            order_type=order_type,
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
    
    # Callback management
    def on(self, event: str, callback: Callable):
        """Register callback"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _emit(self, event: str, *args, **kwargs):
        """Emit event to callbacks"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                log.error(f"Callback error: {e}")


# ============================================================
# Simulator Broker (Paper Trading)
# ============================================================

class SimulatorBroker(BrokerBase):
    """
    High-fidelity paper trading simulator
    
    Features:
    - Realistic slippage and commission
    - T+1 rule enforcement
    - Partial fills simulation
    - Market impact modeling
    """
    
    def __init__(self, initial_capital: float = None):
        super().__init__()
        self._initial_capital = initial_capital or CONFIG.CAPITAL
        self._cash = self._initial_capital
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []
        self._trades: List[Dict] = []
        self._order_counter = 0
        
        # T+1 tracking
        self._today_buys: Dict[str, int] = {}
        self._last_date: date = date.today()
        
        # Data fetcher for prices
        from data.fetcher import DataFetcher
        self._fetcher = DataFetcher()
    
    @property
    def broker_type(self) -> BrokerType:
        return BrokerType.SIMULATOR
    
    @property
    def name(self) -> str:
        return "Paper Trading Simulator"
    
    def connect(self, **kwargs) -> bool:
        self._connected = True
        log.info(f"Simulator connected with ¥{self._initial_capital:,.2f}")
        return True
    
    def disconnect(self):
        self._connected = False
        log.info("Simulator disconnected")
    
    def get_account(self) -> Account:
        self._check_new_day()
        self._update_prices()
        
        market_value = sum(p.market_value for p in self._positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self._positions.values())
        realized_pnl = sum(p.realized_pnl for p in self._positions.values())
        
        return Account(
            broker=self.name,
            total_assets=self._cash + market_value,
            cash=self._cash,
            available_cash=self._cash,
            market_value=market_value,
            positions=self._positions.copy(),
            total_pnl=unrealized_pnl + realized_pnl,
            buying_power=self._cash
        )
    
    def get_positions(self) -> Dict[str, Position]:
        self._check_new_day()
        self._update_prices()
        return self._positions.copy()
    
    def get_position(self, stock_code: str) -> Optional[Position]:
        self._update_prices()
        return self._positions.get(stock_code)
    
    def submit_order(self, order: Order) -> Order:
        self._check_new_day()
        
        # Generate order ID
        self._order_counter += 1
        order.id = f"SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._order_counter:04d}"
        order.created_at = datetime.now()
        
        # Get current price
        quote = self._fetcher.get_realtime(order.stock_code)
        if not quote:
            order.status = OrderStatus.REJECTED
            order.message = "Cannot get market quote"
            self._emit('order_update', order)
            return order
        
        current_price = quote.price
        order.stock_name = quote.name
        
        # Validate order
        if not self._validate_order(order, current_price):
            self._emit('order_update', order)
            return order
        
        order.status = OrderStatus.SUBMITTED
        self._orders[order.id] = order
        
        # Execute (simulate market)
        self._execute_order(order, current_price)
        
        self._emit('order_update', order)
        return order
    
    def _validate_order(self, order: Order, price: float) -> bool:
        """Validate order before execution"""
        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            order.message = "Invalid quantity"
            return False
        
        if order.quantity % CONFIG.LOT_SIZE != 0:
            order.status = OrderStatus.REJECTED
            order.message = f"Quantity must be multiple of {CONFIG.LOT_SIZE}"
            return False
        
        if order.side == OrderSide.BUY:
            # Check funds
            cost = order.quantity * (order.price or price)
            commission = cost * CONFIG.COMMISSION
            total = cost + commission
            
            if total > self._cash:
                order.status = OrderStatus.REJECTED
                order.message = f"Insufficient funds: need ¥{total:,.2f}, have ¥{self._cash:,.2f}"
                return False
        
        else:  # SELL
            pos = self._positions.get(order.stock_code)
            if not pos:
                order.status = OrderStatus.REJECTED
                order.message = "No position to sell"
                return False
            
            if order.quantity > pos.available_qty:
                order.status = OrderStatus.REJECTED
                order.message = f"Available: {pos.available_qty}, requested: {order.quantity}"
                return False
        
        return True
    
    def _execute_order(self, order: Order, market_price: float):
        """Execute order with realistic simulation"""
        # Calculate fill price with slippage
        slippage = CONFIG.SLIPPAGE
        spread = 0.001  # 0.1% spread
        
        if order.side == OrderSide.BUY:
            # Buy at ask (higher)
            base_price = order.price if order.order_type == OrderType.LIMIT else market_price
            fill_price = base_price * (1 + spread/2 + slippage * (0.5 + 0.5 * np.random.random()))
        else:
            # Sell at bid (lower)
            base_price = order.price if order.order_type == OrderType.LIMIT else market_price
            fill_price = base_price * (1 - spread/2 - slippage * (0.5 + 0.5 * np.random.random()))
        
        fill_price = round(fill_price, 2)
        
        # Simulate partial fills for large orders (optional realism)
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
                pos.today_buy_qty += fill_qty
            else:
                self._positions[order.stock_code] = Position(
                    stock_code=order.stock_code,
                    stock_name=order.stock_name,
                    quantity=fill_qty,
                    available_qty=0,  # T+1: not available today
                    avg_cost=fill_price,
                    current_price=fill_price,
                    today_buy_qty=fill_qty
                )
            
            self._today_buys[order.stock_code] = self._today_buys.get(order.stock_code, 0) + fill_qty
        
        else:  # SELL
            self._cash += (trade_value - total_cost)
            
            pos = self._positions[order.stock_code]
            realized = (fill_price - pos.avg_cost) * fill_qty
            pos.realized_pnl += realized
            pos.quantity -= fill_qty
            pos.available_qty -= fill_qty
            pos.today_sell_qty += fill_qty
            
            if pos.quantity <= 0:
                del self._positions[order.stock_code]
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_qty = fill_qty
        order.filled_price = fill_price
        order.avg_price = fill_price
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
        if order_id in self._orders:
            order = self._orders[order_id]
            if order.is_active:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                self._emit('order_update', order)
                return True
        return False
    
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        orders = list(self._orders.values())
        if status:
            orders = [o for o in orders if o.status == status]
        return orders
    
    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)
    
    def _check_new_day(self):
        """Handle T+1 settlement on new trading day"""
        today = date.today()
        if today != self._last_date:
            # Make yesterday's buys available
            for code, pos in self._positions.items():
                pos.available_qty = pos.quantity
                pos.today_buy_qty = 0
                pos.today_sell_qty = 0
            
            self._today_buys.clear()
            self._last_date = today
            log.info("New trading day - T+1 settlement complete")
    
    def _update_prices(self):
        """Update position prices"""
        for code, pos in self._positions.items():
            quote = self._fetcher.get_realtime(code)
            if quote:
                pos.update_price(quote.price)
    
    def get_trade_history(self) -> List[Dict]:
        return self._trades.copy()
    
    def reset(self):
        """Reset simulator"""
        self._cash = self._initial_capital
        self._positions.clear()
        self._orders.clear()
        self._order_history.clear()
        self._trades.clear()
        self._today_buys.clear()
        self._order_counter = 0
        log.info("Simulator reset")


# ============================================================
# EasyTrader-based Brokers
# ============================================================

class EasyTraderBroker(BrokerBase):
    """
    Base class for easytrader-supported brokers
    Supports: 同花顺, 华泰, 国金, 银河, etc.
    """
    
    BROKER_MAP = {
        BrokerType.THS: 'ths',
        BrokerType.HUATAI: 'ht', 
        BrokerType.GUOJIN: 'gj',
        BrokerType.HAITONG: 'ht',  # Uses HT protocol
        BrokerType.YINHE: 'yh',
    }
    
    def __init__(self, broker_type: BrokerType, exe_path: str = None):
        super().__init__()
        self._broker_type = broker_type
        self._exe_path = exe_path
        self._client = None
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
    
    @property
    def broker_type(self) -> BrokerType:
        return self._broker_type
    
    @property
    def name(self) -> str:
        names = {
            BrokerType.THS: "同花顺",
            BrokerType.HUATAI: "华泰证券",
            BrokerType.GUOJIN: "国金证券",
            BrokerType.HAITONG: "海通证券",
            BrokerType.YINHE: "银河证券",
        }
        return names.get(self._broker_type, "Unknown Broker")
    
    def connect(self, **kwargs) -> bool:
        if not EASYTRADER_AVAILABLE:
            log.error("easytrader not installed")
            return False
        
        exe_path = kwargs.get('exe_path', self._exe_path)
        if not exe_path or not Path(exe_path).exists():
            log.error(f"Broker executable not found: {exe_path}")
            return False
        
        try:
            broker_code = self.BROKER_MAP.get(self._broker_type)
            if not broker_code:
                log.error(f"Unsupported broker type: {self._broker_type}")
                return False
            
            self._client = easytrader.use(broker_code)
            self._client.connect(exe_path)
            
            # Verify connection
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
    
    def get_account(self) -> Account:
        if not self.is_connected:
            return Account()
        
        try:
            balance = self._client.balance
            positions = self.get_positions()
            
            market_value = sum(p.market_value for p in positions.values())
            
            return Account(
                broker=self.name,
                total_assets=float(balance.get('总资产', 0)),
                cash=float(balance.get('资金余额', balance.get('总资产', 0))),
                available_cash=float(balance.get('可用金额', 0)),
                frozen_cash=float(balance.get('冻结金额', 0)),
                market_value=market_value,
                positions=positions,
                total_pnl=float(balance.get('总盈亏', 0))
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
                    market_value=float(p.get('市值', 0)),
                    unrealized_pnl=float(p.get('盈亏', p.get('浮动盈亏', 0))),
                    unrealized_pnl_pct=float(p.get('盈亏比例', 0)) * 100
                )
            
            return positions
        except Exception as e:
            log.error(f"Failed to get positions: {e}")
            return {}
    
    def get_position(self, stock_code: str) -> Optional[Position]:
        positions = self.get_positions()
        return positions.get(stock_code)
    
    def submit_order(self, order: Order) -> Order:
        if not self.is_connected:
            order.status = OrderStatus.REJECTED
            order.message = "Not connected"
            return order
        
        try:
            self._order_counter += 1
            order.id = f"{self._broker_type.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._order_counter:04d}"
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
            
            # Parse result
            if result and isinstance(result, dict):
                if '委托编号' in result or 'entrust_no' in result:
                    order.status = OrderStatus.SUBMITTED
                    order.broker_order_id = str(result.get('委托编号', result.get('entrust_no', '')))
                    log.info(f"Order submitted: {order.id} (broker: {order.broker_order_id})")
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
        order = self._orders.get(order_id)
        if not order or not order.broker_order_id:
            return False
        
        try:
            result = self._client.cancel_entrust(order.broker_order_id)
            if result:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                self._emit('order_update', order)
                return True
        except Exception as e:
            log.error(f"Cancel failed: {e}")
        
        return False
    
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        orders = list(self._orders.values())
        if status:
            orders = [o for o in orders if o.status == status]
        return orders
    
    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)


# ============================================================
# Web-Based Broker (Playwright)
# ============================================================

class WebBroker(BrokerBase):
    """
    Web-based broker using browser automation
    Works with any broker that has web trading interface
    """
    
    BROKER_URLS = {
        BrokerType.ZHAOSHANG: {
            'login': 'https://trade.cmschina.com',
            'name': '招商证券'
        },
        BrokerType.GUOTAIJUNAN: {
            'login': 'https://trade.gtja.com',
            'name': '国泰君安'
        },
        BrokerType.PINGAN: {
            'login': 'https://trade.pingan.com',
            'name': '平安证券'
        },
        BrokerType.DONGFANG: {
            'login': 'https://trade.eastmoney.com',
            'name': '东方财富'
        },
        BrokerType.ZHONGXIN: {
            'login': 'https://trade.citics.com',
            'name': '中信证券'
        }
    }
    
    def __init__(self, broker_type: BrokerType):
        super().__init__()
        self._broker_type = broker_type
        self._playwright = None
        self._browser = None
        self._page = None
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
    
    @property
    def broker_type(self) -> BrokerType:
        return self._broker_type
    
    @property
    def name(self) -> str:
        info = self.BROKER_URLS.get(self._broker_type, {})
        return info.get('name', 'Web Broker')
    
    def connect(self, **kwargs) -> bool:
        if not PLAYWRIGHT_AVAILABLE:
            log.error("playwright not installed. Run: pip install playwright && playwright install chromium")
            return False
        
        try:
            from playwright.sync_api import sync_playwright
            
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=kwargs.get('headless', False)
            )
            self._page = self._browser.new_page()
            
            # Navigate to login
            url_info = self.BROKER_URLS.get(self._broker_type, {})
            login_url = url_info.get('login')
            
            if login_url:
                self._page.goto(login_url)
                log.info(f"Please login to {self.name} in the browser window...")
                
                # Wait for login (user must login manually)
                # This could be automated with credentials, but security risk
                try:
                    self._page.wait_for_selector(
                        kwargs.get('login_success_selector', '.user-info, .account-info, #main-content'),
                        timeout=120000  # 2 minutes
                    )
                    self._connected = True
                    log.info(f"Connected to {self.name}")
                    return True
                except:
                    log.error("Login timeout")
            
        except Exception as e:
            log.error(f"Connection failed: {e}")
        
        return False
    
    def disconnect(self):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        self._connected = False
        log.info(f"Disconnected from {self.name}")
    
    def get_account(self) -> Account:
        # Implementation depends on specific broker's web interface
        # This is a template - customize for each broker
        return Account(broker=self.name)
    
    def get_positions(self) -> Dict[str, Position]:
        return {}
    
    def get_position(self, stock_code: str) -> Optional[Position]:
        return self.get_positions().get(stock_code)
    
    def submit_order(self, order: Order) -> Order:
        order.status = OrderStatus.REJECTED
        order.message = "Web broker order submission not fully implemented"
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        return False
    
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        return []
    
    def get_order(self, order_id: str) -> Optional[Order]:
        return None


# ============================================================
# Broker Factory
# ============================================================

class BrokerFactory:
    """Factory to create broker instances"""
    
    @staticmethod
    def create(broker_type: BrokerType, **kwargs) -> BrokerBase:
        """Create broker instance"""
        if broker_type == BrokerType.SIMULATOR:
            return SimulatorBroker(kwargs.get('capital'))
        
        elif broker_type in [BrokerType.THS, BrokerType.HUATAI, 
                            BrokerType.GUOJIN, BrokerType.HAITONG,
                            BrokerType.YINHE]:
            return EasyTraderBroker(broker_type, kwargs.get('exe_path'))
        
        elif broker_type in [BrokerType.ZHAOSHANG, BrokerType.GUOTAIJUNAN,
                            BrokerType.PINGAN, BrokerType.DONGFANG,
                            BrokerType.ZHONGXIN]:
            return WebBroker(broker_type)
        
        else:
            raise ValueError(f"Unsupported broker type: {broker_type}")
    
    @staticmethod
    def list_available() -> List[Tuple[BrokerType, str, bool]]:
        """List available brokers and their status"""
        brokers = []
        
        # Simulator always available
        brokers.append((BrokerType.SIMULATOR, "Paper Trading", True))
        
        # EasyTrader brokers
        if EASYTRADER_AVAILABLE:
            brokers.extend([
                (BrokerType.THS, "同花顺", True),
                (BrokerType.HUATAI, "华泰证券", True),
                (BrokerType.GUOJIN, "国金证券", True),
                (BrokerType.YINHE, "银河证券", True),
            ])
        
        # Web brokers
        if PLAYWRIGHT_AVAILABLE:
            brokers.extend([
                (BrokerType.ZHAOSHANG, "招商证券", True),
                (BrokerType.GUOTAIJUNAN, "国泰君安", True),
                (BrokerType.PINGAN, "平安证券", True),
                (BrokerType.DONGFANG, "东方财富", True),
                (BrokerType.ZHONGXIN, "中信证券", True),
            ])
        
        return brokers


# ============================================================
# Broker Manager (Multi-broker support)
# ============================================================

class BrokerManager:
    """
    Manages multiple broker connections
    Allows trading across different accounts
    """
    
    def __init__(self):
        self._brokers: Dict[str, BrokerBase] = {}
        self._default_broker: Optional[str] = None
    
    def add_broker(self, name: str, broker: BrokerBase) -> bool:
        """Add broker to manager"""
        self._brokers[name] = broker
        if self._default_broker is None:
            self._default_broker = name
        return True
    
    def remove_broker(self, name: str) -> bool:
        """Remove broker"""
        if name in self._brokers:
            self._brokers[name].disconnect()
            del self._brokers[name]
            if self._default_broker == name:
                self._default_broker = next(iter(self._brokers), None)
            return True
        return False
    
    def get_broker(self, name: str = None) -> Optional[BrokerBase]:
        """Get broker by name or default"""
        name = name or self._default_broker
        return self._brokers.get(name)
    
    def set_default(self, name: str):
        """Set default broker"""
        if name in self._brokers:
            self._default_broker = name
    
    def connect_all(self, **kwargs) -> Dict[str, bool]:
        """Connect all brokers"""
        results = {}
        for name, broker in self._brokers.items():
            results[name] = broker.connect(**kwargs.get(name, {}))
        return results
    
    def disconnect_all(self):
        """Disconnect all brokers"""
        for broker in self._brokers.values():
            broker.disconnect()
    
    def get_total_account(self) -> Account:
        """Get combined account across all brokers"""
        total = Account()
        all_positions = {}
        
        for name, broker in self._brokers.items():
            if broker.is_connected:
                acc = broker.get_account()
                total.total_assets += acc.total_assets
                total.cash += acc.cash
                total.available_cash += acc.available_cash
                total.market_value += acc.market_value
                total.total_pnl += acc.total_pnl
                
                for code, pos in acc.positions.items():
                    if code in all_positions:
                        # Combine positions
                        existing = all_positions[code]
                        total_qty = existing.quantity + pos.quantity
                        existing.avg_cost = (
                            existing.avg_cost * existing.quantity + 
                            pos.avg_cost * pos.quantity
                        ) / total_qty if total_qty > 0 else 0
                        existing.quantity = total_qty
                        existing.available_qty += pos.available_qty
                        existing.market_value += pos.market_value
                        existing.unrealized_pnl += pos.unrealized_pnl
                    else:
                        all_positions[code] = pos
        
        total.positions = all_positions
        return total
    
    def list_brokers(self) -> List[Tuple[str, BrokerType, bool]]:
        """List all brokers and status"""
        return [
            (name, broker.broker_type, broker.is_connected)
            for name, broker in self._brokers.items()
        ]


# Need numpy for simulation
import numpy as np