"""
Unified Broker Interface - Uses canonical types from core.types

Supports:
- Paper Trading (Simulator)
- 同花顺 (THS)
- 华泰证券 (HT)
- 招商证券 (ZSZQ)
- 国金证券 (GJ)
- 银河证券 (YH)
"""
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path
import threading
import uuid
import time
import json

from config import CONFIG
from core.types import (
    Order, OrderSide, OrderType, OrderStatus,
    Position, Account, Fill
)
from utils.logger import log


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
    def get_position(self, symbol: str) -> Optional[Position]:
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
    def get_quote(self, symbol: str) -> Optional[float]:
        """Get current price for a stock"""
        pass
    
    # === Convenience Methods ===
    
    def buy(self, symbol: str, qty: int, price: float = None) -> Order:
        """Place buy order"""
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            quantity=qty,
            price=price or 0.0
        )
        return self.submit_order(order)
    
    def sell(self, symbol: str, qty: int, price: float = None) -> Order:
        """Place sell order"""
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            quantity=qty,
            price=price or 0.0
        )
        return self.submit_order(order)
    
    def sell_all(self, symbol: str, price: float = None) -> Optional[Order]:
        """Sell entire position"""
        pos = self.get_position(symbol)
        if pos and pos.available_qty > 0:
            return self.sell(symbol, pos.available_qty, price)
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
        self._initial_capital = initial_capital or CONFIG.capital
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
    
    def get_quote(self, symbol: str) -> Optional[float]:
        """Get current price"""
        fetcher = self._get_fetcher()
        quote = fetcher.get_realtime(symbol)
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
                positions=dict(self._positions),
                initial_capital=self._initial_capital,
                realized_pnl=realized_pnl,
                last_updated=datetime.now()
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
    
    def submit_order(self, order: Order) -> Order:
        import random
        
        with self._lock:
            self._check_settlement()
            
            order.id = f"SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
            order.created_at = datetime.now()
            
            # Get current price
            current_price = self.get_quote(order.symbol)
            
            if current_price is None or current_price <= 0:
                order.status = OrderStatus.REJECTED
                order.message = "Cannot get market quote"
                self._emit('order_update', order)
                return order
            
            # Get stock name
            fetcher = self._get_fetcher()
            quote = fetcher.get_realtime(order.symbol)
            order.name = quote.name if quote else order.symbol
            
            # Use limit price if provided, otherwise market price
            exec_price = order.price if order.price > 0 else current_price
            
            # Validate order
            validation = self._validate_order(order, exec_price)
            if not validation[0]:
                order.status = OrderStatus.REJECTED
                order.message = validation[1]
                self._emit('order_update', order)
                return order
            
            order.status = OrderStatus.SUBMITTED
            self._orders[order.id] = order
            
            # Execute immediately
            self._execute_order(order, exec_price)
            
            self._emit('order_update', order)
            return order
    
    def _validate_order(self, order: Order, price: float) -> Tuple[bool, str]:
        """Validate order"""
        if order.quantity <= 0:
            return False, "Quantity must be positive"
        
        lot_size = getattr(CONFIG, 'LOT_SIZE', 100)
        if order.quantity % lot_size != 0:
            return False, f"Quantity must be multiple of {lot_size}"
        
        if order.side == OrderSide.BUY:
            cost = order.quantity * price
            commission = cost * CONFIG.COMMISSION
            total = cost + commission
            
            if total > self._cash:
                return False, f"Insufficient funds: need ¥{total:,.2f}, have ¥{self._cash:,.2f}"
            
            # Check position limit
            existing_value = 0.0
            existing_pos = self._positions.get(order.symbol)
            if existing_pos:
                existing_value = existing_pos.quantity * price
            
            new_total_value = existing_value + (order.quantity * price)
            equity = self._cash + sum(p.market_value for p in self._positions.values())
            
            if equity > 0:
                max_pct = getattr(CONFIG, 'MAX_POSITION_PCT', 15.0)
                position_pct = new_total_value / equity * 100
                if position_pct > max_pct:
                    return False, f"Position too large: {position_pct:.1f}% (max: {max_pct}%)"
            
        else:  # SELL
            pos = self._positions.get(order.symbol)
            
            if not pos:
                return False, f"No position in {order.symbol}"
            
            if order.quantity > pos.available_qty:
                return False, f"Available: {pos.available_qty}, requested: {order.quantity}"
        
        return True, "OK"
    
    def _execute_order(self, order: Order, market_price: float):
        """Execute order with realistic simulation"""
        import random
        
        slippage = CONFIG.SLIPPAGE
        
        if order.side == OrderSide.BUY:
            fill_price = market_price * (1 + slippage * (0.5 + 0.5 * random.random()))
        else:
            fill_price = market_price * (1 - slippage * (0.5 + 0.5 * random.random()))
        
        fill_price = round(fill_price, 2)
        fill_qty = order.quantity
        
        trade_value = fill_qty * fill_price
        commission = trade_value * CONFIG.COMMISSION
        stamp_tax = trade_value * CONFIG.STAMP_TAX if order.side == OrderSide.SELL else 0
        total_cost = commission + stamp_tax
        
        if order.side == OrderSide.BUY:
            self._cash -= (trade_value + total_cost)
            
            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                total_qty = pos.quantity + fill_qty
                pos.avg_cost = (pos.avg_cost * pos.quantity + fill_price * fill_qty) / total_qty
                pos.quantity = total_qty
            else:
                self._positions[order.symbol] = Position(
                    symbol=order.symbol,
                    name=order.name,
                    quantity=fill_qty,
                    available_qty=0,  # T+1
                    avg_cost=fill_price,
                    current_price=fill_price
                )
            
            self._purchase_dates[order.symbol] = date.today()
            
        else:  # SELL
            self._cash += (trade_value - total_cost)
            
            pos = self._positions[order.symbol]
            gross_pnl = (fill_price - pos.avg_cost) * fill_qty
            realized = gross_pnl - total_cost
            pos.realized_pnl += realized
            pos.quantity -= fill_qty
            pos.available_qty -= fill_qty
            
            if pos.quantity <= 0:
                del self._positions[order.symbol]
                if order.symbol in self._purchase_dates:
                    del self._purchase_dates[order.symbol]
        
        order.status = OrderStatus.FILLED
        order.filled_qty = fill_qty
        order.filled_price = fill_price
        order.avg_price = fill_price
        order.commission = total_cost
        order.updated_at = datetime.now()
        order.filled_at = datetime.now()
        
        # Create Fill record
        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=fill_price,
            commission=commission,
            stamp_tax=stamp_tax,
            timestamp=datetime.now()
        )
        
        self._trades.append({
            'order_id': order.id,
            'symbol': order.symbol,
            'name': order.name,
            'side': order.side.value,
            'quantity': fill_qty,
            'price': fill_price,
            'value': trade_value,
            'commission': commission,
            'stamp_tax': stamp_tax,
            'timestamp': datetime.now()
        })
        self._order_history.append(order)
        
        log.info(
            f"[SIM] {order.side.value.upper()} {fill_qty} {order.symbol} "
            f"@ ¥{fill_price:.2f} (cost: ¥{total_cost:.2f})"
        )
        
        self._emit('trade', order, fill)
    
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
                return [o for o in self._orders.values() if o.is_active]
            return list(self._orders.values())
    
    def _check_settlement(self):
        """Handle T+1 settlement on new trading day"""
        today = date.today()
        if today != self._last_settlement_date:
            for symbol, pos in self._positions.items():
                pos.available_qty = pos.quantity
            self._last_settlement_date = today
            log.info("T+1 settlement: all shares now available")
    
    def _update_prices(self):
        """Update all position prices"""
        for symbol, pos in self._positions.items():
            price = self.get_quote(symbol)
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
    
    def reconcile(self) -> Dict:
        """Reconcile internal state"""
        with self._lock:
            return {
                'cash_diff': 0.0,
                'position_diffs': [],
                'missing_positions': [],
                'extra_positions': [],
                'reconciled': True,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_execution_journal(self) -> List[Dict]:
        """Get execution journal for audit"""
        with self._lock:
            return [
                {
                    'timestamp': t['timestamp'].isoformat(),
                    'order_id': t['order_id'],
                    'symbol': t['symbol'],
                    'side': t['side'],
                    'quantity': t['quantity'],
                    'price': t['price'],
                    'commission': t['commission'],
                    'stamp_tax': t.get('stamp_tax', 0)
                }
                for t in self._trades
            ]


# ============================================================
# THS Broker (Real Trading)
# ============================================================

class THSBroker(BrokerInterface):
    """
    TongHuaShun (同花顺) broker integration via easytrader.
    
    Also works with:
    - 华泰证券 (Huatai) - 'ht'
    - 国金证券 (Guojin) - 'gj'
    - 银河证券 (Yinhe) - 'yh'
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
    
    def get_quote(self, symbol: str) -> Optional[float]:
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        quote = fetcher.get_realtime(symbol)
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
                positions=positions,
                last_updated=datetime.now()
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
                    symbol=code,
                    name=p.get('证券名称', ''),
                    quantity=int(p.get('股票余额', p.get('当前持仓', 0))),
                    available_qty=int(p.get('可卖余额', p.get('可用余额', 0))),
                    avg_cost=float(p.get('成本价', p.get('买入成本', 0))),
                    current_price=float(p.get('当前价', p.get('最新价', 0))),
                    unrealized_pnl=float(p.get('盈亏', p.get('浮动盈亏', 0))),
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
            order.message = "Not connected"
            return order
        
        try:
            self._order_counter += 1
            order.id = f"THS_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._order_counter:04d}"
            order.created_at = datetime.now()
            
            if order.side == OrderSide.BUY:
                if order.order_type == OrderType.MARKET:
                    result = self._client.market_buy(order.symbol, order.quantity)
                else:
                    result = self._client.buy(order.symbol, order.quantity, order.price)
            else:
                if order.order_type == OrderType.MARKET:
                    result = self._client.market_sell(order.symbol, order.quantity)
                else:
                    result = self._client.sell(order.symbol, order.quantity, order.price)
            
            if result and isinstance(result, dict):
                if '委托编号' in result or 'entrust_no' in result:
                    order.status = OrderStatus.SUBMITTED
                    order.broker_id = str(result.get('委托编号', result.get('entrust_no', '')))
                    order.message = f"委托编号: {order.broker_id}"
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
        if not order or not order.broker_id:
            return False
        
        try:
            self._client.cancel_entrust(order.broker_id)
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
# ZSZQ Broker (招商证券)
# ============================================================

class ZSZQBroker(BrokerInterface):
    """
    招商证券 (Zhao Shang Zheng Quan) broker integration
    
    Uses easytrader with 'universal' type or direct API
    """
    
    def __init__(self):
        super().__init__()
        self._client = None
        self._connected = False
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
        
        try:
            import easytrader
            self._easytrader = easytrader
            self._available = True
        except ImportError:
            self._easytrader = None
            self._available = False
            log.warning("easytrader not installed - ZSZQ trading unavailable")
    
    @property
    def name(self) -> str:
        return "招商证券"
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None
    
    def connect(self, exe_path: str = None, **kwargs) -> bool:
        if not self._available:
            log.error("easytrader not installed")
            return False
        
        exe_path = exe_path or kwargs.get('broker_path') or CONFIG.BROKER_PATH
        
        if not exe_path:
            log.error("Broker executable path not configured")
            return False
        
        if not Path(exe_path).exists():
            log.error(f"Broker executable not found: {exe_path}")
            return False
        
        try:
            self._client = self._easytrader.use('universal')
            self._client.connect(exe_path)
            
            balance = self._client.balance
            if balance:
                self._connected = True
                log.info(f"Connected to {self.name}")
                return True
                
        except Exception as e:
            log.error(f"ZSZQ connection failed: {e}")
            
            try:
                self._client = self._easytrader.use('ths')
                self._client.prepare(
                    user=kwargs.get('user', ''),
                    password=kwargs.get('password', '')
                )
                self._client.connect(exe_path)
                
                if self._client.balance:
                    self._connected = True
                    log.info(f"Connected to {self.name} (alternative method)")
                    return True
            except Exception as e2:
                log.error(f"Alternative connection also failed: {e2}")
        
        return False
    
    def disconnect(self):
        self._client = None
        self._connected = False
        log.info(f"Disconnected from {self.name}")
    
    def get_quote(self, symbol: str) -> Optional[float]:
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        quote = fetcher.get_realtime(symbol)
        return quote.price if quote and quote.price > 0 else None
    
    def get_account(self) -> Account:
        if not self.is_connected:
            return Account()
        
        try:
            balance = self._client.balance
            positions = self.get_positions()
            
            cash = float(
                balance.get('资金余额') or 
                balance.get('总资产') or 
                balance.get('可用资金') or 
                0
            )
            
            available = float(
                balance.get('可用金额') or 
                balance.get('可用资金') or 
                balance.get('可取资金') or 
                cash
            )
            
            frozen = float(balance.get('冻结金额', 0) or 0)
            
            return Account(
                broker_name=self.name,
                cash=cash,
                available=available,
                frozen=frozen,
                positions=positions,
                last_updated=datetime.now()
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
                    p.get('证券代码') or 
                    p.get('股票代码') or 
                    ''
                ).zfill(6)
                
                if not code or code == '000000':
                    continue
                
                positions[code] = Position(
                    symbol=code,
                    name=p.get('证券名称') or p.get('股票名称') or '',
                    quantity=int(p.get('股票余额') or p.get('持仓数量') or p.get('当前持仓') or 0),
                    available_qty=int(p.get('可卖余额') or p.get('可用余额') or p.get('可卖数量') or 0),
                    avg_cost=float(p.get('成本价') or p.get('买入成本') or p.get('参考成本价') or 0),
                    current_price=float(p.get('当前价') or p.get('最新价') or p.get('市价') or 0),
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
            self._order_counter += 1
            order.id = f"ZSZQ_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._order_counter:04d}"
            order.created_at = datetime.now()
            
            if order.side == OrderSide.BUY:
                if order.order_type == OrderType.MARKET:
                    result = self._client.market_buy(order.symbol, order.quantity)
                else:
                    result = self._client.buy(order.symbol, order.quantity, order.price)
            else:
                if order.order_type == OrderType.MARKET:
                    result = self._client.market_sell(order.symbol, order.quantity)
                else:
                    result = self._client.sell(order.symbol, order.quantity, order.price)
            
            if result and isinstance(result, dict):
                entrust_no = result.get('委托编号') or result.get('entrust_no') or result.get('order_id')
                
                if entrust_no:
                    order.status = OrderStatus.SUBMITTED
                    order.broker_id = str(entrust_no)
                    order.message = f"委托编号: {entrust_no}"
                    log.info(f"Order submitted: {order.id} -> {entrust_no}")
                else:
                    order.status = OrderStatus.REJECTED
                    order.message = str(result.get('msg') or result.get('message') or result)
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
    
    def cancel_order(self, order_id: str) -> bool:
        if not self.is_connected:
            return False
        
        order = self._orders.get(order_id)
        if not order or not order.broker_id:
            return False
        
        try:
            self._client.cancel_entrust(order.broker_id)
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
            return [o for o in self._orders.values() if o.is_active]
        return list(self._orders.values())


# ============================================================
# Broker Factory
# ============================================================

def create_broker(mode: str = None, **kwargs) -> BrokerInterface:
    """
    Factory function to create appropriate broker.
    
    Args:
        mode: 'simulation', 'paper', 'live', 'ths', 'ht', 'gj', 'yh', 'zszq'
        **kwargs: Additional arguments for broker
    """
    from config import TradingMode
    
    if mode is None:
        mode = CONFIG.trading_mode.value if hasattr(CONFIG.trading_mode, 'value') else str(CONFIG.trading_mode)
    
    mode = mode.lower()
    
    if mode in ['simulation', 'paper']:
        return SimulatorBroker(kwargs.get('capital', CONFIG.capital))
    elif mode == 'live':
        broker_type = kwargs.get('broker_type', 'ths')
        if broker_type in ['zszq', 'zhaoshang', '招商']:
            return ZSZQBroker()
        return THSBroker(broker_type=broker_type)
    elif mode in ['ths', 'ht', 'gj', 'yh']:
        return THSBroker(broker_type=mode)
    elif mode in ['zszq', 'zhaoshang', '招商']:
        return ZSZQBroker()
    else:
        log.warning(f"Unknown broker mode: {mode}, using simulator")
        return SimulatorBroker()