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
from datetime import datetime, date, timedelta
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
from utils.logger import get_logger

log = get_logger(__name__)


# ============================================================
# Abstract Broker Interface
# ============================================================

class BrokerInterface(ABC):
    """
    Abstract broker interface - all brokers must implement this.
    Thread-safe design with callbacks for order updates.
    
    CRITICAL: For live trading correctness, brokers MUST implement:
    - get_fills(): Return new fills since last call
    - get_order_status(): Get current status of an order
    - sync_order(): Sync single order with broker state
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._callbacks: Dict[str, List[Callable]] = {
            'order_update': [],
            'trade': [],
            'error': [],
        }
        # Mapping from our order.id to broker's entrust number
        self._order_id_to_broker_id: Dict[str, str] = {}
        self._broker_id_to_order_id: Dict[str, str] = {}
    
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
        """Submit order - MUST NOT modify order.id, use broker_id instead"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order by our order_id"""
        pass
    
    @abstractmethod
    def get_orders(self, active_only: bool = True) -> List[Order]:
        """Get orders"""
        pass
    
    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[float]:
        """Get current price for a stock"""
        pass
    
    # ============================================================
    # CRITICAL: Fill and Status Sync Methods for Live Trading
    # ============================================================
    
    @abstractmethod
    def get_fills(self, since: datetime = None) -> List[Fill]:
        """
        Get fills/trades since last call or since timestamp.
        Each Fill MUST have:
        - order_id: Our internal order ID (not broker's)
        - All fill details (quantity, price, commission, etc.)
        
        Implementation should track which fills have been returned
        to avoid duplicates.
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get current status of an order from broker.
        Uses our internal order_id, maps to broker_id internally.
        """
        pass
    
    @abstractmethod
    def sync_order(self, order: Order) -> Order:
        """
        Sync order state with broker.
        Updates order.status, order.filled_qty, order.avg_price, etc.
        Returns updated order.
        """
        pass
    
    def get_broker_id(self, order_id: str) -> Optional[str]:
        """Get broker's entrust number for our order_id"""
        return self._order_id_to_broker_id.get(order_id)
    
    def get_order_id(self, broker_id: str) -> Optional[str]:
        """Get our order_id for broker's entrust number"""
        return self._broker_id_to_order_id.get(broker_id)
    
    def register_order_mapping(self, order_id: str, broker_id: str):
        """Register mapping between our order_id and broker's entrust number"""
        with self._lock:
            self._order_id_to_broker_id[order_id] = broker_id
            self._broker_id_to_order_id[broker_id] = order_id
    
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
    - Full fill sync support
    """
    
    def __init__(self, initial_capital: float = None):
        super().__init__()
        self._initial_capital = initial_capital or CONFIG.capital
        self._cash = self._initial_capital
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []
        self._fills: List[Fill] = []
        self._unsent_fills: List[Fill] = []  # Fills not yet returned by get_fills()
        self._connected = False
        
        # T+1 tracking
        self._purchase_dates: Dict[str, date] = {}
        self._last_settlement_date = date.today()
        
        # Data fetcher (lazy init)
        self._fetcher = None
        
        # Fill ID counter
        self._fill_counter = 0
    
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
            log.info(f"Simulator connected with {self._initial_capital:,.2f}")
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
        """Submit order - preserves order.id, sets broker_id"""
        import random
        
        with self._lock:
            self._check_settlement()
            
            # CRITICAL: Do NOT modify order.id - set broker_id instead
            order.broker_id = f"SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
            order.created_at = order.created_at or datetime.now()
            order.submitted_at = datetime.now()
            
            # Register mapping
            self.register_order_mapping(order.id, order.broker_id)
            
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
            if not order.name and quote:
                order.name = quote.name
            
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
            
            # Execute immediately (simulator fills instantly)
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
                return False, f"Insufficient funds: need {total:,.2f}, have {self._cash:,.2f}"
            
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
        
        # Create Fill record with our order.id (not broker_id)
        self._fill_counter += 1
        fill = Fill(
            id=f"FILL_SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._fill_counter:06d}",
            order_id=order.id,  # CRITICAL: Use internal order.id
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=fill_price,
            commission=commission,
            stamp_tax=stamp_tax,
            timestamp=datetime.now()
        )
        
        self._fills.append(fill)
        self._unsent_fills.append(fill)  # Track for get_fills()
        
        self._order_history.append(order)
        
        log.info(
            f"[SIM] {order.side.value.upper()} {fill_qty} {order.symbol} "
            f"@ {fill_price:.2f} (cost: {total_cost:.2f})"
        )
        
        self._emit('trade', order, fill)
    
    def get_fills(self, since: datetime = None) -> List[Fill]:
        """Get fills not yet returned"""
        with self._lock:
            if since:
                fills = [f for f in self._fills if f.timestamp and f.timestamp >= since]
            else:
                # Return unsent fills
                fills = list(self._unsent_fills)
                self._unsent_fills.clear()
            return fills
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status"""
        with self._lock:
            order = self._orders.get(order_id)
            return order.status if order else None
    
    def sync_order(self, order: Order) -> Order:
        """Sync order with simulator state"""
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
                return [o for o in self._orders.values() if o.is_active]
            return list(self._orders.values())
    
    def _check_settlement(self):
        """Handle T+1 settlement on new trading day"""
        from core.constants import is_trading_day
        
        today = date.today()
        if today != self._last_settlement_date and is_trading_day(today):
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
                    'timestamp': f.timestamp
                }
                for f in self._fills
            ]
    
    def reset(self):
        """Reset simulator to initial state"""
        with self._lock:
            self._cash = self._initial_capital
            self._positions.clear()
            self._orders.clear()
            self._order_history.clear()
            self._fills.clear()
            self._unsent_fills.clear()
            self._purchase_dates.clear()
            self._last_settlement_date = date.today()
            self._order_id_to_broker_id.clear()
            self._broker_id_to_order_id.clear()
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


# ============================================================
# THS Broker (Real Trading)
# ============================================================

class THSBroker(BrokerInterface):
    """
    TongHuaShun (同花顺) broker integration via easytrader.
    
    CRITICAL: Implements full fill sync for live trading correctness.
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
        self._seen_fill_ids: set = set()  # Track seen fills to avoid duplicates
        self._last_fill_check: datetime = None
        
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
                self._last_fill_check = datetime.now()
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
            # CRITICAL: Preserve order.id, set broker_id
            order.created_at = order.created_at or datetime.now()
            order.submitted_at = datetime.now()
            
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
                entrust_no = result.get('委托编号') or result.get('entrust_no')
                if entrust_no:
                    order.status = OrderStatus.SUBMITTED
                    order.broker_id = str(entrust_no)
                    order.message = f"Entrust: {order.broker_id}"
                    
                    # Register mapping
                    self.register_order_mapping(order.id, order.broker_id)
                    
                    log.info(f"Order submitted: {order.id} -> broker {order.broker_id}")
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
    
    def get_fills(self, since: datetime = None) -> List[Fill]:
        """
        Get fills from broker.
        CRITICAL: Maps broker entrust numbers back to our order IDs.
        """
        if not self.is_connected:
            return []
        
        fills = []
        
        try:
            # Get today's trades from broker
            trades = self._client.today_trades
            
            for trade in trades:
                # Create unique fill ID
                fill_id = f"{trade.get('成交编号', '')}"
                if not fill_id or fill_id in self._seen_fill_ids:
                    continue
                
                self._seen_fill_ids.add(fill_id)
                
                # Map broker entrust number to our order ID
                broker_entrust = str(trade.get('委托编号', ''))
                our_order_id = self.get_order_id(broker_entrust)
                
                if not our_order_id:
                    log.warning(f"Unknown entrust number: {broker_entrust}")
                    continue
                
                # Parse side
                trade_side = trade.get('买卖标志', trade.get('操作', ''))
                if '买' in str(trade_side):
                    side = OrderSide.BUY
                else:
                    side = OrderSide.SELL
                
                fill = Fill(
                    id=fill_id,
                    order_id=our_order_id,  # Our order ID, not broker's
                    symbol=str(trade.get('证券代码', '')).zfill(6),
                    side=side,
                    quantity=int(trade.get('成交数量', 0)),
                    price=float(trade.get('成交价格', 0)),
                    commission=float(trade.get('手续费', 0) or 0),
                    stamp_tax=float(trade.get('印花税', 0) or 0),
                    timestamp=datetime.now()
                )
                
                fills.append(fill)
                log.info(f"Fill received: {fill.id} for order {our_order_id}")
                
        except Exception as e:
            log.error(f"Failed to get fills: {e}")
        
        return fills
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status from broker"""
        if not self.is_connected:
            return None
        
        broker_id = self.get_broker_id(order_id)
        if not broker_id:
            return None
        
        try:
            entrusts = self._client.today_entrusts
            
            for entrust in entrusts:
                if str(entrust.get('委托编号', '')) == broker_id:
                    status_str = entrust.get('委托状态', entrust.get('状态', ''))
                    return self._parse_status(status_str)
            
            return None
            
        except Exception as e:
            log.error(f"Failed to get order status: {e}")
            return None
    
    def _parse_status(self, status_str: str) -> OrderStatus:
        """Parse broker status string to OrderStatus"""
        status_str = str(status_str).lower()
        
        if '全部成交' in status_str or '已成' in status_str:
            return OrderStatus.FILLED
        elif '部分成交' in status_str:
            return OrderStatus.PARTIAL
        elif '已报' in status_str or '已委托' in status_str:
            return OrderStatus.ACCEPTED
        elif '已撤' in status_str or '撤单' in status_str:
            return OrderStatus.CANCELLED
        elif '废单' in status_str or '拒绝' in status_str:
            return OrderStatus.REJECTED
        else:
            return OrderStatus.SUBMITTED
    
    def sync_order(self, order: Order) -> Order:
        """Sync order state with broker"""
        if not self.is_connected:
            return order
        
        broker_id = self.get_broker_id(order.id)
        if not broker_id:
            return order
        
        try:
            entrusts = self._client.today_entrusts
            
            for entrust in entrusts:
                if str(entrust.get('委托编号', '')) == broker_id:
                    order.status = self._parse_status(entrust.get('委托状态', ''))
                    order.filled_qty = int(entrust.get('成交数量', 0) or 0)
                    
                    avg_price = entrust.get('成交均价', entrust.get('成交价格', 0))
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
    
    CRITICAL: Implements full fill sync for live trading correctness.
    """
    
    def __init__(self):
        super().__init__()
        self._client = None
        self._connected = False
        self._orders: Dict[str, Order] = {}
        self._seen_fill_ids: set = set()
        
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
            # CRITICAL: Preserve order.id
            order.created_at = order.created_at or datetime.now()
            order.submitted_at = datetime.now()
            
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
                    order.message = f"Entrust: {entrust_no}"
                    
                    # Register mapping
                    self.register_order_mapping(order.id, order.broker_id)
                    
                    log.info(f"Order submitted: {order.id} -> broker {order.broker_id}")
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
    
    def get_fills(self, since: datetime = None) -> List[Fill]:
        """Get fills from broker"""
        if not self.is_connected:
            return []
        
        fills = []
        
        try:
            trades = self._client.today_trades
            
            for trade in trades:
                fill_id = f"{trade.get('成交编号', '')}"
                if not fill_id or fill_id in self._seen_fill_ids:
                    continue
                
                self._seen_fill_ids.add(fill_id)
                
                broker_entrust = str(trade.get('委托编号', ''))
                our_order_id = self.get_order_id(broker_entrust)
                
                if not our_order_id:
                    log.warning(f"Unknown entrust number: {broker_entrust}")
                    continue
                
                trade_side = trade.get('买卖标志', trade.get('操作', ''))
                if '买' in str(trade_side):
                    side = OrderSide.BUY
                else:
                    side = OrderSide.SELL
                
                fill = Fill(
                    id=fill_id,
                    order_id=our_order_id,
                    symbol=str(trade.get('证券代码', '')).zfill(6),
                    side=side,
                    quantity=int(trade.get('成交数量', 0)),
                    price=float(trade.get('成交价格', 0)),
                    commission=float(trade.get('手续费', 0) or 0),
                    stamp_tax=float(trade.get('印花税', 0) or 0),
                    timestamp=datetime.now()
                )
                
                fills.append(fill)
                
        except Exception as e:
            log.error(f"Failed to get fills: {e}")
        
        return fills
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status from broker"""
        if not self.is_connected:
            return None
        
        broker_id = self.get_broker_id(order_id)
        if not broker_id:
            return None
        
        try:
            entrusts = self._client.today_entrusts
            
            for entrust in entrusts:
                if str(entrust.get('委托编号', '')) == broker_id:
                    status_str = entrust.get('委托状态', entrust.get('状态', ''))
                    return self._parse_status(status_str)
            
            return None
            
        except Exception as e:
            log.error(f"Failed to get order status: {e}")
            return None
    
    def _parse_status(self, status_str: str) -> OrderStatus:
        """Parse broker status string to OrderStatus"""
        status_str = str(status_str).lower()
        
        if '全部成交' in status_str or '已成' in status_str:
            return OrderStatus.FILLED
        elif '部分成交' in status_str:
            return OrderStatus.PARTIAL
        elif '已报' in status_str or '已委托' in status_str:
            return OrderStatus.ACCEPTED
        elif '已撤' in status_str or '撤单' in status_str:
            return OrderStatus.CANCELLED
        elif '废单' in status_str or '拒绝' in status_str:
            return OrderStatus.REJECTED
        else:
            return OrderStatus.SUBMITTED
    
    def sync_order(self, order: Order) -> Order:
        """Sync order state with broker"""
        if not self.is_connected:
            return order
        
        broker_id = self.get_broker_id(order.id)
        if not broker_id:
            return order
        
        try:
            entrusts = self._client.today_entrusts
            
            for entrust in entrusts:
                if str(entrust.get('委托编号', '')) == broker_id:
                    order.status = self._parse_status(entrust.get('委托状态', ''))
                    order.filled_qty = int(entrust.get('成交数量', 0) or 0)
                    
                    avg_price = entrust.get('成交均价', entrust.get('成交价格', 0))
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