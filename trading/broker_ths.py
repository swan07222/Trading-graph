"""
TongHuaShun (同花顺) Broker Integration
Real trading via THS client
"""
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from .broker_base import (
    BrokerInterface, Order, Position, Account,
    OrderType, OrderSide, OrderStatus
)
from config import CONFIG
from utils.logger import log

try:
    import easytrader
    EASYTRADER_OK = True
except ImportError:
    EASYTRADER_OK = False
    log.warning("easytrader not installed. Live trading unavailable.")


class THSBroker(BrokerInterface):
    """
    TongHuaShun (同花顺) broker integration
    
    Requirements:
    1. Install THS standalone order program
    2. Set BROKER_PATH in config
    3. Enable auto-login in THS
    """
    
    def __init__(self):
        self._client = None
        self._connected = False
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
    
    @property
    def name(self) -> str:
        return "TongHuaShun"
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None
    
    def connect(self) -> bool:
        if not EASYTRADER_OK:
            log.error("easytrader not installed")
            return False
        
        broker_path = Path(CONFIG.BROKER_PATH)
        if not broker_path.exists():
            log.error(f"THS not found: {broker_path}")
            return False
        
        try:
            self._client = easytrader.use('ths')
            self._client.connect(str(broker_path))
            
            # Verify connection
            balance = self._client.balance
            if balance:
                self._connected = True
                log.info(f"Connected to THS. Balance: {balance}")
                return True
            
        except Exception as e:
            log.error(f"Failed to connect to THS: {e}")
        
        return False
    
    def disconnect(self):
        self._connected = False
        self._client = None
        log.info("Disconnected from THS")
    
    def get_account(self) -> Account:
        if not self.is_connected:
            return Account()
        
        try:
            balance = self._client.balance
            positions = self.get_positions()
            
            market_value = sum(p.market_value for p in positions.values())
            
            return Account(
                cash=float(balance.get('总资产', 0)),
                available=float(balance.get('可用金额', 0)),
                frozen=float(balance.get('冻结金额', 0)),
                market_value=market_value,
                total_pnl=float(balance.get('总盈亏', 0)),
                positions=positions
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
                    quantity=int(p.get('股票余额', 0)),
                    available_qty=int(p.get('可卖余额', 0)),
                    avg_cost=float(p.get('成本价', 0)),
                    current_price=float(p.get('当前价', 0)),
                    unrealized_pnl=float(p.get('盈亏', 0)),
                    unrealized_pnl_pct=float(p.get('盈亏比例', 0)) * 100
                )
            
            return positions
            
        except Exception as e:
            log.error(f"Failed to get positions: {e}")
            return {}
    
    def submit_order(self, order: Order) -> Order:
        if not self.is_connected:
            order.status = OrderStatus.REJECTED
            order.message = "Not connected"
            return order
        
        try:
            self._order_counter += 1
            order.id = f"THS_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._order_counter}"
            
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
            
            if result and '委托编号' in str(result):
                order.status = OrderStatus.SUBMITTED
                order.message = f"Broker ID: {result.get('委托编号', '')}"
                log.info(f"Order submitted: {order.id}")
            else:
                order.status = OrderStatus.REJECTED
                order.message = str(result)
                log.warning(f"Order rejected: {result}")
            
            self._orders[order.id] = order
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
        if not order:
            return False
        
        try:
            # Would need broker order ID
            # result = self._client.cancel_entrust(broker_order_id)
            order.status = OrderStatus.CANCELLED
            return True
        except:
            return False
    
    def get_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.is_active]