"""
Simulator Broker - Paper trading with realistic simulation
"""
import random
import threading
import time
from datetime import datetime, date
from typing import Dict, List, Optional

from .broker_base import (
    BrokerInterface, Order, Position, Account,
    OrderType, OrderSide, OrderStatus
)
from config import CONFIG
from data.fetcher import DataFetcher
from utils.logger import log


class SimulatorBroker(BrokerInterface):
    """
    Paper trading simulator with realistic behavior:
    - Slippage simulation
    - Commission calculation
    - T+1 rule enforcement
    - Partial fills
    """
    
    def __init__(self, initial_capital: float = None):
        self._initial_capital = initial_capital or CONFIG.CAPITAL
        self._cash = self._initial_capital
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
        self._connected = False
        
        self._fetcher = DataFetcher()
        
        # Track buy dates for T+1
        self._buy_dates: Dict[str, date] = {}
        
        # Trade history
        self._trades: List[Dict] = []
    
    @property
    def name(self) -> str:
        return "Simulator"
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def connect(self) -> bool:
        self._connected = True
        log.info(f"Simulator connected with ¥{self._initial_capital:,.2f}")
        return True
    
    def disconnect(self):
        self._connected = False
        log.info("Simulator disconnected")
    
    def get_account(self) -> Account:
        self._update_prices()
        
        market_value = sum(p.market_value for p in self._positions.values())
        total_pnl = sum(p.unrealized_pnl + p.realized_pnl for p in self._positions.values())
        
        return Account(
            cash=self._cash,
            available=self._cash,
            frozen=0,
            market_value=market_value,
            total_pnl=total_pnl,
            positions=self._positions.copy()
        )
    
    def get_positions(self) -> Dict[str, Position]:
        self._update_prices()
        return self._positions.copy()
    
    def submit_order(self, order: Order) -> Order:
        self._order_counter += 1
        order.id = f"SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._order_counter}"
        order.status = OrderStatus.SUBMITTED
        
        # Get current price
        quote = self._fetcher.get_realtime(order.stock_code)
        if not quote:
            order.status = OrderStatus.REJECTED
            order.message = "Cannot get market quote"
            return order
        
        current_price = quote.price
        
        # Validate order
        if not self._validate_order(order, current_price):
            return order
        
        # Execute market orders immediately
        if order.order_type == OrderType.MARKET:
            self._execute_order(order, current_price)
        else:
            # Limit order - check if it can be filled
            if order.side == OrderSide.BUY and order.price >= current_price:
                self._execute_order(order, order.price)
            elif order.side == OrderSide.SELL and order.price <= current_price:
                self._execute_order(order, order.price)
            else:
                # Store pending order
                self._orders[order.id] = order
        
        return order
    
    def _validate_order(self, order: Order, price: float) -> bool:
        """Validate order before execution"""
        if order.side == OrderSide.BUY:
            # Check lot size
            if order.quantity % CONFIG.LOT_SIZE != 0:
                order.status = OrderStatus.REJECTED
                order.message = f"Quantity must be multiple of {CONFIG.LOT_SIZE}"
                return False
            
            # Check cash
            cost = order.quantity * (order.price or price)
            commission = cost * CONFIG.COMMISSION
            total_cost = cost + commission
            
            if total_cost > self._cash:
                order.status = OrderStatus.REJECTED
                order.message = "Insufficient funds"
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
            
            # T+1 check
            if CONFIG.T_PLUS_1:
                buy_date = self._buy_dates.get(order.stock_code)
                if buy_date == date.today():
                    order.status = OrderStatus.REJECTED
                    order.message = "T+1: Cannot sell shares bought today"
                    return False
        
        return True
    
    def _execute_order(self, order: Order, market_price: float):
        """Execute order with realistic simulation"""
        # Calculate fill price with slippage
        slippage = CONFIG.SLIPPAGE
        
        if order.side == OrderSide.BUY:
            fill_price = market_price * (1 + slippage + random.uniform(0, slippage))
        else:
            fill_price = market_price * (1 - slippage - random.uniform(0, slippage))
        
        fill_price = round(fill_price, 2)
        
        # Calculate costs
        trade_value = order.quantity * fill_price
        commission = trade_value * CONFIG.COMMISSION
        
        stamp_tax = 0
        if order.side == OrderSide.SELL:
            stamp_tax = trade_value * CONFIG.STAMP_TAX
        
        total_cost = commission + stamp_tax
        
        # Update position and cash
        if order.side == OrderSide.BUY:
            self._cash -= (trade_value + total_cost)
            
            if order.stock_code in self._positions:
                pos = self._positions[order.stock_code]
                total_qty = pos.quantity + order.quantity
                pos.avg_cost = (pos.avg_cost * pos.quantity + fill_price * order.quantity) / total_qty
                pos.quantity = total_qty
                # New shares not available until tomorrow
            else:
                quote = self._fetcher.get_realtime(order.stock_code)
                self._positions[order.stock_code] = Position(
                    stock_code=order.stock_code,
                    stock_name=quote.name if quote else order.stock_code,
                    quantity=order.quantity,
                    available_qty=0,  # T+1
                    avg_cost=fill_price,
                    current_price=fill_price
                )
            
            self._buy_dates[order.stock_code] = date.today()
        
        else:  # SELL
            self._cash += (trade_value - total_cost)
            
            pos = self._positions[order.stock_code]
            realized = (fill_price - pos.avg_cost) * order.quantity
            pos.realized_pnl += realized
            pos.quantity -= order.quantity
            pos.available_qty -= order.quantity
            
            if pos.quantity == 0:
                del self._positions[order.stock_code]
                if order.stock_code in self._buy_dates:
                    del self._buy_dates[order.stock_code]
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_qty = order.quantity
        order.filled_price = fill_price
        order.updated_at = datetime.now()
        
        # Record trade
        self._trades.append({
            'order_id': order.id,
            'code': order.stock_code,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': fill_price,
            'commission': commission,
            'stamp_tax': stamp_tax,
            'timestamp': datetime.now()
        })
        
        log.info(
            f"Order filled: {order.side.value.upper()} {order.quantity} "
            f"{order.stock_code} @ ¥{fill_price:.2f} (cost: ¥{total_cost:.2f})"
        )
        
        if self.on_trade:
            self.on_trade(order)
    
    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            del self._orders[order_id]
            return True
        return False
    
    def get_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.is_active]
    
    def _update_prices(self):
        """Update position prices"""
        for code, pos in self._positions.items():
            quote = self._fetcher.get_realtime(code)
            if quote:
                pos.update_price(quote.price)
                
                # Update available quantity for next day
                if self._buy_dates.get(code) != date.today():
                    pos.available_qty = pos.quantity
    
    def reset(self):
        """Reset simulator to initial state"""
        self._cash = self._initial_capital
        self._positions.clear()
        self._orders.clear()
        self._buy_dates.clear()
        self._trades.clear()
        self._order_counter = 0
        log.info("Simulator reset")
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        return self._trades.copy()
    
    def get_daily_summary(self) -> Dict:
        """Get daily trading summary"""
        today_trades = [t for t in self._trades if t['timestamp'].date() == date.today()]
        
        total_buy = sum(t['quantity'] * t['price'] for t in today_trades if t['side'] == 'buy')
        total_sell = sum(t['quantity'] * t['price'] for t in today_trades if t['side'] == 'sell')
        total_commission = sum(t['commission'] + t.get('stamp_tax', 0) for t in today_trades)
        
        return {
            'date': date.today(),
            'trades': len(today_trades),
            'buy_value': total_buy,
            'sell_value': total_sell,
            'commission': total_commission,
            'net_value': total_sell - total_buy - total_commission
        } 