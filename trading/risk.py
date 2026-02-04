"""
Enhanced Risk Management System
"""
from typing import Tuple, Dict, Optional, List
from datetime import date, datetime, timedelta
from dataclasses import dataclass
import numpy as np

from config import CONFIG
from .broker import Account, OrderSide, Position
from utils.logger import log
from utils.security import get_audit_log


@dataclass
class RiskMetrics:
    """Real-time risk metrics"""
    equity: float
    available: float
    daily_pnl: float
    daily_pnl_pct: float
    positions_count: int
    total_exposure: float
    exposure_pct: float
    largest_position_pct: float
    var_95: float  # Value at Risk
    max_drawdown: float
    can_trade: bool
    warnings: List[str]


class RiskManager:
    """
    Enhanced Risk Management System
    
    Features:
    1. Position size limits (per stock and total)
    2. Daily loss limits with circuit breakers
    3. Value at Risk (VaR) calculation
    4. Maximum drawdown tracking
    5. Concentration risk monitoring
    6. Order validation
    7. Audit logging
    """
    
    def __init__(self, account: Account):
        self.account = account
        self.initial_equity = account.equity
        self.daily_start_equity = account.equity
        self.peak_equity = account.equity
        self.trades_today = 0
        self._last_date = date.today()
        
        # Historical returns for VaR
        self._returns_history: List[float] = []
        self._max_returns_history = 252  # 1 year
        
        # Audit
        self._audit = get_audit_log()
    
    def update(self, account: Account):
        """Update with current account state"""
        old_equity = self.account.equity
        self.account = account
        
        # Track peak for drawdown
        if account.equity > self.peak_equity:
            self.peak_equity = account.equity
        
        # Check for new day
        today = date.today()
        if today != self._last_date:
            self._new_day(old_equity)
            self._last_date = today
        
        # Track returns
        if old_equity > 0:
            ret = (account.equity - old_equity) / old_equity
            self._returns_history.append(ret)
            if len(self._returns_history) > self._max_returns_history:
                self._returns_history.pop(0)
    
    def _new_day(self, last_equity: float):
        """Reset for new trading day"""
        log.info("Risk manager: New trading day")
        self.daily_start_equity = last_equity
        self.trades_today = 0
    
    def record_trade(self):
        """Record trade execution"""
        self.trades_today += 1
    
    def get_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        warnings = []
        
        # Basic metrics
        equity = self.account.equity
        available = self.account.available
        
        # Daily P&L
        if self.daily_start_equity > 0:
            daily_pnl = equity - self.daily_start_equity
            daily_pnl_pct = daily_pnl / self.daily_start_equity * 100
        else:
            daily_pnl = 0
            daily_pnl_pct = 0
        
        # Exposure
        total_exposure = sum(p.market_value for p in self.account.positions.values())
        exposure_pct = total_exposure / equity * 100 if equity > 0 else 0
        
        # Largest position
        largest_position_pct = 0
        if self.account.positions:
            largest = max(p.market_value for p in self.account.positions.values())
            largest_position_pct = largest / equity * 100 if equity > 0 else 0
        
        # VaR (95% confidence, 1-day)
        var_95 = self._calculate_var(0.95)
        
        # Drawdown
        if self.peak_equity > 0:
            max_drawdown = (self.peak_equity - equity) / self.peak_equity * 100
        else:
            max_drawdown = 0
        
        # Check limits and generate warnings
        can_trade = True
        
        if daily_pnl_pct <= -CONFIG.MAX_DAILY_LOSS_PCT:
            warnings.append(f"DAILY LOSS LIMIT REACHED: {daily_pnl_pct:.1f}%")
            can_trade = False
            self._audit.log_risk_event('daily_loss_limit', {'pnl_pct': daily_pnl_pct})
        
        if exposure_pct > 90:
            warnings.append(f"HIGH EXPOSURE: {exposure_pct:.1f}%")
        
        if largest_position_pct > CONFIG.MAX_POSITION_PCT:
            warnings.append(f"POSITION CONCENTRATION: {largest_position_pct:.1f}%")
        
        if max_drawdown > 10:
            warnings.append(f"SIGNIFICANT DRAWDOWN: {max_drawdown:.1f}%")
        
        if len(self.account.positions) >= CONFIG.MAX_POSITIONS:
            warnings.append(f"MAX POSITIONS REACHED: {len(self.account.positions)}")
        
        return RiskMetrics(
            equity=equity,
            available=available,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            positions_count=len(self.account.positions),
            total_exposure=total_exposure,
            exposure_pct=exposure_pct,
            largest_position_pct=largest_position_pct,
            var_95=var_95,
            max_drawdown=max_drawdown,
            can_trade=can_trade,
            warnings=warnings
        )
    
    def _calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(self._returns_history) < 20:
            # Not enough data, use heuristic
            return self.account.equity * 0.02  # 2% default
        
        returns = np.array(self._returns_history)
        var_pct = np.percentile(returns, (1 - confidence) * 100)
        return abs(var_pct * self.account.equity)
    
    def check_order(
        self,
        code: str,
        side: OrderSide,
        quantity: int,
        price: float
    ) -> Tuple[bool, str]:
        """Comprehensive order validation"""
        if price <= 0:
            return False, "Invalid price"
        
        # Check daily limits
        metrics = self.get_metrics()
        if not metrics.can_trade:
            return False, "Trading suspended: daily loss limit reached"
        
        if side == OrderSide.BUY:
            return self._check_buy_order(code, quantity, price, metrics)
        else:
            return self._check_sell_order(code, quantity)
    
    def _check_buy_order(
        self, 
        code: str, 
        quantity: int, 
        price: float,
        metrics: RiskMetrics
    ) -> Tuple[bool, str]:
        """Validate buy order"""
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if quantity % CONFIG.LOT_SIZE != 0:
            return False, f"Quantity must be multiple of {CONFIG.LOT_SIZE}"
        
        # Cost calculation
        cost = quantity * price
        commission = cost * CONFIG.COMMISSION
        total_cost = cost + commission
        
        if total_cost > self.account.available:
            return False, f"Insufficient funds: need ¥{total_cost:,.2f}"
        
        # Position limit including existing
        existing_value = 0
        existing_pos = self.account.positions.get(code)
        if existing_pos:
            existing_value = existing_pos.quantity * price
        
        new_total_value = existing_value + (quantity * price)
        
        if self.account.equity > 0:
            position_pct = new_total_value / self.account.equity * 100
            if position_pct > CONFIG.MAX_POSITION_PCT:
                return False, f"Position too large: {position_pct:.1f}%"
        
        # Max positions
        if code not in self.account.positions:
            if len(self.account.positions) >= CONFIG.MAX_POSITIONS:
                return False, f"Max positions reached: {CONFIG.MAX_POSITIONS}"
        
        # VaR check
        if metrics.var_95 > self.account.equity * 0.05:
            # If 1-day VaR > 5% of equity, require smaller position
            max_additional = self.account.equity * 0.02
            if cost > max_additional:
                return False, f"High risk: reduce position to ¥{max_additional:,.0f}"
        
        return True, "OK"
    
    def _check_sell_order(self, code: str, quantity: int) -> Tuple[bool, str]:
        """Validate sell order"""
        pos = self.account.positions.get(code)
        
        if not pos:
            return False, f"No position in {code}"
        
        if quantity > pos.available_qty:
            return False, f"Available: {pos.available_qty}, requested: {quantity}"
        
        return True, "OK"
    
    def calculate_position_size(
        self,
        entry: float,
        stop_loss: float,
        confidence: float = 1.0,
        signal_strength: float = 1.0
    ) -> int:
        """Calculate optimal position size using Kelly-inspired formula"""
        risk_per_share = abs(entry - stop_loss)
        
        if risk_per_share <= 0 or entry <= 0:
            return 0
        
        # Base risk from config
        base_risk_pct = CONFIG.RISK_PER_TRADE / 100
        
        # Adjust for confidence and signal strength
        adjusted_risk_pct = base_risk_pct * confidence * signal_strength
        
        # Kelly-inspired adjustment (fractional Kelly)
        kelly_fraction = 0.25  # Use 25% Kelly
        win_rate = 0.55  # Assume 55% win rate
        payoff = 2.0  # Assume 2:1 reward/risk
        kelly = (win_rate * payoff - (1 - win_rate)) / payoff
        kelly_adjusted = kelly * kelly_fraction * adjusted_risk_pct
        
        # Calculate shares
        risk_amount = self.account.equity * kelly_adjusted
        shares = int(risk_amount / risk_per_share)
        shares = (shares // CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        
        # Apply limits
        max_value = self.account.equity * (CONFIG.MAX_POSITION_PCT / 100)
        max_shares = int(max_value / entry / CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        shares = min(shares, max_shares)
        
        # Affordability check
        max_affordable = int(self.account.available / entry / CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        shares = min(shares, max_affordable)
        
        return max(0, shares)