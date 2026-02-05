"""
Portfolio Management - Track positions and performance
Uses unified broker types
"""
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np

from config import CONFIG
from core.types import Position, Account
from utils.logger import log


@dataclass
class Trade:
    """Individual trade record"""
    id: str
    stock_code: str
    stock_name: str
    side: str  # "buy" or "sell"
    quantity: int
    price: float
    value: float
    commission: float
    stamp_tax: float
    timestamp: datetime
    pnl: float = 0.0  # Realized P&L for sell trades


@dataclass
class DailyPerformance:
    """Daily performance record"""
    date: date
    starting_equity: float
    ending_equity: float
    pnl: float
    pnl_pct: float
    trades: int
    wins: int
    losses: int


@dataclass
class PortfolioStats:
    """Portfolio statistics"""
    total_value: float
    cash: float
    positions_value: float
    total_pnl: float
    total_pnl_pct: float
    realized_pnl: float
    unrealized_pnl: float
    
    # Performance metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Current
    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl: float
    weekly_pnl_pct: float
    monthly_pnl: float
    monthly_pnl_pct: float


class Portfolio:
    """
    Portfolio management system
    
    Features:
    - Track all positions and trades
    - Calculate performance metrics
    - Risk analysis
    - Persistence
    """
    
    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or CONFIG.CAPITAL
        self.cash = self.initial_capital
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_performance: List[DailyPerformance] = []
        
        # Track equity curve
        self.equity_history: List[Tuple[datetime, float]] = [
            (datetime.now(), self.initial_capital)
        ]
        
        # Peak for drawdown calculation
        self._peak_equity = self.initial_capital
        self._max_drawdown = 0.0
        
        # Daily tracking
        self._daily_start_equity = self.initial_capital
        self._daily_start_date = date.today()
        
        # Load saved state
        self._load()
    
    @property
    def positions_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())
    
    @property
    def equity(self) -> float:
        return self.cash + self.positions_value
    
    @property
    def total_pnl(self) -> float:
        return self.equity - self.initial_capital
    
    @property
    def total_pnl_pct(self) -> float:
        if self.initial_capital <= 0:
            return 0
        return (self.equity / self.initial_capital - 1) * 100
    
    def update_from_account(self, account: Account):
        """Update portfolio from broker account"""
        self.cash = account.cash
        self.positions = account.positions.copy()
        
        self._update_equity()
    
    def record_trade(self, trade: Trade):
        """Record a completed trade"""
        self.trades.append(trade)
        
        # Note: Don't update cash here if using update_from_account
        # The broker account is the source of truth
        
        self._update_equity()
        self._save()
    
    def _update_equity(self):
        """Update equity tracking"""
        current_equity = self.equity
        
        # Update equity history
        self.equity_history.append((datetime.now(), current_equity))
        
        # Update peak and drawdown
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown
        
        # Check for new day
        today = date.today()
        if today != self._daily_start_date:
            # Record previous day's performance
            daily_pnl = current_equity - self._daily_start_equity
            daily_pnl_pct = daily_pnl / self._daily_start_equity * 100 if self._daily_start_equity > 0 else 0
            
            # Count today's trades
            today_trades = [t for t in self.trades if t.timestamp.date() == self._daily_start_date]
            wins = len([t for t in today_trades if t.side == "sell" and t.pnl > 0])
            losses = len([t for t in today_trades if t.side == "sell" and t.pnl < 0])
            
            self.daily_performance.append(DailyPerformance(
                date=self._daily_start_date,
                starting_equity=self._daily_start_equity,
                ending_equity=current_equity,
                pnl=daily_pnl,
                pnl_pct=daily_pnl_pct,
                trades=len(today_trades),
                wins=wins,
                losses=losses
            ))
            
            self._daily_start_equity = current_equity
            self._daily_start_date = today
    
    def get_stats(self) -> PortfolioStats:
        """Calculate comprehensive portfolio statistics"""
        # Basic values
        total_value = self.equity
        
        # Realized and unrealized P&L
        realized_pnl = sum(t.pnl for t in self.trades if t.side == "sell")
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        
        # Trade statistics
        sell_trades = [t for t in self.trades if t.side == "sell"]
        total_trades = len(sell_trades)
        
        if total_trades > 0:
            winning_trades = len([t for t in sell_trades if t.pnl > 0])
            losing_trades = len([t for t in sell_trades if t.pnl < 0])
            win_rate = winning_trades / total_trades
            
            wins = [t.pnl for t in sell_trades if t.pnl > 0]
            losses = [t.pnl for t in sell_trades if t.pnl < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0
            largest_win = max(wins) if wins else 0
            largest_loss = abs(min(losses)) if losses else 0
            
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        else:
            winning_trades = losing_trades = 0
            win_rate = profit_factor = avg_win = avg_loss = 0
            largest_win = largest_loss = 0
        
        # Risk metrics
        returns = self._calculate_returns()
        
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            downside = returns[returns < 0]
            if len(downside) > 0 and np.std(downside) > 0:
                sortino_ratio = np.mean(returns) / np.std(downside) * np.sqrt(252)
            else:
                sortino_ratio = 0
        else:
            sharpe_ratio = sortino_ratio = 0
        
        annual_return = self.total_pnl_pct
        calmar_ratio = annual_return / (self._max_drawdown * 100) if self._max_drawdown > 0 else 0
        
        # Period P&L
        daily_pnl = self.equity - self._daily_start_equity
        daily_pnl_pct = daily_pnl / self._daily_start_equity * 100 if self._daily_start_equity > 0 else 0
        
        week_ago = date.today() - timedelta(days=7)
        week_start_equity = self._get_equity_at_date(week_ago)
        weekly_pnl = self.equity - week_start_equity
        weekly_pnl_pct = weekly_pnl / week_start_equity * 100 if week_start_equity > 0 else 0
        
        month_ago = date.today() - timedelta(days=30)
        month_start_equity = self._get_equity_at_date(month_ago)
        monthly_pnl = self.equity - month_start_equity
        monthly_pnl_pct = monthly_pnl / month_start_equity * 100 if month_start_equity > 0 else 0
        
        return PortfolioStats(
            total_value=total_value,
            cash=self.cash,
            positions_value=self.positions_value,
            total_pnl=self.total_pnl,
            total_pnl_pct=self.total_pnl_pct,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            max_drawdown=self._max_drawdown * self._peak_equity,
            max_drawdown_pct=self._max_drawdown * 100,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl=weekly_pnl,
            weekly_pnl_pct=weekly_pnl_pct,
            monthly_pnl=monthly_pnl,
            monthly_pnl_pct=monthly_pnl_pct
        )
    
    def _calculate_returns(self) -> np.ndarray:
        """Calculate daily returns from equity history"""
        if len(self.equity_history) < 2:
            return np.array([])
        
        equities = [e[1] for e in self.equity_history]
        returns = np.diff(equities) / np.array(equities[:-1])
        return returns
    
    def _get_equity_at_date(self, target_date: date) -> float:
        """Get equity value at a specific date"""
        for ts, equity in reversed(self.equity_history):
            if ts.date() <= target_date:
                return equity
        return self.initial_capital
    
    def get_equity_curve(self) -> List[Tuple[datetime, float]]:
        """Get equity curve data"""
        return self.equity_history.copy()
    
    def get_trade_history(self) -> List[Trade]:
        """Get all trades"""
        return self.trades.copy()
    
    def get_position_summary(self) -> List[Dict]:
        """Get summary of current positions"""
        summary = []
        
        for code, pos in self.positions.items():
            summary.append({
                'code': code,
                'name': pos.name,
                'quantity': pos.quantity,
                'available': pos.available_qty,
                'cost': pos.avg_cost,
                'price': pos.current_price,
                'value': pos.market_value,
                'pnl': pos.unrealized_pnl,
                'pnl_pct': pos.unrealized_pnl_pct
            })
        
        # Sort by value descending
        summary.sort(key=lambda x: x['value'], reverse=True)
        
        return summary
    
    def _save(self):
        """Save portfolio state"""
        path = CONFIG.DATA_DIR / "portfolio.json"
        
        data = {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'peak_equity': self._peak_equity,
            'max_drawdown': self._max_drawdown,
            'trades': [
                {
                    **asdict(t),
                    'timestamp': t.timestamp.isoformat()
                }
                for t in self.trades[-1000:]  # Keep last 1000 trades
            ],
            'equity_history': [
                (ts.isoformat(), eq)
                for ts, eq in self.equity_history[-1000:]
            ],
            'saved_at': datetime.now().isoformat()
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save portfolio: {e}")
    
    def _load(self):
        """Load saved portfolio state"""
        path = CONFIG.DATA_DIR / "portfolio.json"
        
        if not path.exists():
            return
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.initial_capital = data.get('initial_capital', self.initial_capital)
            self.cash = data.get('cash', self.initial_capital)
            self._peak_equity = data.get('peak_equity', self.initial_capital)
            self._max_drawdown = data.get('max_drawdown', 0)
            
            # Load trades
            self.trades = []
            for t in data.get('trades', []):
                t['timestamp'] = datetime.fromisoformat(t['timestamp'])
                self.trades.append(Trade(**t))
            
            # Load equity history
            self.equity_history = [
                (datetime.fromisoformat(ts), eq)
                for ts, eq in data.get('equity_history', [])
            ]
            
            log.info("Portfolio state loaded")
            
        except Exception as e:
            log.warning(f"Failed to load portfolio: {e}")
    
    def reset(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.daily_performance.clear()
        self.equity_history = [(datetime.now(), self.initial_capital)]
        self._peak_equity = self.initial_capital
        self._max_drawdown = 0.0
        self._daily_start_equity = self.initial_capital
        self._daily_start_date = date.today()
        
        # Delete saved state
        path = CONFIG.DATA_DIR / "portfolio.json"
        if path.exists():
            path.unlink()
        
        log.info("Portfolio reset")