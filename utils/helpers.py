"""
Helper Functions
"""
from datetime import datetime, timedelta
from typing import List
import numpy as np


def format_number(n: float, decimals: int = 2) -> str:
    """Format number with commas"""
    if abs(n) >= 1e8:
        return f"{n/1e8:.{decimals}f}亿"
    elif abs(n) >= 1e4:
        return f"{n/1e4:.{decimals}f}万"
    else:
        return f"{n:,.{decimals}f}"


def format_pct(n: float, decimals: int = 2) -> str:
    """Format percentage"""
    sign = "+" if n > 0 else ""
    return f"{sign}{n:.{decimals}f}%"


def format_price(n: float) -> str:
    """Format price"""
    return f"¥{n:,.2f}"


def get_trading_dates(start: datetime, end: datetime) -> List[datetime]:
    """Get list of trading dates (exclude weekends)"""
    dates = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            dates.append(current)
        current += timedelta(days=1)
    return dates


def calculate_sharpe(returns: np.ndarray, risk_free: float = 0.03) -> float:
    """Calculate annualized Sharpe ratio"""
    if len(returns) < 2:
        return 0
    
    excess = returns - risk_free / 252
    if np.std(excess) == 0:
        return 0
    
    return np.mean(excess) / np.std(excess) * np.sqrt(252)


def calculate_max_drawdown(equity: np.ndarray) -> float:
    """Calculate maximum drawdown"""
    if len(equity) == 0:
        return 0
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / (running_max + 1e-8)
    return abs(np.min(drawdown))