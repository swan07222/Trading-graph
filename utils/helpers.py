# utils/helpers.py
"""
Helper functions for formatting, date handling, and performance metrics.

FIXES APPLIED:
- Added NaN/Inf guards to all formatting functions
- calculate_sharpe uses np.isclose instead of == 0
- calculate_max_drawdown handles edge cases (empty, all-zero, negative)
- get_trading_dates documents the weekend-only limitation
- Added input validation to metric functions
- Added type hints throughout
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Union

import numpy as np


# =====================================================================
# Formatting
# =====================================================================


def format_number(n: float, decimals: int = 2) -> str:
    """
    Format a number with Chinese unit suffixes (亿, 万).

    Args:
        n: Number to format
        decimals: Decimal places

    Returns:
        Formatted string

    Examples:
        >>> format_number(123456789)
        '1.23亿'
        >>> format_number(12345)
        '1.23万'
        >>> format_number(float('nan'))
        'N/A'
    """
    if math.isnan(n):
        return "N/A"
    if math.isinf(n):
        return "∞" if n > 0 else "-∞"

    abs_n = abs(n)
    if abs_n >= 1e8:
        return f"{n / 1e8:.{decimals}f}亿"
    elif abs_n >= 1e4:
        return f"{n / 1e4:.{decimals}f}万"
    else:
        return f"{n:,.{decimals}f}"


def format_pct(n: float, decimals: int = 2) -> str:
    """
    Format a number as a percentage with sign.

    Args:
        n: Percentage value (e.g., 5.25 for 5.25%)
        decimals: Decimal places

    Returns:
        Formatted string like "+5.25%" or "-3.10%"

    Examples:
        >>> format_pct(5.25)
        '+5.25%'
        >>> format_pct(-3.1)
        '-3.10%'
        >>> format_pct(0.0)
        '0.00%'
        >>> format_pct(float('nan'))
        'N/A'
    """
    if math.isnan(n):
        return "N/A"
    if math.isinf(n):
        return "+∞%" if n > 0 else "-∞%"

    sign = "+" if n > 0 else ""
    return f"{sign}{n:.{decimals}f}%"


def format_price(n: float, currency: str = "¥") -> str:
    """
    Format a price with currency symbol.

    Args:
        n: Price value
        currency: Currency symbol (default: ¥)

    Returns:
        Formatted string like "¥1,234.56"

    Examples:
        >>> format_price(1234.5)
        '¥1,234.50'
        >>> format_price(float('nan'))
        'N/A'
    """
    if math.isnan(n):
        return "N/A"
    if math.isinf(n):
        return f"{currency}∞" if n > 0 else f"-{currency}∞"

    return f"{currency}{n:,.2f}"


# =====================================================================
# Date Utilities
# =====================================================================


def get_trading_dates(
    start: datetime,
    end: datetime,
    exclude_weekends: bool = True,
) -> List[datetime]:
    """
    Get list of potential trading dates between start and end (inclusive).

    NOTE: This only filters weekends. It does NOT account for Chinese
    market holidays (Spring Festival, National Day, etc.). For
    production use, integrate a holiday calendar (e.g., exchange_calendars
    or chinese_calendar package).

    Args:
        start: Start date (inclusive)
        end: End date (inclusive)
        exclude_weekends: If True, exclude Saturday and Sunday

    Returns:
        List of datetime objects
    """
    if start > end:
        return []

    dates: list[datetime] = []
    current = start
    while current <= end:
        if not exclude_weekends or current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


# =====================================================================
# Performance Metrics
# =====================================================================


def calculate_sharpe(
    returns: Union[np.ndarray, Sequence[float]],
    risk_free_annual: float = 0.03,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Array of periodic returns (e.g., daily returns as decimals)
        risk_free_annual: Annual risk-free rate (default: 3%)
        periods_per_year: Trading periods per year (default: 252 for daily)

    Returns:
        Annualized Sharpe ratio, or 0.0 if insufficient data or zero volatility

    Examples:
        >>> returns = np.array([0.01, 0.02, -0.01, 0.005, 0.015])
        >>> calculate_sharpe(returns)  # some positive number
    """
    returns = np.asarray(returns, dtype=np.float64)

    # Filter out NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return 0.0

    # Convert annual risk-free rate to per-period rate
    rf_per_period = risk_free_annual / periods_per_year
    excess = returns - rf_per_period

    std = np.std(excess, ddof=1)

    # Use isclose instead of == 0 for floating point safety
    if np.isclose(std, 0.0) or np.isnan(std):
        return 0.0

    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def calculate_max_drawdown(
    equity: Union[np.ndarray, Sequence[float]],
) -> float:
    """
    Calculate maximum drawdown from an equity curve.

    Args:
        equity: Array of portfolio values over time (must be positive)

    Returns:
        Maximum drawdown as a positive fraction (e.g., 0.25 for 25%)
        Returns 0.0 if equity curve is empty or non-positive

    Examples:
        >>> equity = np.array([100, 110, 105, 95, 100])
        >>> calculate_max_drawdown(equity)  # ≈ 0.1364 (from 110 to 95)
    """
    equity = np.asarray(equity, dtype=np.float64)

    # Filter NaN
    equity = equity[~np.isnan(equity)]

    if len(equity) == 0:
        return 0.0

    # Equity must be positive for drawdown to be meaningful
    if np.any(equity <= 0):
        return 0.0

    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max
    max_dd = np.max(drawdown)

    return float(max_dd) if not np.isnan(max_dd) else 0.0