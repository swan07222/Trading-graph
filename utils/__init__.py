# utils/__init__.py
"""
Utility modules for the trading system.

Exports:
- Logging: get_logger, setup_logging, teardown_logging, log
- Formatting: format_number, format_pct, format_price
- Date utilities: get_trading_dates
- Metrics: calculate_sharpe, calculate_max_drawdown
- Cancellation: CancellationToken, CancelledException, cancellable_operation
- Atomic I/O: atomic_write_bytes, atomic_write_json, etc.
"""
from .logger import get_logger, setup_logging, teardown_logging, log
from .helpers import (
    format_number,
    format_pct,
    format_price,
    get_trading_dates,
    calculate_sharpe,
    calculate_max_drawdown,
)
from .cancellation import CancellationToken, CancelledException, cancellable_operation
from .atomic_io import (
    atomic_write_bytes,
    atomic_write_text,
    atomic_write_json,
    atomic_pickle_dump,
    atomic_torch_save,
    read_bytes,
    read_text,
    read_json,
    pickle_load,
    torch_load,
    safe_remove,
    ensure_parent_dir,
)

__all__ = [
    # Logger
    "log",
    "get_logger",
    "setup_logging",
    "teardown_logging",
    # Formatters
    "format_number",
    "format_pct",
    "format_price",
    # Date utilities
    "get_trading_dates",
    # Metrics
    "calculate_sharpe",
    "calculate_max_drawdown",
    # Cancellation
    "CancellationToken",
    "CancelledException",
    "cancellable_operation",
    # Atomic I/O — Writers
    "atomic_write_bytes",
    "atomic_write_text",
    "atomic_write_json",
    "atomic_pickle_dump",
    "atomic_torch_save",
    # Atomic I/O — Readers
    "read_bytes",
    "read_text",
    "read_json",
    "pickle_load",
    "torch_load",
    # Atomic I/O — Utilities
    "safe_remove",
    "ensure_parent_dir",
]