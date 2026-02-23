# utils/__init__.py
from .atomic_io import (
    atomic_pickle_dump,
    atomic_torch_save,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    ensure_parent_dir,
    pickle_load,
    read_bytes,
    read_json,
    read_text,
    safe_remove,
    torch_load,
)
from .cancellation import CancellationToken, CancelledException, cancellable_operation
from .helpers import (
    calculate_max_drawdown,
    calculate_sharpe,
    format_number,
    format_pct,
    format_price,
    get_trading_dates,
)
from .json_io import read_json_safe, write_json_safe, read_jsonl, write_jsonl
from .lazy_imports import lazy_get, LazyImport, CachedLazyImport, make_lazy_getter
from .logger import get_logger, log, setup_logging, teardown_logging
from .serialization import (
    safe_dataclass_from_dict,
    dataclass_to_dict,
    to_serializable,
)
from .type_utils import (
    safe_float,
    safe_int,
    safe_str,
    safe_attr,
    safe_float_attr,
    safe_int_attr,
    safe_str_attr,
    clamp,
)

__all__ = [
    "log",
    "get_logger",
    "setup_logging",
    "teardown_logging",
    "format_number",
    "format_pct",
    "format_price",
    "get_trading_dates",
    "calculate_sharpe",
    "calculate_max_drawdown",
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
    # JSON I/O
    "read_json_safe",
    "write_json_safe",
    "read_jsonl",
    "write_jsonl",
    # Lazy imports
    "lazy_get",
    "LazyImport",
    "CachedLazyImport",
    "make_lazy_getter",
    # Serialization
    "safe_dataclass_from_dict",
    "dataclass_to_dict",
    "to_serializable",
    # Type utilities
    "safe_float",
    "safe_int",
    "safe_str",
    "safe_attr",
    "safe_float_attr",
    "safe_int_attr",
    "safe_str_attr",
    "clamp",
]