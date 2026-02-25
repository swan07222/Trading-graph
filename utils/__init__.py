"""Utility package exports with lazy imports.

This keeps package import lightweight and avoids importing optional heavy
dependencies (for example PyTorch) unless explicitly requested.
"""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    # Logging
    "log": (".logger", "log"),
    "get_logger": (".logger", "get_logger"),
    "setup_logging": (".logger", "setup_logging"),
    "teardown_logging": (".logger", "teardown_logging"),
    # Helpers
    "format_number": (".helpers", "format_number"),
    "format_pct": (".helpers", "format_pct"),
    "format_price": (".helpers", "format_price"),
    "get_trading_dates": (".helpers", "get_trading_dates"),
    "calculate_sharpe": (".helpers", "calculate_sharpe"),
    "calculate_max_drawdown": (".helpers", "calculate_max_drawdown"),
    # Cancellation
    "CancellationToken": (".cancellation", "CancellationToken"),
    "CancelledException": (".cancellation", "CancelledException"),
    "cancellable_operation": (".cancellation", "cancellable_operation"),
    # Atomic I/O
    "atomic_write_bytes": (".atomic_io", "atomic_write_bytes"),
    "atomic_write_text": (".atomic_io", "atomic_write_text"),
    "atomic_write_json": (".atomic_io", "atomic_write_json"),
    "atomic_pickle_dump": (".atomic_io", "atomic_pickle_dump"),
    "atomic_torch_save": (".atomic_io", "atomic_torch_save"),
    "read_bytes": (".atomic_io", "read_bytes"),
    "read_text": (".atomic_io", "read_text"),
    "read_json": (".atomic_io", "read_json"),
    "pickle_load": (".atomic_io", "pickle_load"),
    "torch_load": (".atomic_io", "torch_load"),
    "safe_remove": (".atomic_io", "safe_remove"),
    "ensure_parent_dir": (".atomic_io", "ensure_parent_dir"),
    # JSON I/O
    "read_json_safe": (".json_io", "read_json_safe"),
    "write_json_safe": (".json_io", "write_json_safe"),
    "read_jsonl": (".json_io", "read_jsonl"),
    "write_jsonl": (".json_io", "write_jsonl"),
    # Lazy imports helpers
    "lazy_get": (".lazy_imports", "lazy_get"),
    "LazyImport": (".lazy_imports", "LazyImport"),
    "CachedLazyImport": (".lazy_imports", "CachedLazyImport"),
    "make_lazy_getter": (".lazy_imports", "make_lazy_getter"),
    # Serialization
    "safe_dataclass_from_dict": (".serialization", "safe_dataclass_from_dict"),
    "dataclass_to_dict": (".serialization", "dataclass_to_dict"),
    "to_serializable": (".serialization", "to_serializable"),
    # Type utils
    "safe_float": (".type_utils", "safe_float"),
    "safe_int": (".type_utils", "safe_int"),
    "safe_str": (".type_utils", "safe_str"),
    "safe_attr": (".type_utils", "safe_attr"),
    "safe_float_attr": (".type_utils", "safe_float_attr"),
    "safe_int_attr": (".type_utils", "safe_int_attr"),
    "safe_str_attr": (".type_utils", "safe_str_attr"),
    "clamp": (".type_utils", "clamp"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(str(name))
    if target is None:
        raise AttributeError(f"module 'utils' has no attribute {name!r}")
    mod_name, attr_name = target
    module = import_module(mod_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
