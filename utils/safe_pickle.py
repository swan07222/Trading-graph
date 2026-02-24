# utils/safe_pickle.py
"""Safe pickle deserialization with restricted classes.

This module provides secure alternatives to pickle.load/loads that
prevent arbitrary code execution by restricting which classes can
be deserialized.

SECURITY: Never unpickle data from untrusted sources without using
these restricted classes.
"""
from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import IO, Any, Set, Type

from utils.logger import get_logger

log = get_logger(__name__)

# Default safe classes for unpickling
DEFAULT_SAFE_CLASSES: Set[str] = {
    # Builtins
    "builtins.list",
    "builtins.dict",
    "builtins.tuple",
    "builtins.set",
    "builtins.frozenset",
    "builtins.str",
    "builtins.bytes",
    "builtins.bytearray",
    "builtins.int",
    "builtins.float",
    "builtins.bool",
    "builtins.complex",
    "builtins.slice",
    "builtins.range",
    "builtins.type",
    "builtins.object",
    # Datetime
    "datetime.datetime",
    "datetime.date",
    "datetime.time",
    "datetime.timedelta",
    "datetime.timezone",
    # Collections
    "collections.OrderedDict",
    "collections.defaultdict",
    "collections.deque",
    "collections.namedtuple",
    # Numpy (common in ML)
    "numpy.ndarray",
    "numpy.dtype",
    "numpy.float64",
    "numpy.float32",
    "numpy.int64",
    "numpy.int32",
    "numpy.bool_",
    # Pandas
    "pandas.DataFrame",
    "pandas.Series",
    "pandas.Index",
    "pandas.MultiIndex",
    "pandas.Categorical",
}


class RestrictedUnpickler(pickle.Unpickler):
    """Restricted unpickler that only allows safe classes.
    
    Usage:
        data = RestrictedUnpickler(file).load()
    
    Or with custom safe classes:
        data = RestrictedUnpickler(
            file,
            safe_classes={"mymodule.MyClass"}
        ).load()
    """
    
    def __init__(
        self,
        file: IO[bytes],
        *,
        safe_classes: Set[str] | None = None,
        extra_classes: dict[str, Type] | None = None,
    ) -> None:
        super().__init__(file)
        self._safe_classes = safe_classes or DEFAULT_SAFE_CLASSES
        self._extra_classes = extra_classes or {}
    
    def find_class(self, module: str, name: str) -> Type:
        """Override to restrict which classes can be unpickled.
        
        Only allows classes in the safe_classes set or extra_classes dict.
        Raises pickle.UnpicklingError for disallowed classes.
        """
        full_name = f"{module}.{name}"
        
        # Check extra classes first
        if full_name in self._extra_classes:
            return self._extra_classes[full_name]
        
        # Check safe classes
        if full_name in self._safe_classes:
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError) as e:
                raise pickle.UnpicklingError(
                    f"Safe class {full_name} not available: {e}"
                ) from e
        
        # Reject everything else
        raise pickle.UnpicklingError(
            f"Global '{full_name}' is forbidden for security reasons. "
            f"Add it to safe_classes if you trust this source."
        )


def safe_pickle_load(
    file: IO[bytes],
    *,
    safe_classes: Set[str] | None = None,
    extra_classes: dict[str, Type] | None = None,
    max_bytes: int = -1,
) -> Any:
    """Safely load pickle data from a file.
    
    Args:
        file: File object opened in binary read mode
        safe_classes: Set of allowed class names (default: DEFAULT_SAFE_CLASSES)
        extra_classes: Dict mapping class names to class objects
        max_bytes: Maximum file size in bytes (-1 for unlimited)
    
    Returns:
        Unpickled object
    
    Raises:
        pickle.UnpicklingError: If disallowed class is encountered
        ValueError: If file exceeds max_bytes
    """
    if max_bytes > 0:
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Seek back to start
        if size > max_bytes:
            raise ValueError(
                f"Pickle file is {size:,} bytes, exceeding limit of {max_bytes:,}"
            )
    
    unpickler = RestrictedUnpickler(
        file,
        safe_classes=safe_classes,
        extra_classes=extra_classes,
    )
    return unpickler.load()


def safe_pickle_loads(
    data: bytes,
    *,
    safe_classes: Set[str] | None = None,
    extra_classes: dict[str, Type] | None = None,
    max_bytes: int = -1,
) -> Any:
    """Safely load pickle data from bytes.
    
    Args:
        data: Pickled bytes data
        safe_classes: Set of allowed class names
        extra_classes: Dict mapping class names to class objects
        max_bytes: Maximum data size in bytes (-1 for unlimited)
    
    Returns:
        Unpickled object
    
    Raises:
        pickle.UnpicklingError: If disallowed class is encountered
        ValueError: If data exceeds max_bytes
    """
    if max_bytes > 0 and len(data) > max_bytes:
        raise ValueError(
            f"Pickle data is {len(data):,} bytes, exceeding limit of {max_bytes:,}"
        )
    
    file = io.BytesIO(data)
    unpickler = RestrictedUnpickler(
        file,
        safe_classes=safe_classes,
        extra_classes=extra_classes,
    )
    return unpickler.load()


def safe_pickle_dump(
    obj: Any,
    file: IO[bytes],
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> None:
    """Safely dump object to pickle file.
    
    This is just a wrapper around pickle.dump with explicit protocol.
    The security comes from the loading side (safe_pickle_load).
    
    Args:
        obj: Object to pickle
        file: File object opened in binary write mode
        protocol: Pickle protocol version (default: HIGHEST_PROTOCOL)
    """
    pickle.dump(obj, file, protocol=protocol)


def safe_pickle_dumps(
    obj: Any,
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> bytes:
    """Safely dump object to pickle bytes.
    
    Args:
        obj: Object to pickle
        protocol: Pickle protocol version
    
    Returns:
        Pickled bytes data
    """
    return pickle.dumps(obj, protocol=protocol)


def pickle_load_safe(
    path: Path | str,
    *,
    safe_classes: Set[str] | None = None,
    extra_classes: dict[str, Type] | None = None,
    max_bytes: int = 100 * 1024 * 1024,  # 100MB default limit
) -> Any:
    """Safely load pickle data from a file path.
    
    Args:
        path: Path to pickle file
        safe_classes: Set of allowed class names
        extra_classes: Dict mapping class names to class objects
        max_bytes: Maximum file size in bytes
    
    Returns:
        Unpickled object
    """
    path = Path(path)
    with open(path, "rb") as f:
        return safe_pickle_load(
            f,
            safe_classes=safe_classes,
            extra_classes=extra_classes,
            max_bytes=max_bytes,
        )


def pickle_dump_safe(
    obj: Any,
    path: Path | str,
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> None:
    """Safely dump object to a pickle file.
    
    Args:
        obj: Object to pickle
        path: Path to output file
        protocol: Pickle protocol version
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        safe_pickle_dump(obj, f, protocol=protocol)


def is_pickle_file(path: Path | str) -> bool:
    """Check if a file appears to be a pickle file.
    
    Looks at the first few bytes for pickle protocol markers.
    """
    path = Path(path)
    if not path.exists():
        return False
    
    try:
        with open(path, "rb") as f:
            header = f.read(2)
        if len(header) < 2:
            return False
        # Check for pickle protocol markers
        return (
            header[0:1] in (b"(", b"c", b"d", b"g", b"h", b"i", b"j", b"k", b"l", b"m", b"n", b"o", b"r", b"s", b"t", b"u", b"x", b"}") or
            (header[0] == 0x80 and 2 <= header[1] <= 5)  # Protocol 2-5
        )
    except (OSError, IOError):
        return False


def migrate_pickle_to_json(
    pickle_path: Path | str,
    json_path: Path | str | None = None,
    delete_pickle: bool = False,
) -> Path:
    """Migrate a pickle file to JSON format (if possible).
    
    Only works for simple data structures (dicts, lists, strings, numbers).
    
    Args:
        pickle_path: Path to pickle file
        json_path: Path for output JSON (default: same name with .json)
        delete_pickle: Whether to delete the original pickle file
    
    Returns:
        Path to the new JSON file
    
    Raises:
        ValueError: If pickle contains non-JSON-serializable data
    """
    pickle_path = Path(pickle_path)
    
    if json_path is None:
        json_path = pickle_path.with_suffix(".json")
    else:
        json_path = Path(json_path)
    
    # Load with safe unpickler (only simple types)
    data = pickle_load_safe(
        pickle_path,
        safe_classes={
            "builtins.dict",
            "builtins.list",
            "builtins.tuple",
            "builtins.str",
            "builtins.int",
            "builtins.float",
            "builtins.bool",
            "builtins.bytes",
        },
    )
    
    # Verify JSON serializable
    try:
        json.dumps(data)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Pickle data is not JSON serializable: {e}"
        ) from e
    
    # Write JSON
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    if delete_pickle:
        pickle_path.unlink()
    
    return json_path


# Convenience functions for common use cases

def load_model_weights(path: Path | str) -> Any:
    """Load ML model weights safely.
    
    Adds common ML framework classes to safe list.
    """
    extra_classes = {}
    
    # Try to add torch classes
    try:
        import torch
        extra_classes.update({
            "torch.Tensor": torch.Tensor,
            "torch.nn.parameter.Parameter": torch.nn.parameter.Parameter,
        })
    except (ImportError, OSError):
        pass
    
    # Try to add numpy classes
    try:
        import numpy as np
        extra_classes.update({
            "numpy.ndarray": np.ndarray,
            "numpy.dtype": np.dtype,
        })
    except (ImportError, OSError):
        pass
    
    return pickle_load_safe(
        path,
        extra_classes=extra_classes,
        max_bytes=500 * 1024 * 1024,  # 500MB for models
    )


def load_cached_data(path: Path | str) -> Any:
    """Load cached data safely.
    
    More restrictive - only simple data structures.
    """
    return pickle_load_safe(
        path,
        safe_classes={
            "builtins.dict",
            "builtins.list",
            "builtins.tuple",
            "builtins.str",
            "builtins.int",
            "builtins.float",
            "builtins.bool",
            "builtins.bytes",
            "datetime.datetime",
            "datetime.date",
            "collections.OrderedDict",
        },
        max_bytes=50 * 1024 * 1024,  # 50MB for cache
    )
