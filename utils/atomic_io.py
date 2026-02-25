# utils/atomic_io.py

from __future__ import annotations

import hashlib
import importlib
import json
import os
import pickle
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from utils.logger import get_logger
from utils.safe_pickle import safe_pickle_load, safe_pickle_loads

log = get_logger(__name__)

_TORCH_MODULE: Any | None = None
_TORCH_MODULE_CHECKED = False


def _get_torch_module() -> Any | None:
    """Lazily import torch to avoid hard dependency at module import time."""
    global _TORCH_MODULE_CHECKED, _TORCH_MODULE
    if _TORCH_MODULE_CHECKED:
        return _TORCH_MODULE
    _TORCH_MODULE_CHECKED = True
    try:
        _TORCH_MODULE = importlib.import_module("torch")
    except Exception:
        _TORCH_MODULE = None
    return _TORCH_MODULE

__all__ = [
    "atomic_write_bytes",
    "atomic_write_text",
    "atomic_write_json",
    "atomic_pickle_dump",
    "atomic_torch_save",
    "read_bytes",
    "read_text",
    "read_json",
    "pickle_load",
    "pickle_load_bytes",
    "torch_load",
    "artifact_checksum_path",
    "write_checksum_sidecar",
    "verify_checksum_sidecar",
    "safe_remove",
    "ensure_parent_dir",
]

# Bounded lock cache — prevents memory leak in long-running apps

_MAX_LOCKS = 256
_lock_cache: OrderedDict[str, threading.Lock] = OrderedDict()
_cache_lock = threading.Lock()

def _get_lock(path: Path) -> threading.RLock:
    """Get or create a lock for a specific file path.

    Uses a bounded LRU cache to prevent unbounded memory growth.
    Uses RLock (reentrant) to prevent deadlocks when the same thread
    needs to acquire the lock multiple times.
    """
    key = str(path.resolve())
    with _cache_lock:
        if key in _lock_cache:
            _lock_cache.move_to_end(key)
            return _lock_cache[key]

        lock = threading.RLock()
        _lock_cache[key] = lock

        while len(_lock_cache) > _MAX_LOCKS:
            _lock_cache.popitem(last=False)

        return lock

# Directory-level lock cache

_dir_lock_cache: OrderedDict[str, threading.RLock] = OrderedDict()
_dir_cache_lock = threading.Lock()

def _get_dir_lock(path: Path) -> threading.RLock:
    """Get or create a lock for the DIRECTORY containing path.

    This serializes all atomic writes to the same directory,
    preventing fd reuse races when multiple threads write to
    different files in the same directory simultaneously.
    Uses RLock (reentrant) to prevent deadlocks.
    """
    key = str(path.parent.resolve())
    with _dir_cache_lock:
        if key in _dir_lock_cache:
            _dir_lock_cache.move_to_end(key)
            return _dir_lock_cache[key]

        lock = threading.RLock()
        _dir_lock_cache[key] = lock

        while len(_dir_lock_cache) > _MAX_LOCKS:
            _dir_lock_cache.popitem(last=False)

        return lock

def _make_tmp_path(path: Path) -> Path:
    """Create a unique temp file path in the same directory as target.

    Uses PID and thread ID to avoid collisions between processes/threads.
    """
    pid = os.getpid()
    tid = threading.get_ident()
    suffix = f".{pid}.{tid}.tmp"
    return path.with_suffix(path.suffix + suffix)

def _fsync_file(f: Any) -> None:
    """Flush and fsync a file object.

    FIX EBADF: Wrapped in try/except OSError. If the fd is somehow
    invalid, we continue. flush() already pushes data to OS buffers.
    FIX: Explicitly handle all edge cases to prevent file descriptor leaks.
    """
    try:
        f.flush()
    except Exception:
        # flush() can fail if file is already closed
        pass
    
    try:
        fd = f.fileno()
        os.fsync(fd)
    except (OSError, ValueError, AttributeError, TypeError):
        # fileno() can raise ValueError if file is closed or
        # AttributeError if f doesn't support fileno (StringIO, etc.)
        # OSError if fd is invalid (e.g., already closed)
        pass

def _cleanup_tmp(tmp: Path) -> None:
    """Remove temp file if it exists. Never raises."""
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass

def _safe_replace(src: Path, dst: Path, max_retries: int = 3) -> None:
    """Replace dst with src, with retry logic for Windows.

    FIX REPLACE: On Windows, os.replace can fail with PermissionError
    if the target file is momentarily open by another process (e.g.
    antivirus scanner). A brief retry handles this transient condition.
    """
    for attempt in range(max_retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.05 * (attempt + 1))
            else:
                raise

def atomic_write_bytes(
    path: str | Path,
    data: bytes,
    use_lock: bool = True,
) -> None:
    """Atomically write bytes to a file.

    Uses write-to-temp-then-rename pattern to prevent corruption.

    Args:
        path: Target file path
        data: Bytes to write
        use_lock: If True (default), acquire a directory-level thread lock

    Raises:
        TypeError: If data is not bytes
        OSError: If write or rename fails
    """
    if not isinstance(data, bytes):
        raise TypeError(f"data must be bytes, got {type(data).__name__}")

    path = Path(path)
    tmp = _make_tmp_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lock = _get_dir_lock(path) if use_lock else None

    lock_acquired = False
    if lock:
        lock.acquire()
        lock_acquired = True
    try:
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                _fsync_file(f)
            _safe_replace(tmp, path)
        except BaseException:
            _cleanup_tmp(tmp)
            raise
    finally:
        if lock_acquired:
            lock.release()

def atomic_write_text(
    path: str | Path,
    text: str,
    encoding: str = "utf-8",
    use_lock: bool = True,
) -> None:
    """Atomically write text to a file.

    Args:
        path: Target file path
        text: Text string to write
        encoding: Text encoding (default: utf-8)
        use_lock: If True (default), acquire a directory-level thread lock
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    atomic_write_bytes(path, text.encode(encoding), use_lock=use_lock)

def atomic_write_json(
    path: str | Path,
    obj: Any,
    indent: int = 2,
    ensure_ascii: bool = False,
    use_lock: bool = True,
) -> None:
    """Atomically write a JSON-serializable object to a file.

    Args:
        path: Target file path
        obj: JSON-serializable object
        indent: JSON indentation (default: 2)
        ensure_ascii: If True, escape non-ASCII characters
        use_lock: If True (default), acquire a directory-level thread lock

    Raises:
        TypeError: If obj is not JSON-serializable
    """
    json_str = json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent)
    atomic_write_bytes(path, json_str.encode("utf-8"), use_lock=use_lock)

def atomic_pickle_dump(
    path: str | Path,
    obj: Any,
    protocol: int | None = None,
    use_lock: bool = True,
    write_checksum: bool = True,
) -> None:
    """Atomically pickle an object to a file.

    Args:
        path: Target file path
        obj: Object to pickle
        protocol: Pickle protocol (default: HIGHEST_PROTOCOL)
        use_lock: If True (default), acquire a directory-level thread lock

    Raises:
        pickle.PicklingError: If obj cannot be pickled
    """
    if protocol is None:
        protocol = pickle.HIGHEST_PROTOCOL
    data = pickle.dumps(obj, protocol=protocol)
    atomic_write_bytes(path, data, use_lock=use_lock)
    if write_checksum:
        try:
            write_checksum_sidecar(path)
        except OSError:
            pass

def atomic_torch_save(
    path: str | Path,
    obj: Any,
    use_lock: bool = True,
    write_checksum: bool = True,
) -> None:
    """Atomically save a PyTorch object to a file.

    FIX EBADF: Opens file handle ourselves, passes to torch.save(),
    then fsyncs the SAME handle. No close/reopen gap.

    Args:
        path: Target file path
        obj: Object to save (state_dict, checkpoint, etc.)
        use_lock: If True (default), acquire a directory-level thread lock

    Raises:
        ImportError: If torch is not installed
    """
    torch_mod = _get_torch_module()
    if torch_mod is None:
        raise ImportError(
            "PyTorch is required for atomic_torch_save. "
            "Install with: pip install torch"
        )

    path = Path(path)
    tmp = _make_tmp_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lock = _get_dir_lock(path) if use_lock else None

    saved = False
    lock_acquired = False
    if lock:
        lock.acquire()
        lock_acquired = True
    try:
        try:
            with open(tmp, "wb") as f:
                torch_mod.save(obj, f)
                _fsync_file(f)
            _safe_replace(tmp, path)
            saved = True
        except BaseException:
            _cleanup_tmp(tmp)
            raise
    finally:
        if lock_acquired:
            lock.release()

    # Write checksum after releasing the directory lock. Otherwise this path
    # can deadlock by trying to reacquire the same non-reentrant lock.
    if saved and write_checksum:
        try:
            write_checksum_sidecar(path)
        except OSError:
            pass

_DEFAULT_MAX_PICKLE_BYTES = 500 * 1024 * 1024  # 500 MB
_CHECKSUM_SUFFIX = ".sha256"

def read_bytes(path: str | Path) -> bytes:
    """Read entire file contents as bytes.

    Note: Not truly atomic. If another process is performing an
    atomic write to the same path, you will always read a complete
    version (old or new) thanks to the rename pattern.

    Args:
        path: File path

    Returns:
        File contents as bytes

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    with open(Path(path), "rb") as f:
        return f.read()

def read_text(
    path: str | Path,
    encoding: str = "utf-8",
) -> str:
    """Read file contents as text.

    Args:
        path: File path
        encoding: Text encoding (default: utf-8)

    Returns:
        File contents as string
    """
    return read_bytes(path).decode(encoding)

def read_json(path: str | Path) -> Any:
    """Read and parse a JSON file.

    Args:
        path: File path

    Returns:
        Parsed JSON object

    Raises:
        json.JSONDecodeError: If file is not valid JSON
    """
    return json.loads(read_text(path))

def artifact_checksum_path(path: str | Path) -> Path:
    """Return sidecar checksum path for an artifact.

    Example:
        model.pt -> model.pt.sha256
    """
    p = Path(path)
    return p.with_suffix(p.suffix + _CHECKSUM_SUFFIX)

def _sha256_file(path: Path) -> str:
    """Compute SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def write_checksum_sidecar(path: str | Path) -> Path:
    """Write SHA-256 sidecar for the artifact and return sidecar path."""
    src = Path(path)
    sidecar = artifact_checksum_path(src)
    payload = f"{_sha256_file(src)}\n"
    atomic_write_text(sidecar, payload)
    return sidecar

def verify_checksum_sidecar(
    path: str | Path,
    *,
    require: bool = True,
) -> bool:
    """Verify artifact checksum sidecar.

    Args:
        path: Artifact file path.
        require: When True, missing sidecar fails verification.

    Returns:
        True when verification passes, else False.
    """
    src = Path(path)
    sidecar = artifact_checksum_path(src)

    if not sidecar.exists():
        return not require

    try:
        expected_raw = sidecar.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    expected = expected_raw.split()[0].lower() if expected_raw else ""
    if not expected:
        return False

    try:
        actual = _sha256_file(src)
    except OSError:
        return False
    return actual == expected

def pickle_load(
    path: str | Path,
    max_bytes: int = _DEFAULT_MAX_PICKLE_BYTES,
    verify_checksum: bool = False,
    require_checksum: bool = True,
    allow_unsafe: bool = False,
    safe_classes: set[str] | None = None,
) -> Any:
    """Load a pickled object from a file.

    SECURITY FIX: Now uses safe_pickle_load by default with restricted classes.
    Only set allow_unsafe=True for legacy code migration.

    Args:
        path: File path
        max_bytes: Maximum file size to load (default: 500 MB).
                   Set to 0 to disable size check.
        verify_checksum: Verify checksum sidecar before loading
        require_checksum: Require checksum sidecar to exist
        allow_unsafe: If True, uses unsafe pickle.load (NOT RECOMMENDED)
        safe_classes: Custom set of allowed classes (only used if allow_unsafe=False)

    Returns:
        Deserialized object

    Raises:
        ValueError: If file exceeds max_bytes or checksum verification fails
        pickle.UnpicklingError: If disallowed class is encountered (safe mode)
    """
    path = Path(path)

    if verify_checksum and not verify_checksum_sidecar(path, require=require_checksum):
        raise ValueError(
            f"Checksum verification failed for pickle artifact: {path}"
        )

    if max_bytes > 0:
        size = path.stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"Pickle file {path} is {size:,} bytes, "
                f"exceeding limit of {max_bytes:,} bytes"
            )

    if allow_unsafe:
        log.warning(
            "pickle_load called with allow_unsafe=True - "
            "this is unsafe and will be removed in a future version. "
            "Migrate to safe_pickle_load from utils.safe_pickle"
        )
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        # Use safe unpickler by default
        # FIX: Open file and pass file object to safe_pickle_load
        with open(path, "rb") as f:
            return safe_pickle_load(
                f,  # Pass file object, not path
                safe_classes=safe_classes,
                max_bytes=max_bytes,
            )


def pickle_load_bytes(
    data: bytes,
    max_bytes: int = _DEFAULT_MAX_PICKLE_BYTES,
    allow_unsafe: bool = False,
    safe_classes: set[str] | None = None,
) -> Any:
    """Load a pickled object from in-memory bytes.

    SECURITY FIX: Now uses safe_pickle_loads by default with restricted classes.
    Only set allow_unsafe=True for legacy code migration.

    Args:
        data: Pickled bytes data
        max_bytes: Maximum payload size (0 to disable check)
        allow_unsafe: If True, uses unsafe pickle.loads (NOT RECOMMENDED)
        safe_classes: Custom set of allowed classes (only used if allow_unsafe=False)

    Returns:
        Deserialized object

    Raises:
        TypeError: If data is not bytes
        ValueError: If data exceeds max_bytes
        pickle.UnpicklingError: If disallowed class is encountered (safe mode)
    """
    if not isinstance(data, bytes):
        raise TypeError(f"data must be bytes, got {type(data).__name__}")

    if max_bytes > 0 and len(data) > max_bytes:
        raise ValueError(
            f"Pickle payload is {len(data):,} bytes, "
            f"exceeding limit of {max_bytes:,} bytes"
        )

    if allow_unsafe:
        log.warning(
            "pickle_load_bytes called with allow_unsafe=True - "
            "this is unsafe and will be removed in a future version."
        )
        return pickle.loads(data)
    else:
        # Use safe unpickler by default
        return safe_pickle_loads(data, safe_classes=safe_classes, max_bytes=0)

def torch_load(
    path: str | Path,
    map_location: str | None = None,
    weights_only: bool = True,
    verify_checksum: bool = False,
    require_checksum: bool = True,
    allow_unsafe: bool = False,
) -> Any:
    """Load a PyTorch object from a file.

    Args:
        path: File path
        map_location: Device mapping (e.g., 'cpu', 'cuda:0')
        weights_only: If True, only load weights — safer (default: True)

    Returns:
        Loaded PyTorch object

    Raises:
        ImportError: If torch is not installed
    """
    torch_mod = _get_torch_module()
    if torch_mod is None:
        raise ImportError(
            "PyTorch is required for torch_load. "
            "Install with: pip install torch"
        )
    if (not bool(weights_only)) and (not bool(allow_unsafe)):
        raise ValueError(
            "Unsafe torch checkpoint load requires explicit allow_unsafe=True"
        )
    if verify_checksum and not verify_checksum_sidecar(path, require=require_checksum):
        raise ValueError(
            f"Checksum verification failed for torch artifact: {path}"
        )
    target = str(Path(path))
    try:
        return torch_mod.load(
            target,
            map_location=map_location,
            weights_only=weights_only,
        )
    except TypeError:
        # Older torch versions may not support the weights_only argument.
        if not bool(allow_unsafe):
            raise
        return torch_mod.load(
            target,
            map_location=map_location,
        )

def safe_remove(path: str | Path) -> bool:
    """Remove a file, returning False instead of raising if it fails.

    Args:
        path: File path

    Returns:
        True if removed, False otherwise
    """
    try:
        Path(path).unlink(missing_ok=True)
        return True
    except OSError:
        return False

def ensure_parent_dir(path: str | Path) -> Path:
    """Ensure the parent directory of a path exists.

    Args:
        path: File path

    Returns:
        The path as a Path object
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
