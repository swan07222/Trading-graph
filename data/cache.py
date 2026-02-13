# data/cache.py
import gzip
import hashlib
import os
import pickle
import sys
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)

T = TypeVar("T")
_MANUAL_DELETE_ENV = "TRADING_MANUAL_CACHE_DELETE"


def _cache_delete_allowed() -> bool:
    return os.environ.get(_MANUAL_DELETE_ENV, "0") == "1"

@dataclass
class CacheStats:
    """Thread-safe cache statistics."""

    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0
    total_sets: int = 0
    total_evictions: int = 0

    def increment(self, field_name: str, amount: int = 1):
        with self._lock:
            setattr(self, field_name, getattr(self, field_name) + amount)

    @property
    def total_hits(self) -> int:
        return self.l1_hits + self.l2_hits + self.l3_hits

    @property
    def total_misses(self) -> int:
        return self.l3_misses  # Final miss

    @property
    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        return self.total_hits / max(total, 1)

@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    data: Any
    created_at: datetime
    size_bytes: int = 0
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)

# L1: In-memory LRU with TTL

_SENTINEL = object()

class LRUCache:
    """Thread-safe LRU cache with TTL enforcement."""

    def __init__(self, max_items: int = 1000, max_size_mb: int = 500):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_items = max_items
        self._max_size = max_size_mb * 1024 * 1024
        self._current_size = 0
        self._lock = threading.RLock()

    def get(
        self, key: str, max_age_hours: float = None
    ) -> Any:
        """
        Get value from cache.

        Returns _SENTINEL (module-level) on miss so that ``None``
        can be a valid cached value.
        """
        with self._lock:
            if key not in self._cache:
                return _SENTINEL

            entry = self._cache[key]

            if max_age_hours is not None:
                age_hours = (
                    datetime.now() - entry.created_at
                ).total_seconds() / 3600
                if age_hours > max_age_hours:
                    # Expired — evict
                    self._evict_key(key)
                    return _SENTINEL

            entry.access_count += 1
            entry.last_access = datetime.now()
            self._cache.move_to_end(key)
            return entry.data

    def set(self, key: str, value: Any, size_bytes: int = 0):
        with self._lock:
            if key in self._cache:
                self._evict_key(key)

            if size_bytes <= 0:
                size_bytes = self._estimate_size(value)

            # Evict until there's room
            while (
                len(self._cache) >= self._max_items
                or self._current_size + size_bytes > self._max_size
            ):
                if not self._cache:
                    break
                self._evict_oldest()

            entry = CacheEntry(
                data=value,
                created_at=datetime.now(),
                size_bytes=size_bytes,
            )
            self._cache[key] = entry
            self._current_size += size_bytes

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                self._evict_key(key)
                return True
            return False

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._current_size = 0

    def _evict_key(self, key: str):
        """Remove a specific key (caller must hold lock)."""
        entry = self._cache.pop(key, None)
        if entry:
            self._current_size -= entry.size_bytes

    def _evict_oldest(self):
        """Remove oldest entry (caller must hold lock)."""
        if self._cache:
            _, entry = self._cache.popitem(last=False)
            self._current_size -= entry.size_bytes

    @staticmethod
    def _estimate_size(value: Any) -> int:
        """Estimate object memory footprint."""
        if isinstance(value, pd.DataFrame):
            return int(value.memory_usage(deep=True).sum())
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, (str, bytes)):
            return len(value)
        if isinstance(value, dict):
            try:
                return sys.getsizeof(value) + sum(
                    sys.getsizeof(k) + LRUCache._estimate_size(v)
                    for k, v in value.items()
                )
            except (TypeError, RecursionError):
                return sys.getsizeof(value)
        if isinstance(value, (list, tuple)):
            try:
                return sys.getsizeof(value) + sum(
                    LRUCache._estimate_size(item) for item in value
                )
            except (TypeError, RecursionError):
                return sys.getsizeof(value)
        # Use sys.getsizeof as a rough fallback
        try:
            return sys.getsizeof(value)
        except TypeError:
            return 1024  # Default fallback

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def size_mb(self) -> float:
        return self._current_size / (1024 * 1024)

# Disk cache (L2 / L3)

class DiskCache:
    """
    Disk-based cache with atomic writes.

    Uses write-to-temp-then-rename to prevent corruption on crash.

    FIX C4: Closes file descriptor immediately after mkstemp to prevent
    fd leak and avoid Windows PermissionError when renaming.
    """

    def __init__(self, cache_dir: Path, compress: bool = False):
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._compress = compress
        self._lock = threading.RLock()

    def _key_to_path(self, key: str) -> Path:
        # SHA-256 avoids MD5 collision risk
        h = hashlib.sha256(key.encode()).hexdigest()
        ext = ".pkl.gz" if self._compress else ".pkl"
        return self._dir / f"{h}{ext}"

    def get(
        self, key: str, max_age_hours: float = None
    ) -> Any:
        """
        Read from disk cache.

        Returns _SENTINEL on miss/expiry/error.
        """
        path = self._key_to_path(key)

        if not path.exists():
            return _SENTINEL

        if max_age_hours is not None:
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                age_hours = (
                    datetime.now() - mtime
                ).total_seconds() / 3600
                if age_hours > max_age_hours:
                    return _SENTINEL
            except OSError:
                return _SENTINEL

        try:
            with self._lock:
                if self._compress:
                    with gzip.open(path, "rb") as f:
                        return pickle.load(f)
                else:
                    with open(path, "rb") as f:
                        return pickle.load(f)
        except Exception as e:
            log.warning(f"Cache read error for key hash {path.stem}: {e}")
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return _SENTINEL

    def set(self, key: str, value: Any):
        """
        Atomic write: temp file → rename.

        FIX C4: Close the file descriptor from mkstemp IMMEDIATELY
        before opening the file with gzip.open or open(). This prevents:
        1. File descriptor leak (the fd stays open until finally block)
        2. Windows PermissionError (two handles to same file)
        """
        path = self._key_to_path(key)
        tmp_path: Path | None = None

        try:
            with self._lock:
                # Create temp file - mkstemp returns (fd, path)
                fd, tmp_path_str = tempfile.mkstemp(
                    dir=str(self._dir), suffix=".tmp"
                )

                # FIX C4: Close fd IMMEDIATELY - we'll reopen with gzip/open
                try:
                    os.close(fd)
                except OSError:
                    pass

                tmp_path = Path(tmp_path_str)

                # Now write to the file (no fd leak)
                if self._compress:
                    with gzip.open(tmp_path, "wb") as f:
                        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(tmp_path, "wb") as f:
                        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

                tmp_path.replace(path)
                tmp_path = None  # Successfully moved, don't delete in finally

        except Exception as e:
            log.warning(f"Cache write error: {e}")
        finally:
            # Clean up temp file if it still exists (rename failed or exception)
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def delete(self, key: str) -> bool:
        if not _cache_delete_allowed():
            log.warning("Cache delete blocked for key=%s (manual override required)", key)
            return False
        path = self._key_to_path(key)
        try:
            if path.exists():
                path.unlink()
                return True
        except OSError:
            pass
        return False

    def clear(self, older_than_hours: float = None):
        """Clear cache, optionally only old files."""
        if not _cache_delete_allowed():
            log.warning("Cache clear blocked (manual override required)")
            return
        now = datetime.now()
        patterns = ["*.pkl", "*.pkl.gz"]
        for pattern in patterns:
            for path in self._dir.glob(pattern):
                try:
                    if older_than_hours is not None:
                        mtime = datetime.fromtimestamp(path.stat().st_mtime)
                        age = (now - mtime).total_seconds() / 3600
                        if age <= older_than_hours:
                            continue
                    path.unlink()
                except OSError:
                    pass

class TieredCache:
    """
    Three-tier caching system.

    L1: Fast in-memory LRU with TTL
    L2: Disk cache (fast reads)
    L3: Compressed disk (persistent, longer TTL)
    """

    # Bounded thread pool for L3 background writes
    _L3_MAX_WORKERS = 4

    def __init__(self):
        self._l1 = LRUCache(
            max_items=500,
            max_size_mb=CONFIG.data.max_memory_cache_mb,
        )
        self._l2 = DiskCache(CONFIG.cache_dir / "l2", compress=False)
        self._l3 = DiskCache(CONFIG.cache_dir / "l3", compress=True)
        self._stats = CacheStats()
        self._lock = threading.RLock()

        # Bounded executor for L3 writes (replaces unbounded Thread())
        from concurrent.futures import ThreadPoolExecutor

        self._l3_executor = ThreadPoolExecutor(
            max_workers=self._L3_MAX_WORKERS,
            thread_name_prefix="cache_l3",
        )

    def get(
        self, key: str, max_age_hours: float = None
    ) -> Any | None:
        """Get value with tiered lookup. Returns None on miss."""
        max_age = max_age_hours or CONFIG.data.cache_ttl_hours

        # L1: Memory (with TTL)
        value = self._l1.get(key, max_age_hours=max_age)
        if value is not _SENTINEL:
            self._stats.increment("l1_hits")
            return value
        self._stats.increment("l1_misses")

        # L2: Disk
        value = self._l2.get(key, max_age)
        if value is not _SENTINEL:
            self._stats.increment("l2_hits")
            self._l1.set(key, value)  # Promote
            return value
        self._stats.increment("l2_misses")

        # L3: Compressed disk (longer TTL)
        value = self._l3.get(key, max_age * 24)
        if value is not _SENTINEL:
            self._stats.increment("l3_hits")
            self._l1.set(key, value)
            self._l2.set(key, value)
            return value
        self._stats.increment("l3_misses")

        return None

    def set(self, key: str, value: Any, persist: bool = True):
        """Store value in cache tiers."""
        self._stats.increment("total_sets")

        # Always L1
        self._l1.set(key, value)

        if persist:
            # L2 synchronously (fast)
            self._l2.set(key, value)
            # L3 asynchronously via bounded pool
            self._l3_executor.submit(self._l3.set, key, value)

    def delete(self, key: str):
        """Delete from all tiers."""
        if not _cache_delete_allowed():
            log.warning("Tiered cache delete blocked for key=%s (manual override required)", key)
            return
        self._l1.delete(key)
        self._l2.delete(key)
        self._l3.delete(key)

    def clear(
        self,
        tier: str = None,
        older_than_hours: float = None,
    ):
        """Clear cache (optionally one tier or old entries only)."""
        if not _cache_delete_allowed():
            log.warning("Tiered cache clear blocked (manual override required)")
            return
        if tier is None or tier == "l1":
            self._l1.clear()
        if tier is None or tier == "l2":
            self._l2.clear(older_than_hours)
        if tier is None or tier == "l3":
            self._l3.clear(older_than_hours)

    def get_stats(self) -> CacheStats:
        return self._stats

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
        max_age_hours: float = None,
        persist: bool = True,
    ) -> T:
        """
        Get from cache or compute and store.

        Uses a single stats-counted lookup before compute,
        then a raw L1 check after compute to avoid double-counting.
        """
        # First lookup (stats counted once)
        value = self.get(key, max_age_hours)
        if value is not None:
            return value

        value = compute_fn()

        with self._lock:
            # Double-check with raw L1 lookup (no stats)
            existing = self._l1.get(
                key, max_age_hours=max_age_hours or CONFIG.data.cache_ttl_hours
            )
            if existing is not _SENTINEL:
                return existing

            if value is not None:
                self.set(key, value, persist)

        return value

    def shutdown(self):
        """Shutdown L3 background writer pool."""
        try:
            self._l3_executor.shutdown(wait=True, cancel_futures=False)
        except TypeError:
            # Python < 3.9 doesn't have cancel_futures
            self._l3_executor.shutdown(wait=True)

_cache: TieredCache | None = None
_cache_lock = threading.Lock()

def get_cache() -> TieredCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = TieredCache()
    return _cache

def cached(
    key_fn: Callable[..., str] = None,
    max_age_hours: float = None,
    persist: bool = True,
):
    """
    Caching decorator.

    Args:
        key_fn: Function to compute cache key from args/kwargs.
                If None, a default key is derived from module + function name + args.
        max_age_hours: TTL override.
        persist: Whether to persist to disk tiers.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{func.__module__}.{func.__qualname__}:{args}:{kwargs}"

            result = _cache.get(key, max_age_hours) if _cache else None
            if result is not None:
                return result

            result = func(*args, **kwargs)

            if result is not None and _cache:
                _cache.set(key, result, persist)

            return result

        return wrapper

    return decorator
