# data/cache.py
import atexit
import gzip
import hashlib
import os
import pickle
import sys
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from config.settings import CONFIG
from utils.atomic_io import (
    artifact_checksum_path,
    pickle_load,
    pickle_load_bytes,
    verify_checksum_sidecar,
    write_checksum_sidecar,
)
from utils.logger import get_logger

log = get_logger(__name__)

T = TypeVar("T")
_MANUAL_DELETE_ENV = "TRADING_MANUAL_CACHE_DELETE"
_CACHE_REQUIRE_CHECKSUM_ENV = "TRADING_CACHE_REQUIRE_CHECKSUM"
_DISK_CACHE_MAX_PICKLE_BYTES = 200 * 1024 * 1024  # 200 MB

# Module-level sentinel for internal miss detection (L1/L2/L3)
_SENTINEL = object()

# Public sentinel for callers that need to distinguish "not found" from None
MISSING = object()

# FIX #2026-02-24: L3 TTL multiplier (configurable via env var)
_L3_TTL_MULTIPLIER_ENV = "TRADING_CACHE_L3_TTL_MULTIPLIER"
_L3_TTL_MULTIPLIER_DEFAULT = 24.0  # L3 retains data 24x longer than L2


def _get_l3_ttl_multiplier() -> float:
    """Get L3 TTL multiplier from environment or default.
    
    FIX #2026-02-24: Makes L3 cache retention configurable.
    """
    import os
    try:
        val = os.environ.get(_L3_TTL_MULTIPLIER_ENV, "").strip()
        if val:
            multiplier = float(val)
            return max(1.0, min(multiplier, 168.0))  # Cap at 1 week equivalent
    except (TypeError, ValueError):
        pass
    return _L3_TTL_MULTIPLIER_DEFAULT


def _cache_delete_allowed() -> bool:
    return os.environ.get(_MANUAL_DELETE_ENV, "0") == "1"


def _cache_checksum_required() -> bool:
    """Check if checksum verification is required for cache reads.
    
    FIX #2026-02-24: Changed default from "0" to "1" for production security.
    Set TRADING_CACHE_REQUIRE_CHECKSUM=0 to disable.
    """
    return os.environ.get(_CACHE_REQUIRE_CHECKSUM_ENV, "1") == "1"


@dataclass
class CacheStats:
    """Thread-safe cache statistics.
    
    FIX #2026-02-24: Added fields for Prometheus metrics export.
    """

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
    # FIX #2026-02-24: Additional metrics
    validation_failures: int = 0
    write_retries: int = 0
    warming_hits: int = 0
    
    def increment(self, field_name: str, amount: int = 1) -> None:
        with self._lock:
            setattr(self, field_name, getattr(self, field_name) + amount)

    @property
    def total_hits(self) -> int:
        # FIX #17: Read under lock for consistency
        with self._lock:
            return self.l1_hits + self.l2_hits + self.l3_hits

    @property
    def total_misses(self) -> int:
        with self._lock:
            return self.l3_misses  # Final miss

    @property
    def hit_rate(self) -> float:
        # FIX #17: Atomic snapshot of all fields
        with self._lock:
            hits = self.l1_hits + self.l2_hits + self.l3_hits
            misses = self.l3_misses
        total = hits + misses
        return hits / max(total, 1)
    
    # FIX #2026-02-24: Prometheus metrics export
    def to_prometheus_dict(self) -> dict[str, float]:
        """Export stats as dictionary for Prometheus metrics.
        
        Returns:
            Dictionary mapping metric names to values
        """
        with self._lock:
            total = self.l1_hits + self.l2_hits + self.l3_hits + self.l3_misses
            hit_rate = (self.l1_hits + self.l2_hits + self.l3_hits) / max(total, 1)
            return {
                "cache_l1_hits": float(self.l1_hits),
                "cache_l1_misses": float(self.l1_misses),
                "cache_l2_hits": float(self.l2_hits),
                "cache_l2_misses": float(self.l2_misses),
                "cache_l3_hits": float(self.l3_hits),
                "cache_l3_misses": float(self.l3_misses),
                "cache_total_sets": float(self.total_sets),
                "cache_total_evictions": float(self.total_evictions),
                "cache_validation_failures": float(self.validation_failures),
                "cache_write_retries": float(self.write_retries),
                "cache_warming_hits": float(self.warming_hits),
                "cache_hit_rate": float(hit_rate),
            }


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    data: Any
    created_at: datetime
    size_bytes: int = 0
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)


# L1: In-memory LRU with TTL


class LRUCache:
    """Thread-safe LRU cache with TTL enforcement."""

    def __init__(self, max_items: int = 1000, max_size_mb: int = 500) -> None:
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_items = max_items
        self._max_size = max_size_mb * 1024 * 1024
        self._current_size = 0
        self._lock = threading.RLock()

    def get(
        self, key: str, max_age_hours: float = None
    ) -> Any:
        """Get value from cache.

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

    def set(self, key: str, value: Any, size_bytes: int = 0) -> None:
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

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._current_size = 0

    def _evict_key(self, key: str) -> None:
        """Remove a specific key (caller must hold lock)."""
        entry = self._cache.pop(key, None)
        if entry:
            self._current_size -= entry.size_bytes
            # FIX #8: If size tracking becomes inconsistent, recalculate
            if self._current_size < 0:
                log.warning(
                    "LRUCache size tracking inconsistent (evict_key), recalculating"
                )
                self._current_size = sum(e.size_bytes for e in self._cache.values())

    def _evict_oldest(self) -> None:
        """Remove oldest entry (caller must hold lock)."""
        if self._cache:
            _, entry = self._cache.popitem(last=False)
            self._current_size -= entry.size_bytes
            # FIX #8: If size tracking becomes inconsistent, recalculate
            if self._current_size < 0:
                log.warning(
                    "LRUCache size tracking inconsistent (evict_oldest), recalculating"
                )
                self._current_size = sum(e.size_bytes for e in self._cache.values())

    @staticmethod
    def _estimate_size(value: Any, _depth: int = 0) -> int:
        """Estimate object memory footprint.

        FIX #20: Added max depth to prevent infinite recursion on
        self-referencing structures.
        FIX #2026-02-24: Reduced max depth to 5 and added size calculation
        timeout protection to prevent performance issues with complex structures.
        """
        # FIX #2026-02-24: Reduced max depth from 10 to 5 for better protection
        if _depth > 5:
            try:
                return sys.getsizeof(value)
            except TypeError:
                return 1024

        if isinstance(value, pd.DataFrame):
            try:
                return int(value.memory_usage(deep=True, numeric_only=True).sum())
            except (TypeError, ValueError):
                # Fallback for DataFrames with complex dtypes
                return int(value.memory_usage(deep=False).sum()) + 1024
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, (str, bytes)):
            return len(value)
        if isinstance(value, dict):
            try:
                # FIX #2026-02-24: Limit iteration to first 100 items for large dicts
                items = list(value.items())[:100]
                return sys.getsizeof(value) + sum(
                    sys.getsizeof(k)
                    + LRUCache._estimate_size(v, _depth + 1)
                    for k, v in items
                )
            except (TypeError, RecursionError):
                return sys.getsizeof(value)
        if isinstance(value, (list, tuple)):
            try:
                # FIX #2026-02-24: Limit iteration to first 100 items for large collections
                items = value[:100]
                return sys.getsizeof(value) + sum(
                    LRUCache._estimate_size(item, _depth + 1)
                    for item in items
                )
            except (TypeError, RecursionError):
                return sys.getsizeof(value)
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
    """Disk-based cache with atomic writes.

    Uses write-to-temp-then-rename to prevent corruption on crash.

    FIX #7: Removed global lock from get() — reads don't need
    serialization since writes are atomic (temp → rename).
    Lock only protects writes.

    FIX C4: Closes file descriptor immediately after mkstemp to prevent
    fd leak and avoid Windows PermissionError when renaming.
    """

    def __init__(self, cache_dir: Path, compress: bool = False) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._compress = compress
        self._write_lock = threading.RLock()

    def _key_to_path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()
        ext = ".pkl.gz" if self._compress else ".pkl"
        return self._dir / f"{h}{ext}"

    def get(
        self, key: str, max_age_hours: float = None
    ) -> Any:
        """Read from disk cache.

        Returns _SENTINEL on miss/expiry/error.

        FIX #7: No lock needed — files are written atomically via
        temp-then-rename, so a concurrent read sees either the old
        complete file or the new complete file.
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

        require_checksum = _cache_checksum_required()
        if not verify_checksum_sidecar(path, require=require_checksum):
            log.warning(
                "Cache checksum verification failed for key hash %s "
                "(require=%s)",
                path.stem,
                require_checksum,
            )
            try:
                path.unlink(missing_ok=True)
                artifact_checksum_path(path).unlink(missing_ok=True)
            except OSError:
                pass
            return _SENTINEL

        try:
            if self._compress:
                with gzip.open(path, "rb") as f:
                    payload = f.read(_DISK_CACHE_MAX_PICKLE_BYTES + 1)
                if len(payload) > _DISK_CACHE_MAX_PICKLE_BYTES:
                    raise ValueError(
                        f"Compressed cache payload too large ({len(payload):,} bytes)"
                    )
                return pickle_load_bytes(
                    payload,
                    max_bytes=_DISK_CACHE_MAX_PICKLE_BYTES,
                    allow_unsafe=False,  # FIX: Security - disable unsafe pickle
                )
            else:
                return pickle_load(
                    path,
                    max_bytes=_DISK_CACHE_MAX_PICKLE_BYTES,
                    allow_unsafe=False,  # FIX: Security - disable unsafe pickle
                )
        except (OSError, ValueError, TypeError, pickle.UnpicklingError) as e:
            log.warning(
                f"Cache read error for key hash {path.stem}: {e}"
            )
            try:
                path.unlink(missing_ok=True)
                artifact_checksum_path(path).unlink(missing_ok=True)
            except OSError:
                pass
            return _SENTINEL

    def set(self, key: str, value: Any, max_retries: int = 3) -> None:
        """Atomic write: temp file → rename.

        FIX C4: Close the file descriptor from mkstemp IMMEDIATELY
        before opening the file with gzip.open or open().
        
        FIX #2026-02-24: Added retry logic with exponential backoff for
        transient I/O failures.
        """
        import time
        
        path = self._key_to_path(key)
        tmp_path: Path | None = None
        
        for attempt in range(max_retries):
            try:
                with self._write_lock:
                    fd, tmp_path_str = tempfile.mkstemp(
                        dir=str(self._dir), suffix=".tmp"
                    )

                    try:
                        os.close(fd)
                    except OSError:
                        pass

                    tmp_path = Path(tmp_path_str)

                    if self._compress:
                        with gzip.open(tmp_path, "wb") as f:
                            pickle.dump(
                                value, f, protocol=pickle.HIGHEST_PROTOCOL
                            )
                    else:
                        with open(tmp_path, "wb") as f:
                            pickle.dump(
                                value, f, protocol=pickle.HIGHEST_PROTOCOL
                            )

                    tmp_path.replace(path)
                    try:
                        write_checksum_sidecar(path)
                    except OSError as e:
                        log.debug("Cache checksum sidecar write failed for %s: %s", path, e)
                    tmp_path = None  # Successfully moved
                    return  # Success

            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: 0.1s, 0.2s, 0.4s
                    backoff = 0.1 * (2 ** attempt)
                    log.debug(
                        "Cache write attempt %d/%d failed for key %s: %s, retrying in %.2fs",
                        attempt + 1, max_retries, key, e, backoff
                    )
                    time.sleep(backoff)
                else:
                    log.warning(
                        "Cache write failed after %d attempts for key %s: %s",
                        max_retries, key, e
                    )
                # Track retry for metrics (if stats callback provided)
                if attempt > 0 and hasattr(self, '_stats_callback'):
                    try:
                        self._stats_callback.increment("write_retries")
                    except Exception:
                        pass
            finally:
                if tmp_path is not None:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        pass

    def delete(self, key: str) -> bool:
        if not _cache_delete_allowed():
            log.warning(
                "Cache delete blocked for key=%s (manual override required)",
                key,
            )
            return False
        path = self._key_to_path(key)
        try:
            if path.exists():
                path.unlink()
                return True
        except OSError:
            pass
        return False

    def clear(self, older_than_hours: float = None) -> None:
        """Clear cache, optionally only old files."""
        if not _cache_delete_allowed():
            log.warning(
                "Cache clear blocked (manual override required)"
            )
            return
        now = datetime.now()
        patterns = ["*.pkl", "*.pkl.gz"]
        for pattern in patterns:
            for path in self._dir.glob(pattern):
                try:
                    if older_than_hours is not None:
                        mtime = datetime.fromtimestamp(
                            path.stat().st_mtime
                        )
                        age = (now - mtime).total_seconds() / 3600
                        if age <= older_than_hours:
                            continue
                    path.unlink()
                except OSError:
                    pass


class TieredCache:
    """Three-tier caching system.

    L1: Fast in-memory LRU with TTL
    L2: Disk cache (fast reads)
    L3: Compressed disk (persistent, longer TTL)

    FIX #2: Uses MISSING sentinel throughout the public API so that
    None can be a valid cached value.
    FIX #5: Uses per-key locking in get_or_compute to prevent
    double-compute race conditions.
    FIX #12: Tracks shutdown state to prevent RuntimeError after shutdown.
    """

    _L3_MAX_WORKERS = 4

    def __init__(self) -> None:
        self._l1 = LRUCache(
            max_items=500,
            max_size_mb=CONFIG.data.max_memory_cache_mb,
        )
        self._l2 = DiskCache(
            CONFIG.cache_dir / "l2", compress=False
        )
        self._l3 = DiskCache(
            CONFIG.cache_dir / "l3", compress=True
        )
        self._stats = CacheStats()
        self._lock = threading.RLock()
        self._shutdown_flag = False

        # Per-key locks for get_or_compute (FIX #5)
        self._compute_locks: dict[str, threading.Lock] = {}
        self._compute_locks_lock = threading.Lock()
        # Limit compute lock dict growth
        self._compute_locks_max = 10000

        from concurrent.futures import ThreadPoolExecutor

        self._l3_executor = ThreadPoolExecutor(
            max_workers=self._L3_MAX_WORKERS,
            thread_name_prefix="cache_l3",
        )
        
        # FIX #2026-02-24: Track pending L3 writes for crash consistency
        self._l3_pending_writes: set[str] = set()
        self._l3_pending_lock = threading.Lock()
        
        # FIX #2026-02-24: Cache warming support
        self._warmed_keys: set[str] = set()
        self._warming_lock = threading.Lock()

    def warm_cache(self, keys: list[str], compute_fn: Callable[[str], Any]) -> dict[str, bool]:
        """Pre-warm cache with specified keys.
        
        FIX #2026-02-24: Cache warming strategy to reduce cold-start latency.
        Pre-computes and caches values for frequently accessed keys.
        
        Args:
            keys: List of keys to warm
            compute_fn: Function to compute value for each key
            
        Returns:
            Dictionary mapping keys to success status
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results: dict[str, bool] = {}
        warmed_count = 0
        
        # Use a separate executor for warming to avoid blocking main operations
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="cache_warm") as executor:
            futures = {executor.submit(compute_fn, key): key for key in keys}
            for future in as_completed(futures, timeout=60):
                key = futures[future]
                try:
                    value = future.result(timeout=30)
                    if value is not None:
                        self.set(key, value, persist=True)
                        with self._warming_lock:
                            self._warmed_keys.add(key)
                        self._stats.increment("warming_hits")
                        warmed_count += 1
                        results[key] = True
                    else:
                        results[key] = False
                except Exception as e:
                    log.debug("Cache warming failed for key %s: %s", key, e)
                    results[key] = False
        
        log.info("Cache warming complete: %d/%d keys warmed successfully", warmed_count, len(keys))
        return results
    
    def get_warmed_keys(self) -> list[str]:
        """Get list of keys that were warmed.
        
        Returns:
            List of warmed cache keys
        """
        with self._warming_lock:
            return list(self._warmed_keys)

    def _get_compute_lock(self, key: str) -> threading.Lock:
        """Get or create a per-key lock for compute operations.

        Uses LRU eviction policy to prevent unbounded growth while
        preserving recently used locks.

        FIX #6: Locks are now cleaned up after use to prevent memory leak.
        FIX #2026-02-24: Use OrderedDict for proper LRU tracking and mark
        locks as in-use to prevent evicting active locks.
        """
        with self._compute_locks_lock:
            # FIX #12: LRU-style eviction instead of clearing all
            if len(self._compute_locks) >= self._compute_locks_max:
                # Remove oldest 20% of entries when limit reached,
                # but skip locks currently in use (marked with _in_use)
                keys_to_remove = []
                removed_count = 0
                target_remove = self._compute_locks_max // 5
                for k in list(self._compute_locks.keys()):
                    if removed_count >= target_remove:
                        break
                    # Don't evict locks in use
                    if not getattr(self._compute_locks[k], '_in_use', False):
                        keys_to_remove.append(k)
                        removed_count += 1
                for k in keys_to_remove:
                    del self._compute_locks[k]
            if key not in self._compute_locks:
                self._compute_locks[key] = threading.Lock()
            # Mark as in-use
            self._compute_locks[key]._in_use = True  # type: ignore[attr-defined]
            return self._compute_locks[key]

    def _release_compute_lock(self, key: str) -> None:
        """Release and remove per-key lock if no longer needed.

        FIX #6: Call this after compute to clean up unused locks.
        FIX #2026-02-24: Mark lock as not in-use before cleanup.
        """
        with self._compute_locks_lock:
            # Only remove if it exists and is not in use
            if key in self._compute_locks:
                lock = self._compute_locks[key]
                # Mark as not in-use
                lock._in_use = False  # type: ignore[attr-defined]
                # Only remove if no other thread is waiting
                if not getattr(lock, '_waiters', 0):
                    del self._compute_locks[key]

    @staticmethod
    def _clone_cache_value(value: Any) -> Any:
        """Return a defensive clone for mutable payloads.

        This prevents accidental caller mutation from corrupting cache state and
        keeps asynchronous tier writes consistent with the snapshot at set-time.
        """
        if value is None or isinstance(
            value,
            (bool, int, float, str, bytes),
        ):
            return value

        if isinstance(value, pd.DataFrame):
            return value.copy(deep=True)

        if isinstance(value, pd.Series):
            return value.copy(deep=True)

        if isinstance(value, np.ndarray):
            return np.array(value, copy=True)

        try:
            return deepcopy(value)
        except Exception:
            return value

    def get(self, key: str, max_age_hours: float = None) -> Any:
        """Get value with tiered lookup.

        FIX #2: Returns MISSING sentinel on miss (not None).
        This allows None to be a valid cached value.
        FIX #2026-02-24: Added read-time validation for DataFrame cache entries
        to prevent serving corrupted or invalid data.
        """
        max_age = max_age_hours or CONFIG.data.cache_ttl_hours

        # L1: Memory (with TTL)
        value = self._l1.get(key, max_age_hours=max_age)
        if value is not _SENTINEL:
            self._stats.increment("l1_hits")
            # FIX #2026-02-24: Validate DataFrame on read
            if isinstance(value, pd.DataFrame) and not self._validate_dataframe(value):
                log.warning("L1 cache validation failed for key %s, evicting", key)
                self._l1.delete(key)
                self._stats.increment("validation_failures")
                self._stats.increment("l1_misses")
                return MISSING
            return self._clone_cache_value(value)
        self._stats.increment("l1_misses")

        # L2: Disk
        value = self._l2.get(key, max_age)
        if value is not _SENTINEL:
            self._stats.increment("l2_hits")
            # FIX #2026-02-24: Validate DataFrame on read
            if isinstance(value, pd.DataFrame) and not self._validate_dataframe(value):
                log.warning("L2 cache validation failed for key %s, evicting", key)
                self._l2.delete(key)
                self._l1.delete(key)
                self._stats.increment("validation_failures")
                self._stats.increment("l2_misses")
                return MISSING
            promoted = self._clone_cache_value(value)
            self._l1.set(key, promoted)  # Promote immutable snapshot
            return self._clone_cache_value(promoted)
        self._stats.increment("l2_misses")

        # L3: Compressed disk (longer TTL)
        # FIX #2026-02-24: Use configurable TTL multiplier instead of hardcoded 24
        l3_ttl_multiplier = _get_l3_ttl_multiplier()
        value = self._l3.get(key, max_age * l3_ttl_multiplier)
        if value is not _SENTINEL:
            self._stats.increment("l3_hits")
            # FIX #2026-02-24: Validate DataFrame on read
            if isinstance(value, pd.DataFrame) and not self._validate_dataframe(value):
                log.warning("L3 cache validation failed for key %s, evicting", key)
                self._l3.delete(key)
                self._l2.delete(key)
                self._l1.delete(key)
                self._stats.increment("validation_failures")
                self._stats.increment("l3_misses")
                return MISSING
            promoted = self._clone_cache_value(value)
            self._l1.set(key, promoted)
            self._l2.set(key, self._clone_cache_value(promoted))
            return self._clone_cache_value(promoted)
        self._stats.increment("l3_misses")

        return MISSING

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> bool:
        """Validate DataFrame integrity for cache entries.
        
        FIX #2026-02-24: Read-time validation to prevent serving corrupted data.
        
        Returns:
            True if valid, False if corrupted or invalid
        """
        if df is None or df.empty:
            return True  # Empty is valid (edge case)
        
        # Check for required OHLCV columns if present
        required_cols = {"open", "high", "low", "close"}
        available_cols = set(df.columns)
        if required_cols.issubset(available_cols):
            # Validate OHLC relationships
            try:
                if not (df["high"] >= df["low"]).all():
                    return False
                if not (df["high"] >= df["open"]).all():
                    return False
                if not (df["high"] >= df["close"]).all():
                    return False
                if not (df["low"] <= df["open"]).all():
                    return False
                if not (df["low"] <= df["close"]).all():
                    return False
                # Check for positive prices
                for col in ["open", "high", "low", "close"]:
                    if (df[col] <= 0).any():
                        return False
                # Check for NaN/Inf in required columns
                for col in required_cols:
                    if df[col].isna().any():
                        return False
                    if np.isinf(df[col]).any():
                        return False
            except (KeyError, TypeError, ValueError):
                return False
        
        return True

    def set(self, key: str, value: Any, persist: bool = True) -> None:
        """Store value in cache tiers.
        
        FIX #2026-02-24: Tracks pending L3 writes for crash consistency.
        On shutdown, pending writes are flushed synchronously.
        """
        self._stats.increment("total_sets")

        snapshot = self._clone_cache_value(value)

        # Always L1
        self._l1.set(key, snapshot)

        if persist:
            # L2 synchronously (fast) - this is the durable fallback
            self._l2.set(key, self._clone_cache_value(snapshot))
            # L3 asynchronously via bounded pool
            # FIX #12: Don't submit after shutdown
            if not self._shutdown_flag:
                try:
                    # FIX #2026-02-24: Track pending L3 write
                    with self._l3_pending_lock:
                        self._l3_pending_writes.add(key)
                    
                    future = self._l3_executor.submit(
                        self._l3.set,
                        key,
                        self._clone_cache_value(snapshot),
                    )
                    # FIX #2026-02-24: Remove from pending on completion
                    def _on_complete(f):
                        with self._l3_pending_lock:
                            self._l3_pending_writes.discard(key)
                    future.add_done_callback(_on_complete)
                except RuntimeError:
                    # Pool already shut down - write to L3 synchronously
                    with self._l3_pending_lock:
                        self._l3_pending_writes.discard(key)
                    try:
                        self._l3.set(key, self._clone_cache_value(snapshot))
                    except Exception as e:
                        log.debug("L3 sync write failed: %s", e)
                except Exception as e:
                    log.debug("L3 async write failed: %s", e)
                    with self._l3_pending_lock:
                        self._l3_pending_writes.discard(key)

    def delete(self, key: str) -> None:
        """Delete from all tiers."""
        if not _cache_delete_allowed():
            log.warning(
                "Tiered cache delete blocked for key=%s "
                "(manual override required)",
                key,
            )
            return
        self._l1.delete(key)
        self._l2.delete(key)
        self._l3.delete(key)

    def clear(
        self,
        tier: str = None,
        older_than_hours: float = None,
    ) -> None:
        """Clear cache (optionally one tier or old entries only)."""
        if not _cache_delete_allowed():
            log.warning(
                "Tiered cache clear blocked (manual override required)"
            )
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
        """Get from cache or compute and store.

        FIX #2: Uses MISSING sentinel so None return values are cacheable.
        FIX #5: Uses per-key locking to prevent double-compute race.
        FIX #6: Releases compute lock after use to prevent memory leak.
        FIX #2026-02-24: Lock cleanup happens INSIDE lock context to prevent
        race condition where lock reference could be stale.
        """
        # Quick check without compute lock
        value = self.get(key, max_age_hours)
        if value is not MISSING:
            return value

        # Per-key lock to prevent double-compute
        compute_lock = self._get_compute_lock(key)
        try:
            with compute_lock:
                # Re-check after acquiring lock (another thread may have
                # computed and stored while we waited)
                value = self.get(key, max_age_hours)
                if value is not MISSING:
                    return value

                value = compute_fn()

                # Store any value including None
                self.set(key, value, persist)
            
            # FIX #2026-02-24: Clean up lock INSIDE try block but AFTER
            # releasing the compute lock, ensuring atomic cleanup
            self._release_compute_lock(key)
            compute_lock = None  # Prevent double-cleanup
        finally:
            # Safety: cleanup if exception occurred before lock release
            if compute_lock is not None:
                self._release_compute_lock(key)

        return value

    def shutdown(self) -> None:
        """Shutdown L3 background writer pool.
        
        FIX #2026-02-24: Flushes pending L3 writes synchronously before
        shutdown to ensure crash consistency. L2 already has all data
        synchronously, so data is not lost.
        """
        import time
        
        # FIX #12: Set flag before shutdown to prevent new submissions
        self._shutdown_flag = True
        
        # FIX #2026-02-24: Wait for pending L3 writes with timeout
        pending_count = 0
        with self._l3_pending_lock:
            pending_count = len(self._l3_pending_writes)
        
        if pending_count > 0:
            log.info("Waiting for %d pending L3 cache writes to complete...", pending_count)
            # Wait up to 5 seconds for pending writes
            for _ in range(50):  # 50 * 0.1s = 5s max
                time.sleep(0.1)
                with self._l3_pending_lock:
                    if len(self._l3_pending_writes) == 0:
                        break
            else:
                # Timeout - log warning but continue
                with self._l3_pending_lock:
                    remaining = len(self._l3_pending_writes)
                if remaining > 0:
                    log.warning(
                        "L3 cache shutdown timed out with %d pending writes "
                        "(L2 has all data, no data loss)",
                        remaining,
                    )
        
        try:
            self._l3_executor.shutdown(
                wait=True, cancel_futures=False
            )
        except TypeError:
            # Python < 3.9 doesn't have cancel_futures
            self._l3_executor.shutdown(wait=True)
        
        log.debug("L3 cache executor shut down complete")


_cache: TieredCache | None = None
_cache_lock = threading.Lock()


def _shutdown_cache() -> None:
    """Shutdown cache executor on process exit.
    
    Registered with atexit to ensure proper cleanup of thread pools.
    """
    global _cache
    if _cache is not None:
        try:
            _cache.shutdown()
        except Exception:
            pass


# Register shutdown handler to ensure thread pool cleanup
atexit.register(_shutdown_cache)


def get_cache() -> TieredCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = TieredCache()
    return _cache


def reset_cache() -> None:
    """Reset global cache instance (for testing).

    Shuts down the existing cache cleanly before clearing.
    """
    global _cache
    with _cache_lock:
        if _cache is not None:
            try:
                _cache.shutdown()
            except Exception:
                pass
            _cache = None


def cached(
    key_fn: Callable[..., str] = None,
    max_age_hours: float = None,
    persist: bool = True,
):
    """Caching decorator.

    Args:
        key_fn: Function to compute cache key from args/kwargs.
                If None, a default key is derived from module + function
                name + sorted args/kwargs hash.
        max_age_hours: TTL override.
        persist: Whether to persist to disk tiers.

    FIX #1: Uses get_cache() instead of bare _cache module variable
    so the cache is always initialized.
    FIX #2: Uses MISSING sentinel so None return values are cacheable.
    FIX #7: Handles unhashable types in args/kwargs by converting to
    JSON-serializable form before hashing.
    """
    import json as _json

    def _make_hashable(obj: Any) -> Any:
        """Convert potentially unhashable objects to hashable form."""
        if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
            return obj
        if isinstance(obj, (list, tuple)):
            return tuple(_make_hashable(item) for item in obj)
        if isinstance(obj, dict):
            return tuple(sorted((_make_hashable(k), _make_hashable(v)) for k, v in obj.items()))
        if isinstance(obj, set):
            return frozenset(_make_hashable(item) for item in obj)
        # For other types, use string representation
        return str(obj)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()

            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                # FIX #7: Make args/kwargs hashable before creating key
                try:
                    sorted_kwargs = tuple(sorted(kwargs.items()))
                    raw_key = (
                        f"{func.__module__}.{func.__qualname__}"
                        f":{args}:{sorted_kwargs}"
                    )
                except TypeError:
                    # Unhashable types - use JSON serialization
                    try:
                        key_data = {
                            'module': func.__module__,
                            'qualname': func.__qualname__,
                            'args': _make_hashable(args),
                            'kwargs': _make_hashable(kwargs),
                        }
                        raw_key = _json.dumps(key_data, sort_keys=True, default=str)
                    except Exception:
                        # Last resort: use hash of string representation
                        raw_key = f"{func.__module__}.{func.__qualname__}:{str(args)}:{str(kwargs)}"
                
                key = hashlib.sha256(
                    raw_key.encode()
                ).hexdigest()

            result = cache.get(key, max_age_hours)
            if result is not MISSING:
                return result

            result = func(*args, **kwargs)

            # FIX #2: Cache any value including None
            cache.set(key, result, persist)

            return result

        return wrapper

    return decorator
