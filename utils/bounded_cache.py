"""Bounded cache utilities for Trading Graph.

This module provides memory-safe cache implementations with LRU eviction
to prevent memory leaks in long-running trading sessions.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Generic, TypeVar

K = TypeVar('K')
V = TypeVar('V')


class BoundedCache(Generic[K, V]):
    """Thread-safe LRU cache with maximum size limit.
    
    Features:
    - Automatic LRU eviction when max_size is reached
    - Thread-safe with RLock
    - TTL support (optional)
    - Hit/miss statistics
    
    Example:
        >>> cache = BoundedCache[str, dict](max_size=500)
        >>> cache.set("600519", {"open": 100.0, "close": 102.0})
        >>> data = cache.get("600519")
        >>> print(cache.stats())
        {'hits': 1, 'misses': 0, 'size': 1}
    """
    
    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: float | None = None,
    ) -> None:
        """Initialize bounded cache.
        
        Args:
            max_size: Maximum number of items to keep in cache
            ttl_seconds: Optional time-to-live for cache entries
        """
        self._max_size = max(max_size, 1)
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._timestamps: dict[K, float] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: K) -> V | None:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Value if found and not expired, None otherwise
        """
        import time
        
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            # Check TTL if enabled
            if self._ttl_seconds is not None:
                age = time.time() - self._timestamps.get(key, 0)
                if age > self._ttl_seconds:
                    del self._cache[key]
                    if key in self._timestamps:
                        del self._timestamps[key]
                    self._misses += 1
                    return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
    
    def set(self, key: K, value: V) -> None:
        """Set value in cache with LRU eviction.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        import time
        
        with self._lock:
            if key in self._cache:
                # Update existing key
                self._cache.move_to_end(key)
                self._cache[key] = value
                if self._ttl_seconds is not None:
                    self._timestamps[key] = time.time()
            else:
                # Add new key
                if len(self._cache) >= self._max_size:
                    # Evict oldest
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    if oldest_key in self._timestamps:
                        del self._timestamps[oldest_key]
                
                self._cache[key] = value
                if self._ttl_seconds is not None:
                    self._timestamps[key] = time.time()
    
    def delete(self, key: K) -> bool:
        """Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._timestamps:
                    del self._timestamps[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def __len__(self) -> int:
        """Return current cache size."""
        with self._lock:
            return len(self._cache)
    
    def __contains__(self, key: K) -> bool:
        """Check if key is in cache (without updating LRU)."""
        with self._lock:
            return key in self._cache
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with hits, misses, size, max_size, hit_rate
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "max_size": self._max_size,
                "hit_rate": round(hit_rate, 2),
            }
    
    def keys(self) -> list[K]:
        """Return list of cached keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def values(self) -> list[V]:
        """Return list of cached values."""
        with self._lock:
            return list(self._cache.values())
    
    def items(self) -> list[tuple[K, V]]:
        """Return list of cached items."""
        with self._lock:
            return list(self._cache.items())


class BoundedDict(Generic[K, V]):
    """Simpler bounded dict without LRU behavior.
    
    Use this when you don't need LRU eviction but still want
    to bound memory usage. Oldest entries are evicted when full.
    
    Example:
        >>> cache = BoundedDict[str, float](max_size=1000)
        >>> cache.set("600519", 102.5)
    """
    
    def __init__(self, max_size: int = 1000) -> None:
        """Initialize bounded dict.
        
        Args:
            max_size: Maximum number of items
        """
        self._max_size = max(max_size, 1)
        self._dict: dict[K, V] = {}
        self._order: list[K] = []
        self._lock = threading.RLock()
    
    def get(self, key: K) -> V | None:
        """Get value without modifying order."""
        with self._lock:
            return self._dict.get(key)
    
    def set(self, key: K, value: V) -> None:
        """Set value with FIFO eviction."""
        with self._lock:
            if key not in self._dict:
                if len(self._dict) >= self._max_size:
                    # Remove oldest
                    oldest = self._order.pop(0)
                    del self._dict[oldest]
                self._order.append(key)
            self._dict[key] = value
    
    def delete(self, key: K) -> bool:
        """Delete key."""
        with self._lock:
            if key in self._dict:
                del self._dict[key]
                self._order.remove(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items."""
        with self._lock:
            self._dict.clear()
            self._order.clear()
    
    def __len__(self) -> int:
        """Return current size."""
        with self._lock:
            return len(self._dict)
    
    def __contains__(self, key: K) -> bool:
        """Check if key exists."""
        with self._lock:
            return key in self._dict


# Module-level cache registry for cleanup
_cache_registry: list[BoundedCache | BoundedDict] = []


def register_cache(cache: BoundedCache | BoundedDict) -> None:
    """Register cache for cleanup on shutdown.
    
    Args:
        cache: Cache instance to register
    """
    _cache_registry.append(cache)


def cleanup_all_caches() -> None:
    """Clear all registered caches."""
    for cache in _cache_registry:
        cache.clear()


def get_all_stats() -> dict[str, Any]:
    """Get statistics for all registered caches.
    
    Returns:
        Dictionary with stats for each cache
    """
    stats = {}
    for i, cache in enumerate(_cache_registry):
        if isinstance(cache, BoundedCache):
            stats[f"cache_{i}"] = cache.stats()
        else:
            stats[f"dict_{i}"] = {"size": len(cache), "max_size": cache._max_size}
    return stats
