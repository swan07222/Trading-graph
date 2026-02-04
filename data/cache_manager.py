"""
High-Performance Cache Manager with LRU + Disk + Memory Tiers
"""
import pickle
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from collections import OrderedDict
from functools import lru_cache
import numpy as np
import pandas as pd

from config import CONFIG


class TieredCache:
    """
    Three-tier caching system:
    1. L1: In-memory LRU (fastest, limited size)
    2. L2: Memory-mapped files (fast, larger)
    3. L3: Compressed disk (persistent)
    """
    
    def __init__(self, max_memory_mb: int = 500):
        self._l1_cache: OrderedDict = OrderedDict()
        self._l1_max_items = 100
        self._l1_lock = threading.RLock()
        
        self._l2_dir = CONFIG.CACHE_DIR / "l2"
        self._l2_dir.mkdir(exist_ok=True)
        
        self._l3_dir = CONFIG.CACHE_DIR / "l3"
        self._l3_dir.mkdir(exist_ok=True)
        
        self._stats = {'l1_hits': 0, 'l2_hits': 0, 'l3_hits': 0, 'misses': 0}
    
    def _key_hash(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str, max_age_hours: float = 24) -> Optional[Any]:
        """Get from cache with tiered lookup"""
        h = self._key_hash(key)
        now = datetime.now()
        
        # L1: Memory
        with self._l1_lock:
            if h in self._l1_cache:
                data, timestamp = self._l1_cache[h]
                if (now - timestamp).total_seconds() / 3600 < max_age_hours:
                    self._l1_cache.move_to_end(h)
                    self._stats['l1_hits'] += 1
                    return data
                else:
                    del self._l1_cache[h]
        
        # L2: Memory-mapped
        l2_path = self._l2_dir / f"{h}.pkl"
        if l2_path.exists():
            try:
                mtime = datetime.fromtimestamp(l2_path.stat().st_mtime)
                if (now - mtime).total_seconds() / 3600 < max_age_hours:
                    with open(l2_path, 'rb') as f:
                        data = pickle.load(f)
                    self._promote_to_l1(h, data)
                    self._stats['l2_hits'] += 1
                    return data
            except:
                pass
        
        # L3: Compressed disk
        l3_path = self._l3_dir / f"{h}.pkl.gz"
        if l3_path.exists():
            try:
                import gzip
                mtime = datetime.fromtimestamp(l3_path.stat().st_mtime)
                if (now - mtime).total_seconds() / 3600 < max_age_hours:
                    with gzip.open(l3_path, 'rb') as f:
                        data = pickle.load(f)
                    self._promote_to_l2(h, data)
                    self._promote_to_l1(h, data)
                    self._stats['l3_hits'] += 1
                    return data
            except:
                pass
        
        self._stats['misses'] += 1
        return None
    
    def set(self, key: str, data: Any, persist: bool = True):
        """Store in cache"""
        h = self._key_hash(key)
        
        # Always L1
        self._promote_to_l1(h, data)
        
        # L2 for DataFrames > 1KB
        if persist:
            self._save_to_l2(h, data)
    
    def _promote_to_l1(self, h: str, data: Any):
        with self._l1_lock:
            self._l1_cache[h] = (data, datetime.now())
            self._l1_cache.move_to_end(h)
            
            while len(self._l1_cache) > self._l1_max_items:
                self._l1_cache.popitem(last=False)
    
    def _save_to_l2(self, h: str, data: Any):
        try:
            path = self._l2_dir / f"{h}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    
    def _promote_to_l2(self, h: str, data: Any):
        self._save_to_l2(h, data)
    
    def get_stats(self) -> Dict:
        total = sum(self._stats.values())
        return {
            **self._stats,
            'hit_rate': (total - self._stats['misses']) / max(total, 1)
        }
    
    def clear(self, older_than_hours: float = None):
        """Clear cache, optionally only old items"""
        with self._l1_lock:
            if older_than_hours is None:
                self._l1_cache.clear()
            else:
                now = datetime.now()
                expired = [
                    k for k, (_, ts) in self._l1_cache.items()
                    if (now - ts).total_seconds() / 3600 > older_than_hours
                ]
                for k in expired:
                    del self._l1_cache[k]


# Global cache instance
_cache = TieredCache()


def cached(key_fn: Callable = None, max_age_hours: float = 4):
    """Decorator for cached functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{func.__name__}:{args}:{kwargs}"
            
            result = _cache.get(key, max_age_hours)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            if result is not None:
                _cache.set(key, result)
            return result
        return wrapper
    return decorator