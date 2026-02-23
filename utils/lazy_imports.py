"""Lazy import utilities for deferred module loading.

This module provides centralized lazy import helpers to replace
duplicated implementations across the codebase and handle circular imports.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from importlib import import_module
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def lazy_get(module: str, name: str) -> Any:
    """Lazily import and return an attribute from a module.

    Args:
        module: Module name to import from.
        name: Attribute name to retrieve.

    Returns:
        The requested attribute.
    """
    return getattr(import_module(module), name)


class LazyImport(Generic[T]):
    """Lazy import descriptor for module-level attributes.

    Usage:
        _oms = LazyImport("trading.oms", "get_oms")

        def some_function():
            oms = _oms()  # Import happens on first call
    """

    def __init__(self, module: str, name: str) -> None:
        self._module_name = module
        self._attr_name = name
        self._module: Any = None

    def _import(self) -> Any:
        """Import and cache the module."""
        if self._module is None:
            self._module = import_module(self._module_name)
        return getattr(self._module, self._attr_name)

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Call the imported callable."""
        return self._import()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Get attribute from the imported module."""
        return getattr(self._import(), name)


class CachedLazyImport(Generic[T]):
    """Lazy import with TTL-based caching for expensive lookups.

    Usage:
        _get_oms = CachedLazyImport("trading.oms", "get_oms", ttl=5.0)

        def some_function():
            oms = _get_oms()  # Cached for TTL seconds
    """

    def __init__(self, module: str, name: str, ttl: float = 5.0) -> None:
        self._module_name = module
        self._attr_name = name
        self._ttl = ttl
        self._module: Any = None
        self._cache_val: T | None = None
        self._cache_ts: float = 0.0

    def _import(self) -> Callable[..., T]:
        """Import and return the callable."""
        if self._module is None:
            self._module = import_module(self._module_name)
        return getattr(self._module, self._attr_name)

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Call the imported callable with TTL caching."""
        now = time.time()
        if self._cache_val is not None and (now - self._cache_ts) < self._ttl:
            return self._cache_val

        result = self._import()(*args, **kwargs)
        self._cache_val = result
        self._cache_ts = now
        return result


def make_lazy_getter(module: str, name: str, *, cache: bool = False, ttl: float = 5.0) -> Callable[..., Any]:
    """Create a lazy getter function.

    Args:
        module: Module name.
        name: Attribute/function name.
        cache: Whether to cache results.
        ttl: Cache TTL in seconds (only used if cache=True).

    Returns:
        A callable that performs lazy import on first call.
    """
    if cache:
        lazy = CachedLazyImport(module, name, ttl=ttl)
    else:
        lazy = LazyImport(module, name)
    return lazy
