from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Generic, TypeVar

T = TypeVar("T")


class FetcherRegistry(Generic[T]):
    """Thread-safe keyed registry for fetcher-like shared instances."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._instances: dict[str, T] = {}

    def get_or_create(self, key: str, factory: Callable[[], T]) -> T:
        cached = self._instances.get(key)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._instances.get(key)
            if cached is None:
                cached = factory()
                self._instances[key] = cached
            return cached

    def reset(self, *, instance: str | None = None) -> None:
        with self._lock:
            if instance is None:
                self._instances.clear()
                return
            key = str(instance).strip()
            if not key:
                key = "default"
            self._instances.pop(key, None)


_default_registry: FetcherRegistry[object] = FetcherRegistry()
_active_registry_ctx: contextvars.ContextVar[FetcherRegistry[object]] = (
    contextvars.ContextVar(
        "trading_fetcher_registry",
        default=_default_registry,
    )
)


def get_active_fetcher_registry() -> FetcherRegistry[object]:
    return _active_registry_ctx.get()


def get_default_fetcher_registry() -> FetcherRegistry[object]:
    return _default_registry


def set_active_fetcher_registry(
    registry: FetcherRegistry[object],
) -> contextvars.Token[FetcherRegistry[object]]:
    return _active_registry_ctx.set(registry)


def reset_active_fetcher_registry(token: contextvars.Token[FetcherRegistry[object]]) -> None:
    _active_registry_ctx.reset(token)


def reset_default_fetcher_registry() -> None:
    _default_registry.reset()


@contextmanager
def use_fetcher_registry(
    registry: FetcherRegistry[object],
) -> Iterator[FetcherRegistry[object]]:
    token = set_active_fetcher_registry(registry)
    try:
        yield registry
    finally:
        reset_active_fetcher_registry(token)
