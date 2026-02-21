from __future__ import annotations

import os
import threading
from collections.abc import Callable

from data.fetcher_registry import (
    FetcherRegistry,
    get_active_fetcher_registry,
    get_default_fetcher_registry,
    reset_default_fetcher_registry,
)

_FETCHER_SCOPE_THREAD = "thread"
_FETCHER_SCOPE_PROCESS = "process"


def resolve_fetcher_scope() -> str:
    raw = str(
        os.environ.get("TRADING_FETCHER_SCOPE", _FETCHER_SCOPE_THREAD)
    ).strip().lower()
    if raw in (_FETCHER_SCOPE_THREAD, _FETCHER_SCOPE_PROCESS):
        return raw
    return _FETCHER_SCOPE_THREAD


def resolve_fetcher_instance_key(*, instance: str | None = None) -> str:
    key = str(instance or "").strip()
    if key:
        return key
    if resolve_fetcher_scope() == _FETCHER_SCOPE_PROCESS:
        return "default"
    return f"thread:{threading.get_ident()}"


def get_fetcher_instance(
    *,
    create: Callable[[], object],
    disable_singletons: bool,
    instance: str | None = None,
    force_new: bool = False,
    registry: FetcherRegistry[object] | None = None,
) -> object:
    if force_new or disable_singletons:
        return create()
    target_registry = registry or get_active_fetcher_registry()
    key = resolve_fetcher_instance_key(instance=instance)
    return target_registry.get_or_create(key, create)


def reset_fetcher_instances(
    *,
    instance: str | None = None,
    registry: FetcherRegistry[object] | None = None,
) -> None:
    key = None
    if instance is not None:
        key = resolve_fetcher_instance_key(instance=instance)

    if registry is not None:
        registry.reset(instance=key)
        return

    default_registry = get_default_fetcher_registry()
    active_registry = get_active_fetcher_registry()

    if instance is None:
        reset_default_fetcher_registry()
        if active_registry is not default_registry:
            active_registry.reset(instance=None)
        return

    default_registry.reset(instance=key)
    if active_registry is not default_registry:
        active_registry.reset(instance=key)
