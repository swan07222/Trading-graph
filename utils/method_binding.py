from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)


def bind_methods(
    target_cls: type[Any],
    bindings: Mapping[str, Any],
    *,
    static_methods: set[str] | None = None,
    context: str = "",
) -> None:
    """Bind callables to a class with minimal validation.

    Args:
        target_cls: Class receiving bound attributes.
        bindings: Mapping of attribute name to callable.
        static_methods: Method names that should be bound as ``staticmethod``.
        context: Optional label used in debug/error messages.
    """
    if not isinstance(bindings, Mapping):
        raise TypeError("bindings must be a mapping")

    static_names = set(static_methods or set())
    label = str(context or f"{target_cls.__module__}.{target_cls.__name__}")

    for name, fn in bindings.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}: binding names must be non-empty strings")
        if not callable(fn):
            raise TypeError(
                f"{label}: binding for {name!r} must be callable, got {type(fn).__name__}"
            )

        existing = getattr(target_cls, name, None)
        if existing is not None and existing is not fn:
            log.debug("%s: overriding existing binding for %s", label, name)

        bound = staticmethod(fn) if name in static_names else fn
        setattr(target_cls, name, bound)
