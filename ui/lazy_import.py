from __future__ import annotations

from importlib import import_module
from typing import Any


def lazy_get(module: str, name: str) -> Any:
    return getattr(import_module(module), name)
