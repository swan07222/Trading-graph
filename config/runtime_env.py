from __future__ import annotations

import os
from typing import Final

_TRUTHY_ENV: Final[frozenset[str]] = frozenset(
    {"1", "true", "yes", "on"}
)


def env_flag(name: str, default: str = "0") -> bool:
    """Read boolean-like env flags using a shared truthy policy."""
    return str(os.environ.get(name, default)).strip().lower() in _TRUTHY_ENV


def env_text(name: str, default: str = "") -> str:
    """Read text env value with deterministic string normalization."""
    return str(os.environ.get(name, default) or "")
