from __future__ import annotations

import json
from typing import TypeAlias

RecoverableExceptions: TypeAlias = tuple[type[BaseException], ...]

COMMON_RECOVERABLE_EXCEPTIONS: RecoverableExceptions = (
    ArithmeticError,
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

JSON_RECOVERABLE_EXCEPTIONS: RecoverableExceptions = (
    *COMMON_RECOVERABLE_EXCEPTIONS,
    json.JSONDecodeError,
)
