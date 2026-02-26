from __future__ import annotations

import json
from typing import TypeAlias

RecoverableExceptions: TypeAlias = tuple[type[BaseException], ...]

# Backwards-compatible exception groups used throughout the codebase.
# Recovery orchestration (retry/fallback managers) has been removed.
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
    ConnectionError,
    ConnectionResetError,
    ConnectionAbortedError,
    ConnectionRefusedError,
    BrokenPipeError,
    InterruptedError,
    RecursionError,
    StopIteration,
    GeneratorExit,
    Warning,
    UserWarning,
    ResourceWarning,
)

JSON_RECOVERABLE_EXCEPTIONS: RecoverableExceptions = (
    *COMMON_RECOVERABLE_EXCEPTIONS,
    json.JSONDecodeError,
)

NETWORK_RECOVERABLE_EXCEPTIONS: RecoverableExceptions = (
    ConnectionError,
    ConnectionResetError,
    ConnectionAbortedError,
    ConnectionRefusedError,
    BrokenPipeError,
    TimeoutError,
    OSError,
    RuntimeError,
)

DISK_IO_RECOVERABLE_EXCEPTIONS: RecoverableExceptions = (
    OSError,
    IOError,
    EOFError,
    RuntimeError,
    TypeError,
    ValueError,
)

MODEL_LOAD_RECOVERABLE_EXCEPTIONS: RecoverableExceptions = (
    *NETWORK_RECOVERABLE_EXCEPTIONS,
    *DISK_IO_RECOVERABLE_EXCEPTIONS,
    ImportError,
    AttributeError,
    KeyError,
    IndexError,
    TypeError,
    ValueError,
    RuntimeError,
)

__all__ = [
    "RecoverableExceptions",
    "COMMON_RECOVERABLE_EXCEPTIONS",
    "JSON_RECOVERABLE_EXCEPTIONS",
    "NETWORK_RECOVERABLE_EXCEPTIONS",
    "DISK_IO_RECOVERABLE_EXCEPTIONS",
    "MODEL_LOAD_RECOVERABLE_EXCEPTIONS",
]
