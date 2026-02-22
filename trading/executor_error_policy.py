from __future__ import annotations

import queue
from typing import Any

from config import CONFIG, TradingMode

# Programming errors should usually fail fast so defects are visible.
PROGRAMMING_EXCEPTIONS = (
    AttributeError,
    IndexError,
    KeyError,
    TypeError,
    ValueError,
)

# Broad compatibility bucket used by legacy call sites.
SOFT_FAIL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    OverflowError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    ZeroDivisionError,
    queue.Empty,
    queue.Full,
)


def is_live_mode(mode: Any) -> bool:
    raw = getattr(mode, "value", mode)
    return str(raw or "").strip().lower() == TradingMode.LIVE.value


def should_escalate_exception(mode: Any, exc: BaseException) -> bool:
    """
    Decide whether an exception should be re-raised instead of soft-failed.

    The default policy is fail-fast for bug-like exceptions; this can be relaxed
    via configuration for legacy environments.
    """
    sec_cfg = getattr(CONFIG, "security", None)
    strict_runtime = bool(
        getattr(sec_cfg, "strict_runtime_exception_policy", True)
    )
    if not strict_runtime:
        return False

    if isinstance(exc, PROGRAMMING_EXCEPTIONS):
        return True

    # In live mode, optionally escalate arithmetic corruption signals too.
    if is_live_mode(mode) and bool(
        getattr(sec_cfg, "escalate_runtime_errors_live", True)
    ):
        return isinstance(exc, (OverflowError, ZeroDivisionError))

    return False

