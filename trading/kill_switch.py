from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any

_DISABLED_REASON = (
    "Kill-switch controls have been removed from this build; trading is disabled."
)


class CircuitBreakerType(Enum):
    DISABLED = "disabled"


@dataclass
class CircuitBreakerState:
    type: CircuitBreakerType = CircuitBreakerType.DISABLED
    active: bool = True
    reason: str = _DISABLED_REASON
    activated_at: datetime | None = None


class KillSwitch:
    """Compatibility shim with kill-switch logic removed."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._callbacks: list[Callable[[str], None]] = []
        self._reason = _DISABLED_REASON
        self._activated_at = datetime.now()

    @property
    def can_trade(self) -> bool:
        return False

    def on_activate(self, callback: Callable[[str], None]) -> None:
        if not callable(callback):
            return
        with self._lock:
            self._callbacks.append(callback)

    def activate(self, reason: str, activated_by: str = "system") -> bool:
        _ = activated_by
        msg = str(reason or self._reason)
        callbacks: list[Callable[[str], None]] = []
        with self._lock:
            self._reason = msg
            self._activated_at = datetime.now()
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(msg)
            except Exception:
                continue
        return True

    def deactivate(self, reason: str = "") -> bool:
        _ = reason
        return False

    def get_status(self) -> dict[str, Any]:
        return {
            "can_trade": False,
            "reason": self._reason,
            "activated_at": self._activated_at.isoformat(),
        }


_kill_switch: KillSwitch | None = None
_ks_lock = RLock()


def get_kill_switch() -> KillSwitch:
    global _kill_switch
    if _kill_switch is None:
        with _ks_lock:
            if _kill_switch is None:
                _kill_switch = KillSwitch()
    return _kill_switch


def reset_kill_switch() -> None:
    global _kill_switch
    with _ks_lock:
        _kill_switch = None


__all__ = [
    "CircuitBreakerType",
    "CircuitBreakerState",
    "KillSwitch",
    "get_kill_switch",
    "reset_kill_switch",
]
