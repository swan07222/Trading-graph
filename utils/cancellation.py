# utils/cancellation.py
from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager

from utils.logger import get_logger

log = get_logger(__name__)


class CancelledException(Exception):
    """Raised when a cancellable operation is cancelled."""
    pass

class CancellationToken:
    """Thread-safe cancellation token for cooperative cancellation.

    Usage:
        token = CancellationToken()

        # In worker thread:
        for epoch in range(num_epochs):
            token.raise_if_cancelled()
            # ... do work ...

        # From main thread:
        token.cancel()

    As a callable stop_flag:
        while not token():
            # ... do work ...
    """

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._callbacks: list[Callable] = []
        self._lock = threading.Lock()

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled.is_set()

    def cancel(self) -> None:
        """Request cancellation and invoke registered callbacks.

        Thread-safe. Idempotent - safe to call multiple times.
        """
        if self._cancelled.is_set():
            return

        self._cancelled.set()

        with self._lock:
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                log.warning("Cancellation callback failed: %s", e)

    def on_cancel(self, callback: Callable) -> Callable:
        """Register a callback to run when cancellation is requested.

        If already cancelled, the callback fires immediately.

        Args:
            callback: Zero-argument callable

        Returns:
            The callback (for use as a decorator)
        """
        with self._lock:
            self._callbacks.append(callback)

        # If already cancelled, fire immediately
        if self._cancelled.is_set():
            try:
                callback()
            except Exception as e:
                log.warning("Immediate cancellation callback failed: %s", e)

        return callback

    def remove_callback(self, callback: Callable) -> bool:
        """Remove a previously registered callback.

        Args:
            callback: The callback to remove

        Returns:
            True if callback was found and removed
        """
        with self._lock:
            try:
                self._callbacks.remove(callback)
                return True
            except ValueError:
                return False

    def raise_if_cancelled(self) -> None:
        """Raise CancelledException if cancellation was requested.

        Use this as a checkpoint in loops:
            for batch in data:
                token.raise_if_cancelled()
                process(batch)
        """
        if self._cancelled.is_set():
            raise CancelledException("Operation was cancelled")

    def wait(self, timeout: float | None = None) -> bool:
        """Block until cancellation is requested or timeout expires.

        Args:
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            True if cancelled, False if timeout expired
        """
        return self._cancelled.wait(timeout)

    def __call__(self) -> bool:
        """Check cancellation state. For use as a stop_flag callable.

        Returns:
            True if cancelled, False otherwise
        """
        return self._cancelled.is_set()

    def __bool__(self) -> bool:
        """Always returns True.

        This ensures `if token:` doesn't accidentally skip logic.
        Use `token.is_cancelled` or `token()` to check state.
        """
        return True

    def reset(self, clear_callbacks: bool = True) -> None:
        """Reset the token for reuse.

        Args:
            clear_callbacks: If True (default), remove all callbacks.
                           Set False to keep callbacks across resets.
        """
        self._cancelled.clear()
        if clear_callbacks:
            with self._lock:
                self._callbacks.clear()

@contextmanager
def cancellable_operation(token: CancellationToken | None = None):
    """Context manager for cancellable operations.

    Yields a check function that raises CancelledException if
    the token has been cancelled.

    Usage:
        with cancellable_operation(token) as check:
            for i in range(1000):
                check()  # raises CancelledException if cancelled
                do_work(i)

    Args:
        token: CancellationToken or None (no-op if None)
    """
    if token is None:
        # No-op check function
        yield lambda: None
    else:
        token.raise_if_cancelled()  # Check before starting
        yield token.raise_if_cancelled
