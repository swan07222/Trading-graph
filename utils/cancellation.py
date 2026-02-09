"""
Cooperative Cancellation Support

Provides thread-safe cancellation mechanism for long-running operations.
Use instead of QThread.terminate() which is unsafe.
"""
import threading
from typing import Callable, Optional
from contextlib import contextmanager


class CancellationToken:
    """
    Thread-safe cancellation token.
    
    Usage:
        token = CancellationToken()
        
        # In worker thread:
        for epoch in range(epochs):
            if token.is_cancelled:
                break
            # ... do work ...
        
        # From main thread:
        token.cancel()
    """
    
    def __init__(self):
        self._cancelled = threading.Event()
        self._callbacks: list = []
        self._lock = threading.Lock()
    
    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested"""
        return self._cancelled.is_set()
    
    def cancel(self):
        """Request cancellation"""
        self._cancelled.set()
        
        with self._lock:
            for callback in self._callbacks:
                try:
                    callback()
                except:
                    pass
    
    def on_cancel(self, callback: Callable):
        """Register a callback to be called on cancellation"""
        with self._lock:
            self._callbacks.append(callback)
    
    def raise_if_cancelled(self):
        """Raise exception if cancelled"""
        if self.is_cancelled:
            raise CancelledException("Operation was cancelled")
    
    def wait(self, timeout: float = None) -> bool:
        """Wait for cancellation."""
        return self._cancelled.wait(timeout)
    
    def __call__(self) -> bool:
        """Allow using token as callable stop_flag"""
        return self.is_cancelled

    def __bool__(self) -> bool:
        """
        Always True so that 'if token:' doesn't skip checks.
        Use token.is_cancelled to check cancellation state.
        
        FIX: Previously returned is_cancelled which meant
        bool(fresh_token) was False, breaking 'if stop_flag:' patterns.
        """
        return True
    
    def reset(self):
        """Reset the token for reuse"""
        self._cancelled.clear()


class CancelledException(Exception):
    """Raised when an operation is cancelled"""
    pass


@contextmanager
def cancellable_operation(token: Optional[CancellationToken] = None):
    """
    Context manager for cancellable operations.
    
    Usage:
        with cancellable_operation(token) as check_cancelled:
            for i in range(1000):
                check_cancelled()  # Raises if cancelled
                # ... do work ...
    """
    if token is None:
        yield lambda: None
    else:
        def check():
            token.raise_if_cancelled()
        yield check