from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeAlias

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


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY_IMMEDIATE = "retry_immediate"
    RETRY_EXPONENTIAL = "retry_exponential"
    RETRY_LINEAR = "retry_linear"
    FALLBACK = "fallback"
    ROLLBACK = "rollback"
    SKIP = "skip"
    ABORT = "abort"


@dataclass
class RecoveryContext:
    """Context information for recovery operations.
    
    Attributes:
        operation: Name of the operation being recovered
        error: The exception that triggered recovery
        attempt: Current attempt number (1-indexed)
        max_attempts: Maximum number of recovery attempts
        strategy: Recovery strategy to use
        metadata: Additional context-specific metadata
        start_time: When the operation started
        recovery_start_time: When recovery started
    """
    operation: str
    error: BaseException | None = None
    attempt: int = 1
    max_attempts: int = 3
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY_EXPONENTIAL
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    recovery_start_time: float = field(default_factory=time.time)
    
    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time since operation start."""
        return time.time() - self.start_time
    
    @property
    def recovery_elapsed_seconds(self) -> float:
        """Time spent in recovery."""
        return time.time() - self.recovery_start_time
    
    @property
    def can_retry(self) -> bool:
        """Check if more retry attempts are available."""
        return self.attempt < self.max_attempts
    
    @property
    def should_abort(self) -> bool:
        """Check if recovery should be aborted."""
        return self.strategy == RecoveryStrategy.ABORT
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "operation": self.operation,
            "error_type": type(self.error).__name__ if self.error else None,
            "error_message": str(self.error) if self.error else None,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "strategy": self.strategy.value,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "recovery_elapsed_seconds": round(self.recovery_elapsed_seconds, 3),
            "metadata": self.metadata,
        }


@dataclass
class RecoveryResult:
    """Result of a recovery operation.
    
    Attributes:
        success: Whether recovery was successful
        context: The recovery context
        result: The recovered result (if any)
        fallback_used: Whether a fallback was used
        message: Human-readable status message
    """
    success: bool
    context: RecoveryContext
    result: Any = None
    fallback_used: bool = False
    message: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "success": self.success,
            "operation": self.context.operation,
            "attempts_made": self.context.attempt,
            "fallback_used": self.fallback_used,
            "message": self.message,
            "total_elapsed_seconds": round(self.context.elapsed_seconds, 3),
        }


def retry_with_recovery(
    func: Callable,
    operation: str,
    max_attempts: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 10.0,
    exponential: bool = True,
    recoverable_exceptions: RecoverableExceptions = COMMON_RECOVERABLE_EXCEPTIONS,
    on_retry: Callable[[RecoveryContext], None] | None = None,
    on_success: Callable[[RecoveryResult], None] | None = None,
    on_failure: Callable[[RecoveryResult], None] | None = None,
    fallback: Callable | None = None,
) -> RecoveryResult:
    """Execute a function with automatic retry and recovery.
    
    Args:
        func: Function to execute
        operation: Operation name for logging
        max_attempts: Maximum number of attempts
        base_delay: Base delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential: Use exponential backoff if True
        recoverable_exceptions: Exceptions that trigger retry
        on_retry: Callback invoked before each retry
        on_success: Callback invoked on success
        on_failure: Callback invoked on failure
        fallback: Fallback function if all retries fail
    
    Returns:
        RecoveryResult with success status and result
    
    Example:
        result = retry_with_recovery(
            fetch_data,
            operation="fetch_market_data",
            max_attempts=5,
            on_retry=lambda ctx: log.warning(f"Retry {ctx.attempt}"),
        )
        if result.success:
            process(result.result)
    """
    context = RecoveryContext(
        operation=operation,
        max_attempts=max_attempts,
    )
    
    last_error: BaseException | None = None
    
    for attempt in range(1, max_attempts + 1):
        context.attempt = attempt
        
        try:
            result = func()
            
            recovery_result = RecoveryResult(
                success=True,
                context=context,
                result=result,
                message=f"Operation '{operation}' succeeded on attempt {attempt}",
            )
            
            if on_success:
                on_success(recovery_result)
            
            return recovery_result
            
        except recoverable_exceptions as e:
            last_error = e
            context.error = e
            
            if context.can_retry:
                if on_retry:
                    on_retry(context)
                
                # Calculate delay
                if exponential:
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                else:
                    delay = min(base_delay * attempt, max_delay)
                
                time.sleep(delay)
            else:
                break
    
    # All retries exhausted
    recovery_result = RecoveryResult(
        success=False,
        context=context,
        message=f"Operation '{operation}' failed after {max_attempts} attempts: {last_error}",
    )
    
    # Try fallback if available
    if fallback:
        try:
            fallback_result = fallback()
            recovery_result.success = True
            recovery_result.result = fallback_result
            recovery_result.fallback_used = True
            recovery_result.message = f"Operation '{operation}' succeeded using fallback"
            
            if on_success:
                on_success(recovery_result)
            
            return recovery_result
        except Exception as fallback_error:
            recovery_result.message += f"; fallback also failed: {fallback_error}"
    
    if on_failure:
        on_failure(recovery_result)
    
    return recovery_result
