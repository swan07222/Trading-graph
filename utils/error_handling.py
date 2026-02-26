# utils/error_handling.py
"""
Enhanced Error Handling Framework

FIXES:
- Consistent error handling across application
- User-friendly error messages
- Automatic error recovery
- Error telemetry and alerting
"""

from __future__ import annotations

import functools
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

from utils.logger import get_logger

log = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK = "network"
    DATABASE = "database"
    MODEL = "model"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    INTERNAL = "internal"
    EXTERNAL = "external"


@dataclass
class ErrorContext:
    """Error context for debugging."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    user_message: str
    module: str
    function: str
    line_number: int
    stack_trace: str
    original_exception: Optional[Exception] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "user_message": self.user_message,
            "module": self.module,
            "function": self.function,
            "line_number": self.line_number,
            "stack_trace": self.stack_trace,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "metadata": self.metadata,
        }


class TradingGraphError(Exception):
    """Base exception for Trading Graph errors."""
    
    def __init__(
        self,
        message: str,
        user_message: Optional[str] = None,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        original_exception: Optional[Exception] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.user_message = user_message or self._generate_user_message(message)
        self.category = category
        self.severity = severity
        self.original_exception = original_exception
        self.metadata = metadata or {}
    
    def _generate_user_message(self, technical_message: str) -> str:
        """Generate user-friendly message from technical message."""
        # Map technical messages to user-friendly ones
        user_messages = {
            "connection": "Unable to connect to data source. Please check your network connection.",
            "timeout": "Request timed out. Please try again.",
            "authentication": "Authentication failed. Please check your credentials.",
            "authorization": "Access denied. You don't have permission for this action.",
            "rate_limit": "Too many requests. Please wait a moment and try again.",
            "model": "Model prediction failed. Please try retraining the model.",
            "data": "Data validation failed. Please check your input data.",
            "config": "Configuration error. Please check your settings.",
        }
        
        technical_message_lower = technical_message.lower()
        for key, user_msg in user_messages.items():
            if key in technical_message_lower:
                return user_msg
        
        return "An unexpected error occurred. Please try again or contact support."


# Specific exception types
class NetworkError(TradingGraphError):
    """Network-related errors."""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, category=ErrorCategory.NETWORK, **kwargs)


class DatabaseError(TradingGraphError):
    """Database-related errors."""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, category=ErrorCategory.DATABASE, **kwargs)


class ModelError(TradingGraphError):
    """Model-related errors."""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, category=ErrorCategory.MODEL, **kwargs)


class ValidationError(TradingGraphError):
    """Validation-related errors."""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, category=ErrorCategory.VALIDATION, **kwargs)


class AuthenticationError(TradingGraphError):
    """Authentication-related errors."""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, category=ErrorCategory.AUTHENTICATION, **kwargs)


class RateLimitError(TradingGraphError):
    """Rate limit errors."""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, category=ErrorCategory.RATE_LIMIT, **kwargs)


T = TypeVar('T')


def handle_errors(
    default_return: Optional[T] = None,
    retry_count: int = 0,
    retry_delay: float = 1.0,
    retry_exceptions: tuple = (NetworkError,),
    log_level: str = "error",
    raise_on_failure: bool = True,
) -> Callable:
    """
    Decorator for consistent error handling.
    
    FIXES:
    1. Consistent error handling across application
    2. Automatic retry for transient failures
    3. User-friendly error messages
    4. Error telemetry
    
    Usage:
        @handle_errors(default_return=None, retry_count=3)
        def fetch_data(symbol: str) -> Optional[Data]:
            ...
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(retry_count + 1):
                try:
                    return fn(*args, **kwargs)
                    
                except retry_exceptions as e:
                    last_exception = e
                    
                    if attempt < retry_count:
                        log.warning(
                            f"Retryable error in {fn.__name__}: {e}. "
                            f"Attempt {attempt + 1}/{retry_count + 1}. "
                            f"Retrying in {retry_delay}s..."
                        )
                        import time
                        time.sleep(retry_delay)
                    else:
                        log.error(
                            f"All retries exhausted for {fn.__name__}: {e}"
                        )
                        
                except TradingGraphError as e:
                    # Already handled exception
                    log.log(
                        getattr(log, log_level),
                        f"{fn.__name__}: {e.user_message}"
                    )
                    if raise_on_failure:
                        raise
                    return default_return  # type: ignore
                    
                except Exception as e:
                    # Unexpected exception
                    error_context = _create_error_context(
                        exception=e,
                        module=fn.__module__,
                        function=fn.__name__,
                    )
                    
                    log.error(
                        f"Unexpected error in {fn.__name__}: {error_context.user_message}"
                    )
                    log.debug(f"Stack trace: {error_context.stack_trace}")
                    
                    if raise_on_failure:
                        raise TradingGraphError(
                            message=str(e),
                            user_message=error_context.user_message,
                            severity=ErrorSeverity.ERROR,
                            original_exception=e,
                        )
                    return default_return  # type: ignore
            
            # All retries exhausted
            if raise_on_failure and last_exception:
                raise TradingGraphError(
                    message=str(last_exception),
                    user_message="Operation failed after multiple attempts. Please try again later.",
                    severity=ErrorSeverity.ERROR,
                    original_exception=last_exception,
                )
            return default_return  # type: ignore
        
        return wrapper
    return decorator


def _create_error_context(
    exception: Exception,
    module: str,
    function: str,
) -> ErrorContext:
    """Create error context from exception."""
    import hashlib
    import time
    
    # Get stack trace
    stack_trace = traceback.format_exc()
    
    # Get line number from traceback
    tb = exception.__traceback__
    line_number = tb.tb_lineno if tb else 0
    
    # Generate error ID
    error_id = hashlib.sha256(
        f"{module}.{function}.{time.time()}".encode()
    ).hexdigest()[:16]
    
    # Determine severity
    severity = ErrorSeverity.ERROR
    if isinstance(exception, (AuthenticationError, RateLimitError)):
        severity = ErrorSeverity.WARNING
    elif isinstance(exception, (NetworkError, DatabaseError)):
        severity = ErrorSeverity.ERROR
    
    # Generate user message
    user_message = _generate_user_friendly_message(exception)
    
    return ErrorContext(
        error_id=error_id,
        timestamp=datetime.now(),
        severity=severity,
        category=_categorize_exception(exception),
        message=str(exception),
        user_message=user_message,
        module=module,
        function=function,
        line_number=line_number,
        stack_trace=stack_trace,
        original_exception=exception,
    )


def _categorize_exception(exception: Exception) -> ErrorCategory:
    """Categorize exception for proper handling."""
    if isinstance(exception, NetworkError):
        return ErrorCategory.NETWORK
    elif isinstance(exception, DatabaseError):
        return ErrorCategory.DATABASE
    elif isinstance(exception, ModelError):
        return ErrorCategory.MODEL
    elif isinstance(exception, ValidationError):
        return ErrorCategory.VALIDATION
    elif isinstance(exception, AuthenticationError):
        return ErrorCategory.AUTHENTICATION
    elif isinstance(exception, RateLimitError):
        return ErrorCategory.RATE_LIMIT
    else:
        return ErrorCategory.INTERNAL


def _generate_user_friendly_message(exception: Exception) -> str:
    """Generate user-friendly error message."""
    if isinstance(exception, TradingGraphError):
        return exception.user_message
    
    exception_str = str(exception).lower()
    
    if "connection" in exception_str or "network" in exception_str:
        return "Unable to connect. Please check your network connection."
    elif "timeout" in exception_str:
        return "Request timed out. Please try again."
    elif "auth" in exception_str:
        return "Authentication failed. Please check your credentials."
    elif "permission" in exception_str or "access" in exception_str:
        return "Access denied. You don't have permission for this action."
    elif "rate" in exception_str or "limit" in exception_str:
        return "Too many requests. Please wait and try again."
    elif "model" in exception_str or "prediction" in exception_str:
        return "Model operation failed. Please try retraining."
    elif "data" in exception_str or "validation" in exception_str:
        return "Data validation failed. Please check your input."
    elif "config" in exception_str or "setting" in exception_str:
        return "Configuration error. Please check your settings."
    else:
        return "An unexpected error occurred. Please try again or contact support."


class ErrorRecovery:
    """
    Error recovery strategies.
    
    FIX: Automatic error recovery
    """
    
    @staticmethod
    def recover_from_network_error(
        fn: Callable,
        *args: Any,
        fallback_sources: Optional[list] = None,
        **kwargs: Any,
    ) -> Any:
        """Attempt recovery from network error."""
        try:
            return fn(*args, **kwargs)
        except NetworkError as e:
            log.warning(f"Network error, attempting recovery: {e}")
            
            # Try fallback sources
            if fallback_sources:
                for source in fallback_sources:
                    try:
                        log.info(f"Trying fallback source: {source}")
                        return source(*args, **kwargs)
                    except Exception as fallback_error:
                        log.warning(f"Fallback failed: {fallback_error}")
                        continue
            
            # Re-raise if all fallbacks fail
            raise
    
    @staticmethod
    def recover_from_model_error(
        fn: Callable,
        *args: Any,
        use_fallback_model: bool = True,
        fallback_model: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Any:
        """Attempt recovery from model error."""
        try:
            return fn(*args, **kwargs)
        except ModelError as e:
            log.warning(f"Model error, attempting recovery: {e}")
            
            # Try fallback model
            if use_fallback_model and fallback_model:
                try:
                    log.info("Using fallback model")
                    return fallback_model(*args, **kwargs)
                except Exception as fallback_error:
                    log.warning(f"Fallback model failed: {fallback_error}")
            
            # Return default prediction
            return {"prediction": 0.0, "confidence": 0.0, "fallback": True}
    
    @staticmethod
    def recover_from_database_error(
        fn: Callable,
        *args: Any,
        use_cache: bool = True,
        cache_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Attempt recovery from database error."""
        try:
            return fn(*args, **kwargs)
        except DatabaseError as e:
            log.warning(f"Database error, attempting recovery: {e}")
            
            # Try cache
            if use_cache and cache_key:
                from utils.performance_optimizer import get_optimizer
                optimizer = get_optimizer()
                cached = optimizer.cache.get(cache_key)
                if cached:
                    log.info("Using cached data")
                    return cached
            
            # Re-raise if recovery fails
            raise


class ErrorTelemetry:
    """
    Error telemetry and alerting.
    
    FIX: Error monitoring and alerting
    """
    
    def __init__(self):
        self._error_counts: dict[str, int] = {}
        self._alert_thresholds: dict[str, int] = {
            ErrorSeverity.CRITICAL.value: 1,
            ErrorSeverity.FATAL.value: 1,
            ErrorSeverity.ERROR.value: 10,
        }
    
    def record_error(self, context: ErrorContext) -> None:
        """Record error for telemetry."""
        key = f"{context.category.value}:{context.severity.value}"
        self._error_counts[key] = self._error_counts.get(key, 0) + 1
        
        # Check alert thresholds
        threshold = self._alert_thresholds.get(context.severity.value, 100)
        if self._error_counts[key] >= threshold:
            self._send_alert(context)
            self._error_counts[key] = 0  # Reset counter
    
    def _send_alert(self, context: ErrorContext) -> None:
        """Send alert for critical errors."""
        log.critical(
            f"ALERT: {context.severity.value} error rate exceeded threshold. "
            f"Category: {context.category.value}, "
            f"Message: {context.user_message}"
        )
        # In production, integrate with alerting systems (PagerDuty, Slack, etc.)


# Global error telemetry
_telemetry = ErrorTelemetry()


def get_telemetry() -> ErrorTelemetry:
    """Get error telemetry instance."""
    return _telemetry
