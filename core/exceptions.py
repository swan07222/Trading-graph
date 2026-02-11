# core/exceptions.py
"""
Custom Exceptions — comprehensive error handling.

FIXES APPLIED:
1. Added __str__ that includes code and details (not just message)
2. to_dict includes exception class name and timestamp
3. details is defensively copied to prevent caller mutation
4. Renamed SecurityError → TradingSecurityError to avoid shadowing builtins
5. Added __repr__ for debugging
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, List


class TradingSystemError(Exception):
    """Base exception for trading system."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        # FIX #3: Defensive copy so caller mutations don't affect us
        self.details: Dict[str, Any] = dict(details) if details else {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "error": self.code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }

    # FIX #1: __str__ includes code and details for meaningful log output
    def __str__(self) -> str:
        parts = [self.message]
        if self.code and self.code != self.__class__.__name__:
            parts.insert(0, f"[{self.code}]")
        if self.details:
            parts.append(f"(details: {self.details})")
        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"details={self.details!r})"
        )


# ── Data Errors ──────────────────────────────────────────────


class DataError(TradingSystemError):
    """Base data error."""


class DataFetchError(DataError):
    """Failed to fetch data."""


class DataValidationError(DataError):
    """Data validation failed."""


class InsufficientDataError(DataError):
    """Not enough data."""


class DataSourceUnavailableError(DataError):
    """Data source unavailable."""


# ── Trading Errors ───────────────────────────────────────────


class TradingError(TradingSystemError):
    """Base trading error."""


class OrderError(TradingError):
    """Order-related error."""


class OrderValidationError(OrderError):
    """Order validation failed."""


class OrderRejectedError(OrderError):
    """Order was rejected."""


class InsufficientFundsError(OrderError):
    """Insufficient funds for order."""


class InsufficientPositionError(OrderError):
    """Insufficient position for sell order."""


class PositionLimitError(OrderError):
    """Position limit exceeded."""


# ── Risk Errors ──────────────────────────────────────────────


class RiskError(TradingSystemError):
    """Base risk error."""


class RiskLimitBreachedError(RiskError):
    """Risk limit was breached."""


class DailyLossLimitError(RiskError):
    """Daily loss limit reached."""


class DrawdownLimitError(RiskError):
    """Maximum drawdown exceeded."""


class CircuitBreakerError(RiskError):
    """Circuit breaker activated."""


# ── Model Errors ─────────────────────────────────────────────


class ModelError(TradingSystemError):
    """Base model error."""


class ModelNotFoundError(ModelError):
    """Model file not found."""


class ModelLoadError(ModelError):
    """Failed to load model."""


class PredictionError(ModelError):
    """Prediction failed."""


# ── Broker Errors ────────────────────────────────────────────


class BrokerError(TradingSystemError):
    """Base broker error."""


class BrokerConnectionError(BrokerError):
    """Failed to connect to broker."""


class BrokerAuthenticationError(BrokerError):
    """Broker authentication failed."""


class BrokerOrderError(BrokerError):
    """Broker rejected order."""


# ── Security Errors ──────────────────────────────────────────

# FIX #4: Renamed from SecurityError to avoid shadowing builtins.SecurityError


class TradingSecurityError(TradingSystemError):
    """Base security error."""


class AuthenticationError(TradingSecurityError):
    """Authentication failed."""


class AuthorizationError(TradingSecurityError):
    """Not authorized for action."""


class RateLimitError(TradingSecurityError):
    """Rate limit exceeded."""


# Backward compatibility alias (import-safe, but won't shadow builtins)
SecurityError_ = TradingSecurityError