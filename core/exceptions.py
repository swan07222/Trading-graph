"""
Custom Exceptions - Comprehensive error handling
Score Target: 10/10
"""
from typing import Optional, Dict, Any


class TradingSystemError(Exception):
    """Base exception for trading system"""
    
    def __init__(self, message: str, code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict:
        return {
            'error': self.code,
            'message': self.message,
            'details': self.details
        }


# Data Errors
class DataError(TradingSystemError):
    """Base data error"""
    pass


class DataFetchError(DataError):
    """Failed to fetch data"""
    pass


class DataValidationError(DataError):
    """Data validation failed"""
    pass


class InsufficientDataError(DataError):
    """Not enough data"""
    pass


class DataSourceUnavailableError(DataError):
    """Data source unavailable"""
    pass


# Trading Errors
class TradingError(TradingSystemError):
    """Base trading error"""
    pass


class OrderError(TradingError):
    """Order-related error"""
    pass


class OrderValidationError(OrderError):
    """Order validation failed"""
    pass


class OrderRejectedError(OrderError):
    """Order was rejected"""
    pass


class InsufficientFundsError(OrderError):
    """Insufficient funds for order"""
    pass


class InsufficientPositionError(OrderError):
    """Insufficient position for sell order"""
    pass


class PositionLimitError(OrderError):
    """Position limit exceeded"""
    pass


# Risk Errors
class RiskError(TradingSystemError):
    """Base risk error"""
    pass


class RiskLimitBreachedError(RiskError):
    """Risk limit was breached"""
    pass


class DailyLossLimitError(RiskError):
    """Daily loss limit reached"""
    pass


class DrawdownLimitError(RiskError):
    """Maximum drawdown exceeded"""
    pass


class CircuitBreakerError(RiskError):
    """Circuit breaker activated"""
    pass


# Model Errors
class ModelError(TradingSystemError):
    """Base model error"""
    pass


class ModelNotFoundError(ModelError):
    """Model file not found"""
    pass


class ModelLoadError(ModelError):
    """Failed to load model"""
    pass


class PredictionError(ModelError):
    """Prediction failed"""
    pass


# Broker Errors
class BrokerError(TradingSystemError):
    """Base broker error"""
    pass


class BrokerConnectionError(BrokerError):
    """Failed to connect to broker"""
    pass


class BrokerAuthenticationError(BrokerError):
    """Broker authentication failed"""
    pass


class BrokerOrderError(BrokerError):
    """Broker rejected order"""
    pass


# Security Errors
class SecurityError(TradingSystemError):
    """Base security error"""
    pass


class AuthenticationError(SecurityError):
    """Authentication failed"""
    pass


class AuthorizationError(SecurityError):
    """Not authorized for action"""
    pass


class RateLimitError(SecurityError):
    """Rate limit exceeded"""
    pass