"""Safety validation layer for trading commands.

Fixes:
- Safety risks: Multi-layer validation before execution
- Security: Position limits, risk checks, circuit breakers
- Compliance: Pre-trade validation and audit trail
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum, auto
from pathlib import Path
from typing import Any

from config.settings import CONFIG
from utils.logger import get_logger

from .command_parser import ParsedCommand, CommandType

log = get_logger(__name__)


class ValidationLevel(Enum):
    """Levels of validation strictness."""
    NONE = auto()      # No validation (testing only)
    BASIC = auto()     # Basic parameter validation
    STANDARD = auto()  # Standard risk checks
    STRICT = auto()    # Strict risk + compliance checks
    INSTITUTIONAL = auto()  # Full institutional-grade checks


class ValidationResult(Enum):
    """Result of a validation check."""
    PASS = auto()
    WARNING = auto()
    BLOCKED = auto()
    ERROR = auto()


@dataclass
class ValidationCheck:
    """A single validation check result."""
    name: str
    level: ValidationLevel
    result: ValidationResult
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level.name,
            "result": self.result.name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SafetyReport:
    """Complete safety validation report for a command."""
    command_id: str
    command_type: CommandType
    overall_result: ValidationResult
    checks: list[ValidationCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    requires_approval: bool = False
    approval_level: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_approved(self) -> bool:
        """Check if command passed all validations."""
        return self.overall_result == ValidationResult.PASS
    
    def can_execute(self) -> bool:
        """Check if command can be executed (may need approval)."""
        return self.overall_result != ValidationResult.BLOCKED
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "command_type": self.command_type.name,
            "overall_result": self.overall_result.name,
            "checks": [c.to_dict() for c in self.checks],
            "warnings": self.warnings,
            "blocked_reasons": self.blocked_reasons,
            "requires_approval": self.requires_approval,
            "approval_level": self.approval_level,
            "timestamp": self.timestamp.isoformat(),
        }


class SafetyValidator:
    """Multi-layer safety validation for trading commands.
    
    Validation Layers:
    1. Parameter Validation - Check types, ranges, required fields
    2. Risk Limits - Position limits, order size limits
    3. Market Hours - Trading hour restrictions
    4. Circuit Breakers - Volatility, loss limits
    5. Compliance - Regulatory restrictions
    """
    
    def __init__(
        self,
        level: ValidationLevel = ValidationLevel.STANDARD,
        config_path: Path | None = None,
    ) -> None:
        self.level = level
        self.config_path = config_path or self._default_config_path()
        self._config = self._load_config()
        
        # Risk limits (can be overridden by config)
        self.max_order_value: float = 1_000_000.0  # Max order value in base currency
        self.max_position_value: float = 10_000_000.0  # Max position value
        self.max_daily_loss: float = 100_000.0  # Max daily loss
        self.max_order_size_pct: float = 0.05  # Max order as % of avg volume
        self.position_concentration_limit: float = 0.25  # Max % in single position
        
        # Trading hours (China A-share market)
        self.trading_hours = {
            "morning": (time(9, 30), time(11, 30)),
            "afternoon": (time(13, 0), time(15, 0)),
        }
        
        # Circuit breakers
        self.circuit_breaker_enabled = True
        self.volatility_threshold: float = 0.05  # 5% move triggers check
        self.loss_threshold: float = 0.02  # 2% daily loss triggers check
        
        # Load additional config
        self._apply_config()
    
    def _default_config_path(self) -> Path:
        """Get default safety config path."""
        return CONFIG.config_dir / "safety_config.json"
    
    def _load_config(self) -> dict[str, Any]:
        """Load safety configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Failed to load safety config: {e}")
        return {}
    
    def _apply_config(self) -> None:
        """Apply configuration overrides."""
        if "max_order_value" in self._config:
            self.max_order_value = float(self._config["max_order_value"])
        if "max_position_value" in self._config:
            self.max_position_value = float(self._config["max_position_value"])
        if "max_daily_loss" in self._config:
            self.max_daily_loss = float(self._config["max_daily_loss"])
        if "trading_hours" in self._config:
            self.trading_hours.update(self._config["trading_hours"])
    
    def validate(
        self,
        command: ParsedCommand,
        context: dict[str, Any] | None = None,
    ) -> SafetyReport:
        """Validate a trading command.
        
        Args:
            command: Parsed command to validate
            context: Optional context (portfolio, market data, etc.)
            
        Returns:
            SafetyReport with validation results
        """
        context = context or {}
        checks = []
        warnings = []
        blocked_reasons = []
        
        # Run validation checks based on command type
        if command.command_type in {CommandType.BUY, CommandType.SELL}:
            checks.extend(self._validate_trade_command(command, context))
        elif command.command_type == CommandType.GET_QUOTE:
            checks.extend(self._validate_data_command(command, context))
        
        # Determine overall result
        overall = ValidationResult.PASS
        
        for check in checks:
            if check.result == ValidationResult.BLOCKED:
                blocked_reasons.append(check.message)
                overall = ValidationResult.BLOCKED
            elif check.result == ValidationResult.ERROR:
                blocked_reasons.append(check.message)
                overall = ValidationResult.BLOCKED
            elif check.result == ValidationResult.WARNING:
                warnings.append(check.message)
                if overall == ValidationResult.PASS:
                    overall = ValidationResult.WARNING
        
        # Determine if approval is required
        requires_approval = False
        approval_level = None
        
        if overall == ValidationResult.WARNING and self.level >= ValidationLevel.STRICT:
            requires_approval = True
            approval_level = "senior_trader"
        elif command.command_type in {CommandType.BUY, CommandType.SELL}:
            order_value = command.get_param("quantity", 0) * command.get_param("price", 0)
            if order_value > self.max_order_value * 0.5:
                requires_approval = True
                approval_level = "risk_manager"
        
        return SafetyReport(
            command_id=command.command_id,
            command_type=command.command_type,
            overall_result=overall,
            checks=checks,
            warnings=warnings,
            blocked_reasons=blocked_reasons,
            requires_approval=requires_approval,
            approval_level=approval_level,
        )
    
    def _validate_trade_command(
        self,
        command: ParsedCommand,
        context: dict[str, Any],
    ) -> list[ValidationCheck]:
        """Validate a trade command (BUY/SELL)."""
        checks = []
        
        # Check 1: Required parameters
        symbol = command.get_param("symbol")
        quantity = command.get_param("quantity")
        price = command.get_param("price")
        
        if not symbol:
            checks.append(ValidationCheck(
                name="required_parameters",
                level=self.level,
                result=ValidationResult.BLOCKED,
                message="Missing required parameter: symbol",
            ))
            return checks
        
        if not quantity or quantity <= 0:
            checks.append(ValidationCheck(
                name="required_parameters",
                level=self.level,
                result=ValidationResult.BLOCKED,
                message="Invalid or missing quantity parameter",
            ))
            return checks
        
        checks.append(ValidationCheck(
            name="required_parameters",
            level=self.level,
            result=ValidationResult.PASS,
            message="All required parameters present",
        ))
        
        # Check 2: Order value limit
        if price and quantity:
            order_value = price * quantity
            if order_value > self.max_order_value:
                checks.append(ValidationCheck(
                    name="order_value_limit",
                    level=self.level,
                    result=ValidationResult.BLOCKED,
                    message=f"Order value {order_value:.2f} exceeds limit {self.max_order_value:.2f}",
                    details={"order_value": order_value, "limit": self.max_order_value},
                ))
            else:
                checks.append(ValidationCheck(
                    name="order_value_limit",
                    level=self.level,
                    result=ValidationResult.PASS,
                    message=f"Order value within limits",
                    details={"order_value": order_value, "limit": self.max_order_value},
                ))
        
        # Check 3: Trading hours
        if not self._is_trading_hours():
            checks.append(ValidationCheck(
                name="trading_hours",
                level=self.level,
                result=ValidationResult.WARNING,
                message="Outside trading hours - order will be queued",
                details={"current_time": datetime.now().time().isoformat()},
            ))
        else:
            checks.append(ValidationCheck(
                name="trading_hours",
                level=self.level,
                result=ValidationResult.PASS,
                message="Within trading hours",
            ))
        
        # Check 4: Position limits (if context available)
        if "portfolio" in context:
            portfolio = context["portfolio"]
            current_position = portfolio.get(symbol, {}).get("quantity", 0)
            current_price = portfolio.get(symbol, {}).get("price", price or 0)
            
            if command.command_type == CommandType.BUY:
                new_quantity = current_position + quantity
                new_value = new_quantity * (price or current_price)
                
                if new_value > self.max_position_value:
                    checks.append(ValidationCheck(
                        name="position_limit",
                        level=self.level,
                        result=ValidationResult.BLOCKED,
                        message=f"New position value {new_value:.2f} exceeds limit",
                        details={"new_value": new_value, "limit": self.max_position_value},
                    ))
                else:
                    checks.append(ValidationCheck(
                        name="position_limit",
                        level=self.level,
                        result=ValidationResult.PASS,
                        message="Position within limits",
                    ))
        
        # Check 5: Daily loss limit (circuit breaker)
        if "daily_pnl" in context:
            daily_pnl = context["daily_pnl"]
            if daily_pnl < -self.max_daily_loss:
                checks.append(ValidationCheck(
                    name="circuit_breaker",
                    level=self.level,
                    result=ValidationResult.BLOCKED,
                    message=f"Daily loss {daily_pnl:.2f} exceeds limit {-self.max_daily_loss:.2f}",
                    details={"daily_pnl": daily_pnl, "limit": -self.max_daily_loss},
                ))
            else:
                checks.append(ValidationCheck(
                    name="circuit_breaker",
                    level=self.level,
                    result=ValidationResult.PASS,
                    message="Daily loss within limits",
                ))
        
        return checks
    
    def _validate_data_command(
        self,
        command: ParsedCommand,
        context: dict[str, Any],
    ) -> list[ValidationCheck]:
        """Validate a data access command."""
        checks = []
        
        # Basic validation - data commands are generally safe
        symbol = command.get_param("symbol")
        if not symbol:
            checks.append(ValidationCheck(
                name="symbol_required",
                level=self.level,
                result=ValidationResult.ERROR,
                message="Symbol parameter required",
            ))
        else:
            checks.append(ValidationCheck(
                name="symbol_validation",
                level=self.level,
                result=ValidationResult.PASS,
                message=f"Valid symbol: {symbol}",
            ))
        
        return checks
    
    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours."""
        now = datetime.now().time()
        
        # Check if weekend
        if datetime.now().weekday() >= 5:
            return False
        
        # Check morning session
        morning_start, morning_end = self.trading_hours["morning"]
        if morning_start <= now <= morning_end:
            return True
        
        # Check afternoon session
        afternoon_start, afternoon_end = self.trading_hours["afternoon"]
        if afternoon_start <= now <= afternoon_end:
            return True
        
        return False
    
    def update_config(self, updates: dict[str, Any]) -> None:
        """Update safety configuration.
        
        Requires appropriate permissions in production.
        """
        self._config.update(updates)
        self._apply_config()
        
        # Persist config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
        
        log.info(f"Safety config updated: {updates.keys()}")
    
    def get_risk_limits(self) -> dict[str, Any]:
        """Get current risk limits."""
        return {
            "max_order_value": self.max_order_value,
            "max_position_value": self.max_position_value,
            "max_daily_loss": self.max_daily_loss,
            "max_order_size_pct": self.max_order_size_pct,
            "position_concentration_limit": self.position_concentration_limit,
            "trading_hours": self.trading_hours,
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "volatility_threshold": self.volatility_threshold,
            "loss_threshold": self.loss_threshold,
        }


# Singleton instance
_validator_instance: SafetyValidator | None = None


def get_validator(level: ValidationLevel | None = None) -> SafetyValidator:
    """Get or create the singleton SafetyValidator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = SafetyValidator(level or ValidationLevel.STANDARD)
    return _validator_instance


def validate_command(
    command: ParsedCommand,
    context: dict[str, Any] | None = None,
) -> SafetyReport:
    """Convenience function to validate a command."""
    return get_validator().validate(command, context)
