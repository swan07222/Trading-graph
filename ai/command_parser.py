"""Deterministic command parsing for trading control.

Fixes:
- Non-determinism: Structured command extraction with validation
- Safety: Explicit command types with parameter validation
- Audit: Every command is parsed, validated, and logged
- Security: No natural language execution without parsing
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)


class CommandType(Enum):
    """Types of trading commands.
    
    Each command type has specific parameters and validation rules.
    """
    # Market data commands
    GET_QUOTE = auto()
    GET_HISTORY = auto()
    GET_ORDERBOOK = auto()
    
    # Trading commands
    BUY = auto()
    SELL = auto()
    CANCEL_ORDER = auto()
    MODIFY_ORDER = auto()
    
    # Portfolio commands
    GET_POSITION = auto()
    GET_PORTFOLIO = auto()
    GET_PNL = auto()
    
    # Analysis commands
    RUN_ANALYSIS = auto()
    GET_PREDICTION = auto()
    RUN_BACKTEST = auto()
    GET_SENTIMENT = auto()
    
    # Model commands
    TRAIN_MODEL = auto()
    EVALUATE_MODEL = auto()
    EXPORT_MODEL = auto()
    
    # System commands
    HEALTH_CHECK = auto()
    GET_METRICS = auto()
    START_STREAM = auto()
    STOP_STREAM = auto()
    
    # Chat commands (non-trading)
    CHAT = auto()
    EXPLAIN = auto()
    SUMMARIZE = auto()
    
    # Unknown/unparsed
    UNKNOWN = auto()


@dataclass
class CommandParameter:
    """A validated command parameter."""
    name: str
    value: Any
    param_type: type
    required: bool = True
    min_value: float | None = None
    max_value: float | None = None
    choices: list[str] | None = None
    pattern: str | None = None


@dataclass
class ParsedCommand:
    """A parsed and validated trading command.
    
    This is the ONLY way to execute trading actions.
    Natural language is converted to ParsedCommand first.
    """
    command_type: CommandType
    parameters: dict[str, CommandParameter] = field(default_factory=dict)
    raw_input: str = ""
    confidence: float = 1.0  # Parsing confidence (0-1)
    requires_confirmation: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    command_id: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "command_type": self.command_type.name,
            "parameters": {
                k: {"name": v.name, "value": v.value, "type": v.param_type.__name__}
                for k, v in self.parameters.items()
            },
            "raw_input": self.raw_input,
            "confidence": self.confidence,
            "requires_confirmation": self.requires_confirmation,
            "created_at": self.created_at.isoformat(),
            "command_id": self.command_id,
        }
    
    def get_param(self, name: str, default: Any = None) -> Any:
        """Safely get a parameter value."""
        param = self.parameters.get(name)
        return param.value if param else default


class CommandParser:
    """Deterministic command parser for trading control.
    
    Converts natural language to structured commands with validation.
    
    Usage:
        parser = CommandParser()
        cmd = parser.parse("Buy 100 shares of AAPL at $150")
        if cmd.command_type == CommandType.BUY:
            # Execute with validated parameters
            quantity = cmd.get_param("quantity")
            symbol = cmd.get_param("symbol")
    """
    
    # Command patterns (regex)
    PATTERNS = {
        CommandType.BUY: [
            r"(?:buy|purchase|go\s+long|enter\s+long)\s+(?P<quantity>\d+)\s*(?:shares|units|lots)?\s*(?:of|@)?\s*(?P<symbol>[\w\.]+)(?:\s*(?:at|@)\s*\$?(?P<price>[\d\.]+))?",
            r"(?P<symbol>[\w\.]+)\s*(?:buy|purchase)\s*(?P<quantity>\d+)",
        ],
        CommandType.SELL: [
            r"(?:sell|liquidate|go\s+short|enter\s+short)\s+(?P<quantity>\d+)\s*(?:shares|units|lots)?\s*(?:of|@)?\s*(?P<symbol>[\w\.]+)(?:\s*(?:at|@)\s*\$?(?P<price>[\d\.]+))?",
            r"(?P<symbol>[\w\.]+)\s*(?:sell)\s*(?P<quantity>\d+)",
        ],
        CommandType.GET_QUOTE: [
            r"(?:get|show|fetch)\s*(?:quote|price|last)\s*(?:for)?\s*(?P<symbol>[\w\.]+)",
            r"(?P<symbol>[\w\.]+)\s*(?:price|quote|last)",
        ],
        CommandType.GET_PREDICTION: [
            r"(?:predict|forecast|analyze)\s*(?:for)?\s*(?P<symbol>[\w\.]+)",
            r"(?P<symbol>[\w\.]+)\s*(?:prediction|forecast)",
        ],
        CommandType.GET_SENTIMENT: [
            r"(?:sentiment|analyze\s+sentiment)\s*(?:for)?\s*(?P<symbol>[\w\.]+)",
        ],
        CommandType.RUN_BACKTEST: [
            r"(?:backtest|run\s+backtest)\s*(?:for)?\s*(?P<symbol>[\w\.]+)?\s*(?P<params>.*)?",
        ],
        CommandType.GET_PORTFOLIO: [
            r"(?:portfolio|positions|holdings|get\s+portfolio)",
        ],
        CommandType.HEALTH_CHECK: [
            r"(?:health|status|system\s+check|health\s+check)",
        ],
        CommandType.CHAT: [
            r"(?:chat|talk|ask|question)\s*[:\s]*(?P<message>.+)",
        ],
        CommandType.EXPLAIN: [
            r"(?:explain|why|how)\s+(?P<topic>.+)",
        ],
        CommandType.SUMMARIZE: [
            r"(?:summarize|summary|tl;dr)\s+(?P<topic>.+)",
        ],
    }
    
    # Parameter validators
    PARAM_VALIDATORS = {
        "symbol": {
            "type": str,
            "pattern": r"^[\w\.]+$",
            "required": True,
        },
        "quantity": {
            "type": int,
            "min_value": 1,
            "required": True,
        },
        "price": {
            "type": float,
            "min_value": 0.01,
            "required": False,
        },
        "message": {
            "type": str,
            "required": True,
        },
        "topic": {
            "type": str,
            "required": True,
        },
    }
    
    def __init__(self) -> None:
        self._compiled_patterns = {}
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        for cmd_type, patterns in self.PATTERNS.items():
            self._compiled_patterns[cmd_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def parse(self, input_text: str) -> ParsedCommand:
        """Parse natural language input into a structured command.
        
        Args:
            input_text: User input text
            
        Returns:
            ParsedCommand with validated parameters
        """
        input_text = input_text.strip()
        
        # Try each command type
        for cmd_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(input_text)
                if match:
                    return self._build_command(cmd_type, match, input_text)
        
        # No match found
        return ParsedCommand(
            command_type=CommandType.UNKNOWN,
            raw_input=input_text,
            confidence=0.0,
            command_id=self._generate_command_id(),
        )
    
    def _build_command(
        self,
        cmd_type: CommandType,
        match: re.Match,
        raw_input: str,
    ) -> ParsedCommand:
        """Build a ParsedCommand from regex match."""
        params = {}
        
        # Extract matched groups
        for name, value in match.groupdict().items():
            if value is not None:
                validator = self.PARAM_VALIDATORS.get(name, {})
                param_type = validator.get("type", str)
                
                # Type conversion and validation
                try:
                    if param_type == int:
                        value = int(value)
                    elif param_type == float:
                        value = float(value)
                    elif param_type == str:
                        value = str(value).strip()
                    
                    # Validate constraints
                    if validator.get("min_value") is not None:
                        if value < validator["min_value"]:
                            raise ValueError(f"{name} below minimum")
                    
                    if validator.get("max_value") is not None:
                        if value > validator["max_value"]:
                            raise ValueError(f"{name} above maximum")
                    
                    if validator.get("pattern") is not None:
                        if not re.match(validator["pattern"], value):
                            raise ValueError(f"{name} doesn't match pattern")
                    
                    params[name] = CommandParameter(
                        name=name,
                        value=value,
                        param_type=param_type,
                        required=validator.get("required", False),
                        min_value=validator.get("min_value"),
                        max_value=validator.get("max_value"),
                        pattern=validator.get("pattern"),
                    )
                    
                except (ValueError, TypeError) as e:
                    log.warning(f"Parameter validation failed for {name}: {e}")
                    continue
        
        # Determine if confirmation is required
        requires_confirmation = cmd_type in {
            CommandType.BUY,
            CommandType.SELL,
            CommandType.CANCEL_ORDER,
            CommandType.MODIFY_ORDER,
            CommandType.TRAIN_MODEL,
        }
        
        return ParsedCommand(
            command_type=cmd_type,
            parameters=params,
            raw_input=raw_input,
            confidence=0.9,  # High confidence since pattern matched
            requires_confirmation=requires_confirmation,
            command_id=self._generate_command_id(),
        )
    
    def parse_json(self, json_str: str) -> ParsedCommand:
        """Parse a JSON command (for programmatic use).
        
        This is the safest way to execute commands programmatically.
        
        Example:
            {
                "command_type": "BUY",
                "parameters": {
                    "symbol": {"value": "AAPL", "type": "str"},
                    "quantity": {"value": 100, "type": "int"}
                }
            }
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        
        cmd_type_str = data.get("command_type", "UNKNOWN").upper()
        try:
            cmd_type = CommandType[cmd_type_str]
        except KeyError:
            raise ValueError(f"Unknown command type: {cmd_type_str}")
        
        params = {}
        for name, param_data in data.get("parameters", {}).items():
            value = param_data.get("value")
            type_name = param_data.get("type", "str")
            
            param_type = {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
            }.get(type_name, str)
            
            if param_type in (int, float):
                try:
                    value = param_type(value)
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid {type_name} value for {name}")
            
            params[name] = CommandParameter(
                name=name,
                value=value,
                param_type=param_type,
                required=param_data.get("required", False),
            )
        
        requires_confirmation = data.get(
            "requires_confirmation",
            cmd_type in {CommandType.BUY, CommandType.SELL}
        )
        
        return ParsedCommand(
            command_type=cmd_type,
            parameters=params,
            raw_input=json_str,
            confidence=1.0,  # JSON is explicit
            requires_confirmation=requires_confirmation,
            command_id=self._generate_command_id(),
        )
    
    def _generate_command_id(self) -> str:
        """Generate a unique command ID for audit tracking."""
        import uuid
        return f"cmd_{uuid.uuid4().hex[:12]}"
    
    def get_command_help(self, cmd_type: CommandType) -> str:
        """Get help text for a command type."""
        help_texts = {
            CommandType.BUY: "Buy shares. Example: 'Buy 100 shares of AAPL at $150'",
            CommandType.SELL: "Sell shares. Example: 'Sell 50 shares of MSFT'",
            CommandType.GET_QUOTE: "Get current quote. Example: 'Get quote for GOOGL'",
            CommandType.GET_PREDICTION: "Get AI prediction. Example: 'Predict TSLA'",
            CommandType.GET_SENTIMENT: "Get sentiment analysis. Example: 'Sentiment for NVDA'",
            CommandType.RUN_BACKTEST: "Run backtest. Example: 'Backtest AAPL 2023-2024'",
            CommandType.GET_PORTFOLIO: "View portfolio positions",
            CommandType.HEALTH_CHECK: "Check system health",
            CommandType.CHAT: "Chat with AI assistant",
            CommandType.EXPLAIN: "Explain a concept. Example: 'Explain RSI divergence'",
        }
        return help_texts.get(cmd_type, "No help available")
    
    def list_commands(self) -> list[dict[str, Any]]:
        """List all available commands."""
        return [
            {
                "name": cmd.name,
                "value": cmd.value,
                "help": self.get_command_help(cmd),
            }
            for cmd in CommandType
        ]


# Singleton instance
_parser_instance: CommandParser | None = None


def get_parser() -> CommandParser:
    """Get the singleton CommandParser instance."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = CommandParser()
    return _parser_instance


def parse_command(input_text: str) -> ParsedCommand:
    """Convenience function to parse a command."""
    return get_parser().parse(input_text)
