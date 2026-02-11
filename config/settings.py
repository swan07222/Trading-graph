# config/settings.py
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime, time
import threading

_SENTINEL = object()

# Minimal logger that doesn't depend on our logger module
# (avoids circular import: settings → logger → settings)
_log = logging.getLogger("config.settings")


class TradingMode(Enum):
    SIMULATION = "simulation"
    PAPER = "paper"
    LIVE = "live"


class RiskProfile(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class MarketType(Enum):
    A_SHARE = "a_share"
    HK = "hk"
    US = "us"


@dataclass
class DataConfig:
    """Data layer configuration."""
    cache_ttl_hours: float = 4.0
    max_memory_cache_mb: int = 500
    parallel_downloads: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    min_history_days: int = 200
    feature_lookback: int = 60
    poll_interval_seconds: float = 3.0


@dataclass
class ModelConfig:
    """ML model configuration."""
    sequence_length: int = 60
    hidden_size: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.3
    num_classes: int = 3

    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.0005
    early_stop_patience: int = 15
    weight_decay: float = 0.01

    # Splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Prediction
    prediction_horizon: int = 5
    up_threshold: float = 2.0
    down_threshold: float = -2.0
    embargo_bars: int = 10
    min_confidence: float = 0.55

    # Signal thresholds
    strong_buy_threshold: float = 0.65
    buy_threshold: float = 0.55
    sell_threshold: float = 0.55
    strong_sell_threshold: float = 0.65


@dataclass
class TradingConfig:
    """Trading rules configuration."""
    lot_size: int = 100
    commission: float = 0.00025
    stamp_tax: float = 0.001
    slippage: float = 0.001
    t_plus_1: bool = True
    allow_short: bool = False

    price_limit_pct: float = 10.0
    st_price_limit_pct: float = 5.0

    market_open_am: time = field(default_factory=lambda: time(9, 30))
    market_close_am: time = field(default_factory=lambda: time(11, 30))
    market_open_pm: time = field(default_factory=lambda: time(13, 0))
    market_close_pm: time = field(default_factory=lambda: time(15, 0))


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_pct: float = 15.0
    max_portfolio_risk_pct: float = 30.0
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 15.0
    max_positions: int = 10
    risk_per_trade_pct: float = 2.0
    var_confidence: float = 0.95
    kelly_fraction: float = 0.25

    # Circuit breakers
    circuit_breaker_loss_pct: float = 5.0
    circuit_breaker_duration_minutes: int = 60
    max_orders_per_minute: int = 10
    max_orders_per_day: int = 100

    # Kill switch
    kill_switch_loss_pct: float = 8.0
    kill_switch_drawdown_pct: float = 20.0


@dataclass
class SecurityConfig:
    """Security configuration."""
    encrypt_credentials: bool = True
    audit_logging: bool = True
    require_2fa_for_live: bool = True
    max_session_hours: int = 8
    ip_whitelist: List[str] = field(default_factory=list)


@dataclass
class AlertConfig:
    """Alerting configuration."""
    enabled: bool = True
    email_enabled: bool = False
    email_recipients: List[str] = field(default_factory=list)
    smtp_server: str = ""
    smtp_port: int = 587
    sms_enabled: bool = False
    webhook_enabled: bool = False
    webhook_url: str = ""

    # Alert thresholds
    large_loss_alert_pct: float = 2.0
    position_concentration_alert_pct: float = 20.0
    connection_loss_alert_seconds: int = 30

    from_email: str = ""
    smtp_username: str = ""
    smtp_password_key: str = "smtp_password"


@dataclass
class AutoTradeConfig:
    """
    Auto-trading configuration.

    Controls the autonomous trading engine that can execute trades
    without manual confirmation when enabled.
    """
    # Master enable/disable
    enabled: bool = False

    # Signal filters — minimum thresholds for auto-execution
    min_confidence: float = 0.70
    min_signal_strength: float = 0.60
    min_model_agreement: float = 0.65

    # Which signals to auto-trade
    allow_strong_buy: bool = True
    allow_buy: bool = True
    allow_sell: bool = True
    allow_strong_sell: bool = True
    allow_hold: bool = False  # HOLD never auto-trades by default

    # Position limits for auto-trading (can be tighter than manual)
    max_auto_positions: int = 5
    max_auto_position_pct: float = 10.0
    max_auto_order_value: float = 50000.0

    # Timing
    scan_interval_seconds: int = 60
    cooldown_after_trade_seconds: int = 300
    max_trades_per_day: int = 10
    max_trades_per_stock_per_day: int = 2

    # Safety
    require_market_open: bool = True
    require_broker_connected: bool = True
    pause_on_high_volatility: bool = True
    volatility_pause_threshold: float = 5.0  # ATR% above this pauses

    # Auto-stop loss management
    auto_stop_loss: bool = True
    trailing_stop_enabled: bool = False
    trailing_stop_pct: float = 3.0

    # Notification
    notify_on_trade: bool = True
    notify_on_skip: bool = False

    # Paper trading safety — require explicit confirmation for live
    confirm_live_auto_trade: bool = True


def _safe_dataclass_from_dict(dc_instance, data: Dict) -> List[str]:
    """
    Apply dict values to a dataclass instance with type checking.
    Returns list of warnings for bad values.
    """
    warnings_list = []
    if not isinstance(data, dict):
        return [f"Expected dict, got {type(data).__name__}"]

    dc_fields = {f.name: f for f in fields(dc_instance)}

    for key, value in data.items():
        if key not in dc_fields:
            warnings_list.append(f"Unknown field '{key}' — ignored")
            continue

        current_value = getattr(dc_instance, key)

        try:
            # FIX: Check bool BEFORE int (since bool is subclass of int)
            if isinstance(current_value, bool):
                if isinstance(value, bool):
                    setattr(dc_instance, key, value)
                elif isinstance(value, (int, str)):
                    setattr(dc_instance, key, bool(value))
                else:
                    warnings_list.append(
                        f"Bad type for bool field '{key}': {type(value).__name__}"
                    )
            elif isinstance(current_value, int) and isinstance(value, (int, float)):
                setattr(dc_instance, key, int(value))
            elif isinstance(current_value, float) and isinstance(value, (int, float)):
                setattr(dc_instance, key, float(value))
            elif isinstance(current_value, str) and isinstance(value, str):
                setattr(dc_instance, key, value)
            elif isinstance(current_value, list) and isinstance(value, list):
                setattr(dc_instance, key, value)
            elif isinstance(current_value, time) and isinstance(value, str):
                # Parse "HH:MM" or "HH:MM:SS"
                parts = [int(p) for p in value.split(":")]
                setattr(dc_instance, key, time(*parts))
            else:
                warnings_list.append(
                    f"Type mismatch for '{key}': "
                    f"expected {type(current_value).__name__}, "
                    f"got {type(value).__name__}"
                )
        except (TypeError, ValueError) as e:
            warnings_list.append(f"Bad value for '{key}': {value!r} — {e}")

    return warnings_list


def _dataclass_to_dict(dc_instance) -> Dict:
    """Serialize a dataclass to dict, handling special types."""
    result = {}
    for f in fields(dc_instance):
        value = getattr(dc_instance, f.name)
        if isinstance(value, time):
            result[f.name] = value.strftime("%H:%M:%S")
        elif isinstance(value, Enum):
            result[f.name] = value.value
        else:
            result[f.name] = value
    return result


class Config:
    """
    Production-grade configuration manager.

    Usage:
        from config.settings import CONFIG
        print(CONFIG.model.learning_rate)
        print(CONFIG.LEARNING_RATE)  # Legacy alias

    Ensure config/__init__.py contains:
        from config.settings import CONFIG, TradingMode, RiskProfile, MarketType
        __all__ = ["CONFIG", "TradingMode", "RiskProfile", "MarketType"]
    """

    _instance: Optional[Config] = None
    _instance_lock = threading.RLock()

    def __new__(cls) -> Config:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        FIX #9: Destroy singleton for testing.
        Call between tests to get a fresh Config.
        """
        with cls._instance_lock:
            cls._instance = None

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True
        self._lock = threading.RLock()
        self._config_file = Path(__file__).parent.parent / "config.json"
        self._env_prefix = "TRADING_"
        self._validation_warnings: List[str] = []

        # Sub-configs
        self.data = DataConfig()
        self.model = ModelConfig()
        self.trading = TradingConfig()
        self.risk = RiskConfig()
        self.security = SecurityConfig()
        self.alerts = AlertConfig()
        self.auto_trade = AutoTradeConfig()

        # Main settings
        self.capital: float = 100_000.0
        self.trading_mode: TradingMode = TradingMode.SIMULATION
        self.risk_profile: RiskProfile = RiskProfile.MODERATE
        self.market_type: MarketType = MarketType.A_SHARE
        self.broker_path: str = ""

        # Paths
        self._base_dir = Path(__file__).parent.parent
        self._model_dir_override: Optional[str] = None

        # FIX #6: Sentinel-based caching for path properties
        self._data_dir_cached: Any = _SENTINEL
        self._model_dir_cached: Any = _SENTINEL
        self._model_dir_cached_override: Any = _SENTINEL
        self._log_dir_cached: Any = _SENTINEL
        self._cache_dir_cached: Any = _SENTINEL
        self._audit_dir_cached: Any = _SENTINEL

        # Stock pool
        self.stock_pool: List[str] = [
            "600519", "601318", "600036", "000858", "600900",
            "002594", "300750", "002475", "300059", "002230",
            "000333", "000651", "600887", "603288", "600276",
            "300760", "300015", "601166", "601398", "600030",
        ]

        # Training settings
        self.min_stocks_for_training: int = 5
        self.auto_learn_epochs: int = 50

        # Load from file/env
        # FIX LOCK: _load() is called from __init__ so we do NOT acquire
        # self._lock here (it would deadlock on RLock from __getattr__
        # during initialization). _load is only called once from __init__
        # which is already protected by _instance_lock.
        self._load()
        # FIX #1: Validate but don't crash — collect warnings
        self._validate()

    # ==================== LEGACY COMPATIBILITY ====================

    # FIX #8: Single unified mapping
    _LEGACY_MAP: Dict[str, str] = {
        # Model aliases
        "SEQUENCE_LENGTH": "model.sequence_length",
        "PREDICTION_HORIZON": "model.prediction_horizon",
        "NUM_CLASSES": "model.num_classes",
        "HIDDEN_SIZE": "model.hidden_size",
        "DROPOUT": "model.dropout",
        "EPOCHS": "model.epochs",
        "BATCH_SIZE": "model.batch_size",
        "LEARNING_RATE": "model.learning_rate",
        "WEIGHT_DECAY": "model.weight_decay",
        "EARLY_STOP_PATIENCE": "model.early_stop_patience",
        "MIN_CONFIDENCE": "model.min_confidence",
        "BUY_THRESHOLD": "model.buy_threshold",
        "SELL_THRESHOLD": "model.sell_threshold",
        "STRONG_BUY_THRESHOLD": "model.strong_buy_threshold",
        "STRONG_SELL_THRESHOLD": "model.strong_sell_threshold",
        "UP_THRESHOLD": "model.up_threshold",
        "DOWN_THRESHOLD": "model.down_threshold",
        "EMBARGO_BARS": "model.embargo_bars",
        "TRAIN_RATIO": "model.train_ratio",
        "VAL_RATIO": "model.val_ratio",
        "TEST_RATIO": "model.test_ratio",
        # Trading aliases
        "COMMISSION": "trading.commission",
        "STAMP_TAX": "trading.stamp_tax",
        "SLIPPAGE": "trading.slippage",
        "LOT_SIZE": "trading.lot_size",
        # Risk aliases
        "MAX_POSITION_PCT": "risk.max_position_pct",
        "MAX_DAILY_LOSS_PCT": "risk.max_daily_loss_pct",
        "MAX_POSITIONS": "risk.max_positions",
        "RISK_PER_TRADE": "risk.risk_per_trade_pct",
        # Path aliases
        "DATA_DIR": "_prop_data_dir",
        "dataDir": "_prop_data_dir",
        "MODEL_DIR": "_prop_model_dir",
        "modelDir": "_prop_model_dir",
        "CACHE_DIR": "_prop_cache_dir",
        "cacheDir": "_prop_cache_dir",
        "AUDIT_DIR": "_prop_audit_dir",
        "auditDir": "_prop_audit_dir",
        "LOG_DIR": "_prop_log_dir",
    }

    def _get_nested(self, path: str) -> Any:
        """Get a dotted attribute path like 'model.sequence_length'."""
        parts = path.split(".")
        obj = self
        for part in parts:
            obj = getattr(obj, part)
        return obj

    def __getattr__(self, name: str) -> Any:
        """
        Legacy compatibility: CONFIG.SEQUENCE_LENGTH → CONFIG.model.sequence_length

        Only called when normal attribute lookup fails,
        so @property definitions always take precedence.

        FIX GETATTR: Guard against recursive calls during __init__
        by checking _initialized. During __init__, sub-configs may not
        exist yet, so we must not try to resolve legacy names.
        """
        # FIX GETATTR: During __init__, _initialized may not exist yet
        # (object.__getattribute__ hasn't set it). Use object.__getattribute__
        # to check without recursing.
        try:
            initialized = object.__getattribute__(self, '_initialized')
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute {name!r}"
            )

        if not initialized:
            raise AttributeError(
                f"{self.__class__.__name__} not yet initialized, "
                f"cannot access {name!r}"
            )

        mapping = Config._LEGACY_MAP.get(name)
        if mapping is None:
            # Stock pool alias
            if name == "STOCK_POOL":
                return self.stock_pool
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute {name!r}"
            )

        # Path properties — delegate to the property
        if mapping.startswith("_prop_"):
            prop_name = mapping[6:]  # strip "_prop_"
            return getattr(self, prop_name)

        return self._get_nested(mapping)

    # ==================== EXPLICIT PROPERTIES ====================

    @property
    def CAPITAL(self) -> float:
        return self.capital

    @CAPITAL.setter
    def CAPITAL(self, value: float) -> None:
        # FIX SETATTR: Validate positive value
        value = float(value)
        if value <= 0:
            _log.warning(f"Capital must be positive, got {value} — clamping to 1.0")
            value = 1.0
        self.capital = value

    @property
    def MIN_STOCKS_FOR_TRAINING(self) -> int:
        return self.min_stocks_for_training

    @property
    def TRADING_MODE(self) -> TradingMode:
        return self.trading_mode

    @TRADING_MODE.setter
    def TRADING_MODE(self, value: TradingMode) -> None:
        self.trading_mode = value

    @property
    def BROKER_PATH(self) -> str:
        return self.broker_path

    @BROKER_PATH.setter
    def BROKER_PATH(self, value: str) -> None:
        self.broker_path = str(value) if value else ""

    @property
    def AUTO_LEARN_EPOCHS(self) -> int:
        return self.auto_learn_epochs

    # ==================== PATH PROPERTIES ====================
    # FIX #11: No directory creation in getters. Use ensure_dirs() explicitly.

    @property
    def BASE_DIR(self) -> Path:
        return self._base_dir

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def data_dir(self) -> Path:
        if self._data_dir_cached is _SENTINEL:
            self._data_dir_cached = self._base_dir / "data_storage"
        return self._data_dir_cached

    @property
    def model_dir(self) -> Path:
        override = self._model_dir_override
        if (
            self._model_dir_cached is _SENTINEL
            or self._model_dir_cached_override is not override
        ):
            if override:
                self._model_dir_cached = Path(override)
            else:
                self._model_dir_cached = self._base_dir / "models_saved"
            self._model_dir_cached_override = override
        return self._model_dir_cached

    @property
    def log_dir(self) -> Path:
        if self._log_dir_cached is _SENTINEL:
            self._log_dir_cached = self._base_dir / "logs"
        return self._log_dir_cached

    @property
    def cache_dir(self) -> Path:
        if self._cache_dir_cached is _SENTINEL:
            self._cache_dir_cached = self._base_dir / "cache"
        return self._cache_dir_cached

    @property
    def audit_dir(self) -> Path:
        if self._audit_dir_cached is _SENTINEL:
            self._audit_dir_cached = self._base_dir / "audit"
        return self._audit_dir_cached

    def ensure_dirs(self) -> None:
        """
        FIX #11: Create all directories explicitly.
        Call once at startup, not as a side effect of reading config.
        """
        for d in (
            self.data_dir,
            self.model_dir,
            self.log_dir,
            self.cache_dir,
            self.audit_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def set_model_dir(self, path: str) -> None:
        """Override model directory path — invalidates cache."""
        with self._lock:
            self._model_dir_override = str(path) if path else None
            self._model_dir_cached = _SENTINEL
            self._model_dir_cached_override = _SENTINEL

    # ==================== LOADING ====================

    def _load(self) -> None:
        """
        Load configuration from file and environment.

        FIX LOCK: NOT locked here — called from __init__ which is
        already serialized by _instance_lock. Acquiring self._lock
        here would be redundant and could cause issues if _apply_dict
        triggers __getattr__ during init.
        """
        if self._config_file.exists():
            try:
                with open(self._config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._apply_dict(data, _from_init=True)
            except Exception as e:
                _log.warning("Failed to load config file: %s", e)

        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load from environment variables."""
        env_mappings = {
            "CAPITAL": ("capital", float),
            "TRADING_MODE": (
                "trading_mode",
                lambda x: TradingMode(x.lower()),
            ),
            "RISK_PROFILE": (
                "risk_profile",
                lambda x: RiskProfile(x.lower()),
            ),
            "MAX_POSITION_PCT": ("risk.max_position_pct", float),
            "MAX_DAILY_LOSS_PCT": ("risk.max_daily_loss_pct", float),
            "AUTO_TRADE_ENABLED": (
                "auto_trade.enabled",
                lambda x: x.lower() in ("true", "1", "yes"),
            ),
        }

        for env_key, (attr_path, converter) in env_mappings.items():
            full_key = f"{self._env_prefix}{env_key}"
            value = os.environ.get(full_key)
            if value is not None and value.strip():
                try:
                    self._set_nested(attr_path, converter(value.strip()))
                except Exception as e:
                    # FIX #3: Log instead of silently ignoring
                    _log.warning(
                        "Failed to apply env %s=%r: %s", full_key, value, e
                    )

    def _apply_dict(self, data: Dict, _from_init: bool = False) -> None:
        """
        Apply dictionary to config with type checking.

        Args:
            data: Configuration dictionary
            _from_init: If True, skip lock acquisition (caller holds init lock)
        """
        if not _from_init:
            self._lock.acquire()

        try:
            self._apply_dict_inner(data)
        finally:
            if not _from_init:
                self._lock.release()

    def _apply_dict_inner(self, data: Dict) -> None:
        """Inner implementation of _apply_dict without locking."""
        # Sub-config dataclasses
        sub_configs = {
            "data": self.data,
            "model": self.model,
            "trading": self.trading,
            "risk": self.risk,
            "security": self.security,
            "alerts": self.alerts,
            "auto_trade": self.auto_trade,
        }

        for key, value in data.items():
            # Handle sub-config dicts
            if key in sub_configs:
                if isinstance(value, dict):
                    warnings_list = _safe_dataclass_from_dict(
                        sub_configs[key], value
                    )
                    for w in warnings_list:
                        _log.warning("Config %s: %s", key, w)
                else:
                    _log.warning(
                        "Expected dict for '%s', got %s — ignored",
                        key,
                        type(value).__name__,
                    )
                continue

            # Handle enums
            if key == "trading_mode" and isinstance(value, str):
                try:
                    self.trading_mode = TradingMode(value.lower())
                except ValueError as e:
                    _log.warning("Bad trading_mode %r: %s", value, e)
                continue

            if key == "risk_profile" and isinstance(value, str):
                try:
                    self.risk_profile = RiskProfile(value.lower())
                except ValueError as e:
                    _log.warning("Bad risk_profile %r: %s", value, e)
                continue

            if key == "market_type" and isinstance(value, str):
                try:
                    self.market_type = MarketType(value.lower())
                except ValueError as e:
                    _log.warning("Bad market_type %r: %s", value, e)
                continue

            # Handle known top-level attributes with type checking
            if hasattr(self, key) and not key.startswith("_"):
                current = getattr(self, key)
                try:
                    # FIX: Check bool before int (bool is subclass of int)
                    if isinstance(current, bool) and isinstance(value, (bool, int)):
                        setattr(self, key, bool(value))
                    elif isinstance(current, float) and isinstance(
                        value, (int, float)
                    ):
                        setattr(self, key, float(value))
                    elif isinstance(current, int) and isinstance(
                        value, (int, float)
                    ):
                        setattr(self, key, int(value))
                    elif isinstance(current, str) and isinstance(value, str):
                        setattr(self, key, value)
                    elif isinstance(current, list) and isinstance(value, list):
                        setattr(self, key, value)
                    else:
                        _log.warning(
                            "Type mismatch for '%s': expected %s, got %s",
                            key,
                            type(current).__name__,
                            type(value).__name__,
                        )
                except Exception as e:
                    _log.warning("Failed to set '%s': %s", key, e)
            else:
                _log.debug("Unknown config key '%s' — ignored", key)

    def _set_nested(self, path: str, value: Any) -> None:
        """Set nested attribute like 'risk.max_position_pct'."""
        parts = path.split(".")
        obj = self
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    # ==================== VALIDATION ====================

    def _validate(self) -> None:
        """
        FIX #1: Validate configuration without raising.
        Collects warnings accessible via self.validation_warnings.
        Only raises on truly fatal errors (capital <= 0 in LIVE mode).

        FIX VALIDATE: Additional checks for threshold consistency.
        FIX RATIO: Auto-correct split ratios that don't sum to 1.0.
        """
        self._validation_warnings.clear()

        if self.capital <= 0:
            self._validation_warnings.append("Capital must be positive")

        if not (0 < self.model.train_ratio < 1):
            self._validation_warnings.append("Invalid train ratio")

        if self.model.embargo_bars < self.model.prediction_horizon:
            self._validation_warnings.append(
                "Embargo must be >= prediction horizon"
            )

        # FIX VALIDATE: Check threshold relationship
        if self.model.down_threshold >= 0:
            self._validation_warnings.append(
                f"down_threshold should be negative, got {self.model.down_threshold}"
            )
        if self.model.up_threshold <= 0:
            self._validation_warnings.append(
                f"up_threshold should be positive, got {self.model.up_threshold}"
            )
        if self.model.down_threshold >= self.model.up_threshold:
            self._validation_warnings.append(
                f"down_threshold ({self.model.down_threshold}) must be < "
                f"up_threshold ({self.model.up_threshold})"
            )

        # FIX RATIO: Check and auto-correct split ratios
        ratio_sum = (
            self.model.train_ratio
            + self.model.val_ratio
            + self.model.test_ratio
        )
        if abs(ratio_sum - 1.0) >= 0.001:
            self._validation_warnings.append(
                f"Split ratios must sum to 1.0, got {ratio_sum:.4f} — "
                f"auto-correcting test_ratio"
            )
            # Auto-correct: adjust test_ratio to make sum = 1.0
            corrected_test = 1.0 - self.model.train_ratio - self.model.val_ratio
            if corrected_test > 0:
                self.model.test_ratio = round(corrected_test, 4)
                _log.info(
                    f"Auto-corrected test_ratio to {self.model.test_ratio}"
                )
            else:
                self._validation_warnings.append(
                    "Cannot auto-correct: train_ratio + val_ratio >= 1.0"
                )

        # Auto-trade validation
        if self.auto_trade.enabled:
            if self.auto_trade.min_confidence < 0.5:
                self._validation_warnings.append(
                    "Auto-trade min_confidence should be >= 0.5 for safety"
                )
            if self.auto_trade.max_auto_positions > self.risk.max_positions:
                self._validation_warnings.append(
                    "Auto-trade max_positions exceeds risk max_positions"
                )
            if self.auto_trade.max_auto_position_pct > self.risk.max_position_pct:
                self._validation_warnings.append(
                    "Auto-trade max_position_pct exceeds risk max_position_pct"
                )

        for w in self._validation_warnings:
            _log.warning("Config validation: %s", w)

        # Only raise for fatal issues in live mode
        if (
            self.trading_mode == TradingMode.LIVE
            and self._validation_warnings
        ):
            raise ValueError(
                f"Configuration errors (LIVE mode): "
                f"{'; '.join(self._validation_warnings)}"
            )

    @property
    def validation_warnings(self) -> List[str]:
        """Access validation warnings without re-validating."""
        return list(self._validation_warnings)

    # ==================== SAVE / RELOAD ====================

    def save(self) -> None:
        """
        FIX #4: Save ALL configuration including sub-configs.
        FIX SAVE: Use atomic_write_json for consistency with other writers.
        """
        with self._lock:
            data = {
                "capital": self.capital,
                "trading_mode": self.trading_mode.value,
                "risk_profile": self.risk_profile.value,
                "market_type": self.market_type.value,
                "stock_pool": self.stock_pool,
                "broker_path": self.broker_path,
                "min_stocks_for_training": self.min_stocks_for_training,
                "auto_learn_epochs": self.auto_learn_epochs,
                # Sub-configs
                "data": _dataclass_to_dict(self.data),
                "model": _dataclass_to_dict(self.model),
                "trading": _dataclass_to_dict(self.trading),
                "risk": _dataclass_to_dict(self.risk),
                "security": _dataclass_to_dict(self.security),
                "alerts": _dataclass_to_dict(self.alerts),
                "auto_trade": _dataclass_to_dict(self.auto_trade),
            }

        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                from utils.atomic_io import atomic_write_json
                atomic_write_json(
                    self._config_file, data, indent=2, use_lock=True
                )
            except ImportError:
                # Fallback: manual atomic write with resilient fsync
                tmp_path = self._config_file.with_suffix(".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
                tmp_path.replace(self._config_file)

        except Exception as e:
            _log.error("Failed to save config: %s", e)

    def reload(self) -> None:
        """
        FIX #5: Hot-reload — reset sub-configs to defaults, then re-apply.
        FIX RELOAD: Invalidate path caches after reset.
        """
        with self._lock:
            # Reset sub-configs to defaults
            self.data = DataConfig()
            self.model = ModelConfig()
            self.trading = TradingConfig()
            self.risk = RiskConfig()
            self.security = SecurityConfig()
            self.alerts = AlertConfig()
            self.auto_trade = AutoTradeConfig()

            # FIX RELOAD: Invalidate all path caches
            self._data_dir_cached = _SENTINEL
            self._model_dir_cached = _SENTINEL
            self._model_dir_cached_override = _SENTINEL
            self._log_dir_cached = _SENTINEL
            self._cache_dir_cached = _SENTINEL
            self._audit_dir_cached = _SENTINEL

            # Re-load from file/env
            if self._config_file.exists():
                try:
                    with open(self._config_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._apply_dict_inner(data)
                except Exception as e:
                    _log.warning("Failed to reload config file: %s", e)

            self._load_from_env()
            self._validate()

    # ==================== MARKET HOURS ====================

    def is_market_open(self) -> bool:
        """
        FIX #10: Robust market-open check with proper fallback.
        """
        try:
            from zoneinfo import ZoneInfo

            now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
        except Exception:
            now = datetime.now()

        # Weekend check (always available)
        if now.weekday() >= 5:
            return False

        # Try holiday calendar if available
        try:
            from core.constants import is_trading_day

            if not is_trading_day(now.date()):
                return False
        except ImportError:
            pass  # No holiday calendar — weekday check is our best effort

        current_time = now.time()
        t = self.trading

        morning = t.market_open_am <= current_time <= t.market_close_am
        afternoon = t.market_open_pm <= current_time <= t.market_close_pm

        return morning or afternoon

    # ==================== UTILITIES ====================

    def get_min_data_required(self) -> int:
        """Get minimum data points required for training."""
        return (
            self.model.sequence_length
            + self.model.prediction_horizon
            + self.model.embargo_bars
            + 50  # Buffer
        )

    def __repr__(self) -> str:
        return (
            f"Config(mode={self.trading_mode.value}, "
            f"capital={self.capital}, "
            f"risk={self.risk_profile.value}, "
            f"auto_trade={'ON' if self.auto_trade.enabled else 'OFF'})"
        )


# Global config instance
CONFIG = Config()