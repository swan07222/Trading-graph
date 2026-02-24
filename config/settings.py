# config/settings.py
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from pathlib import Path
from typing import Any

from config.settings_utils import (
    _coerce_bool,
    _dataclass_to_dict,
    _safe_dataclass_from_dict,
)

_SENTINEL = object()

# Minimal logger that doesn't depend on our logger module
# (avoids circular import: settings 鈫?logger 鈫?settings)
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
    poll_interval_seconds: float = 1.0
    # Cleaning defaults prioritize preserving original bar truth.
    truth_preserving_cleaning: bool = True
    # Aggressive intraday repair can modify bars and should stay opt-in.
    aggressive_intraday_repair: bool = False
    # Synthetic intraday timestamps are disabled by default.
    synthesize_intraday_index: bool = False
    # Session cache retention controls (CSV-per-symbol compaction).
    session_cache_retention_days: int = 45
    session_cache_max_rows_per_symbol: int = 12000
    session_cache_compact_every_writes: int = 240
    session_cache_max_file_mb: float = 8.0
    # Daily-history quorum gate before writing internet bars into local DB.
    history_quorum_required_sources: int = 2
    history_quorum_tolerance_bps: float = 80.0
    history_quorum_min_ratio: float = 0.55
    
    # China network optimization
    china_network_optimized: bool = True
    china_endpoint_probe_interval: int = 120  # seconds
    china_proxy_enabled: bool = False
    china_proxy_url: str = ""
    china_dns_servers: list[str] = field(default_factory=lambda: [
        "114.114.114.114",
        "223.5.5.5",
        "119.29.29.29",
    ])
    # Provider priority for China (higher = preferred)
    china_provider_priority: dict[str, float] = field(default_factory=lambda: {
        "eastmoney": 1.2,
        "jin10": 1.3,
        "sina": 1.1,
        "tencent": 1.1,
        "xueqiu": 1.0,
    })


@dataclass
class ModelConfig:
    """ML model configuration."""

    sequence_length: int = 60
    hidden_size: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.3
    num_classes: int = 3

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.0005
    early_stop_patience: int = 15
    weight_decay: float = 0.01

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    prediction_horizon: int = 5
    up_threshold: float = 2.0
    down_threshold: float = -2.0
    embargo_bars: int = 10
    min_confidence: float = 0.70  # Increased to 70% for higher accuracy predictions
    # Require checksum sidecars for model/scaler artifacts at load time.
    require_artifact_checksum: bool = True
    # Unsafe legacy checkpoint/pickle fallback is opt-in only.
    allow_unsafe_artifact_load: bool = False

    strong_buy_threshold: float = 0.72
    buy_threshold: float = 0.60
    sell_threshold: float = 0.60
    strong_sell_threshold: float = 0.72


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
    enable_multi_venue: bool = False
    venue_priority: list[str] = field(default_factory=list)
    venue_failover_cooldown_seconds: int = 30
    broker_plugin: str = ""
    broker_plugin_kwargs: dict[str, Any] = field(default_factory=dict)


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
    min_expected_edge_pct: float = 0.30
    min_risk_reward_ratio: float = 1.25
    max_position_scale: float = 1.25
    quote_staleness_seconds: float = 5.0
    max_quote_deviation_bps: float = 80.0

    circuit_breaker_loss_pct: float = 5.0
    circuit_breaker_duration_minutes: int = 60
    max_orders_per_minute: int = 10
    max_orders_per_day: int = 100

    kill_switch_loss_pct: float = 8.0
    kill_switch_drawdown_pct: float = 20.0


@dataclass
class SecurityConfig:
    """Security configuration."""

    encrypt_credentials: bool = True
    audit_logging: bool = True
    audit_hash_chain: bool = True
    require_2fa_for_live: bool = True
    two_factor_ttl_minutes: int = 30
    require_live_trade_permission: bool = True
    strict_live_governance: bool = True
    min_live_approvals: int = 2
    block_trading_when_unhealthy: bool = True
    block_trading_when_degraded: bool = False
    auto_pause_auto_trader_on_degraded: bool = True
    max_session_hours: int = 8
    ip_whitelist: list[str] = field(default_factory=list)
    audit_retention_days: int = 365
    audit_auto_prune: bool = True
    enable_runtime_lease: bool = True
    runtime_lease_backend: str = "sqlite"
    runtime_lease_cluster: str = "execution_engine"
    runtime_lease_path: str = ""
    runtime_lease_node_id: str = ""
    runtime_lease_ttl_seconds: float = 20.0


@dataclass
class AlertConfig:
    """Alerting configuration."""

    enabled: bool = True
    email_enabled: bool = False
    email_recipients: list[str] = field(default_factory=list)
    smtp_server: str = ""
    smtp_port: int = 587
    sms_enabled: bool = False
    webhook_enabled: bool = False
    webhook_url: str = ""

    large_loss_alert_pct: float = 2.0
    position_concentration_alert_pct: float = 20.0
    connection_loss_alert_seconds: int = 30

    from_email: str = ""
    smtp_username: str = ""
    smtp_password_key: str = "smtp_password"


@dataclass
class AutoTradeConfig:
    """Auto-trading configuration.

    Controls the autonomous trading engine that can execute trades
    without manual confirmation when enabled.
    """

    # Master enable/disable
    enabled: bool = False

    # Signal filters 鈥?minimum thresholds for auto-execution
    min_confidence: float = 0.78
    min_signal_strength: float = 0.70
    min_model_agreement: float = 0.72

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

    scan_interval_seconds: int = 60
    cooldown_after_trade_seconds: int = 300
    max_trades_per_day: int = 10
    max_trades_per_stock_per_day: int = 2

    require_market_open: bool = True
    require_broker_connected: bool = True
    pause_on_high_volatility: bool = True
    volatility_pause_threshold: float = 5.0  # ATR% above this pauses

    # Auto-stop loss management
    auto_stop_loss: bool = True
    trailing_stop_enabled: bool = False
    trailing_stop_pct: float = 3.0

    notify_on_trade: bool = True
    notify_on_skip: bool = False

    # Paper trading safety 鈥?require explicit confirmation for live
    confirm_live_auto_trade: bool = True
    # Reject delayed/fallback realtime quotes at submission time.
    block_on_stale_realtime: bool = True
    # Auto-disable live auto-trade when drift guard raises an alarm.
    auto_disable_on_model_drift: bool = True
    model_drift_pause_seconds: int = 3600

    # FIX #16: Add __post_init__ to automatically validate on instantiation
    def __post_init__(self) -> None:
        """Automatically validate configuration on instantiation.
        
        FIX #16: Ensures invalid auto-trade settings are caught early
        rather than causing unexpected behavior at runtime.
        """
        errors = self.validate()
        if errors:
            # Log warnings but don't raise — allows partial configs to load
            from utils.logger import get_logger
            log = get_logger(__name__)
            for error in errors:
                log.warning(f"AutoTradeConfig validation warning: {error}")
            # Store validation errors for later inspection
            object.__setattr__(self, '_validation_errors', errors)

    # FIX #16: Add validation method for configuration consistency
    def validate(self) -> list[str]:
        """Validate auto-trade configuration consistency.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        # Validate confidence thresholds are in valid range (0, 1]
        if not (0.0 < self.min_confidence <= 1.0):
            errors.append(f"min_confidence must be in (0, 1], got {self.min_confidence}")
        if not (0.0 < self.min_signal_strength <= 1.0):
            errors.append(f"min_signal_strength must be in (0, 1], got {self.min_signal_strength}")
        if not (0.0 < self.min_model_agreement <= 1.0):
            errors.append(f"min_model_agreement must be in (0, 1], got {self.min_model_agreement}")

        # Validate position limits are positive
        if self.max_auto_positions <= 0:
            errors.append(f"max_auto_positions must be positive, got {self.max_auto_positions}")
        if self.max_auto_position_pct <= 0:
            errors.append(f"max_auto_position_pct must be positive, got {self.max_auto_position_pct}")
        if self.max_auto_order_value <= 0:
            errors.append(f"max_auto_order_value must be positive, got {self.max_auto_order_value}")

        # Validate timing constraints
        if self.scan_interval_seconds < 10:
            errors.append(f"scan_interval_seconds must be >= 10, got {self.scan_interval_seconds}")
        if self.cooldown_after_trade_seconds < 60:
            errors.append(f"cooldown_after_trade_seconds must be >= 60, got {self.cooldown_after_trade_seconds}")
        if self.max_trades_per_day <= 0:
            errors.append(f"max_trades_per_day must be positive, got {self.max_trades_per_day}")

        # Validate volatility threshold
        if self.volatility_pause_threshold <= 0:
            errors.append(f"volatility_pause_threshold must be positive, got {self.volatility_pause_threshold}")

        # Validate trailing stop
        if self.trailing_stop_enabled and self.trailing_stop_pct <= 0:
            errors.append(f"trailing_stop_pct must be positive when enabled, got {self.trailing_stop_pct}")

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


@dataclass
class PrecisionConfig:
    """Precision-oriented controls for higher hit-rate/lower-frequency operation.

    All options are optional and default-safe (disabled) so existing workflows
    keep behaving the same unless explicitly enabled.
    """

    enabled: bool = False

    # Predictor runtime gating
    min_confidence: float = 0.78
    min_agreement: float = 0.72
    max_entropy: float = 0.35
    min_edge: float = 0.14

    # Regime-aware threshold routing (trend/range/high-vol)
    regime_routing: bool = True
    range_confidence_boost: float = 0.06
    high_vol_confidence_boost: float = 0.08
    high_vol_atr_pct: float = 0.035  # 3.5%

    # Training labels: make neutral band cost-aware.
    profit_aware_labels: bool = False
    label_cost_buffer_pct: float = 0.20
    min_label_edge_pct: float = 0.25

    # Auto-trader: optionally allow only strong signals.
    force_strong_signals_auto_trade: bool = True
    block_auto_trade_on_short_history_fallback: bool = True
    fail_closed_on_quality_gate_error: bool = True

    # Auto-learner threshold tuning
    enable_threshold_tuning: bool = True
    min_tuning_samples: int = 12
    tuning_min_trade_rate: float = 0.03
    tuning_max_candidates_per_axis: int = 7

    # Auto-learner holdout acceptance (adaptive, regime-aware).
    validation_min_predictions: int = 5
    validation_min_accept_lb: float = 0.30
    validation_max_accuracy_degradation: float = 0.15
    validation_max_confidence_degradation: float = 0.18
    validation_confidence_z: float = 1.64
    validation_confidence_margin: float = 0.03
    validation_max_train_holdout_gap: float = 0.40
    validation_high_vol_return_pct: float = 1.2
    validation_low_signal_edge: float = 0.10
    validation_high_vol_relax: float = 0.05
    validation_low_signal_tighten: float = 0.04

    # Persistence for learned threshold profile
    profile_filename: str = "precision_thresholds.json"


class Config:
    """Production-grade configuration manager.

    Usage:
        from config.settings import CONFIG
        print(CONFIG.model.learning_rate)
        print(CONFIG.LEARNING_RATE)  # Legacy alias

    Ensure config/__init__.py contains:
        from config.settings import CONFIG, TradingMode, RiskProfile, MarketType
        __all__ = ["CONFIG", "TradingMode", "RiskProfile", "MarketType"]
    """

    _instance: Config | None = None
    _instance_lock = threading.RLock()

    def __new__(cls) -> Config:
        """Thread-safe singleton creation with double-checked locking.

        Uses RLock for reentrant safety and explicit memory barriers
        through the lock to prevent instruction reordering issues.
        
        FIX C3: Set _initialized to True here in __new__ to prevent
        race condition where two threads could both pass the check in
        __init__ and both attempt initialization.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    # FIX: Set _initialized to True immediately to prevent
                    # race condition during concurrent initialization
                    object.__setattr__(inst, '_initialized', True)
                    object.__setattr__(inst, '_lock', threading.RLock())
                    cls._instance = inst
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Destroy singleton for testing.
        Call between tests to get a fresh Config.
        """
        with cls._instance_lock:
            if cls._instance is not None:
                # Clean up any resources before resetting
                try:
                    cls._instance._lock.release()
                except RuntimeError:
                    pass  # Lock was not held
            cls._instance = None

    def __init__(self) -> None:
        """Thread-safe initialization with explicit barrier.

        FIX C3: Now uses object.__getattribute__ to safely check
        _initialized flag, preventing race condition where two threads
        could both pass the check and both attempt initialization.
        """
        # Early exit if already initialized - use object.__getattribute__
        # to avoid triggering __getattr__ during partial initialization
        try:
            initialized = object.__getattribute__(self, '_initialized')
            if initialized and hasattr(self, '_lock'):
                # Check if fully initialized by checking for data attribute
                if hasattr(self, 'data'):
                    return
        except AttributeError:
            # _initialized not set yet, proceed with initialization
            pass

        # Acquire lock for initialization
        with self._lock:
            # Double-check after acquiring lock
            if hasattr(self, 'data'):
                return

            # Mark as initialized
            object.__setattr__(self, '_initialized', True)
            self._config_file = Path(__file__).parent.parent / "config.json"
            self._env_prefix = "TRADING_"
            self._validation_warnings: list[str] = []

            # Sub-configs
            self.data = DataConfig()
            self.model = ModelConfig()
            self.trading = TradingConfig()
            self.risk = RiskConfig()
            self.security = SecurityConfig()
            self.alerts = AlertConfig()
            self.auto_trade = AutoTradeConfig()
            self.precision = PrecisionConfig()

            self.capital: float = 100_000.0
            self.trading_mode: TradingMode = TradingMode.SIMULATION
            self.risk_profile: RiskProfile = RiskProfile.MODERATE
            self.market_type: MarketType = MarketType.A_SHARE
            self.broker_path: str = ""
            self.runtime_checkpoint_seconds: float = 5.0
            self.runtime_watchdog_stall_seconds: float = 25.0
            self.runtime_lease_heartbeat_seconds: float = 5.0

            self._base_dir = Path(__file__).parent.parent
            self._model_dir_override: str | None = None

            # Sentinel-based caching for path properties
            # Use `object` as sentinel type to avoid Any propagation
            self._data_dir_cached: Path | object = _SENTINEL
            self._model_dir_cached: Path | object = _SENTINEL
            self._model_dir_cached_override: str | None | object = _SENTINEL
            self._log_dir_cached: Path | object = _SENTINEL
            self._cache_dir_cached: Path | object = _SENTINEL
            self._audit_dir_cached: Path | object = _SENTINEL

            self.stock_pool: list[str] = [
                "600519",
                "601318",
                "600036",
                "000858",
                "600900",
                "002594",
                "300750",
                "002475",
                "300059",
                "002230",
                "000333",
                "000651",
                "600887",
                "603288",
                "600276",
                "300760",
                "300015",
                "601166",
                "601398",
                "600030",
            ]

            self.min_stocks_for_training: int = 5
            self.auto_learn_epochs: int = 50

            self._load()
            self._validate()

    # ==================== LEGACY COMPATIBILITY ====================

    _LEGACY_MAP: dict[str, str] = {
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
        "COMMISSION": "trading.commission",
        "STAMP_TAX": "trading.stamp_tax",
        "SLIPPAGE": "trading.slippage",
        "LOT_SIZE": "trading.lot_size",
        "MAX_POSITION_PCT": "risk.max_position_pct",
        "MAX_DAILY_LOSS_PCT": "risk.max_daily_loss_pct",
        "MAX_POSITIONS": "risk.max_positions",
        "RISK_PER_TRADE": "risk.risk_per_trade_pct",
        "QUOTE_STALENESS_SECONDS": "risk.quote_staleness_seconds",
        "POLL_INTERVAL_SECONDS": "data.poll_interval_seconds",
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
        """Legacy compatibility: CONFIG.SEQUENCE_LENGTH 鈫?CONFIG.model.sequence_length.

        Only called when normal attribute lookup fails,
        so @property definitions always take precedence.
        """
        try:
            initialized = object.__getattribute__(self, "_initialized")
        except AttributeError as exc:
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute {name!r}"
            ) from exc

        if not initialized:
            raise AttributeError(
                f"{self.__class__.__name__} not yet initialized, "
                f"cannot access {name!r}"
            )

        mapping = Config._LEGACY_MAP.get(name)
        if mapping is None:
            if name == "STOCK_POOL":
                return self.stock_pool
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute {name!r}"
            )

        # Path properties 鈥?delegate to the property
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
        value = float(value)
        if value <= 0:
            _log.warning(
                f"Capital must be positive, got {value} 鈥?clamping to 1.0"
            )
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
        return self._data_dir_cached  # type: ignore[return-value]

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
        return self._model_dir_cached  # type: ignore[return-value]

    @property
    def log_dir(self) -> Path:
        if self._log_dir_cached is _SENTINEL:
            self._log_dir_cached = self._base_dir / "logs"
        return self._log_dir_cached  # type: ignore[return-value]

    @property
    def cache_dir(self) -> Path:
        if self._cache_dir_cached is _SENTINEL:
            self._cache_dir_cached = self._base_dir / "cache"
        return self._cache_dir_cached  # type: ignore[return-value]

    @property
    def audit_dir(self) -> Path:
        if self._audit_dir_cached is _SENTINEL:
            self._audit_dir_cached = self._base_dir / "audit"
        return self._audit_dir_cached  # type: ignore[return-value]

    def ensure_dirs(self) -> None:
        """Create all directories explicitly. Call once at startup."""
        for d in (
            self.data_dir,
            self.model_dir,
            self.log_dir,
            self.cache_dir,
            self.audit_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def set_model_dir(self, path: str) -> None:
        """Override model directory path 鈥?invalidates cache."""
        with self._lock:
            self._model_dir_override = str(path) if path else None
            self._model_dir_cached = _SENTINEL
            self._model_dir_cached_override = _SENTINEL

    # ==================== LOADING ====================

    def _load(self) -> None:
        """Load configuration from file and environment."""
        if self._config_file.exists():
            try:
                with open(self._config_file, encoding="utf-8") as f:
                    data = json.load(f)
                self._apply_dict(data, _from_init=True)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as e:
                _log.warning("Failed to load config file: %s", e)

        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load from environment variables."""

        def _env_bool(raw: str) -> bool:
            ok, parsed = _coerce_bool(raw)
            if not ok:
                raise ValueError(f"invalid boolean value: {raw!r}")
            return parsed

        env_mappings = {
            "CAPITAL": ("capital", float),
            "TRADING_MODE": (
                "trading_mode",
                lambda x: TradingMode(x.strip().lower()),
            ),
            "RISK_PROFILE": (
                "risk_profile",
                lambda x: RiskProfile(x.strip().lower()),
            ),
            "MAX_POSITION_PCT": ("risk.max_position_pct", float),
            "MAX_DAILY_LOSS_PCT": ("risk.max_daily_loss_pct", float),
            "AUTO_TRADE_ENABLED": (
                "auto_trade.enabled",
                _env_bool,
            ),
            "ALLOW_UNSAFE_ARTIFACT_LOAD": (
                "model.allow_unsafe_artifact_load",
                _env_bool,
            ),
            "REQUIRE_ARTIFACT_CHECKSUM": (
                "model.require_artifact_checksum",
                _env_bool,
            ),
            "RUNTIME_LEASE_ENABLED": (
                "security.enable_runtime_lease",
                _env_bool,
            ),
            "RUNTIME_LEASE_BACKEND": (
                "security.runtime_lease_backend",
                str,
            ),
            "RUNTIME_LEASE_CLUSTER": (
                "security.runtime_lease_cluster",
                str,
            ),
            "RUNTIME_LEASE_PATH": ("security.runtime_lease_path", str),
            "RUNTIME_LEASE_NODE_ID": (
                "security.runtime_lease_node_id",
                str,
            ),
            "RUNTIME_LEASE_TTL_SECONDS": (
                "security.runtime_lease_ttl_seconds",
                float,
            ),
            "RUNTIME_CHECKPOINT_SECONDS": (
                "runtime_checkpoint_seconds",
                float,
            ),
            "RUNTIME_WATCHDOG_STALL_SECONDS": (
                "runtime_watchdog_stall_seconds",
                float,
            ),
            "RUNTIME_LEASE_HEARTBEAT_SECONDS": (
                "runtime_lease_heartbeat_seconds",
                float,
            ),
        }

        for env_key, (attr_path, converter) in env_mappings.items():
            full_key = f"{self._env_prefix}{env_key}"
            value = os.environ.get(full_key)
            if value is not None and value.strip():
                try:
                    self._set_nested(attr_path, converter(value.strip()))  # type: ignore[operator]
                except (AttributeError, TypeError, ValueError) as e:
                    _log.warning(
                        "Failed to apply env %s=%r: %s",
                        full_key,
                        value,
                        e,
                    )

    def _apply_dict(self, data: dict, _from_init: bool = False) -> None:
        """Apply dictionary to config with type checking."""
        if not _from_init:
            self._lock.acquire()

        try:
            self._apply_dict_inner(data)
        finally:
            if not _from_init:
                self._lock.release()

    def _apply_dict_inner(self, data: dict) -> None:
        """Inner implementation of _apply_dict without locking."""
        sub_configs = {
            "data": self.data,
            "model": self.model,
            "trading": self.trading,
            "risk": self.risk,
            "security": self.security,
            "alerts": self.alerts,
            "auto_trade": self.auto_trade,
            "precision": self.precision,
        }

        for key, value in data.items():
            if key in sub_configs:
                if isinstance(value, dict):
                    warnings_list = _safe_dataclass_from_dict(
                        sub_configs[key], value
                    )
                    for w in warnings_list:
                        _log.warning("Config %s: %s", key, w)
                else:
                    _log.warning(
                        "Expected dict for '%s', got %s 鈥?ignored",
                        key,
                        type(value).__name__,
                    )
                continue

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

            if hasattr(self, key) and not key.startswith("_"):
                current = getattr(self, key)
                try:
                    if isinstance(current, bool):
                        ok, parsed = _coerce_bool(value)
                        if ok:
                            setattr(self, key, parsed)
                        else:
                            _log.warning(
                                "Bad value for bool '%s': %r",
                                key,
                                value,
                            )
                    elif isinstance(current, float) and isinstance(
                        value, (int, float)
                    ):
                        setattr(self, key, float(value))
                    elif isinstance(current, int) and isinstance(
                        value, (int, float)
                    ):
                        setattr(self, key, int(value))
                    elif isinstance(current, str) and isinstance(
                        value, str
                    ):
                        setattr(self, key, value)
                    elif isinstance(current, list) and isinstance(
                        value, list
                    ):
                        setattr(self, key, value)
                    else:
                        _log.warning(
                            "Type mismatch for '%s': expected %s, got %s",
                            key,
                            type(current).__name__,
                            type(value).__name__,
                        )
                except (AttributeError, TypeError, ValueError) as e:
                    _log.warning("Failed to set '%s': %s", key, e)
            else:
                _log.debug("Unknown config key '%s' 鈥?ignored", key)

    def _set_nested(self, path: str, value: Any) -> None:
        """Set nested attribute like 'risk.max_position_pct'."""
        parts = path.split(".")
        obj = self
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    # ==================== VALIDATION ====================

    def _validate(self) -> None:
        """Validate configuration without raising (except LIVE mode).
        Collects warnings accessible via self.validation_warnings.
        """
        self._validation_warnings.clear()

        if self.capital <= 0:
            self._validation_warnings.append("Capital must be positive")

        if self.data.history_quorum_required_sources < 2:
            self._validation_warnings.append(
                "data.history_quorum_required_sources should be >= 2"
            )
        if self.data.history_quorum_tolerance_bps <= 0:
            self._validation_warnings.append(
                "data.history_quorum_tolerance_bps should be > 0"
            )
        if not (0.0 < self.data.history_quorum_min_ratio <= 1.0):
            self._validation_warnings.append(
                "data.history_quorum_min_ratio should be in (0, 1]"
            )

        if not (0 < self.model.train_ratio < 1):
            self._validation_warnings.append("Invalid train ratio")

        if self.model.embargo_bars <= self.model.prediction_horizon:
            self._validation_warnings.append(
                "Embargo must be > prediction horizon (not >=)"
            )

        if bool(self.model.allow_unsafe_artifact_load):
            self._validation_warnings.append(
                "model.allow_unsafe_artifact_load=true weakens artifact security"
            )

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

        ratio_sum = (
            self.model.train_ratio
            + self.model.val_ratio
            + self.model.test_ratio
        )
        if abs(ratio_sum - 1.0) >= 0.001:
            self._validation_warnings.append(
                f"Split ratios must sum to 1.0, got {ratio_sum:.4f} 鈥?"
                f"auto-correcting test_ratio"
            )
            corrected_test = (
                1.0 - self.model.train_ratio - self.model.val_ratio
            )
            if corrected_test > 0:
                self.model.test_ratio = round(corrected_test, 4)
                _log.info(
                    f"Auto-corrected test_ratio to {self.model.test_ratio}"
                )
            else:
                self._validation_warnings.append(
                    "Cannot auto-correct: train_ratio + val_ratio >= 1.0"
                )

        if self.auto_trade.enabled:
            if self.auto_trade.min_confidence < 0.5:
                self._validation_warnings.append(
                    "Auto-trade min_confidence should be >= 0.5 for safety"
                )
            if (
                self.auto_trade.max_auto_positions
                > self.risk.max_positions
            ):
                self._validation_warnings.append(
                    "Auto-trade max_positions exceeds risk max_positions"
                )
            if (
                self.auto_trade.max_auto_position_pct
                > self.risk.max_position_pct
            ):
                self._validation_warnings.append(
                    "Auto-trade max_position_pct exceeds risk max_position_pct"
                )
            if self.auto_trade.model_drift_pause_seconds < 60:
                self._validation_warnings.append(
                    "auto_trade.model_drift_pause_seconds should be >= 60"
                )

        if (
            self.precision.min_confidence < 0
            or self.precision.min_confidence > 1
        ):
            self._validation_warnings.append(
                f"precision.min_confidence out of range: {self.precision.min_confidence}"
            )
        if (
            self.precision.min_agreement < 0
            or self.precision.min_agreement > 1
        ):
            self._validation_warnings.append(
                f"precision.min_agreement out of range: {self.precision.min_agreement}"
            )
        if (
            self.precision.max_entropy < 0
            or self.precision.max_entropy > 1
        ):
            self._validation_warnings.append(
                f"precision.max_entropy out of range: {self.precision.max_entropy}"
            )
        if (
            self.precision.min_edge < 0
            or self.precision.min_edge > 1
        ):
            self._validation_warnings.append(
                f"precision.min_edge out of range: {self.precision.min_edge}"
            )
        if self.precision.min_tuning_samples < 1:
            self._validation_warnings.append(
                f"precision.min_tuning_samples should be >= 1, got {self.precision.min_tuning_samples}"
            )
        if not (0.0 < self.precision.tuning_min_trade_rate <= 1.0):
            self._validation_warnings.append(
                f"precision.tuning_min_trade_rate out of range: {self.precision.tuning_min_trade_rate}"
            )
        if self.precision.tuning_max_candidates_per_axis < 3:
            self._validation_warnings.append(
                "precision.tuning_max_candidates_per_axis should be >= 3"
            )
        if self.precision.validation_min_predictions < 1:
            self._validation_warnings.append(
                "precision.validation_min_predictions should be >= 1"
            )
        if (
            self.precision.validation_min_accept_lb < 0
            or self.precision.validation_min_accept_lb > 1
        ):
            self._validation_warnings.append(
                f"precision.validation_min_accept_lb out of range: {self.precision.validation_min_accept_lb}"
            )
        if (
            self.precision.validation_max_accuracy_degradation < 0
            or self.precision.validation_max_accuracy_degradation > 1
        ):
            self._validation_warnings.append(
                "precision.validation_max_accuracy_degradation out of range"
            )
        if (
            self.precision.validation_max_confidence_degradation < 0
            or self.precision.validation_max_confidence_degradation > 1
        ):
            self._validation_warnings.append(
                "precision.validation_max_confidence_degradation out of range"
            )
        if (
            self.precision.validation_max_train_holdout_gap < 0
            or self.precision.validation_max_train_holdout_gap > 1
        ):
            self._validation_warnings.append(
                "precision.validation_max_train_holdout_gap out of range"
            )
        if self.precision.validation_confidence_z <= 0:
            self._validation_warnings.append(
                "precision.validation_confidence_z should be > 0"
            )
        if self.precision.validation_confidence_margin < 0:
            self._validation_warnings.append(
                "precision.validation_confidence_margin should be >= 0"
            )
        if self.precision.validation_high_vol_return_pct <= 0:
            self._validation_warnings.append(
                "precision.validation_high_vol_return_pct should be > 0"
            )
        if (
            self.precision.validation_low_signal_edge < 0
            or self.precision.validation_low_signal_edge > 1
        ):
            self._validation_warnings.append(
                "precision.validation_low_signal_edge out of range"
            )
        if self.precision.validation_high_vol_relax < 0:
            self._validation_warnings.append(
                "precision.validation_high_vol_relax should be >= 0"
            )
        if self.precision.validation_low_signal_tighten < 0:
            self._validation_warnings.append(
                "precision.validation_low_signal_tighten should be >= 0"
            )

        for w in self._validation_warnings:
            _log.warning("Config validation: %s", w)

        if (
            self.trading_mode == TradingMode.LIVE
            and self._validation_warnings
        ):
            raise ValueError(
                f"Configuration errors (LIVE mode): "
                f"{'; '.join(self._validation_warnings)}"
            )

    @property
    def validation_warnings(self) -> list[str]:
        """Access validation warnings without re-validating."""
        return list(self._validation_warnings)

    # ==================== SAVE / RELOAD ====================

    def save(self) -> None:
        """Save ALL configuration including sub-configs."""
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
                "precision": _dataclass_to_dict(self.precision),
            }

        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                from utils.atomic_io import atomic_write_json

                atomic_write_json(
                    self._config_file, data, indent=2, use_lock=True
                )
            except ImportError:
                tmp_path = self._config_file.with_suffix(".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
                tmp_path.replace(self._config_file)

        except (OSError, TypeError, ValueError) as e:
            _log.error("Failed to save config: %s", e)

    def reload(self) -> None:
        """Hot-reload 鈥?reset sub-configs to defaults, then re-apply."""
        with self._lock:
            self.data = DataConfig()
            self.model = ModelConfig()
            self.trading = TradingConfig()
            self.risk = RiskConfig()
            self.security = SecurityConfig()
            self.alerts = AlertConfig()
            self.auto_trade = AutoTradeConfig()
            self.precision = PrecisionConfig()

            self._data_dir_cached = _SENTINEL
            self._model_dir_cached = _SENTINEL
            self._model_dir_cached_override = _SENTINEL
            self._log_dir_cached = _SENTINEL
            self._cache_dir_cached = _SENTINEL
            self._audit_dir_cached = _SENTINEL

            if self._config_file.exists():
                try:
                    with open(self._config_file, encoding="utf-8") as f:
                        data = json.load(f)
                    self._apply_dict_inner(data)
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as e:
                    _log.warning("Failed to reload config file: %s", e)

            self._load_from_env()
            self._validate()

    # ==================== MARKET HOURS ====================

    def is_market_open(self) -> bool:
        """FIX: Robust market-open check with proper timezone handling.

        Always compares naive time objects to avoid tz-aware vs tz-naive
        mismatch with TradingConfig time fields.
        """
        try:
            from zoneinfo import ZoneInfo

            now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
        except (ImportError, ModuleNotFoundError):
            now = datetime.now()

        # Weekend check
        if now.weekday() >= 5:
            return False

        try:
            from core.constants import is_trading_day

            if not is_trading_day(now.date()):
                return False
        except Exception as exc:
            _log.debug("Trading-day lookup failed in is_market_open: %s", exc)

        # FIX #18: Always use naive time for comparison with TradingConfig
        # time fields which are naive
        current_time = now.time().replace(tzinfo=None)
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


CONFIG = Config()
