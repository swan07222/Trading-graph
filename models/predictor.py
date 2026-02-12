# models/predictor.py
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import os
import json

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)

class Signal(Enum):
    """Trading signal types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class TradingLevels:
    """Trading price levels"""
    entry: float = 0.0
    stop_loss: float = 0.0
    stop_loss_pct: float = 0.0
    target_1: float = 0.0
    target_1_pct: float = 0.0
    target_2: float = 0.0
    target_2_pct: float = 0.0
    target_3: float = 0.0
    target_3_pct: float = 0.0

@dataclass
class PositionSize:
    """Position sizing information"""
    shares: int = 0
    value: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    expected_edge_pct: float = 0.0
    risk_reward_ratio: float = 0.0

@dataclass
class Prediction:
    """Complete prediction result"""
    stock_code: str
    stock_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    signal: Signal = Signal.HOLD
    signal_strength: float = 0.0
    confidence: float = 0.0

    prob_up: float = 0.33
    prob_neutral: float = 0.34
    prob_down: float = 0.33

    current_price: float = 0.0
    price_history: List[float] = field(default_factory=list)
    predicted_prices: List[float] = field(default_factory=list)

    rsi: float = 50.0
    macd_signal: str = "NEUTRAL"
    trend: str = "NEUTRAL"

    levels: TradingLevels = field(default_factory=TradingLevels)

    position: PositionSize = field(default_factory=PositionSize)

    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    interval: str = "1d"
    horizon: int = 5

    # Extra fields for UI/signal generator
    model_agreement: float = 1.0
    entropy: float = 0.0

    # ATR from features (used internally for levels)
    atr_pct_value: float = 0.02

class Predictor:
    """
    AI Stock Predictor with real-time capabilities.

    Features:
    - Ensemble model predictions
    - Multi-step price forecasting
    - Real-time chart updates
    - Multiple interval support (1m, 5m, 1d, etc.)
    - Prediction caching with configurable TTL
    """

    _CACHE_TTL: float = 5.0
    _MAX_CACHE_SIZE: int = 200

    def __init__(
        self,
        capital: float = None,
        interval: str = "1d",
        prediction_horizon: int = None,
    ):
        self.capital = float(capital or CONFIG.CAPITAL)
        self.interval = str(interval).lower()
        self.horizon = int(prediction_horizon or CONFIG.PREDICTION_HORIZON)

        self._predict_lock = threading.RLock()

        # Components (lazy loaded)
        self.ensemble = None
        self.forecaster = None
        self.processor = None
        self.feature_engine = None
        self.fetcher = None

        self._feature_cols: List[str] = []

        # Prediction cache: code -> (timestamp, Prediction)
        self._pred_cache: Dict[str, Tuple[float, Prediction]] = {}
        self._cache_lock = threading.Lock()

        # Track if constructor params were overridden by model metadata
        self._requested_interval = self.interval
        self._requested_horizon = self.horizon
        self._high_precision = self._load_high_precision_config()

        self._load_models()

    def _load_high_precision_config(self) -> Dict[str, float]:
        """
        Optional high-precision runtime gate.
        Disabled by default so existing behavior is unchanged.
        """
        def _env_bool(name: str, default: bool = False) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return str(raw).strip().lower() in ("1", "true", "yes", "on")

        def _env_float(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if raw is None:
                return float(default)
            try:
                return float(raw)
            except Exception:
                return float(default)

        precision_cfg = getattr(CONFIG, "precision", None)
        enabled_default = bool(getattr(precision_cfg, "enabled", False))
        min_conf_default = max(
            float(CONFIG.model.min_confidence),
            float(getattr(precision_cfg, "min_confidence", 0.70)),
        )
        min_agree_default = max(
            float(getattr(precision_cfg, "min_agreement", 0.65)),
            0.65,
        )
        max_entropy_default = float(getattr(precision_cfg, "max_entropy", 0.45))
        min_edge_default = float(getattr(precision_cfg, "min_edge", 0.10))
        cfg = {
            "enabled": 1.0 if _env_bool("TRADING_HIGH_PRECISION_MODE", enabled_default) else 0.0,
            "min_confidence": _env_float("TRADING_HP_MIN_CONFIDENCE", min_conf_default),
            "min_agreement": _env_float("TRADING_HP_MIN_AGREEMENT", min_agree_default),
            "max_entropy": _env_float("TRADING_HP_MAX_ENTROPY", max_entropy_default),
            "min_edge": _env_float("TRADING_HP_MIN_EDGE", min_edge_default),
            "regime_routing": 1.0 if bool(getattr(precision_cfg, "regime_routing", True)) else 0.0,
            "range_conf_boost": float(getattr(precision_cfg, "range_confidence_boost", 0.04)),
            "high_vol_conf_boost": float(getattr(precision_cfg, "high_vol_confidence_boost", 0.05)),
            "high_vol_atr_pct": float(getattr(precision_cfg, "high_vol_atr_pct", 0.035)),
        }

        # Load learned profile if available; env vars still have final override.
        try:
            profile_name = str(getattr(precision_cfg, "profile_filename", "precision_thresholds.json"))
            profile_path = CONFIG.data_dir / profile_name
            if profile_path.exists():
                data = json.loads(profile_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    prof = data.get("thresholds", data)
                    if isinstance(prof, dict):
                        for key in ("min_confidence", "min_agreement", "max_entropy", "min_edge"):
                            if key in prof:
                                cfg[key] = float(prof[key])
                        cfg["profile_loaded"] = 1.0
        except Exception as e:
            log.debug("High precision profile load failed: %s", e)

        # Final environment override (highest priority)
        cfg["min_confidence"] = _env_float("TRADING_HP_MIN_CONFIDENCE", cfg["min_confidence"])
        cfg["min_agreement"] = _env_float("TRADING_HP_MIN_AGREEMENT", cfg["min_agreement"])
        cfg["max_entropy"] = _env_float("TRADING_HP_MAX_ENTROPY", cfg["max_entropy"])
        cfg["min_edge"] = _env_float("TRADING_HP_MIN_EDGE", cfg["min_edge"])

        if cfg["enabled"] > 0:
            log.info(
                "High Precision Mode enabled: min_conf=%.2f min_agree=%.2f "
                "max_entropy=%.2f min_edge=%.2f",
                cfg["min_confidence"],
                cfg["min_agreement"],
                cfg["max_entropy"],
                cfg["min_edge"],
            )
        return cfg

    # =========================================================================
    # =========================================================================

    def _load_models(self) -> bool:
        """Load all required models with robust fallback."""
        try:
            from data.processor import DataProcessor
            from data.features import FeatureEngine
            from data.fetcher import get_fetcher
            from models.ensemble import EnsembleModel

            self.processor = DataProcessor()
            self.feature_engine = FeatureEngine()
            self.fetcher = get_fetcher()
            self._feature_cols = self.feature_engine.get_feature_columns()

            model_dir = CONFIG.MODEL_DIR

            # Pick best ensemble + scaler pair
            chosen_ens, chosen_scl = self._find_best_model_pair(model_dir)

            if chosen_scl and chosen_scl.exists():
                self.processor.load_scaler(str(chosen_scl))
            else:
                legacy = model_dir / "scaler_1d_5.pkl"
                if legacy.exists():
                    self.processor.load_scaler(str(legacy))
                else:
                    log.warning(
                        "No scaler found — predictions may be inaccurate"
                    )

            self.ensemble = None
            if chosen_ens and chosen_ens.exists():
                input_size = (
                    self.processor.n_features or len(self._feature_cols)
                )
                ens = EnsembleModel(input_size=input_size)
                if ens.load(str(chosen_ens)) and ens.models:
                    self.ensemble = ens

                    loaded_interval = str(
                        getattr(self.ensemble, "interval", self.interval)
                    )
                    loaded_horizon = int(
                        getattr(
                            self.ensemble, "prediction_horizon",
                            self.horizon
                        )
                    )

                    if (
                        loaded_interval != self._requested_interval
                        or loaded_horizon != self._requested_horizon
                    ):
                        log.info(
                            f"Model metadata overrides constructor: "
                            f"interval {self._requested_interval}"
                            f"->{loaded_interval}, "
                            f"horizon {self._requested_horizon}"
                            f"->{loaded_horizon}"
                        )

                    self.interval = loaded_interval
                    self.horizon = loaded_horizon

                    log.info(
                        f"Ensemble loaded from {chosen_ens.name} "
                        f"(interval={self.interval}, "
                        f"horizon={self.horizon})"
                    )
                else:
                    log.warning(f"Failed to load ensemble: {chosen_ens}")
            else:
                log.warning("No ensemble model found")

            # Load forecaster (optional)
            self._load_forecaster()

            return True

        except ImportError as e:
            log.error(f"Missing dependency for models: {e}")
            return False
        except Exception as e:
            log.error(f"Failed to load models: {e}")
            return False

    def _find_best_model_pair(self, model_dir):
        """Find the best ensemble + scaler file pair."""
        from pathlib import Path

        req_ens = model_dir / f"ensemble_{self.interval}_{self.horizon}.pt"
        req_scl = model_dir / f"scaler_{self.interval}_{self.horizon}.pkl"

        if req_ens.exists():
            return req_ens, req_scl if req_scl.exists() else None

        # Fallback 1: common default
        fb_ens = model_dir / "ensemble_1d_5.pt"
        fb_scl = model_dir / "scaler_1d_5.pkl"
        if fb_ens.exists():
            return fb_ens, fb_scl if fb_scl.exists() else None

        # Fallback 2: any available ensemble, prefer matching scaler
        ensembles = sorted(
            model_dir.glob("ensemble_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for ep in ensembles:
            parts = ep.stem.split("_", 2)
            if len(parts) == 3:
                sp = model_dir / f"scaler_{parts[1]}_{parts[2]}.pkl"
                if sp.exists():
                    return ep, sp

        if ensembles:
            return ensembles[0], None

        return None, None

    def _load_forecaster(self):
        """Load TCN forecaster for price curve prediction."""
        try:
            import torch
            from models.networks import TCNModel

            forecast_path = (
                CONFIG.MODEL_DIR
                / f"forecast_{self.interval}_{self.horizon}.pt"
            )
            if not forecast_path.exists():
                forecast_path = CONFIG.MODEL_DIR / "forecast_1d_5.pt"

            if not forecast_path.exists():
                log.debug("No forecaster model found")
                return

            data = torch.load(
                forecast_path, map_location="cpu", weights_only=False
            )

            required_keys = {"input_size", "horizon", "arch", "state_dict"}
            if not required_keys.issubset(data.keys()):
                log.warning(
                    f"Forecaster checkpoint missing keys: "
                    f"{required_keys - set(data.keys())}"
                )
                return

            self.forecaster = TCNModel(
                input_size=data["input_size"],
                hidden_size=data["arch"]["hidden_size"],
                num_classes=data["horizon"],
                dropout=data["arch"]["dropout"],
            )
            self.forecaster.load_state_dict(data["state_dict"])
            self.forecaster.eval()

            log.info(f"Forecaster loaded: horizon={data['horizon']}")

        except ImportError:
            log.debug("TCNModel not available — forecaster disabled")
            self.forecaster = None
        except Exception as e:
            log.debug(f"Forecaster not loaded: {e}")
            self.forecaster = None

    # =========================================================================
    # =========================================================================

    def predict(
        self,
        stock_code: str,
        use_realtime_price: bool = True,
        interval: str = None,
        forecast_minutes: int = None,
        lookback_bars: int = None,
        skip_cache: bool = False,
    ) -> Prediction:
        """
        Make full prediction with price forecast.

        Args:
            stock_code: Stock code to predict
            use_realtime_price: Whether to use real-time price
            interval: Data interval (1m, 5m, 1d, etc.)
            forecast_minutes: Number of bars to forecast
            lookback_bars: Historical bars to use
            skip_cache: If True, bypass prediction cache (for real-time)

        Returns:
            Prediction object with all fields populated
        """
        with self._predict_lock:
            interval = str(interval or self.interval).lower()
            horizon = int(forecast_minutes or self.horizon)
            lookback = int(
                lookback_bars or (1400 if interval == "1m" else 600)
            )

            code = self._clean_code(stock_code)

            if not skip_cache:
                cached = self._get_cached_prediction(code)
                if cached is not None:
                    return cached

            pred = Prediction(
                stock_code=code,
                timestamp=datetime.now(),
                interval=interval,
                horizon=horizon,
            )

            try:
                df = self._fetch_data(
                    code, interval, lookback, use_realtime_price
                )
                if (
                    df is None
                    or df.empty
                    or len(df) < CONFIG.SEQUENCE_LENGTH
                ):
                    data_len = len(df) if df is not None else 0
                    pred.warnings.append(
                        f"Insufficient data: got {data_len} bars, "
                        f"need {CONFIG.SEQUENCE_LENGTH}"
                    )
                    return pred

                pred.stock_name = self._get_stock_name(code, df)
                pred.current_price = float(df["close"].iloc[-1])
                pred.price_history = df["close"].tail(180).tolist()

                min_rows = getattr(
                    self.feature_engine, 'MIN_ROWS',
                    CONFIG.SEQUENCE_LENGTH
                )
                if len(df) < min_rows:
                    pred.warnings.append(
                        f"Insufficient data for features: "
                        f"{len(df)} < {min_rows}"
                    )
                    return pred

                df = self.feature_engine.create_features(df)
                self._extract_technicals(df, pred)

                X = self.processor.prepare_inference_sequence(
                    df, self._feature_cols
                )

                if self.ensemble:
                    self._apply_ensemble_prediction(X, pred)

                pred.predicted_prices = self._generate_forecast(
                    X, pred.current_price, horizon, pred.atr_pct_value
                )
                pred.levels = self._calculate_levels(pred)
                pred.position = self._calculate_position(pred)
                self._generate_reasons(pred)

                self._set_cached_prediction(code, pred)

            except Exception as e:
                log.error(
                    f"Prediction failed for {code}: {e}",
                    exc_info=True
                )
                pred.warnings.append(
                    f"Prediction error: {type(e).__name__}: {str(e)}"
                )

            return pred

    def predict_quick_batch(
        self,
        stock_codes: List[str],
        use_realtime_price: bool = True,
        interval: str = None,
        lookback_bars: int = None,
    ) -> List[Prediction]:
        """Quick batch prediction without full forecasting."""
        with self._predict_lock:
            interval = str(interval or self.interval).lower()
            lookback = int(
                lookback_bars or (1400 if interval == "1m" else 300)
            )

            predictions: List[Prediction] = []

            for stock_code in stock_codes:
                try:
                    code = self._clean_code(stock_code)
                    if not code:
                        continue

                    pred = Prediction(
                        stock_code=code,
                        timestamp=datetime.now(),
                        interval=interval,
                    )

                    min_rows = getattr(
                        self.feature_engine, 'MIN_ROWS',
                        CONFIG.SEQUENCE_LENGTH
                    )

                    df = self._fetch_data(
                        code, interval, lookback, use_realtime_price
                    )
                    if (
                        df is None
                        or df.empty
                        or len(df) < CONFIG.SEQUENCE_LENGTH
                        or len(df) < min_rows
                    ):
                        continue

                    pred.current_price = float(df["close"].iloc[-1])

                    df = self.feature_engine.create_features(df)
                    X = self.processor.prepare_inference_sequence(
                        df, self._feature_cols
                    )

                    if self.ensemble:
                        self._apply_ensemble_prediction(X, pred)

                    predictions.append(pred)

                except Exception as e:
                    log.debug(
                        f"Quick prediction failed for {stock_code}: {e}"
                    )

            return predictions

    def get_realtime_forecast_curve(
        self,
        stock_code: str,
        interval: str = None,
        horizon_steps: int = None,
        lookback_bars: int = None,
        use_realtime_price: bool = True,
    ) -> Tuple[List[float], List[float]]:
        """
        Get real-time forecast curve for charting.

        Returns:
            (actual_prices, predicted_prices)
        """
        with self._predict_lock:
            interval = str(interval or self.interval).lower()
            horizon = int(horizon_steps or self.horizon)
            lookback = int(
                lookback_bars or (1400 if interval == "1m" else 600)
            )

            code = self._clean_code(stock_code)

            try:
                min_rows = getattr(
                    self.feature_engine, 'MIN_ROWS',
                    CONFIG.SEQUENCE_LENGTH
                )

                df = self._fetch_data(
                    code, interval, lookback, use_realtime_price
                )

                if (
                    df is None
                    or df.empty
                    or len(df) < CONFIG.SEQUENCE_LENGTH
                    or len(df) < min_rows
                ):
                    return [], []

                actual = df["close"].tail(180).tolist()
                current_price = float(df["close"].iloc[-1])

                df = self.feature_engine.create_features(df)
                X = self.processor.prepare_inference_sequence(
                    df, self._feature_cols
                )

                atr_pct = self._get_atr_pct(df)

                predicted = self._generate_forecast(
                    X, current_price, horizon, atr_pct
                )

                return actual, predicted

            except Exception as e:
                log.warning(f"Forecast curve failed for {code}: {e}")
                return [], []

    def get_top_picks(
        self,
        stock_codes: List[str],
        n: int = 10,
        signal_type: str = "buy",
    ) -> List[Prediction]:
        """Get top N stock picks based on signal type."""
        with self._predict_lock:
            predictions = self.predict_quick_batch(stock_codes)

            if signal_type.lower() == "buy":
                filtered = [
                    p for p in predictions
                    if p.signal in [Signal.STRONG_BUY, Signal.BUY]
                    and p.confidence >= CONFIG.MIN_CONFIDENCE
                ]
            else:
                filtered = [
                    p for p in predictions
                    if p.signal in [Signal.STRONG_SELL, Signal.SELL]
                    and p.confidence >= CONFIG.MIN_CONFIDENCE
                ]

            filtered.sort(key=lambda x: x.confidence, reverse=True)

            return filtered[:n]

    # =========================================================================
    # =========================================================================

    def _apply_ensemble_prediction(
        self, X: np.ndarray, pred: Prediction
    ):
        """Apply ensemble prediction with bounds checking."""
        ensemble_pred = self.ensemble.predict(X)

        probs = ensemble_pred.probabilities
        n_classes = len(probs)

        pred.prob_down = float(probs[0]) if n_classes > 0 else 0.33
        pred.prob_neutral = float(probs[1]) if n_classes > 1 else 0.34
        pred.prob_up = float(probs[2]) if n_classes > 2 else 0.33

        pred.prob_down = max(0.0, min(1.0, pred.prob_down))
        pred.prob_neutral = max(0.0, min(1.0, pred.prob_neutral))
        pred.prob_up = max(0.0, min(1.0, pred.prob_up))

        pred.confidence = float(
            max(0.0, min(1.0, ensemble_pred.confidence))
        )

        pred.model_agreement = float(
            getattr(ensemble_pred, "agreement", 1.0)
        )
        pred.entropy = float(
            getattr(ensemble_pred, "entropy", 0.0)
        )

        pred.signal = self._determine_signal(ensemble_pred, pred)
        pred.signal_strength = self._calculate_signal_strength(
            ensemble_pred, pred
        )
        self._apply_high_precision_gate(pred)

    def _apply_high_precision_gate(self, pred: Prediction) -> None:
        """Optionally downgrade weak actionable predictions to HOLD."""
        cfg = self._high_precision
        if not cfg or cfg.get("enabled", 0.0) <= 0:
            return
        if pred.signal == Signal.HOLD:
            return

        reasons: List[str] = []

        # Regime-aware confidence floor: range/high-vol require stronger evidence.
        required_conf = float(cfg["min_confidence"])
        if cfg.get("regime_routing", 0.0) > 0:
            if str(pred.trend).upper() == "SIDEWAYS":
                required_conf += float(cfg.get("range_conf_boost", 0.0))
            if float(pred.atr_pct_value) >= float(cfg.get("high_vol_atr_pct", 0.035)):
                required_conf += float(cfg.get("high_vol_conf_boost", 0.0))

        if pred.confidence < required_conf:
            reasons.append(
                f"confidence {pred.confidence:.2f} < {required_conf:.2f}"
            )
        if pred.model_agreement < cfg["min_agreement"]:
            reasons.append(
                f"agreement {pred.model_agreement:.2f} < {cfg['min_agreement']:.2f}"
            )
        if pred.entropy > cfg["max_entropy"]:
            reasons.append(
                f"entropy {pred.entropy:.2f} > {cfg['max_entropy']:.2f}"
            )
        edge = abs(float(pred.prob_up) - float(pred.prob_down))
        if edge < cfg["min_edge"]:
            reasons.append(f"edge {edge:.2f} < {cfg['min_edge']:.2f}")

        if not reasons:
            return

        old_signal = pred.signal.value
        pred.signal = Signal.HOLD
        pred.signal_strength = min(float(pred.signal_strength), 0.49)
        pred.warnings.append(
            "High Precision Mode filtered signal "
            f"{old_signal} -> HOLD ({'; '.join(reasons[:3])})"
        )

    # =========================================================================
    # =========================================================================

    def _get_cached_prediction(self, code: str) -> Optional[Prediction]:
        """Get cached prediction if still valid."""
        with self._cache_lock:
            entry = self._pred_cache.get(code)
            if entry is not None:
                ts, pred = entry
                if (time.time() - ts) < self._CACHE_TTL:
                    return pred
                del self._pred_cache[code]
        return None

    def _set_cached_prediction(self, code: str, pred: Prediction):
        """Cache a prediction result with bounded size."""
        with self._cache_lock:
            self._pred_cache[code] = (time.time(), pred)

            if len(self._pred_cache) > self._MAX_CACHE_SIZE:
                now = time.time()
                expired = [
                    k for k, (ts, _) in self._pred_cache.items()
                    if (now - ts) > self._CACHE_TTL
                ]
                for k in expired:
                    del self._pred_cache[k]

                # If still too large, evict oldest
                if len(self._pred_cache) > self._MAX_CACHE_SIZE:
                    sorted_keys = sorted(
                        self._pred_cache.keys(),
                        key=lambda k: self._pred_cache[k][0]
                    )
                    for k in sorted_keys[:len(sorted_keys) // 2]:
                        del self._pred_cache[k]

    def invalidate_cache(self, code: str = None):
        """Invalidate cache for a specific code or all codes."""
        with self._cache_lock:
            if code:
                self._pred_cache.pop(code, None)
            else:
                self._pred_cache.clear()

    # =========================================================================
    # =========================================================================

    def _fetch_data(
        self,
        code: str,
        interval: str,
        lookback: int,
        use_realtime: bool,
    ) -> Optional[pd.DataFrame]:
        """Fetch stock data with minimum data requirement."""
        try:
            from data.fetcher import BARS_PER_DAY

            interval = str(interval).lower()
            bpd = float(BARS_PER_DAY.get(interval, 1))
            min_days = 14
            min_bars = int(max(min_days * bpd, min_days))
            bars = int(max(int(lookback), int(min_bars)))

            df = self.fetcher.get_history(
                code,
                interval=interval,
                bars=bars,
                use_cache=True,
                update_db=True,
            )
            if df is None or df.empty:
                return None

            if use_realtime:
                try:
                    quote = self.fetcher.get_realtime(code)
                    if quote and quote.price > 0:
                        df.loc[df.index[-1], "close"] = float(
                            quote.price
                        )
                        df.loc[df.index[-1], "high"] = max(
                            float(df["high"].iloc[-1]),
                            float(quote.price)
                        )
                        df.loc[df.index[-1], "low"] = min(
                            float(df["low"].iloc[-1]),
                            float(quote.price)
                        )
                except Exception:
                    pass

            return df

        except Exception as e:
            log.warning(f"Failed to fetch data for {code}: {e}")
            return None

    # =========================================================================
    # =========================================================================

    def _generate_forecast(
        self,
        X: np.ndarray,
        current_price: float,
        horizon: int,
        atr_pct: float = 0.02,
    ) -> List[float]:
        """Generate price forecast using forecaster or ensemble."""
        if current_price <= 0:
            return []

        if self.forecaster is not None:
            try:
                import torch

                self.forecaster.eval()
                with torch.inference_mode():
                    X_tensor = torch.FloatTensor(X)
                    returns, _ = self.forecaster(X_tensor)
                    returns = returns[0].cpu().numpy()

                prices = [current_price]
                for r in returns[:horizon]:
                    next_price = prices[-1] * (1 + float(r) / 100)
                    next_price = max(
                        next_price, current_price * 0.5
                    )
                    next_price = min(
                        next_price, current_price * 2.0
                    )
                    prices.append(float(next_price))

                return prices[1:]

            except Exception as e:
                log.debug(f"Forecaster failed: {e}")

        # Fallback: ensemble-guided stochastic forecast
        if self.ensemble:
            try:
                pred = self.ensemble.predict(X)

                probs = pred.probabilities
                direction = (
                    (float(probs[2]) if len(probs) > 2 else 0.33)
                    - (float(probs[0]) if len(probs) > 0 else 0.33)
                )

                volatility = max(atr_pct, 0.005)

                prices = []
                price = current_price

                # Deterministic seed per stock+time for consistency
                seed = int(current_price * 100 + horizon) % (2**31)
                rng = np.random.RandomState(seed)

                for i in range(horizon):
                    # Mean-reverting drift with ensemble direction
                    drift = direction * volatility * 0.3
                    noise = rng.normal(0, volatility * 0.5)
                    decay = 1.0 - (i / (horizon * 2))
                    change = drift * decay + noise
                    price = price * (1 + change)

                    price = max(price, current_price * 0.5)
                    price = min(price, current_price * 2.0)

                    prices.append(float(price))

                return prices

            except Exception as e:
                log.debug(f"Ensemble forecast failed: {e}")

        return [current_price] * horizon

    # =========================================================================
    # =========================================================================

    def _determine_signal(
        self, ensemble_pred, pred: Prediction
    ) -> Signal:
        """Determine trading signal from prediction."""
        confidence = float(ensemble_pred.confidence)
        predicted_class = int(ensemble_pred.predicted_class)

        if predicted_class == 2:  # UP
            if confidence >= CONFIG.STRONG_BUY_THRESHOLD:
                return Signal.STRONG_BUY
            elif confidence >= CONFIG.BUY_THRESHOLD:
                return Signal.BUY
        elif predicted_class == 0:  # DOWN
            if confidence >= CONFIG.STRONG_SELL_THRESHOLD:
                return Signal.STRONG_SELL
            elif confidence >= CONFIG.SELL_THRESHOLD:
                return Signal.SELL

        return Signal.HOLD

    def _calculate_signal_strength(
        self, ensemble_pred, pred: Prediction
    ) -> float:
        """Calculate signal strength 0-1."""
        confidence = float(ensemble_pred.confidence)
        agreement = float(getattr(ensemble_pred, "agreement", 1.0))
        entropy_inv = 1.0 - float(
            getattr(ensemble_pred, "entropy", 0.0)
        )

        return float(
            np.clip(
                (confidence + agreement + entropy_inv) / 3.0,
                0.0, 1.0
            )
        )

    # =========================================================================
    # =========================================================================

    def _calculate_levels(self, pred: Prediction) -> TradingLevels:
        """Calculate trading levels using actual ATR from features."""
        price = pred.current_price

        if price <= 0:
            return TradingLevels()

        # Use actual ATR percentage from features, with floor
        atr_pct = max(pred.atr_pct_value, 0.005)

        levels = TradingLevels(entry=price)

        if pred.signal in [Signal.STRONG_BUY, Signal.BUY]:
            levels.stop_loss = price * (1 - atr_pct * 1.5)
            levels.target_1 = price * (1 + atr_pct * 1.5)
            levels.target_2 = price * (1 + atr_pct * 3.0)
            levels.target_3 = price * (1 + atr_pct * 5.0)
        elif pred.signal in [Signal.STRONG_SELL, Signal.SELL]:
            levels.stop_loss = price * (1 + atr_pct * 1.5)
            levels.target_1 = price * (1 - atr_pct * 1.5)
            levels.target_2 = price * (1 - atr_pct * 3.0)
            levels.target_3 = price * (1 - atr_pct * 5.0)
        else:
            levels.stop_loss = price * (1 - atr_pct)
            levels.target_1 = price * (1 + atr_pct)
            levels.target_2 = price * (1 + atr_pct * 2.0)
            levels.target_3 = price * (1 + atr_pct * 3.5)

        if price > 0:
            levels.stop_loss_pct = (levels.stop_loss / price - 1) * 100
            levels.target_1_pct = (levels.target_1 / price - 1) * 100
            levels.target_2_pct = (levels.target_2 / price - 1) * 100
            levels.target_3_pct = (levels.target_3 / price - 1) * 100

        return levels

    # =========================================================================
    # =========================================================================

    def _calculate_position(self, pred: Prediction) -> PositionSize:
        """Calculate position size using risk, quality and expected-edge gating."""
        price = pred.current_price

        if price <= 0:
            return PositionSize()

        stop_distance, reward_distance = self._resolve_trade_distances(pred)
        if stop_distance <= 0 or reward_distance <= 0:
            return PositionSize()

        risk_pct = float(CONFIG.RISK_PER_TRADE) / 100.0
        quality_scale = self._quality_scale(pred)
        edge = self._expected_edge(pred, price, stop_distance, reward_distance)
        rr_ratio = reward_distance / max(stop_distance, 1e-9)
        min_edge = max(0.0, float(CONFIG.risk.min_expected_edge_pct) / 100.0)
        min_rr = max(0.1, float(CONFIG.risk.min_risk_reward_ratio))

        if rr_ratio < min_rr or edge <= 0.0:
            return PositionSize(
                expected_edge_pct=edge * 100.0,
                risk_reward_ratio=rr_ratio,
            )

        edge_scale = 1.0
        if min_edge > 0:
            edge_scale = float(np.clip(edge / min_edge, 0.0, CONFIG.risk.max_position_scale))

        risk_amount = self.capital * risk_pct * quality_scale * edge_scale

        lot_size = max(1, CONFIG.LOT_SIZE)

        shares = int(risk_amount / stop_distance)
        shares = (shares // lot_size) * lot_size

        if shares < lot_size:
            shares = lot_size

        max_value = self.capital * (CONFIG.MAX_POSITION_PCT / 100)
        if shares * price > max_value:
            shares = int(max_value / price)
            shares = (shares // lot_size) * lot_size

        # Final guard: ensure shares > 0 and affordable
        if shares <= 0:
            shares = lot_size

        if shares * price > self.capital:
            # Can't afford even one lot
            return PositionSize()

        return PositionSize(
            shares=int(shares),
            value=float(shares * price),
            risk_amount=float(shares * stop_distance),
            risk_pct=float(
                (shares * stop_distance / self.capital) * 100
            ),
            expected_edge_pct=float(edge * 100.0),
            risk_reward_ratio=float(rr_ratio),
        )

    def _resolve_trade_distances(self, pred: Prediction) -> Tuple[float, float]:
        """Resolve stop and reward distances from level plan."""
        price = float(pred.current_price)
        if price <= 0:
            return 0.0, 0.0

        stop_distance = abs(price - float(pred.levels.stop_loss))
        if stop_distance <= 0:
            stop_distance = price * 0.02

        d1 = abs(float(pred.levels.target_1) - price)
        d2 = abs(float(pred.levels.target_2) - price)
        # Weighted reward estimate: partial at target_1 plus runner to target_2.
        reward_distance = (0.7 * d1) + (0.3 * d2)
        if reward_distance <= 0:
            reward_distance = d1 if d1 > 0 else stop_distance

        return stop_distance, reward_distance

    def _quality_scale(self, pred: Prediction) -> float:
        """Scale risk by signal quality (confidence and strength)."""
        conf = float(np.clip(pred.confidence, 0.0, 1.0))
        strength = float(np.clip(pred.signal_strength, 0.0, 1.0))
        agreement = float(np.clip(pred.model_agreement, 0.0, 1.0))
        quality = (0.5 * conf) + (0.35 * strength) + (0.15 * agreement)
        return float(np.clip(quality, 0.25, CONFIG.risk.max_position_scale))

    def _expected_edge(
        self,
        pred: Prediction,
        price: float,
        stop_distance: float,
        reward_distance: float,
    ) -> float:
        """
        Estimate expected edge after costs.

        Returns decimal edge (e.g. 0.003 means +0.3% expected value).
        """
        if price <= 0:
            return 0.0

        if pred.signal in (Signal.BUY, Signal.STRONG_BUY):
            p_win = float(np.clip(pred.prob_up, 0.0, 1.0))
            p_loss = float(np.clip(pred.prob_down, 0.0, 1.0))
            side = "buy"
        elif pred.signal in (Signal.SELL, Signal.STRONG_SELL):
            p_win = float(np.clip(pred.prob_down, 0.0, 1.0))
            p_loss = float(np.clip(pred.prob_up, 0.0, 1.0))
            side = "sell"
        else:
            return 0.0

        reward_pct = reward_distance / price
        risk_pct = stop_distance / price
        gross_edge = (p_win * reward_pct) - (p_loss * risk_pct)
        cost_pct = self._round_trip_cost_pct(side)
        return float(gross_edge - cost_pct)

    def _round_trip_cost_pct(self, side: str) -> float:
        """Estimate round-trip friction cost as decimal percentage."""
        commission = max(float(CONFIG.COMMISSION), 0.0)
        slippage = max(float(CONFIG.SLIPPAGE), 0.0)
        stamp_tax = max(float(CONFIG.STAMP_TAX), 0.0)

        if side == "sell":
            # Short-cover path includes one sell leg with stamp tax.
            return (2.0 * commission) + (2.0 * slippage) + stamp_tax
        # Long round trip: buy then sell, stamp tax on sell leg.
        return (2.0 * commission) + (2.0 * slippage) + stamp_tax

    # =========================================================================
    # =========================================================================

    def _extract_technicals(self, df: pd.DataFrame, pred: Prediction):
        """
        Extract technical indicators from dataframe.

        IMPORTANT: FeatureEngine normalizes indicators:
        - rsi_14 = raw_rsi/100 - 0.5  (range: -0.5 to 0.5)
        - macd_hist = hist/close*100   (scale-invariant percentage)
        - ma_ratio_5_20 = (ma5/ma20 - 1) * 100
        """
        try:
            # RSI: reverse FeatureEngine normalization
            if "rsi_14" in df.columns:
                normalized_rsi = float(df["rsi_14"].iloc[-1])
                # FeatureEngine does: rsi_14 = raw_rsi/100 - 0.5
                # So: raw_rsi = (normalized_rsi + 0.5) * 100
                raw_rsi = (normalized_rsi + 0.5) * 100.0
                pred.rsi = float(np.clip(raw_rsi, 0.0, 100.0))

            if "macd_hist" in df.columns:
                macd_hist = float(df["macd_hist"].iloc[-1])
                if macd_hist > 0.001:
                    pred.macd_signal = "BULLISH"
                elif macd_hist < -0.001:
                    pred.macd_signal = "BEARISH"
                else:
                    pred.macd_signal = "NEUTRAL"

            if "ma_ratio_5_20" in df.columns:
                ma_ratio = float(df["ma_ratio_5_20"].iloc[-1])
                if ma_ratio > 1.0:
                    pred.trend = "UPTREND"
                elif ma_ratio < -1.0:
                    pred.trend = "DOWNTREND"
                else:
                    pred.trend = "SIDEWAYS"

            pred.atr_pct_value = self._get_atr_pct(df)

        except Exception as e:
            log.debug(f"Technical extraction error: {e}")

    def _get_atr_pct(self, df: pd.DataFrame) -> float:
        """Get ATR as a decimal fraction (e.g., 0.02 for 2%)."""
        try:
            if "atr_pct" in df.columns:
                atr = float(df["atr_pct"].iloc[-1])
                # atr_pct from FeatureEngine is: atr_14 / close * 100
                return max(atr / 100.0, 0.005)
        except Exception:
            pass
        return 0.02  # default 2%

    # =========================================================================
    # =========================================================================

    def _generate_reasons(self, pred: Prediction):
        """Generate analysis reasons and warnings."""
        reasons = []
        warnings = []

        if pred.confidence >= 0.7:
            reasons.append(
                f"High AI confidence: {pred.confidence:.0%}"
            )
        elif pred.confidence >= 0.6:
            reasons.append(
                f"Moderate AI confidence: {pred.confidence:.0%}"
            )
        else:
            warnings.append(
                f"Low AI confidence: {pred.confidence:.0%}"
            )

        if pred.model_agreement < 0.6:
            warnings.append(
                f"Low model agreement: {pred.model_agreement:.0%}"
            )

        if pred.prob_up > 0.5:
            reasons.append(
                f"AI predicts UP with {pred.prob_up:.0%} probability"
            )
        elif pred.prob_down > 0.5:
            reasons.append(
                f"AI predicts DOWN with {pred.prob_down:.0%} probability"
            )

        if pred.rsi > 70:
            warnings.append(f"RSI overbought: {pred.rsi:.0f}")
        elif pred.rsi < 30:
            warnings.append(f"RSI oversold: {pred.rsi:.0f}")
        else:
            reasons.append(f"RSI neutral: {pred.rsi:.0f}")

        # Signal-trend alignment
        if (
            pred.signal in [Signal.STRONG_BUY, Signal.BUY]
            and pred.trend == "UPTREND"
        ):
            reasons.append("Signal aligned with uptrend")
        elif (
            pred.signal in [Signal.STRONG_SELL, Signal.SELL]
            and pred.trend == "DOWNTREND"
        ):
            reasons.append("Signal aligned with downtrend")
        elif (
            pred.trend != "SIDEWAYS"
            and pred.signal != Signal.HOLD
        ):
            warnings.append(f"Signal against trend ({pred.trend})")

        if pred.macd_signal != "NEUTRAL":
            reasons.append(f"MACD: {pred.macd_signal}")

        if pred.entropy > 0.8:
            warnings.append(
                f"High prediction uncertainty "
                f"(entropy: {pred.entropy:.2f})"
            )

        pred.reasons = reasons
        pred.warnings = warnings

    # =========================================================================
    # =========================================================================

    def _get_stock_name(self, code: str, df: pd.DataFrame) -> str:
        """Get stock name from fetcher."""
        try:
            quote = self.fetcher.get_realtime(code)
            if quote and quote.name:
                return str(quote.name)
        except Exception:
            pass
        return ""

    def _clean_code(self, code: str) -> str:
        """
        Clean and normalize stock code.
        Delegates to DataFetcher when available.
        """
        if self.fetcher is not None:
            try:
                return self.fetcher.clean_code(code)
            except Exception:
                pass

        if not code:
            return ""
        code = str(code).strip()
        code = "".join(c for c in code if c.isdigit())
        return code.zfill(6) if code else ""
