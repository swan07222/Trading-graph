# models/predictor.py
import copy
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

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
    raw_confidence: float = 0.0

    prob_up: float = 0.33
    prob_neutral: float = 0.34
    prob_down: float = 0.33

    current_price: float = 0.0
    price_history: list[float] = field(default_factory=list)
    predicted_prices: list[float] = field(default_factory=list)

    rsi: float = 50.0
    macd_signal: str = "NEUTRAL"
    trend: str = "NEUTRAL"

    levels: TradingLevels = field(default_factory=TradingLevels)

    position: PositionSize = field(default_factory=PositionSize)

    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    interval: str = "1m"
    horizon: int = 30

    # Extra fields for UI/signal generator
    model_agreement: float = 1.0
    entropy: float = 0.0
    model_margin: float = 0.0
    brier_score: float = 0.0
    uncertainty_score: float = 0.5
    tail_risk_score: float = 0.5

    # ATR from features (used internally for levels)
    atr_pct_value: float = 0.02

    # Forecast uncertainty bands (same length as predicted_prices)
    predicted_prices_low: list[float] = field(default_factory=list)
    predicted_prices_high: list[float] = field(default_factory=list)

    # News-aware context (used by UI/details and forecast tilt)
    news_sentiment: float = 0.0
    news_confidence: float = 0.0
    news_count: int = 0

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
    _CACHE_TTL_REALTIME: float = 1.2
    _MAX_CACHE_SIZE: int = 200
    _NEWS_CACHE_TTL_INTRADAY: float = 45.0
    _NEWS_CACHE_TTL_SWING: float = 180.0

    def __init__(
        self,
        capital: float = None,
        interval: str = "1m",
        prediction_horizon: int = None,
    ):
        self.capital = float(capital or CONFIG.CAPITAL)
        self.interval = str(interval).lower()
        intraday_tokens = {"1m", "2m", "3m", "5m", "15m", "30m", "60m", "1h"}
        default_horizon = (
            30
            if self.interval in intraday_tokens
            else int(CONFIG.PREDICTION_HORIZON)
        )
        self.horizon = int(
            prediction_horizon
            if prediction_horizon is not None
            else default_horizon
        )

        self._predict_lock = threading.RLock()

        # Components (lazy loaded)
        self.ensemble = None
        self.forecaster = None
        self._forecaster_horizon: int = 0
        self.processor = None
        self.feature_engine = None
        self.fetcher = None

        self._feature_cols: list[str] = []

        # Prediction cache: code -> (timestamp, Prediction)
        self._pred_cache: dict[str, tuple[float, Prediction]] = {}
        self._cache_lock = threading.Lock()
        # News sentiment cache: code -> (ts, sentiment, confidence, count)
        self._news_cache: dict[str, tuple[float, float, float, int]] = {}
        self._news_cache_lock = threading.Lock()

        # Track if constructor params were overridden by model metadata
        self._requested_interval = self.interval
        self._requested_horizon = self.horizon
        self._loaded_model_interval = self.interval
        self._loaded_model_horizon = self.horizon
        self._high_precision = self._load_high_precision_config()
        self._loaded_ensemble_path: Path | None = None
        self._trained_stock_codes: list[str] = []
        self._trained_stock_last_train: dict[str, str] = {}
        self._model_artifact_sig: str = ""
        self._last_model_reload_attempt_ts: float = 0.0
        self._model_reload_cooldown_s: float = 15.0

        self._load_models()
        self._model_artifact_sig = self._model_artifact_signature()
        self._last_model_reload_attempt_ts = time.monotonic()

    def _load_high_precision_config(self) -> dict[str, float]:
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
            except Exception as e:
                log.debug("Invalid float env override %s=%r: %s", name, raw, e)
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
            from data.features import FeatureEngine
            from data.fetcher import get_fetcher
            from data.processor import DataProcessor
            from models.ensemble import EnsembleModel

            self.processor = DataProcessor()
            self.feature_engine = FeatureEngine()
            self.fetcher = get_fetcher()
            self._feature_cols = self.feature_engine.get_feature_columns()

            model_dir = CONFIG.MODEL_DIR

            # Pick best ensemble + scaler pair
            chosen_ens, chosen_scl = self._find_best_model_pair(model_dir)
            if chosen_scl is None:
                chosen_scl = self._find_best_scaler_checkpoint(model_dir)

            if chosen_scl and chosen_scl.exists():
                if not self.processor.load_scaler(str(chosen_scl)):
                    log.warning("Failed loading scaler from %s", chosen_scl)
            else:
                log.warning(
                    "No scaler found for requested interval/horizon (%s/%s)",
                    self.interval,
                    self.horizon,
                )

            self.ensemble = None
            self._loaded_ensemble_path = None
            self._trained_stock_codes = []
            self._trained_stock_last_train = {}
            if chosen_ens and chosen_ens.exists():
                self._loaded_ensemble_path = Path(chosen_ens)
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

                    # Keep runtime interval/horizon aligned to constructor
                    # (UI/requested settings). Loaded model metadata is tracked
                    # separately for diagnostics and compatibility checks.
                    self._loaded_model_interval = str(loaded_interval).lower()
                    self._loaded_model_horizon = int(loaded_horizon)
                    self._trained_stock_codes = (
                        self._extract_trained_stocks_from_ensemble()
                        or self._load_trained_stocks_from_manifest(chosen_ens)
                        or self._load_trained_stocks_from_learner_state(
                            loaded_interval,
                            loaded_horizon,
                        )
                    )
                    self._trained_stock_last_train = (
                        self._extract_trained_stock_last_train_from_ensemble()
                        or self._load_trained_stock_last_train_from_manifest(chosen_ens)
                    )
                    if self._trained_stock_last_train and self._trained_stock_codes:
                        allowed = {
                            self._normalize_stock_code(x)
                            for x in list(self._trained_stock_codes or [])
                        }
                        allowed = {x for x in allowed if x}
                        if allowed:
                            self._trained_stock_last_train = {
                                c: ts
                                for c, ts in self._trained_stock_last_train.items()
                                if c in allowed
                            }

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

        req_ens = model_dir / f"ensemble_{self.interval}_{self.horizon}.pt"
        req_scl = model_dir / f"scaler_{self.interval}_{self.horizon}.pkl"

        if req_ens.exists() and req_scl.exists():
            return req_ens, req_scl

        if req_ens.exists() and not req_scl.exists():
            log.warning(
                "Exact ensemble exists but scaler missing for %s/%s",
                self.interval,
                self.horizon,
            )

        # Same interval, nearest available horizon with scaler.
        same_interval: list[tuple[int, float, Path, Path]] = []
        for ep in model_dir.glob(f"ensemble_{self.interval}_*.pt"):
            parts = ep.stem.split("_", 2)
            if len(parts) != 3:
                continue
            cand_h = self._parse_artifact_horizon(parts[2])
            if cand_h is None:
                continue
            sp = self._find_scaler_for_interval_horizon(
                model_dir=model_dir,
                interval=self.interval,
                horizon=cand_h,
            )
            if sp is None:
                continue
            same_interval.append(
                (abs(cand_h - int(self.horizon)), -ep.stat().st_mtime, ep, sp)
            )
        if same_interval:
            same_interval.sort(key=lambda x: (x[0], x[1]))
            _delta, _mtime_neg, ep, sp = same_interval[0]
            return ep, sp

        log.warning(
            "No interval-matched model pair found for interval=%s horizon=%s",
            self.interval,
            self.horizon,
        )
        return None, None

    @staticmethod
    def _parse_artifact_horizon(token: str) -> int | None:
        """Extract horizon integer from artifact token (supports suffixes)."""
        s = str(token or "").strip()
        if not s:
            return None
        m = re.match(r"^(\d+)", s)
        if m is None:
            return None
        try:
            return int(m.group(1))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _find_scaler_for_interval_horizon(
        model_dir: Path,
        interval: str,
        horizon: int,
    ) -> Path | None:
        """Find exact or suffixed scaler checkpoint for interval+horizon."""
        exact = model_dir / f"scaler_{interval}_{int(horizon)}.pkl"
        if exact.exists():
            return exact

        matches = list(model_dir.glob(f"scaler_{interval}_{int(horizon)}*.pkl"))
        if not matches:
            return None
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    def _find_best_forecaster_checkpoint(self, model_dir: Path) -> Path | None:
        """Find best-matching forecaster for current interval+horizon."""
        req_path = model_dir / f"forecast_{self.interval}_{self.horizon}.pt"
        if req_path.exists():
            return req_path

        same_interval: list[tuple[int, float, Path]] = []
        for fp in model_dir.glob(f"forecast_{self.interval}_*.pt"):
            parts = fp.stem.split("_", 2)
            if len(parts) != 3:
                continue
            cand_h = self._parse_artifact_horizon(parts[2])
            if cand_h is None:
                continue
            same_interval.append(
                (abs(cand_h - int(self.horizon)), -fp.stat().st_mtime, fp)
            )
        if same_interval:
            same_interval.sort(key=lambda x: (x[0], x[1]))
            _delta, _mtime_neg, fp = same_interval[0]
            return fp

        return None

    def _find_best_scaler_checkpoint(self, model_dir: Path) -> Path | None:
        """Find best-matching scaler for current interval+horizon."""
        req_path = model_dir / f"scaler_{self.interval}_{self.horizon}.pkl"
        if req_path.exists():
            return req_path

        same_interval: list[tuple[int, float, Path]] = []
        for sp in model_dir.glob(f"scaler_{self.interval}_*.pkl"):
            parts = sp.stem.split("_", 2)
            if len(parts) != 3:
                continue
            cand_h = self._parse_artifact_horizon(parts[2])
            if cand_h is None:
                continue
            same_interval.append(
                (abs(cand_h - int(self.horizon)), -sp.stat().st_mtime, sp)
            )
        if same_interval:
            same_interval.sort(key=lambda x: (x[0], x[1]))
            _delta, _mtime_neg, sp = same_interval[0]
            return sp

        return None

    def _model_artifact_signature(self) -> str:
        """Compact signature of model artifacts for hot-reload checks."""
        try:
            model_dir = Path(CONFIG.MODEL_DIR)
            if not model_dir.exists():
                return ""
            parts: list[str] = []
            for pattern in ("ensemble_*.pt", "forecast_*.pt", "scaler_*.pkl"):
                files = list(model_dir.glob(pattern))
                latest = 0.0
                for fp in files:
                    try:
                        latest = max(latest, float(fp.stat().st_mtime))
                    except OSError:
                        continue
                parts.append(f"{pattern}:{len(files)}:{latest:.3f}")
            return "|".join(parts)
        except Exception as e:
            log.debug("Model artifact signature check failed: %s", e)
            return ""

    def _models_ready_for_runtime(self) -> bool:
        """Whether runtime has enough loaded artifacts for stable inference."""
        processor = getattr(self, "processor", None)
        ensemble = getattr(self, "ensemble", None)
        forecaster = getattr(self, "forecaster", None)
        scaler_ready = bool(
            processor is not None and getattr(processor, "is_fitted", True)
        )
        return scaler_ready and (ensemble is not None or forecaster is not None)

    def _maybe_reload_models(self, *, reason: str = "", force: bool = False) -> bool:
        """
        Opportunistically reload models when artifacts appear/rotate on disk.

        This keeps long-running UI sessions in sync without requiring restart
        after training finishes.
        """
        # Unit tests may construct Predictor via __new__ with a minimal fixture.
        # In that case, skip runtime hot-reload logic.
        if (
            not hasattr(self, "_model_artifact_sig")
            or not hasattr(self, "_model_reload_cooldown_s")
        ):
            return False

        now_ts = time.monotonic()
        current_sig = self._model_artifact_signature()
        prev_sig = str(getattr(self, "_model_artifact_sig", ""))
        sig_changed = bool(current_sig) and current_sig != prev_sig
        ready = self._models_ready_for_runtime()
        last_reload_ts = float(getattr(self, "_last_model_reload_attempt_ts", 0.0))
        cooldown_s = float(getattr(self, "_model_reload_cooldown_s", 15.0))
        cooldown_ok = (
            (now_ts - last_reload_ts)
            >= cooldown_s
        )
        should_reload = bool(force) or ((not ready or sig_changed) and cooldown_ok)
        if not should_reload:
            return False

        self._last_model_reload_attempt_ts = now_ts
        before_ready = ready
        ok = bool(self._load_models())
        self._model_artifact_sig = self._model_artifact_signature()
        after_ready = self._models_ready_for_runtime()

        if before_ready != after_ready or sig_changed:
            log.info(
                "Predictor model reload (%s): ok=%s scaler=%s ensemble=%s forecaster=%s",
                reason or "runtime",
                ok,
                bool(self.processor is not None and getattr(self.processor, "is_fitted", False)),
                bool(self.ensemble is not None),
                bool(self.forecaster is not None),
            )
        return ok

    def _extract_trained_stocks_from_ensemble(self) -> list[str]:
        if self.ensemble is None:
            return []
        try:
            info = self.ensemble.get_model_info()
            if not isinstance(info, dict):
                return []
            out = [
                str(x).strip()
                for x in list(info.get("trained_stock_codes", []) or [])
                if str(x).strip()
            ]
            return out
        except Exception as e:
            log.debug("Failed reading trained stocks from ensemble metadata: %s", e)
            return []

    @staticmethod
    def _normalize_stock_code(value: object) -> str:
        code = "".join(ch for ch in str(value or "").strip() if ch.isdigit())
        return code if len(code) == 6 else ""

    @classmethod
    def _sanitize_last_train_map(cls, payload: object) -> dict[str, str]:
        if not isinstance(payload, dict):
            return {}
        out: dict[str, str] = {}
        for raw_code, raw_ts in payload.items():
            code = cls._normalize_stock_code(raw_code)
            if not code:
                continue
            ts = str(raw_ts or "").strip()
            if not ts:
                continue
            out[code] = ts
        return out

    def _extract_trained_stock_last_train_from_ensemble(self) -> dict[str, str]:
        if self.ensemble is None:
            return {}
        try:
            info = self.ensemble.get_model_info()
            if not isinstance(info, dict):
                return {}
            return self._sanitize_last_train_map(
                info.get("trained_stock_last_train", {})
            )
        except Exception as e:
            log.debug(
                "Failed reading trained-stock last-train from ensemble metadata: %s",
                e,
            )
            return {}

    def _load_trained_stocks_from_manifest(self, ensemble_path: Path) -> list[str]:
        """Fallback for legacy model loads when ensemble metadata is missing."""
        try:
            stem = Path(ensemble_path).stem
            p = Path(ensemble_path).parent / f"model_manifest_{stem}.json"
            if not p.exists():
                return []
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return []
            out = [
                str(x).strip()
                for x in list(data.get("trained_stock_codes", []) or [])
                if str(x).strip()
            ]
            return out
        except Exception as e:
            log.debug("Failed reading trained stocks from manifest %s: %s", ensemble_path, e)
            return []

    def _load_trained_stock_last_train_from_manifest(
        self,
        ensemble_path: Path,
    ) -> dict[str, str]:
        """Fallback for legacy loads when ensemble metadata is missing."""
        try:
            stem = Path(ensemble_path).stem
            p = Path(ensemble_path).parent / f"model_manifest_{stem}.json"
            if not p.exists():
                return {}
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            return self._sanitize_last_train_map(
                data.get("trained_stock_last_train", {})
            )
        except Exception as e:
            log.debug(
                "Failed reading trained-stock last-train from manifest %s: %s",
                ensemble_path,
                e,
            )
            return {}

    def _load_trained_stocks_from_learner_state(
        self,
        interval: str,
        horizon: int,
    ) -> list[str]:
        """
        Last-resort fallback for legacy artifacts missing stock metadata.
        Uses persisted learner_state only when interval+horizon match.
        """
        try:
            p = Path(CONFIG.data_dir) / "learner_state.json"
            if not p.exists():
                return []
            raw = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return []
            data = raw.get("_data", raw)
            if not isinstance(data, dict):
                return []

            iv = str(data.get("last_interval", "")).strip().lower()
            try:
                hz = int(data.get("last_horizon", 0) or 0)
            except (TypeError, ValueError):
                hz = 0

            if iv != str(interval).strip().lower() or hz != int(horizon):
                return []

            replay = data.get("replay", {})
            if not isinstance(replay, dict):
                replay = {}
            candidates = list(replay.get("buffer", []) or [])

            if not candidates:
                rot = data.get("rotator", {})
                if isinstance(rot, dict):
                    candidates = list(rot.get("processed", []) or [])

            out: list[str] = []
            seen: set[str] = set()
            for x in candidates:
                code = "".join(c for c in str(x).strip() if c.isdigit())
                if len(code) != 6 or code in seen:
                    continue
                seen.add(code)
                out.append(code)
            return out
        except Exception as e:
            log.debug("Failed reading trained stocks from learner state: %s", e)
            return []

    def get_trained_stock_codes(self, limit: int | None = None) -> list[str]:
        """
        Return stock codes used in the currently loaded training artifact.
        """
        codes = list(self._trained_stock_codes)
        if limit is not None:
            try:
                n = max(0, int(limit))
                return codes[:n]
            except (TypeError, ValueError) as e:
                log.debug("Invalid trained stock code limit %r: %s", limit, e)
        return codes

    def get_trained_stock_last_train(self) -> dict[str, str]:
        """Return per-stock last-train timestamps from loaded model artifacts."""
        return dict(self._trained_stock_last_train or {})

    def _load_forecaster(self):
        """Load TCN forecaster for price curve prediction."""
        try:
            import torch

            from models.networks import TCNModel

            self.forecaster = None
            self._forecaster_horizon = 0
            forecast_path = self._find_best_forecaster_checkpoint(
                CONFIG.MODEL_DIR
            )

            if forecast_path is None:
                log.debug(
                    "No matching forecaster model found for interval=%s horizon=%s",
                    self.interval,
                    self.horizon,
                )
                return

            if (
                self.processor is not None
                and not self.processor._verify_artifact_checksum(forecast_path)
            ):
                return

            allow_unsafe = bool(
                getattr(
                    getattr(CONFIG, "model", None),
                    "allow_unsafe_artifact_load",
                    False,
                )
            )

            def _load_checkpoint(weights_only: bool):
                try:
                    from utils.atomic_io import torch_load

                    return torch_load(
                        forecast_path,
                        map_location="cpu",
                        weights_only=weights_only,
                    )
                except TypeError as exc:
                    # Older torch versions may not support weights_only.
                    if not allow_unsafe:
                        raise RuntimeError(
                            "Secure forecaster load requires torch weights_only support"
                        ) from exc
                    return torch.load(forecast_path, map_location="cpu")
                except ImportError as exc:
                    if not allow_unsafe:
                        raise RuntimeError(
                            "Secure forecaster load requires utils.atomic_io.torch_load"
                        ) from exc
                    return torch.load(forecast_path, map_location="cpu")

            try:
                data = _load_checkpoint(weights_only=True)
            except Exception as exc:
                if not allow_unsafe:
                    log.error(
                        "Forecaster secure load failed for %s and unsafe fallback is disabled: %s",
                        forecast_path,
                        exc,
                    )
                    return
                log.warning(
                    "Forecaster weights-only load failed for %s; "
                    "falling back to unsafe legacy checkpoint load: %s",
                    forecast_path,
                    exc,
                )
                data = _load_checkpoint(weights_only=False)

            required_keys = {"input_size", "horizon", "arch", "state_dict"}
            if not required_keys.issubset(data.keys()):
                log.warning(
                    f"Forecaster checkpoint missing keys: "
                    f"{required_keys - set(data.keys())}"
                )
                return

            horizon = int(data.get("horizon", 0) or 0)
            if horizon <= 0:
                log.warning("Forecaster checkpoint has invalid horizon: %s", horizon)
                return

            self.forecaster = TCNModel(
                input_size=data["input_size"],
                hidden_size=data["arch"]["hidden_size"],
                num_classes=horizon,
                dropout=data["arch"]["dropout"],
            )
            self.forecaster.load_state_dict(data["state_dict"])
            self.forecaster.eval()
            self._forecaster_horizon = horizon

            log.info(
                "Forecaster loaded: %s (interval=%s, horizon=%s)",
                forecast_path.name,
                self.interval,
                horizon,
            )

        except ImportError:
            log.debug("TCNModel not available - forecaster disabled")
            self.forecaster = None
            self._forecaster_horizon = 0
        except Exception as e:
            log.debug(f"Forecaster not loaded: {e}")
            self.forecaster = None
            self._forecaster_horizon = 0

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
        history_allow_online: bool = True,
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
            history_allow_online: If False, use local cache/DB/session only

        Returns:
            Prediction object with all fields populated
        """
        with self._predict_lock:
            self._maybe_reload_models(reason="predict")
            interval = self._normalize_interval_token(interval)
            horizon = int(forecast_minutes or self.horizon)
            lookback = int(
                lookback_bars or self._default_lookback_bars(interval)
            )

            code = self._clean_code(stock_code)
            cache_key = (
                f"{code}:{interval}:{horizon}:"
                f"{'rt' if bool(use_realtime_price) else 'hist'}"
            )
            cache_ttl = self._get_cache_ttl(
                use_realtime=bool(use_realtime_price),
                interval=interval,
            )

            if not skip_cache:
                cached = self._get_cached_prediction(
                    cache_key, ttl=cache_ttl
                )
                if cached is not None:
                    return cached

            pred = Prediction(
                stock_code=code,
                timestamp=datetime.now(),
                interval=interval,
                horizon=horizon,
            )

            try:
                min_rows = int(
                    getattr(
                        self.feature_engine,
                        "MIN_ROWS",
                        CONFIG.SEQUENCE_LENGTH,
                    )
                )
                required_rows = int(max(CONFIG.SEQUENCE_LENGTH, min_rows))

                df = self._fetch_data(
                    code,
                    interval,
                    lookback,
                    use_realtime_price,
                    history_allow_online=bool(history_allow_online),
                )

                if not self._has_required_rows(df, required_rows):
                    data_len = len(df) if isinstance(df, pd.DataFrame) else 0
                    short_ready = self._has_required_rows(
                        df,
                        max(20, int(required_rows // 3)),
                    )
                    if short_ready and self._bootstrap_short_history_prediction(
                        pred,
                        df,
                        horizon=horizon,
                        required_rows=required_rows,
                    ):
                        # Keep short-history fallback aligned with news context.
                        self._apply_news_influence(pred, code, interval)
                        self._set_cached_prediction(cache_key, pred)
                        return pred
                    self._populate_minimal_snapshot(pred, code)
                    pred.warnings.append(
                        f"Insufficient data: got {data_len} bars, "
                        f"need {required_rows}"
                    )
                    return pred

                pred.stock_name = self._get_stock_name(code, df)
                pred.current_price = float(df["close"].iloc[-1])
                pred.price_history = df["close"].tail(180).tolist()

                df = self.feature_engine.create_features(df)
                self._extract_technicals(df, pred)

                X = self.processor.prepare_inference_sequence(
                    df, self._feature_cols
                )

                if self.ensemble:
                    self._apply_ensemble_prediction(X, pred)

                news_bias = self._apply_news_influence(pred, code, interval)
                self._refresh_prediction_uncertainty(pred)
                self._apply_tail_risk_guard(pred)

                try:
                    pred.predicted_prices = self._generate_forecast(
                        X,
                        pred.current_price,
                        horizon,
                        pred.atr_pct_value,
                        sequence_signature=self._sequence_signature(X),
                        seed_context=f"{code}:{interval}",
                        recent_prices=df["close"].tail(min(560, len(df))).tolist(),
                        news_bias=news_bias,
                    )
                except TypeError:
                    # Compatibility for tests/mocks overriding _generate_forecast.
                    pred.predicted_prices = self._generate_forecast(
                        X,
                        pred.current_price,
                        horizon,
                        pred.atr_pct_value,
                        sequence_signature=self._sequence_signature(X),
                        seed_context=f"{code}:{interval}",
                        recent_prices=df["close"].tail(min(560, len(df))).tolist(),
                    )
                self._build_prediction_bands(pred)
                pred.levels = self._calculate_levels(pred)
                pred.position = self._calculate_position(pred)
                self._generate_reasons(pred)

                self._set_cached_prediction(cache_key, pred)

            except Exception as e:
                log.error(
                    f"Prediction failed for {code}: {e}",
                    exc_info=True
                )
                pred.warnings.append(
                    f"Prediction error: {type(e).__name__}: {str(e)}"
                )

            return pred

    @staticmethod
    def _has_required_rows(df: pd.DataFrame | None, required_rows: int) -> bool:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return False
        try:
            return int(len(df)) >= int(max(1, required_rows))
        except Exception as e:
            log.debug("Failed required-row check (required=%r): %s", required_rows, e)
            return False

    def _populate_minimal_snapshot(self, pred: Prediction, code: str) -> None:
        """
        Populate minimal quote context when full feature history is unavailable.
        This keeps UI chart/price widgets responsive instead of returning all zeros.
        """
        if pred.current_price > 0:
            return
        fetcher = getattr(self, "fetcher", None)
        if fetcher is None:
            return

        try:
            quote = fetcher.get_realtime(code)
        except Exception as e:
            log.debug("Realtime quote unavailable for %s: %s", code, e)
            quote = None

        if quote is not None:
            try:
                px = float(getattr(quote, "price", 0) or 0)
            except (TypeError, ValueError):
                px = 0.0
            if px > 0:
                pred.current_price = px
                pred.price_history = [px]
                qname = str(getattr(quote, "name", "") or "").strip()
                if qname and not pred.stock_name:
                    pred.stock_name = qname
                return

        fallback_iv = self._normalize_interval_token(
            getattr(pred, "interval", None) or self.interval
        )
        try:
            df = fetcher.get_history(
                code,
                interval=fallback_iv,
                bars=1,
                use_cache=True,
                update_db=False,
                allow_online=False,
            )
        except TypeError:
            df = fetcher.get_history(
                code,
                interval=fallback_iv,
                bars=1,
                use_cache=True,
                update_db=False,
            )
        except Exception as e:
            log.debug("Fallback history fetch failed for %s: %s", code, e)
            df = None

        if isinstance(df, pd.DataFrame) and not df.empty:
            try:
                px = float(df["close"].iloc[-1] or 0)
            except (TypeError, ValueError):
                px = 0.0
            if px > 0:
                pred.current_price = px
                pred.price_history = [px]

    def _bootstrap_short_history_prediction(
        self,
        pred: Prediction,
        df: pd.DataFrame,
        *,
        horizon: int,
        required_rows: int,
    ) -> bool:
        """
        Produce a lightweight fallback prediction when history is short.
        This avoids HOLD(0%) outputs during startup warm-up windows.
        """
        if df is None or df.empty or "close" not in df.columns:
            return False

        close = pd.to_numeric(df["close"], errors="coerce")
        close = close.replace([np.inf, -np.inf], np.nan).dropna()
        if len(close) < 20:
            return False

        current_price = float(close.iloc[-1] or 0.0)
        if current_price <= 0:
            return False

        pred.stock_name = self._get_stock_name(pred.stock_code, df)
        pred.current_price = current_price
        pred.price_history = [
            float(v) for v in close.tail(180).tolist() if float(v) > 0
        ]
        if not pred.price_history:
            pred.price_history = [current_price]

        recent = close.tail(min(36, len(close)))
        pct = recent.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        drift = float(pct.tail(min(12, len(pct))).mean()) if not pct.empty else 0.0
        vol = (
            float(pct.tail(min(20, len(pct))).std(ddof=0))
            if len(pct) > 1 else abs(drift) * 1.5
        )
        if not np.isfinite(drift):
            drift = 0.0
        if not np.isfinite(vol):
            vol = 0.0
        base_move = float(recent.iloc[-1] / max(float(recent.iloc[0]), 1e-8) - 1.0)
        score = (drift * 700.0) + (base_move * 6.0)
        direction = float(np.tanh(score))
        depth_ratio = float(min(1.0, len(close) / max(float(required_rows), 1.0)))

        pred.confidence = float(
            np.clip(0.36 + (0.24 * depth_ratio) + (0.28 * abs(direction)), 0.36, 0.78)
        )
        pred.raw_confidence = float(pred.confidence)
        pred.model_agreement = float(np.clip(0.52 + (0.35 * depth_ratio), 0.0, 1.0))
        pred.entropy = float(
            np.clip(
                0.62 - (0.35 * abs(direction)) + (0.25 * max(0.0, vol - 0.02)),
                0.05,
                0.95,
            )
        )
        pred.model_margin = float(np.clip(abs(direction) * 0.32, 0.01, 0.25))

        neutral = float(
            np.clip(
                0.45 - (0.20 * abs(direction)) + (0.90 * max(0.0, vol - 0.015)),
                0.12,
                0.74,
            )
        )
        up = float(np.clip((1.0 - neutral) * (0.5 + 0.5 * direction), 0.02, 0.95))
        down = float(max(0.0, 1.0 - neutral - up))
        total = up + neutral + down
        if total <= 0:
            up, neutral, down = 0.33, 0.34, 0.33
            total = 1.0
        pred.prob_up = float(up / total)
        pred.prob_neutral = float(neutral / total)
        pred.prob_down = float(down / total)

        if direction >= 0.55 and pred.confidence >= 0.70:
            pred.signal = Signal.STRONG_BUY
        elif direction >= 0.30 and pred.confidence >= 0.54:
            pred.signal = Signal.BUY
        elif direction <= -0.55 and pred.confidence >= 0.70:
            pred.signal = Signal.STRONG_SELL
        elif direction <= -0.30 and pred.confidence >= 0.54:
            pred.signal = Signal.SELL
        else:
            pred.signal = Signal.HOLD

        pred.signal_strength = float(
            np.clip((0.70 * abs(direction)) + (0.30 * pred.confidence), 0.0, 1.0)
        )
        if direction > 0.25:
            pred.trend = "UPTREND"
        elif direction < -0.25:
            pred.trend = "DOWNTREND"
        else:
            pred.trend = "SIDEWAYS"
        pred.atr_pct_value = float(np.clip(max(0.006, vol * 2.5), 0.006, 0.05))

        steps = max(1, int(horizon))
        pred.predicted_prices = []
        px = current_price
        drift = float(np.clip(drift, -0.02, 0.02))
        vol_step = float(np.clip(max(vol, abs(drift) * 1.2), 0.0008, 0.018))
        seed_n = len(close)
        for step in range(1, steps + 1):
            decay = max(0.15, 1.0 - (float(step) / float(max(steps, 1))) * 0.65)
            step_ret = (drift * decay) + (direction * 0.0006 * decay)
            wave = np.sin((float(step) + float(seed_n)) * 0.7) * (vol_step * 0.18)
            step_ret = float(np.clip(step_ret + wave, -0.04, 0.04))
            px = max(0.01, float(px) * (1.0 + step_ret))
            pred.predicted_prices.append(float(px))

        self._refresh_prediction_uncertainty(pred)
        self._apply_tail_risk_guard(pred)
        self._build_prediction_bands(pred)

        try:
            pred.levels = self._calculate_levels(pred)
        except Exception as e:
            log.debug(
                "Short-history levels calculation failed for %s: %s",
                pred.stock_code,
                e,
            )
        try:
            pred.position = self._calculate_position(pred)
        except Exception as e:
            log.debug(
                "Short-history position calculation failed for %s: %s",
                pred.stock_code,
                e,
            )
        try:
            self._generate_reasons(pred)
        except Exception as e:
            log.debug(
                "Short-history reason generation failed for %s: %s",
                pred.stock_code,
                e,
            )

        pred.warnings.append(
            "Short-history fallback used: "
            f"{len(close)} bars available (target {required_rows})"
        )
        return True

    def predict_quick_batch(
        self,
        stock_codes: list[str],
        use_realtime_price: bool = True,
        interval: str = None,
        lookback_bars: int = None,
        history_allow_online: bool = True,
    ) -> list[Prediction]:
        """Quick batch prediction without full forecasting."""
        with self._predict_lock:
            self._maybe_reload_models(reason="quick_batch")
            interval = self._normalize_interval_token(interval)
            lookback = int(
                lookback_bars or self._default_lookback_bars(interval)
            )

            predictions: list[Prediction] = []
            prepared: list[tuple[Prediction, np.ndarray]] = []

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

                    try:
                        df = self._fetch_data(
                            code,
                            interval,
                            lookback,
                            use_realtime_price,
                            history_allow_online=bool(history_allow_online),
                        )
                    except TypeError:
                        # Backward compatibility for tests/mocks overriding _fetch_data.
                        df = self._fetch_data(
                            code,
                            interval,
                            lookback,
                            use_realtime_price,
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
                    prepared.append((pred, X))

                except Exception as e:
                    log.debug(
                        f"Quick prediction failed for {stock_code}: {e}"
                    )

            if not prepared:
                return predictions

            if self.ensemble:
                try:
                    X_batch = np.concatenate(
                        [np.asarray(X, dtype=np.float32) for _, X in prepared],
                        axis=0,
                    )
                    ensemble_results = self.ensemble.predict_batch(
                        X_batch,
                        batch_size=max(8, min(256, len(prepared))),
                    )
                    if len(ensemble_results) == len(prepared):
                        for (pred, _), ens_pred in zip(
                            prepared,
                            ensemble_results,
                            strict=False,
                        ):
                            try:
                                self._apply_ensemble_result(ens_pred, pred)
                            except Exception as e:
                                log.debug(
                                    "Quick batch ensemble apply failed "
                                    f"for {pred.stock_code}: {e}"
                                )
                            predictions.append(pred)
                        return predictions
                    log.debug(
                        "Quick batch ensemble size mismatch: "
                        f"got {len(ensemble_results)}, expected {len(prepared)}"
                    )
                except Exception as e:
                    log.debug(
                        "Quick batch ensemble inference failed; "
                        f"falling back to single predict: {e}"
                    )

            # Fallback path: per-symbol predict.
            for pred, X in prepared:
                if self.ensemble:
                    try:
                        self._apply_ensemble_prediction(X, pred)
                    except Exception as e:
                        log.debug(
                            f"Quick single fallback failed for {pred.stock_code}: {e}"
                        )
                predictions.append(pred)

            return predictions

    def get_realtime_forecast_curve(
        self,
        stock_code: str,
        interval: str = None,
        horizon_steps: int = None,
        lookback_bars: int = None,
        use_realtime_price: bool = True,
        history_allow_online: bool = True,
    ) -> tuple[list[float], list[float]]:
        """
        Get real-time forecast curve for charting.

        Returns:
            (actual_prices, predicted_prices)
        """
        with self._predict_lock:
            self._maybe_reload_models(reason="forecast_curve")
            interval = self._normalize_interval_token(interval)
            horizon = max(
                1,
                int(horizon_steps if horizon_steps is not None else 30),
            )
            lookback = max(
                120,
                int(
                    lookback_bars
                    if lookback_bars is not None
                    else self._default_lookback_bars(interval)
                ),
            )

            code = self._clean_code(stock_code)

            try:
                min_rows = getattr(
                    self.feature_engine, 'MIN_ROWS',
                    CONFIG.SEQUENCE_LENGTH
                )
                window = int(
                    max(lookback, int(min_rows), int(CONFIG.SEQUENCE_LENGTH))
                )

                try:
                    df = self.fetcher.get_history(
                        code,
                        interval=interval,
                        bars=window,
                        use_cache=True,
                        update_db=False,
                        allow_online=bool(history_allow_online),
                    )
                except TypeError:
                    df = self.fetcher.get_history(
                        code,
                        interval=interval,
                        bars=window,
                        use_cache=True,
                        update_db=False,
                    )

                # Merge latest session bars (including partial intraday bars)
                # so realtime guessed curve follows the current live candle.
                if interval in {"1m", "3m", "5m", "15m", "30m", "60m", "1h"}:
                    try:
                        from data.session_cache import get_session_bar_cache

                        s_df = get_session_bar_cache().read_history(
                            symbol=code,
                            interval=interval,
                            bars=window,
                            final_only=False,
                        )
                        if s_df is not None and not s_df.empty:
                            parts: list[pd.DataFrame] = []
                            for part in (df, s_df):
                                if part is None or part.empty:
                                    continue
                                p = part.copy()
                                if not isinstance(p.index, pd.DatetimeIndex):
                                    if "datetime" in p.columns:
                                        p["datetime"] = pd.to_datetime(
                                            p["datetime"],
                                            errors="coerce",
                                        )
                                        p = p.dropna(subset=["datetime"]).set_index("datetime")
                                    elif "timestamp" in p.columns:
                                        p["datetime"] = pd.to_datetime(
                                            p["timestamp"],
                                            errors="coerce",
                                        )
                                        p = p.dropna(subset=["datetime"]).set_index("datetime")
                                parts.append(p)
                            if parts:
                                merged = pd.concat(parts, axis=0)
                                if isinstance(merged.index, pd.DatetimeIndex):
                                    merged = merged[~merged.index.duplicated(keep="last")]
                                    merged = merged.sort_index()
                                df = merged
                    except Exception as e:
                        log.debug("Realtime session-merge skipped for %s: %s", code, e)

                if (
                    df is None
                    or df.empty
                    or len(df) < CONFIG.SEQUENCE_LENGTH
                    or len(df) < min_rows
                ):
                    return [], []

                # Real-time guess should follow the latest candles window.
                df = df.tail(window).copy()

                if use_realtime_price:
                    try:
                        quote = self.fetcher.get_realtime(code)
                        if quote and float(getattr(quote, "price", 0) or 0) > 0:
                            px = float(quote.price)
                            df.loc[df.index[-1], "close"] = px
                            df.loc[df.index[-1], "high"] = max(
                                float(df["high"].iloc[-1]),
                                px,
                            )
                            df.loc[df.index[-1], "low"] = min(
                                float(df["low"].iloc[-1]),
                                px,
                            )
                    except Exception as e:
                        log.debug(
                            "Realtime tail merge skipped while building forecast curve for %s: %s",
                            code,
                            e,
                        )

                df = self._sanitize_history_df(df, interval)
                if (
                    df is None
                    or df.empty
                    or len(df) < CONFIG.SEQUENCE_LENGTH
                    or len(df) < min_rows
                ):
                    return [], []

                actual = df["close"].tail(min(lookback, len(df))).tolist()
                current_price = float(df["close"].iloc[-1])

                scaler_ready = bool(
                    self.processor is not None
                    and getattr(self.processor, "is_fitted", True)
                )
                if not scaler_ready:
                    self._maybe_reload_models(reason="forecast_curve_scaler_missing")
                    scaler_ready = bool(
                        self.processor is not None
                        and getattr(self.processor, "is_fitted", True)
                    )
                if not scaler_ready:
                    log.debug(
                        "Realtime forecast skipped for %s/%s: scaler unavailable",
                        code,
                        interval,
                    )
                    return actual, []

                df = self.feature_engine.create_features(df)
                X = self.processor.prepare_inference_sequence(
                    df, self._feature_cols
                )

                atr_pct = self._get_atr_pct(df)
                news_score, news_conf, news_count = self._get_news_sentiment(
                    code,
                    interval,
                )
                news_bias = self._compute_news_bias(
                    news_score,
                    news_conf,
                    news_count,
                    interval,
                )

                try:
                    predicted = self._generate_forecast(
                        X,
                        current_price,
                        horizon,
                        atr_pct,
                        sequence_signature=self._sequence_signature(X),
                        seed_context=f"{code}:{interval}",
                        recent_prices=actual,
                        news_bias=news_bias,
                    )
                except TypeError:
                    predicted = self._generate_forecast(
                        X,
                        current_price,
                        horizon,
                        atr_pct,
                        sequence_signature=self._sequence_signature(X),
                        seed_context=f"{code}:{interval}",
                        recent_prices=actual,
                    )
                predicted = self._stabilize_forecast_curve(
                    predicted,
                    current_price=current_price,
                    atr_pct=atr_pct,
                )

                return actual, predicted

            except Exception as e:
                log.warning(f"Forecast curve failed for {code}: {e}")
                return [], []

    @staticmethod
    def _stabilize_forecast_curve(
        values: list[float],
        *,
        current_price: float,
        atr_pct: float,
    ) -> list[float]:
        """
        Clamp/smooth forecast curve so one noisy step cannot create
        unrealistic V-shapes in real-time chart updates.
        """
        if not values:
            return []
        px0 = float(current_price or 0.0)
        if px0 <= 0:
            return [float(v) for v in values if float(v) > 0]

        vol = float(np.nan_to_num(atr_pct, nan=0.02, posinf=0.02, neginf=0.02))
        if vol <= 0:
            vol = 0.02

        # Per-step clamp for intraday visualization stability.
        max_step = float(np.clip(vol * 0.55, 0.0025, 0.015))
        prev = px0
        out: list[float] = []
        for raw in values:
            try:
                p = float(raw)
            except (TypeError, ValueError):
                p = prev
            if not np.isfinite(p) or p <= 0:
                p = prev

            lo = prev * (1.0 - max_step)
            hi = prev * (1.0 + max_step)
            p = float(np.clip(p, lo, hi))

            # Mild EMA smoothing to reduce sawtooth artifacts.
            p = float((0.70 * p) + (0.30 * prev))
            out.append(p)
            prev = p
        return out

    def get_top_picks(
        self,
        stock_codes: list[str],
        n: int = 10,
        signal_type: str = "buy",
    ) -> list[Prediction]:
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
        self._apply_ensemble_result(ensemble_pred, pred)

    def _apply_ensemble_result(
        self, ensemble_pred, pred: Prediction
    ):
        """Apply a precomputed ensemble result to a prediction object."""

        probs = np.asarray(
            getattr(ensemble_pred, "probabilities", [0.33, 0.34, 0.33]),
            dtype=float,
        ).reshape(-1)
        n_classes = len(probs)

        pred.prob_down = float(probs[0]) if n_classes > 0 else 0.33
        pred.prob_neutral = float(probs[1]) if n_classes > 1 else 0.34
        pred.prob_up = float(probs[2]) if n_classes > 2 else 0.33

        pred.prob_down = max(0.0, min(1.0, pred.prob_down))
        pred.prob_neutral = max(0.0, min(1.0, pred.prob_neutral))
        pred.prob_up = max(0.0, min(1.0, pred.prob_up))
        p_sum = pred.prob_down + pred.prob_neutral + pred.prob_up
        if p_sum > 0:
            pred.prob_down /= p_sum
            pred.prob_neutral /= p_sum
            pred.prob_up /= p_sum
        else:
            pred.prob_down, pred.prob_neutral, pred.prob_up = (
                0.33,
                0.34,
                0.33,
            )

        pred.confidence = float(
            max(0.0, min(1.0, getattr(ensemble_pred, "confidence", 0.0)))
        )
        pred.raw_confidence = float(
            max(
                0.0,
                min(
                    1.0,
                    getattr(
                        ensemble_pred,
                        "raw_confidence",
                        getattr(ensemble_pred, "confidence", 0.0),
                    ),
                ),
            )
        )

        pred.model_agreement = float(
            getattr(ensemble_pred, "agreement", 1.0)
        )
        pred.entropy = float(
            getattr(ensemble_pred, "entropy", 0.0)
        )
        pred.model_margin = float(
            getattr(ensemble_pred, "margin", 0.10)
        )
        pred.brier_score = float(
            max(0.0, getattr(ensemble_pred, "brier_score", 0.0))
        )

        pred.signal = self._determine_signal(ensemble_pred, pred)
        pred.signal_strength = self._calculate_signal_strength(
            ensemble_pred, pred
        )
        self._refresh_prediction_uncertainty(pred)
        self._apply_high_precision_gate(pred)
        self._apply_runtime_signal_quality_gate(pred)
        self._apply_tail_risk_guard(pred)

    @staticmethod
    def _append_warning_once(pred: Prediction, message: str) -> None:
        """Append warning only once to avoid noisy duplicates."""
        text = str(message).strip()
        if not text:
            return
        existing = [str(x) for x in list(pred.warnings or [])]
        if text in existing:
            return
        pred.warnings.append(text)

    def _refresh_prediction_uncertainty(self, pred: Prediction) -> None:
        """
        Derive uncertainty and tail-risk from signal quality metrics.

        Also moderates confidence when entropy/adverse-risk is high to avoid
        over-confident chart narratives.
        """
        conf = float(np.clip(getattr(pred, "confidence", 0.0), 0.0, 1.0))
        raw_conf = float(np.clip(getattr(pred, "raw_confidence", conf), 0.0, 1.0))
        agreement = float(np.clip(getattr(pred, "model_agreement", 1.0), 0.0, 1.0))
        entropy = float(np.clip(getattr(pred, "entropy", 0.0), 0.0, 1.0))
        margin = float(np.clip(getattr(pred, "model_margin", 0.10), 0.0, 1.0))
        edge = float(abs(float(pred.prob_up) - float(pred.prob_down)))

        atr = float(np.nan_to_num(getattr(pred, "atr_pct_value", 0.02), nan=0.02))
        atr = float(np.clip(atr, 0.003, 0.12))
        vol_scale = float(np.clip(atr / 0.030, 0.0, 2.0))

        if pred.signal in (Signal.BUY, Signal.STRONG_BUY):
            adverse_prob = float(np.clip(pred.prob_down, 0.0, 1.0))
        elif pred.signal in (Signal.SELL, Signal.STRONG_SELL):
            adverse_prob = float(np.clip(pred.prob_up, 0.0, 1.0))
        else:
            adverse_prob = float(
                np.clip(max(pred.prob_up, pred.prob_down), 0.0, 1.0)
            )

        quality = (
            (0.46 * conf)
            + (0.22 * agreement)
            + (0.20 * (1.0 - entropy))
            + (0.12 * margin)
        )
        uncertainty = float(
            np.clip(
                (1.0 - quality)
                + (0.18 * (1.0 - edge))
                + (0.14 * vol_scale),
                0.0,
                1.0,
            )
        )
        tail_risk = float(
            np.clip(
                (0.58 * adverse_prob)
                + (0.24 * entropy)
                + (0.18 * max(0.0, vol_scale - 1.0)),
                0.0,
                1.0,
            )
        )

        # Confidence moderation only when risk is materially elevated.
        penalty = (
            (max(0.0, entropy - 0.62) * 0.35)
            + (max(0.0, tail_risk - 0.58) * 0.30)
            + (max(0.0, 0.55 - agreement) * 0.28)
        )
        if margin < 0.04:
            penalty += 0.05
        if penalty > 0.0:
            old_conf = conf
            conf = float(np.clip(conf - penalty, 0.0, 1.0))
            pred.confidence = conf
            if (old_conf - conf) >= 0.08:
                self._append_warning_once(
                    pred,
                    "Confidence moderated due to uncertainty/tail-risk conditions",
                )

        if raw_conf > 0 and conf > raw_conf:
            pred.confidence = raw_conf

        pred.uncertainty_score = float(uncertainty)
        pred.tail_risk_score = float(tail_risk)

    def _apply_tail_risk_guard(self, pred: Prediction) -> None:
        """
        Block actionable signals when adverse-tail probability is too high.
        """
        if pred.signal == Signal.HOLD:
            return
        conf = float(np.clip(getattr(pred, "confidence", 0.0), 0.0, 1.0))
        tail_risk = float(np.clip(getattr(pred, "tail_risk_score", 0.0), 0.0, 1.0))
        uncertainty = float(
            np.clip(getattr(pred, "uncertainty_score", 0.0), 0.0, 1.0)
        )

        reasons: list[str] = []
        if tail_risk >= 0.72 and conf < 0.88:
            reasons.append(f"tail_risk {tail_risk:.2f}")
        if uncertainty >= 0.82 and conf < 0.86:
            reasons.append(f"uncertainty {uncertainty:.2f}")

        if not reasons:
            return

        old_signal = pred.signal.value
        pred.signal = Signal.HOLD
        pred.signal_strength = min(float(pred.signal_strength), 0.49)
        self._append_warning_once(
            pred,
            "Tail-risk guard filtered signal "
            f"{old_signal} -> HOLD ({'; '.join(reasons[:2])})",
        )

    def _build_prediction_bands(self, pred: Prediction) -> None:
        """
        Build per-step prediction intervals to visualize uncertainty.
        """
        values = [
            float(v)
            for v in list(getattr(pred, "predicted_prices", []) or [])
            if float(v) > 0 and np.isfinite(float(v))
        ]
        if not values:
            pred.predicted_prices_low = []
            pred.predicted_prices_high = []
            return

        uncertainty = float(
            np.clip(getattr(pred, "uncertainty_score", 0.5), 0.0, 1.0)
        )
        tail_risk = float(np.clip(getattr(pred, "tail_risk_score", 0.5), 0.0, 1.0))
        conf = float(np.clip(getattr(pred, "confidence", 0.0), 0.0, 1.0))
        atr = float(
            np.clip(
                np.nan_to_num(getattr(pred, "atr_pct_value", 0.02), nan=0.02),
                0.003,
                0.12,
            )
        )
        n = max(1, len(values))

        base_width = float(
            np.clip(
                atr
                * (0.60 + (1.10 * uncertainty) + (0.75 * tail_risk))
                * (1.05 + (0.35 * (1.0 - conf))),
                0.004,
                0.30,
            )
        )

        lows: list[float] = []
        highs: list[float] = []
        for i, px in enumerate(values, start=1):
            growth = 1.0 + (float(i) / float(n)) * (0.85 + (0.65 * uncertainty))
            width = float(np.clip(base_width * growth, 0.004, 0.35))
            lo = max(0.01, float(px) * (1.0 - width))
            hi = max(lo + 1e-6, float(px) * (1.0 + width))
            lows.append(float(lo))
            highs.append(float(hi))

        pred.predicted_prices_low = lows
        pred.predicted_prices_high = highs

    def _apply_high_precision_gate(self, pred: Prediction) -> None:
        """Optionally downgrade weak actionable predictions to HOLD."""
        cfg = self._high_precision
        if not cfg or cfg.get("enabled", 0.0) <= 0:
            return
        if pred.signal == Signal.HOLD:
            return

        reasons: list[str] = []

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

    def _apply_runtime_signal_quality_gate(self, pred: Prediction) -> None:
        """
        Always-on runtime guard to reduce low-quality actionable signals.
        This improves precision by preferring HOLD when edge quality is weak.
        """
        if pred.signal == Signal.HOLD:
            return

        reasons: list[str] = []
        conf = float(np.clip(pred.confidence, 0.0, 1.0))
        agreement = float(np.clip(pred.model_agreement, 0.0, 1.0))
        entropy = float(np.clip(pred.entropy, 0.0, 1.0))
        edge = float(pred.prob_up) - float(pred.prob_down)
        trend = str(pred.trend).upper()

        if pred.signal in (Signal.BUY, Signal.STRONG_BUY) and edge < 0.03:
            reasons.append(f"edge {edge:.2f} too weak for long")
        if pred.signal in (Signal.SELL, Signal.STRONG_SELL) and edge > -0.03:
            reasons.append(f"edge {edge:.2f} too weak for short")
        if agreement < 0.50 and conf < 0.78:
            reasons.append(
                f"agreement/conf weak ({agreement:.2f}/{conf:.2f})"
            )
        if entropy > 0.78 and conf < 0.80:
            reasons.append(f"high entropy {entropy:.2f}")
        if trend == "SIDEWAYS" and conf < 0.72:
            reasons.append("sideways regime with low confidence")
        if (
            trend == "UPTREND"
            and pred.signal in (Signal.SELL, Signal.STRONG_SELL)
            and conf < 0.86
        ):
            reasons.append("counter-trend short lacks conviction")
        if (
            trend == "DOWNTREND"
            and pred.signal in (Signal.BUY, Signal.STRONG_BUY)
            and conf < 0.86
        ):
            reasons.append("counter-trend long lacks conviction")
        if pred.atr_pct_value >= 0.04 and conf < 0.76:
            reasons.append("high volatility requires stronger confidence")

        if not reasons:
            return

        old_signal = pred.signal.value
        pred.signal = Signal.HOLD
        pred.signal_strength = min(float(pred.signal_strength), 0.49)
        pred.warnings.append(
            "Runtime quality gate filtered signal "
            f"{old_signal} -> HOLD ({'; '.join(reasons[:3])})"
        )

    # =========================================================================
    # =========================================================================

    def _get_cache_ttl(self, use_realtime: bool, interval: str) -> float:
        """
        Adaptive cache TTL.
        Real-time paths get shorter TTL to reduce stale guesses.
        """
        base = float(self._CACHE_TTL)
        if not use_realtime:
            return base
        intraday = str(interval).lower() in {"1m", "3m", "5m", "15m", "30m", "60m"}
        if intraday:
            return float(max(0.2, min(base, self._CACHE_TTL_REALTIME)))
        return float(max(0.2, min(base, 2.0)))

    def _get_cached_prediction(
        self, cache_key: str, ttl: float | None = None
    ) -> Prediction | None:
        """Get cached prediction if still valid."""
        ttl_s = float(self._CACHE_TTL if ttl is None else ttl)
        with self._cache_lock:
            entry = self._pred_cache.get(cache_key)
            if entry is not None:
                ts, pred = entry
                if (time.time() - ts) < ttl_s:
                    return copy.deepcopy(pred)
                del self._pred_cache[cache_key]
        return None

    def _set_cached_prediction(self, cache_key: str, pred: Prediction):
        """Cache a prediction result with bounded size."""
        with self._cache_lock:
            self._pred_cache[cache_key] = (time.time(), copy.deepcopy(pred))

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

    def _news_cache_ttl(self, interval: str) -> float:
        """News sentiment cache TTL by interval profile."""
        if self._is_intraday_interval(interval):
            return float(self._NEWS_CACHE_TTL_INTRADAY)
        return float(self._NEWS_CACHE_TTL_SWING)

    def _ensure_news_cache_state(self) -> None:
        """Lazy-init news cache fields for tests using Predictor.__new__()."""
        if not hasattr(self, "_news_cache") or self._news_cache is None:
            self._news_cache = {}
        if not hasattr(self, "_news_cache_lock") or self._news_cache_lock is None:
            self._news_cache_lock = threading.Lock()

    def _get_news_sentiment(
        self,
        stock_code: str,
        interval: str,
    ) -> tuple[float, float, int]:
        """
        Return (sentiment, confidence, count) for stock news.
        Sentiment is in [-1, 1], confidence in [0, 1].
        """
        self._ensure_news_cache_state()
        code = self._clean_code(stock_code)
        if not code:
            return 0.0, 0.0, 0

        ttl = self._news_cache_ttl(interval)
        now = time.time()
        with self._news_cache_lock:
            rec = self._news_cache.get(code)
            if rec is not None:
                ts, s, conf, cnt = rec
                if (now - float(ts)) < ttl:
                    return float(s), float(conf), int(cnt)

        try:
            from data.news import get_news_aggregator

            agg = get_news_aggregator()
            summary = agg.get_sentiment_summary(code)
            score = float(summary.get("overall_sentiment", 0.0) or 0.0)
            conf = float(summary.get("confidence", 0.0) or 0.0)
            cnt = int(summary.get("total", 0) or 0)

            score = float(np.clip(score, -1.0, 1.0))
            conf = float(np.clip(conf, 0.0, 1.0))
            cnt = max(0, int(cnt))

        except Exception as e:
            log.debug("News sentiment lookup failed for %s: %s", code, e)
            score, conf, cnt = 0.0, 0.0, 0

        with self._news_cache_lock:
            self._news_cache[code] = (now, float(score), float(conf), int(cnt))
            if len(self._news_cache) > 500:
                oldest = sorted(
                    self._news_cache.items(),
                    key=lambda kv: float(kv[1][0]),
                )[:180]
                for k, _ in oldest:
                    self._news_cache.pop(k, None)

        return float(score), float(conf), int(cnt)

    def _compute_news_bias(
        self,
        sentiment: float,
        confidence: float,
        count: int,
        interval: str,
    ) -> float:
        """
        Convert news metrics into a bounded directional bias.
        Positive => bullish tilt, negative => bearish tilt.
        """
        s = float(np.clip(np.nan_to_num(sentiment, nan=0.0), -1.0, 1.0))
        conf = float(np.clip(np.nan_to_num(confidence, nan=0.0), 0.0, 1.0))
        cnt = max(0, int(count))

        coverage = float(np.clip(cnt / 24.0, 0.0, 1.0))
        eff_conf = conf * (0.30 + (0.70 * coverage))
        raw = s * eff_conf
        cap = 0.14 if self._is_intraday_interval(interval) else 0.20
        return float(np.clip(raw, -cap, cap))

    def _apply_news_influence(
        self,
        pred: Prediction,
        stock_code: str,
        interval: str,
    ) -> float:
        """
        Blend news sentiment into class probabilities and confidence.
        Returns the directional bias used for forecast shaping.
        """
        sentiment, conf, count = self._get_news_sentiment(stock_code, interval)
        pred.news_sentiment = float(sentiment)
        pred.news_confidence = float(conf)
        pred.news_count = int(count)

        news_bias = self._compute_news_bias(sentiment, conf, count, interval)
        if abs(news_bias) <= 1e-8:
            return 0.0

        shift = float(min(0.18, abs(news_bias) * 0.55))
        if news_bias > 0:
            moved = min(float(pred.prob_down), shift)
            pred.prob_down = float(pred.prob_down - moved)
            pred.prob_up = float(pred.prob_up + moved)
        else:
            moved = min(float(pred.prob_up), shift)
            pred.prob_up = float(pred.prob_up - moved)
            pred.prob_down = float(pred.prob_down + moved)

        # Keep probabilities normalized.
        pred.prob_down = float(np.clip(pred.prob_down, 0.0, 1.0))
        pred.prob_neutral = float(np.clip(pred.prob_neutral, 0.0, 1.0))
        pred.prob_up = float(np.clip(pred.prob_up, 0.0, 1.0))
        p_sum = float(pred.prob_down + pred.prob_neutral + pred.prob_up)
        if p_sum <= 0:
            pred.prob_down, pred.prob_neutral, pred.prob_up = 0.33, 0.34, 0.33
        else:
            pred.prob_down /= p_sum
            pred.prob_neutral /= p_sum
            pred.prob_up /= p_sum

        edge = float(pred.prob_up - pred.prob_down)
        aligned = (edge == 0.0) or ((edge > 0) == (news_bias > 0))
        conf_delta = min(0.10, abs(news_bias) * (0.35 if aligned else 0.18))
        if aligned:
            pred.confidence = float(np.clip(pred.confidence + conf_delta, 0.0, 1.0))
        else:
            pred.confidence = float(np.clip(pred.confidence - (conf_delta * 0.6), 0.0, 1.0))

        # News can upgrade HOLD when the post-blend edge is meaningful.
        if pred.signal == Signal.HOLD and pred.confidence >= 0.56:
            if edge >= 0.08:
                pred.signal = Signal.BUY
            elif edge <= -0.08:
                pred.signal = Signal.SELL

        # News can also dampen contradictory directional signals.
        if pred.signal in (Signal.BUY, Signal.STRONG_BUY) and edge < 0:
            pred.signal = Signal.HOLD
        elif pred.signal in (Signal.SELL, Signal.STRONG_SELL) and edge > 0:
            pred.signal = Signal.HOLD

        if count > 0:
            direction = "bullish" if news_bias > 0 else "bearish"
            msg = (
                f"News sentiment tilt: {direction} "
                f"({sentiment:+.2f}, conf {conf:.2f}, n={count})"
            )
            if msg not in pred.reasons:
                pred.reasons.append(msg)

        return float(news_bias)

    def _normalize_interval_token(self, interval: str | None) -> str:
        """Normalize common provider/UI aliases."""
        iv = str(interval or self.interval).strip().lower()
        aliases = {
            "1h": "60m",
            "60min": "60m",
            "60mins": "60m",
            "daily": "1d",
            "day": "1d",
            "1day": "1d",
            "1440m": "1d",
        }
        return aliases.get(iv, iv)

    def _is_intraday_interval(self, interval: str) -> bool:
        iv = self._normalize_interval_token(interval)
        return iv not in {"1d", "1wk", "1mo"}

    def _bar_safety_caps(self, interval: str) -> tuple[float, float]:
        """Return (max_jump_pct, max_range_pct) for OHLC history cleaning."""
        iv = self._normalize_interval_token(interval)
        if iv == "1m":
            return 0.08, 0.03
        if iv == "5m":
            return 0.10, 0.05
        if iv in ("15m", "30m"):
            return 0.14, 0.08
        if iv in ("60m",):
            return 0.18, 0.12
        if iv in ("1d", "1wk", "1mo"):
            return 0.24, 0.22
        return 0.20, 0.15

    def _sanitize_ohlc_row(
        self,
        o: float,
        h: float,
        low: float,
        c: float,
        *,
        interval: str,
        ref_close: float | None = None,
    ) -> tuple[float, float, float, float] | None:
        """Clean one OHLC row and reject malformed spikes."""
        try:
            o = float(o or 0.0)
            h = float(h or 0.0)
            low = float(low or 0.0)
            c = float(c or 0.0)
        except (TypeError, ValueError):
            return None
        if not all(np.isfinite(v) for v in (o, h, low, c)):
            return None
        if c <= 0:
            return None

        if o <= 0:
            o = c
        if h <= 0:
            h = max(o, c)
        if low <= 0:
            low = min(o, c)
        if h < low:
            h, low = low, h

        jump_cap, range_cap = self._bar_safety_caps(interval)
        ref = float(ref_close or 0.0)
        if not np.isfinite(ref) or ref <= 0:
            ref = 0.0

        if ref > 0:
            jump = abs(c / ref - 1.0)
            hard_jump_cap = max(
                jump_cap * 1.7,
                0.12 if self._is_intraday_interval(interval) else jump_cap,
            )
            if jump > hard_jump_cap:
                return None

        anchor = ref if ref > 0 else c
        if anchor <= 0:
            anchor = c
        if ref > 0:
            effective_range_cap = float(range_cap)
        else:
            bootstrap_cap = (
                0.30
                if not self._is_intraday_interval(interval)
                else float(min(0.24, max(jump_cap, range_cap * 2.0)))
            )
            effective_range_cap = float(max(range_cap, bootstrap_cap))

        max_body = float(anchor) * float(
            max(jump_cap * 1.25, effective_range_cap * 0.9)
        )
        if max_body > 0 and abs(o - c) > max_body:
            if ref > 0 and abs(c / ref - 1.0) <= max(jump_cap * 1.2, 0.10):
                o = ref
            else:
                o = c

        top = max(o, c)
        bot = min(o, c)
        if h < top:
            h = top
        if low > bot:
            low = bot
        if h < low:
            h, low = low, h

        max_range = float(anchor) * float(effective_range_cap)
        curr_range = max(0.0, h - low)
        if max_range > 0 and curr_range > max_range:
            body = max(0.0, top - bot)
            if body > max_range:
                o = c
                top = c
                bot = c
                body = 0.0
            wick_allow = max(0.0, max_range - body)
            h = min(h, top + (wick_allow * 0.5))
            low = max(low, bot - (wick_allow * 0.5))
            if h < low:
                h, low = low, h

        if anchor > 0 and (h - low) > (float(anchor) * float(effective_range_cap) * 1.05):
            return None

        o = min(max(o, low), h)
        c = min(max(c, low), h)
        return o, h, low, c

    def _intraday_session_mask(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """Best-effort CN intraday trading-session filter."""
        if idx.size <= 0:
            return np.zeros(0, dtype=bool)

        ts = idx
        try:
            if ts.tz is None:
                ts = ts.tz_localize("Asia/Shanghai", ambiguous="NaT", nonexistent="shift_forward")
            else:
                ts = ts.tz_convert("Asia/Shanghai")
        except Exception as e:
            log.debug("Session mask timezone conversion fallback triggered: %s", e)
            try:
                ts = idx.tz_localize(None)
            except Exception as inner_e:
                log.debug("Session mask timezone fallback failed: %s", inner_e)
                ts = idx

        weekday = np.asarray(ts.weekday, dtype=int)
        mins = (np.asarray(ts.hour, dtype=int) * 60) + np.asarray(ts.minute, dtype=int)

        t_cfg = CONFIG.trading
        am_open = (int(t_cfg.market_open_am.hour) * 60) + int(t_cfg.market_open_am.minute)
        am_close = (int(t_cfg.market_close_am.hour) * 60) + int(t_cfg.market_close_am.minute)
        pm_open = (int(t_cfg.market_open_pm.hour) * 60) + int(t_cfg.market_open_pm.minute)
        pm_close = (int(t_cfg.market_close_pm.hour) * 60) + int(t_cfg.market_close_pm.minute)

        is_weekday = weekday < 5
        in_am = (mins >= am_open) & (mins <= am_close)
        in_pm = (mins >= pm_open) & (mins <= pm_close)
        return np.asarray(is_weekday & (in_am | in_pm), dtype=bool)

    def _sanitize_history_df(
        self,
        df: pd.DataFrame | None,
        interval: str,
    ) -> pd.DataFrame:
        """
        Normalize history rows before features/inference.
        Fixes malformed open=0 intraday rows and drops out-of-session noise.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        iv = self._normalize_interval_token(interval)
        work = df.copy()
        has_dt_index = isinstance(work.index, pd.DatetimeIndex)

        if not has_dt_index:
            dt = None
            if "datetime" in work.columns:
                dt = pd.to_datetime(work["datetime"], errors="coerce")
            elif "timestamp" in work.columns:
                dt = pd.to_datetime(work["timestamp"], errors="coerce")
            if dt is not None:
                valid_dt = dt.notna()
                valid_count = int(valid_dt.sum())
                valid_ratio = (
                    float(valid_count) / float(len(work))
                    if len(work) > 0
                    else 0.0
                )
                if valid_count > 0 and valid_ratio >= 0.80:
                    work = work.assign(_dt=dt).dropna(subset=["_dt"]).set_index("_dt")
                    has_dt_index = isinstance(work.index, pd.DatetimeIndex)

        if has_dt_index:
            work = work[~work.index.duplicated(keep="last")].sort_index()

        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col not in work.columns:
                work[col] = 0.0
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

        work = work[np.isfinite(work["close"]) & (work["close"] > 0)].copy()
        if work.empty:
            return pd.DataFrame()

        if self._is_intraday_interval(iv) and has_dt_index:
            mask = self._intraday_session_mask(work.index)
            if mask.size == len(work):
                work = work.loc[mask].copy()
            if work.empty:
                return pd.DataFrame()

        cleaned_rows: list[dict] = []
        cleaned_idx: list = []
        prev_close: float | None = None
        prev_date = None

        for idx, row in work.iterrows():
            try:
                c = float(row.get("close", 0) or 0)
                o = float(row.get("open", c) or c)
                h = float(row.get("high", c) or c)
                low = float(row.get("low", c) or c)
            except (TypeError, ValueError):
                continue

            idx_date = idx.date() if hasattr(idx, "date") else None
            ref_close = prev_close
            if (
                self._is_intraday_interval(iv)
                and has_dt_index
                and prev_date is not None
                and idx_date is not None
                and idx_date != prev_date
            ):
                # First bar of a new day can gap against prior close.
                ref_close = None

            sanitized = self._sanitize_ohlc_row(
                o,
                h,
                low,
                c,
                interval=iv,
                ref_close=ref_close,
            )
            if sanitized is None:
                continue
            o, h, low, c = sanitized

            row_out = row.to_dict()
            row_out["open"] = float(o)
            row_out["high"] = float(h)
            row_out["low"] = float(low)
            row_out["close"] = float(c)
            cleaned_rows.append(row_out)
            cleaned_idx.append(idx)
            prev_close = float(c)
            prev_date = idx_date

        if not cleaned_rows:
            return pd.DataFrame()

        out = pd.DataFrame(cleaned_rows)
        if has_dt_index:
            out.index = pd.DatetimeIndex(cleaned_idx)
            out = out[~out.index.duplicated(keep="last")].sort_index()
        else:
            out.index = pd.Index(cleaned_idx)
        return out

    def invalidate_cache(self, code: str = None):
        """Invalidate cache for a specific code or all codes."""
        with self._cache_lock:
            if code:
                key = str(code).strip()
                code6 = self._clean_code(key)
                for k in list(self._pred_cache.keys()):
                    if k == key or (code6 and str(k).startswith(f"{code6}:")):
                        self._pred_cache.pop(k, None)
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
        history_allow_online: bool = True,
    ) -> pd.DataFrame | None:
        """Fetch stock data with minimum data requirement."""
        try:
            from data.fetcher import BARS_PER_DAY

            interval = self._normalize_interval_token(interval)
            bpd = float(BARS_PER_DAY.get(interval, 1))
            min_days = (
                7
                if interval in {"1m", "2m", "3m", "5m", "15m", "30m", "60m", "1h"}
                else 14
            )
            min_bars = int(max(min_days * bpd, min_days))
            bars = int(max(int(lookback), int(min_bars)))

            try:
                df = self.fetcher.get_history(
                    code,
                    interval=interval,
                    bars=bars,
                    use_cache=True,
                    update_db=True,
                    allow_online=bool(history_allow_online),
                )
            except TypeError:
                df = self.fetcher.get_history(
                    code,
                    interval=interval,
                    bars=bars,
                    use_cache=True,
                    update_db=True,
                )
            if df is None or df.empty:
                return None

            df = self._sanitize_history_df(df, interval)
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
                except Exception as e:
                    log.debug("Realtime quote merge failed for %s: %s", code, e)

            df = self._sanitize_history_df(df, interval)
            if df is None or df.empty:
                return None

            return df

        except Exception as e:
            log.warning(f"Failed to fetch data for {code}: {e}")
            return None

    def _default_lookback_bars(self, interval: str | None) -> int:
        """
        Default history depth for inference.
        Intraday intervals use a true 7-day window (e.g. 1m => 1680 bars).
        """
        iv = self._normalize_interval_token(interval)
        try:
            from data.fetcher import BARS_PER_DAY, INTERVAL_MAX_DAYS
            bpd = float(BARS_PER_DAY.get(iv, 1.0))
            max_days = int(INTERVAL_MAX_DAYS.get(iv, 7))
        except Exception as e:
            log.debug("Falling back to default lookback constants for interval=%s: %s", iv, e)
            bpd = float({
                "1m": 240.0,
                "2m": 120.0,
                "3m": 80.0,
                "5m": 48.0,
                "15m": 16.0,
                "30m": 8.0,
                "60m": 4.0,
                "1h": 4.0,
                "1d": 1.0,
            }.get(iv, 1.0))
            max_days = 7 if iv in {"1m", "2m", "3m", "5m", "15m", "30m", "60m", "1h"} else 365

        if iv in {"1d", "1wk", "1mo"}:
            return max(60, int(round(min(365, max_days) * max(1.0, bpd))))

        days = max(1, min(7, max_days))
        bars = int(round(float(days) * max(1.0, bpd)))
        return max(120, bars)

    # =========================================================================
    # =========================================================================

    def _sequence_signature(self, X: np.ndarray) -> float:
        """Stable numeric signature for the latest feature sequence."""
        try:
            arr = np.asarray(X, dtype=float).reshape(-1)
            if arr.size <= 0:
                return 0.0
            tail = arr[-min(64, arr.size):]
            weights = np.arange(1, tail.size + 1, dtype=float)
            return float(np.sum(np.round(tail, 5) * weights))
        except (TypeError, ValueError):
            return 0.0

    def _forecast_seed(
        self,
        current_price: float,
        sequence_signature: float,
        direction_hint: float,
        horizon: int,
        seed_context: str = "",
        recent_prices: list[float] | None = None,
    ) -> int:
        """
        Deterministic seed for forecast noise.
        Includes symbol/interval context to avoid repeated template curves
        when feature signatures are similar across symbols.
        """
        ctx_hash = 0
        for ch in str(seed_context or ""):
            ctx_hash = ((ctx_hash * 131) + ord(ch)) & 0x7FFFFFFF

        recent_hash = 0
        if recent_prices is not None:
            try:
                rp = np.array(
                    [float(p) for p in recent_prices if float(p) > 0],
                    dtype=float,
                )
                if rp.size > 0:
                    tail = rp[-min(12, rp.size):]
                    weights = np.arange(1, tail.size + 1, dtype=float)
                    recent_hash = int(
                        abs(np.sum(np.round(tail, 4) * weights) * 10.0)
                    )
            except (TypeError, ValueError):
                recent_hash = 0

        seed = (
            int(abs(float(current_price)) * 100)
            ^ int(abs(float(sequence_signature)) * 1000)
            ^ int((float(direction_hint) + 1.0) * 100000)
            ^ int(max(1, int(horizon)) * 131)
            ^ int(ctx_hash)
            ^ int(recent_hash)
        ) % (2**31 - 1)

        return 1 if seed == 0 else int(seed)

    def _generate_forecast(
        self,
        X: np.ndarray,
        current_price: float,
        horizon: int,
        atr_pct: float = 0.02,
        sequence_signature: float = 0.0,
        seed_context: str = "",
        recent_prices: list[float] | None = None,
        news_bias: float = 0.0,
    ) -> list[float]:
        """Generate price forecast using forecaster or ensemble."""
        if current_price <= 0:
            return []
        horizon = max(1, int(horizon))
        atr_pct = float(np.nan_to_num(atr_pct, nan=0.02, posinf=0.02, neginf=0.02))
        if atr_pct <= 0:
            atr_pct = 0.02
        news_bias = float(
            np.clip(
                np.nan_to_num(news_bias, nan=0.0, posinf=0.0, neginf=0.0),
                -0.50,
                0.50,
            )
        )

        direction_hint = 0.0
        hint_confidence = 0.5
        hint_entropy = 0.5
        if self.ensemble is not None:
            try:
                hint_pred = self.ensemble.predict(X)
                probs_hint = getattr(hint_pred, "probabilities", None)
                if probs_hint is not None and len(probs_hint) >= 3:
                    direction_hint = (
                        float(probs_hint[2]) - float(probs_hint[0])
                    )
                hint_confidence = float(
                    np.clip(getattr(hint_pred, "confidence", 0.5), 0.0, 1.0)
                )
                hint_entropy = float(
                    np.clip(getattr(hint_pred, "entropy", 0.5), 0.0, 1.0)
                )
            except Exception as e:
                log.debug("Ensemble direction hint unavailable: %s", e)
                direction_hint = 0.0
                hint_confidence = 0.5
                hint_entropy = 0.5
        direction_hint = float(
            np.nan_to_num(direction_hint, nan=0.0, posinf=0.0, neginf=0.0)
        )
        if abs(news_bias) > 1e-8:
            direction_hint = float(
                np.clip(direction_hint + (news_bias * 0.65), -1.0, 1.0)
            )
        hint_confidence = float(
            np.clip(
                np.nan_to_num(hint_confidence, nan=0.5, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            )
        )
        hint_entropy = float(
            np.clip(
                np.nan_to_num(hint_entropy, nan=0.5, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            )
        )

        quality_scale = float(
            np.clip(
                (0.35 + (0.65 * hint_confidence)) * (1.0 - (0.55 * hint_entropy)),
                0.35,
                1.0,
            )
        )

        if self.forecaster is not None:
            try:
                import torch

                self.forecaster.eval()
                with torch.inference_mode():
                    X_tensor = torch.FloatTensor(X)
                    returns, _ = self.forecaster(X_tensor)
                    returns_arr = np.asarray(
                        returns[0].detach().cpu().numpy(), dtype=float
                    ).reshape(-1)

                if returns_arr.size <= 0:
                    raise ValueError("Forecaster produced empty output")

                returns_arr = np.nan_to_num(
                    returns_arr, nan=0.0, posinf=0.0, neginf=0.0
                )

                neutral_mode = abs(direction_hint) < 0.10
                neutral_bias = 0.0
                if neutral_mode and returns_arr.size > 0:
                    neutral_bias = float(
                        np.mean(returns_arr[:min(8, returns_arr.size)])
                    )

                prices_arr = np.array(
                    [float(p) for p in (recent_prices or []) if float(p) > 0],
                    dtype=float,
                )
                recent_mu_pct = 0.0
                if prices_arr.size >= 6:
                    rets = np.diff(np.log(prices_arr[-min(90, prices_arr.size):]))
                    if rets.size > 0:
                        recent_mu_pct = float(
                            np.clip(np.mean(rets) * 100.0, -0.25, 0.25)
                        )

                step_cap_pct = float(
                    np.clip(max(float(atr_pct), 0.0035) * 140.0, 0.18, 3.0)
                )
                if neutral_mode:
                    step_cap_pct = min(
                        step_cap_pct,
                        float(max(float(atr_pct) * 70.0, 0.35)),
                    )
                step_cap_pct = max(
                    step_cap_pct * quality_scale,
                    0.12 if neutral_mode else 0.20,
                )
                news_drift_pct = float(
                    news_bias
                    * step_cap_pct
                    * (0.14 if neutral_mode else 0.28)
                )

                # Deterministic symbol-specific residual to avoid template-like tails.
                seed = self._forecast_seed(
                    current_price=current_price,
                    sequence_signature=sequence_signature,
                    direction_hint=direction_hint,
                    horizon=horizon,
                    seed_context=seed_context,
                    recent_prices=recent_prices,
                )
                rng = np.random.RandomState(seed)

                tail_window = returns_arr[-min(10, returns_arr.size):]
                tail_mu = float(np.mean(tail_window)) if tail_window.size > 0 else 0.0
                tail_sigma = float(np.std(tail_window)) if tail_window.size > 0 else 0.0
                tail_sigma_floor = step_cap_pct * (0.05 if neutral_mode else 0.10)
                tail_sigma = float(
                    np.clip(
                        tail_sigma,
                        max(0.01, tail_sigma_floor),
                        max(0.06, step_cap_pct * 0.45),
                    )
                )

                prev_eps = 0.0
                prev_ret = 0.0
                prev_model_ret = 0.0

                prices = [current_price]
                for i in range(horizon):
                    if i < returns_arr.size:
                        raw_ret = float(returns_arr[i])
                    else:
                        extra_i = i - returns_arr.size + 1
                        decay = float(
                            np.exp(
                                -extra_i / max(4.0, float(horizon) * 0.22)
                            )
                        )
                        tail_target = (
                            (tail_mu * (0.55 + (0.45 * decay)))
                            + (recent_mu_pct * (0.45 * (1.0 - decay)))
                        )
                        tail_noise = float(
                            rng.normal(
                                0.0,
                                tail_sigma * (0.35 + (0.65 * decay)),
                            )
                        )
                        raw_ret = (
                            (0.74 * prev_model_ret)
                            + (0.26 * tail_target)
                            + tail_noise
                        )
                    prev_model_ret = raw_ret
                    r_val = raw_ret

                    if neutral_mode:
                        r_val = ((r_val - neutral_bias) * 0.45) + (recent_mu_pct * 0.35)
                        mean_pull_pct = (-(prices[-1] / current_price - 1.0)) * 22.0
                        r_val += mean_pull_pct
                    else:
                        r_val = (r_val * 0.84) + (recent_mu_pct * 0.16)

                    if abs(news_drift_pct) > 1e-9:
                        news_decay = max(
                            0.35,
                            1.0 - (float(i) / max(3.0, float(horizon) * 1.25)),
                        )
                        r_val += float(news_drift_pct * news_decay)

                    noise_scale = 0.55 + (0.45 * quality_scale)
                    eps_scale = step_cap_pct * (0.06 if neutral_mode else 0.10) * noise_scale
                    eps = (0.62 * prev_eps) + float(rng.normal(0.0, eps_scale))
                    prev_eps = eps
                    r_val += eps
                    r_val = (0.78 * r_val) + (0.22 * prev_ret)
                    r_val = float(np.clip(r_val, -step_cap_pct, step_cap_pct))
                    prev_ret = r_val
                    next_price = prices[-1] * (1 + r_val / 100)
                    next_price = max(
                        next_price, current_price * 0.5
                    )
                    next_price = min(
                        next_price, current_price * 2.0
                    )
                    prices.append(float(next_price))

                out = prices[1:]
                if neutral_mode:
                    neutral_cap = max(float(atr_pct) * 0.55, 0.0045)
                    lo = current_price * (1.0 - neutral_cap)
                    hi = current_price * (1.0 + neutral_cap)
                    out = [float(np.clip(p, lo, hi)) for p in out]
                    if len(out) >= 2:
                        for i in range(1, len(out)):
                            out[i] = float((0.68 * out[i]) + (0.32 * out[i - 1]))
                        out = [float(np.clip(p, lo, hi)) for p in out]

                return out

            except Exception as e:
                log.debug(f"Forecaster failed: {e}")

        # Fallback: ensemble-guided path shaped by recent symbol behavior.
        if self.ensemble:
            try:
                pred = self.ensemble.predict(X)

                probs = np.asarray(
                    getattr(pred, "probabilities", [0.33, 0.34, 0.33]),
                    dtype=float,
                ).reshape(-1)
                probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                if probs.size < 3:
                    probs = np.pad(probs, (0, 3 - probs.size), constant_values=0.0)
                prob_sum = float(np.sum(probs[:3]))
                if prob_sum <= 0:
                    probs = np.array([0.33, 0.34, 0.33], dtype=float)
                else:
                    probs = probs[:3] / prob_sum
                direction = (
                    (float(probs[2]) if len(probs) > 2 else 0.33)
                    - (float(probs[0]) if len(probs) > 0 else 0.33)
                )
                if abs(news_bias) > 1e-8:
                    direction = float(
                        np.clip(direction + (news_bias * 0.70), -1.0, 1.0)
                    )
                confidence = float(np.clip(getattr(pred, "confidence", 0.5), 0.0, 1.0))
                entropy = float(np.clip(getattr(pred, "entropy", 0.5), 0.0, 1.0))
                quality_scale = float(
                    np.clip(
                        (0.35 + (0.65 * confidence)) * (1.0 - (0.55 * entropy)),
                        0.35,
                        1.0,
                    )
                )
                neutral_mode = abs(direction) < 0.10

                volatility = max(float(atr_pct), 0.005) * quality_scale
                if neutral_mode:
                    volatility = min(volatility, 0.012)

                prices_arr = np.array(
                    [float(p) for p in (recent_prices or []) if float(p) > 0],
                    dtype=float,
                )
                if prices_arr.size >= 4:
                    rets = np.diff(np.log(prices_arr[-min(80, prices_arr.size):]))
                    ret_mu = float(np.clip(np.mean(rets), -0.02, 0.02))
                    ret_sigma = float(
                        np.clip(
                            np.std(rets),
                            0.0006,
                            max(volatility * 0.8, 0.03),
                        )
                    )
                else:
                    ret_mu = 0.0
                    ret_sigma = float(max(volatility * 0.45, 0.001))
                if neutral_mode:
                    ret_mu *= 0.35
                    ret_sigma = max(ret_sigma * 0.55, 0.0004)
                else:
                    ret_sigma *= (0.70 + (0.30 * quality_scale))

                prices = []
                price = current_price

                # Deterministic seed using sequence signature to keep each symbol distinct.
                seed = self._forecast_seed(
                    current_price=current_price,
                    sequence_signature=sequence_signature,
                    direction_hint=direction,
                    horizon=horizon,
                    seed_context=seed_context,
                    recent_prices=recent_prices,
                )
                rng = np.random.RandomState(seed)

                for i in range(horizon):
                    decay = 1.0 - (i / (horizon * 1.8))
                    drift_scale = 0.10 if neutral_mode else 0.20
                    mu_scale = 0.20 if neutral_mode else 0.35
                    drift = (direction * volatility * drift_scale) + (ret_mu * mu_scale)
                    if abs(news_bias) > 1e-8:
                        drift += (
                            news_bias
                            * volatility
                            * (0.10 if neutral_mode else 0.22)
                            * decay
                        )
                    noise = float(
                        rng.normal(
                            0.0,
                            ret_sigma * ((0.35 if neutral_mode else 0.55) + (0.45 * decay)),
                        )
                    )
                    mean_revert = (-(0.18 if neutral_mode else 0.10)) * (
                        (price / current_price) - 1.0
                    )
                    change = drift + noise + mean_revert
                    if neutral_mode:
                        max_step = max(volatility * 1.3, 0.007)
                    else:
                        max_step = max(volatility * 2.2, 0.02)
                    change = float(np.clip(change, -max_step, max_step))
                    price = price * (1 + change)

                    price = max(price, current_price * 0.5)
                    price = min(price, current_price * 2.0)
                    if neutral_mode:
                        neutral_cap = max(volatility, 0.008)
                        price = float(
                            np.clip(
                                price,
                                current_price * (1.0 - neutral_cap),
                                current_price * (1.0 + neutral_cap),
                            )
                        )

                    prices.append(float(price))

                return prices

            except Exception as e:
                log.debug(f"Ensemble forecast failed: {e}")

        # Last resort: deterministic micro-trajectory from recent volatility
        # instead of a flat line when model artifacts are unavailable.
        prices_arr = np.array(
            [float(p) for p in (recent_prices or []) if float(p) > 0],
            dtype=float,
        )
        if prices_arr.size >= 4:
            rets = np.diff(np.log(prices_arr[-min(120, prices_arr.size):]))
            ret_mu = float(np.clip(np.mean(rets), -0.01, 0.01))
            ret_sigma = float(
                np.clip(
                    np.std(rets),
                    max(float(atr_pct) * 0.02, 0.0003),
                    max(float(atr_pct) * 0.18, 0.0060),
                )
            )
        else:
            ret_mu = 0.0
            ret_sigma = float(max(float(atr_pct) * 0.05, 0.0006))

        max_step = float(max(float(atr_pct) * 0.22, 0.0012))
        drift = float(np.clip(ret_mu + (news_bias * 0.0012), -max_step * 0.45, max_step * 0.45))
        seed = self._forecast_seed(
            current_price=current_price,
            sequence_signature=sequence_signature,
            direction_hint=direction_hint,
            horizon=horizon,
            seed_context=seed_context,
            recent_prices=recent_prices,
        )
        rng = np.random.RandomState(seed)

        out: list[float] = []
        price = float(current_price)
        prev_eps = 0.0
        for i in range(horizon):
            decay = max(0.25, 1.0 - (float(i) / float(max(horizon, 1))) * 0.65)
            eps = (0.55 * prev_eps) + float(
                rng.normal(0.0, ret_sigma * (0.45 + (0.55 * decay)))
            )
            prev_eps = eps
            mean_revert = -0.08 * ((price / max(float(current_price), 1e-8)) - 1.0)
            change = float(np.clip(drift + eps + mean_revert, -max_step, max_step))
            if abs(change) < 1e-7:
                change = float(((-1.0) ** i) * max_step * 0.03)
            price = float(np.clip(price * (1.0 + change), current_price * 0.5, current_price * 2.0))
            out.append(price)
        return out

    # =========================================================================
    # =========================================================================

    def _determine_signal(
        self, ensemble_pred, pred: Prediction
    ) -> Signal:
        """Determine trading signal from prediction."""
        confidence = float(ensemble_pred.confidence)
        predicted_class = int(ensemble_pred.predicted_class)
        edge = float(np.clip(pred.prob_up - pred.prob_down, -1.0, 1.0))
        is_sideways = str(pred.trend).upper() == "SIDEWAYS"
        edge_floor = 0.06 if is_sideways else 0.04
        strong_edge_floor = max(0.12, edge_floor * 2.0)

        if predicted_class == 2:  # UP
            if edge < edge_floor:
                return Signal.HOLD
            if confidence >= CONFIG.STRONG_BUY_THRESHOLD:
                if edge >= strong_edge_floor:
                    return Signal.STRONG_BUY
                return Signal.BUY
            elif confidence >= CONFIG.BUY_THRESHOLD:
                return Signal.BUY
        elif predicted_class == 0:  # DOWN
            if edge > -edge_floor:
                return Signal.HOLD
            if confidence >= CONFIG.STRONG_SELL_THRESHOLD:
                if edge <= -strong_edge_floor:
                    return Signal.STRONG_SELL
                return Signal.SELL
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

    def _resolve_trade_distances(self, pred: Prediction) -> tuple[float, float]:
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
        except Exception as e:
            log.debug("ATR extraction failed; using default: %s", e)
        return 0.02  # default 2%

    # =========================================================================
    # =========================================================================

    def _generate_reasons(self, pred: Prediction):
        """Generate analysis reasons and warnings."""
        existing_reasons = list(pred.reasons or [])
        existing_warnings = list(pred.warnings or [])
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

        uncertainty = float(np.clip(getattr(pred, "uncertainty_score", 0.5), 0.0, 1.0))
        tail_risk = float(np.clip(getattr(pred, "tail_risk_score", 0.5), 0.0, 1.0))
        if uncertainty >= 0.70:
            warnings.append(f"Wide uncertainty regime (score: {uncertainty:.2f})")
        else:
            reasons.append(f"Uncertainty score: {uncertainty:.2f}")

        if tail_risk >= 0.60:
            warnings.append(f"Elevated tail-event risk ({tail_risk:.2f})")
        else:
            reasons.append(f"Tail-event risk: {tail_risk:.2f}")

        low_band = list(getattr(pred, "predicted_prices_low", []) or [])
        high_band = list(getattr(pred, "predicted_prices_high", []) or [])
        if low_band and high_band and len(low_band) == len(high_band):
            try:
                lo_last = float(low_band[-1])
                hi_last = float(high_band[-1])
                ref = max(float(pred.current_price), 1e-8)
                spread_pct = float((hi_last - lo_last) / ref * 100.0)
                if spread_pct >= 6.0:
                    warnings.append(
                        f"Forecast interval is wide ({spread_pct:.1f}% at horizon)"
                    )
                else:
                    reasons.append(
                        f"Forecast interval width: {spread_pct:.1f}% at horizon"
                    )
            except Exception as e:
                log.debug("Failed computing forecast interval reason for %s: %s", pred.stock_code, e)

        pred.reasons = existing_reasons + [
            msg for msg in reasons if msg not in existing_reasons
        ]
        pred.warnings = existing_warnings + [
            msg for msg in warnings if msg not in existing_warnings
        ]

    # =========================================================================
    # =========================================================================

    def _get_stock_name(self, code: str, df: pd.DataFrame) -> str:
        """Get stock name from fetcher."""
        del df
        try:
            quote = self.fetcher.get_realtime(code)
            if quote and quote.name:
                return str(quote.name)
        except Exception as e:
            log.debug("Stock name lookup failed for %s: %s", code, e)
        return ""

    def _clean_code(self, code: str) -> str:
        """
        Clean and normalize stock code.
        Delegates to DataFetcher when available.
        """
        if self.fetcher is not None:
            try:
                return self.fetcher.clean_code(code)
            except Exception as e:
                log.debug("Fetcher clean_code failed for %r: %s", code, e)

        if not code:
            return ""
        code = str(code).strip()
        code = "".join(c for c in code if c.isdigit())
        return code.zfill(6) if code else ""
