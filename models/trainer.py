# models/trainer.py
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config.settings import CONFIG
from data.features import FeatureEngine
from data.fetcher import get_fetcher
from data.processor import DataProcessor
from models.ensemble import EnsembleModel
from models.trainer_data_ops import _assess_raw_data_quality as _assess_raw_data_quality_impl
from models.trainer_eval_ops import _build_explainability_samples as _build_explainability_samples_impl
from models.trainer_eval_ops import _build_quality_gate as _build_quality_gate_impl
from models.trainer_eval_ops import _build_trading_stress_tests as _build_trading_stress_tests_impl
from models.trainer_data_ops import _create_sequences_from_splits as _create_sequences_from_splits_impl
from models.trainer_data_ops import _effective_confidence_floor as _effective_confidence_floor_impl
from models.trainer_eval_ops import _evaluate as _evaluate_impl
from models.trainer_data_ops import _fallback_temporal_validation_split as _fallback_temporal_validation_split_impl
from models.trainer_data_ops import _fetch_raw_data as _fetch_raw_data_impl
from models.trainer_data_ops import prepare_data as _prepare_data_impl
from models.trainer_data_ops import _rebalance_train_samples as _rebalance_train_samples_impl
from models.trainer_eval_ops import _run_drift_guard as _run_drift_guard_impl
from models.trainer_eval_ops import _simulate_trading as _simulate_trading_impl
from models.trainer_data_ops import _split_single_stock as _split_single_stock_impl
from models.trainer_eval_ops import _trade_masks as _trade_masks_impl
from models.trainer_eval_ops import _trade_quality_thresholds as _trade_quality_thresholds_impl
from models.trainer_eval_ops import _walk_forward_validate as _walk_forward_validate_impl
from utils.cancellation import CancelledException
from utils.logger import get_logger

log = get_logger(__name__)

# Import atomic_io at module level (not inside loops)
try:
    from utils.atomic_io import (
        atomic_torch_save,
        atomic_write_json,
        read_json,
    )
except ImportError:
    atomic_torch_save = None
    atomic_write_json = None
    read_json = None

_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_SEED)

# Epsilon for division-by-zero protection
_EPS = 1e-8

# Stop check interval for batch loops (check every N batches)
_STOP_CHECK_INTERVAL = 10
_TRAINING_INTERVAL_LOCK = "1m"
_MIN_1M_LOOKBACK_BARS = 10080
_DEFAULT_ENSEMBLE_MODELS = ["lstm", "gru", "tcn", "transformer", "hybrid"]
_WALK_FORWARD_FOLDS = 3
_MIN_WALK_FORWARD_SAMPLES = 180
_OVERFIT_VAL_ACC_DROP_WARN = 0.06
_OVERFIT_LOSS_GAP_WARN = 0.35
_DRIFT_WARN_SCORE_DROP = 0.08
_DRIFT_BLOCK_SCORE_DROP = 0.16
_DRIFT_WARN_ACC_DROP = 0.05
_DRIFT_BLOCK_ACC_DROP = 0.10
_MIN_BASELINE_RISK_SCORE = 0.52
_MIN_BASELINE_PROFIT_FACTOR = 1.05
_MAX_BASELINE_DRAWDOWN = 0.25
_MIN_BASELINE_TRADES = 5
_DATA_QUALITY_MAX_NAN_RATIO = 0.04
_DATA_QUALITY_MAX_NONPOS_PRICE_RATIO = 0.0
_DATA_QUALITY_MAX_BROKEN_OHLC_RATIO = 0.001
_DATA_QUALITY_MIN_VALID_SYMBOL_RATIO = 0.55
_INCREMENTAL_REGIME_BLOCK_LEVELS = {"high"}
_STRESS_COST_MULTIPLIERS = (1.0, 1.5, 2.0)
_TAIL_STRESS_QUANTILE = 0.90
_MIN_TAIL_STRESS_SAMPLES = 24
_TAIL_EVENT_SHOCK_MIN_PCT = 1.0
_TAIL_EVENT_SHOCK_MAX_PCT = 6.0


def _write_artifact_checksum(path: Path) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    payload = f"{h.hexdigest()}\n"
    try:
        from utils.atomic_io import atomic_write_text

        atomic_write_text(checksum_path, payload)
    except Exception:
        checksum_path.write_text(payload, encoding="utf-8")

def _resolve_learning_rate(explicit_lr: float | None = None) -> float:
    """
    Resolve learning rate from multiple sources in priority order:
    1. Explicit parameter (highest priority)
    2. Thread-local override from auto_learner
    3. CONFIG.model.learning_rate (default)

    This centralizes LR resolution so both classifier and forecaster
    use the same value.
    """
    if explicit_lr is not None:
        return float(explicit_lr)

    try:
        from models.auto_learner import get_effective_learning_rate
        return get_effective_learning_rate()
    except ImportError:
        pass

    return CONFIG.model.learning_rate

class Trainer:
    """
    Complete training pipeline with proper data handling.

    CRITICAL: Features are computed and labels are created WITHIN each
    temporal split to prevent leakage.
    """

    def __init__(self) -> None:
        self.fetcher = get_fetcher()
        self.processor = DataProcessor()
        self.feature_engine = FeatureEngine()

        self.ensemble: EnsembleModel | None = None
        self.history: dict = {}
        self.input_size: int = 0

        self.interval: str = "1m"
        self.prediction_horizon: int = CONFIG.PREDICTION_HORIZON

        # Incremental training flag 鈥?set externally by auto_learner.
        self._skip_scaler_fit: bool = False
        self._last_data_quality_summary: dict[str, Any] = {}
        self._last_consistency_guard: dict[str, Any] = {}

    @staticmethod
    def _normalize_model_names(model_names: list[str] | None) -> list[str]:
        """Normalize requested ensemble model names and enforce 5-model default."""
        raw = list(model_names or _DEFAULT_ENSEMBLE_MODELS)
        out: list[str] = []
        seen: set[str] = set()
        for name in raw:
            key = str(name or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        if not out:
            return list(_DEFAULT_ENSEMBLE_MODELS)
        return out

    @staticmethod
    def _enforce_training_interval(interval: str | None) -> str:
        req = str(interval or _TRAINING_INTERVAL_LOCK).strip().lower()
        if req != _TRAINING_INTERVAL_LOCK:
            log.info(
                "Trainer interval locked to %s (requested=%s)",
                _TRAINING_INTERVAL_LOCK,
                req,
            )
        return _TRAINING_INTERVAL_LOCK

    @staticmethod
    def _default_lookback_bars(interval: str) -> int:
        """
        Default training lookback.
        1m training uses at least 10080 bars.
        Other intraday intervals use a strict 7-day window.
        """
        iv = str(interval or "1m").strip().lower()
        try:
            from data.fetcher import BARS_PER_DAY, INTERVAL_MAX_DAYS
            bpd = float(BARS_PER_DAY.get(iv, 1.0))
            max_days = int(INTERVAL_MAX_DAYS.get(iv, 7))
        except Exception as e:
            log.debug(
                "Using fallback lookback constants for interval=%s due to fetcher constants error: %s",
                iv,
                e,
            )
            bpd = float({
                "1m": 240.0,
                "2m": 120.0,
                "5m": 48.0,
                "15m": 16.0,
                "30m": 8.0,
                "60m": 4.0,
                "1h": 4.0,
                "1d": 1.0,
            }.get(iv, 1.0))
            max_days = (
                7
                if iv in {"1m", "2m", "5m", "15m", "30m", "60m", "1h"}
                else 500
            )

        if iv in {"1d", "1wk", "1mo"}:
            annual_days = min(365, max_days)
            return max(
                200,
                min(2400, int(round(float(annual_days) * max(1.0, bpd)))),
            )

        if iv == _TRAINING_INTERVAL_LOCK:
            return int(max(_MIN_1M_LOOKBACK_BARS, CONFIG.SEQUENCE_LENGTH + 20))

        days = max(1, min(7, max_days))
        return max(
            int(CONFIG.SEQUENCE_LENGTH) + 20,
            int(round(float(days) * max(1.0, bpd))),
        )

    # =========================================================================
    # =========================================================================

    @staticmethod
    def _should_stop(stop_flag: Any) -> bool:
        """Check if training should stop 鈥?handles multiple stop flag types."""
        if stop_flag is None:
            return False

        is_cancelled = getattr(stop_flag, "is_cancelled", None)
        if is_cancelled is not None:
            try:
                return bool(
                    is_cancelled() if callable(is_cancelled) else is_cancelled
                )
            except Exception as e:
                log.debug("Stop-flag is_cancelled evaluation failed: %s", e)
                return False

        if callable(stop_flag):
            try:
                return bool(stop_flag())
            except Exception as e:
                log.debug("Stop-flag callable evaluation failed: %s", e)
                return False

        return False

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        return float(numerator) / float(max(1.0, float(denominator)))

    @staticmethod
    def _sanitize_raw_history(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Sort and deduplicate history to preserve chronological integrity."""
        if df is None or len(df) == 0:
            return pd.DataFrame(), {
                "rows_before": 0,
                "rows_after": 0,
                "duplicates_removed": 0,
            }

        out = df.copy()
        rows_before = int(len(out))
        duplicates_removed = 0

        try:
            out = out.sort_index()
        except Exception as e:
            log.debug("Raw history sort_index failed; preserving original order: %s", e)

        try:
            dup_mask = out.index.duplicated(keep="last")
            duplicates_removed = int(np.sum(dup_mask))
            if duplicates_removed > 0:
                out = out.loc[~dup_mask]
        except Exception as e:
            log.debug("Raw history duplicate cleanup failed: %s", e)
            duplicates_removed = 0

        return out, {
            "rows_before": rows_before,
            "rows_after": int(len(out)),
            "duplicates_removed": duplicates_removed,
        }
    def _split_and_fit_scaler(
        self,
        raw_data: dict[str, pd.DataFrame],
        feature_cols: list[str],
        horizon: int,
        interval: str,
    ) -> tuple[dict[str, dict[str, pd.DataFrame]], bool]:
        """
        Split all stocks temporally, compute features per split,
        and fit scaler on training data.

        Returns:
            (split_data, has_valid_data)
        """
        all_train_features = []
        split_data: dict[str, dict[str, pd.DataFrame]] = {}

        for code, df_raw in raw_data.items():
            splits = self._split_single_stock(df_raw, horizon, feature_cols)
            if splits is None:
                log.debug(f"Insufficient split data for {code}")
                continue

            split_data[code] = splits

            train_df = splits["train"]
            if len(train_df) > 0 and "label" in train_df.columns:
                avail_cols = [c for c in feature_cols if c in train_df.columns]
                if len(avail_cols) == len(feature_cols):
                    train_features = train_df[feature_cols].values
                    valid_mask = ~train_df["label"].isna()
                    if valid_mask.sum() > 0:
                        all_train_features.append(train_features[valid_mask])

        if not all_train_features:
            return split_data, False

        expected_features = int(len(feature_cols))
        has_fitted_scaler = bool(self.processor.is_fitted)
        fitted_features = int(self.processor.n_features or 0)
        feature_count_matches = (
            has_fitted_scaler and fitted_features == expected_features
        )
        should_skip = self._skip_scaler_fit and feature_count_matches

        if should_skip:
            log.info(
                f"Skipping scaler refit (incremental mode, "
                f"existing scaler has {self.processor.n_features} features)"
            )
        else:
            if self._skip_scaler_fit and not self.processor.is_fitted:
                log.warning(
                    "_skip_scaler_fit=True but no prior scaler exists 鈥?"
                    "fitting new scaler"
                )

            if (
                self._skip_scaler_fit
                and self.processor.is_fitted
                and not feature_count_matches
            ):
                log.warning(
                    "Incremental scaler feature mismatch "
                    f"(loaded={fitted_features}, expected={expected_features}) "
                    "-> refitting scaler"
                )

            combined_train_features = np.concatenate(all_train_features, axis=0)
            self.processor.fit_scaler(
                combined_train_features,
                interval=interval,
                horizon=horizon,
            )
            log.info(
                f"Scaler fitted on {len(combined_train_features)} training samples"
            )

        return split_data, True
    @staticmethod
    def _combine_arrays(
        storage: dict[str, list],
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Combine arrays from multiple stocks."""
        if not storage["X"]:
            return None, None, None
        X = np.concatenate(storage["X"])
        y = np.concatenate(storage["y"])
        r = np.concatenate(storage["r"]) if storage["r"] else np.zeros(len(y))
        return X, y, r

    @staticmethod
    def _to_1d_float_array(values: Any) -> np.ndarray:
        """Convert array-like input to a finite 1D float array."""
        if values is None:
            return np.zeros((0,), dtype=np.float64)
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        return arr[np.isfinite(arr)]

    def _summarize_regime_shift(
        self,
        train_returns: np.ndarray | None,
        val_returns: np.ndarray | None,
        test_returns: np.ndarray | None,
    ) -> dict[str, Any]:
        """
        Detect train/eval regime drift from return distribution changes.

        Returns a score and a confidence-floor boost recommendation.
        """
        train = self._to_1d_float_array(train_returns)
        val = self._to_1d_float_array(val_returns)
        test = self._to_1d_float_array(test_returns)
        eval_parts = [x for x in (val, test) if len(x) > 0]

        if len(train) < 40 or not eval_parts:
            return {
                "detected": False,
                "level": "unknown",
                "score": 0.0,
                "confidence_boost": 0.0,
                "reason": "insufficient_samples",
            }

        eval_arr = np.concatenate(eval_parts)
        if len(eval_arr) < 40:
            return {
                "detected": False,
                "level": "unknown",
                "score": 0.0,
                "confidence_boost": 0.0,
                "reason": "insufficient_eval_samples",
            }

        train_vol = float(np.std(train))
        eval_vol = float(np.std(eval_arr))
        vol_ratio = eval_vol / (train_vol + _EPS)

        train_tail = float(np.percentile(np.abs(train), 90))
        eval_tail = float(np.percentile(np.abs(eval_arr), 90))
        tail_ratio = eval_tail / (train_tail + _EPS)

        mean_shift = abs(float(np.mean(eval_arr) - np.mean(train))) / (
            train_vol + _EPS
        )

        score = (
            0.45 * abs(float(np.log(max(vol_ratio, _EPS))))
            + 0.35 * abs(float(np.log(max(tail_ratio, _EPS))))
            + 0.20 * (min(mean_shift, 4.0) / 4.0)
        )

        if score >= 0.85:
            level = "high"
            confidence_boost = 0.10
        elif score >= 0.55:
            level = "elevated"
            confidence_boost = 0.05
        else:
            level = "normal"
            confidence_boost = 0.00

        return {
            "detected": bool(score >= 0.55),
            "level": level,
            "score": float(score),
            "confidence_boost": float(confidence_boost),
            "train_volatility": float(train_vol),
            "eval_volatility": float(eval_vol),
            "volatility_ratio": float(vol_ratio),
            "tail_ratio": float(tail_ratio),
            "mean_shift_z": float(mean_shift),
        }
    @staticmethod
    def _assess_overfitting(history: dict[str, Any] | None) -> dict[str, Any]:
        """Detect overfitting from train/val loss divergence and late val-acc drop."""
        if not history:
            return {
                "detected": False,
                "flagged_models": [],
                "per_model": {},
            }

        def _float_series(values: list[Any]) -> list[float]:
            out: list[float] = []
            for item in values or []:
                try:
                    v = float(item)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    out.append(v)
            return out

        per_model: dict[str, dict[str, float | bool]] = {}
        flagged: list[str] = []

        for name, model_hist in history.items():
            train_loss = _float_series(list(model_hist.get("train_loss", [])))
            val_loss = _float_series(list(model_hist.get("val_loss", [])))
            val_acc = _float_series(list(model_hist.get("val_acc", [])))

            loss_gap_ratio = 0.0
            if train_loss and val_loss:
                loss_gap_ratio = max(
                    0.0,
                    (val_loss[-1] - train_loss[-1]) / (abs(train_loss[-1]) + _EPS),
                )

            val_acc_drop = 0.0
            if val_acc:
                val_acc_drop = max(0.0, max(val_acc) - val_acc[-1])

            is_overfit = bool(
                loss_gap_ratio > _OVERFIT_LOSS_GAP_WARN
                or val_acc_drop > _OVERFIT_VAL_ACC_DROP_WARN
            )

            per_model[name] = {
                "loss_gap_ratio": float(loss_gap_ratio),
                "val_acc_drop": float(val_acc_drop),
                "overfit_warning": bool(is_overfit),
            }
            if is_overfit:
                flagged.append(str(name))

        return {
            "detected": bool(flagged),
            "flagged_models": flagged,
            "per_model": per_model,
        }
    @staticmethod
    def _risk_adjusted_score(metrics: dict[str, Any]) -> float:
        """
        Compute a deployment score using risk-first metrics, not accuracy alone.
        """
        accuracy = float(np.clip(metrics.get("accuracy", 0.0), 0.0, 1.0))
        trading = metrics.get("trading", {}) or {}

        sharpe = float(trading.get("sharpe_ratio", 0.0))
        profit_factor = float(trading.get("profit_factor", 0.0))
        max_drawdown = float(np.clip(trading.get("max_drawdown", 1.0), 0.0, 1.0))
        excess_return = float(trading.get("excess_return", 0.0)) / 100.0
        win_rate = float(np.clip(trading.get("win_rate", 0.0), 0.0, 1.0))
        trades = float(max(0.0, trading.get("trades", 0.0)))
        trade_coverage = float(np.clip(trading.get("trade_coverage", 0.0), 0.0, 1.0))
        avg_trade_conf = float(
            np.clip(trading.get("avg_trade_confidence", 0.0), 0.0, 1.0)
        )

        sharpe_score = 0.5 + (0.5 * float(np.tanh(sharpe / 2.0)))
        pf_score = 0.5 + (0.5 * float(np.tanh(max(-1.0, profit_factor - 1.0))))
        excess_score = 0.5 + (0.5 * float(np.tanh(excess_return * 3.0)))
        drawdown_score = 1.0 - max_drawdown
        participation = (0.5 * float(np.tanh(trades / 25.0))) + (0.5 * trade_coverage)

        score = (
            0.22 * accuracy
            + 0.22 * sharpe_score
            + 0.20 * pf_score
            + 0.14 * excess_score
            + 0.12 * drawdown_score
            + 0.06 * win_rate
            + 0.02 * participation
            + 0.02 * avg_trade_conf
        )

        if trades < 5:
            score *= 0.8

        return float(np.clip(score, 0.0, 1.0))
    @staticmethod
    def _drift_baseline_path(interval: str, horizon: int) -> Path:
        return Path(CONFIG.DATA_DIR) / (
            f"training_drift_baseline_{interval}_{horizon}.json"
        )

    @staticmethod
    def _candidate_artifact_path(live_path: Path) -> Path:
        """Build side-by-side candidate artifact path."""
        return live_path.with_name(
            f"{live_path.stem}.candidate{live_path.suffix}"
        )

    @staticmethod
    def _promote_candidate_artifact(
        candidate_path: Path,
        live_path: Path,
    ) -> bool:
        """Atomically promote a candidate artifact to live."""
        try:
            if not candidate_path.exists():
                return False
            live_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.replace(live_path)
            return True
        except Exception as e:
            log.warning(
                "Failed to promote artifact %s -> %s: %s",
                candidate_path,
                live_path,
                e,
            )
            return False

    @staticmethod
    def _meets_baseline_quality_floor(metrics: dict[str, Any]) -> bool:
        """Only stable/usable runs should become drift baselines."""
        return bool(
            float(metrics.get("risk_adjusted_score", 0.0))
            >= _MIN_BASELINE_RISK_SCORE
            and float(metrics.get("profit_factor", 0.0))
            >= _MIN_BASELINE_PROFIT_FACTOR
            and float(metrics.get("max_drawdown", 1.0))
            <= _MAX_BASELINE_DRAWDOWN
            and int(metrics.get("trades", 0)) >= _MIN_BASELINE_TRADES
        )

    @staticmethod
    def _read_json_safely(path: Path) -> dict[str, Any] | None:
        try:
            if read_json is not None:
                data = read_json(path)
            else:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception as e:
            log.debug("Failed reading JSON from %s: %s", path, e)
            return None

    @staticmethod
    def _write_json_safely(path: Path, payload: dict[str, Any]) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if atomic_write_json is not None:
                atomic_write_json(path, payload, indent=2, use_lock=True)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            return True
        except Exception as e:
            log.warning("Failed to write %s: %s", path, e)
            return False
    def train(
        self,
        stock_codes: list[str] = None,
        epochs: int = None,
        batch_size: int = None,
        model_names: list[str] = None,
        callback: Callable = None,
        stop_flag: Any = None,
        save_model: bool = True,
        incremental: bool = False,
        interval: str = "1m",
        prediction_horizon: int = None,
        lookback_bars: int = None,
        learning_rate: float = None,
    ) -> dict:
        """
        Train complete pipeline:
        1) Classification ensemble for trading signals
        2) Multi-step forecaster for AI-generated price curves
        """

        epochs = int(epochs or CONFIG.EPOCHS)
        batch_size = int(batch_size or CONFIG.BATCH_SIZE)
        interval = self._enforce_training_interval(interval)
        horizon = int(prediction_horizon or CONFIG.PREDICTION_HORIZON)
        lookback = int(
            lookback_bars
            if lookback_bars is not None
            else self._default_lookback_bars(interval)
        )
        if interval == _TRAINING_INTERVAL_LOCK:
            lookback = int(max(lookback, _MIN_1M_LOOKBACK_BARS))
        model_names = self._normalize_model_names(model_names)

        self.interval = interval
        self.prediction_horizon = horizon

        effective_lr = _resolve_learning_rate(learning_rate)
        live_ensemble_path = CONFIG.MODEL_DIR / f"ensemble_{interval}_{horizon}.pt"
        live_scaler_path = CONFIG.MODEL_DIR / f"scaler_{interval}_{horizon}.pkl"
        live_forecast_path = CONFIG.MODEL_DIR / f"forecast_{interval}_{horizon}.pt"
        candidate_ensemble_path = self._candidate_artifact_path(
            live_ensemble_path
        )
        candidate_scaler_path = self._candidate_artifact_path(live_scaler_path)
        candidate_forecast_path = self._candidate_artifact_path(
            live_forecast_path
        )

        log.info("=" * 70)
        log.info("STARTING TRAINING PIPELINE (Classifier + Forecaster)")
        log.info(
            f"Interval: {interval}, Horizon: {horizon} bars, "
            f"Lookback: {lookback}"
        )
        log.info(
            f"Incremental: {incremental}, "
            f"Skip scaler fit: {self._skip_scaler_fit}, "
            f"LR: {effective_lr:.6f}"
        )
        log.info("=" * 70)

        stocks = stock_codes or CONFIG.STOCK_POOL
        feature_cols = self.feature_engine.get_feature_columns()

        # --- Phase 1: Fetch raw data ---
        raw_data = self._fetch_raw_data(
            stocks, interval, lookback, stop_flag=stop_flag
        )

        if self._should_stop(stop_flag):
            return {"status": "cancelled"}

        if not raw_data:
            raise ValueError("No valid stock data available for training")

        log.info(f"Loaded {len(raw_data)} stocks successfully")

        # --- Phase 2: Split and fit scaler ---
        split_data, scaler_ok = self._split_and_fit_scaler(
            raw_data, feature_cols, horizon, interval
        )

        if not scaler_ok and not self.processor.is_fitted:
            raise ValueError("No valid training data after split")

        trained_stock_codes = [
            str(c).strip() for c in split_data.keys() if str(c).strip()
        ]

        # --- Phase 3: Create classifier sequences ---
        storage = self._create_sequences_from_splits(
            split_data, feature_cols, include_returns=True
        )

        X_train, y_train, r_train = self._combine_arrays(storage["train"])
        X_val, y_val, r_val = self._combine_arrays(storage["val"])
        X_test, y_test, r_test = self._combine_arrays(storage["test"])

        if X_train is None or len(X_train) == 0:
            raise ValueError("No training sequences available")

        self.input_size = int(X_train.shape[2])
        regime_profile = self._summarize_regime_shift(r_train, r_val, r_test)
        data_quality_summary = dict(self._last_data_quality_summary or {})
        effective_incremental = bool(incremental)
        incremental_guard = {
            "requested": bool(incremental),
            "effective": bool(incremental),
            "blocked": False,
            "reason": (
                "incremental_requested"
                if bool(incremental)
                else "full_retrain_requested"
            ),
            "scaler_refit": False,
        }

        log.info(
            f"Training data: {len(X_train)} samples, "
            f"{self.input_size} features"
        )
        if regime_profile.get("level") not in {"unknown", None}:
            log.info(
                "Regime profile: level=%s, score=%.3f, vol_ratio=%.2f",
                regime_profile.get("level", "unknown"),
                float(regime_profile.get("score", 0.0)),
                float(regime_profile.get("volatility_ratio", 1.0)),
            )

        regime_level = str(regime_profile.get("level", "unknown")).strip().lower()
        if effective_incremental and regime_level in _INCREMENTAL_REGIME_BLOCK_LEVELS:
            effective_incremental = False
            incremental_guard["blocked"] = True
            incremental_guard["reason"] = "high_regime_shift_requires_full_retrain"
            incremental_guard["effective"] = False
            log.warning(
                "Incremental training blocked for this cycle due to regime level=%s "
                "(score=%.3f); forcing full retrain path",
                regime_level,
                float(regime_profile.get("score", 0.0)),
            )

            # If an old scaler was reused, refit to current distribution.
            if self._skip_scaler_fit:
                self._skip_scaler_fit = False
                split_data, scaler_ok = self._split_and_fit_scaler(
                    raw_data, feature_cols, horizon, interval
                )
                if not scaler_ok and not self.processor.is_fitted:
                    raise ValueError("No valid training data after scaler refit")

                trained_stock_codes = [
                    str(c).strip() for c in split_data.keys() if str(c).strip()
                ]
                storage = self._create_sequences_from_splits(
                    split_data, feature_cols, include_returns=True
                )
                X_train, y_train, r_train = self._combine_arrays(storage["train"])
                X_val, y_val, r_val = self._combine_arrays(storage["val"])
                X_test, y_test, r_test = self._combine_arrays(storage["test"])
                if X_train is None or len(X_train) == 0:
                    raise ValueError(
                        "No training sequences available after scaler refit"
                    )
                self.input_size = int(X_train.shape[2])
                regime_profile = self._summarize_regime_shift(r_train, r_val, r_test)
                incremental_guard["scaler_refit"] = True

        incremental_guard["effective"] = bool(effective_incremental)
        log.info(
            "Incremental guard: requested=%s effective=%s blocked=%s reason=%s",
            bool(incremental_guard.get("requested")),
            bool(incremental_guard.get("effective")),
            bool(incremental_guard.get("blocked")),
            str(incremental_guard.get("reason", "")),
        )

        # Noise-vs-signal hardening:
        # downsample weak-return windows and upsample tail events.
        (
            X_train,
            y_train,
            r_train,
            sampling_guard,
        ) = self._rebalance_train_samples(X_train, y_train, r_train)
        if bool(sampling_guard.get("enabled", False)):
            log.info(
                "Sampling guard enabled: train=%s->%s, low_cut=%.3f%%, tail_cut=%.3f%%, "
                "low_drop=%s, tail_dup=%s",
                int(sampling_guard.get("input_samples", 0)),
                int(sampling_guard.get("output_samples", 0)),
                float(sampling_guard.get("low_signal_cutoff_pct", 0.0)),
                float(sampling_guard.get("tail_cutoff_pct", 0.0)),
                int(sampling_guard.get("low_signal_downsampled", 0)),
                int(sampling_guard.get("tail_upsampled", 0)),
            )
        else:
            log.info(
                "Sampling guard skipped: %s",
                str(sampling_guard.get("reason", "not_applicable")),
            )

        # --- Phase 4: Train classifier ensemble ---
        if effective_incremental:
            ensemble_path = live_ensemble_path
            if ensemble_path.exists():
                temp_ensemble = EnsembleModel(
                    input_size=self.input_size,
                    model_names=model_names,
                )
                if temp_ensemble.load(str(ensemble_path)):
                    required = list(model_names or _DEFAULT_ENSEMBLE_MODELS)
                    missing = [
                        m for m in required
                        if m not in set(temp_ensemble.models.keys())
                    ]
                    for name in missing:
                        try:
                            temp_ensemble._init_model(name)
                        except Exception as e:
                            log.warning(
                                "Failed to add missing model '%s' to ensemble: %s",
                                name,
                                e,
                            )
                    if missing:
                        temp_ensemble._normalize_weights()
                        log.info(
                            "Incremental ensemble upgraded with missing models: %s",
                            ", ".join(missing),
                        )
                    self.ensemble = temp_ensemble
                    log.info(
                        "Loaded existing ensemble for incremental training"
                    )
                else:
                    log.warning(
                        "Failed to load existing ensemble 鈥?"
                        "training from scratch"
                    )
                    self.ensemble = EnsembleModel(
                        input_size=self.input_size,
                        model_names=model_names,
                    )
            else:
                self.ensemble = EnsembleModel(
                    input_size=self.input_size,
                    model_names=model_names,
                )
        else:
            self.ensemble = EnsembleModel(
                input_size=self.input_size,
                model_names=model_names,
            )

        self.ensemble.interval = str(interval)
        self.ensemble.prediction_horizon = int(horizon)
        self.ensemble.trained_stock_codes = list(trained_stock_codes)
        trained_at = datetime.now().isoformat(timespec="seconds")
        known_last_train = dict(getattr(self.ensemble, "trained_stock_last_train", {}) or {})
        trained_code_set = {
            "".join(ch for ch in str(x).strip() if ch.isdigit())
            for x in list(trained_stock_codes or [])
        }
        trained_code_set = {c for c in trained_code_set if len(c) == 6}
        fresh_last_train = {
            code: str(known_last_train.get(code, "")).strip()
            for code in trained_code_set
            if str(known_last_train.get(code, "")).strip()
        }
        for code in trained_code_set:
            fresh_last_train[code] = trained_at
        self.ensemble.trained_stock_last_train = fresh_last_train

        if X_val is None or len(X_val) == 0:
            (
                X_train,
                y_train,
                r_train,
                X_val,
                y_val,
                r_val_fallback,
            ) = self._fallback_temporal_validation_split(
                X_train, y_train, r_train
            )
            if r_val is None or len(r_val) == 0:
                r_val = r_val_fallback
            if X_val is None or len(X_val) == 0:
                raise ValueError(
                    "No validation samples available after temporal fallback"
                )

        log.info(
            f"Training classifier: {len(X_train)} train, "
            f"{len(X_val)} val samples"
        )

        history = self.ensemble.train(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            callback=callback,
            stop_flag=stop_flag,
            learning_rate=effective_lr,
        )

        if self._should_stop(stop_flag):
            log.info("Training stopped by user")
            return {"status": "cancelled", "history": history}

        # --- Phase 5: Train forecaster ---
        # FIX CANCEL2: _train_forecaster now re-raises CancelledException
        forecaster_trained = False
        try:
            forecaster_trained = self._train_forecaster(
                split_data, feature_cols, horizon, interval,
                batch_size, epochs, stop_flag, save_model,
                effective_lr,
                forecast_save_path=(
                    candidate_forecast_path if save_model else None
                ),
            )
        except CancelledException:
            log.info("Training cancelled during forecaster phase")
            return {"status": "cancelled", "history": history}

        # --- Phase 6: Evaluate on test set ---
        test_metrics = {}
        if (
            X_test is not None
            and y_test is not None
            and r_test is not None
            and len(X_test) > 0
            and len(y_test) > 0
            and len(r_test) > 0
        ):
            eval_n = int(min(2000, len(X_test), len(y_test), len(r_test)))
            test_metrics = self._evaluate(
                X_test[:eval_n],
                y_test[:eval_n],
                r_test[:eval_n],
                regime_profile=regime_profile,
            )
            log.info(
                f"Test accuracy: {test_metrics.get('accuracy', 0):.2%}"
            )

        overfit_report = self._assess_overfitting(history)
        walk_forward = self._walk_forward_validate(
            X_val,
            y_val,
            r_val,
            X_test,
            y_test,
            r_test,
            regime_profile=regime_profile,
        )
        risk_adjusted_score = self._risk_adjusted_score(test_metrics)
        drift_guard = self._run_drift_guard(
            interval=interval,
            horizon=horizon,
            test_metrics=test_metrics,
            risk_adjusted_score=risk_adjusted_score,
        )
        quality_gate = self._build_quality_gate(
            test_metrics=test_metrics,
            walk_forward=walk_forward,
            overfit_report=overfit_report,
            drift_guard=drift_guard,
            data_quality=data_quality_summary,
            incremental_guard=incremental_guard,
        )

        log.info(
            "Quality gate: score=%.3f, action=%s",
            float(quality_gate.get("risk_adjusted_score", 0.0)),
            str(quality_gate.get("recommended_action", "shadow_mode_recommended")),
        )

        deployment = {
            "deployed": False,
            "reason": "save_model_disabled",
            "live_paths": {
                "ensemble": str(live_ensemble_path),
                "scaler": str(live_scaler_path),
                "forecast": str(live_forecast_path),
            },
            "candidate_paths": {
                "ensemble": str(candidate_ensemble_path),
                "scaler": str(candidate_scaler_path),
                "forecast": str(candidate_forecast_path),
            },
            "promoted_artifacts": [],
            "promotion_errors": [],
        }

        if save_model:
            deployment["reason"] = "quality_gate_block"
            try:
                self.processor.save_scaler(str(candidate_scaler_path))
                if self.ensemble is None:
                    raise RuntimeError("ensemble_not_initialized")
                self.ensemble.save(str(candidate_ensemble_path))
            except Exception as e:
                deployment["reason"] = f"candidate_save_failed:{e}"
                deployment["promotion_errors"] = [
                    str(deployment["reason"])
                ]
                log.warning(
                    "Failed to save candidate artifacts for deployment: %s", e
                )
            else:
                if bool(quality_gate.get("passed", False)):
                    promoted: list[str] = []
                    failed: list[str] = []

                    for name, candidate_path, live_path in [
                        ("ensemble", candidate_ensemble_path, live_ensemble_path),
                        ("scaler", candidate_scaler_path, live_scaler_path),
                    ]:
                        if self._promote_candidate_artifact(
                            candidate_path,
                            live_path,
                        ):
                            promoted.append(name)
                        else:
                            failed.append(f"{name}_promotion_failed")

                    if forecaster_trained:
                        if candidate_forecast_path.exists():
                            if self._promote_candidate_artifact(
                                candidate_forecast_path,
                                live_forecast_path,
                            ):
                                promoted.append("forecast")
                            else:
                                failed.append("forecast_promotion_failed")
                        else:
                            failed.append("forecast_candidate_missing")

                    deployment["promoted_artifacts"] = promoted
                    deployment["promotion_errors"] = failed
                    if failed:
                        deployment["reason"] = "promotion_failed"
                        log.warning(
                            "Model quality passed but deployment promotion failed: %s",
                            ", ".join(failed),
                        )
                    else:
                        deployment["deployed"] = True
                        deployment["reason"] = "deploy_ok"
                        log.info(
                            "Deployment completed: %s",
                            ", ".join(promoted) if promoted else "none",
                        )
                else:
                    action = str(
                        quality_gate.get(
                            "recommended_action",
                            "shadow_mode_recommended",
                        )
                    )
                    deployment["reason"] = f"quality_gate_block:{action}"
                    log.warning(
                        "Deployment blocked by quality gate; keeping candidate artifacts only"
                    )

        best_accuracy = 0.0
        if history:
            for h in history.values():
                val_acc_list = h.get("val_acc", [])
                if val_acc_list:
                    best_accuracy = max(best_accuracy, max(val_acc_list))

        test_acc = test_metrics.get("accuracy", 0.0)
        if best_accuracy == 0.0 and test_acc > 0.0:
            best_accuracy = test_acc

        return {
            "status": "complete",
            "history": history,
            "best_accuracy": float(best_accuracy),
            "test_metrics": test_metrics,
            "risk_adjusted_score": float(risk_adjusted_score),
            "quality_gate": quality_gate,
            "walk_forward_metrics": walk_forward,
            "overfit_report": overfit_report,
            "regime_profile": regime_profile,
            "drift_guard": drift_guard,
            "data_quality": data_quality_summary,
            "consistency_guard": dict(self._last_consistency_guard or {}),
            "incremental_guard": incremental_guard,
            "sampling_guard": sampling_guard,
            "deployment": deployment,
            "input_size": int(self.input_size),
            "num_models": (
                len(self.ensemble.models) if self.ensemble else 0
            ),
            "epochs": int(epochs),
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": (
                int(len(X_test)) if X_test is not None else 0
            ),
            "interval": str(interval),
            "prediction_horizon": int(horizon),
            "trained_stock_count": int(len(trained_stock_codes)),
            "trained_stock_codes": list(trained_stock_codes),
            "trained_stock_last_train": dict(self.ensemble.trained_stock_last_train or {}),
            "trained_at": str(trained_at),
            "forecaster_trained": forecaster_trained,
            "model_path": f"ensemble_{interval}_{horizon}.pt",
            "scaler_path": f"scaler_{interval}_{horizon}.pkl",
            "forecast_path": f"forecast_{interval}_{horizon}.pt",
        }

    # =========================================================================
    # =========================================================================

    def _train_forecaster(
        self,
        split_data: dict[str, dict[str, pd.DataFrame]],
        feature_cols: list[str],
        horizon: int,
        interval: str,
        batch_size: int,
        epochs: int,
        stop_flag: Any,
        save_model: bool,
        learning_rate: float,
        forecast_save_path: Path | None = None,
    ) -> bool:
        """
        Train multi-step forecaster. Returns True if successful.

        FIX CANCEL2: Re-raises CancelledException instead of swallowing it,
        so the caller (train()) knows to return cancelled status.
        """
        if self._should_stop(stop_flag):
            raise CancelledException()

        from models.networks import TCNModel

        Xf_train_list, Yf_train_list = [], []
        Xf_val_list, Yf_val_list = [], []

        for code, splits in split_data.items():
            if self._should_stop(stop_flag):
                raise CancelledException()

            for split_name, storage_x, storage_y in [
                ("train", Xf_train_list, Yf_train_list),
                ("val", Xf_val_list, Yf_val_list),
            ]:
                split_df = splits[split_name]
                if len(split_df) >= CONFIG.SEQUENCE_LENGTH + horizon + 5:
                    try:
                        Xf, Yf = self.processor.prepare_forecast_sequences(
                            split_df,
                            feature_cols,
                            horizon=horizon,
                            fit_scaler=False,
                        )
                        if len(Xf) > 0:
                            storage_x.append(Xf)
                            storage_y.append(Yf)
                    except Exception as e:
                        log.debug(
                            f"Forecast sequence failed for "
                            f"{code}/{split_name}: {e}"
                        )

        if not Xf_train_list or not Xf_val_list:
            log.warning("Forecaster training skipped: insufficient data")
            return False

        try:
            Xf_train = np.concatenate(Xf_train_list, axis=0)
            Yf_train = np.concatenate(Yf_train_list, axis=0)
            Xf_val = np.concatenate(Xf_val_list, axis=0)
            Yf_val = np.concatenate(Yf_val_list, axis=0)

            device = "cuda" if torch.cuda.is_available() else "cpu"

            forecaster = TCNModel(
                input_size=self.input_size,
                hidden_size=CONFIG.model.hidden_size,
                num_classes=horizon,
                dropout=CONFIG.model.dropout,
            ).to(device)

            optimizer = torch.optim.AdamW(
                forecaster.parameters(),
                lr=learning_rate,
                weight_decay=CONFIG.model.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
            )

            loss_fn = nn.MSELoss()

            train_loader = DataLoader(
                TensorDataset(
                    torch.FloatTensor(Xf_train),
                    torch.FloatTensor(Yf_train),
                ),
                batch_size=min(512, batch_size),
                shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(
                    torch.FloatTensor(Xf_val),
                    torch.FloatTensor(Yf_val),
                ),
                batch_size=1024,
                shuffle=False,
            )

            best_val_loss = float("inf")
            best_state = None
            patience = 0
            max_patience = 10
            fore_epochs = max(10, min(30, epochs // 2))

            log.info(
                f"Training forecaster: {len(Xf_train)} samples, "
                f"horizon={horizon}, lr={learning_rate:.6f}"
            )

            for ep in range(fore_epochs):
                if self._should_stop(stop_flag):
                    log.info("Forecaster training stopped")
                    raise CancelledException()

                forecaster.train()
                train_losses = []
                cancelled = False

                for batch_idx, (xb, yb) in enumerate(train_loader):
                    if (
                        batch_idx % _STOP_CHECK_INTERVAL == 0
                        and batch_idx > 0
                        and self._should_stop(stop_flag)
                    ):
                        cancelled = True
                        break

                    xb = xb.to(device)
                    yb = yb.to(device)

                    optimizer.zero_grad(set_to_none=True)
                    pred, _ = forecaster(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        forecaster.parameters(), 1.0
                    )
                    optimizer.step()
                    train_losses.append(float(loss.item()))

                if cancelled:
                    log.info("Forecaster training cancelled mid-epoch")
                    raise CancelledException()

                forecaster.eval()
                val_losses = []

                with torch.inference_mode():
                    for batch_idx, (xb, yb) in enumerate(val_loader):
                        if (
                            batch_idx % _STOP_CHECK_INTERVAL == 0
                            and batch_idx > 0
                            and self._should_stop(stop_flag)
                        ):
                            raise CancelledException()
                        xb = xb.to(device)
                        yb = yb.to(device)
                        pred, _ = forecaster(xb)
                        val_losses.append(
                            float(loss_fn(pred, yb).item())
                        )

                train_loss = float(np.mean(train_losses))
                val_loss = (
                    float(np.mean(val_losses))
                    if val_losses
                    else float("inf")
                )

                scheduler.step(val_loss)

                log.info(
                    f"Forecaster epoch {ep + 1}/{fore_epochs}: "
                    f"train_mse={train_loss:.6f}, val_mse={val_loss:.6f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in forecaster.state_dict().items()
                    }
                    patience = 0
                else:
                    patience += 1
                    if patience >= max_patience:
                        log.info("Forecaster early stopping")
                        break

            if best_state:
                forecaster.load_state_dict(best_state)

                target_path = forecast_save_path
                if target_path is None and save_model:
                    target_path = (
                        CONFIG.MODEL_DIR
                        / f"forecast_{interval}_{horizon}.pt"
                    )
                if target_path is not None:
                    payload = {
                        "input_size": int(self.input_size),
                        "interval": str(interval),
                        "horizon": int(horizon),
                        "arch": {
                            "hidden_size": int(CONFIG.model.hidden_size),
                            "dropout": float(CONFIG.model.dropout),
                        },
                        "state_dict": forecaster.state_dict(),
                    }

                    if atomic_torch_save is not None:
                        atomic_torch_save(target_path, payload)
                    else:
                        torch.save(payload, target_path)

                    try:
                        _write_artifact_checksum(Path(target_path))
                    except Exception as exc:
                        log.warning(
                            "Failed writing forecaster checksum sidecar for %s: %s",
                            target_path,
                            exc,
                        )

                    log.info(f"Forecaster saved: {target_path}")

                return True

        except CancelledException:
            # FIX CANCEL2: Re-raise so train() can handle it
            log.info("Forecaster training cancelled")
            raise
        except Exception as e:
            log.error(f"Forecaster training failed: {e}")
            import traceback
            traceback.print_exc()

        return False
    def get_ensemble(self) -> EnsembleModel | None:
        """Get the trained ensemble model."""
        return self.ensemble

    def save_training_report(
        self,
        results: dict[str, Any],
        path: str | None = None,
    ) -> None:
        """Save training report to file."""
        import json

        path = path or str(CONFIG.DATA_DIR / "training_report.json")

        def convert(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        report = convert(results)
        report["timestamp"] = datetime.now().isoformat()

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        try:
            from utils.atomic_io import atomic_write_json
            atomic_write_json(path, report)
        except ImportError:
            with open(path, "w") as f:
                json.dump(report, f, indent=2)

        log.info(f"Training report saved to {path}")

Trainer._assess_raw_data_quality = _assess_raw_data_quality_impl
Trainer._split_single_stock = _split_single_stock_impl
Trainer._fetch_raw_data = _fetch_raw_data_impl
Trainer._create_sequences_from_splits = _create_sequences_from_splits_impl
Trainer._rebalance_train_samples = _rebalance_train_samples_impl
Trainer._effective_confidence_floor = _effective_confidence_floor_impl
Trainer._fallback_temporal_validation_split = _fallback_temporal_validation_split_impl
Trainer._walk_forward_validate = _walk_forward_validate_impl
Trainer._build_quality_gate = _build_quality_gate_impl
Trainer._run_drift_guard = _run_drift_guard_impl
Trainer._trade_quality_thresholds = _trade_quality_thresholds_impl
Trainer._trade_masks = staticmethod(_trade_masks_impl)
Trainer._build_explainability_samples = _build_explainability_samples_impl
Trainer.prepare_data = _prepare_data_impl
Trainer._build_trading_stress_tests = _build_trading_stress_tests_impl
Trainer._evaluate = _evaluate_impl
Trainer._simulate_trading = _simulate_trading_impl
