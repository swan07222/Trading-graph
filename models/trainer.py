# models/trainer.py
from __future__ import annotations

import json
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from config.settings import CONFIG
from data.features import FeatureEngine
from data.fetcher import get_fetcher
from data.processor import DataProcessor
from models.ensemble import EnsembleModel
from models.trainer_data_ops import _assess_raw_data_quality as _assess_raw_data_quality_impl
from models.trainer_data_ops import (
    _create_sequences_from_splits as _create_sequences_from_splits_impl,
)
from models.trainer_data_ops import _effective_confidence_floor as _effective_confidence_floor_impl
from models.trainer_data_ops import (
    _fallback_temporal_validation_split as _fallback_temporal_validation_split_impl,
)
from models.trainer_data_ops import _fetch_raw_data as _fetch_raw_data_impl
from models.trainer_data_ops import _rebalance_train_samples as _rebalance_train_samples_impl
from models.trainer_data_ops import _split_single_stock as _split_single_stock_impl
from models.trainer_data_ops import _validate_temporal_split_integrity as _validate_temporal_split_integrity_impl
from models.trainer_data_ops import prepare_data as _prepare_data_impl
from models.trainer_eval_ops import (
    _build_explainability_samples as _build_explainability_samples_impl,
)
from models.trainer_eval_ops import _build_quality_gate as _build_quality_gate_impl
from models.trainer_eval_ops import _build_trading_stress_tests as _build_trading_stress_tests_impl
from models.trainer_eval_ops import _evaluate as _evaluate_impl
from models.trainer_eval_ops import _run_drift_guard as _run_drift_guard_impl
from models.trainer_eval_ops import _simulate_trading as _simulate_trading_impl
from models.trainer_eval_ops import _trade_masks as _trade_masks_impl
from models.trainer_eval_ops import _trade_quality_thresholds as _trade_quality_thresholds_impl
from models.trainer_eval_ops import _walk_forward_validate as _walk_forward_validate_impl
from models.trainer_pipeline_ops import _train_forecaster as _train_forecaster_impl
from models.trainer_pipeline_ops import train as _train_impl
from utils.logger import get_logger

log = get_logger(__name__)

# Import atomic_io at module level (not inside loops)
try:
    from utils.atomic_io import (
        atomic_write_json,
        read_json,
    )
except ImportError:
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

_TRAINING_INTERVAL_LOCK = "1m"
# FIX 1M: Reduced from 10080 to 480 bars - free sources provide 1-2 days of 1m data
_MIN_1M_LOOKBACK_BARS = 480
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
_STRESS_COST_MULTIPLIERS = (1.0, 1.5, 2.0)
_TAIL_STRESS_QUANTILE = 0.90
_MIN_TAIL_STRESS_SAMPLES = 24
_TAIL_EVENT_SHOCK_MIN_PCT = 1.0
_TAIL_EVENT_SHOCK_MAX_PCT = 6.0


class Trainer:
    """Complete training pipeline with proper data handling.

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

        # Incremental training flag 閳?set externally by auto_learner.
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
        """Default training lookback.
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
        """Check if training should stop 閳?handles multiple stop flag types."""
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
        """Split all stocks temporally, compute features per split,
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
                    "_skip_scaler_fit=True but no prior scaler exists 閳?"
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
        """Detect train/eval regime drift from return distribution changes.

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
        """Compute a deployment score using risk-first metrics, not accuracy alone."""
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
        return _train_impl(
            self,
            stock_codes=stock_codes,
            epochs=epochs,
            batch_size=batch_size,
            model_names=model_names,
            callback=callback,
            stop_flag=stop_flag,
            save_model=save_model,
            incremental=incremental,
            interval=interval,
            prediction_horizon=prediction_horizon,
            lookback_bars=lookback_bars,
            learning_rate=learning_rate,
        )

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
        return _train_forecaster_impl(
            self,
            split_data=split_data,
            feature_cols=feature_cols,
            horizon=horizon,
            interval=interval,
            batch_size=batch_size,
            epochs=epochs,
            stop_flag=stop_flag,
            save_model=save_model,
            learning_rate=learning_rate,
            forecast_save_path=forecast_save_path,
        )

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
Trainer._validate_temporal_split_integrity = _validate_temporal_split_integrity_impl
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
