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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config.settings import CONFIG
from data.features import FeatureEngine
from data.fetcher import get_fetcher
from data.processor import DataProcessor
from models.ensemble import EnsembleModel
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

    def __init__(self):
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

    def _assess_raw_data_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Validate raw OHLC quality before feature engineering.

        This blocks obvious bad data that can create misleading backtests.
        """
        rows = int(len(df)) if df is not None else 0
        report: dict[str, Any] = {
            "rows": rows,
            "passed": False,
            "nan_ratio": 1.0,
            "nonpos_price_ratio": 1.0,
            "broken_ohlc_ratio": 1.0,
            "reasons": [],
        }

        if rows <= 0 or df is None:
            report["reasons"] = ["empty_frame"]
            return report

        col_map = {str(c).strip().lower(): c for c in list(df.columns)}
        required = ["open", "high", "low", "close"]
        missing = [name for name in required if name not in col_map]
        if missing:
            report["reasons"] = [f"missing_columns:{','.join(sorted(missing))}"]
            return report

        try:
            open_s = pd.to_numeric(df[col_map["open"]], errors="coerce")
            high_s = pd.to_numeric(df[col_map["high"]], errors="coerce")
            low_s = pd.to_numeric(df[col_map["low"]], errors="coerce")
            close_s = pd.to_numeric(df[col_map["close"]], errors="coerce")
        except Exception as e:
            log.debug("OHLC numeric coercion failed: %s", e)
            report["reasons"] = ["ohlc_numeric_coercion_failed"]
            return report

        values = np.column_stack(
            [
                open_s.to_numpy(dtype=np.float64),
                high_s.to_numpy(dtype=np.float64),
                low_s.to_numpy(dtype=np.float64),
                close_s.to_numpy(dtype=np.float64),
            ]
        )
        total_cells = float(max(1, values.size))
        nan_cells = float(np.isnan(values).sum())
        nan_ratio = self._safe_ratio(nan_cells, total_cells)

        nonpos_rows = float(
            np.sum(
                (open_s <= 0).to_numpy()
                | (high_s <= 0).to_numpy()
                | (low_s <= 0).to_numpy()
                | (close_s <= 0).to_numpy()
            )
        )
        nonpos_ratio = self._safe_ratio(nonpos_rows, rows)

        broken_rows = float(
            np.sum(
                ((high_s < low_s).to_numpy())
                | ((high_s < open_s).to_numpy())
                | ((high_s < close_s).to_numpy())
                | ((low_s > open_s).to_numpy())
                | ((low_s > close_s).to_numpy())
            )
        )
        broken_ratio = self._safe_ratio(broken_rows, rows)

        reasons: list[str] = []
        if nan_ratio > _DATA_QUALITY_MAX_NAN_RATIO:
            reasons.append("nan_ratio_high")
        if nonpos_ratio > _DATA_QUALITY_MAX_NONPOS_PRICE_RATIO:
            reasons.append("non_positive_prices")
        if broken_ratio > _DATA_QUALITY_MAX_BROKEN_OHLC_RATIO:
            reasons.append("invalid_ohlc_relations")

        report.update(
            {
                "nan_ratio": float(nan_ratio),
                "nonpos_price_ratio": float(nonpos_ratio),
                "broken_ohlc_ratio": float(broken_ratio),
                "passed": bool(len(reasons) == 0),
                "reasons": reasons,
            }
        )
        return report

    # =========================================================================
    # =========================================================================

    def _split_single_stock(
        self,
        df_raw: pd.DataFrame,
        horizon: int,
        feature_cols: list[str],
    ) -> dict[str, pd.DataFrame] | None:
        """
        Split a single stock's RAW data temporally, compute features
        and labels WITHIN each split, and invalidate warmup rows.
        """
        n = len(df_raw)
        embargo = max(int(CONFIG.EMBARGO_BARS), horizon)
        seq_len = int(CONFIG.SEQUENCE_LENGTH)
        feature_lookback = int(CONFIG.data.feature_lookback)

        train_end = int(n * CONFIG.TRAIN_RATIO)
        val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
        val_start = train_end + embargo
        test_start = val_end + embargo

        if train_end < seq_len + 50:
            return None
        if val_start >= val_end or test_start >= n:
            return None

        train_raw = df_raw.iloc[:train_end].copy()

        val_raw_begin = max(0, val_start - feature_lookback)
        val_raw = df_raw.iloc[val_raw_begin:val_end].copy()

        test_raw_begin = max(0, test_start - feature_lookback)
        test_raw = df_raw.iloc[test_raw_begin:].copy()

        min_rows = self.feature_engine.MIN_ROWS
        for name, split_raw in [
            ("train", train_raw),
            ("val", val_raw),
            ("test", test_raw),
        ]:
            if len(split_raw) < min_rows:
                log.debug(
                    f"Split '{name}' has {len(split_raw)} rows < "
                    f"{min_rows} minimum for features"
                )
                if name == "train":
                    return None

        try:
            train_df = self.feature_engine.create_features(train_raw)
        except ValueError as e:
            log.debug(f"Train feature creation failed: {e}")
            return None

        try:
            val_df = self.feature_engine.create_features(val_raw)
        except ValueError:
            val_df = pd.DataFrame()

        try:
            test_df = self.feature_engine.create_features(test_raw)
        except ValueError:
            test_df = pd.DataFrame()

        for split_name, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            if len(split_df) == 0:
                continue
            missing = set(feature_cols) - set(split_df.columns)
            if missing:
                log.debug(f"Missing features in {split_name} split: {missing}")
                if split_name == "train":
                    return None

        if len(train_df) > 0:
            train_df = self.processor.create_labels(train_df, horizon=horizon)
        if len(val_df) > 0:
            val_df = self.processor.create_labels(val_df, horizon=horizon)
        if len(test_df) > 0:
            test_df = self.processor.create_labels(test_df, horizon=horizon)

        # FIX WARMUP: Clamp warmup indices to actual DataFrame length
        warmup_val = val_start - val_raw_begin
        warmup_test = test_start - test_raw_begin

        if warmup_val > 0 and len(val_df) > 0 and "label" in val_df.columns:
            clamp_val = min(warmup_val, len(val_df))
            if clamp_val > 0:
                val_df.iloc[
                    :clamp_val,
                    val_df.columns.get_loc("label"),
                ] = np.nan
        if warmup_test > 0 and len(test_df) > 0 and "label" in test_df.columns:
            clamp_test = min(warmup_test, len(test_df))
            if clamp_test > 0:
                test_df.iloc[
                    :clamp_test,
                    test_df.columns.get_loc("label"),
                ] = np.nan

        return {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

    # =========================================================================
    # SHARED DATA PREPARATION (eliminates duplication)
    # =========================================================================

    def _fetch_raw_data(
        self,
        stocks: list[str],
        interval: str,
        bars: int,
        stop_flag: Any = None,
        verbose: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Fetch raw OHLCV data for all stocks."""
        raw_data: dict[str, pd.DataFrame] = {}
        short_1m_codes: list[str] = []
        quality_reports: dict[str, dict[str, Any]] = {}
        reject_counts: dict[str, int] = {}
        consistency_guard: dict[str, Any] = {
            "reconcile_attempted": False,
            "pending_count": 0,
            "pending_codes": [],
        }
        pending_codes: set[str] = set()

        try:
            reconcile_fn = getattr(self.fetcher, "reconcile_pending_cache_sync", None)
            if callable(reconcile_fn):
                consistency_guard["reconcile_attempted"] = True
                try:
                    reconcile_report = reconcile_fn(
                        codes=list(stocks or []),
                        interval=interval,
                    )
                except TypeError:
                    reconcile_report = reconcile_fn()
                if isinstance(reconcile_report, dict):
                    consistency_guard["reconcile_report"] = dict(reconcile_report)

            pending_fn = getattr(self.fetcher, "get_pending_reconcile_codes", None)
            if callable(pending_fn):
                pending_raw = pending_fn(interval=interval)
                pending_codes = {
                    str(x).strip()
                    for x in list(pending_raw or [])
                    if str(x).strip()
                }
                consistency_guard["pending_count"] = int(len(pending_codes))
                consistency_guard["pending_codes"] = sorted(list(pending_codes))
        except Exception as exc:
            consistency_guard["error"] = str(exc)
            log.debug("Consistency preflight failed: %s", exc)

        self._last_consistency_guard = dict(consistency_guard)
        iterator = tqdm(stocks, desc="Loading stocks") if verbose else stocks

        for code in iterator:
            if self._should_stop(stop_flag):
                log.info("Data fetch stopped by user")
                break

            try:
                code_clean = ""
                try:
                    clean_fn = getattr(self.fetcher, "clean_code", None)
                    if callable(clean_fn):
                        code_clean = str(clean_fn(code) or "").strip()
                except Exception:
                    code_clean = ""
                if not code_clean:
                    digits = "".join(ch for ch in str(code).strip() if ch.isdigit())
                    code_clean = digits if len(digits) == 6 else ""

                if code_clean and code_clean in pending_codes:
                    reason = "pending_reconcile_consistency"
                    reject_counts[reason] = int(reject_counts.get(reason, 0)) + 1
                    quality_reports[str(code)] = {
                        "passed": False,
                        "reasons": [reason],
                    }
                    log.warning(
                        "Skipping %s for training until refresh reconcile completes",
                        code,
                    )
                    continue

                try:
                    df = self.fetcher.get_history(
                        code,
                        bars=bars,
                        interval=interval,
                        use_cache=True,
                        update_db=True,
                        allow_online=True,
                        refresh_intraday_after_close=True,
                    )
                except TypeError:
                    df = self.fetcher.get_history(
                        code,
                        bars=bars,
                        interval=interval,
                        use_cache=True,
                        update_db=True,
                    )
                if df is None or df.empty:
                    log.debug(f"No data for {code}")
                    continue

                df, sanitize_meta = self._sanitize_raw_history(df)
                q_report = self._assess_raw_data_quality(df)
                q_report["duplicates_removed"] = int(
                    sanitize_meta.get("duplicates_removed", 0)
                )
                quality_reports[str(code)] = q_report
                if not bool(q_report.get("passed", False)):
                    for reason in list(q_report.get("reasons", []) or []):
                        key = str(reason).strip().lower() or "unknown"
                        reject_counts[key] = int(reject_counts.get(key, 0)) + 1
                    log.warning(
                        "Rejecting %s due to raw data quality: %s",
                        code,
                        ",".join(
                            [str(x) for x in list(q_report.get("reasons", []) or [])]
                        ),
                    )
                    continue

                min_required = int(CONFIG.SEQUENCE_LENGTH + 80)
                if len(df) < min_required:
                    log.debug(
                        f"Insufficient data for {code}: "
                        f"{len(df)} bars (need {min_required})"
                    )
                    continue

                if (
                    str(interval).strip().lower() == _TRAINING_INTERVAL_LOCK
                    and len(df) < _MIN_1M_LOOKBACK_BARS
                ):
                    short_1m_codes.append(str(code))

                raw_data[code] = df

            except Exception as e:
                log.warning(f"Error fetching {code}: {e}")

        if short_1m_codes:
            log.warning(
                "1m training target is %s bars; %s stock(s) currently below target "
                "and will use best-available history (examples: %s)",
                _MIN_1M_LOOKBACK_BARS,
                len(short_1m_codes),
                ", ".join(short_1m_codes[:8]),
            )

        symbols_checked = int(len(quality_reports))
        symbols_passed = int(
            np.sum([1 for x in quality_reports.values() if bool(x.get("passed"))])
        )
        symbols_rejected = max(0, symbols_checked - symbols_passed)
        valid_ratio = (
            float(symbols_passed / symbols_checked) if symbols_checked > 0 else 0.0
        )
        top_reject_reasons = [
            k
            for k, _ in sorted(
                reject_counts.items(),
                key=lambda kv: (-int(kv[1]), str(kv[0])),
            )[:5]
        ]

        self._last_data_quality_summary = {
            "symbols_checked": symbols_checked,
            "symbols_passed": symbols_passed,
            "symbols_rejected": symbols_rejected,
            "valid_symbol_ratio": float(valid_ratio),
            "top_reject_reasons": top_reject_reasons,
            "consistency_guard": dict(self._last_consistency_guard or {}),
        }

        if symbols_checked > 0:
            log.info(
                "Data quality gate: %s/%s symbols passed (%.1f%%)",
                symbols_passed,
                symbols_checked,
                float(valid_ratio * 100.0),
            )
            if symbols_rejected > 0:
                log.warning(
                    "Data quality rejects: %s symbol(s), top reason(s): %s",
                    symbols_rejected,
                    ", ".join(top_reject_reasons) if top_reject_reasons else "n/a",
                )

        return raw_data

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

    def _create_sequences_from_splits(
        self,
        split_data: dict[str, dict[str, pd.DataFrame]],
        feature_cols: list[str],
        include_returns: bool = True,
    ) -> dict[str, dict[str, list]]:
        """Create sequences for train/val/test from split data."""
        storage = {
            "train": {"X": [], "y": [], "r": []},
            "val": {"X": [], "y": [], "r": []},
            "test": {"X": [], "y": [], "r": []},
        }

        for code, splits in split_data.items():
            for split_name in ("train", "val", "test"):
                split_df = splits[split_name]
                if len(split_df) >= CONFIG.SEQUENCE_LENGTH + 5:
                    try:
                        X, y, r = self.processor.prepare_sequences(
                            split_df,
                            feature_cols,
                            fit_scaler=False,
                        )
                        if len(X) > 0:
                            storage[split_name]["X"].append(X)
                            storage[split_name]["y"].append(y)
                            if include_returns:
                                storage[split_name]["r"].append(r)
                    except Exception as e:
                        log.debug(
                            f"Sequence creation failed for {code}/{split_name}: {e}"
                        )

        return storage

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

    def _rebalance_train_samples(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        r_train: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
        """
        Reduce low-signal noise and increase tail-event exposure in training.

        This is deterministic and only activates on sufficiently large,
        non-flat return distributions.
        """
        n = int(len(X_train))
        report: dict[str, Any] = {
            "enabled": False,
            "reason": "not_applicable",
            "input_samples": int(n),
            "output_samples": int(n),
            "low_signal_cutoff_pct": 0.0,
            "tail_cutoff_pct": 0.0,
            "low_signal_ratio": 0.0,
            "tail_ratio": 0.0,
            "low_signal_downsampled": 0,
            "tail_upsampled": 0,
        }

        if (
            X_train is None
            or y_train is None
            or r_train is None
            or n < 240
        ):
            report["reason"] = "insufficient_samples"
            return X_train, y_train, r_train, report

        r_arr = np.asarray(r_train, dtype=np.float64).reshape(-1)
        if len(r_arr) != n:
            report["reason"] = "returns_length_mismatch"
            return X_train, y_train, r_train, report

        finite_mask = np.isfinite(r_arr)
        if int(np.sum(finite_mask)) < int(max(200, n * 0.75)):
            report["reason"] = "insufficient_finite_returns"
            return X_train, y_train, r_train, report

        idx_all = np.arange(n, dtype=np.int64)
        idx_valid = idx_all[finite_mask]
        abs_returns = np.abs(r_arr[finite_mask])
        if len(abs_returns) < 200:
            report["reason"] = "insufficient_valid_returns"
            return X_train, y_train, r_train, report

        low_cut = float(np.percentile(abs_returns, 40))
        tail_cut = float(np.percentile(abs_returns, 90))
        if (
            not np.isfinite(low_cut)
            or not np.isfinite(tail_cut)
            or tail_cut <= (low_cut + _EPS)
            or tail_cut <= 0.02
        ):
            report["reason"] = "flat_or_low_dispersion_returns"
            return X_train, y_train, r_train, report

        low_mask_valid = abs_returns <= low_cut
        tail_mask_valid = abs_returns >= tail_cut
        low_idx = idx_valid[low_mask_valid]
        tail_idx = idx_valid[tail_mask_valid]

        if len(low_idx) < 40 or len(tail_idx) < 12:
            report["reason"] = "insufficient_low_or_tail_events"
            return X_train, y_train, r_train, report

        core_idx = idx_valid[~low_mask_valid]
        if len(core_idx) <= 0:
            report["reason"] = "empty_core_after_filter"
            return X_train, y_train, r_train, report

        keep_low_target = max(
            int(round(len(core_idx) * 0.50)),
            int(round(len(idx_valid) * 0.18)),
        )
        keep_low_target = int(min(len(low_idx), keep_low_target))
        if keep_low_target <= 0:
            keep_low_target = min(len(low_idx), 20)

        seed = int(
            _SEED
            + n
            + int(np.clip(np.mean(abs_returns) * 1000.0, 0.0, 100000.0))
        )
        rng = np.random.RandomState(seed)

        if keep_low_target < len(low_idx):
            kept_low = np.sort(
                rng.choice(low_idx, size=keep_low_target, replace=False)
            )
        else:
            kept_low = np.sort(low_idx)

        rebalance_idx = np.concatenate([np.sort(core_idx), kept_low], axis=0)

        current_tail = int(np.sum(np.isin(rebalance_idx, tail_idx)))
        target_tail = int(round(len(rebalance_idx) * 0.20))
        needed_tail = max(0, target_tail - current_tail)
        max_tail_dup = int(
            min(
                max(0, len(rebalance_idx) // 3),
                max(0, len(tail_idx) * 3),
            )
        )
        dup_tail = int(min(needed_tail, max_tail_dup))
        if dup_tail > 0:
            extra_tail = rng.choice(tail_idx, size=dup_tail, replace=True)
            rebalance_idx = np.concatenate([rebalance_idx, extra_tail], axis=0)
        else:
            dup_tail = 0

        if len(rebalance_idx) < max(120, int(round(n * 0.35))):
            report["reason"] = "rebalance_too_aggressive"
            return X_train, y_train, r_train, report

        perm = np.arange(len(rebalance_idx), dtype=np.int64)
        rng.shuffle(perm)
        rebalance_idx = rebalance_idx[perm]

        X_out = X_train[rebalance_idx]
        y_out = y_train[rebalance_idx]
        r_out = r_arr[rebalance_idx]

        report.update(
            {
                "enabled": True,
                "reason": "ok",
                "output_samples": int(len(X_out)),
                "low_signal_cutoff_pct": float(low_cut),
                "tail_cutoff_pct": float(tail_cut),
                "low_signal_ratio": float(len(low_idx) / max(1, len(idx_valid))),
                "tail_ratio": float(len(tail_idx) / max(1, len(idx_valid))),
                "low_signal_downsampled": int(
                    max(0, len(low_idx) - len(kept_low))
                ),
                "tail_upsampled": int(dup_tail),
                "seed": int(seed),
            }
        )
        return X_out, y_out, r_out, report

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

    def _effective_confidence_floor(
        self, regime_profile: dict[str, Any] | None = None
    ) -> float:
        """Resolve dynamic confidence floor (base + regime-aware boost)."""
        base = float(CONFIG.MIN_CONFIDENCE)
        boost = 0.0
        if regime_profile:
            try:
                boost = float(regime_profile.get("confidence_boost", 0.0))
            except (TypeError, ValueError):
                boost = 0.0
        return float(min(0.95, max(base, base + boost)))

    def _fallback_temporal_validation_split(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        r_train: np.ndarray | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
    ]:
        """
        Build a temporal validation holdout with embargo when val split is empty.

        This avoids random slicing leakage by preserving order and introducing
        a non-overlap gap before validation.
        """
        total = int(len(X_train))
        if total <= 2:
            empty_x = np.zeros((0, *X_train.shape[1:]), dtype=X_train.dtype)
            empty_y = np.zeros((0,), dtype=y_train.dtype)
            empty_r = (
                np.zeros((0,), dtype=r_train.dtype)
                if r_train is not None
                else None
            )
            return X_train, y_train, r_train, empty_x, empty_y, empty_r

        holdout = max(1, total // 7)
        raw_gap = max(
            int(CONFIG.EMBARGO_BARS),
            int(self.prediction_horizon),
            1,
        )
        gap = max(1, min(raw_gap, max(1, total // 8)))

        train_end = max(1, total - holdout - gap)
        val_start = min(total, train_end + gap)
        if val_start >= total:
            val_start = max(1, total - holdout)
            train_end = max(1, val_start - 1)

        X_tr = X_train[:train_end]
        y_tr = y_train[:train_end]
        X_val = X_train[val_start:]
        y_val = y_train[val_start:]

        if len(X_val) == 0:
            split = max(1, total - 1)
            X_tr = X_train[:split]
            y_tr = y_train[:split]
            X_val = X_train[split:]
            y_val = y_train[split:]
            gap = 0

        if r_train is None:
            r_tr = None
            r_val = None
        else:
            r_tr = r_train[: len(X_tr)]
            r_val = r_train[val_start: val_start + len(X_val)]
            if len(r_val) == 0:
                r_val = r_train[len(X_tr): len(X_tr) + len(X_val)]

        log.info(
            "Validation fallback used temporal holdout with embargo "
            "(train=%s, val=%s, gap=%s)",
            len(X_tr),
            len(X_val),
            gap,
        )
        return X_tr, y_tr, r_tr, X_val, y_val, r_val

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

    def _walk_forward_validate(
        self,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        r_val: np.ndarray | None,
        X_test: np.ndarray | None,
        y_test: np.ndarray | None,
        r_test: np.ndarray | None,
        regime_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate stability on contiguous forward windows.

        This is a post-train diagnostic (not re-training folds) used to
        detect unstable model behavior across recent slices.
        """
        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        r_parts: list[np.ndarray] = []

        for X_arr, y_arr, r_arr in [
            (X_val, y_val, r_val),
            (X_test, y_test, r_test),
        ]:
            if (
                X_arr is None
                or y_arr is None
                or r_arr is None
                or len(X_arr) == 0
                or len(y_arr) == 0
                or len(r_arr) == 0
            ):
                continue
            n = int(min(len(X_arr), len(y_arr), len(r_arr)))
            if n <= 0:
                continue
            X_parts.append(X_arr[:n])
            y_parts.append(y_arr[:n])
            r_parts.append(r_arr[:n])

        if not X_parts:
            return {
                "enabled": False,
                "reason": "no_eval_data",
            }

        X_eval = np.concatenate(X_parts, axis=0)
        y_eval = np.concatenate(y_parts, axis=0)
        r_eval = np.concatenate(r_parts, axis=0)

        max_samples = 3000
        if len(X_eval) > max_samples:
            X_eval = X_eval[-max_samples:]
            y_eval = y_eval[-max_samples:]
            r_eval = r_eval[-max_samples:]

        fold_count = int(_WALK_FORWARD_FOLDS)
        min_fold_size = max(int(CONFIG.SEQUENCE_LENGTH), 64)
        min_required = max(
            _MIN_WALK_FORWARD_SAMPLES,
            fold_count * min_fold_size,
        )
        if len(X_eval) < min_required:
            return {
                "enabled": False,
                "reason": (
                    f"insufficient_samples (need>={min_required}, "
                    f"got={len(X_eval)})"
                ),
            }

        fold_size = len(X_eval) // fold_count
        fold_results: list[dict[str, Any]] = []

        for fold in range(fold_count):
            start = int(fold * fold_size)
            end = (
                int(len(X_eval))
                if fold == fold_count - 1
                else int((fold + 1) * fold_size)
            )
            if end - start < min_fold_size:
                continue

            metrics = self._evaluate(
                X_eval[start:end],
                y_eval[start:end],
                r_eval[start:end],
                regime_profile=regime_profile,
            )
            fold_results.append(
                {
                    "fold": int(fold + 1),
                    "start": int(start),
                    "end": int(end),
                    "accuracy": float(metrics.get("accuracy", 0.0)),
                    "risk_adjusted_score": float(
                        metrics.get("risk_adjusted_score", 0.0)
                    ),
                    "sharpe_ratio": float(
                        (metrics.get("trading", {}) or {}).get(
                            "sharpe_ratio", 0.0
                        )
                    ),
                }
            )

        if not fold_results:
            return {
                "enabled": False,
                "reason": "fold_construction_failed",
            }

        accs = np.array(
            [x["accuracy"] for x in fold_results], dtype=np.float64
        )
        scores = np.array(
            [x["risk_adjusted_score"] for x in fold_results],
            dtype=np.float64,
        )

        acc_mean = float(np.mean(accs))
        acc_std = float(np.std(accs))
        score_mean = float(np.mean(scores))
        score_std = float(np.std(scores))

        stability = 1.0 - (
            0.6 * (acc_std / (acc_mean + _EPS))
            + 0.4 * (score_std / (abs(score_mean) + 0.1))
        )
        stability = float(np.clip(stability, 0.0, 1.0))

        return {
            "enabled": True,
            "folds": fold_results,
            "mean_accuracy": acc_mean,
            "std_accuracy": acc_std,
            "mean_risk_adjusted_score": score_mean,
            "std_risk_adjusted_score": score_std,
            "stability_score": stability,
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

    def _build_quality_gate(
        self,
        test_metrics: dict[str, Any],
        walk_forward: dict[str, Any],
        overfit_report: dict[str, Any],
        drift_guard: dict[str, Any],
        data_quality: dict[str, Any] | None = None,
        incremental_guard: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Combine diagnostics into a single deployment recommendation."""
        trading = test_metrics.get("trading", {}) or {}
        stress_tests = test_metrics.get("stress_tests", {}) or {}
        risk_score = self._risk_adjusted_score(test_metrics)
        walk_enabled = bool(walk_forward.get("enabled", False))
        walk_stability = float(
            walk_forward.get("stability_score", 0.0) if walk_enabled else 0.0
        )

        dq = data_quality if isinstance(data_quality, dict) else {}
        symbols_checked = int(dq.get("symbols_checked", 0))
        valid_symbol_ratio = float(
            np.clip(dq.get("valid_symbol_ratio", 1.0), 0.0, 1.0)
        )
        data_quality_ok = bool(
            symbols_checked <= 0
            or valid_symbol_ratio >= _DATA_QUALITY_MIN_VALID_SYMBOL_RATIO
        )

        ig = incremental_guard if isinstance(incremental_guard, dict) else {}
        incremental_ok = not bool(ig.get("blocked", False))
        tail_stress_ok = bool(stress_tests.get("tail_guard_passed", True))
        cost_resilience_ok = bool(
            stress_tests.get("cost_resilience_passed", True)
        )
        trade_count = int(max(0, trading.get("trades", 0)))

        checks = {
            "risk_score": bool(risk_score >= 0.52),
            "profit_factor": bool(float(trading.get("profit_factor", 0.0)) >= 1.05),
            "drawdown": bool(float(trading.get("max_drawdown", 1.0)) <= 0.25),
            "trade_count": bool(trade_count >= _MIN_BASELINE_TRADES),
            "overfit": not bool(overfit_report.get("detected", False)),
            "drift": str(drift_guard.get("action", "")) != "rollback_recommended",
            "walk_forward": (not walk_enabled) or bool(walk_stability >= 0.35),
            "data_quality": bool(data_quality_ok),
            "tail_stress": bool(tail_stress_ok),
            "cost_resilience": bool(cost_resilience_ok),
            "incremental_guard": bool(incremental_ok),
        }

        reasons: list[str] = []
        if not checks["risk_score"]:
            reasons.append("risk_adjusted_score_below_threshold")
        if not checks["profit_factor"]:
            reasons.append("profit_factor_below_1.05")
        if not checks["drawdown"]:
            reasons.append("max_drawdown_above_25pct")
        if not checks["trade_count"]:
            reasons.append("insufficient_trade_count")
        if not checks["overfit"]:
            reasons.append("overfitting_detected")
        if not checks["drift"]:
            reasons.append("drift_guard_block")
        if not checks["walk_forward"]:
            reasons.append("walk_forward_instability")
        if not checks["data_quality"]:
            reasons.append("data_quality_gate_failed")
        if not checks["tail_stress"]:
            reasons.append("tail_stress_failure")
        if not checks["cost_resilience"]:
            reasons.append("cost_sensitivity_failure")
        if not checks["incremental_guard"]:
            reasons.append("incremental_regime_guard_block")

        if str(drift_guard.get("action", "")) == "rollback_recommended":
            action = "rollback_recommended"
            passed = False
        elif all(checks.values()):
            action = "deploy_ok"
            passed = True
        else:
            action = "shadow_mode_recommended"
            passed = False

        return {
            "passed": bool(passed),
            "recommended_action": action,
            "checks": checks,
            "failed_reasons": reasons,
            "risk_adjusted_score": float(risk_score),
            "walk_forward_stability": float(walk_stability),
            "data_quality_ratio": float(valid_symbol_ratio),
            "stress_tests": stress_tests if isinstance(stress_tests, dict) else {},
            "incremental_guard": ig,
        }

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

    def _run_drift_guard(
        self,
        interval: str,
        horizon: int,
        test_metrics: dict[str, Any],
        risk_adjusted_score: float,
    ) -> dict[str, Any]:
        """Compare current run with previous baseline and recommend rollout mode."""
        path = self._drift_baseline_path(interval, horizon)
        baseline = self._read_json_safely(path) if path.exists() else None

        trading = test_metrics.get("trading", {}) or {}
        current = {
            "accuracy": float(test_metrics.get("accuracy", 0.0)),
            "risk_adjusted_score": float(risk_adjusted_score),
            "sharpe_ratio": float(trading.get("sharpe_ratio", 0.0)),
            "profit_factor": float(trading.get("profit_factor", 0.0)),
            "max_drawdown": float(trading.get("max_drawdown", 0.0)),
            "excess_return": float(trading.get("excess_return", 0.0)),
            "trades": int(trading.get("trades", 0)),
        }

        baseline_metrics: dict[str, Any] = {}
        score_drop = 0.0
        accuracy_drop = 0.0
        action = "no_baseline"

        if baseline:
            baseline_metrics = dict(baseline.get("metrics", {}) or {})
            prev_score = float(baseline_metrics.get("risk_adjusted_score", 0.0))
            prev_acc = float(baseline_metrics.get("accuracy", 0.0))
            score_drop = max(0.0, prev_score - current["risk_adjusted_score"])
            accuracy_drop = max(0.0, prev_acc - current["accuracy"])

            if (
                score_drop >= _DRIFT_BLOCK_SCORE_DROP
                or accuracy_drop >= _DRIFT_BLOCK_ACC_DROP
            ):
                action = "rollback_recommended"
            elif (
                score_drop >= _DRIFT_WARN_SCORE_DROP
                or accuracy_drop >= _DRIFT_WARN_ACC_DROP
            ):
                action = "shadow_mode_recommended"
            else:
                action = "deploy_ok"

        meets_floor = self._meets_baseline_quality_floor(current)
        baseline_update_block_reason = ""
        should_update_baseline = baseline is None and meets_floor

        if baseline and meets_floor:
            prev_score = float(
                (baseline.get("metrics", {}) or {}).get(
                    "risk_adjusted_score", -1.0
                )
            )
            should_update_baseline = bool(
                current["risk_adjusted_score"] >= prev_score
            )
        elif not meets_floor:
            baseline_update_block_reason = "quality_floor_not_met"

        baseline_updated = False
        if should_update_baseline:
            payload = {
                "updated_at": datetime.now().isoformat(),
                "interval": str(interval),
                "horizon": int(horizon),
                "metrics": current,
            }
            baseline_updated = self._write_json_safely(path, payload)

        return {
            "baseline_path": str(path),
            "baseline_found": bool(baseline is not None),
            "baseline_metrics": baseline_metrics,
            "current_metrics": current,
            "score_drop": float(score_drop),
            "accuracy_drop": float(accuracy_drop),
            "action": action,
            "meets_quality_floor": bool(meets_floor),
            "baseline_update_block_reason": str(baseline_update_block_reason),
            "baseline_updated": bool(baseline_updated),
        }

    def _trade_quality_thresholds(
        self, confidence_floor: float | None = None
    ) -> dict[str, float]:
        """Thresholds for confidence-first no-trade filtering."""
        precision_cfg = getattr(CONFIG, "precision", None)

        min_confidence = max(
            float(CONFIG.MIN_CONFIDENCE),
            float(
                confidence_floor
                if confidence_floor is not None
                else CONFIG.MIN_CONFIDENCE
            ),
        )
        min_agreement = 0.55
        max_entropy = 0.80
        min_edge = 0.03

        if precision_cfg is not None:
            min_agreement = max(
                min_agreement,
                float(getattr(precision_cfg, "min_agreement", min_agreement))
                - 0.10,
            )
            max_entropy = min(
                max_entropy,
                float(getattr(precision_cfg, "max_entropy", max_entropy))
                + 0.15,
            )
            min_edge = max(
                min_edge,
                float(getattr(precision_cfg, "min_edge", min_edge)) * 0.5,
            )

        min_margin = max(0.04, min(0.20, (min_edge * 0.8) + 0.02))

        return {
            "min_confidence": float(min_confidence),
            "min_agreement": float(min_agreement),
            "max_entropy": float(max_entropy),
            "min_margin": float(min_margin),
            "min_edge": float(min_edge),
        }

    @staticmethod
    def _trade_masks(
        preds: np.ndarray,
        confs: np.ndarray,
        agreements: np.ndarray | None,
        entropies: np.ndarray | None,
        margins: np.ndarray | None,
        edges: np.ndarray | None,
        thresholds: dict[str, float],
    ) -> dict[str, np.ndarray]:
        """Build per-signal eligibility masks for no-trade filtering."""
        n = int(len(preds))
        is_up = np.asarray(preds).reshape(-1) == 2
        conf_ok = np.asarray(confs, dtype=np.float64).reshape(-1) >= float(
            thresholds["min_confidence"]
        )

        if agreements is not None and len(agreements) == n:
            agreement_ok = np.asarray(agreements, dtype=np.float64).reshape(-1) >= float(
                thresholds["min_agreement"]
            )
        else:
            agreement_ok = np.ones(n, dtype=bool)

        if entropies is not None and len(entropies) == n:
            entropy_ok = np.asarray(entropies, dtype=np.float64).reshape(-1) <= float(
                thresholds["max_entropy"]
            )
        else:
            entropy_ok = np.ones(n, dtype=bool)

        if margins is not None and len(margins) == n:
            margin_ok = np.asarray(margins, dtype=np.float64).reshape(-1) >= float(
                thresholds["min_margin"]
            )
        else:
            margin_ok = np.ones(n, dtype=bool)

        if edges is not None and len(edges) == n:
            edge_ok = np.asarray(edges, dtype=np.float64).reshape(-1) >= float(
                thresholds["min_edge"]
            )
        else:
            edge_ok = np.ones(n, dtype=bool)

        eligible = is_up & conf_ok & agreement_ok & entropy_ok & margin_ok & edge_ok

        return {
            "is_up": is_up,
            "conf": conf_ok,
            "agreement": agreement_ok,
            "entropy": entropy_ok,
            "margin": margin_ok,
            "edge": edge_ok,
            "eligible": eligible,
        }

    def _build_explainability_samples(
        self,
        predictions: list[Any],
        sample_count: int,
        thresholds: dict[str, float],
        masks: dict[str, np.ndarray],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Create compact per-decision diagnostics for top-confidence samples."""
        if sample_count <= 0 or not predictions:
            return []

        n = int(min(sample_count, len(predictions)))
        rank = sorted(
            range(n),
            key=lambda i: float(getattr(predictions[i], "confidence", 0.0)),
            reverse=True,
        )
        out: list[dict[str, Any]] = []

        for idx in rank[: int(max(1, limit))]:
            pred = predictions[idx]
            pred_cls = int(getattr(pred, "predicted_class", 1))
            probs = np.asarray(
                getattr(pred, "probabilities", np.array([0.0, 0.0, 0.0])),
                dtype=np.float64,
            ).reshape(-1)
            probs3 = [float(x) for x in probs[:3]]
            while len(probs3) < 3:
                probs3.append(0.0)

            reasons: list[str] = []
            if pred_cls != 2:
                reasons.append("predicted_not_up")
            if not bool(masks["conf"][idx]):
                reasons.append("low_confidence")
            if not bool(masks["agreement"][idx]):
                reasons.append("low_agreement")
            if not bool(masks["entropy"][idx]):
                reasons.append("high_entropy")
            if not bool(masks["margin"][idx]):
                reasons.append("low_margin")
            if not bool(masks["edge"][idx]):
                reasons.append("low_edge")

            action = "TRADE_LONG" if bool(masks["eligible"][idx]) else "NO_TRADE"
            reason = reasons[0] if reasons else "passed_all_filters"

            out.append(
                {
                    "index": int(idx),
                    "predicted_class": int(pred_cls),
                    "confidence": float(getattr(pred, "confidence", 0.0)),
                    "agreement": float(getattr(pred, "agreement", 0.0)),
                    "entropy": float(getattr(pred, "entropy", 1.0)),
                    "margin": float(getattr(pred, "margin", 0.0)),
                    "edge": float(abs(probs3[2] - probs3[0])),
                    "probabilities": probs3,
                    "action": action,
                    "primary_reason": reason,
                    "thresholds": {
                        "min_confidence": float(thresholds["min_confidence"]),
                        "min_agreement": float(thresholds["min_agreement"]),
                        "max_entropy": float(thresholds["max_entropy"]),
                        "min_margin": float(thresholds["min_margin"]),
                        "min_edge": float(thresholds["min_edge"]),
                    },
                }
            )

        return out

    # =========================================================================
    # prepare_data (standalone, used by external callers)
    # =========================================================================

    def prepare_data(
        self,
        stock_codes: list[str] = None,
        min_samples_per_stock: int = 100,
        verbose: bool = True,
        interval: str = "1m",
        prediction_horizon: int = None,
        lookback_bars: int = None,
    ) -> tuple:
        """
        Prepare training data with proper temporal split.

        Returns:
            Tuple of (X_train, y_train, r_train,
                       X_val,   y_val,   r_val,
                       X_test,  y_test,  r_test)
        """
        stocks = stock_codes or CONFIG.STOCK_POOL
        interval = self._enforce_training_interval(interval)
        horizon = int(prediction_horizon or CONFIG.PREDICTION_HORIZON)
        bars = int(
            lookback_bars
            if lookback_bars is not None
            else self._default_lookback_bars(interval)
        )
        if interval == _TRAINING_INTERVAL_LOCK:
            bars = int(max(bars, _MIN_1M_LOOKBACK_BARS))

        self.interval = interval
        self.prediction_horizon = horizon

        log.info(f"Preparing data for {len(stocks)} stocks...")
        log.info(
            f"Interval: {interval}, Horizon: {horizon}, Lookback: {bars}"
        )
        log.info(
            f"Temporal split: Train={CONFIG.TRAIN_RATIO:.0%}, "
            f"Val={CONFIG.VAL_RATIO:.0%}, Test={CONFIG.TEST_RATIO:.0%}"
        )

        feature_cols = self.feature_engine.get_feature_columns()

        raw_data = self._fetch_raw_data(stocks, interval, bars, verbose=verbose)

        if not raw_data:
            raise ValueError("No valid stock data available for training")

        log.info(f"Successfully loaded {len(raw_data)} stocks")

        split_data, scaler_ok = self._split_and_fit_scaler(
            raw_data, feature_cols, horizon, interval
        )

        if not scaler_ok and not self.processor.is_fitted:
            raise ValueError("No valid training data after split")

        storage = self._create_sequences_from_splits(split_data, feature_cols)

        X_train, y_train, r_train = self._combine_arrays(storage["train"])
        X_val, y_val, r_val = self._combine_arrays(storage["val"])
        X_test, y_test, r_test = self._combine_arrays(storage["test"])

        if X_train is None or len(X_train) == 0:
            raise ValueError("No training sequences available")

        self.input_size = int(X_train.shape[2])

        log.info("Data prepared:")
        log.info(f"  Train: {len(X_train)} samples")
        log.info(f"  Val:   {len(X_val) if X_val is not None else 0} samples")
        log.info(f"  Test:  {len(X_test) if X_test is not None else 0} samples")
        log.info(f"  Input size: {self.input_size} features")

        if len(y_train) > 0:
            dist = self.processor.get_class_distribution(y_train)
            log.info(
                f"  Class distribution: DOWN={dist['DOWN']}, "
                f"NEUTRAL={dist['NEUTRAL']}, UP={dist['UP']}"
            )

        scaler_path = CONFIG.MODEL_DIR / f"scaler_{interval}_{horizon}.pkl"
        self.processor.save_scaler(str(scaler_path))

        # FIX EMPTY: Return properly shaped empty arrays instead of 1D empty
        seq_len = int(CONFIG.SEQUENCE_LENGTH)
        n_feat = self.input_size

        def safe(arr, is_X=False):
            if arr is not None:
                return arr
            if is_X:
                return np.zeros((0, seq_len, n_feat), dtype=np.float32)
            return np.zeros((0,), dtype=np.float32)

        return (
            safe(X_train, True), safe(y_train), safe(r_train),
            safe(X_val, True), safe(y_val), safe(r_val),
            safe(X_test, True), safe(y_test), safe(r_test),
        )

    # =========================================================================
    # MAIN train() METHOD
    # =========================================================================

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

    # =========================================================================
    # =========================================================================

    def _build_trading_stress_tests(
        self,
        preds: np.ndarray,
        confs: np.ndarray,
        returns: np.ndarray,
        agreements: np.ndarray | None,
        entropies: np.ndarray | None,
        margins: np.ndarray | None,
        edges: np.ndarray | None,
        confidence_floor: float,
        masks: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """
        Build robustness diagnostics for tail events and higher trading costs.

        These checks are used to reduce live deployment of fragile models.
        """
        cost_scenarios: list[dict[str, Any]] = []
        for multiplier in _STRESS_COST_MULTIPLIERS:
            sim = self._simulate_trading(
                preds,
                confs,
                returns,
                agreements=agreements,
                entropies=entropies,
                margins=margins,
                edges=edges,
                confidence_floor=confidence_floor,
                masks=masks,
                cost_multiplier=float(multiplier),
            )
            cost_scenarios.append(
                {
                    "cost_multiplier": float(multiplier),
                    "profit_factor": float(sim.get("profit_factor", 0.0)),
                    "max_drawdown": float(sim.get("max_drawdown", 0.0)),
                    "excess_return": float(sim.get("excess_return", 0.0)),
                    "trades": int(sim.get("trades", 0)),
                }
            )

        base_cost = cost_scenarios[0] if cost_scenarios else {}
        high_cost = cost_scenarios[-1] if cost_scenarios else {}
        cost_resilience_passed = True
        cost_resilience_reason = "insufficient_trades"
        if int(base_cost.get("trades", 0)) >= _MIN_BASELINE_TRADES:
            base_pf = max(float(base_cost.get("profit_factor", 0.0)), _EPS)
            high_pf = float(high_cost.get("profit_factor", 0.0))
            pf_drop = max(0.0, (base_pf - high_pf) / base_pf)
            high_dd = float(high_cost.get("max_drawdown", 1.0))
            cost_resilience_passed = bool(
                high_pf >= 0.90 and pf_drop <= 0.45 and high_dd <= 0.35
            )
            cost_resilience_reason = (
                "ok" if cost_resilience_passed else "high_cost_degradation"
            )

        returns_arr = np.asarray(returns, dtype=np.float64).reshape(-1)
        abs_returns = np.abs(returns_arr)
        tail_block: dict[str, Any] = {
            "enabled": False,
            "reason": "insufficient_samples",
            "tail_samples": 0,
            "quantile": float(_TAIL_STRESS_QUANTILE),
        }
        tail_guard_passed = True
        tail_guard_reason = "insufficient_samples"

        if len(abs_returns) >= _MIN_TAIL_STRESS_SAMPLES:
            threshold = float(
                np.percentile(abs_returns, float(_TAIL_STRESS_QUANTILE * 100.0))
            )
            tail_mask = abs_returns >= threshold
            tail_samples = int(np.sum(tail_mask))
            min_tail_samples = max(8, int(round(len(abs_returns) * 0.08)))

            if tail_samples >= min_tail_samples:
                tail_preds = np.asarray(preds)[tail_mask]
                tail_confs = np.asarray(confs)[tail_mask]
                tail_returns = returns_arr[tail_mask]

                tail_masks = {
                    name: np.asarray(val, dtype=bool)[tail_mask]
                    for name, val in masks.items()
                    if isinstance(val, np.ndarray) and len(val) == len(tail_mask)
                }
                tail_sim = self._simulate_trading(
                    tail_preds,
                    tail_confs,
                    tail_returns,
                    agreements=(
                        np.asarray(agreements)[tail_mask]
                        if agreements is not None and len(agreements) == len(tail_mask)
                        else None
                    ),
                    entropies=(
                        np.asarray(entropies)[tail_mask]
                        if entropies is not None and len(entropies) == len(tail_mask)
                        else None
                    ),
                    margins=(
                        np.asarray(margins)[tail_mask]
                        if margins is not None and len(margins) == len(tail_mask)
                        else None
                    ),
                    edges=(
                        np.asarray(edges)[tail_mask]
                        if edges is not None and len(edges) == len(tail_mask)
                        else None
                    ),
                    confidence_floor=confidence_floor,
                    masks=tail_masks if tail_masks else None,
                )

                shock_pct = float(
                    np.clip(
                        np.percentile(abs_returns, 95) * 1.5,
                        _TAIL_EVENT_SHOCK_MIN_PCT,
                        _TAIL_EVENT_SHOCK_MAX_PCT,
                    )
                )
                shock_sim = self._simulate_trading(
                    preds,
                    confs,
                    returns_arr,
                    agreements=agreements,
                    entropies=entropies,
                    margins=margins,
                    edges=edges,
                    confidence_floor=confidence_floor,
                    masks=masks,
                    cost_multiplier=1.5,
                    stress_return_shock_pct=shock_pct,
                )

                tail_guard_passed = True
                tail_guard_reason = "insufficient_trades"
                if int(shock_sim.get("trades", 0)) >= _MIN_BASELINE_TRADES:
                    shock_pf = float(shock_sim.get("profit_factor", 0.0))
                    shock_dd = float(shock_sim.get("max_drawdown", 1.0))
                    tail_guard_passed = bool(shock_pf >= 0.90 and shock_dd <= 0.35)
                    tail_guard_reason = (
                        "ok" if tail_guard_passed else "tail_event_fragile"
                    )

                tail_block = {
                    "enabled": True,
                    "reason": "ok",
                    "tail_samples": int(tail_samples),
                    "quantile": float(_TAIL_STRESS_QUANTILE),
                    "threshold_abs_return_pct": float(threshold),
                    "tail_metrics": {
                        "profit_factor": float(tail_sim.get("profit_factor", 0.0)),
                        "max_drawdown": float(tail_sim.get("max_drawdown", 0.0)),
                        "excess_return": float(tail_sim.get("excess_return", 0.0)),
                        "trades": int(tail_sim.get("trades", 0)),
                    },
                    "shock_pct": float(shock_pct),
                    "shock_metrics": {
                        "profit_factor": float(shock_sim.get("profit_factor", 0.0)),
                        "max_drawdown": float(shock_sim.get("max_drawdown", 0.0)),
                        "excess_return": float(shock_sim.get("excess_return", 0.0)),
                        "trades": int(shock_sim.get("trades", 0)),
                    },
                }
            else:
                tail_block = {
                    "enabled": False,
                    "reason": "insufficient_tail_samples",
                    "tail_samples": int(tail_samples),
                    "quantile": float(_TAIL_STRESS_QUANTILE),
                }

        return {
            "cost_scenarios": cost_scenarios,
            "cost_resilience_passed": bool(cost_resilience_passed),
            "cost_resilience_reason": str(cost_resilience_reason),
            "tail_event": tail_block,
            "tail_guard_passed": bool(tail_guard_passed),
            "tail_guard_reason": str(tail_guard_reason),
        }

    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        r: np.ndarray,
        regime_profile: dict[str, Any] | None = None,
    ) -> dict:
        """Evaluate model on test data."""
        try:
            from sklearn.metrics import (
                confusion_matrix,
                precision_recall_fscore_support,
            )
            metrics_backend = "sklearn"
        except Exception as e:
            log.warning(
                "scikit-learn metrics unavailable (%s); "
                "falling back to numpy metrics in evaluation",
                e,
            )
            metrics_backend = "numpy"

            def confusion_matrix(y_true, y_pred, labels):  # type: ignore[redef]
                labels_arr = np.asarray(labels, dtype=int).reshape(-1)
                out = np.zeros((len(labels_arr), len(labels_arr)), dtype=np.int64)
                pos = {int(v): i for i, v in enumerate(labels_arr.tolist())}
                for t, p in zip(y_true, y_pred, strict=False):
                    ti = pos.get(int(t))
                    pi = pos.get(int(p))
                    if ti is None or pi is None:
                        continue
                    out[ti, pi] += 1
                return out

            def precision_recall_fscore_support(  # type: ignore[redef]
                y_true,
                y_pred,
                labels=None,
                average=None,
                zero_division=0,
            ):
                del average  # current caller uses labels=[2], average=None
                labels_arr = np.asarray(labels if labels is not None else [2], dtype=int)
                p_list: list[float] = []
                r_list: list[float] = []
                f_list: list[float] = []
                s_list: list[int] = []
                for lbl in labels_arr.tolist():
                    tp = int(np.sum((y_true == lbl) & (y_pred == lbl)))
                    fp = int(np.sum((y_true != lbl) & (y_pred == lbl)))
                    fn = int(np.sum((y_true == lbl) & (y_pred != lbl)))
                    support = int(np.sum(y_true == lbl))
                    precision = (
                        (tp / (tp + fp))
                        if (tp + fp) > 0
                        else float(zero_division)
                    )
                    recall = (
                        (tp / (tp + fn))
                        if (tp + fn) > 0
                        else float(zero_division)
                    )
                    denom = precision + recall
                    f1 = (
                        (2.0 * precision * recall / denom)
                        if denom > 0.0
                        else float(zero_division)
                    )
                    p_list.append(float(precision))
                    r_list.append(float(recall))
                    f_list.append(float(f1))
                    s_list.append(int(support))
                return (
                    np.asarray(p_list, dtype=np.float64),
                    np.asarray(r_list, dtype=np.float64),
                    np.asarray(f_list, dtype=np.float64),
                    np.asarray(s_list, dtype=np.int64),
                )

        empty_result = {
            "accuracy": 0.0,
            "trading": {},
            "stress_tests": {},
            "confusion_matrix": [],
            "up_precision": 0.0,
            "up_recall": 0.0,
            "up_f1": 0.0,
            "risk_adjusted_score": 0.0,
            "regime_profile": regime_profile or {},
            "explainability": {
                "samples": [],
                "filters": {},
            },
        }

        if len(X) == 0 or len(y) == 0:
            return empty_result

        predictions = self.ensemble.predict_batch(X)
        pred_classes = np.array(
            [p.predicted_class for p in predictions]
        )

        if len(pred_classes) == 0:
            return empty_result

        min_len = min(len(pred_classes), len(y), len(r))
        pred_classes = pred_classes[:min_len]
        y_eval = y[:min_len]
        r_eval = r[:min_len]

        cm = confusion_matrix(y_eval, pred_classes, labels=[0, 1, 2])

        # FIX EVAL: Use average='binary' style with safe extraction
        # labels=[2] with average=None returns arrays of length 1
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_eval,
            pred_classes,
            labels=[2],
            average=None,
            zero_division=0,
        )

        up_precision = float(pr[0]) if len(pr) > 0 else 0.0
        up_recall = float(rc[0]) if len(rc) > 0 else 0.0
        up_f1 = float(f1[0]) if len(f1) > 0 else 0.0

        confidences = np.array(
            [p.confidence for p in predictions[:min_len]]
        )
        agreements = np.array(
            [float(getattr(p, "agreement", 0.0)) for p in predictions[:min_len]]
        )
        entropies = np.array(
            [float(getattr(p, "entropy", 1.0)) for p in predictions[:min_len]]
        )
        margins = np.array(
            [float(getattr(p, "margin", 0.0)) for p in predictions[:min_len]]
        )
        prob_up = np.array(
            [float(getattr(p, "prob_up", 0.0)) for p in predictions[:min_len]]
        )
        prob_down = np.array(
            [float(getattr(p, "prob_down", 0.0)) for p in predictions[:min_len]]
        )
        edges = np.abs(prob_up - prob_down)

        accuracy = float(np.mean(pred_classes == y_eval))

        class_acc = {}
        for c in range(CONFIG.NUM_CLASSES):
            mask = y_eval == c
            if mask.sum() > 0:
                class_acc[c] = float(np.mean(pred_classes[mask] == c))
            else:
                class_acc[c] = 0.0

        confidence_floor = self._effective_confidence_floor(regime_profile)
        thresholds = self._trade_quality_thresholds(confidence_floor)
        masks = self._trade_masks(
            pred_classes,
            confidences,
            agreements,
            entropies,
            margins,
            edges,
            thresholds,
        )
        trading_metrics = self._simulate_trading(
            pred_classes,
            confidences,
            r_eval,
            agreements=agreements,
            entropies=entropies,
            margins=margins,
            edges=edges,
            confidence_floor=confidence_floor,
            masks=masks,
        )
        stress_tests = self._build_trading_stress_tests(
            preds=pred_classes,
            confs=confidences,
            returns=r_eval,
            agreements=agreements,
            entropies=entropies,
            margins=margins,
            edges=edges,
            confidence_floor=confidence_floor,
            masks=masks,
        )

        explainability_samples = self._build_explainability_samples(
            predictions=predictions,
            sample_count=min_len,
            thresholds=thresholds,
            masks=masks,
            limit=8,
        )

        risk_adjusted_score = self._risk_adjusted_score(
            {
                "accuracy": accuracy,
                "trading": trading_metrics,
            }
        )

        return {
            "accuracy": accuracy,
            "class_accuracy": class_acc,
            "mean_confidence": (
                float(np.mean(confidences))
                if len(confidences) > 0
                else 0.0
            ),
            "trading": trading_metrics,
            "stress_tests": stress_tests,
            "confusion_matrix": cm.tolist(),
            "up_precision": up_precision,
            "up_recall": up_recall,
            "up_f1": up_f1,
            "risk_adjusted_score": float(risk_adjusted_score),
            "regime_profile": regime_profile or {},
            "explainability": {
                "samples": explainability_samples,
                "filters": thresholds,
            },
            "metrics_backend": metrics_backend,
        }

    def _simulate_trading(
        self,
        preds: np.ndarray,
        confs: np.ndarray,
        returns: np.ndarray,
        agreements: np.ndarray | None = None,
        entropies: np.ndarray | None = None,
        margins: np.ndarray | None = None,
        edges: np.ndarray | None = None,
        confidence_floor: float | None = None,
        masks: dict[str, np.ndarray] | None = None,
        cost_multiplier: float = 1.0,
        stress_return_shock_pct: float = 0.0,
    ) -> dict:
        """
        Simulate trading with proper compounding and consistent units.

        FIX TRADE: Accumulates returns over the actual holding period
        instead of using a single returns[entry_idx] value.
        FIX COST: Stamp tax only on sells (China A-share rule).
        FIX m7: Use log-sum instead of np.prod to prevent overflow.
        PLUS: No-trade filtering for low-quality predictions.
        """
        thresholds = self._trade_quality_thresholds(confidence_floor)
        if masks is None:
            masks = self._trade_masks(
                preds=preds,
                confs=confs,
                agreements=agreements,
                entropies=entropies,
                margins=margins,
                edges=edges,
                thresholds=thresholds,
            )

        eligible_mask = np.asarray(masks.get("eligible", []), dtype=bool)
        up_mask = np.asarray(masks.get("is_up", []), dtype=bool)
        conf_mask = np.asarray(masks.get("conf", []), dtype=bool)
        agreement_mask = np.asarray(masks.get("agreement", []), dtype=bool)
        entropy_mask = np.asarray(masks.get("entropy", []), dtype=bool)
        margin_mask = np.asarray(masks.get("margin", []), dtype=bool)
        edge_mask = np.asarray(masks.get("edge", []), dtype=bool)

        position = eligible_mask.astype(float)

        horizon = self.prediction_horizon

        # FIX COST: Commission on both sides, stamp tax only on sell.
        cost_scale = float(max(0.0, cost_multiplier))
        entry_costs = cost_scale * (CONFIG.COMMISSION + CONFIG.SLIPPAGE)
        exit_costs = cost_scale * (
            CONFIG.COMMISSION + CONFIG.SLIPPAGE + CONFIG.STAMP_TAX
        )
        shock_decimal = float(max(0.0, stress_return_shock_pct)) / 100.0

        entries = np.diff(position, prepend=0) > 0
        exits = np.diff(position, prepend=0) < 0

        trades_decimal = []
        trade_confidences = []
        in_position = False
        entry_idx = 0

        for i in range(len(position)):
            if entries[i] and not in_position:
                in_position = True
                entry_idx = i
            elif (exits[i] or i == len(position) - 1) and in_position:
                # FIX TRADE: Accumulate returns over holding period
                # Each returns[j] is a percentage return for that period
                exit_idx = i
                holding_returns = returns[entry_idx:exit_idx + 1]

                if len(holding_returns) > 0:
                    # Convert percentage returns to factors, compound them
                    factors = 1.0 + holding_returns / 100.0
                    safe_factors = np.maximum(factors, _EPS)
                    cumulative = np.exp(np.sum(np.log(safe_factors)))
                    trade_return = cumulative - 1.0 - shock_decimal
                    trade_return -= entry_costs + exit_costs
                    trades_decimal.append(trade_return)
                    trade_confidences.append(
                        float(np.mean(confs[entry_idx:exit_idx + 1]))
                    )

                in_position = False

        num_trades = len(trades_decimal)
        avg_trade_confidence = (
            float(np.mean(trade_confidences))
            if trade_confidences
            else 0.0
        )

        if num_trades > 0:
            trades = np.array(trades_decimal)

            # FIX m7: Use log-sum to prevent overflow/underflow
            safe_factors = np.maximum(1 + trades, _EPS)
            total_return_factor = np.exp(np.sum(np.log(safe_factors)))
            total_return_pct = (total_return_factor - 1) * 100

            wins = trades[trades > 0]
            losses = trades[trades < 0]

            win_rate = len(wins) / num_trades

            gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
            gross_loss = (
                abs(np.sum(losses)) if len(losses) > 0 else _EPS
            )
            profit_factor = gross_profit / gross_loss

            if len(trades) > 1 and np.std(trades) > 0:
                avg_holding = max(horizon, 1)
                trades_per_year = 252 / avg_holding
                sharpe = (
                    np.mean(trades)
                    / np.std(trades)
                    * np.sqrt(trades_per_year)
                )
            else:
                sharpe = 0.0

            # FIX m7: Use log-sum for cumulative returns too
            log_returns = np.log(safe_factors)
            cumulative_log = np.cumsum(log_returns)
            cumulative = np.exp(cumulative_log)

            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / (running_max + _EPS)
            max_drawdown = (
                abs(float(np.min(drawdown)))
                if len(drawdown) > 0
                else 0.0
            )

        else:
            total_return_pct = 0.0
            win_rate = 0.0
            profit_factor = 0.0
            sharpe = 0.0
            max_drawdown = 0.0

        if len(returns) > 0 and horizon > 0:
            period_returns = returns / 100.0
            indices = list(
                range(0, len(period_returns), max(horizon, 1))
            )
            bh_returns = period_returns[indices]

            # FIX m7: Use log-sum for buy-hold too
            safe_bh = np.maximum(1 + bh_returns, _EPS)
            buyhold_factor = np.exp(np.sum(np.log(safe_bh)))
            buyhold_return_pct = (buyhold_factor - 1) * 100
        else:
            buyhold_return_pct = 0.0

        signal_count = int(len(preds))
        up_signals = int(np.sum(up_mask)) if len(up_mask) == signal_count else 0
        eligible_signals = (
            int(np.sum(eligible_mask)) if len(eligible_mask) == signal_count else 0
        )
        rejected_low_conf = (
            int(np.sum(up_mask & ~conf_mask))
            if len(conf_mask) == signal_count and up_signals > 0
            else 0
        )
        rejected_low_agreement = (
            int(np.sum(up_mask & ~agreement_mask))
            if len(agreement_mask) == signal_count and up_signals > 0
            else 0
        )
        rejected_high_entropy = (
            int(np.sum(up_mask & ~entropy_mask))
            if len(entropy_mask) == signal_count and up_signals > 0
            else 0
        )
        rejected_low_margin = (
            int(np.sum(up_mask & ~margin_mask))
            if len(margin_mask) == signal_count and up_signals > 0
            else 0
        )
        rejected_low_edge = (
            int(np.sum(up_mask & ~edge_mask))
            if len(edge_mask) == signal_count and up_signals > 0
            else 0
        )

        trade_coverage = float(eligible_signals / max(up_signals, 1))
        no_trade_rate = float(1.0 - (eligible_signals / max(signal_count, 1)))

        return {
            "total_return": float(total_return_pct),
            "buyhold_return": float(buyhold_return_pct),
            "excess_return": float(
                total_return_pct - buyhold_return_pct
            ),
            "trades": num_trades,
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "avg_trade_confidence": float(avg_trade_confidence),
            "trade_coverage": float(trade_coverage),
            "no_trade_rate": float(no_trade_rate),
            "signals_total": int(signal_count),
            "signals_up": int(up_signals),
            "signals_eligible": int(eligible_signals),
            "rejected_low_confidence": int(rejected_low_conf),
            "rejected_low_agreement": int(rejected_low_agreement),
            "rejected_high_entropy": int(rejected_high_entropy),
            "rejected_low_margin": int(rejected_low_margin),
            "rejected_low_edge": int(rejected_low_edge),
            "cost_multiplier": float(cost_scale),
            "stress_return_shock_pct": float(max(0.0, stress_return_shock_pct)),
            "filters": {
                "min_confidence": float(thresholds["min_confidence"]),
                "min_agreement": float(thresholds["min_agreement"]),
                "max_entropy": float(thresholds["max_entropy"]),
                "min_margin": float(thresholds["min_margin"]),
                "min_edge": float(thresholds["min_edge"]),
            },
        }

    # =========================================================================
    # =========================================================================

    def get_ensemble(self) -> EnsembleModel | None:
        """Get the trained ensemble model."""
        return self.ensemble

    def save_training_report(self, results: dict, path: str = None):
        """Save training report to file."""
        import json

        path = path or str(CONFIG.DATA_DIR / "training_report.json")

        def convert(obj):
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

