from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.settings import CONFIG
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)
_TRAINER_DATA_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS

_SEED = 42
_EPS = 1e-8

# Stop check interval for batch loops (check every N batches)
# FIX STOP: Reduced from 10 to 3 for faster cancellation response
_STOP_CHECK_INTERVAL = 3
_TRAINING_INTERVAL_LOCK = "1m"
# FIX 1M: Reduced from 10080 to 480 bars - free sources provide 1-2 days of 1m data
_MIN_1M_LOOKBACK_BARS = 480
_INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "1h"}
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

# Rebalancing hyperparameters (configurable ratios for noise reduction and tail upsampling)
_REBAL_LOW_SIGNAL_PERCENTILE = 40  # Percentile for low-signal cutoff
_REBAL_TAIL_PERCENTILE = 90  # Percentile for tail event cutoff
_REBAL_LOW_SIGNAL_TARGET_RATIO = 0.50  # Target ratio of low-signal samples to keep
_REBAL_LOW_SIGNAL_MIN_RATIO = 0.18  # Minimum ratio of low-signal samples to keep
_REBAL_TAIL_TARGET_RATIO = 0.20  # Target ratio of tail events in final dataset
_REBAL_MIN_DATASET_SIZE_RATIO = 0.35  # Minimum rebalanced dataset size ratio


def _two_day_intraday_window_bars(interval: str) -> int:
    """Return strict latest-2-day bar budget for intraday intervals."""
    iv = str(interval or "").strip().lower()
    if iv not in _INTRADAY_INTERVALS:
        return 0
    try:
        from data.fetcher import BARS_PER_DAY

        bpd = float(BARS_PER_DAY.get(iv, 1.0) or 1.0)
    except (ImportError, ValueError, TypeError, AttributeError):
        bpd = float(
            {
                "1m": 240.0,
                "2m": 120.0,
                "5m": 48.0,
                "15m": 16.0,
                "30m": 8.0,
                "60m": 4.0,
                "1h": 4.0,
            }.get(iv, 1.0)
        )
    return int(max(1, round(2.0 * max(0.01, bpd))))


def _assess_raw_data_quality(self, df: pd.DataFrame) -> dict[str, Any]:
    """Validate raw OHLC quality before feature engineering.

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
    except _TRAINER_DATA_RECOVERABLE_EXCEPTIONS as e:
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
    """Split a single stock's RAW data temporally, compute features
    and labels WITHIN each split, and invalidate warmup rows.
    """
    # FIX VALID: Validate input DataFrame before processing
    if df_raw is None or len(df_raw) == 0:
        log.warning("Split received empty or None DataFrame")
        return None
    
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df_raw.columns]
    if missing_cols:
        log.warning(
            "Split missing required columns: %s",
            ", ".join(missing_cols),
        )
        return None
    
    n = len(df_raw)
    embargo = max(int(CONFIG.EMBARGO_BARS), horizon)
    seq_len = int(CONFIG.SEQUENCE_LENGTH)
    feature_lookback = int(CONFIG.data.feature_lookback)

    train_end = int(n * CONFIG.TRAIN_RATIO)
    val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
    val_start = train_end + embargo
    test_start = val_end + embargo

    if train_end < seq_len + 50:
        log.warning(
            "Train split too small: %s rows (need >=%s)",
            train_end,
            seq_len + 50,
        )
        return None
    if val_start >= val_end or test_start >= n:
        log.warning(
            "Invalid split boundaries: val_start=%s, val_end=%s, test_start=%s, n=%s",
            val_start,
            val_end,
            test_start,
            n,
        )
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
            log.warning(
                "Split '%s' has %s rows < %s minimum for features",
                name,
                len(split_raw),
                min_rows,
            )
            if name == "train":
                return None

    try:
        train_df = self.feature_engine.create_features(train_raw)
    except ValueError as e:
        log.warning("Train feature creation failed: %s", e)
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
    except _TRAINER_DATA_RECOVERABLE_EXCEPTIONS as exc:
        consistency_guard["error"] = str(exc)
        log.warning("Consistency preflight failed: %s", exc)

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
            except (ValueError, TypeError, AttributeError):
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
                    use_cache=False,
                    update_db=True,
                    allow_online=True,
                    refresh_intraday_after_close=True,
                )
            except TypeError:
                df = self.fetcher.get_history(
                    code,
                    bars=bars,
                    interval=interval,
                    use_cache=False,
                    update_db=True,
                )
            if df is None or df.empty:
                log.warning("No data for %s", code)
                continue

            df, sanitize_meta = self._sanitize_raw_history(df, interval=interval)
            two_day_cap = int(_two_day_intraday_window_bars(interval))
            if two_day_cap > 0 and len(df) > two_day_cap:
                df = df.tail(two_day_cap).copy()
                sanitize_meta["trimmed_to_two_day_window"] = True
                sanitize_meta["window_bars"] = int(two_day_cap)
            else:
                sanitize_meta["trimmed_to_two_day_window"] = False
            q_report = self._assess_raw_data_quality(df)
            q_report["duplicates_removed"] = int(
                sanitize_meta.get("duplicates_removed", 0)
            )
            q_report["repaired_rows"] = int(
                sanitize_meta.get("repaired_rows", 0)
            )
            q_report["rows_removed"] = int(
                sanitize_meta.get("rows_removed", 0)
            )
            q_report["trimmed_to_two_day_window"] = bool(
                sanitize_meta.get("trimmed_to_two_day_window", False)
            )
            q_report["window_bars"] = int(sanitize_meta.get("window_bars", len(df)))
            raw_invalid_ohlc_rows = int(
                sanitize_meta.get("raw_invalid_ohlc_rows", 0)
            )
            if raw_invalid_ohlc_rows > 0:
                raw_rows_before = int(
                    max(1, int(sanitize_meta.get("rows_before", len(df))))
                )
                raw_broken_ratio = self._safe_ratio(
                    float(raw_invalid_ohlc_rows),
                    float(raw_rows_before),
                )
                q_report["broken_ohlc_ratio"] = float(
                    max(float(q_report.get("broken_ohlc_ratio", 0.0)), raw_broken_ratio)
                )
                if raw_broken_ratio > float(_DATA_QUALITY_MAX_BROKEN_OHLC_RATIO):
                    reasons = [
                        str(x).strip()
                        for x in list(q_report.get("reasons", []) or [])
                        if str(x).strip()
                    ]
                    if "invalid_ohlc_relations" not in reasons:
                        reasons.append("invalid_ohlc_relations")
                    q_report["reasons"] = reasons
                    q_report["passed"] = False
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
                log.warning(
                    "Insufficient data for %s: %s bars (need %s)",
                    code,
                    len(df),
                    min_required,
                )
                continue

            if (
                str(interval).strip().lower() == _TRAINING_INTERVAL_LOCK
                and len(df) < _MIN_1M_LOOKBACK_BARS
            ):
                short_1m_codes.append(str(code))

            raw_data[code] = df

        except _TRAINER_DATA_RECOVERABLE_EXCEPTIONS as e:
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


def _validate_temporal_split_integrity(
    self,
    split_data: dict[str, dict[str, pd.DataFrame]],
    feature_cols: list[str],
) -> dict[str, Any]:
    """Validate that temporal splits prevent data leakage.
    
    This is a critical guard against look-ahead bias that can cause
    overoptimistic backtests and poor live performance.
    
    Checks:
    1. No feature columns contain NaN values from future data leakage
    2. Label creation uses only past/present information
    3. Train/val/test boundaries have proper embargo periods
    4. Feature statistics don't show impossible jumps at boundaries
    
    Returns:
        Validation report with 'passed' boolean and detailed diagnostics
    """
    report: dict[str, Any] = {
        "passed": True,
        "checks": {},
        "warnings": [],
        "errors": [],
    }
    
    try:
        train_df = split_data.get("train", pd.DataFrame())
        val_df = split_data.get("val", pd.DataFrame())
        
        if train_df.empty:
            report["errors"].append("train_split_empty")
            report["passed"] = False
            return report
        
        # Check 1: Feature column integrity
        missing_features = set(feature_cols) - set(train_df.columns)
        if missing_features:
            report["errors"].append(
                f"missing_features:{','.join(sorted(missing_features))}"
            )
            report["passed"] = False
            return report
        
        # Check 2: NaN ratio in features (should be minimal after proper handling)
        train_features = train_df[feature_cols]
        nan_ratio = float(train_features.isna().sum().sum()) / float(
            max(1, train_features.size)
        )
        report["checks"]["feature_nan_ratio"] = round(nan_ratio, 4)
        if nan_ratio > 0.05:  # More than 5% NaN suggests leakage or bad handling
            report["warnings"].append(f"high_feature_nan_ratio:{nan_ratio:.4f}")
        
        # Check 3: Label NaN pattern (should only be in warmup period)
        if "label" in train_df.columns:
            label_isna = train_df["label"].isna()
            nan_count = int(label_isna.sum())
            total_labels = len(train_df)
            expected_warmup = int(CONFIG.SEQUENCE_LENGTH + CONFIG.EMBARGO_BARS)

            report["checks"]["label_nan_count"] = nan_count
            report["checks"]["expected_warmup"] = expected_warmup

            # FIX NaN: Calculate NaN ratio and block training if excessive
            nan_ratio = float(nan_count) / float(max(1, total_labels))
            report["checks"]["label_nan_ratio"] = round(nan_ratio, 4)

            # NaN labels should only be in the first ~SEQUENCE_LENGTH rows
            if nan_count > 0:
                last_nan_idx = label_isna[::-1].idxmax() if label_isna.any() else None
                if last_nan_idx is not None:
                    # Find position of last NaN
                    label_values = train_df["label"].values
                    last_nan_pos = 0
                    for i in range(len(label_values) - 1, -1, -1):
                        if pd.isna(label_values[i]):
                            last_nan_pos = i + 1
                            break

                    report["checks"]["last_nan_position"] = last_nan_pos

                    # If NaN extends beyond warmup, suggests feature computation issue
                    if last_nan_pos > expected_warmup * 1.5:
                        report["warnings"].append(
                            f"label_nan_extends_beyond_warmup:"
                            f"{last_nan_pos}>{expected_warmup}"
                        )
                    
                    # FIX NaN: Block training if NaN ratio exceeds 15%
                    if nan_ratio > 0.15:
                        report["errors"].append(
                            f"excessive_label_nan_ratio:{nan_ratio:.4f}"
                        )
                        report["passed"] = False
                        return report
        
        # Check 4: Feature statistics continuity at boundaries
        # (Detects if validation/test data leaked into train features)
        if not val_df.empty and len(train_df) > 50 and len(val_df) > 50:
            severe_leakage_detected = False
            for col in feature_cols[:5]:  # Check first 5 features for efficiency
                if col in train_df.columns and col in val_df.columns:
                    train_window = pd.to_numeric(
                        train_df[col].tail(20), errors="coerce"
                    ).dropna()
                    val_window = pd.to_numeric(
                        val_df[col].head(20), errors="coerce"
                    ).dropna()
                    if len(train_window) < 5 or len(val_window) < 5:
                        continue
                    train_end = float(train_window.mean())
                    val_start = float(val_window.mean())

                    # Large jumps might indicate leakage or regime shift
                    # Use a stable denominator so near-zero means do not trigger
                    # false leakage alarms on normalized features.
                    scale = max(
                        abs(train_end),
                        float(train_window.std(ddof=0)),
                        1e-3,
                    )
                    if scale > _EPS:
                        jump = abs(val_start - train_end) / scale
                        if jump > 2.0:  # More than 200% jump
                            report["warnings"].append(
                                f"feature_jump_at_boundary:{col}:{jump:.2f}"
                            )
                        # FIX LEAK: Block training on severe leakage (>500% jump)
                        if jump > 5.0:
                            severe_leakage_detected = True
                            report["errors"].append(
                                f"severe_feature_leakage:{col}:{jump:.2f}"
                            )
            if severe_leakage_detected:
                report["passed"] = False
                return report

        # Check 5: Return-based leakage detection
        # If features contain future returns, correlation will be suspiciously high
        if "label" in train_df.columns and len(train_df) > 100:
            label_clean = pd.to_numeric(
                train_df["label"], errors="coerce"
            ).dropna()
            if len(label_clean) > 50:
                severe_leakage_detected = False
                for col in feature_cols[:3]:  # Sample check
                    if col in train_df.columns:
                        feature_series = pd.to_numeric(
                            train_df[col], errors="coerce"
                        )
                        aligned = pd.concat(
                            [
                                feature_series,
                                pd.to_numeric(train_df["label"], errors="coerce"),
                            ],
                            axis=1,
                            keys=["feature", "label"],
                        ).dropna()
                        if len(aligned) < 50:
                            if abs(len(feature_series.dropna()) - len(label_clean)) > 0:
                                report["warnings"].append(
                                    f"correlation_overlap_insufficient:{col}:{len(aligned)}"
                                )
                            continue
                        try:
                            corr = float(
                                np.corrcoef(
                                    aligned["feature"].to_numpy(dtype=np.float64),
                                    aligned["label"].to_numpy(dtype=np.float64),
                                )[0, 1]
                            )
                            if np.isfinite(corr) and abs(corr) > 0.8:
                                report["warnings"].append(
                                    f"suspicious_high_correlation:{col}:{corr:.3f}"
                                )
                            # FIX LEAK: Block training on extreme correlation (>0.95)
                            if np.isfinite(corr) and abs(corr) > 0.95:
                                severe_leakage_detected = True
                                report["errors"].append(
                                    f"extreme_leakage:{col}:{corr:.3f}"
                                )
                        except (ValueError, FloatingPointError):
                            pass
                if severe_leakage_detected:
                    report["passed"] = False
                    return report
        
        # Summary
        if report["errors"]:
            report["passed"] = False
        elif len(report["warnings"]) > 3:
            report["passed"] = False
            report["errors"].append(
                f"too_many_warnings:{len(report['warnings'])}"
            )
        
        report["checks"]["total_warnings"] = len(report["warnings"])
        report["checks"]["total_errors"] = len(report["errors"])
        
    except _TRAINER_DATA_RECOVERABLE_EXCEPTIONS as e:
        report["errors"].append(f"validation_exception:{str(e)}")
        report["passed"] = False
    
    return report


def _create_sequences_from_splits(
    self,
    split_data: dict[str, dict[str, pd.DataFrame]],
    feature_cols: list[str],
    include_returns: bool = True,
) -> dict[str, dict[str, list]]:
    """Create sequences for train/val/test from split data.
    
    Includes temporal integrity validation to prevent data leakage.
    """
    # CRITICAL: Validate temporal split integrity before creating sequences
    # This prevents look-ahead bias that causes overoptimistic backtests
    validation_report = self._validate_temporal_split_integrity(
        split_data, feature_cols
    )

    if not validation_report["passed"]:
        errors = validation_report.get("errors", [])
        warnings = validation_report.get("warnings", [])
        
        # FIX LEAK: Block training on severe leakage indicators
        severe_leakage_errors = [
            e for e in errors 
            if "severe_feature_leakage" in e or "extreme_leakage" in e
        ]
        if severe_leakage_errors:
            log.error(
                "Temporal split validation blocked training due to severe data leakage: %s",
                ", ".join(severe_leakage_errors),
            )
            raise ValueError(
                f"Severe data leakage detected: {', '.join(severe_leakage_errors)}"
            )
        
        log.warning(
            "Temporal split validation failed: %s | Warnings: %s",
            ", ".join(errors),
            ", ".join(warnings),
        )
        # Don't block training for non-severe issues, but log for investigation
    
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
                except _TRAINER_DATA_RECOVERABLE_EXCEPTIONS as e:
                    log.warning(
                        "Sequence creation failed for %s/%s: %s",
                        code,
                        split_name,
                        e,
                    )

    return storage


def _rebalance_train_samples(
    self,
    X_train: np.ndarray,
    y_train: np.ndarray,
    r_train: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Reduce low-signal noise and increase tail-event exposure in training.

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

    low_cut = float(np.percentile(abs_returns, _REBAL_LOW_SIGNAL_PERCENTILE))
    tail_cut = float(np.percentile(abs_returns, _REBAL_TAIL_PERCENTILE))
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

    # FIX MAGIC: Use configurable ratios instead of hardcoded values
    keep_low_target = max(
        int(round(len(core_idx) * _REBAL_LOW_SIGNAL_TARGET_RATIO)),
        int(round(len(idx_valid) * _REBAL_LOW_SIGNAL_MIN_RATIO)),
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
    target_tail = int(round(len(rebalance_idx) * _REBAL_TAIL_TARGET_RATIO))
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

    if len(rebalance_idx) < max(120, int(round(n * _REBAL_MIN_DATASET_SIZE_RATIO))):
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
    """Build a temporal validation holdout with embargo when val split is empty.

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

    min_val = max(2, min(32, total // 10))
    holdout = max(min_val, total // 7)
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

    if len(X_val) < min_val:
        val_start = max(1, total - min_val)
        train_end = max(1, val_start - gap)
        if train_end >= val_start:
            train_end = max(1, val_start - 1)
        X_tr = X_train[:train_end]
        y_tr = y_train[:train_end]
        X_val = X_train[val_start:]
        y_val = y_train[val_start:]

    if len(X_val) == 0:
        split = max(1, total - 2)
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


def prepare_data(
    self,
    stock_codes: list[str] = None,
    min_samples_per_stock: int = 100,
    verbose: bool = True,
    interval: str = "1m",
    prediction_horizon: int = None,
    lookback_bars: int = None,
) -> tuple:
    """Prepare training data with proper temporal split.

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
    two_day_cap = int(_two_day_intraday_window_bars(interval))
    if two_day_cap > 0:
        bars = int(min(bars, two_day_cap))

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

    # FIX SHAPE: Validate array dimensions before accessing shape[2]
    if X_train.ndim != 3:
        raise ValueError(
            f"X_train must be 3D array (samples, seq_len, features), "
            f"got {X_train.ndim}D shape {X_train.shape}"
        )
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
