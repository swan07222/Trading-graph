# models/trainer.py
from __future__ import annotations

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
    from utils.atomic_io import atomic_torch_save
except ImportError:
    atomic_torch_save = None

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

        self.interval: str = "1d"
        self.prediction_horizon: int = CONFIG.PREDICTION_HORIZON

        # Incremental training flag — set externally by auto_learner.
        self._skip_scaler_fit: bool = False

    # =========================================================================
    # =========================================================================

    @staticmethod
    def _should_stop(stop_flag: Any) -> bool:
        """Check if training should stop — handles multiple stop flag types."""
        if stop_flag is None:
            return False

        is_cancelled = getattr(stop_flag, "is_cancelled", None)
        if is_cancelled is not None:
            try:
                return bool(
                    is_cancelled() if callable(is_cancelled) else is_cancelled
                )
            except Exception:
                return False

        if callable(stop_flag):
            try:
                return bool(stop_flag())
            except Exception:
                return False

        return False

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
        iterator = tqdm(stocks, desc="Loading stocks") if verbose else stocks

        for code in iterator:
            if self._should_stop(stop_flag):
                log.info("Data fetch stopped by user")
                break

            try:
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

                min_required = CONFIG.SEQUENCE_LENGTH + 80
                if len(df) < min_required:
                    log.debug(
                        f"Insufficient data for {code}: "
                        f"{len(df)} bars (need {min_required})"
                    )
                    continue

                raw_data[code] = df

            except Exception as e:
                log.warning(f"Error fetching {code}: {e}")

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

        should_skip = self._skip_scaler_fit and self.processor.is_fitted

        if should_skip:
            log.info(
                f"Skipping scaler refit (incremental mode, "
                f"existing scaler has {self.processor.n_features} features)"
            )
        else:
            if self._skip_scaler_fit and not self.processor.is_fitted:
                log.warning(
                    "_skip_scaler_fit=True but no prior scaler exists — "
                    "fitting new scaler"
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

    # =========================================================================
    # prepare_data (standalone, used by external callers)
    # =========================================================================

    def prepare_data(
        self,
        stock_codes: list[str] = None,
        min_samples_per_stock: int = 100,
        verbose: bool = True,
        interval: str = "1d",
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
        interval = str(interval).lower()
        horizon = int(prediction_horizon or CONFIG.PREDICTION_HORIZON)
        bars = int(lookback_bars or 2000)

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
        interval: str = "1d",
        prediction_horizon: int = None,
        lookback_bars: int = 2400,
        learning_rate: float = None,
    ) -> dict:
        """
        Train complete pipeline:
        1) Classification ensemble for trading signals
        2) Multi-step forecaster for AI-generated price curves
        """

        epochs = int(epochs or CONFIG.EPOCHS)
        batch_size = int(batch_size or CONFIG.BATCH_SIZE)
        interval = str(interval).lower()
        horizon = int(prediction_horizon or CONFIG.PREDICTION_HORIZON)
        lookback = int(lookback_bars)

        self.interval = interval
        self.prediction_horizon = horizon

        effective_lr = _resolve_learning_rate(learning_rate)

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

        log.info(
            f"Training data: {len(X_train)} samples, "
            f"{self.input_size} features"
        )

        scaler_path = CONFIG.MODEL_DIR / f"scaler_{interval}_{horizon}.pkl"
        self.processor.save_scaler(str(scaler_path))

        # --- Phase 4: Train classifier ensemble ---
        if incremental:
            ensemble_path = (
                CONFIG.MODEL_DIR / f"ensemble_{interval}_{horizon}.pt"
            )
            if ensemble_path.exists():
                temp_ensemble = EnsembleModel(
                    input_size=self.input_size,
                    model_names=model_names,
                )
                if temp_ensemble.load(str(ensemble_path)):
                    self.ensemble = temp_ensemble
                    log.info(
                        "Loaded existing ensemble for incremental training"
                    )
                else:
                    log.warning(
                        "Failed to load existing ensemble — "
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

        if X_val is None or len(X_val) == 0:
            split_idx = int(len(X_train) * 0.85)
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
            r_train = r_train[:split_idx] if r_train is not None else None

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

        if save_model:
            ensemble_path = (
                CONFIG.MODEL_DIR / f"ensemble_{interval}_{horizon}.pt"
            )
            self.ensemble.save(str(ensemble_path))
            log.info(f"Ensemble saved: {ensemble_path}")

        # --- Phase 5: Train forecaster ---
        # FIX CANCEL2: _train_forecaster now re-raises CancelledException
        forecaster_trained = False
        try:
            forecaster_trained = self._train_forecaster(
                split_data, feature_cols, horizon, interval,
                batch_size, epochs, stop_flag, save_model,
                effective_lr,
            )
        except CancelledException:
            log.info("Training cancelled during forecaster phase")
            return {"status": "cancelled", "history": history}

        # --- Phase 6: Evaluate on test set ---
        test_metrics = {}
        if X_test is not None and len(X_test) > 0 and r_test is not None:
            test_metrics = self._evaluate(
                X_test[:2000], y_test[:2000], r_test[:2000]
            )
            log.info(
                f"Test accuracy: {test_metrics.get('accuracy', 0):.2%}"
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
    ) -> bool:
        """
        Train multi-step forecaster. Returns True if successful.

        FIX CANCEL2: Re-raises CancelledException instead of swallowing it,
        so the caller (train()) knows to return cancelled status.
        """
        from models.networks import TCNModel

        Xf_train_list, Yf_train_list = [], []
        Xf_val_list, Yf_val_list = [], []

        for code, splits in split_data.items():
            if self._should_stop(stop_flag):
                break

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
                    break

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
                    break

                forecaster.eval()
                val_losses = []

                with torch.inference_mode():
                    for xb, yb in val_loader:
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

                if save_model:
                    forecast_path = (
                        CONFIG.MODEL_DIR
                        / f"forecast_{interval}_{horizon}.pt"
                    )
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
                        atomic_torch_save(forecast_path, payload)
                    else:
                        torch.save(payload, forecast_path)

                    log.info(f"Forecaster saved: {forecast_path}")

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

    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        r: np.ndarray,
    ) -> dict:
        """Evaluate model on test data."""
        from sklearn.metrics import (
            confusion_matrix,
            precision_recall_fscore_support,
        )

        if len(X) == 0 or len(y) == 0:
            return {
                "accuracy": 0.0,
                "trading": {},
                "confusion_matrix": [],
                "up_precision": 0.0,
                "up_recall": 0.0,
                "up_f1": 0.0,
            }

        predictions = self.ensemble.predict_batch(X)
        pred_classes = np.array(
            [p.predicted_class for p in predictions]
        )

        if len(pred_classes) == 0:
            return {
                "accuracy": 0.0,
                "trading": {},
                "confusion_matrix": [],
                "up_precision": 0.0,
                "up_recall": 0.0,
                "up_f1": 0.0,
            }

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
        accuracy = float(np.mean(pred_classes == y_eval))

        class_acc = {}
        for c in range(CONFIG.NUM_CLASSES):
            mask = y_eval == c
            if mask.sum() > 0:
                class_acc[c] = float(np.mean(pred_classes[mask] == c))
            else:
                class_acc[c] = 0.0

        trading_metrics = self._simulate_trading(
            pred_classes, confidences, r_eval
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
            "confusion_matrix": cm.tolist(),
            "up_precision": up_precision,
            "up_recall": up_recall,
            "up_f1": up_f1,
        }

    def _simulate_trading(
        self,
        preds: np.ndarray,
        confs: np.ndarray,
        returns: np.ndarray,
    ) -> dict:
        """
        Simulate trading with proper compounding and consistent units.

        FIX TRADE: Accumulates returns over the actual holding period
        instead of using a single returns[entry_idx] value.
        FIX COST: Stamp tax only on sells (China A-share rule).
        FIX m7: Use log-sum instead of np.prod to prevent overflow.
        """
        confidence_mask = confs >= CONFIG.MIN_CONFIDENCE

        position = np.zeros_like(preds, dtype=float)
        position[preds == 2] = 1
        position = position * confidence_mask

        horizon = self.prediction_horizon

        # FIX COST: Commission on both sides, stamp tax only on sell,
        entry_costs = CONFIG.COMMISSION + CONFIG.SLIPPAGE
        exit_costs = CONFIG.COMMISSION + CONFIG.SLIPPAGE + CONFIG.STAMP_TAX

        entries = np.diff(position, prepend=0) > 0
        exits = np.diff(position, prepend=0) < 0

        trades_decimal = []
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
                    trade_return = cumulative - 1.0 - entry_costs - exit_costs
                    trades_decimal.append(trade_return)

                in_position = False

        num_trades = len(trades_decimal)

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
