# models/trainer.py
"""
Model Trainer - Complete Training Pipeline

FIXES APPLIED:
- Issue 1:  Features computed WITHIN each split (no cross-split leakage)
- Issue 2:  Warmup uses CONFIG.data.feature_lookback (not seq_len)
- Issue 5:  _skip_scaler_fit is a documented public attribute
- Issue 13: min_periods=1 instability handled by warmup invalidation
- Issue 19: interval/horizon set on ensemble before save
- Proper temporal split per stock (no data leakage)
- Scaler fitted only on training data
- Labels created WITHIN each split
- Scaler saved with model for inference
- Proper interval/horizon parameter handling
- Robust forecaster training with error handling
- Proper embargo gap between splits
- best_accuracy fallback to test_acc
"""
import numpy as np
import torch
import random
from typing import Dict, List, Optional, Callable, Tuple, Any
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from config.settings import CONFIG
from data.fetcher import DataFetcher, get_fetcher
from data.processor import DataProcessor
from data.features import FeatureEngine
from models.ensemble import EnsembleModel
from utils.logger import get_logger
from utils.cancellation import CancelledException
from __future__ import annotations
import pandas as pd


log = get_logger(__name__)

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    Complete training pipeline with proper data handling.

    CRITICAL: Features are computed and labels are created WITHIN each
    temporal split to prevent leakage.

    Supports:
    - Classification ensemble for trading signals
    - Multi-step forecaster for price curve prediction
    - Multiple intervals (1m, 5m, 15m, 1d, etc.)
    - Incremental training with _skip_scaler_fit
    """

    def __init__(self):
        self.fetcher = get_fetcher()
        self.processor = DataProcessor()
        self.feature_engine = FeatureEngine()

        self.ensemble: Optional[EnsembleModel] = None
        self.history: Dict = {}
        self.input_size: int = 0

        # Training metadata
        self.interval: str = "1d"
        self.prediction_horizon: int = CONFIG.PREDICTION_HORIZON

        # Incremental training flag — set externally by auto_learner.
        # When True AND the processor already has a fitted scaler,
        # prepare_data / train will NOT refit the scaler.
        self._skip_scaler_fit: bool = False

    def _should_stop(self, stop_flag: Any) -> bool:
        """Check if training should stop - handles multiple stop flag types"""
        if stop_flag is None:
            return False

        # Handle CancellationToken
        is_cancelled = getattr(stop_flag, 'is_cancelled', None)
        if is_cancelled is not None:
            if callable(is_cancelled):
                try:
                    return bool(is_cancelled())
                except Exception:
                    pass
            return bool(is_cancelled)

        # Handle callable
        if callable(stop_flag):
            try:
                return bool(stop_flag())
            except Exception:
                pass

        return False

    # =========================================================================
    # CORE SPLIT LOGIC (shared by prepare_data and train)
    # =========================================================================

    def _split_single_stock(
        self,
        df_raw: pd.DataFrame,
        horizon: int,
        feature_cols: List[str],
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Split a single stock's RAW data temporally, compute features
        and labels WITHIN each split, and invalidate warmup rows.

        Returns dict with keys 'train', 'val', 'test' (DataFrames with
        features + labels) or None if the stock has insufficient data.

        FIX Issue 1 + Issue 2:
        -  Raw OHLCV is split FIRST.
        -  feature_engine.create_features() is called per split so that
           rolling windows never cross the boundary.
        -  Extra ``feature_lookback`` rows are prepended to val/test so
           indicators can warm up; those rows are then label-invalidated.
        """
        n = len(df_raw)
        embargo = max(int(CONFIG.EMBARGO_BARS), horizon)
        seq_len = int(CONFIG.SEQUENCE_LENGTH)
        feature_lookback = int(CONFIG.data.feature_lookback)

        # ---- temporal boundaries ----
        train_end = int(n * CONFIG.TRAIN_RATIO)
        val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
        val_start = train_end + embargo
        test_start = val_end + embargo

        if train_end < seq_len + 50:
            return None
        if val_start >= val_end or test_start >= n:
            return None

        # ---- slice RAW data, with lookback prefix for val/test ----
        train_raw = df_raw.iloc[:train_end].copy()

        val_raw_begin = max(0, val_start - feature_lookback)
        val_raw = df_raw.iloc[val_raw_begin:val_end].copy()

        test_raw_begin = max(0, test_start - feature_lookback)
        test_raw = df_raw.iloc[test_raw_begin:].copy()

        # ---- compute features WITHIN each split ----
        train_df = self.feature_engine.create_features(train_raw)
        val_df = self.feature_engine.create_features(val_raw)
        test_df = self.feature_engine.create_features(test_raw)

        # Verify all feature columns exist
        for split_name, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            missing = set(feature_cols) - set(split_df.columns)
            if missing:
                log.debug(
                    f"Missing features in {split_name} split: {missing}"
                )
                return None

        # ---- create labels WITHIN each split ----
        train_df = self.processor.create_labels(
            train_df, horizon=horizon
        )
        val_df = self.processor.create_labels(
            val_df, horizon=horizon
        )
        test_df = self.processor.create_labels(
            test_df, horizon=horizon
        )

        # ---- invalidate warmup rows (Issue 2: use feature_lookback) ----
        warmup_val = val_start - val_raw_begin
        warmup_test = test_start - test_raw_begin

        if warmup_val > 0 and 'label' in val_df.columns:
            val_df.iloc[
                :warmup_val, val_df.columns.get_loc('label')
            ] = np.nan
        if warmup_test > 0 and 'label' in test_df.columns:
            test_df.iloc[
                :warmup_test, test_df.columns.get_loc('label')
            ] = np.nan

        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
        }

    # =========================================================================
    # prepare_data (standalone, used by external callers)
    # =========================================================================

    def prepare_data(
        self,
        stock_codes: List[str] = None,
        min_samples_per_stock: int = 100,
        verbose: bool = True,
        interval: str = "1d",
        prediction_horizon: int = None,
        lookback_bars: int = None,
    ) -> Tuple:
        """
        Prepare training data with proper temporal split.

        CRITICAL: Each stock's raw data is split temporally BEFORE
        features and labels are computed inside each split.

        Returns:
            Tuple of (X_train, y_train, r_train,
                       X_val,   y_val,   r_val,
                       X_test,  y_test,  r_test)
        """
        import pandas as pd  # local to avoid top-level issues

        stocks = stock_codes or CONFIG.STOCK_POOL
        interval = str(interval).lower()
        horizon = int(prediction_horizon or CONFIG.PREDICTION_HORIZON)
        bars = int(lookback_bars or 2000)

        self.interval = interval
        self.prediction_horizon = horizon

        log.info(f"Preparing data for {len(stocks)} stocks...")
        log.info(
            f"Interval: {interval}, Horizon: {horizon}, "
            f"Lookback: {bars}"
        )
        log.info(
            f"Temporal split: Train={CONFIG.TRAIN_RATIO:.0%}, "
            f"Val={CONFIG.VAL_RATIO:.0%}, "
            f"Test={CONFIG.TEST_RATIO:.0%}"
        )

        feature_cols = self.feature_engine.get_feature_columns()

        # Phase 1: Fetch RAW data (no features yet)
        raw_data: Dict[str, pd.DataFrame] = {}
        iterator = (
            tqdm(stocks, desc="Loading stocks") if verbose else stocks
        )

        for code in iterator:
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

                min_required = (
                    CONFIG.SEQUENCE_LENGTH + min_samples_per_stock
                )
                if len(df) < min_required:
                    log.debug(
                        f"Insufficient data for {code}: "
                        f"{len(df)} bars (need {min_required})"
                    )
                    continue

                raw_data[code] = df

            except Exception as e:
                log.warning(f"Error processing {code}: {e}")

        if not raw_data:
            raise ValueError(
                "No valid stock data available for training"
            )

        log.info(f"Successfully loaded {len(raw_data)} stocks")

        # Phase 2: Split each stock, compute features per split
        all_train_features = []
        split_data: Dict[str, Dict[str, pd.DataFrame]] = {}

        for code, df_raw in raw_data.items():
            splits = self._split_single_stock(
                df_raw, horizon, feature_cols
            )
            if splits is None:
                log.debug(
                    f"Insufficient split data for {code}"
                )
                continue

            split_data[code] = splits

            # Collect training features for scaler fitting
            train_df = splits['train']
            train_features = train_df[feature_cols].values
            valid_mask = ~train_df['label'].isna()
            if valid_mask.sum() > 0:
                all_train_features.append(
                    train_features[valid_mask]
                )

        if not all_train_features:
            raise ValueError(
                "No valid training data after split"
            )

        # Phase 3: Fit scaler on training data ONLY
        if self._skip_scaler_fit and self.processor.is_fitted:
            log.info(
                f"Skipping scaler refit (incremental mode, "
                f"existing scaler has "
                f"{self.processor.n_features} features)"
            )
        else:
            log.info("Fitting scaler on training data...")
            combined_train_features = np.concatenate(
                all_train_features, axis=0
            )
            self.processor.fit_scaler(combined_train_features)
            log.info(
                f"Scaler fitted on "
                f"{len(combined_train_features)} training samples"
            )

        # Phase 4: Create sequences for each split
        all_train = {'X': [], 'y': [], 'r': []}
        all_val = {'X': [], 'y': [], 'r': []}
        all_test = {'X': [], 'y': [], 'r': []}

        for code, splits in split_data.items():
            for split_name, split_df, storage in [
                ('train', splits['train'], all_train),
                ('val', splits['val'], all_val),
                ('test', splits['test'], all_test),
            ]:
                if len(split_df) >= CONFIG.SEQUENCE_LENGTH + 5:
                    X, y, r = self.processor.prepare_sequences(
                        split_df,
                        feature_cols,
                        fit_scaler=False,
                    )
                    if len(X) > 0:
                        storage['X'].append(X)
                        storage['y'].append(y)
                        storage['r'].append(r)

        # Phase 5: Combine all stocks
        def combine_arrays(
            storage: Dict,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if not storage['X']:
                return np.array([]), np.array([]), np.array([])
            return (
                np.concatenate(storage['X']),
                np.concatenate(storage['y']),
                np.concatenate(storage['r']),
            )

        X_train, y_train, r_train = combine_arrays(all_train)
        X_val, y_val, r_val = combine_arrays(all_val)
        X_test, y_test, r_test = combine_arrays(all_test)

        self.input_size = (
            X_train.shape[2] if len(X_train) > 0 else 0
        )

        log.info("Data prepared:")
        log.info(f"  Train: {len(X_train)} samples")
        log.info(f"  Val:   {len(X_val)} samples")
        log.info(f"  Test:  {len(X_test)} samples")
        log.info(f"  Input size: {self.input_size} features")

        if len(y_train) > 0:
            dist = self.processor.get_class_distribution(y_train)
            log.info(
                f"  Class distribution: DOWN={dist['DOWN']}, "
                f"NEUTRAL={dist['NEUTRAL']}, UP={dist['UP']}"
            )

        # Save scaler
        scaler_path = (
            CONFIG.MODEL_DIR / f"scaler_{interval}_{horizon}.pkl"
        )
        self.processor.save_scaler(str(scaler_path))

        return (
            X_train, y_train, r_train,
            X_val, y_val, r_val,
            X_test, y_test, r_test,
        )

    # =========================================================================
    # MAIN train() METHOD
    # =========================================================================

    def train(
        self,
        stock_codes: List[str] = None,
        epochs: int = None,
        batch_size: int = None,
        model_names: List[str] = None,
        callback: Callable = None,
        stop_flag: Any = None,
        save_model: bool = True,
        incremental: bool = False,
        interval: str = "1d",
        prediction_horizon: int = None,
        lookback_bars: int = 2400,
    ) -> Dict:
        """
        Train complete pipeline:
        1) Classification ensemble for trading signals
        2) Multi-step forecaster for AI-generated price curves

        Returns:
            Dictionary with training results and metrics
        """
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from models.networks import TCNModel
        import pandas as pd

        # Validate and set parameters
        epochs = int(epochs or CONFIG.EPOCHS)
        batch_size = int(batch_size or CONFIG.BATCH_SIZE)
        interval = str(interval).lower()
        horizon = int(
            prediction_horizon or CONFIG.PREDICTION_HORIZON
        )
        lookback = int(lookback_bars)

        self.interval = interval
        self.prediction_horizon = horizon

        log.info("=" * 70)
        log.info(
            "STARTING TRAINING PIPELINE (Classifier + Forecaster)"
        )
        log.info(
            f"Interval: {interval}, Horizon: {horizon} bars, "
            f"Lookback: {lookback}"
        )
        log.info(
            f"Incremental: {incremental}, "
            f"Skip scaler fit: {self._skip_scaler_fit}"
        )
        log.info("=" * 70)

        # --- Phase 1: Fetch RAW data ---
        stocks = stock_codes or CONFIG.STOCK_POOL
        feature_cols = self.feature_engine.get_feature_columns()

        raw_data: Dict[str, pd.DataFrame] = {}
        for code in tqdm(stocks, desc="Fetching data"):
            if self._should_stop(stop_flag):
                log.info(
                    "Training stopped by user during data fetch"
                )
                return {"status": "cancelled"}

            try:
                df = self.fetcher.get_history(
                    code,
                    interval=interval,
                    bars=lookback,
                    use_cache=True,
                    update_db=True,
                )
                if df is None or df.empty:
                    continue

                # Only check raw length; features will add columns
                if len(df) < CONFIG.SEQUENCE_LENGTH + 80:
                    continue

                raw_data[code] = df
            except Exception as e:
                log.debug(f"Error processing {code}: {e}")

        if not raw_data:
            raise ValueError(
                "No valid stock data available for training"
            )

        log.info(
            f"Loaded {len(raw_data)} stocks successfully"
        )

        # --- Phase 2: Split and compute features per split ---
        all_train_features = []
        split_data: Dict[str, Dict[str, pd.DataFrame]] = {}

        for code, df_raw in raw_data.items():
            splits = self._split_single_stock(
                df_raw, horizon, feature_cols
            )
            if splits is None:
                continue

            split_data[code] = splits

            train_df = splits['train']
            train_features = train_df[feature_cols].values
            valid_mask = ~train_df['label'].isna()
            if int(valid_mask.sum()) > 0:
                all_train_features.append(
                    train_features[valid_mask]
                )

        if not all_train_features:
            raise ValueError(
                "No valid training data after split"
            )

        # Fit scaler on training data only
        if self._skip_scaler_fit and self.processor.is_fitted:
            log.info(
                f"Skipping scaler refit (incremental mode, "
                f"existing scaler has "
                f"{self.processor.n_features} features)"
            )
        else:
            combined_train_features = np.concatenate(
                all_train_features, axis=0
            )
            self.processor.fit_scaler(combined_train_features)
            log.info(
                f"Scaler fitted on "
                f"{len(combined_train_features)} samples"
            )

        # Create sequences for classifier
        all_train_seq = {"X": [], "y": []}
        all_val_seq = {"X": [], "y": []}
        all_test_seq = {"X": [], "y": []}

        for code, splits in split_data.items():
            for split_df, storage in [
                (splits["train"], all_train_seq),
                (splits["val"], all_val_seq),
                (splits["test"], all_test_seq),
            ]:
                if len(split_df) >= CONFIG.SEQUENCE_LENGTH + 5:
                    X, y, _ = self.processor.prepare_sequences(
                        split_df,
                        feature_cols,
                        fit_scaler=False,
                    )
                    if len(X) > 0:
                        storage["X"].append(X)
                        storage["y"].append(y)

        if not all_train_seq["X"]:
            raise ValueError(
                "No training sequences available"
            )

        X_train = np.concatenate(all_train_seq["X"])
        y_train = np.concatenate(all_train_seq["y"])
        X_val = (
            np.concatenate(all_val_seq["X"])
            if all_val_seq["X"]
            else None
        )
        y_val = (
            np.concatenate(all_val_seq["y"])
            if all_val_seq["y"]
            else None
        )
        X_test = (
            np.concatenate(all_test_seq["X"])
            if all_test_seq["X"]
            else None
        )
        y_test = (
            np.concatenate(all_test_seq["y"])
            if all_test_seq["y"]
            else None
        )

        self.input_size = int(X_train.shape[2])

        log.info(
            f"Training data: {len(X_train)} samples, "
            f"{self.input_size} features"
        )

        # Save scaler
        scaler_path = (
            CONFIG.MODEL_DIR / f"scaler_{interval}_{horizon}.pkl"
        )
        self.processor.save_scaler(str(scaler_path))

        # --- Phase 3: Train classifier ensemble ---
        self.ensemble = EnsembleModel(
            input_size=self.input_size,
            model_names=model_names,
        )
        # Issue 19: set interval/horizon before training so save() uses them
        self.ensemble.interval = str(interval)
        self.ensemble.prediction_horizon = int(horizon)

        if incremental:
            ensemble_path = (
                CONFIG.MODEL_DIR
                / f"ensemble_{interval}_{horizon}.pt"
            )
            if ensemble_path.exists():
                self.ensemble.load(str(ensemble_path))
                log.info(
                    "Loaded existing ensemble for "
                    "incremental training"
                )

        # Ensure we have validation data
        if X_val is None or len(X_val) == 0:
            split_idx = int(len(X_train) * 0.85)
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]

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
        )

        if self._should_stop(stop_flag):
            log.info("Training stopped by user")
            return {
                "status": "cancelled",
                "history": history,
            }

        # Calibrate ensemble
        self.ensemble.calibrate(X_val, y_val)

        if save_model:
            ensemble_path = (
                CONFIG.MODEL_DIR
                / f"ensemble_{interval}_{horizon}.pt"
            )
            self.ensemble.save(str(ensemble_path))
            log.info(f"Ensemble saved: {ensemble_path}")

        # --- Phase 4: Train forecaster ---
        Xf_train_list, Yf_train_list = [], []
        Xf_val_list, Yf_val_list = [], []

        for code, splits in split_data.items():
            if self._should_stop(stop_flag):
                break

            tr = splits["train"]
            va = splits["val"]

            if (
                len(tr)
                >= CONFIG.SEQUENCE_LENGTH + horizon + 5
            ):
                Xf, Yf = (
                    self.processor.prepare_forecast_sequences(
                        tr,
                        feature_cols,
                        horizon=horizon,
                        fit_scaler=False,
                    )
                )
                if len(Xf) > 0:
                    Xf_train_list.append(Xf)
                    Yf_train_list.append(Yf)

            if (
                len(va)
                >= CONFIG.SEQUENCE_LENGTH + horizon + 5
            ):
                Xf, Yf = (
                    self.processor.prepare_forecast_sequences(
                        va,
                        feature_cols,
                        horizon=horizon,
                        fit_scaler=False,
                    )
                )
                if len(Xf) > 0:
                    Xf_val_list.append(Xf)
                    Yf_val_list.append(Yf)

        forecaster_trained = False

        if Xf_train_list and Xf_val_list:
            try:
                Xf_train = np.concatenate(
                    Xf_train_list, axis=0
                )
                Yf_train = np.concatenate(
                    Yf_train_list, axis=0
                )
                Xf_val = np.concatenate(Xf_val_list, axis=0)
                Yf_val = np.concatenate(Yf_val_list, axis=0)

                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                )

                forecaster = TCNModel(
                    input_size=self.input_size,
                    hidden_size=CONFIG.model.hidden_size,
                    num_classes=horizon,
                    dropout=CONFIG.model.dropout,
                ).to(device)

                optimizer = torch.optim.AdamW(
                    forecaster.parameters(),
                    lr=CONFIG.model.learning_rate,
                    weight_decay=CONFIG.model.weight_decay,
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
                    f"Training forecaster: "
                    f"{len(Xf_train)} samples, "
                    f"horizon={horizon}"
                )

                for ep in range(fore_epochs):
                    if self._should_stop(stop_flag):
                        log.info(
                            "Forecaster training stopped"
                        )
                        break

                    forecaster.train()
                    train_losses = []

                    for xb, yb in train_loader:
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
                        train_losses.append(
                            float(loss.item())
                        )

                    forecaster.eval()
                    val_losses = []

                    with torch.inference_mode():
                        for xb, yb in val_loader:
                            xb = xb.to(device)
                            yb = yb.to(device)
                            pred, _ = forecaster(xb)
                            val_losses.append(
                                float(
                                    loss_fn(pred, yb).item()
                                )
                            )

                    train_loss = float(
                        np.mean(train_losses)
                    )
                    val_loss = (
                        float(np.mean(val_losses))
                        if val_losses
                        else float("inf")
                    )

                    log.info(
                        f"Forecaster epoch "
                        f"{ep + 1}/{fore_epochs}: "
                        f"train_mse={train_loss:.6f}, "
                        f"val_mse={val_loss:.6f}"
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
                            log.info(
                                "Forecaster early stopping"
                            )
                            break

                if best_state:
                    forecaster.load_state_dict(best_state)
                    forecaster_trained = True

                    if save_model:
                        forecast_path = (
                            CONFIG.MODEL_DIR
                            / f"forecast_{interval}_{horizon}.pt"
                        )
                        payload = {
                            "input_size": int(
                                self.input_size
                            ),
                            "interval": str(interval),
                            "horizon": int(horizon),
                            "arch": {
                                "hidden_size": int(
                                    CONFIG.model.hidden_size
                                ),
                                "dropout": float(
                                    CONFIG.model.dropout
                                ),
                            },
                            "state_dict": forecaster.state_dict(),
                        }

                        try:
                            from utils.atomic_io import (
                                atomic_torch_save,
                            )

                            atomic_torch_save(
                                forecast_path, payload
                            )
                        except ImportError:
                            torch.save(payload, forecast_path)

                        log.info(
                            f"Forecaster saved: "
                            f"{forecast_path}"
                        )

            except Exception as e:
                log.error(
                    f"Forecaster training failed: {e}"
                )
                import traceback

                traceback.print_exc()
        else:
            log.warning(
                "Forecaster training skipped: "
                "insufficient data"
            )

        # --- Phase 5: Evaluate on test set ---
        test_acc = 0.0
        if X_test is not None and len(X_test) > 0:
            try:
                preds = self.ensemble.predict_batch(
                    X_test[:2000]
                )
                pred_cls = np.array(
                    [p.predicted_class for p in preds]
                )
                test_acc = float(
                    np.mean(
                        pred_cls == y_test[: len(pred_cls)]
                    )
                )
                log.info(f"Test accuracy: {test_acc:.2%}")
            except Exception as e:
                log.warning(
                    f"Test evaluation failed: {e}"
                )

        # Calculate best validation accuracy
        best_accuracy = 0.0
        if history:
            for h in history.values():
                val_acc_list = h.get("val_acc", [])
                if val_acc_list:
                    best_accuracy = max(
                        best_accuracy, max(val_acc_list)
                    )

        # Use test accuracy as fallback
        if best_accuracy == 0.0 and test_acc > 0.0:
            best_accuracy = test_acc

        return {
            "status": "complete",
            "history": history,
            "best_accuracy": float(best_accuracy),
            "test_metrics": {"accuracy": test_acc},
            "input_size": int(self.input_size),
            "num_models": (
                len(self.ensemble.models)
                if self.ensemble
                else 0
            ),
            "epochs": int(epochs),
            "train_samples": int(len(X_train)),
            "val_samples": (
                int(len(X_val)) if X_val is not None else 0
            ),
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
    # EVALUATION
    # =========================================================================

    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        r: np.ndarray,
    ) -> Dict:
        """Evaluate model on test data"""
        from sklearn.metrics import (
            precision_recall_fscore_support,
            confusion_matrix,
        )

        if len(X) == 0 or len(y) == 0:
            return {
                'accuracy': 0,
                'trading': {},
                'confusion_matrix': [],
                'up_precision': 0,
                'up_recall': 0,
                'up_f1': 0,
            }

        predictions = self.ensemble.predict_batch(X)
        pred_classes = np.array(
            [p.predicted_class for p in predictions]
        )

        if len(pred_classes) == 0:
            return {
                'accuracy': 0,
                'trading': {},
                'confusion_matrix': [],
                'up_precision': 0,
                'up_recall': 0,
                'up_f1': 0,
            }

        cm = confusion_matrix(
            y, pred_classes, labels=[0, 1, 2]
        )
        pr, rc, f1, _ = precision_recall_fscore_support(
            y,
            pred_classes,
            labels=[2],
            average=None,
            zero_division=0,
        )

        metrics_extra = {
            "confusion_matrix": cm.tolist(),
            "up_precision": (
                float(pr[0]) if len(pr) > 0 else 0.0
            ),
            "up_recall": (
                float(rc[0]) if len(rc) > 0 else 0.0
            ),
            "up_f1": (
                float(f1[0]) if len(f1) > 0 else 0.0
            ),
        }

        confidences = np.array(
            [p.confidence for p in predictions]
        )
        accuracy = float(np.mean(pred_classes == y))

        # Per-class accuracy
        class_acc = {}
        for c in range(CONFIG.NUM_CLASSES):
            mask = y == c
            if mask.sum() > 0:
                class_acc[c] = float(
                    np.mean(pred_classes[mask] == c)
                )
            else:
                class_acc[c] = 0.0

        # Trading simulation
        trading_metrics = self._simulate_trading(
            pred_classes, confidences, r
        )

        return {
            'accuracy': accuracy,
            'class_accuracy': class_acc,
            'mean_confidence': (
                float(np.mean(confidences))
                if len(confidences) > 0
                else 0.0
            ),
            'trading': trading_metrics,
            **metrics_extra,
        }

    def _simulate_trading(
        self,
        preds: np.ndarray,
        confs: np.ndarray,
        returns: np.ndarray,
    ) -> Dict:
        """Simulate trading with proper handling of horizon returns."""
        confidence_mask = confs >= CONFIG.MIN_CONFIDENCE

        position = np.zeros_like(preds, dtype=float)
        position[preds == 2] = 1  # UP -> Long
        position = position * confidence_mask

        horizon = self.prediction_horizon
        costs_pct = (
            (
                CONFIG.COMMISSION * 2
                + CONFIG.SLIPPAGE * 2
                + CONFIG.STAMP_TAX
            )
            * 100
        )

        entries = np.diff(position, prepend=0) > 0
        exits = np.diff(position, prepend=0) < 0

        trades = []
        in_position = False
        entry_idx = 0

        for i in range(len(position)):
            if entries[i] and not in_position:
                in_position = True
                entry_idx = i
            elif (
                exits[i] or i == len(position) - 1
            ) and in_position:
                if entry_idx < len(returns):
                    trade_return = (
                        returns[entry_idx] - costs_pct
                    )
                    trades.append(trade_return)
                in_position = False

        num_trades = len(trades)

        if num_trades > 0:
            trades = np.array(trades)
            trades_decimal = trades / 100

            total_return = (
                np.prod(1 + trades_decimal) - 1
            ) * 100

            wins = trades[trades > 0]
            losses = trades[trades < 0]

            win_rate = (
                len(wins) / num_trades
                if num_trades > 0
                else 0
            )

            gross_profit = (
                np.sum(wins) if len(wins) > 0 else 0
            )
            gross_loss = (
                abs(np.sum(losses))
                if len(losses) > 0
                else 1e-8
            )
            profit_factor = gross_profit / gross_loss

            if len(trades) > 1 and np.std(trades) > 0:
                avg_holding = horizon
                trades_per_year = 252 / avg_holding
                sharpe = (
                    np.mean(trades)
                    / np.std(trades)
                    * np.sqrt(trades_per_year)
                )
            else:
                sharpe = 0

            cumulative = np.cumsum(trades_decimal)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            max_drawdown = (
                abs(np.min(drawdown))
                if len(drawdown) > 0
                else 0
            )

        else:
            total_return = 0
            win_rate = 0
            profit_factor = 0
            sharpe = 0
            max_drawdown = 0

        avg_return = (
            np.mean(returns) if len(returns) > 0 else 0
        )
        num_periods = (
            len(returns) // horizon if horizon > 0 else 1
        )
        buyhold_return = avg_return * num_periods / 100

        return {
            'total_return': total_return,
            'buyhold_return': buyhold_return * 100,
            'excess_return': total_return
            - buyhold_return * 100,
            'trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
        }

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_ensemble(self) -> Optional[EnsembleModel]:
        """Get the trained ensemble model"""
        return self.ensemble

    def save_training_report(
        self, results: Dict, path: str = None
    ):
        """Save training report to file"""
        import json

        path = path or str(
            CONFIG.DATA_DIR / "training_report.json"
        )

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
        report['timestamp'] = datetime.now().isoformat()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)

        log.info(f"Training report saved to {path}")