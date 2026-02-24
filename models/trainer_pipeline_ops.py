from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import CONFIG
from models.ensemble import EnsembleModel as _DefaultEnsembleModel
from utils.cancellation import CancelledException
from utils.logger import get_logger

log = get_logger(__name__)

try:
    from utils.atomic_io import atomic_torch_save
except ImportError:
    atomic_torch_save = None

# FIX STOP: Reduced from 10 to 3 for faster cancellation response
_STOP_CHECK_INTERVAL = 3
_TRAINING_INTERVAL_LOCK = "1m"
# FIX 1M: Reduced from 10080 to 480 bars - free sources provide 1-2 days of 1m data
_MIN_1M_LOOKBACK_BARS = 480
_DEFAULT_ENSEMBLE_MODELS = ["lstm", "gru", "tcn", "transformer", "hybrid"]
_INCREMENTAL_REGIME_BLOCK_LEVELS = {"high"}


def _write_artifact_checksum(path: Path) -> None:
    import hashlib

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
    except (ImportError, OSError, IOError):
        checksum_path.write_text(payload, encoding="utf-8")


def _resolve_learning_rate(explicit_lr: float | None = None) -> float:
    if explicit_lr is not None:
        return float(explicit_lr)

    try:
        from models.auto_learner import get_effective_learning_rate

        return get_effective_learning_rate()
    except ImportError:
        pass

    return CONFIG.model.learning_rate


def _resolve_ensemble_model_class() -> type[Any]:
    """Resolve the ensemble class via models.trainer when available.

    This keeps runtime behavior unchanged while preserving testability when
    tests monkeypatch `models.trainer.EnsembleModel`.
    """
    try:
        from models import trainer as trainer_mod

        candidate = getattr(trainer_mod, "EnsembleModel", None)
        if candidate is not None:
            return candidate
    except Exception as e:
        log.debug("Falling back to default EnsembleModel resolver: %s", e)
    return _DefaultEnsembleModel


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
    """Train complete pipeline:
    1) Classification ensemble for trading signals
    2) Multi-step forecaster for AI-generated price curves.
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
    ensemble_cls = _resolve_ensemble_model_class()

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

    # --- Phase 1.5: Early regime detection for incremental training ---
    # FIX RACE: Check regime shift BEFORE processing data to avoid wasting resources
    # on incremental training that will be blocked anyway
    if incremental:
        # Quick preliminary regime check using raw data statistics
        preliminary_regime = self._preliminary_regime_check(raw_data, interval)
        if preliminary_regime.get("level") == "high":
            log.warning(
                "Preliminary regime check indicates high shift (score=%.3f); "
                "forcing full retrain before data processing",
                float(preliminary_regime.get("score", 0.0)),
            )
            incremental = False  # Force full retrain early

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

    # FIX SHAPE: Validate array dimensions before accessing shape[2]
    if X_train.ndim != 3:
        raise ValueError(
            f"X_train must be 3D array (samples, seq_len, features), "
            f"got {X_train.ndim}D shape {X_train.shape}"
        )
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
            # FIX SHAPE: Validate array dimensions before accessing shape[2]
            if X_train.ndim != 3:
                raise ValueError(
                    f"X_train must be 3D array (samples, seq_len, features), "
                    f"got {X_train.ndim}D shape {X_train.shape}"
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
            temp_ensemble = ensemble_cls(
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
                    "Failed to load existing ensemble 閳?"
                    "training from scratch"
                )
                self.ensemble = ensemble_cls(
                    input_size=self.input_size,
                    model_names=model_names,
                )
        else:
            self.ensemble = ensemble_cls(
                input_size=self.input_size,
                model_names=model_names,
            )
    else:
        self.ensemble = ensemble_cls(
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
                    deployment["deployed"] = False
                    log.error(
                        "Deployment FAILED: Model quality passed but artifact promotion failed: %s",
                        ", ".join(failed),
                    )
                    # FIX DEPLOY: Raise exception to ensure caller knows deployment failed
                    raise RuntimeError(
                        f"Deployment failed: unable to promote artifacts: {', '.join(failed)}"
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
    """Train multi-step forecaster. Returns True if successful.

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
        # FIX CANCEL4: Re-raise immediately without catching in outer Exception
        log.info("Forecaster training cancelled")
        raise
    except Exception as e:
        log.error(f"Forecaster training failed: {e}")
        import traceback
        traceback.print_exc()
        # FIX CANCEL4: Don't catch CancelledException in generic handler
        if isinstance(e, CancelledException):
            raise

    return False

