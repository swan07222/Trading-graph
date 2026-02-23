# models/ensemble.py

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)

try:
    from utils.cancellation import CancelledException
except ImportError:
    class CancelledException(Exception):  # type: ignore[no-redef]
        pass

def _get_effective_learning_rate() -> float:
    """Get effective learning rate, checking thread-local override first."""
    try:
        from models.auto_learner import get_effective_learning_rate
        return float(get_effective_learning_rate())
    except ImportError:
        pass

    return float(CONFIG.model.learning_rate)

@dataclass
class EnsemblePrediction:
    """Single-sample prediction result from the ensemble."""

    probabilities: np.ndarray
    predicted_class: int
    confidence: float
    raw_confidence: float
    entropy: float
    agreement: float
    margin: float
    brier_score: float
    individual_predictions: dict[str, np.ndarray]

    @property
    def prob_up(self) -> float:
        return float(self.probabilities[2]) if len(self.probabilities) > 2 else 0.0

    @property
    def prob_neutral(self) -> float:
        return float(self.probabilities[1]) if len(self.probabilities) > 1 else 0.0

    @property
    def prob_down(self) -> float:
        return float(self.probabilities[0]) if len(self.probabilities) > 0 else 0.0

    @property
    def is_confident(self) -> bool:
        return bool(self.confidence >= float(CONFIG.model.min_confidence))

def _build_amp_context(device: str) -> tuple[Callable[[], Any], Any | None]:
    """Return (context_factory, GradScaler_or_None) compatible with torch >= 1.9."""
    use_amp = device == "cuda"
    if not use_amp:
        return (lambda: nullcontext()), None

    # torch >= 2.0 path
    try:
        from torch.amp import GradScaler, autocast
        return (lambda: autocast("cuda", enabled=True)), GradScaler("cuda", enabled=True)
    except (ImportError, TypeError):
        pass

    # torch 1.x path
    try:
        from torch.cuda.amp import GradScaler, autocast
        return (lambda: autocast(**{"enabled": True})), GradScaler(enabled=True)
    except (ImportError, TypeError):
        pass

    return (lambda: nullcontext()), None

class EnsembleModel:
    """Ensemble of multiple neural networks with calibrated weighted voting."""

    _MODEL_CLASSES: dict | None = None  # class-level cache

    def __init__(
        self,
        input_size: int,
        model_names: list[str] | None = None,
    ):
        """Initialize ensemble with the configured model set."""
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")

        self.input_size: int = int(input_size)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        self._lock = threading.RLock()  # reentrant for nested calls
        self.temperature: float = 1.0

        # Metadata - updated by train() and load()
        self.interval: str = "1d"
        self.prediction_horizon: int = int(CONFIG.model.prediction_horizon)
        self.trained_stock_codes: list[str] = []
        self.trained_stock_last_train: dict[str, str] = {}

        model_names = model_names or ["lstm", "gru", "tcn", "transformer", "hybrid"]

        self.models: dict[str, nn.Module] = {}
        self.weights: dict[str, float] = {}

        cls_map = self._get_model_classes()
        for name in model_names:
            if name in cls_map:
                self._init_model(name)
            else:
                log.warning(
                    f"Unknown model name '{name}', available: {list(cls_map.keys())}"
                )

        self._normalize_weights()
        total_params = sum(
            sum(p.numel() for p in m.parameters()) for m in self.models.values()
        )
        log.info(
            f"Ensemble ready: models={list(self.models.keys())}, "
            f"params={total_params:,}, device={self.device}"
        )

    # ------------------------------------------------------------------
    # Artifact integrity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _artifact_checksum_path(path: Path) -> Path:
        from utils.atomic_io import artifact_checksum_path

        return artifact_checksum_path(path)

    def _write_artifact_checksum(self, path: Path) -> None:
        try:
            from utils.atomic_io import write_checksum_sidecar

            write_checksum_sidecar(path)
        except Exception as exc:
            log.warning("Failed writing checksum sidecar for %s: %s", path, exc)

    def _verify_artifact_checksum(self, path: Path) -> bool:
        checksum_path = self._artifact_checksum_path(path)
        require_checksum = bool(
            getattr(getattr(CONFIG, "model", None), "require_artifact_checksum", True)
        )
        try:
            from utils.atomic_io import verify_checksum_sidecar

            ok = bool(
                verify_checksum_sidecar(
                    path,
                    require=require_checksum,
                )
            )
            if not ok:
                log.error(
                    "Checksum verification failed for %s (sidecar=%s, require=%s)",
                    path,
                    checksum_path,
                    require_checksum,
                )
            return ok
        except Exception as exc:
            log.error("Checksum verification failed for %s: %s", path, exc)
            return False

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    @classmethod
    def _get_model_classes(cls) -> dict:
        if cls._MODEL_CLASSES is None:
            from .networks import (
                GRUModel,
                HybridModel,
                LSTMModel,
                TCNModel,
                TransformerModel,
            )
            cls._MODEL_CLASSES = {
                "lstm": LSTMModel,
                "transformer": TransformerModel,
                "gru": GRUModel,
                "tcn": TCNModel,
                "hybrid": HybridModel,
            }
        return cls._MODEL_CLASSES

    def _init_model(
        self,
        name: str,
        hidden_size: int | None = None,
        dropout: float | None = None,
        num_classes: int | None = None,
    ) -> None:
        cls_map = self._get_model_classes()
        try:
            model = cls_map[name](
                input_size=self.input_size,
                hidden_size=hidden_size or CONFIG.model.hidden_size,
                num_classes=num_classes or CONFIG.model.num_classes,
                dropout=dropout or CONFIG.model.dropout,
            )
            model.to(self.device)
            self.models[name] = model
            self.weights[name] = 1.0
            log.debug(f"Initialised {name}")
        except Exception as e:
            log.error(f"Failed to initialise {name}: {e}")

    def _normalize_weights(self) -> None:
        # FIX NORM: Handle empty weights dict gracefully
        if not self.weights:
            return
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        else:
            # All weights are zero - set uniform
            n = len(self.weights)
            if n > 0:
                self.weights = {k: 1.0 / n for k in self.weights}

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def _should_stop(stop_flag: Any) -> bool:
        """Check whether training should be aborted."""
        if stop_flag is None:
            return False

        # Object with .is_cancelled attribute (property or field)
        is_cancelled = getattr(stop_flag, "is_cancelled", None)
        if is_cancelled is not None:
            try:
                return bool(is_cancelled() if callable(is_cancelled) else is_cancelled)
            except Exception:
                return False

        if callable(stop_flag):
            try:
                return bool(stop_flag())
            except Exception:
                return False

        return False

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def calibrate(
        self, X_val: np.ndarray, y_val: np.ndarray, batch_size: int = 512
    ) -> None:
        """Temperature-scale the ensemble's weighted logits on a held-out set."""
        if len(X_val) == 0 or len(y_val) == 0:
            log.warning("Empty validation data for calibration - skipping")
            return

        dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with self._lock:
            models = list(self.models.items())
            weights = dict(self.weights)

        if not models:
            log.warning("No models available for calibration")
            return

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(self.device)
            weighted_logits: torch.Tensor | None = None

            with torch.inference_mode():
                for name, model in models:
                    model.eval()
                    logits, _ = model(batch_X)
                    w = weights.get(name, 1.0 / max(1, len(models)))
                    if weighted_logits is None:
                        weighted_logits = logits * w
                    else:
                        weighted_logits = weighted_logits + logits * w

            if weighted_logits is not None:
                all_logits.append(weighted_logits.cpu())
                all_labels.append(batch_y)

        if not all_logits:
            return

        combined_logits = torch.cat(all_logits)
        combined_labels = torch.cat(all_labels)

        best_temp = 1.0
        best_nll = float("inf")

        # FIX CALIB: Finer temperature grid with more granularity
        temp_grid = [
            0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75,
            0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0,
        ]

        for temp in temp_grid:
            nll = F.cross_entropy(combined_logits / temp, combined_labels).item()
            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        with self._lock:
            self.temperature = best_temp

        log.info(f"Calibration: temperature={best_temp:.2f}, NLL={best_nll:.4f}")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int | None = None,
        batch_size: int | None = None,
        callback: Callable | None = None,
        stop_flag: Any = None,
        interval: str | None = None,
        horizon: int | None = None,
        learning_rate: float | None = None,
    ) -> dict:
        """
        Train all models in the ensemble.

        Args:
            X_train: Training features (N, seq_len, n_features)
            y_train: Training labels (N,)
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs (default: from CONFIG)
            batch_size: Batch size (default: from CONFIG)
            callback: Optional callback(model_name, epoch, val_acc)
            stop_flag: Optional cancellation token/callable
            interval: Data interval metadata
            horizon: Prediction horizon metadata
            learning_rate: Explicit learning rate override

        Returns:
            Dict mapping model name to training history
        """
        with self._lock:
            if interval is not None:
                self.interval = str(interval)
            if horizon is not None:
                self.prediction_horizon = int(horizon)

        if not self.models:
            log.warning("No models to train")
            return {}

        epochs = epochs or CONFIG.model.epochs
        batch_size = batch_size or CONFIG.model.batch_size

        if learning_rate is not None:
            effective_lr = float(learning_rate)
        else:
            effective_lr = _get_effective_learning_rate()

        if X_train.shape[-1] != self.input_size:
            raise ValueError(
                f"X_train feature dim {X_train.shape[-1]} != input_size {self.input_size}"
            )

        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

        pin = self.device == "cuda"
        sample_count = int(len(train_ds))
        # Windows process-spawn overhead can dominate small training jobs.
        if sample_count < 5000 or os.name == "nt":
            n_workers = 0
        else:
            n_workers = min(4, max(0, (os.cpu_count() or 1) - 1))
        if n_workers <= 0:
            n_workers = 0
        persist = n_workers > 0

        loader_kw = dict(
            num_workers=n_workers, pin_memory=pin,
            persistent_workers=persist if n_workers > 0 else False,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kw)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kw)

        # Class weights (inverse-frequency, smoothed)
        counts = np.bincount(y_train, minlength=CONFIG.model.num_classes).astype(np.float64)
        inv_freq = 1.0 / (counts + len(y_train) * 0.01)  # Laplace-smoothed
        inv_freq /= inv_freq.sum()
        class_weights = torch.FloatTensor(inv_freq).to(self.device)

        history: dict[str, dict] = {}
        val_accuracies: dict[str, float] = {}

        log.info(f"Training with learning_rate={effective_lr:.6f}")

        for name, model in list(self.models.items()):
            if self._should_stop(stop_flag):
                log.info("Training cancelled")
                break

            log.info(f"Training {name} ...")
            model_hist, best_acc = self._train_single_model(
                model=model,
                name=name,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                epochs=epochs,
                learning_rate=effective_lr,
                callback=callback,
                stop_flag=stop_flag,
            )
            history[name] = model_hist
            val_accuracies[name] = best_acc

        self._update_weights(val_accuracies)

        # Skip expensive calibration when stop was requested or no model ran.
        if val_accuracies and len(X_val) >= 128 and not self._should_stop(stop_flag):
            self.calibrate(X_val, y_val)

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return history

    def _train_single_model(
        self,
        model: nn.Module,
        name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        epochs: int,
        learning_rate: float,
        callback: Callable | None = None,
        stop_flag: Any = None,
    ) -> tuple[dict, float]:
        """
        Train one model with early stopping, warmup + cosine schedule,
        optional AMP.

        FIX RESTORE: best_state is always restored even when
        CancelledException is raised, preventing a half-trained model
        from being left as the active model.
        """

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=CONFIG.model.weight_decay,
        )

        warmup_epochs = max(1, epochs // 10)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        amp_ctx, scaler = _build_amp_context(self.device)
        use_amp = scaler is not None

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0
        patience_counter = 0
        best_state: dict | None = None
        patience_limit = int(CONFIG.model.early_stop_patience)

        _STOP_CHECK_INTERVAL = 10

        try:
            for epoch in range(int(epochs)):
                if self._should_stop(stop_flag):
                    break

                # --- train ---
                model.train()
                train_losses: list[float] = []
                cancelled = False

                for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                    if batch_idx % _STOP_CHECK_INTERVAL == 0 and batch_idx > 0:
                        if self._should_stop(stop_flag):
                            cancelled = True
                            break

                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    if use_amp:
                        with amp_ctx():
                            logits, _ = model(batch_X)
                            loss = criterion(logits, batch_y)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits, _ = model(batch_X)
                        loss = criterion(logits, batch_y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    train_losses.append(float(loss.detach()))

                if cancelled:
                    break

                scheduler.step()

                # --- validate ---
                model.eval()
                val_losses: list[float] = []
                correct = 0
                total = 0

                with torch.inference_mode():
                    for batch_idx, (batch_X, batch_y) in enumerate(val_loader):
                        if batch_idx % _STOP_CHECK_INTERVAL == 0 and batch_idx > 0:
                            if self._should_stop(stop_flag):
                                cancelled = True
                                break

                        batch_X = batch_X.to(self.device, non_blocking=True)
                        batch_y = batch_y.to(self.device, non_blocking=True)

                        with amp_ctx():
                            logits, _ = model(batch_X)
                            loss = criterion(logits, batch_y)

                        val_losses.append(float(loss))
                        preds = logits.argmax(dim=-1)
                        correct += int((preds == batch_y).sum())
                        total += len(batch_y)

                if cancelled:
                    break

                train_loss = float(np.mean(train_losses)) if train_losses else 0.0
                val_loss = float(np.mean(val_losses)) if val_losses else 0.0
                val_acc = correct / max(total, 1)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= patience_limit:
                        log.info(f"{name}: early stop at epoch {epoch + 1}")
                        break

                # FIX CANCEL: Propagate CancelledException from callback
                if callback is not None:
                    try:
                        callback(name, epoch, val_acc)
                    except CancelledException:
                        log.info(f"{name}: cancelled via callback at epoch {epoch + 1}")
                        raise
                    except Exception:
                        pass

        except CancelledException:
            # FIX RESTORE: Still restore best state before re-raising
            if best_state is not None:
                model.load_state_dict(best_state)
                model.to(self.device)
                log.info(f"{name}: restored best state before cancellation (acc={best_val_acc:.2%})")
            raise

        # Normal completion - restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        log.info(f"{name} done - best val acc: {best_val_acc:.2%}")
        return history, best_val_acc

    def _update_weights(self, val_accuracies: dict[str, float]) -> None:
        """
        Update ensemble weights from validation accuracies.

        If training is partial (e.g., cancelled mid-cycle), only the trained
        models are reweighted while untrained models retain their prior mass.
        This avoids artificially boosting untrained models with placeholder
        scores.
        """
        if not val_accuracies:
            return

        with self._lock:
            names_all = list(self.models.keys())
            current = {
                n: float(self.weights.get(n, 0.0))
                for n in names_all
            }

        trained_names = [n for n in names_all if n in val_accuracies]
        if not trained_names:
            return

        accs = np.array(
            [float(val_accuracies[n]) for n in trained_names],
            dtype=np.float64,
        )
        if accs.size == 0:
            return
        accs = np.nan_to_num(accs, nan=0.0, posinf=1.0, neginf=0.0)

        temperature = 0.5
        shifted = (accs - np.max(accs)) / temperature
        exp_w = np.exp(shifted)
        exp_sum = float(exp_w.sum())
        if not np.isfinite(exp_sum) or exp_sum <= 0.0:
            trained_dist = np.full(accs.size, 1.0 / float(accs.size))
        else:
            trained_dist = exp_w / exp_sum

        if len(trained_names) == len(names_all):
            new_weights = {
                n: float(w)
                for n, w in zip(trained_names, trained_dist, strict=False)
            }
        else:
            untrained_names = [n for n in names_all if n not in trained_names]
            untrained_raw = np.array(
                [max(0.0, current.get(n, 0.0)) for n in untrained_names],
                dtype=np.float64,
            )
            untrained_mass = float(np.sum(untrained_raw))
            if not np.isfinite(untrained_mass):
                untrained_mass = 0.0

            # Keep room for newly trained models even when stale mass is large.
            untrained_mass = min(max(0.0, untrained_mass), 0.85)
            trained_mass = max(0.15, 1.0 - untrained_mass)
            if trained_mass > 1.0:
                trained_mass = 1.0
            untrained_mass = 1.0 - trained_mass

            if untrained_names:
                raw_sum = float(untrained_raw.sum())
                if raw_sum > 0:
                    untrained_dist = untrained_raw / raw_sum
                else:
                    untrained_dist = np.full(
                        len(untrained_names),
                        1.0 / float(len(untrained_names)),
                    )
            else:
                untrained_dist = np.array([], dtype=np.float64)

            new_weights = {}
            for n, w in zip(trained_names, trained_dist, strict=False):
                new_weights[n] = float(w * trained_mass)
            for n, w in zip(untrained_names, untrained_dist, strict=False):
                new_weights[n] = float(w * untrained_mass)

        total = float(sum(new_weights.values()))
        if not np.isfinite(total) or total <= 0.0:
            uniform = 1.0 / float(max(1, len(names_all)))
            new_weights = {n: uniform for n in names_all}
        else:
            new_weights = {n: float(v / total) for n, v in new_weights.items()}

        with self._lock:
            self.weights = new_weights

        log.info(f"Ensemble weights: {self.weights}")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """
        Predict a single sample.

        Args:
            X: Input array of shape (seq_len, n_features) or (1, seq_len, n_features)

        Returns:
            EnsemblePrediction with probabilities, class, confidence, etc.

        Raises:
            ValueError: If input shape is invalid
            RuntimeError: If no models are available
        """
        if X.ndim == 2:
            X = X[np.newaxis]
        if X.ndim != 3 or X.shape[0] != 1:
            raise ValueError(
                f"predict() expects a single sample, got shape {X.shape}. "
                f"Use predict_batch() for multiple samples."
            )
        results = self.predict_batch(X, batch_size=1)
        if not results:
            raise RuntimeError("Prediction failed - no models available")
        return results[0]

    def predict_batch(
        self, X: np.ndarray, batch_size: int = 1024
    ) -> list[EnsemblePrediction]:
        """
        Batch prediction.

        Args:
            X: Input array of shape (N, seq_len, n_features)
            batch_size: Processing batch size

        Returns:
            List of EnsemblePrediction, one per sample
        """
        if X is None or len(X) == 0:
            return []

        if X.ndim == 2:
            X = X[np.newaxis]
        if X.ndim != 3:
            raise ValueError(f"Expected 2D or 3D input, got {X.ndim}D")

        if X.shape[-1] != self.input_size:
            raise ValueError(
                f"Feature dim {X.shape[-1]} != expected input_size {self.input_size}"
            )

        with self._lock:
            models = list(self.models.items())
            weights = dict(self.weights)
            temp = max(self.temperature, 0.1)

        if not models:
            log.warning("No models available for prediction")
            return []

        num_classes = CONFIG.model.num_classes
        max_entropy = float(np.log(num_classes)) if num_classes > 1 else 1.0
        results: list[EnsemblePrediction] = []
        n = len(X)

        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            X_t = torch.FloatTensor(X[start:end]).to(self.device)

            per_model_probs: dict[str, np.ndarray] = {}
            weighted_logits: torch.Tensor | None = None

            with torch.inference_mode():
                for name, model in models:
                    model.eval()
                    logits, _ = model(X_t)
                    per_model_probs[name] = F.softmax(logits, dim=-1).cpu().numpy()

                    w = weights.get(name, 1.0 / max(1, len(models)))
                    if weighted_logits is None:
                        weighted_logits = logits * w
                    else:
                        weighted_logits = weighted_logits + logits * w

            # FIX EMPTY: Skip batch if no logits produced (shouldn't happen
            # with the models check above, but defensive)
            if weighted_logits is None:
                log.warning(f"No logits produced for batch {start}:{end}")
                continue

            final_probs = F.softmax(weighted_logits / temp, dim=-1).cpu().numpy()

            for i in range(end - start):
                probs = final_probs[i]
                pred_cls = int(np.argmax(probs))
                raw_conf = float(np.max(probs))

                probs_safe = np.clip(probs, 1e-8, 1.0)
                ent = float(-np.sum(probs_safe * np.log(probs_safe)))
                ent_norm = ent / max_entropy if max_entropy > 0 else 0.0

                sorted_probs = np.sort(probs)
                margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) >= 2 else 0.0

                model_preds = [
                    int(np.argmax(per_model_probs[m][i])) for m in per_model_probs
                ]
                if model_preds:
                    most_common = max(set(model_preds), key=model_preds.count)
                    agreement = model_preds.count(most_common) / len(model_preds)
                else:
                    agreement = 0.0

                # Reliability-adjusted confidence:
                # lower when entropy is high or model agreement is weak.
                rel = max(0.0, min(1.0, 0.65 + 0.35 * agreement))
                ent_penalty = max(0.0, min(1.0, 1.0 - 0.25 * ent_norm))
                margin_boost = max(0.8, min(1.1, 0.8 + margin))
                conf = max(0.0, min(1.0, raw_conf * rel * ent_penalty * margin_boost))

                target = np.zeros_like(probs)
                target[pred_cls] = 1.0
                brier = float(np.mean((probs - target) ** 2))

                indiv = {m: per_model_probs[m][i] for m in per_model_probs}

                results.append(
                    EnsemblePrediction(
                        probabilities=probs,
                        predicted_class=pred_cls,
                        confidence=conf,
                        raw_confidence=raw_conf,
                        entropy=float(ent_norm),
                        agreement=float(agreement),
                        margin=margin,
                        brier_score=brier,
                        individual_predictions=indiv,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        """
        Save ensemble atomically.

        FIX SAVE: Captures state_dict copies under lock to prevent
        concurrent model mutation during serialization.

        Args:
            path: Target file path (default: ensemble_{interval}_{horizon}.pt)
        """
        from datetime import datetime

        try:
            from utils.atomic_io import atomic_torch_save, atomic_write_json
        except ImportError:
            atomic_torch_save = None
            atomic_write_json = None

        with self._lock:
            interval = str(self.interval)
            horizon = int(self.prediction_horizon)

            if path is None:
                save_dir = Path(CONFIG.model_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                path_obj = save_dir / f"ensemble_{interval}_{horizon}.pt"
            else:
                path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # FIX SAVE: Copy state dicts under lock to prevent mutation
            model_states = {}
            for n, m in self.models.items():
                model_states[n] = {
                    k: v.detach().cpu().clone()
                    for k, v in m.state_dict().items()
                }

            state = {
                "input_size": self.input_size,
                "model_names": list(self.models.keys()),
                "models": model_states,
                "weights": dict(self.weights),
                "temperature": self.temperature,
                "meta": {
                    "interval": interval,
                    "prediction_horizon": horizon,
                    "trained_stock_codes": list(self.trained_stock_codes),
                    "trained_stock_last_train": dict(
                        self.trained_stock_last_train or {}
                    ),
                },
                "arch": {
                    "hidden_size": CONFIG.model.hidden_size,
                    "dropout": CONFIG.model.dropout,
                    "num_classes": CONFIG.model.num_classes,
                },
            }

        # Save outside lock (I/O bound)
        if atomic_torch_save is not None:
            atomic_torch_save(path_obj, state)
        else:
            torch.save(state, path_obj)

        try:
            self._write_artifact_checksum(path_obj)
        except Exception as exc:
            log.warning("Failed writing ensemble checksum sidecar for %s: %s", path_obj, exc)

        manifest = {
            "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "saved_at": datetime.now().isoformat(),
            "ensemble_path": path_obj.name,
            "scaler_path": f"scaler_{interval}_{horizon}.pkl",
            "input_size": self.input_size,
            "num_models": len(model_states),
            "temperature": self.temperature,
            "interval": interval,
            "prediction_horizon": horizon,
            "trained_stock_count": len(self.trained_stock_codes),
            "trained_stock_codes": list(self.trained_stock_codes),
            "trained_stock_last_train": dict(self.trained_stock_last_train or {}),
        }

        manifest_path = path_obj.parent / f"model_manifest_{path_obj.stem}.json"

        if atomic_write_json is not None:
            atomic_write_json(manifest_path, manifest)
        else:
            import json
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

        log.info(f"Ensemble saved -> {path_obj}")

    def load(self, path: str | Path | None = None) -> bool:
        """
        Load ensemble from file.

        Args:
            path: Source file path (default: ensemble_1m_30.pt)

        Returns:
            True if load succeeded, False otherwise
        """
        path_obj = Path(path) if path is not None else (Path(CONFIG.model_dir) / "ensemble_1m_30.pt")
        if not path_obj.exists():
            log.warning(f"No saved model at {path_obj}")
            return False

        try:
            if not self._verify_artifact_checksum(path_obj):
                return False

            allow_unsafe = bool(
                getattr(getattr(CONFIG, "model", None), "allow_unsafe_artifact_load", False)
            )
            require_checksum = bool(
                getattr(getattr(CONFIG, "model", None), "require_artifact_checksum", True)
            )

            def _load_checkpoint(weights_only: bool) -> dict[str, Any]:
                from utils.atomic_io import torch_load

                return torch_load(
                    path_obj,
                    map_location=self.device,
                    weights_only=weights_only,
                    verify_checksum=True,
                    require_checksum=require_checksum,
                    allow_unsafe=allow_unsafe,
                )

            try:
                state = _load_checkpoint(weights_only=True)
            except (OSError, RuntimeError, TypeError, ValueError, ImportError) as exc:
                if not allow_unsafe:
                    log.error(
                        "Ensemble secure load failed for %s and unsafe fallback is disabled: %s",
                        path_obj,
                        exc,
                    )
                    return False
                log.warning(
                    "Ensemble weights-only load failed for %s; "
                    "falling back to unsafe legacy checkpoint load: %s",
                    path_obj,
                    exc,
                )
                state = _load_checkpoint(weights_only=False)

            if not isinstance(state, dict):
                log.error(f"Invalid ensemble file format (not dict): {path_obj}")
                return False
            if "input_size" not in state or "models" not in state:
                log.error(f"Ensemble file missing required keys: {path_obj}")
                return False
            if not isinstance(state.get("models"), dict):
                log.error(f"Invalid ensemble 'models' payload: {path_obj}")
                return False

            with self._lock:
                self.input_size = int(state["input_size"])

                meta = state.get("meta", {})
                self.interval = meta.get("interval", "1m")
                self.prediction_horizon = int(
                    meta.get("prediction_horizon", CONFIG.model.prediction_horizon)
                )
                self.trained_stock_codes = [
                    str(x).strip()
                    for x in list(meta.get("trained_stock_codes", []) or [])
                    if str(x).strip()
                ]
                raw_last_train = meta.get("trained_stock_last_train", {})
                if not isinstance(raw_last_train, dict):
                    raw_last_train = {}
                clean_last_train: dict[str, str] = {}
                for k, v in raw_last_train.items():
                    code = "".join(ch for ch in str(k).strip() if ch.isdigit())
                    if len(code) != 6:
                        continue
                    ts = str(v or "").strip()
                    if not ts:
                        continue
                    clean_last_train[code] = ts
                if clean_last_train:
                    self.trained_stock_last_train = clean_last_train
                else:
                    self.trained_stock_last_train = {}

                arch = state.get("arch", {})
                h = int(arch.get("hidden_size", CONFIG.model.hidden_size))
                d = float(arch.get("dropout", CONFIG.model.dropout))
                c = int(arch.get("num_classes", CONFIG.model.num_classes))

                model_names = state.get("model_names", list(state["models"].keys()))
                cls_map = self._get_model_classes()

                self.models = {}
                self.weights = {}

                for name in model_names:
                    if name not in cls_map:
                        log.warning(f"Skipping unknown model type: {name}")
                        continue
                    if name not in state["models"]:
                        log.warning(f"Skipping model {name}: no saved state")
                        continue

                    self._init_model(name, hidden_size=h, dropout=d, num_classes=c)

                    if name not in self.models:
                        # _init_model failed
                        continue

                    try:
                        self.models[name].load_state_dict(state["models"][name])
                    except RuntimeError as e:
                        log.error(f"Shape mismatch loading {name}: {e}")
                        del self.models[name]
                        self.weights.pop(name, None)
                        continue

                    self.models[name].eval()

                saved_w = state.get("weights", {})
                for name in list(self.models.keys()):
                    self.weights[name] = float(saved_w.get(name, 1.0))

                self._normalize_weights()
                self.temperature = float(state.get("temperature", 1.0))

            if not self.models:
                log.error(f"No models successfully loaded from {path_obj}")
                return False

            log.info(
                f"Ensemble loaded: {list(self.models.keys())}, "
                f"interval={self.interval}, horizon={self.prediction_horizon}"
            )
            return True

        except Exception as e:
            log.error(f"Failed to load ensemble from {path_obj}: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the ensemble models."""
        with self._lock:
            return {
                "input_size": self.input_size,
                "interval": self.interval,
                "prediction_horizon": self.prediction_horizon,
                "temperature": self.temperature,
                "models": list(self.models.keys()),
                "weights": dict(self.weights),
                "total_params": sum(
                    sum(p.numel() for p in m.parameters())
                    for m in self.models.values()
                ),
                "device": self.device,
                "trained_stock_count": int(len(self.trained_stock_codes)),
                "trained_stock_codes": list(self.trained_stock_codes),
                "trained_stock_last_train": dict(
                    self.trained_stock_last_train or {}
                ),
            }

    def set_eval_mode(self) -> None:
        """Set all models to evaluation mode."""
        with self._lock:
            for model in self.models.values():
                model.eval()

    def set_train_mode(self) -> None:
        """Set all models to training mode."""
        with self._lock:
            for model in self.models.values():
                model.train()

    def to(self, device: str) -> EnsembleModel:
        """
        Move all models to specified device.

        Args:
            device: Target device ('cpu', 'cuda', 'cuda:0', etc.)

        Returns:
            self for chaining
        """
        with self._lock:
            self.device = device
            for model in self.models.values():
                model.to(device)
        return self

    def __repr__(self) -> str:
        with self._lock:
            names = list(self.models.keys())
            total_p = sum(
                sum(p.numel() for p in m.parameters()) for m in self.models.values()
            )
        return (
            f"EnsembleModel(input_size={self.input_size}, models={names}, "
            f"params={total_p:,}, device={self.device}, temp={self.temperature:.2f})"
        )

    def __len__(self) -> int:
        """Return number of models in ensemble."""
        with self._lock:
            return len(self.models)
