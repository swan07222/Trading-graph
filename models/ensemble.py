# models/ensemble.py
"""
Ensemble Model — combines multiple neural networks with weighted voting.

Key fixes vs original:
    - Thread-safe train/predict/save/load with proper locking
    - Unified _should_stop logic
    - train() accepts interval/horizon so save() uses correct values
    - predict() handles arbitrary batch sizes; predict_batch() is the workhorse
    - Input shape validation against self.input_size
    - AMP handled cleanly via a single helper
    - Proper GPU memory cleanup after training
    - Learning-rate warmup before cosine annealing
    - Robust calibration with finer temperature grid

FIXES APPLIED:
    - FIX C1 SUPPORT: Added get_effective_learning_rate() import and usage
      to support thread-local LR override from auto_learner.py
    - train() now accepts optional learning_rate parameter for explicit override
    - Improved docstrings and type hints
"""
from __future__ import annotations

import os
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Thread-local LR support (FIX C1)
# ---------------------------------------------------------------------------

def _get_effective_learning_rate() -> float:
    """
    Get effective learning rate, checking thread-local override first.
    
    This supports the thread-local LR pattern from auto_learner.py
    which avoids global CONFIG mutation race conditions.
    """
    try:
        from models.auto_learner import get_effective_learning_rate
        return get_effective_learning_rate()
    except ImportError:
        pass
    
    return CONFIG.model.learning_rate


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------

@dataclass
class EnsemblePrediction:
    """Single-sample prediction result from the ensemble."""

    probabilities: np.ndarray
    predicted_class: int
    confidence: float
    entropy: float
    agreement: float
    individual_predictions: Dict[str, np.ndarray]

    @property
    def prob_up(self) -> float:
        return float(self.probabilities[2]) if len(self.probabilities) > 2 else 0.0

    @property
    def prob_neutral(self) -> float:
        return float(self.probabilities[1]) if len(self.probabilities) > 1 else 0.0

    @property
    def prob_down(self) -> float:
        return float(self.probabilities[0])

    @property
    def is_confident(self) -> bool:
        return self.confidence >= CONFIG.model.min_confidence


# ---------------------------------------------------------------------------
# AMP helpers
# ---------------------------------------------------------------------------

def _build_amp_context(device: str):
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
        return (lambda: autocast(enabled=True)), GradScaler(enabled=True)
    except (ImportError, TypeError):
        pass

    return (lambda: nullcontext()), None


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class EnsembleModel:
    """
    Ensemble of multiple neural networks with calibrated weighted voting.
    
    Supports:
    - Multiple model architectures (LSTM, GRU, TCN, Transformer, Hybrid)
    - Weighted voting based on validation accuracy
    - Temperature-scaled calibration
    - Thread-safe operations
    - Incremental training
    """

    _MODEL_CLASSES: Optional[Dict] = None  # class-level cache

    def __init__(
        self,
        input_size: int,
        model_names: Optional[List[str]] = None,
    ):
        """
        Initialize ensemble with specified models.
        
        Args:
            input_size: Number of input features
            model_names: List of model types to include 
                        (default: ['lstm', 'gru', 'tcn'])
        """
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")

        self.input_size: int = int(input_size)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        self._lock = threading.RLock()  # reentrant for nested calls
        self.temperature: float = 1.0

        # Metadata — updated by train() and load()
        self.interval: str = "1d"
        self.prediction_horizon: int = int(CONFIG.model.prediction_horizon)

        model_names = model_names or ["lstm", "gru", "tcn"]

        self.models: Dict[str, nn.Module] = {}
        self.weights: Dict[str, float] = {}

        cls_map = self._get_model_classes()
        for name in model_names:
            if name in cls_map:
                self._init_model(name)

        self._normalize_weights()
        total_params = sum(
            sum(p.numel() for p in m.parameters()) for m in self.models.values()
        )
        log.info(
            f"Ensemble ready: models={list(self.models.keys())}, "
            f"params={total_params:,}, device={self.device}"
        )

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    @classmethod
    def _get_model_classes(cls) -> Dict:
        if cls._MODEL_CLASSES is None:
            from .networks import (
                LSTMModel, TransformerModel, GRUModel, TCNModel, HybridModel,
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
        hidden_size: Optional[int] = None,
        dropout: Optional[float] = None,
        num_classes: Optional[int] = None,
    ):
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

    def _normalize_weights(self):
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    # ------------------------------------------------------------------
    # Stopping helper
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

        # Plain callable
        if callable(stop_flag):
            try:
                return bool(stop_flag())
            except Exception:
                return False

        return False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self, X_val: np.ndarray, y_val: np.ndarray, batch_size: int = 512
    ):
        """Temperature-scale the ensemble's weighted logits on a held-out set."""
        dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with self._lock:
            models = list(self.models.items())
            weights = dict(self.weights)

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(self.device)
            weighted_logits: Optional[torch.Tensor] = None

            with torch.inference_mode():
                for name, model in models:
                    model.eval()
                    logits, _ = model(batch_X)
                    w = weights.get(name, 1.0 / max(1, len(models)))
                    weighted_logits = (
                        logits * w
                        if weighted_logits is None
                        else weighted_logits + logits * w
                    )

            if weighted_logits is not None:
                all_logits.append(weighted_logits.cpu())
                all_labels.append(batch_y)

        if not all_logits:
            return

        combined_logits = torch.cat(all_logits)
        combined_labels = torch.cat(all_labels)

        best_temp = 1.0
        best_nll = float("inf")

        for temp in [
            0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0,
        ]:
            nll = F.cross_entropy(combined_logits / temp, combined_labels).item()
            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        with self._lock:
            self.temperature = best_temp

        log.info(f"Calibration: temperature={best_temp:.2f}, NLL={best_nll:.4f}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        callback: Optional[Callable] = None,
        stop_flag: Any = None,
        interval: Optional[str] = None,
        horizon: Optional[int] = None,
        learning_rate: Optional[float] = None,  # FIX C1: Explicit LR parameter
    ) -> Dict:
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
            learning_rate: Explicit learning rate override (FIX C1)
        
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
        
        # FIX C1: Use explicit LR if provided, else try thread-local, else CONFIG
        if learning_rate is not None:
            effective_lr = float(learning_rate)
        else:
            effective_lr = _get_effective_learning_rate()

        # Validate shapes
        if X_train.shape[-1] != self.input_size:
            raise ValueError(
                f"X_train feature dim {X_train.shape[-1]} != input_size {self.input_size}"
            )

        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

        pin = self.device == "cuda"
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

        history: Dict[str, Dict] = {}
        val_accuracies: Dict[str, float] = {}

        log.info(f"Training with learning_rate={effective_lr:.6f}")

        for name, model in list(self.models.items()):
            if self._should_stop(stop_flag):
                log.info("Training cancelled")
                break

            log.info(f"Training {name} …")
            model_hist, best_acc = self._train_single_model(
                model=model,
                name=name,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                epochs=epochs,
                learning_rate=effective_lr,  # Pass LR to single model training
                callback=callback,
                stop_flag=stop_flag,
            )
            history[name] = model_hist
            val_accuracies[name] = best_acc

        self._update_weights(val_accuracies)

        if len(X_val) > 0:
            self.calibrate(X_val, y_val)

        # Cleanup
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
        learning_rate: float,  # FIX C1: Accept LR as parameter
        callback: Optional[Callable] = None,
        stop_flag: Any = None,
    ) -> Tuple[Dict, float]:
        """Train one model with early stopping, warmup + cosine schedule, optional AMP."""

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,  # FIX C1: Use passed LR instead of CONFIG
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

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0
        patience_counter = 0
        best_state: Optional[Dict] = None
        patience_limit = int(CONFIG.model.early_stop_patience)

        for epoch in range(int(epochs)):
            if self._should_stop(stop_flag):
                break

            # --- train ---
            model.train()
            train_losses: list[float] = []

            for batch_X, batch_y in train_loader:
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

            scheduler.step()

            # --- validate ---
            model.eval()
            val_losses: list[float] = []
            correct = 0
            total = 0

            with torch.inference_mode():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)

                    with amp_ctx():
                        logits, _ = model(batch_X)
                        loss = criterion(logits, batch_y)

                    val_losses.append(float(loss))
                    preds = logits.argmax(dim=-1)
                    correct += int((preds == batch_y).sum())
                    total += len(batch_y)

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

            if callback is not None:
                try:
                    callback(name, epoch, val_acc)
                except Exception:
                    pass

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        log.info(f"{name} done — best val acc: {best_val_acc:.2%}")
        return history, best_val_acc

    def _update_weights(self, val_accuracies: Dict[str, float]):
        """Softmax-weighted ensemble based on validation accuracy."""
        if not val_accuracies:
            return

        names = list(self.models.keys())
        accs = np.array([val_accuracies.get(n, 0.5) for n in names])

        temperature = 0.5
        exp_w = np.exp(accs / temperature)
        exp_w /= exp_w.sum()

        with self._lock:
            self.weights = {n: float(w) for n, w in zip(names, exp_w)}

        log.info(f"Ensemble weights: {self.weights}")
    
        # ------------------------------------------------------------------
    # Prediction
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
            raise RuntimeError("Prediction failed — no models available")
        return results[0]

    def predict_batch(
        self, X: np.ndarray, batch_size: int = 1024
    ) -> List[EnsemblePrediction]:
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

        num_classes = CONFIG.model.num_classes
        max_entropy = float(np.log(num_classes)) if num_classes > 1 else 1.0
        results: List[EnsemblePrediction] = []
        n = len(X)

        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            X_t = torch.FloatTensor(X[start:end]).to(self.device)

            per_model_probs: Dict[str, np.ndarray] = {}
            weighted_logits: Optional[torch.Tensor] = None

            with torch.inference_mode():
                for name, model in models:
                    model.eval()
                    logits, _ = model(X_t)
                    per_model_probs[name] = F.softmax(logits, dim=-1).cpu().numpy()

                    w = weights.get(name, 1.0 / max(1, len(models)))
                    weighted_logits = (
                        logits * w
                        if weighted_logits is None
                        else weighted_logits + logits * w
                    )

            if weighted_logits is None:
                break

            final_probs = F.softmax(weighted_logits / temp, dim=-1).cpu().numpy()

            for i in range(end - start):
                probs = final_probs[i]
                pred_cls = int(np.argmax(probs))
                conf = float(np.max(probs))

                ent = float(-np.sum(probs * np.log(probs + 1e-8)))
                ent_norm = ent / max_entropy

                model_preds = [
                    int(np.argmax(per_model_probs[m][i])) for m in per_model_probs
                ]
                if model_preds:
                    most_common = max(set(model_preds), key=model_preds.count)
                    agreement = model_preds.count(most_common) / len(model_preds)
                else:
                    agreement = 0.0

                indiv = {m: per_model_probs[m][i] for m in per_model_probs}

                results.append(
                    EnsemblePrediction(
                        probabilities=probs,
                        predicted_class=pred_cls,
                        confidence=conf,
                        entropy=float(ent_norm),
                        agreement=float(agreement),
                        individual_predictions=indiv,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None):
        """
        Save ensemble atomically.
        
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
                path = save_dir / f"ensemble_{interval}_{horizon}.pt"
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "input_size": self.input_size,
                "model_names": list(self.models.keys()),
                "models": {n: m.state_dict() for n, m in self.models.items()},
                "weights": dict(self.weights),
                "temperature": self.temperature,
                "meta": {
                    "interval": interval,
                    "prediction_horizon": horizon,
                },
                "arch": {
                    "hidden_size": CONFIG.model.hidden_size,
                    "dropout": CONFIG.model.dropout,
                    "num_classes": CONFIG.model.num_classes,
                },
            }

        if atomic_torch_save is not None:
            atomic_torch_save(path, state)
        else:
            # Fallback to regular torch.save
            import torch
            torch.save(state, path)

        manifest = {
            "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "saved_at": datetime.now().isoformat(),
            "ensemble_path": path.name,
            "scaler_path": f"scaler_{interval}_{horizon}.pkl",
            "input_size": self.input_size,
            "num_models": len(self.models),
            "temperature": self.temperature,
            "interval": interval,
            "prediction_horizon": horizon,
        }

        manifest_path = path.parent / f"model_manifest_{path.stem}.json"
        
        if atomic_write_json is not None:
            atomic_write_json(manifest_path, manifest)
        else:
            # Fallback to regular json write
            import json
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

        log.info(f"Ensemble saved → {path}")

    def load(self, path: Optional[str] = None) -> bool:
        """
        Load ensemble from file.
        
        Args:
            path: Source file path (default: ensemble_1d_5.pt)
        
        Returns:
            True if load succeeded, False otherwise
        """
        import torch
        
        if path is None:
            path = str(CONFIG.model_dir / "ensemble_1d_5.pt")

        path = Path(path)
        if not path.exists():
            log.warning(f"No saved model at {path}")
            return False

        try:
            state = torch.load(path, map_location=self.device, weights_only=False)

            with self._lock:
                self.input_size = int(state["input_size"])

                meta = state.get("meta", {})
                self.interval = meta.get("interval", "1d")
                self.prediction_horizon = int(
                    meta.get("prediction_horizon", CONFIG.model.prediction_horizon)
                )

                arch = state.get("arch", {})
                h = int(arch.get("hidden_size", CONFIG.model.hidden_size))
                d = float(arch.get("dropout", CONFIG.model.dropout))
                c = int(arch.get("num_classes", CONFIG.model.num_classes))

                model_names = state.get("model_names", list(state["models"].keys()))
                cls_map = self._get_model_classes()

                self.models = {}
                self.weights = {}

                for name in model_names:
                    if name not in cls_map or name not in state["models"]:
                        log.warning(f"Skipping unknown/missing model: {name}")
                        continue

                    self._init_model(name, hidden_size=h, dropout=d, num_classes=c)

                    try:
                        self.models[name].load_state_dict(state["models"][name])
                    except RuntimeError as e:
                        log.error(f"Shape mismatch loading {name}: {e}")
                        del self.models[name]
                        del self.weights[name]
                        continue

                    self.models[name].eval()

                saved_w = state.get("weights", {})
                for name in list(self.models.keys()):
                    self.weights[name] = float(saved_w.get(name, 1.0))

                self._normalize_weights()
                self.temperature = float(state.get("temperature", 1.0))

            log.info(
                f"Ensemble loaded: {list(self.models.keys())}, "
                f"interval={self.interval}, horizon={self.prediction_horizon}"
            )
            return True

        except Exception as e:
            log.error(f"Failed to load ensemble from {path}: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, Any]:
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
            }

    def set_eval_mode(self):
        """Set all models to evaluation mode."""
        with self._lock:
            for model in self.models.values():
                model.eval()

    def set_train_mode(self):
        """Set all models to training mode."""
        with self._lock:
            for model in self.models.values():
                model.train()

    def to(self, device: str) -> "EnsembleModel":
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
        return len(self.models)