# models/ensemble.py
"""
Ensemble Model - Combines multiple neural networks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset
import threading

from .networks import LSTMModel, TransformerModel, GRUModel, TCNModel, HybridModel
from config.settings import CONFIG  # FIXED: correct import
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class EnsemblePrediction:
    """Prediction result from ensemble"""
    probabilities: np.ndarray
    predicted_class: int
    confidence: float
    entropy: float
    agreement: float
    individual_predictions: Dict[str, np.ndarray]
    
    @property
    def prob_up(self) -> float:
        return float(self.probabilities[2])
    
    @property
    def prob_neutral(self) -> float:
        return float(self.probabilities[1])
    
    @property
    def prob_down(self) -> float:
        return float(self.probabilities[0])
    
    @property
    def is_confident(self) -> bool:
        return self.confidence >= CONFIG.model.min_confidence


class EnsembleModel:
    """
    Ensemble of multiple neural networks with weighted voting.
    """
    
    MODEL_CLASSES = {
        'lstm': LSTMModel,
        'transformer': TransformerModel,
        'gru': GRUModel,
        'tcn': TCNModel,
        'hybrid': HybridModel,
    }
    
    def __init__(
        self, 
        input_size: int, 
        model_names: List[str] = None
    ):
        self.input_size = input_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
        self._lock = threading.Lock()
        self.temperature = 1.0  # Initialize temperature
        
        model_names = model_names or ['lstm', 'transformer', 'gru', 'tcn']
        
        self.models: Dict[str, nn.Module] = {}
        self.weights: Dict[str, float] = {}
        
        for name in model_names:
            if name in self.MODEL_CLASSES:
                self._init_model(name)
        
        self._normalize_weights()
        
        log.info(f"Ensemble initialized: {list(self.models.keys())} on {self.device}")
    
    def _init_model(
        self, 
        name: str, 
        hidden_size: int = None, 
        dropout: float = None, 
        num_classes: int = None
    ):
        """Initialize a single model with proper params"""
        try:
            model_class = self.MODEL_CLASSES[name]
            model = model_class(
                input_size=self.input_size,
                hidden_size=hidden_size or CONFIG.model.hidden_size,
                num_classes=num_classes or CONFIG.model.num_classes,
                dropout=dropout or CONFIG.model.dropout
            )
            model.to(self.device)
            self.models[name] = model
            self.weights[name] = 1.0
            log.debug(f"Initialized {name} model")
        except Exception as e:
            log.error(f"Failed to initialize {name}: {e}")
    
    def _normalize_weights(self):
        if not self.weights:
            return
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray, batch_size: int = 512):
        """Calibrate using weighted logits"""
        dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_logits = []
        all_labels = []
        
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(self.device)
            
            weighted_logits = None
            for name, model in self.models.items():
                model.eval()
                with torch.no_grad():
                    logits, _ = model(batch_X)
                    weight = self.weights.get(name, 1.0 / len(self.models))
                    if weighted_logits is None:
                        weighted_logits = logits * weight
                    else:
                        weighted_logits += logits * weight
            
            all_logits.append(weighted_logits.cpu())
            all_labels.append(batch_y)
        
        combined_logits = torch.cat(all_logits, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)
        
        best_temp = 1.0
        best_nll = float('inf')
        
        for temp in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
            scaled = combined_logits / temp
            nll = F.cross_entropy(scaled, combined_labels).item()
            
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        
        self.temperature = best_temp
        log.info(f"Calibration complete: temperature={best_temp:.2f}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = None,
        batch_size: int = None,
        callback: Callable = None,
        stop_flag: Any = None
    ) -> Dict:
        """Train all models in the ensemble."""
        epochs = epochs or CONFIG.model.epochs
        batch_size = batch_size or CONFIG.model.batch_size
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Class weights for imbalanced data
        class_counts = np.bincount(y_train, minlength=CONFIG.model.num_classes)
        weights = 1.0 / (class_counts + 1)
        weights = weights / weights.sum()
        class_weights = torch.FloatTensor(weights).to(self.device)
        
        history = {}
        val_accuracies = {}
        
        for name, model in self.models.items():
            if self._should_stop(stop_flag):
                log.info("Training stopped by user")
                break
                
            log.info(f"Training {name}...")
            
            model_history, best_acc = self._train_single_model(
                model=model,
                name=name,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                epochs=epochs,
                callback=callback,
                stop_flag=stop_flag
            )
            
            history[name] = model_history
            val_accuracies[name] = best_acc
        
        self._update_weights(val_accuracies)
        
        # Calibrate after training
        if len(X_val) > 0:
            self.calibrate(X_val, y_val)
        
        return history
    
    def _should_stop(self, stop_flag: Any) -> bool:
        """Check if training should stop - handles various stop flag types"""
        if stop_flag is None:
            return False
        
        if callable(stop_flag):
            try:
                return bool(stop_flag())
            except TypeError:
                pass
        
        is_cancelled = getattr(stop_flag, "is_cancelled", None)
        if is_cancelled is not None:
            if callable(is_cancelled):
                try:
                    return bool(is_cancelled())
                except TypeError:
                    pass
            else:
                return bool(is_cancelled)
        
        try:
            return bool(stop_flag)
        except Exception:
            return False

    def _train_single_model(
        self,
        model: nn.Module,
        name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        epochs: int,
        callback: Callable = None,
        stop_flag: Any = None
    ) -> Tuple[Dict, float]:
        """Train a single model"""
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG.model.learning_rate,
            weight_decay=CONFIG.model.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0
        patience = 0
        best_state = None
        
        for epoch in range(epochs):
            if self._should_stop(stop_flag):
                break
            
            # Training
            model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                logits, _ = model(batch_X)
                loss = criterion(logits, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            scheduler.step()
            
            # Validation
            model.eval()
            val_losses = []
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    logits, _ = model(batch_X)
                    loss = criterion(logits, batch_y)
                    val_losses.append(loss.item())
                    
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == batch_y).sum().item()
                    total += len(batch_y)
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            val_acc = correct / total if total > 0 else 0
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= CONFIG.model.early_stop_patience:
                    log.info(f"{name}: Early stopping at epoch {epoch+1}")
                    break
            
            if callback:
                callback(name, epoch, val_acc)
            
            if (epoch + 1) % 10 == 0:
                log.info(f"{name} Epoch {epoch+1}: val_acc={val_acc:.2%}")
        
        # Load best weights
        if best_state:
            model.load_state_dict(best_state)
            model.to(self.device)
        
        log.info(f"{name} complete. Best accuracy: {best_val_acc:.2%}")
        return history, best_val_acc
    
    def _update_weights(self, val_accuracies: Dict[str, float]):
        """Update model weights based on validation accuracy"""
        if not val_accuracies:
            return
        
        accs = np.array([val_accuracies.get(name, 0.5) for name in self.models.keys()])
        
        temp = 0.5
        weights = np.exp(accs / temp)
        weights = weights / weights.sum()
        
        self.weights = {name: float(w) for name, w in zip(self.models.keys(), weights)}
        
        log.info(f"Updated weights: {self.weights}")
    
    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """Get ensemble prediction for single sample."""
        if X.ndim == 2:
            X = X[np.newaxis, :]
        
        if X.ndim != 3:
            raise ValueError(f"Expected 2D or 3D input, got shape {X.shape}")
        
        if X.shape[0] != 1:
            raise ValueError(
                f"predict() expects single sample (batch_size=1), got {X.shape[0]}. "
                f"Use predict_batch() for multiple samples."
            )
        
        with self._lock:
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            all_logits = []
            all_probs = {}
            
            for name, model in self.models.items():
                model.eval()
                with torch.no_grad():
                    logits, _ = model(X_tensor)
                    all_logits.append((name, logits))
                    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                    all_probs[name] = probs
            
            # Weighted average of logits
            weighted_logits = torch.zeros_like(all_logits[0][1])
            for name, logits in all_logits:
                weight = self.weights.get(name, 1.0 / len(self.models))
                weighted_logits += logits * weight
            
            # Apply temperature scaling
            scaled_logits = weighted_logits / self.temperature
            final_probs = F.softmax(scaled_logits, dim=-1).cpu().numpy()[0]
            
            predicted_class = int(np.argmax(final_probs))
            confidence = float(np.max(final_probs))
            
            # Entropy
            entropy = -np.sum(final_probs * np.log(final_probs + 1e-8))
            max_entropy = np.log(CONFIG.model.num_classes)
            normalized_entropy = entropy / max_entropy
            
            # Agreement
            predictions = [np.argmax(p) for p in all_probs.values()]
            if predictions:
                most_common = max(set(predictions), key=predictions.count)
                agreement = predictions.count(most_common) / len(predictions)
            else:
                agreement = 0.0
            
            return EnsemblePrediction(
                probabilities=final_probs,
                predicted_class=predicted_class,
                confidence=confidence,
                entropy=normalized_entropy,
                agreement=agreement,
                individual_predictions=all_probs
            )
    
    def predict_batch(self, X: np.ndarray, batch_size: int = 1024) -> List[EnsemblePrediction]:
        """Memory-safe batch inference."""
        if X is None or len(X) == 0:
            return []

        out: List[EnsemblePrediction] = []

        with self._lock:
            n = len(X)
            start = 0
            while start < n:
                end = min(n, start + batch_size)
                xb = X[start:end]
                X_tensor = torch.FloatTensor(xb).to(self.device)

                all_logits: Dict[str, torch.Tensor] = {}
                all_probs: Dict[str, np.ndarray] = {}

                for name, model in self.models.items():
                    model.eval()
                    with torch.inference_mode():
                        logits, _ = model(X_tensor)
                        all_logits[name] = logits
                        all_probs[name] = F.softmax(logits, dim=-1).cpu().numpy()

                first = next(iter(all_logits.values()))
                weighted_logits = torch.zeros_like(first)
                for name, logits in all_logits.items():
                    weight = self.weights.get(name, 1.0 / max(1, len(self.models)))
                    weighted_logits += logits * weight

                scaled_logits = weighted_logits / self.temperature
                final_probs = F.softmax(scaled_logits, dim=-1).cpu().numpy()

                for i in range(len(xb)):
                    probs = final_probs[i]
                    sample_probs = {n: all_probs[n][i] for n in self.models.keys()}

                    pred_cls = int(np.argmax(probs))
                    conf = float(np.max(probs))

                    ent = -np.sum(probs * np.log(probs + 1e-8))
                    ent_norm = float(ent / np.log(CONFIG.model.num_classes))

                    model_preds = [int(np.argmax(p)) for p in sample_probs.values()]
                    if model_preds:
                        most_common = max(set(model_preds), key=model_preds.count)
                        agreement = float(model_preds.count(most_common) / len(model_preds))
                    else:
                        agreement = 0.0

                    out.append(EnsemblePrediction(
                        probabilities=probs,
                        predicted_class=pred_cls,
                        confidence=conf,
                        entropy=ent_norm,
                        agreement=agreement,
                        individual_predictions=sample_probs
                    ))

                start = end

        return out
    
    def save(self, path: str = None):
        """Save ensemble to file atomically."""
        from datetime import datetime
        from utils.atomic_io import atomic_torch_save, atomic_write_json

        path = Path(path or (CONFIG.model_dir / "ensemble.pt"))

        with self._lock:
            state = {
                'input_size': self.input_size,
                'model_names': list(self.models.keys()),
                'models': {name: model.state_dict() for name, model in self.models.items()},
                'weights': self.weights,
                'temperature': self.temperature,
                'arch': {
                    'hidden_size': CONFIG.model.hidden_size,
                    'dropout': CONFIG.model.dropout,
                    'num_classes': CONFIG.model.num_classes,
                }
            }

        atomic_torch_save(path, state)

        manifest = {
            "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "saved_at": datetime.now().isoformat(),
            "ensemble_path": path.name,
            "scaler_path": "scaler.pkl",
            "input_size": int(self.input_size),
            "num_models": len(self.models),
            "temperature": float(self.temperature),
        }
        atomic_write_json(path.parent / "model_manifest.json", manifest)

        log.info(f"Ensemble saved atomically to {path}")

    def load(self, path: str = None) -> bool:
        """Load ensemble from file."""
        path = path or str(CONFIG.model_dir / "ensemble.pt")
        
        if not Path(path).exists():
            log.warning(f"No saved model at {path}")
            return False
        
        try:
            with self._lock:
                state = torch.load(path, map_location=self.device, weights_only=False)
                
                self.input_size = state['input_size']
                model_names = state.get('model_names', list(state['models'].keys()))
                
                arch = state.get("arch", {})
                saved_hidden = int(arch.get("hidden_size", CONFIG.model.hidden_size))
                saved_dropout = float(arch.get("dropout", CONFIG.model.dropout))
                saved_classes = int(arch.get("num_classes", CONFIG.model.num_classes))

                self.models = {}
                self.weights = {}
                
                for name in model_names:
                    if name in self.MODEL_CLASSES and name in state['models']:
                        self._init_model(
                            name,
                            hidden_size=saved_hidden,
                            dropout=saved_dropout,
                            num_classes=saved_classes
                        )
                        self.models[name].load_state_dict(state['models'][name])
                        self.models[name].eval()
                
                saved_weights = state.get('weights', {})
                for name in self.models.keys():
                    self.weights[name] = saved_weights.get(name, 1.0)
                
                self._normalize_weights()
                self.temperature = state.get('temperature', 1.0)
            
            log.info(f"Ensemble loaded: {list(self.models.keys())}, temp={self.temperature:.2f}")
            return True
            
        except Exception as e:
            log.error(f"Failed to load ensemble: {e}")
            return False