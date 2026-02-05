"""
Ensemble Model - Combines multiple neural networks

FIXED Issues:
- Consistent weight initialization and loading
- Proper confidence calculation (from softmax, not untrained head)
- Thread-safe operations
- Clean save/load
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset
import threading

from .networks import LSTMModel, TransformerModel, GRUModel, TCNModel, HybridModel
from config import CONFIG
from utils.logger import log


@dataclass
class EnsemblePrediction:
    """Prediction result from ensemble"""
    probabilities: np.ndarray
    predicted_class: int
    confidence: float  # max(probabilities)
    entropy: float  # prediction uncertainty
    agreement: float  # model agreement
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
        return self.confidence >= CONFIG.MIN_CONFIDENCE


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
        self._lock = threading.Lock()
        
        model_names = model_names or ['lstm', 'transformer', 'gru', 'tcn']
        
        self.models: Dict[str, nn.Module] = {}
        self.weights: Dict[str, float] = {}
        
        for name in model_names:
            if name in self.MODEL_CLASSES:
                self._init_model(name)
        
        self._normalize_weights()
        
        log.info(f"Ensemble initialized: {list(self.models.keys())} on {self.device}")
    
    def _init_model(self, name: str):
        try:
            model_class = self.MODEL_CLASSES[name]
            model = model_class(
                input_size=self.input_size,
                hidden_size=CONFIG.HIDDEN_SIZE,
                num_classes=CONFIG.NUM_CLASSES,
                dropout=CONFIG.DROPOUT
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
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = None,
        batch_size: int = None,
        callback: Callable = None,
        stop_flag: Callable = None
    ) -> Dict:
        """Train all models in the ensemble."""
        epochs = epochs or CONFIG.EPOCHS
        batch_size = batch_size or CONFIG.BATCH_SIZE
        
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
        class_counts = np.bincount(y_train, minlength=CONFIG.NUM_CLASSES)
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
        
        return history
    
    def _should_stop(self, stop_flag) -> bool:
        """Check if training should stop - handles various stop flag types"""
        if stop_flag is None:
            return False
        
        # Case 1: stop_flag is a callable (function that returns bool)
        if callable(stop_flag):
            try:
                result = stop_flag()
                return bool(result)
            except TypeError:
                pass
        
        # Case 2: stop_flag has is_cancelled attribute (CancellationToken)
        is_cancelled = getattr(stop_flag, "is_cancelled", None)
        if is_cancelled is not None:
            # Could be a property (returns bool) or method (needs to be called)
            if callable(is_cancelled):
                try:
                    return bool(is_cancelled())
                except TypeError:
                    pass
            else:
                return bool(is_cancelled)
        
        # Case 3: Fallback - try to use it as boolean
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
        stop_flag: Callable = None
    ) -> Tuple[Dict, float]:
        """Train a single model"""
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG.LEARNING_RATE,
            weight_decay=0.01
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
                if patience >= CONFIG.EARLY_STOP_PATIENCE:
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
        """Get ensemble prediction with uncertainty quantification."""
        if X.ndim == 2:
            X = X[np.newaxis, :]
        
        with self._lock:
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            all_probs = {}
            
            for name, model in self.models.items():
                model.eval()
                with torch.no_grad():
                    logits, _ = model(X_tensor)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                    all_probs[name] = probs
            
            # Weighted average
            weighted_probs = np.zeros(CONFIG.NUM_CLASSES)
            
            for name, probs in all_probs.items():
                weight = self.weights.get(name, 1.0 / len(self.models))
                weighted_probs += probs * weight
            
            weighted_probs = weighted_probs / (weighted_probs.sum() + 1e-8)
            
            predicted_class = int(np.argmax(weighted_probs))
            confidence = float(np.max(weighted_probs))
            
            # Entropy
            entropy = -np.sum(weighted_probs * np.log(weighted_probs + 1e-8))
            max_entropy = np.log(CONFIG.NUM_CLASSES)
            normalized_entropy = entropy / max_entropy
            
            # Agreement
            predictions = [np.argmax(p) for p in all_probs.values()]
            if predictions:
                most_common = max(set(predictions), key=predictions.count)
                agreement = predictions.count(most_common) / len(predictions)
            else:
                agreement = 0.0
            
            return EnsemblePrediction(
                probabilities=weighted_probs,
                predicted_class=predicted_class,
                confidence=confidence,
                entropy=normalized_entropy,
                agreement=agreement,
                individual_predictions=all_probs
            )
    
    def predict_batch(self, X: np.ndarray) -> List[EnsemblePrediction]:
        """
        Predict multiple samples efficiently.
        
        Uses batched forward passes for all models to avoid
        per-sample overhead.
        """
        if len(X) == 0:
            return []
        
        with self._lock:
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Get all predictions at once for each model
            all_model_probs = {}
            
            for name, model in self.models.items():
                model.eval()
                with torch.no_grad():
                    logits, _ = model(X_tensor)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    all_model_probs[name] = probs
            
            # Build predictions for each sample
            predictions = []
            
            for i in range(len(X)):
                # Get probs for this sample from each model
                sample_probs = {name: probs[i] for name, probs in all_model_probs.items()}
                
                # Weighted average
                weighted_probs = np.zeros(CONFIG.NUM_CLASSES)
                for name, probs in sample_probs.items():
                    weight = self.weights.get(name, 1.0 / len(self.models))
                    weighted_probs += probs * weight
                
                weighted_probs = weighted_probs / (weighted_probs.sum() + 1e-8)
                
                predicted_class = int(np.argmax(weighted_probs))
                confidence = float(np.max(weighted_probs))
                
                # Entropy
                entropy = -np.sum(weighted_probs * np.log(weighted_probs + 1e-8))
                max_entropy = np.log(CONFIG.NUM_CLASSES)
                normalized_entropy = entropy / max_entropy
                
                # Agreement
                model_predictions = [np.argmax(p) for p in sample_probs.values()]
                if model_predictions:
                    most_common = max(set(model_predictions), key=model_predictions.count)
                    agreement = model_predictions.count(most_common) / len(model_predictions)
                else:
                    agreement = 0.0
                
                predictions.append(EnsemblePrediction(
                    probabilities=weighted_probs,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    entropy=normalized_entropy,
                    agreement=agreement,
                    individual_predictions=sample_probs
                ))
            
            return predictions
    
    def save(self, path: str = None):
        """Save ensemble to file"""
        path = path or str(CONFIG.MODEL_DIR / "ensemble.pt")
        
        with self._lock:
            state = {
                'input_size': self.input_size,
                'model_names': list(self.models.keys()),
                'models': {name: model.state_dict() for name, model in self.models.items()},
                'weights': self.weights,
            }
            
            torch.save(state, path)
        
        log.info(f"Ensemble saved to {path}")
    
    # models/ensemble.py - FIXED load() method ending

    def load(self, path: str = None) -> bool:
        """Load ensemble from file"""
        path = path or str(CONFIG.MODEL_DIR / "ensemble.pt")
        
        if not Path(path).exists():
            log.warning(f"No saved model at {path}")
            return False
        
        try:
            with self._lock:
                state = torch.load(path, map_location=self.device)
                
                self.input_size = state['input_size']
                
                model_names = state.get('model_names', list(state['models'].keys()))
                
                self.models = {}
                self.weights = {}
                
                for name in model_names:
                    if name in self.MODEL_CLASSES and name in state['models']:
                        self._init_model(name)
                        self.models[name].load_state_dict(state['models'][name])
                        self.models[name].eval()
                
                saved_weights = state.get('weights', {})
                for name in self.models.keys():
                    self.weights[name] = saved_weights.get(name, 1.0)
                
                self._normalize_weights()
            
            log.info(f"Ensemble loaded: {list(self.models.keys())}")
            return True
            
        except Exception as e:
            log.error(f"Failed to load ensemble: {e}")
            return False