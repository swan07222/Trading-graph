"""
Advanced Ensemble Model with Dynamic Weighting and Uncertainty Calibration
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

from .advanced_networks import (
    AdvancedLSTMModel, AdvancedTransformerModel, 
    MambaModel, HybridMambaTransformer
)
from .networks import GRUModel, TCNModel
from config import CONFIG
from utils.logger import log


@dataclass
class AdvancedPrediction:
    """Enhanced prediction with full uncertainty quantification"""
    probabilities: np.ndarray
    predicted_class: int
    confidence: float
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    agreement: float
    calibrated_confidence: float
    individual_predictions: Dict[str, np.ndarray]
    prediction_interval: Tuple[float, float]  # 95% CI
    reliability_score: float  # How reliable is this prediction?
    
    @property
    def should_trade(self) -> bool:
        """Whether conditions are good enough to trade"""
        return (
            self.calibrated_confidence >= CONFIG.MIN_CONFIDENCE and
            self.reliability_score >= 0.6 and
            self.agreement >= 0.5 and
            self.epistemic_uncertainty < 0.4
        )
    
    @property
    def prob_up(self) -> float:
        return float(self.probabilities[2])
    
    @property
    def prob_down(self) -> float:
        return float(self.probabilities[0])


class ConfidenceCalibrator:
    """
    Calibrates model confidence using temperature scaling
    Ensures confidence reflects true accuracy
    """
    def __init__(self):
        self.temperature = 1.0
        self.calibrated = False
    
    def fit(self, logits: np.ndarray, labels: np.ndarray):
        """Fit temperature parameter on validation set"""
        from scipy.optimize import minimize_scalar
        
        def nll_loss(T):
            scaled = logits / T
            probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
            log_probs = np.log(probs[np.arange(len(labels)), labels] + 1e-8)
            return -log_probs.mean()
        
        result = minimize_scalar(nll_loss, bounds=(0.1, 10), method='bounded')
        self.temperature = result.x
        self.calibrated = True
        log.info(f"Calibration temperature: {self.temperature:.3f}")
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling"""
        if not self.calibrated:
            return logits
        return logits / self.temperature


class DynamicWeightOptimizer:
    """
    Dynamically adjusts model weights based on recent performance
    Uses online learning to adapt to market regime changes
    """
    def __init__(self, model_names: List[str], learning_rate: float = 0.1):
        self.model_names = model_names
        self.lr = learning_rate
        
        # Initialize uniform weights
        self.weights = {name: 1.0 / len(model_names) for name in model_names}
        
        # Performance tracking (exponential moving average)
        self.accuracy_ema = {name: 0.5 for name in model_names}
        self.ema_alpha = 0.1
        
        # Confidence calibration per model
        self.calibration_error = {name: 0.0 for name in model_names}
    
    def update(self, predictions: Dict[str, int], confidences: Dict[str, float], 
               true_label: int):
        """Update weights based on prediction outcome"""
        for name in self.model_names:
            # Was this model correct?
            correct = 1.0 if predictions[name] == true_label else 0.0
            
            # Update accuracy EMA
            self.accuracy_ema[name] = (
                self.ema_alpha * correct + 
                (1 - self.ema_alpha) * self.accuracy_ema[name]
            )
            
            # Update calibration error
            conf = confidences[name]
            self.calibration_error[name] = (
                self.ema_alpha * abs(conf - correct) +
                (1 - self.ema_alpha) * self.calibration_error[name]
            )
        
        # Recompute weights
        self._update_weights()
    
    def _update_weights(self):
        """Update model weights based on performance"""
        # Score = accuracy * (1 - calibration_error)
        scores = {}
        for name in self.model_names:
            acc = self.accuracy_ema[name]
            cal_err = self.calibration_error[name]
            scores[name] = acc * (1 - cal_err) + 0.01  # Small epsilon
        
        # Normalize to sum to 1
        total = sum(scores.values())
        self.weights = {name: score / total for name, score in scores.items()}
    
    def get_weights(self) -> Dict[str, float]:
        return self.weights.copy()


class AdvancedEnsembleModel:
    """
    State-of-the-art ensemble combining:
    - Multiple advanced architectures
    - Dynamic weight optimization
    - Confidence calibration
    - Uncertainty quantification
    - Monte Carlo dropout for epistemic uncertainty
    """
    
    MODEL_CLASSES = {
        'advanced_lstm': AdvancedLSTMModel,
        'advanced_transformer': AdvancedTransformerModel,
        'mamba': MambaModel,
        'hybrid': HybridMambaTransformer,
        'gru': GRUModel,
        'tcn': TCNModel,
    }
    
    DEFAULT_MODELS = ['advanced_lstm', 'advanced_transformer', 'mamba', 'hybrid']
    
    def __init__(self, input_size: int, model_names: List[str] = None):
        self.input_size = input_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model_names = model_names or self.DEFAULT_MODELS
        
        self.models: Dict[str, nn.Module] = {}
        self.calibrators: Dict[str, ConfidenceCalibrator] = {}
        
        for name in model_names:
            if name in self.MODEL_CLASSES:
                self._init_model(name)
                self.calibrators[name] = ConfidenceCalibrator()
        
        self.weight_optimizer = DynamicWeightOptimizer(list(self.models.keys()))
        
        # Global calibrator for ensemble
        self.ensemble_calibrator = ConfidenceCalibrator()
        
        log.info(f"Advanced ensemble initialized with {len(self.models)} models on {self.device}")
    
    def _init_model(self, name: str):
        """Initialize a model"""
        try:
            model_class = self.MODEL_CLASSES[name]
            
            # Model-specific parameters
            if 'transformer' in name or name == 'hybrid':
                model = model_class(
                    input_size=self.input_size,
                    hidden_size=CONFIG.HIDDEN_SIZE,
                    num_classes=CONFIG.NUM_CLASSES,
                    dropout=CONFIG.DROPOUT
                )
            elif name == 'mamba':
                model = model_class(
                    input_size=self.input_size,
                    hidden_size=CONFIG.HIDDEN_SIZE,
                    num_layers=6,
                    num_classes=CONFIG.NUM_CLASSES,
                    dropout=CONFIG.DROPOUT
                )
            else:
                model = model_class(
                    input_size=self.input_size,
                    hidden_size=CONFIG.HIDDEN_SIZE,
                    num_classes=CONFIG.NUM_CLASSES,
                    dropout=CONFIG.DROPOUT
                )
            
            model.to(self.device)
            self.models[name] = model
            log.info(f"Initialized {name}")
            
        except Exception as e:
            log.error(f"Failed to init {name}: {e}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = None,
        batch_size: int = None,
        callback=None
    ) -> Dict:
        """Train all models with advanced techniques"""
        epochs = epochs or CONFIG.EPOCHS
        batch_size = batch_size or CONFIG.BATCH_SIZE
        
        # Create data loaders
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
        
        # Class weights
        class_counts = np.bincount(y_train, minlength=CONFIG.NUM_CLASSES)
        weights = 1.0 / (class_counts + 1)
        weights = weights / weights.sum()
        class_weights = torch.FloatTensor(weights).to(self.device)
        
        history = {}
        
        for name, model in self.models.items():
            log.info(f"Training {name}...")
            model_history = self._train_model(
                model, name, train_loader, val_loader,
                class_weights, epochs, callback
            )
            history[name] = model_history
            
            # Calibrate confidence
            self._calibrate_model(name, model, val_loader)
        
        # Calibrate ensemble
        self._calibrate_ensemble(X_val, y_val)
        
        return history
    
    def _train_model(
        self,
        model: nn.Module,
        name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        epochs: int,
        callback=None
    ) -> Dict:
        """Train single model with advanced techniques"""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG.LEARNING_RATE,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        
        # Label smoothing cross entropy
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1
        )
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0
        patience = 0
        best_state = None
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                logits, conf = model(batch_X)
                loss = criterion(logits, batch_y)
                
                # Add auxiliary losses if model has them
                if hasattr(model, 'aux_loss'):
                    loss = loss + model.aux_loss
                
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
                    
                    logits, conf = model(batch_X)
                    loss = criterion(logits, batch_y)
                    val_losses.append(loss.item())
                    
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == batch_y).sum().item()
                    total += len(batch_y)
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            val_acc = correct / total
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
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
        
        # Load best model
        if best_state:
            model.load_state_dict(best_state)
            model.to(self.device)
        
        log.info(f"{name} complete. Best: {best_val_acc:.2%}")
        return history
    
    def _calibrate_model(self, name: str, model: nn.Module, val_loader: DataLoader):
        """Calibrate model confidence"""
        model.eval()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                logits, _ = model(batch_X)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(batch_y.numpy())
        
        logits = np.concatenate(all_logits)
        labels = np.concatenate(all_labels)
        
        self.calibrators[name].fit(logits, labels)
    
    def _calibrate_ensemble(self, X_val: np.ndarray, y_val: np.ndarray):
        """Calibrate ensemble predictions"""
        # Get ensemble logits
        ensemble_logits = []
        
        for i in range(len(X_val)):
            X = X_val[i:i+1]
            pred = self._get_raw_ensemble_logits(X)
            ensemble_logits.append(pred)
        
        logits = np.array(ensemble_logits)
        self.ensemble_calibrator.fit(logits, y_val)
    
    def _get_raw_ensemble_logits(self, X: np.ndarray) -> np.ndarray:
        """Get raw ensemble logits (before calibration)"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        weights = self.weight_optimizer.get_weights()
        
        weighted_logits = np.zeros(CONFIG.NUM_CLASSES)
        
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                logits, _ = model(X_tensor)
                logits = logits.cpu().numpy()[0]
                
                # Apply individual calibration
                logits = self.calibrators[name].calibrate(logits)
                
                weighted_logits += weights[name] * logits
        
        return weighted_logits
    
    def predict(self, X: np.ndarray, num_mc_samples: int = 10) -> AdvancedPrediction:
        """
        Get ensemble prediction with full uncertainty quantification
        
        Uses Monte Carlo dropout for epistemic uncertainty
        """
        if X.ndim == 2:
            X = X[np.newaxis, :]
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        weights = self.weight_optimizer.get_weights()
        
        individual_probs = {}
        individual_confs = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                logits, conf = model(X_tensor)
                
                # Apply calibration
                logits = logits.cpu().numpy()[0]
                logits = self.calibrators[name].calibrate(logits)
                
                probs = np.exp(logits) / np.exp(logits).sum()
                individual_probs[name] = probs
                individual_confs[name] = float(conf.cpu().numpy()[0])
        
        # Monte Carlo dropout for epistemic uncertainty
        mc_predictions = []
        for _ in range(num_mc_samples):
            mc_logits = np.zeros(CONFIG.NUM_CLASSES)
            
            for name, model in self.models.items():
                model.train()  # Enable dropout
                with torch.no_grad():
                    logits, _ = model(X_tensor)
                    logits = logits.cpu().numpy()[0]
                    mc_logits += weights[name] * logits
            
            mc_probs = np.exp(mc_logits) / np.exp(mc_logits).sum()
            mc_predictions.append(mc_probs)
        
        # Reset to eval mode
        for model in self.models.values():
            model.eval()
        
        mc_predictions = np.array(mc_predictions)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = mc_predictions.std(axis=0).mean()
        
        # Mean prediction
        mean_probs = mc_predictions.mean(axis=0)
        
        # Apply ensemble calibration
        calibrated_logits = self.ensemble_calibrator.calibrate(np.log(mean_probs + 1e-8))
        calibrated_probs = np.exp(calibrated_logits) / np.exp(calibrated_logits).sum()
        
        # Aleatoric uncertainty (inherent data uncertainty)
        entropy = -np.sum(calibrated_probs * np.log(calibrated_probs + 1e-8))
        aleatoric_uncertainty = entropy / np.log(CONFIG.NUM_CLASSES)  # Normalize
        
        # Model agreement
        predictions = [np.argmax(p) for p in individual_probs.values()]
        most_common = max(set(predictions), key=predictions.count)
        agreement = predictions.count(most_common) / len(predictions)
        
        # Confidence (weighted average)
        weighted_conf = sum(
            individual_confs[name] * weights[name]
            for name in individual_confs
        )
        
        # Calibrated confidence
        calibrated_conf = weighted_conf * (1 - epistemic_uncertainty) * agreement
        
        # Prediction interval (95% CI using MC samples)
        predicted_class = np.argmax(calibrated_probs)
        class_predictions = mc_predictions[:, predicted_class]
        ci_low, ci_high = np.percentile(class_predictions, [2.5, 97.5])
        
        # Reliability score
        reliability = (
            0.3 * agreement +
            0.3 * (1 - epistemic_uncertainty) +
            0.2 * calibrated_conf +
            0.2 * (1 - aleatoric_uncertainty)
        )
        
        return AdvancedPrediction(
            probabilities=calibrated_probs,
            predicted_class=int(predicted_class),
            confidence=weighted_conf,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            agreement=agreement,
            calibrated_confidence=calibrated_conf,
            individual_predictions=individual_probs,
            prediction_interval=(ci_low, ci_high),
            reliability_score=reliability
        )
    
    def update_weights_from_outcome(
        self, 
        X: np.ndarray, 
        true_label: int
    ):
        """Update model weights based on actual outcome (online learning)"""
        if X.ndim == 2:
            X = X[np.newaxis, :]
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                logits, conf = model(X_tensor)
                pred = torch.argmax(logits, dim=-1).item()
                predictions[name] = pred
                confidences[name] = float(conf.cpu().numpy()[0])
        
        self.weight_optimizer.update(predictions, confidences, true_label)
    
    def save(self, path: str = None):
        """Save ensemble"""
        path = path or str(CONFIG.MODEL_DIR / "advanced_ensemble.pt")
        
        state = {
            'input_size': self.input_size,
            'models': {name: model.state_dict() for name, model in self.models.items()},
            'weights': self.weight_optimizer.get_weights(),
            'calibrators': {
                name: cal.temperature 
                for name, cal in self.calibrators.items()
            },
            'ensemble_calibrator': self.ensemble_calibrator.temperature
        }
        
        torch.save(state, path)
        log.info(f"Saved to {path}")
    
    def load(self, path: str = None) -> bool:
        """Load ensemble"""
        path = path or str(CONFIG.MODEL_DIR / "advanced_ensemble.pt")
        
        if not Path(path).exists():
            return False
        
        try:
            state = torch.load(path, map_location=self.device)
            
            self.input_size = state['input_size']
            
            for name, model_state in state['models'].items():
                if name in self.models:
                    self.models[name].load_state_dict(model_state)
                    self.models[name].to(self.device)
                    self.models[name].eval()
            
            if 'weights' in state:
                self.weight_optimizer.weights = state['weights']
            
            if 'calibrators' in state:
                for name, temp in state['calibrators'].items():
                    if name in self.calibrators:
                        self.calibrators[name].temperature = temp
                        self.calibrators[name].calibrated = True
            
            if 'ensemble_calibrator' in state:
                self.ensemble_calibrator.temperature = state['ensemble_calibrator']
                self.ensemble_calibrator.calibrated = True
            
            log.info(f"Loaded from {path}")
            return True
            
        except Exception as e:
            log.error(f"Load failed: {e}")
            return False