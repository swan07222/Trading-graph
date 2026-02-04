"""
Professional Training Pipeline with Best Practices
Implements state-of-the-art training techniques for maximum performance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json
import copy

from config import CONFIG
from data.fetcher import DataFetcher
from data.processor import DataProcessor
from data.advanced_features import AdvancedFeatureEngine
from models.advanced_networks import (
    AdvancedLSTMModel, AdvancedTransformerModel,
    MambaModel, HybridMambaTransformer
)
from models.networks import GRUModel, TCNModel
from models.uncertainty import (
    UncertaintyQuantifier, TemperatureScaling,
    CalibrationMetrics, EvidentialUncertainty
)
from utils.logger import log


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic
    epochs: int = 150
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    
    # Scheduler
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    
    # Regularization
    dropout: float = 0.3
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 0.0
    
    # Training tricks
    gradient_clip: float = 1.0
    use_amp: bool = True  # Automatic mixed precision
    use_ema: bool = True  # Exponential moving average
    ema_decay: float = 0.999
    
    # Early stopping
    patience: int = 20
    min_delta: float = 0.001
    
    # Ensemble
    num_ensemble: int = 5  # Number of models to train
    
    # Data augmentation
    use_augmentation: bool = True
    noise_std: float = 0.01
    time_warp_prob: float = 0.1
    
    # Uncertainty
    use_evidential: bool = False
    mc_dropout_samples: int = 30


class EMAModel:
    """Exponential Moving Average of model weights"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = copy.deepcopy(model)
        self.decay = decay
        
        for param in self.model.parameters():
            param.requires_grad_(False)
    
    def update(self, model: nn.Module):
        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class DataAugmentor:
    """Data augmentation for time series"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.config.use_augmentation:
            return x, y
        
        # Gaussian noise
        if self.config.noise_std > 0:
            noise = torch.randn_like(x) * self.config.noise_std
            x = x + noise
        
        # Time warping (simplified)
        if np.random.random() < self.config.time_warp_prob:
            x = self._time_warp(x)
        
        # Mixup
        if self.config.mixup_alpha > 0:
            x, y = self._mixup(x, y)
        
        return x, y
    
    def _time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Simple time warping by interpolation"""
        batch, seq_len, features = x.shape
        
        # Random stretch/compress factor
        factor = np.random.uniform(0.9, 1.1)
        new_len = int(seq_len * factor)
        
        # Interpolate
        x = x.transpose(1, 2)  # (batch, features, seq)
        x = F.interpolate(x, size=seq_len, mode='linear', align_corners=False)
        x = x.transpose(1, 2)  # (batch, seq, features)
        
        return x
    
    def _mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixup data augmentation"""
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        
        # For classification, we need to handle labels specially
        # Here we just return original labels (simplified)
        return mixed_x, y


class CosineWarmupScheduler:
    """Cosine annealing with warmup"""
    
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, 
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class ProfessionalTrainer:
    """
    Professional-grade training pipeline
    
    Features:
    - Multi-model ensemble training
    - Mixed precision training (AMP)
    - Exponential moving average (EMA)
    - Advanced data augmentation
    - Cosine warmup learning rate
    - Gradient clipping
    - Label smoothing
    - Uncertainty calibration
    - Comprehensive logging
    """
    
    MODEL_REGISTRY = {
        'advanced_lstm': AdvancedLSTMModel,
        'advanced_transformer': AdvancedTransformerModel,
        'mamba': MambaModel,
        'hybrid': HybridMambaTransformer,
        'gru': GRUModel,
        'tcn': TCNModel,
    }
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.fetcher = DataFetcher()
        self.processor = DataProcessor()
        self.feature_engine = AdvancedFeatureEngine()
        
        # Training state
        self.models: Dict[str, nn.Module] = {}
        self.ema_models: Dict[str, EMAModel] = {}
        self.calibrators: Dict[str, TemperatureScaling] = {}
        self.history: Dict[str, List] = {}
        
        # Mixed precision
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Data augmentation
        self.augmentor = DataAugmentor(self.config)
    
    def prepare_data(
        self,
        stock_codes: List[str] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data with advanced features"""
        stocks = stock_codes or CONFIG.STOCK_POOL
        
        log.info(f"Preparing data for {len(stocks)} stocks...")
        
        all_X, all_y, all_r = [], [], []
        
        for i, code in enumerate(stocks):
            if verbose:
                log.info(f"Processing {code} ({i+1}/{len(stocks)})")
            
            try:
                df = self.fetcher.get_history(code, days=2000)
                
                if len(df) < CONFIG.SEQUENCE_LENGTH + 50:
                    continue
                
                # Advanced features
                df = self.feature_engine.create_features(df)
                df = self.processor.create_labels(df)
                
                feature_cols = self.feature_engine.get_feature_columns()
                X, y, r = self.processor.prepare_sequences(df, feature_cols)
                
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
                    all_r.append(r)
                    
            except Exception as e:
                log.error(f"Error processing {code}: {e}")
        
        if not all_X:
            raise ValueError("No data available")
        
        X = np.concatenate(all_X)
        y = np.concatenate(all_y)
        r = np.concatenate(all_r)
        
        # Shuffle
        idx = np.random.permutation(len(X))
        X, y, r = X[idx], y[idx], r[idx]
        
        # Split
        n = len(X)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        log.info(f"Data: Train={train_end}, Val={val_end-train_end}, Test={n-val_end}")
        log.info(f"Classes: DOWN={np.sum(y==0)}, NEUTRAL={np.sum(y==1)}, UP={np.sum(y==2)}")
        
        return (
            X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:]
        )
    
    def train(
        self,
        stock_codes: List[str] = None,
        model_names: List[str] = None,
        callback: Callable = None
    ) -> Dict:
        """
        Full training pipeline
        """
        model_names = model_names or ['advanced_lstm', 'advanced_transformer', 'mamba', 'hybrid']
        
        log.info("=" * 70)
        log.info("PROFESSIONAL TRAINING PIPELINE")
        log.info("=" * 70)
        log.info(f"Device: {self.device}")
        log.info(f"Models: {model_names}")
        log.info(f"Epochs: {self.config.epochs}")
        log.info(f"Mixed Precision: {self.config.use_amp}")
        log.info("=" * 70)
        
        # Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(stock_codes)
        
        input_size = X_train.shape[2]
        
        # Create data loaders
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
        test_loader = self._create_dataloader(X_test, y_test, shuffle=False)
        
        # Class weights
        class_counts = np.bincount(y_train, minlength=CONFIG.NUM_CLASSES)
        weights = 1.0 / (class_counts + 1)
        weights = weights / weights.sum()
        class_weights = torch.FloatTensor(weights).to(self.device)
        
        # Train each model
        all_history = {}
        best_models = {}
        
        for name in model_names:
            log.info(f"\n{'='*60}")
            log.info(f"Training {name.upper()}")
            log.info(f"{'='*60}")
            
            model = self._create_model(name, input_size)
            history, best_state = self._train_single_model(
                model, name, train_loader, val_loader,
                class_weights, callback
            )
            
            all_history[name] = history
            best_models[name] = best_state
            
            # Store model
            self.models[name] = model
        
        # Calibrate models
        log.info("\n" + "="*60)
        log.info("CALIBRATING MODELS")
        log.info("="*60)
        
        for name, model in self.models.items():
            self._calibrate_model(name, model, val_loader)
        
        # Evaluate on test set
        log.info("\n" + "="*60)
        log.info("EVALUATING ON TEST SET")
        log.info("="*60)
        
        test_metrics = self._evaluate(test_loader, y_test)
        
        # Save models
        self._save_models()
        
        # Compile results
        best_accuracy = max(
            max(h['val_acc']) if h['val_acc'] else 0
            for h in all_history.values()
        )
        
        results = {
            'history': all_history,
            'best_accuracy': best_accuracy,
            'test_metrics': test_metrics,
            'input_size': input_size,
            'num_models': len(self.models),
            'calibration_temps': {
                name: cal.temperature 
                for name, cal in self.calibrators.items()
            }
        }
        
        log.info("\n" + "="*60)
        log.info(f"TRAINING COMPLETE!")
        log.info(f"Best Accuracy: {best_accuracy:.2%}")
        log.info(f"Test Accuracy: {test_metrics['accuracy']:.2%}")
        log.info("="*60)
        
        return results
    
    def _create_model(self, name: str, input_size: int) -> nn.Module:
        """Create model instance"""
        model_class = self.MODEL_REGISTRY.get(name)
        
        if model_class is None:
            raise ValueError(f"Unknown model: {name}")
        
        model = model_class(
            input_size=input_size,
            hidden_size=CONFIG.HIDDEN_SIZE,
            num_classes=CONFIG.NUM_CLASSES,
            dropout=self.config.dropout
        )
        
        return model.to(self.device)
    
    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, 
                          shuffle: bool = True) -> DataLoader:
        """Create data loader with optional weighted sampling"""
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.LongTensor(y)
        )
        
        if shuffle:
            # Weighted sampling for class balance
            class_counts = np.bincount(y, minlength=CONFIG.NUM_CLASSES)
            weights = 1.0 / (class_counts + 1)
            sample_weights = weights[y]
            
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(y),
                replacement=True
            )
            
            return DataLoader(
                dataset, 
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=0,
                pin_memory=True
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
    
    def _train_single_model(
        self,
        model: nn.Module,
        name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        callback: Callable = None
    ) -> Tuple[Dict, Dict]:
        """Train a single model with all optimizations"""
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=self.config.warmup_epochs,
            total_epochs=self.config.epochs,
            min_lr=self.config.min_lr
        )
        
        # Loss with label smoothing
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=self.config.label_smoothing
        )
        
        # EMA
        ema = EMAModel(model, self.config.ema_decay) if self.config.use_ema else None
        
        # Training state
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        best_val_acc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training
            model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Data augmentation
                batch_X, batch_y = self.augmentor(batch_X, batch_y)
                
                optimizer.zero_grad()
                
                # Mixed precision
                if self.config.use_amp:
                    with autocast():
                        logits, conf = model(batch_X)
                        loss = criterion(logits, batch_y)
                        
                        if hasattr(model, 'aux_loss'):
                            loss = loss + model.aux_loss
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    logits, conf = model(batch_X)
                    loss = criterion(logits, batch_y)
                    
                    if hasattr(model, 'aux_loss'):
                        loss = loss + model.aux_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    optimizer.step()
                
                train_losses.append(loss.item())
                
                # Update EMA
                if ema:
                    ema.update(model)
            
            # Update scheduler
            scheduler.step(epoch)
            
            # Validation
            val_loss, val_acc = self._validate(model, val_loader, criterion)
            
            # Record history
            history['train_loss'].append(np.mean(train_losses))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Best model
            if val_acc > best_val_acc + self.config.min_delta:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best state (use EMA if available)
                if ema:
                    best_state = {k: v.cpu().clone() for k, v in ema.state_dict().items()}
                else:
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            # Callback
            if callback:
                callback(name, epoch, val_acc)
            
            # Logging
            if (epoch + 1) % 10 == 0:
                log.info(
                    f"{name} Epoch {epoch+1}/{self.config.epochs}: "
                    f"train_loss={history['train_loss'][-1]:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}, "
                    f"lr={history['lr'][-1]:.2e}"
                )
            
            # Early stopping
            if patience_counter >= self.config.patience:
                log.info(f"{name}: Early stopping at epoch {epoch+1}")
                break
        
        # Load best weights
        if best_state:
            model.load_state_dict(best_state)
            model.to(self.device)
        
        # Store EMA model
        if ema:
            self.ema_models[name] = ema
        
        log.info(f"{name} completed. Best accuracy: {best_val_acc:.2%}")
        
        return history, best_state
    
    def _validate(self, model: nn.Module, val_loader: DataLoader, 
                  criterion: nn.Module) -> Tuple[float, float]:
        """Validate model"""
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
        
        return np.mean(val_losses), correct / total
    
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
        
        # Temperature scaling
        calibrator = TemperatureScaling()
        calibrator.fit(logits, labels)
        self.calibrators[name] = calibrator
        
        # Compute calibration metrics
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        calibrated_probs = calibrator.calibrate_probs(probs)
        
        ece_before = CalibrationMetrics.expected_calibration_error(probs, labels)
        ece_after = CalibrationMetrics.expected_calibration_error(calibrated_probs, labels)
        
        log.info(f"{name} calibration: ECE {ece_before:.4f} → {ece_after:.4f}")
    
    def _evaluate(self, test_loader: DataLoader, y_test: np.ndarray) -> Dict:
        """Evaluate ensemble on test set"""
        all_probs = []
        
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(self.device)
            
            batch_probs = []
            for name, model in self.models.items():
                model.eval()
                with torch.no_grad():
                    logits, _ = model(batch_X)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    
                    # Apply calibration
                    if name in self.calibrators:
                        probs = self.calibrators[name].calibrate_probs(probs)
                    
                    batch_probs.append(probs)
            
            # Average ensemble
            ensemble_probs = np.mean(batch_probs, axis=0)
            all_probs.append(ensemble_probs)
        
        probs = np.concatenate(all_probs)
        preds = probs.argmax(axis=1)
        
        # Metrics
        accuracy = (preds == y_test).mean()
        
        # Per-class
        class_acc = {}
        for c in range(CONFIG.NUM_CLASSES):
            mask = y_test == c
            if mask.sum() > 0:
                class_acc[c] = (preds[mask] == c).mean()
        
        # Calibration
        ece = CalibrationMetrics.expected_calibration_error(probs, y_test)
        brier = CalibrationMetrics.brier_score(probs, y_test)
        
        log.info(f"Test Accuracy: {accuracy:.2%}")
        log.info(f"Class Accuracy: DOWN={class_acc.get(0,0):.2%}, "
                f"NEUTRAL={class_acc.get(1,0):.2%}, UP={class_acc.get(2,0):.2%}")
        log.info(f"ECE: {ece:.4f}, Brier: {brier:.4f}")
        
        return {
            'accuracy': accuracy,
            'class_accuracy': class_acc,
            'ece': ece,
            'brier': brier
        }
    
    def _save_models(self):
        """Save all models and calibrators"""
        save_path = CONFIG.MODEL_DIR / "professional_ensemble.pt"
        
        state = {
            'input_size': list(self.models.values())[0].input_size if self.models else 0,
            'models': {
                name: model.state_dict() 
                for name, model in self.models.items()
            },
            'ema_models': {
                name: ema.state_dict()
                for name, ema in self.ema_models.items()
            },
            'calibrators': {
                name: cal.temperature
                for name, cal in self.calibrators.items()
            },
            'config': {
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
            },
            'saved_at': datetime.now().isoformat()
        }
        
        torch.save(state, save_path)
        log.info(f"Models saved to {save_path}")