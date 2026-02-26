# models/training_utils.py
"""Enhanced training utilities for improved model training.

This module provides:
- Advanced learning rate schedulers
- Improved early stopping mechanisms
- Gradient clipping and normalization
- Mixed precision training helpers
- Training progress tracking
- Memory optimization utilities
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

from utils.logger import get_logger

log = get_logger(__name__)


class EarlyStoppingMode(Enum):
    """Early stopping operation modes."""
    MINIMIZE = "minimize"  # For loss
    MAXIMIZE = "maximize"  # For accuracy/metrics


@dataclass
class TrainingMetrics:
    """Container for training metrics with rolling statistics."""
    
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_metrics: list[float] = field(default_factory=list)
    val_metrics: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)
    
    # Rolling statistics for real-time monitoring
    rolling_window: int = 10
    _rolling_losses: deque = field(default_factory=lambda: deque(maxlen=10))
    _rolling_metrics: deque = field(default_factory=lambda: deque(maxlen=10))
    
    @property
    def current_epoch(self) -> int:
        return len(self.train_losses)
    
    @property
    def best_train_loss(self) -> float:
        return min(self.train_losses) if self.train_losses else float('inf')
    
    @property
    def best_val_loss(self) -> float:
        return min(self.val_losses) if self.val_losses else float('inf')
    
    @property
    def best_val_metric(self) -> float:
        return max(self.val_metrics) if self.val_metrics else 0.0
    
    @property
    def rolling_loss_avg(self) -> float:
        return np.mean(self._rolling_losses) if self._rolling_losses else float('inf')
    
    @property
    def rolling_metric_avg(self) -> float:
        return np.mean(self._rolling_metrics) if self._rolling_metrics else 0.0
    
    @property
    def loss_improvement_rate(self) -> float:
        """Calculate rate of loss improvement over recent epochs."""
        if len(self.train_losses) < 5:
            return 0.0
        recent = self.train_losses[-5:]
        if recent[0] == 0:
            return 0.0
        return (recent[0] - recent[-1]) / recent[0]
    
    @property
    def overfitting_score(self) -> float:
        """Measure overfitting as gap between train and val performance."""
        if not self.train_losses or not self.val_losses:
            return 0.0
        return self.val_losses[-1] - self.train_losses[-1]
    
    def add_epoch_results(
        self,
        train_loss: float,
        val_loss: float,
        train_metric: float = 0.0,
        val_metric: float = 0.0,
        lr: float = 0.0,
        epoch_time: float = 0.0,
        grad_norm: float = 0.0,
    ) -> None:
        """Add results from a training epoch."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_metrics.append(train_metric)
        self.val_metrics.append(val_metric)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
        self.gradient_norms.append(grad_norm)
        
        self._rolling_losses.append(val_loss)
        self._rolling_metrics.append(val_metric)
    
    def to_summary_dict(self) -> dict[str, Any]:
        """Convert to summary dictionary for logging/saving."""
        return {
            "current_epoch": self.current_epoch,
            "best_train_loss": self.best_train_loss,
            "best_val_loss": self.best_val_loss,
            "best_val_metric": self.best_val_metric,
            "rolling_loss_avg": self.rolling_loss_avg,
            "rolling_metric_avg": self.rolling_metric_avg,
            "loss_improvement_rate": self.loss_improvement_rate,
            "overfitting_score": self.overfitting_score,
            "avg_epoch_time": np.mean(self.epoch_times) if self.epoch_times else 0.0,
            "avg_gradient_norm": np.mean(self.gradient_norms) if self.gradient_norms else 0.0,
        }


class AdvancedEarlyStopping:
    """Enhanced early stopping with multiple strategies.
    
    Features:
    - Patience-based stopping
    - Minimum delta threshold
    - Cooldown period after LR reduction
    - Rolling window averaging
    - Divergence detection
    """
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: EarlyStoppingMode = EarlyStoppingMode.MINIMIZE,
        cooldown: int = 5,
        min_epochs: int = 20,
        max_epochs: int = 500,
        divergence_threshold: float = 2.0,
        rolling_window: int = 5,
        verbose: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.cooldown = cooldown
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.divergence_threshold = divergence_threshold
        self.rolling_window = rolling_window
        self.verbose = verbose
        
        self.counter = 0
        self.cooldown_counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.divergence_detected = False
        
        self._value_history: deque = deque(maxlen=rolling_window)
    
    def __call__(self, value: float, epoch: int, learning_rate: float = None) -> bool:
        """Check if training should stop.
        
        Args:
            value: Current metric value (loss or accuracy)
            epoch: Current epoch number
            learning_rate: Current learning rate (for logging)
            
        Returns:
            True if training should stop
        """
        if epoch < self.min_epochs:
            self._update_history(value)
            return False
        
        if epoch >= self.max_epochs:
            if self.verbose:
                log.info(f"EarlyStopping: Max epochs ({self.max_epochs}) reached")
            return True
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self._update_history(value)
            return False
        
        # Check for divergence
        if self._check_divergence(value):
            self.divergence_detected = True
            if self.verbose:
                log.warning(
                    f"EarlyStopping: Divergence detected at epoch {epoch} "
                    f"(value={value:.6f})"
                )
            return True
        
        # Get smoothed value
        smoothed_value = self._get_smoothed_value()
        
        # Initialize or check improvement
        if self.best_score is None:
            self.best_score = smoothed_value
            self.best_epoch = epoch
        elif self._is_improvement(smoothed_value):
            self.best_score = smoothed_value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    log.info(
                        f"EarlyStopping: No improvement for {self.patience} epochs "
                        f"(best={self.best_score:.6f} at epoch {self.best_epoch})"
                    )
                return True
        
        self._update_history(value)
        return False
    
    def _is_improvement(self, value: float) -> bool:
        """Check if value is an improvement over best."""
        if self.mode == EarlyStoppingMode.MINIMIZE:
            return value < self.best_score - self.min_delta
        else:
            return value > self.best_score + self.min_delta
    
    def _check_divergence(self, value: float) -> bool:
        """Detect if training is diverging."""
        if len(self._value_history) < 3:
            return False
        
        recent_avg = np.mean(list(self._value_history)[-3:])
        if self.mode == EarlyStoppingMode.MINIMIZE:
            # Loss increasing rapidly
            if self.best_score and recent_avg > self.best_score * self.divergence_threshold:
                return True
        else:
            # Metric decreasing rapidly
            if self.best_score and recent_avg < self.best_score / self.divergence_threshold:
                return True
        return False
    
    def _update_history(self, value: float) -> None:
        """Update value history."""
        self._value_history.append(value)
    
    def _get_smoothed_value(self) -> float:
        """Get smoothed value from recent history."""
        if len(self._value_history) == 0:
            return 0.0
        return np.mean(list(self._value_history))
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.cooldown_counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.divergence_detected = False
        self._value_history.clear()


class GradientClipper:
    """Advanced gradient clipping strategies."""
    
    def __init__(
        self,
        strategy: str = "norm",
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        adaptive: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            strategy: Clipping strategy ('norm', 'value', 'adaptive')
            max_norm: Maximum gradient norm
            norm_type: Norm type (2.0 for L2)
            adaptive: Enable adaptive clipping
            verbose: Log clipping events
        """
        self.strategy = strategy
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.adaptive = adaptive
        self.verbose = verbose
        
        self._grad_history: deque = deque(maxlen=20)
        self._clip_count = 0
        self._total_steps = 0
    
    def clip(self, model: nn.Module) -> float:
        """Apply gradient clipping.
        
        Args:
            model: PyTorch model
            
        Returns:
            Gradient norm before clipping
        """
        self._total_steps += 1
        
        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(self.norm_type)
                total_norm += float(param_norm.item()) ** self.norm_type
        total_norm **= (1.0 / self.norm_type)
        
        self._grad_history.append(total_norm)
        
        # Determine clipping threshold
        clip_norm = self.max_norm
        if self.adaptive and len(self._grad_history) > 5:
            # Adaptive: clip at mean + 2*std of recent gradients
            recent = list(self._grad_history)[-20:]
            adaptive_norm = float(np.mean(recent) + 2 * np.std(recent))
            clip_norm = min(clip_norm, max(0.1, adaptive_norm))
        
        # Apply clipping
        if total_norm > clip_norm:
            self._clip_count += 1
            if self.strategy == "norm":
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=clip_norm,
                    norm_type=self.norm_type,
                )
            elif self.strategy == "value":
                nn.utils.clip_grad_value_(
                    model.parameters(),
                    clip_value=clip_norm,
                )
            
            if self.verbose and self._total_steps % 100 == 0:
                clip_ratio = self._clip_count / self._total_steps
                log.debug(
                    f"Gradient clipping: norm={total_norm:.4f}, "
                    f"clipped to {clip_norm:.4f}, clip_ratio={clip_ratio:.2%}"
                )
        
        return total_norm
    
    @property
    def clip_ratio(self) -> float:
        """Return ratio of clipped gradients."""
        if self._total_steps == 0:
            return 0.0
        return self._clip_count / self._total_steps


class LearningRateScheduler:
    """Enhanced learning rate scheduler with warm restarts."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        strategy: str = "cosine",
        base_lr: float = 1e-3,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        warmup_epochs: int = 5,
        patience: int = 10,
        factor: float = 0.5,
        T_max: int = 50,
        T_mult: int = 2,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            optimizer: PyTorch optimizer
            strategy: Scheduling strategy
            base_lr: Base learning rate
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate (for OneCycle)
            warmup_epochs: Warmup period
            patience: Patience for ReduceLROnPlateau
            factor: Reduction factor
            T_max: Maximum iterations for cosine annealing
            T_mult: Multiplication factor for restarts
            verbose: Log LR changes
        """
        self.optimizer = optimizer
        self.strategy = strategy
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose
        
        self._scheduler: Any = None
        self._epoch = 0
        self._best_score: float | None = None
        self._cooldown_counter = 0
        
        # Initialize scheduler based on strategy
        if strategy == "cosine":
            self._scheduler = CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=min_lr,
            )
        elif strategy == "onecycle":
            self._scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lr,
                epochs=T_max,
                steps_per_epoch=1,  # Manual stepping
            )
        elif strategy == "plateau":
            self._scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                verbose=verbose,
            )
        elif strategy == "warmup_cosine":
            # Custom warmup + cosine
            self._scheduler = None  # Handle manually
            self._warmup_multiplier = 0.0
        
        self._init_lr = base_lr
    
    def step(self, metric: float | None = None) -> float:
        """Step the scheduler.
        
        Args:
            metric: Metric value (for plateau scheduler)
            
        Returns:
            Current learning rate
        """
        self._epoch += 1
        
        if self.strategy == "warmup_cosine":
            self._step_warmup_cosine()
        elif self.strategy == "plateau":
            if metric is not None:
                self._scheduler.step(metric)
        elif self._scheduler is not None:
            self._scheduler.step()
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        if self.verbose and self._epoch % 10 == 0:
            log.debug(f"LR Scheduler: epoch={self._epoch}, lr={current_lr:.6f}")
        
        return current_lr
    
    def _step_warmup_cosine(self) -> None:
        """Custom warmup + cosine annealing step."""
        if self._epoch <= self.warmup_epochs:
            # Linear warmup
            progress = self._epoch / self.warmup_epochs
            lr = self.base_lr * progress
        else:
            # Cosine annealing
            progress = (self._epoch - self.warmup_epochs) / 100  # Assume 100 epochs
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def reset(self, new_base_lr: float | None = None) -> None:
        """Reset scheduler with optional new base LR."""
        if new_base_lr is not None:
            self.base_lr = new_base_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_base_lr
        
        if self._scheduler is not None:
            # Some schedulers support last_epoch parameter
            try:
                self._scheduler.last_epoch = 0
            except AttributeError:
                pass


@dataclass
class BatchProgress:
    """Progress tracking for batch processing."""
    
    total_batches: int
    current_batch: int = 0
    start_time: float = field(default_factory=time.time)
    losses: list[float] = field(default_factory=list)
    
    @property
    def progress(self) -> float:
        return self.current_batch / max(1, self.total_batches)
    
    @property
    def eta_seconds(self) -> float:
        if self.current_batch == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.current_batch
        remaining = self.total_batches - self.current_batch
        return remaining * avg_time
    
    @property
    def avg_loss(self) -> float:
        return np.mean(self.losses) if self.losses else 0.0
    
    def update(self, batch: int, loss: float) -> None:
        """Update batch progress."""
        self.current_batch = batch
        self.losses.append(loss)


def get_gradient_stats(model: nn.Module) -> dict[str, float]:
    """Get comprehensive gradient statistics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of gradient statistics
    """
    stats = {
        "total_norm": 0.0,
        "max_grad": 0.0,
        "min_grad": 0.0,
        "mean_grad": 0.0,
        "std_grad": 0.0,
        "zero_grad_params": 0,
        "total_params": 0,
    }
    
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grad = p.grad.detach().data.flatten()
            grads.append(grad)
            stats["total_params"] += 1
        else:
            stats["zero_grad_params"] += 1
    
    if grads:
        all_grads = torch.cat(grads)
        stats["total_norm"] = float(torch.norm(all_grads, 2.0).item())
        stats["max_grad"] = float(all_grads.max().item())
        stats["min_grad"] = float(all_grads.min().item())
        stats["mean_grad"] = float(all_grads.mean().item())
        stats["std_grad"] = float(all_grads.std().item())
    
    return stats


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def get_memory_usage() -> dict[str, float]:
    """Get current memory usage.
    
    Returns:
        Dictionary of memory statistics in MB
    """
    stats = {}
    
    if torch.cuda.is_available():
        stats["cuda_allocated"] = torch.cuda.memory_allocated() / 1024**2
        stats["cuda_reserved"] = torch.cuda.memory_reserved() / 1024**2
        stats["cuda_max_allocated"] = torch.cuda.memory_max_memory_allocated() / 1024**2
    
    # CPU memory (approximate)
    import gc
    stats["cpu_objects"] = len(gc.get_objects())
    
    return stats


def enable_deterministic_training(seed: int = 42) -> None:
    """Enable deterministic training for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TrainingCheckpoint:
    """Checkpoint manager for training."""
    
    def __init__(
        self,
        save_dir: Path,
        save_best_only: bool = True,
        save_last: bool = True,
        mode: str = "min",
        monitor: str = "val_loss",
        max_checkpoints: int = 3,
    ) -> None:
        """
        Args:
            save_dir: Directory to save checkpoints
            save_best_only: Only save best model
            save_last: Always save last epoch
            mode: Min or max mode
            monitor: Metric to monitor
            max_checkpoints: Maximum checkpoints to keep
        """
        from pathlib import Path
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.mode = mode
        self.monitor = monitor
        self.max_checkpoints = max_checkpoints
        
        self.best_score: float | None = None
        self._checkpoints: list[Path] = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        """Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            score: Current score
            metadata: Additional metadata
            
        Returns:
            Path to saved checkpoint or None
        """
        import torch
        
        # Check if this is the best
        is_best = False
        if self.best_score is None:
            is_best = True
            self.best_score = score
        elif (self.mode == "min" and score < self.best_score) or \
             (self.mode == "max" and score > self.best_score):
            is_best = True
            self.best_score = score
        
        # Save last checkpoint
        if self.save_last:
            last_path = self.save_dir / "checkpoint_last.pt"
            self._save_checkpoint(
                last_path, model, optimizer, epoch, score, metadata
            )
        
        # Save best checkpoint
        if self.save_best_only and not is_best:
            return None
        
        if is_best:
            best_path = self.save_dir / "checkpoint_best.pt"
            self._save_checkpoint(
                best_path, model, optimizer, epoch, score, metadata
            )
            
            # Save with epoch name
            epoch_path = self.save_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            self._save_checkpoint(
                epoch_path, model, optimizer, epoch, score, metadata
            )
            self._checkpoints.append(epoch_path)
            
            # Remove old checkpoints
            while len(self._checkpoints) > self.max_checkpoints:
                old_path = self._checkpoints.pop(0)
                if old_path.exists():
                    old_path.unlink()
            
            return best_path
        
        return None
    
    def _save_checkpoint(
        self,
        path: Path,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save checkpoint to file."""
        import torch
        
        state = {
            "epoch": epoch,
            "score": score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata or {},
        }
        
        torch.save(state, path)
        log.info(f"Checkpoint saved: {path}")
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        checkpoint_type: str = "best",
    ) -> dict[str, Any]:
        """Load checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state
            checkpoint_type: Type of checkpoint ('best', 'last', 'specific')
            
        Returns:
            Checkpoint metadata
        """
        import torch
        
        if checkpoint_type == "best":
            path = self.save_dir / "checkpoint_best.pt"
        elif checkpoint_type == "last":
            path = self.save_dir / "checkpoint_last.pt"
        else:
            path = self.save_dir / checkpoint_type
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, weights_only=True)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        log.info(f"Checkpoint loaded: {path} (epoch {checkpoint['epoch']})")
        
        return {
            "epoch": checkpoint["epoch"],
            "score": checkpoint["score"],
            "metadata": checkpoint.get("metadata", {}),
        }
