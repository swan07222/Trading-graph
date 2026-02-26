"""Training Checkpoint Module.

Provides checkpoint save/load functionality for training sessions with
resume capability.

Features:
- Save training state (epoch, optimizer, model weights)
- Resume from checkpoint
- Checkpoint rotation and retention
- Training progress tracking

Usage:
    from models.training_checkpoint import TrainingCheckpoint
    
    checkpoint = TrainingCheckpoint("my_training")
    
    # Load existing checkpoint or start fresh
    start_epoch = 0
    if checkpoint.has_checkpoint():
        state = checkpoint.load()
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
    
    # Train with checkpointing
    for epoch in range(start_epoch, total_epochs):
        train_one_epoch()
        
        # Save checkpoint
        checkpoint.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": {"loss": loss, "accuracy": acc},
        })
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import CONFIG
from utils.atomic_io import atomic_write_json, read_json
from utils.logger import get_logger

log = get_logger(__name__)

try:
    import torch
    from utils.atomic_io import atomic_torch_save, torch_load
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class TrainingState:
    """Training session state."""
    session_id: str
    start_time: str
    last_checkpoint_time: str
    current_epoch: int = 0
    total_epochs: int = 0
    best_accuracy: float = 0.0
    best_loss: float = float('inf')
    metrics_history: list[dict] = field(default_factory=list)
    model_type: str = ""
    interval: str = ""
    horizon: int = 0
    stocks_trained: list[str] = field(default_factory=list)
    
    @property
    def progress(self) -> float:
        """Calculate training progress (0-1)."""
        if self.total_epochs == 0:
            return 0.0
        return min(1.0, self.current_epoch / self.total_epochs)
    
    @property
    def elapsed_seconds(self) -> float:
        """Calculate elapsed time since start."""
        start = datetime.fromisoformat(self.start_time)
        return (datetime.now() - start).total_seconds()
    
    @property
    def estimated_total_seconds(self) -> float:
        """Estimate total training time."""
        if self.progress == 0:
            return 0.0
        return self.elapsed_seconds / self.progress
    
    @property
    def estimated_remaining_seconds(self) -> float:
        """Estimate remaining time."""
        return self.estimated_total_seconds - self.elapsed_seconds
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "last_checkpoint_time": self.last_checkpoint_time,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "best_accuracy": self.best_accuracy,
            "best_loss": self.best_loss,
            "metrics_history": self.metrics_history[-100:],  # Keep last 100
            "model_type": self.model_type,
            "interval": self.interval,
            "horizon": self.horizon,
            "stocks_trained": self.stocks_trained,
            "progress": round(self.progress, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "estimated_remaining_seconds": round(self.estimated_remaining_seconds, 2),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingState":
        """Deserialize from dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            start_time=data.get("start_time", ""),
            last_checkpoint_time=data.get("last_checkpoint_time", ""),
            current_epoch=data.get("current_epoch", 0),
            total_epochs=data.get("total_epochs", 0),
            best_accuracy=data.get("best_accuracy", 0.0),
            best_loss=data.get("best_loss", float('inf')),
            metrics_history=data.get("metrics_history", []),
            model_type=data.get("model_type", ""),
            interval=data.get("interval", ""),
            horizon=data.get("horizon", 0),
            stocks_trained=data.get("stocks_trained", []),
        )


@dataclass
class CheckpointInfo:
    """Information about a checkpoint file."""
    path: Path
    epoch: int
    timestamp: str
    size_bytes: int
    metrics: dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def from_metadata(cls, meta_path: Path) -> "CheckpointInfo | None":
        """Load checkpoint info from metadata file."""
        try:
            data = read_json(meta_path)
            return cls(
                path=meta_path.with_suffix(".pt"),
                epoch=data.get("epoch", 0),
                timestamp=data.get("timestamp", ""),
                size_bytes=data.get("size_bytes", 0),
                metrics=data.get("metrics", {}),
            )
        except Exception as e:
            log.debug("Failed to load checkpoint metadata: %s", e)
            return None


class TrainingCheckpoint:
    """Training checkpoint manager.
    
    Usage:
        checkpoint = TrainingCheckpoint("ensemble_training")
        
        # Resume or start fresh
        if checkpoint.has_checkpoint():
            state = checkpoint.load()
            # Resume training...
        
        # Save during training
        for epoch in range(total_epochs):
            train()
            checkpoint.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": {"loss": loss},
            })
    """
    
    def __init__(
        self,
        session_name: str,
        checkpoint_dir: Path | str | None = None,
        max_checkpoints: int = 5,
        save_every_epochs: int = 1,
    ) -> None:
        """Initialize training checkpoint.
        
        Args:
            session_name: Unique session identifier
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum checkpoints to retain
            save_every_epochs: Save checkpoint every N epochs
        """
        self.session_name = session_name
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CONFIG.checkpoint_dir / "training"
        self.max_checkpoints = max_checkpoints
        self.save_every_epochs = save_every_epochs
        
        self._lock = threading.Lock()
        self._state: TrainingState | None = None
        
        self._init_directory()
    
    def _init_directory(self) -> None:
        """Initialize checkpoint directory."""
        session_dir = self.session_directory
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing state
        state_path = session_dir / "training_state.json"
        if state_path.exists():
            try:
                data = read_json(state_path)
                self._state = TrainingState.from_dict(data)
                log.info("Loaded training state: epoch %d/%d", 
                        self._state.current_epoch, self._state.total_epochs)
            except Exception as e:
                log.warning("Failed to load training state: %s", e)
    
    @property
    def session_directory(self) -> Path:
        """Get session-specific directory."""
        return self.checkpoint_dir / self.session_name
    
    @property
    def latest_checkpoint_path(self) -> Path | None:
        """Get path to latest checkpoint."""
        session_dir = self.session_directory
        checkpoints = sorted(session_dir.glob("epoch_*.pt"), 
                           key=lambda p: p.stat().st_mtime, 
                           reverse=True)
        return checkpoints[0] if checkpoints else None
    
    @property
    def best_checkpoint_path(self) -> Path | None:
        """Get path to best checkpoint."""
        session_dir = self.session_directory
        best_path = session_dir / "best.pt"
        return best_path if best_path.exists() else None
    
    @property
    def state(self) -> TrainingState | None:
        """Get current training state."""
        return self._state
    
    def has_checkpoint(self) -> bool:
        """Check if a checkpoint exists."""
        return self.latest_checkpoint_path is not None
    
    def initialize_session(
        self,
        total_epochs: int,
        model_type: str = "",
        interval: str = "",
        horizon: int = 0,
        stocks: list[str] | None = None,
    ) -> TrainingState:
        """Initialize a new training session.

        Args:
            total_epochs: Total number of epochs
            model_type: Type of model being trained
            interval: Data interval
            horizon: Prediction horizon
            stocks: List of stocks being trained

        Returns:
            TrainingState for the session
        """
        with self._lock:
            now = datetime.now().isoformat()
            self._state = TrainingState(
                session_id=f"{self.session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                start_time=now,
                last_checkpoint_time=now,
                total_epochs=total_epochs,
                model_type=model_type,
                interval=interval,
                horizon=horizon,
                stocks_trained=stocks or [],
            )
            self._save_state()
            return self._state

    @staticmethod
    def _calculate_checksum(file_path: Path) -> str:
        """Calculate SHA256 checksum of a file.

        FIX 17: Used for checkpoint integrity verification.

        Args:
            file_path: Path to file

        Returns:
            Hex-encoded SHA256 checksum
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            log.warning("Checksum calculation failed for %s: %s", file_path, e)
            return ""

    def _verify_checksum(self, checkpoint_path: Path, expected_checksum: str) -> bool:
        """Verify checkpoint file integrity.

        FIX 17: Validates checkpoint hasn't been corrupted.

        Args:
            checkpoint_path: Path to checkpoint file
            expected_checksum: Expected SHA256 checksum

        Returns:
            True if checksum matches, False otherwise
        """
        if not expected_checksum:
            log.debug("No checksum available for verification")
            return True  # Can't verify without checksum

        actual_checksum = self._calculate_checksum(checkpoint_path)
        if actual_checksum != expected_checksum:
            log.error(
                "Checkpoint integrity check failed for %s: expected %s, got %s",
                checkpoint_path,
                expected_checksum[:16],
                actual_checksum[:16],
            )
            return False

        log.debug("Checkpoint integrity verified for %s", checkpoint_path)
        return True

    def save(
        self,
        checkpoint_data: dict[str, Any],
        epoch: int | None = None,
        metrics: dict[str, float] | None = None,
        is_best: bool = False,
    ) -> Path | None:
        """Save a training checkpoint.
        
        Args:
            checkpoint_data: Data to save (model, optimizer, etc.)
            epoch: Current epoch (overrides checkpoint_data["epoch"])
            metrics: Optional metrics to track
            is_best: Whether this is the best checkpoint so far
        
        Returns:
            Path to saved checkpoint or None if failed
        """
        with self._lock:
            try:
                session_dir = self.session_directory
                session_dir.mkdir(parents=True, exist_ok=True)
                
                epoch = epoch if epoch is not None else checkpoint_data.get("epoch", 0)
                
                # Update state
                if self._state:
                    self._state.current_epoch = epoch + 1
                    self._state.last_checkpoint_time = datetime.now().isoformat()
                    
                    if metrics:
                        self._state.metrics_history.append({
                            "epoch": epoch,
                            "timestamp": datetime.now().isoformat(),
                            **metrics,
                        })
                        
                        # Track best
                        if "accuracy" in metrics and metrics["accuracy"] > self._state.best_accuracy:
                            self._state.best_accuracy = metrics["accuracy"]
                            is_best = True
                        
                        if "loss" in metrics and metrics["loss"] < self._state.best_loss:
                            self._state.best_loss = metrics["loss"]
                
                # Save checkpoint
                checkpoint_path = session_dir / f"epoch_{epoch}.pt"

                if _TORCH_AVAILABLE:
                    atomic_torch_save(checkpoint_path, checkpoint_data)
                else:
                    # Fallback to JSON
                    json_path = session_dir / f"epoch_{epoch}.json"
                    atomic_write_json(json_path, checkpoint_data)
                    checkpoint_path = json_path

                # FIX 17: Calculate and save checksum for integrity verification
                checksum = self._calculate_checksum(checkpoint_path)

                # Save metadata
                metadata = {
                    "epoch": epoch,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics or {},
                    "size_bytes": checkpoint_path.stat().st_size,
                    "session_id": self._state.session_id if self._state else "",
                    "checksum_sha256": checksum,  # FIX 17: Add checksum
                }
                atomic_write_json(session_dir / f"epoch_{epoch}.meta.json", metadata)

                # Save as best
                if is_best:
                    best_path = session_dir / "best.pt"
                    shutil.copy2(checkpoint_path, best_path)
                    # Save checksum for best checkpoint too
                    best_metadata = dict(metadata)
                    best_metadata["checksum_sha256"] = self._calculate_checksum(best_path)
                    atomic_write_json(session_dir / "best.meta.json", best_metadata)
                
                # Save state
                self._save_state()
                
                # Prune old checkpoints
                self._prune_checkpoints()
                
                log.info("Checkpoint saved: epoch %d (%s)", epoch, 
                        "best" if is_best else f"metrics={metrics}" if metrics else "OK")
                return checkpoint_path
                
            except Exception as e:
                log.error("Failed to save checkpoint: %s", e)
                return None
    
    def load(self, checkpoint_path: Path | str | None = None) -> dict[str, Any] | None:
        """Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (default: latest)

        Returns:
            Checkpoint data or None if not found
        """
        with self._lock:
            try:
                if checkpoint_path is None:
                    checkpoint_path = self.latest_checkpoint_path

                if checkpoint_path is None:
                    return None

                checkpoint_path = Path(checkpoint_path)
                if not checkpoint_path.exists():
                    return None

                if checkpoint_path.suffix == ".pt" and _TORCH_AVAILABLE:
                    # FIX 16: Handle weights_only parameter compatibility
                    allow_unsafe = bool(getattr(getattr(CONFIG, "model", None), "allow_unsafe_artifact_load", False))
                    
                    # FIX 17: Verify checksum before loading
                    meta_path = checkpoint_path.with_suffix(".meta.json")
                    expected_checksum = ""
                    if meta_path.exists():
                        try:
                            meta = read_json(meta_path)
                            expected_checksum = meta.get("checksum_sha256", "")
                        except Exception:
                            pass

                    try:
                        # First attempt: weights_only=True (secure, PyTorch >= 2.6)
                        data = torch_load(
                            checkpoint_path,
                            map_location="cpu",
                            weights_only=True,
                            allow_unsafe=allow_unsafe,
                        )
                    except (TypeError, AttributeError) as e:
                        # Second attempt: weights_only=False for older PyTorch or incompatible checkpoints
                        log.debug("weights_only=True failed (%s), retrying with weights_only=False", e)
                        try:
                            data = torch_load(
                                checkpoint_path,
                                map_location="cpu",
                                weights_only=False,
                            )
                            log.warning("Checkpoint loaded with weights_only=False - ensure checkpoint source is trusted")
                        except Exception as e2:
                            log.error("Checkpoint load failed with both weights_only modes: %s", e2)
                            return None
                    except Exception as e:
                        log.error("Checkpoint load failed: %s", e)
                        return None
                    else:
                        # FIX 17: Verify checksum after successful load
                        if expected_checksum and not self._verify_checksum(checkpoint_path, expected_checksum):
                            log.error("Checkpoint integrity verification failed")
                            return None
                else:
                    # Fallback to JSON
                    data = read_json(checkpoint_path)

                log.info("Checkpoint loaded: %s", checkpoint_path)
                return data

            except Exception as e:
                log.warning("Failed to load checkpoint: %s", e)
                return None
    
    def load_best(self) -> dict[str, Any] | None:
        """Load the best checkpoint."""
        return self.load(self.best_checkpoint_path)
    
    def get_checkpoints(self) -> list[CheckpointInfo]:
        """List all checkpoints for this session."""
        checkpoints = []
        session_dir = self.session_directory
        
        for meta_path in session_dir.glob("epoch_*.meta.json"):
            info = CheckpointInfo.from_metadata(meta_path)
            if info:
                checkpoints.append(info)
        
        # Sort by epoch
        checkpoints.sort(key=lambda c: c.epoch)
        return checkpoints
    
    def resume_training(self) -> tuple[int, dict[str, Any]] | None:
        """Resume training from latest checkpoint.
        
        Returns:
            Tuple of (next_epoch, checkpoint_data) or None if no checkpoint
        """
        if not self.has_checkpoint():
            return None
        
        checkpoint_data = self.load()
        if checkpoint_data is None:
            return None
        
        epoch = checkpoint_data.get("epoch", 0)
        return (epoch + 1, checkpoint_data)
    
    def _save_state(self) -> None:
        """Save training state to disk."""
        if not self._state:
            return
        
        state_path = self.session_directory / "training_state.json"
        try:
            atomic_write_json(state_path, self._state.to_dict())
        except Exception as e:
            log.warning("Failed to save training state: %s", e)
    
    def _prune_checkpoints(self) -> None:
        """Remove old checkpoints."""
        session_dir = self.session_directory
        checkpoints = sorted(
            session_dir.glob("epoch_*.meta.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        for old_meta in checkpoints[self.max_checkpoints:]:
            try:
                epoch = old_meta.stem.replace("epoch_", "")
                for suffix in [".pt", ".json", ".meta.json"]:
                    path = session_dir / f"epoch_{epoch}{suffix}"
                    if path.exists():
                        path.unlink()
                log.debug("Pruned old checkpoint: epoch %s", epoch)
            except Exception as e:
                log.debug("Failed to prune checkpoint: %s", e)
    
    def get_progress(self) -> dict[str, Any]:
        """Get training progress information."""
        if not self._state:
            return {"progress": 0.0, "status": "not_started"}
        
        return {
            "progress": self._state.progress,
            "current_epoch": self._state.current_epoch,
            "total_epochs": self._state.total_epochs,
            "elapsed_seconds": self._state.elapsed_seconds,
            "estimated_remaining_seconds": self._state.estimated_remaining_seconds,
            "best_accuracy": self._state.best_accuracy,
            "best_loss": self._state.best_loss,
            "status": "training" if self._state.progress < 1.0 else "completed",
        }
    
    def cleanup(self) -> int:
        """Clean up all checkpoints for this session.
        
        Returns:
            Number of files deleted
        """
        with self._lock:
            deleted = 0
            session_dir = self.session_directory
            
            if not session_dir.exists():
                return 0
            
            for file_path in session_dir.glob("*"):
                try:
                    file_path.unlink()
                    deleted += 1
                except Exception:
                    pass
            
            # Remove directory if empty
            try:
                if not any(session_dir.iterdir()):
                    session_dir.rmdir()
            except Exception:
                pass
            
            log.info("Cleaned up %d checkpoint files for session %s", deleted, self.session_name)
            return deleted


def create_training_checkpoint(
    session_name: str,
    total_epochs: int,
    model_type: str = "",
    interval: str = "",
    horizon: int = 0,
    stocks: list[str] | None = None,
    resume: bool = True,
) -> tuple[TrainingCheckpoint, int, dict | None]:
    """Create or resume a training checkpoint.
    
    Args:
        session_name: Session identifier
        total_epochs: Total epochs for training
        model_type: Type of model
        interval: Data interval
        horizon: Prediction horizon
        stocks: Stocks to train on
        resume: Whether to resume if checkpoint exists
    
    Returns:
        Tuple of (checkpoint, start_epoch, checkpoint_data)
    
    Example:
        checkpoint, start_epoch, state = create_training_checkpoint(
            "ensemble_1m_30",
            total_epochs=100,
            model_type="ensemble",
            interval="1m",
            horizon=30,
        )
        
        for epoch in range(start_epoch, 100):
            train_one_epoch()
            checkpoint.save({...}, epoch=epoch)
    """
    checkpoint = TrainingCheckpoint(session_name)
    
    if resume and checkpoint.has_checkpoint():
        result = checkpoint.resume_training()
        if result:
            start_epoch, checkpoint_data = result
            log.info("Resuming training from epoch %d", start_epoch)
            return (checkpoint, start_epoch, checkpoint_data)
    
    # Initialize new session
    checkpoint.initialize_session(
        total_epochs=total_epochs,
        model_type=model_type,
        interval=interval,
        horizon=horizon,
        stocks=stocks,
    )
    
    return (checkpoint, 0, None)
