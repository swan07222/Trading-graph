"""Centralized Recovery Manager.

Provides unified checkpoint, state persistence, and recovery capabilities
for the trading system.

Features:
- Checkpoint save/restore for training and inference
- State persistence across restarts
- Automatic recovery on failure
- Recovery metrics and audit logging
- Versioned checkpoints with retention policy

Usage:
    from utils.recovery_manager import RecoveryManager
    
    recovery = RecoveryManager()
    
    # Save checkpoint
    recovery.save_checkpoint("training", {"epoch": 10, "model": state_dict})
    
    # Restore checkpoint
    checkpoint = recovery.load_checkpoint("training")
    
    # Run with recovery
    result = recovery.run_with_recovery(
        operation="train_model",
        func=train,
        checkpoint_key="training",
    )
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from config.settings import CONFIG
from utils.atomic_io import atomic_write_json, atomic_torch_save, read_json
from utils.logger import get_logger
from utils.recoverable import (
    COMMON_RECOVERABLE_EXCEPTIONS,
    MODEL_LOAD_RECOVERABLE_EXCEPTIONS,
    RecoveryContext,
    RecoveryResult,
    RecoveryStrategy,
    retry_with_recovery,
)

log = get_logger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    checkpoint_type: str
    created_at: str
    updated_at: str
    version: str = "1.0"
    size_bytes: int = 0
    checksum: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    
    @classmethod
    def create(
        cls,
        checkpoint_type: str,
        labels: dict[str, str] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> "CheckpointMetadata":
        """Create a new checkpoint metadata."""
        now = datetime.now().isoformat()
        checkpoint_id = f"{checkpoint_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return cls(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            created_at=now,
            updated_at=now,
            labels=labels or {},
            metrics=metrics or {},
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_type": self.checkpoint_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "labels": self.labels,
            "metrics": self.metrics,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        """Deserialize from dictionary."""
        return cls(
            checkpoint_id=data.get("checkpoint_id", ""),
            checkpoint_type=data.get("checkpoint_type", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            version=data.get("version", "1.0"),
            size_bytes=data.get("size_bytes", 0),
            checksum=data.get("checksum", ""),
            labels=data.get("labels", {}),
            metrics=data.get("metrics", {}),
            notes=data.get("notes", ""),
        )


@dataclass
class RecoveryState:
    """Current recovery state."""
    last_checkpoint: str | None = None
    last_checkpoint_time: datetime | None = None
    recovery_count: int = 0
    last_recovery_time: datetime | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    total_operations: int = 0
    successful_operations: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations
    
    @property
    def is_healthy(self) -> bool:
        """Check if recovery state is healthy."""
        return (
            self.consecutive_failures < 3 and
            self.success_rate > 0.5
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "last_checkpoint": self.last_checkpoint,
            "last_checkpoint_time": self.last_checkpoint_time.isoformat() if self.last_checkpoint_time else None,
            "recovery_count": self.recovery_count,
            "last_recovery_time": self.last_recovery_time.isoformat() if self.last_recovery_time else None,
            "last_error": self.last_error,
            "consecutive_failures": self.consecutive_failures,
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "success_rate": round(self.success_rate, 4),
            "is_healthy": self.is_healthy,
        }


class RecoveryManager:
    """Centralized recovery manager for the trading system.
    
    Features:
    - Checkpoint management with versioning
    - Automatic state persistence
    - Recovery metrics and monitoring
    - Configurable retention policies
    
    Usage:
        recovery = RecoveryManager()
        
        # Save checkpoint
        recovery.save_checkpoint(
            key="training_epoch_10",
            state={"epoch": 10, "model": model.state_dict()},
            checkpoint_type="training",
        )
        
        # Load latest checkpoint
        checkpoint = recovery.load_latest_checkpoint("training")
        
        # Run operation with automatic recovery
        result = recovery.run_with_recovery(
            operation="fetch_data",
            func=fetch_data,
            max_attempts=5,
        )
    """
    
    def __init__(
        self,
        checkpoint_dir: Path | str | None = None,
        max_checkpoints: int = 10,
        retention_days: int = 7,
    ) -> None:
        """Initialize recovery manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints (default: CONFIG.checkpoint_dir)
            max_checkpoints: Maximum checkpoints to retain per type
            retention_days: Days to retain checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CONFIG.checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.retention_days = retention_days
        
        self._lock = threading.RLock()
        self._state = RecoveryState()
        self._checkpoint_registry: dict[str, CheckpointMetadata] = {}
        
        self._init_directories()
        self._load_state()
    
    def _init_directories(self) -> None:
        """Initialize checkpoint directories."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_dir / "training").mkdir(exist_ok=True)
        (self.checkpoint_dir / "inference").mkdir(exist_ok=True)
        (self.checkpoint_dir / "data").mkdir(exist_ok=True)
        (self.checkpoint_dir / "backups").mkdir(exist_ok=True)
    
    def _load_state(self) -> None:
        """Load recovery state from disk."""
        state_path = self.checkpoint_dir / "recovery_state.json"
        if state_path.exists():
            try:
                data = read_json(state_path)
                if data.get("last_checkpoint_time"):
                    data["last_checkpoint_time"] = datetime.fromisoformat(data["last_checkpoint_time"])
                if data.get("last_recovery_time"):
                    data["last_recovery_time"] = datetime.fromisoformat(data["last_recovery_time"])
                self._state = RecoveryState(**data)
                log.debug("Recovery state loaded: %d operations, %.1f%% success", 
                         self._state.total_operations, self._state.success_rate * 100)
            except Exception as e:
                log.warning("Failed to load recovery state: %s", e)
    
    def _save_state(self) -> None:
        """Save recovery state to disk."""
        state_path = self.checkpoint_dir / "recovery_state.json"
        try:
            atomic_write_json(state_path, self._state.to_dict())
        except Exception as e:
            log.warning("Failed to save recovery state: %s", e)
    
    def save_checkpoint(
        self,
        key: str,
        state: dict[str, Any],
        checkpoint_type: str = "general",
        labels: dict[str, str] | None = None,
        metrics: dict[str, float] | None = None,
        save_metadata: bool = True,
    ) -> str:
        """Save a checkpoint.
        
        Args:
            key: Unique checkpoint key
            state: State dictionary to save
            checkpoint_type: Type of checkpoint (training, inference, data)
            labels: Optional labels for categorization
            metrics: Optional metrics to track
            save_metadata: Whether to save metadata file
        
        Returns:
            Checkpoint ID
        
        Example:
            checkpoint_id = recovery.save_checkpoint(
                key="epoch_10",
                state={"epoch": 10, "model": model.state_dict()},
                checkpoint_type="training",
                metrics={"loss": 0.05, "accuracy": 0.95},
            )
        """
        with self._lock:
            try:
                # Create metadata
                metadata = CheckpointMetadata.create(
                    checkpoint_type=checkpoint_type,
                    labels=labels,
                    metrics=metrics,
                )
                
                # Determine file path
                type_dir = self.checkpoint_dir / checkpoint_type
                type_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint_path = type_dir / f"{key}.pt"
                metadata_path = type_dir / f"{key}.meta.json"
                
                # Calculate checksum
                state_bytes = json.dumps(state, sort_keys=True).encode()
                checksum = hashlib.sha256(state_bytes).hexdigest()
                metadata.checksum = checksum
                metadata.size_bytes = len(state_bytes)
                
                # Save checkpoint (PyTorch format for compatibility)
                try:
                    import torch
                    atomic_torch_save(checkpoint_path, state)
                except ImportError:
                    # Fallback to JSON if torch not available
                    atomic_write_json(checkpoint_path.with_suffix(".json"), state)
                
                # Save metadata
                if save_metadata:
                    atomic_write_json(metadata_path, metadata.to_dict())
                
                # Update registry
                self._checkpoint_registry[key] = metadata
                
                # Update state
                self._state.last_checkpoint = key
                self._state.last_checkpoint_time = datetime.now()
                self._save_state()
                
                # Prune old checkpoints
                self._prune_checkpoints(checkpoint_type)
                
                log.info("Checkpoint saved: %s (%s)", key, checkpoint_type)
                return metadata.checkpoint_id
                
            except Exception as e:
                log.error("Failed to save checkpoint %s: %s", key, e)
                raise
    
    def load_checkpoint(
        self,
        key: str,
        checkpoint_type: str | None = None,
    ) -> dict[str, Any] | None:
        """Load a specific checkpoint.
        
        Args:
            key: Checkpoint key
            checkpoint_type: Optional type filter
        
        Returns:
            State dictionary or None if not found
        """
        with self._lock:
            try:
                if checkpoint_type:
                    checkpoint_path = self.checkpoint_dir / checkpoint_type / f"{key}.pt"
                else:
                    # Search all types
                    checkpoint_path = None
                    for type_dir in self.checkpoint_dir.iterdir():
                        if type_dir.is_dir():
                            candidate = type_dir / f"{key}.pt"
                            if candidate.exists():
                                checkpoint_path = candidate
                                break
                
                if not checkpoint_path or not checkpoint_path.exists():
                    # Try JSON fallback
                    json_path = checkpoint_path.with_suffix(".json") if checkpoint_path else None
                    if json_path and json_path.exists():
                        return read_json(json_path)
                    return None
                
                # Load PyTorch checkpoint
                import torch
                from utils.atomic_io import torch_load
                
                allow_unsafe = bool(getattr(getattr(CONFIG, "model", None), "allow_unsafe_artifact_load", False))
                
                state = torch_load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                    allow_unsafe=allow_unsafe,
                )
                
                log.info("Checkpoint loaded: %s", key)
                return state
                
            except MODEL_LOAD_RECOVERABLE_EXCEPTIONS as e:
                log.warning("Failed to load checkpoint %s: %s", key, e)
                return None
            except Exception as e:
                log.error("Failed to load checkpoint %s: %s", key, e)
                return None
    
    def load_latest_checkpoint(
        self,
        checkpoint_type: str,
    ) -> tuple[str, dict[str, Any]] | None:
        """Load the most recent checkpoint of a type.
        
        Args:
            checkpoint_type: Type of checkpoint
        
        Returns:
            Tuple of (key, state) or None if not found
        
        Example:
            result = recovery.load_latest_checkpoint("training")
            if result:
                key, state = result
                model.load_state_dict(state["model"])
        """
        with self._lock:
            type_dir = self.checkpoint_dir / checkpoint_type
            if not type_dir.exists():
                return None
            
            # Find latest metadata file
            meta_files = sorted(
                type_dir.glob("*.meta.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            
            for meta_path in meta_files:
                try:
                    metadata = CheckpointMetadata.from_dict(read_json(meta_path))
                    key = meta_path.stem.replace(".meta", "")
                    state = self.load_checkpoint(key, checkpoint_type)
                    if state:
                        log.info("Loaded latest checkpoint: %s", key)
                        return (key, state)
                except Exception as e:
                    log.debug("Failed to load metadata %s: %s", meta_path, e)
            
            return None
    
    def list_checkpoints(
        self,
        checkpoint_type: str | None = None,
        limit: int = 100,
    ) -> list[CheckpointMetadata]:
        """List available checkpoints.
        
        Args:
            checkpoint_type: Filter by type (None for all)
            limit: Maximum number to return
        
        Returns:
            List of checkpoint metadata
        """
        checkpoints = []
        
        if checkpoint_type:
            type_dirs = [self.checkpoint_dir / checkpoint_type]
        else:
            type_dirs = [d for d in self.checkpoint_dir.iterdir() if d.is_dir()]
        
        for type_dir in type_dirs:
            if not type_dir.exists():
                continue
            
            for meta_path in type_dir.glob("*.meta.json"):
                try:
                    metadata = CheckpointMetadata.from_dict(read_json(meta_path))
                    checkpoints.append(metadata)
                except Exception:
                    pass
        
        # Sort by creation time
        checkpoints.sort(key=lambda m: m.created_at, reverse=True)
        return checkpoints[:limit]
    
    def delete_checkpoint(self, key: str, checkpoint_type: str) -> bool:
        """Delete a checkpoint.
        
        Args:
            key: Checkpoint key
            checkpoint_type: Type of checkpoint
        
        Returns:
            True if deleted
        """
        with self._lock:
            type_dir = self.checkpoint_dir / checkpoint_type
            checkpoint_path = type_dir / f"{key}.pt"
            metadata_path = type_dir / f"{key}.meta.json"
            
            deleted = False
            for path in [checkpoint_path, metadata_path, checkpoint_path.with_suffix(".json")]:
                if path.exists():
                    try:
                        path.unlink()
                        deleted = True
                    except Exception as e:
                        log.warning("Failed to delete %s: %s", path, e)
            
            if key in self._checkpoint_registry:
                del self._checkpoint_registry[key]
            
            return deleted
    
    def _prune_checkpoints(self, checkpoint_type: str) -> None:
        """Prune old checkpoints.
        
        Args:
            checkpoint_type: Type to prune
        """
        type_dir = self.checkpoint_dir / checkpoint_type
        if not type_dir.exists():
            return
        
        # Get all checkpoints sorted by age
        meta_files = sorted(
            type_dir.glob("*.meta.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        
        # Keep max_checkpoints and delete old ones
        for i, meta_path in enumerate(meta_files):
            try:
                metadata = CheckpointMetadata.from_dict(read_json(meta_path))
                created = datetime.fromisoformat(metadata.created_at)
                
                if i >= self.max_checkpoints or created < cutoff:
                    key = meta_path.stem.replace(".meta", "")
                    self.delete_checkpoint(key, checkpoint_type)
                    log.debug("Pruned old checkpoint: %s", key)
            except Exception as e:
                log.debug("Failed to prune %s: %s", meta_path, e)
    
    def run_with_recovery(
        self,
        operation: str,
        func: Callable,
        checkpoint_key: str | None = None,
        checkpoint_type: str = "general",
        max_attempts: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 10.0,
        recoverable_exceptions: tuple = COMMON_RECOVERABLE_EXCEPTIONS,
        on_checkpoint_save: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> RecoveryResult:
        """Run an operation with automatic recovery.
        
        Args:
            operation: Operation name
            func: Function to run
            checkpoint_key: Key for checkpointing
            checkpoint_type: Type of checkpoint
            max_attempts: Maximum retry attempts
            base_delay: Base delay between retries
            max_delay: Maximum delay
            recoverable_exceptions: Exceptions to recover from
            on_checkpoint_save: Callback after checkpoint save
            **kwargs: Arguments to pass to func
        
        Returns:
            RecoveryResult
        
        Example:
            result = recovery.run_with_recovery(
                operation="train_epoch",
                func=train_one_epoch,
                checkpoint_key="epoch_checkpoint",
                max_attempts=3,
            )
        """
        with self._lock:
            self._state.total_operations += 1
        
        def wrapped_func() -> Any:
            return func(**kwargs)
        
        def on_success(result: RecoveryResult) -> None:
            with self._lock:
                self._state.successful_operations += 1
                self._state.consecutive_failures = 0
                self._save_state()
            
            # Save checkpoint if key provided
            if checkpoint_key and result.result is not None:
                try:
                    if isinstance(result.result, dict):
                        self.save_checkpoint(
                            key=checkpoint_key,
                            state=result.result,
                            checkpoint_type=checkpoint_type,
                        )
                        if on_checkpoint_save:
                            on_checkpoint_save(checkpoint_key)
                except Exception as e:
                    log.warning("Failed to save checkpoint after success: %s", e)
        
        def on_failure(result: RecoveryResult) -> None:
            with self._lock:
                self._state.consecutive_failures += 1
                self._state.last_error = result.message
                self._state.last_recovery_time = datetime.now()
                self._state.recovery_count += 1
                self._save_state()
        
        recovery_result = retry_with_recovery(
            func=wrapped_func,
            operation=operation,
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            recoverable_exceptions=recoverable_exceptions,
            on_success=on_success,
            on_failure=on_failure,
        )
        
        return recovery_result
    
    def get_state(self) -> RecoveryState:
        """Get current recovery state."""
        return self._state
    
    def get_metrics(self) -> dict[str, Any]:
        """Get recovery metrics."""
        with self._lock:
            return {
                "state": self._state.to_dict(),
                "checkpoint_count": len(self._checkpoint_registry),
                "checkpoints_by_type": self._count_by_type(),
                "disk_usage_bytes": self._calculate_disk_usage(),
            }
    
    def _count_by_type(self) -> dict[str, int]:
        """Count checkpoints by type."""
        counts: dict[str, int] = {}
        for metadata in self._checkpoint_registry.values():
            checkpoint_type = metadata.checkpoint_type
            counts[checkpoint_type] = counts.get(checkpoint_type, 0) + 1
        return counts
    
    def _calculate_disk_usage(self) -> int:
        """Calculate total disk usage of checkpoints."""
        total = 0
        for path in self.checkpoint_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total
    
    def export_state(self, output_path: Path | str) -> bool:
        """Export recovery state to file.
        
        Args:
            output_path: Output file path
        
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "state": self._state.to_dict(),
                "checkpoints": [m.to_dict() for m in self._checkpoint_registry.values()],
            }
            
            atomic_write_json(output_path, export_data)
            log.info("Recovery state exported to %s", output_path)
            return True
        except Exception as e:
            log.error("Failed to export recovery state: %s", e)
            return False
    
    def import_state(self, input_path: Path | str) -> bool:
        """Import recovery state from file.
        
        Args:
            input_path: Input file path
        
        Returns:
            True if successful
        """
        try:
            input_path = Path(input_path)
            if not input_path.exists():
                return False
            
            data = read_json(input_path)
            
            with self._lock:
                state_data = data.get("state", {})
                if state_data:
                    if state_data.get("last_checkpoint_time"):
                        state_data["last_checkpoint_time"] = datetime.fromisoformat(state_data["last_checkpoint_time"])
                    if state_data.get("last_recovery_time"):
                        state_data["last_recovery_time"] = datetime.fromisoformat(state_data["last_recovery_time"])
                    self._state = RecoveryState(**state_data)
                
                checkpoints_data = data.get("checkpoints", [])
                for cp_data in checkpoints_data:
                    metadata = CheckpointMetadata.from_dict(cp_data)
                    self._checkpoint_registry[metadata.checkpoint_id] = metadata
                
                self._save_state()
            
            log.info("Recovery state imported from %s", input_path)
            return True
        except Exception as e:
            log.error("Failed to import recovery state: %s", e)
            return False
    
    def cleanup(self) -> dict[str, int]:
        """Clean up old checkpoints and temporary files.
        
        Returns:
            Statistics about cleanup
        """
        stats = {"deleted_checkpoints": 0, "deleted_temp_files": 0, "freed_bytes": 0}
        
        with self._lock:
            # Prune all checkpoint types
            for type_dir in self.checkpoint_dir.iterdir():
                if type_dir.is_dir() and type_dir.name not in {"backups"}:
                    self._prune_checkpoints(type_dir.name)
            
            # Clean temp files
            for tmp_file in self.checkpoint_dir.rglob("*.tmp"):
                try:
                    size = tmp_file.stat().st_size
                    tmp_file.unlink()
                    stats["deleted_temp_files"] += 1
                    stats["freed_bytes"] += size
                except Exception:
                    pass
            
            log.info("Cleanup completed: %d checkpoints, %d temp files, %d bytes freed",
                    stats["deleted_checkpoints"], stats["deleted_temp_files"], stats["freed_bytes"])
        
        return stats


# Global instance
_recovery_manager: RecoveryManager | None = None
_recovery_lock = threading.Lock()


def get_recovery_manager() -> RecoveryManager:
    """Get or create the global recovery manager instance."""
    global _recovery_manager
    
    if _recovery_manager is None:
        with _recovery_lock:
            if _recovery_manager is None:
                _recovery_manager = RecoveryManager()
    
    return _recovery_manager


def reset_recovery_manager() -> None:
    """Reset the global recovery manager (for testing)."""
    global _recovery_manager
    with _recovery_lock:
        _recovery_manager = None
