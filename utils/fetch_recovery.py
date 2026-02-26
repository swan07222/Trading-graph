"""Data Fetching State Recovery.

Provides state persistence and recovery for data fetching operations,
enabling resume of interrupted data collection.

Features:
- Fetch state persistence (symbols, timestamps, progress)
- Resume interrupted fetches
- Batch tracking and recovery
- Cache state management

Usage:
    from utils.fetch_recovery import FetchStateRecovery
    
    recovery = FetchStateRecovery()
    
    # Start fetch session
    session_id = recovery.start_session(
        operation="fetch_history",
        symbols=symbol_list,
        interval="1m",
    )
    
    # During fetch, track progress
    for symbol in symbols:
        data = fetcher.get_history(symbol, interval="1m")
        recovery.mark_complete(session_id, symbol)
        
        # Periodically save state
        if i % 10 == 0:
            recovery.save_state()
    
    # On restart, resume
    if recovery.has_pending_session():
        session = recovery.load_pending_session()
        remaining = session["pending_symbols"]
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from config.settings import CONFIG
from utils.atomic_io import atomic_write_json, read_json
from utils.logger import get_logger
from utils.recoverable import (
    COMMON_RECOVERABLE_EXCEPTIONS,
    RecoveryContext,
    RecoveryResult,
    retry_with_recovery,
)

log = get_logger(__name__)


@dataclass
class FetchSession:
    """Data fetching session state."""
    session_id: str
    operation: str
    created_at: str
    updated_at: str
    status: Literal["pending", "running", "paused", "completed", "failed"] = "pending"
    
    # Symbols to fetch
    total_symbols: int = 0
    completed_symbols: list[str] = field(default_factory=list)
    failed_symbols: list[str] = field(default_factory=list)
    pending_symbols: list[str] = field(default_factory=list)
    
    # Parameters
    interval: str = ""
    horizon: int = 0
    lookback_bars: int = 0
    extra_params: dict[str, Any] = field(default_factory=dict)
    
    # Progress
    progress_percent: float = 0.0
    last_symbol: str = ""
    last_error: str = ""
    retry_counts: dict[str, int] = field(default_factory=dict)
    
    # Timing
    started_at: str | None = None
    completed_at: str | None = None
    estimated_remaining_seconds: float = 0.0
    
    @property
    def completed_count(self) -> int:
        return len(self.completed_symbols)
    
    @property
    def failed_count(self) -> int:
        return len(self.failed_symbols)
    
    @property
    def pending_count(self) -> int:
        return len(self.pending_symbols)
    
    def update_progress(self) -> None:
        """Update progress percentage."""
        if self.total_symbols > 0:
            self.progress_percent = round(
                (self.completed_count / self.total_symbols) * 100, 2
            )
        self.updated_at = datetime.now().isoformat()
        
        # Estimate remaining time
        if self.started_at and self.completed_count > 0:
            start = datetime.fromisoformat(self.started_at)
            elapsed = (datetime.now() - start).total_seconds()
            avg_per_symbol = elapsed / max(1, self.completed_count)
            self.estimated_remaining_seconds = avg_per_symbol * self.pending_count
    
    def mark_complete(self, symbol: str) -> None:
        """Mark a symbol as completed."""
        if symbol in self.pending_symbols:
            self.pending_symbols.remove(symbol)
        if symbol not in self.completed_symbols:
            self.completed_symbols.append(symbol)
        self.last_symbol = symbol
        self.update_progress()
    
    def mark_failed(self, symbol: str, error: str = "") -> None:
        """Mark a symbol as failed."""
        if symbol in self.pending_symbols:
            self.pending_symbols.remove(symbol)
        if symbol not in self.failed_symbols:
            self.failed_symbols.append(symbol)
        if error:
            self.last_error = error
        self.update_progress()
    
    def retry_symbol(self, symbol: str) -> bool:
        """Add a symbol back to pending for retry."""
        if symbol in self.failed_symbols:
            self.failed_symbols.remove(symbol)
            if symbol not in self.pending_symbols:
                self.pending_symbols.append(symbol)
            
            # Track retry count
            self.retry_counts[symbol] = self.retry_counts.get(symbol, 0) + 1
            self.update_progress()
            return True
        return False
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        progress_percent = self.progress_percent
        if self.total_symbols > 0:
            progress_percent = round((self.completed_count / self.total_symbols) * 100, 2)

        return {
            "session_id": self.session_id,
            "operation": self.operation,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "total_symbols": self.total_symbols,
            "completed_symbols": self.completed_symbols,
            "failed_symbols": self.failed_symbols,
            "pending_symbols": self.pending_symbols,
            "interval": self.interval,
            "horizon": self.horizon,
            "lookback_bars": self.lookback_bars,
            "extra_params": self.extra_params,
            "progress_percent": progress_percent,
            "last_symbol": self.last_symbol,
            "last_error": self.last_error,
            "retry_counts": self.retry_counts,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "estimated_remaining_seconds": round(self.estimated_remaining_seconds, 2),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FetchSession":
        """Deserialize from dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            operation=data.get("operation", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            status=data.get("status", "pending"),
            total_symbols=data.get("total_symbols", 0),
            completed_symbols=data.get("completed_symbols", []),
            failed_symbols=data.get("failed_symbols", []),
            pending_symbols=data.get("pending_symbols", []),
            interval=data.get("interval", ""),
            horizon=data.get("horizon", 0),
            lookback_bars=data.get("lookback_bars", 0),
            extra_params=data.get("extra_params", {}),
            progress_percent=data.get("progress_percent", 0.0),
            last_symbol=data.get("last_symbol", ""),
            last_error=data.get("last_error", ""),
            retry_counts=data.get("retry_counts", {}),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            estimated_remaining_seconds=data.get("estimated_remaining_seconds", 0.0),
        )


@dataclass
class BatchState:
    """State for a batch of fetch operations."""
    batch_id: str
    session_id: str
    symbols: list[str]
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    
    @property
    def progress(self) -> float:
        if not self.symbols:
            return 0.0
        return len(self.completed) / len(self.symbols)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "session_id": self.session_id,
            "symbols": self.symbols,
            "status": self.status,
            "completed": self.completed,
            "failed": self.failed,
            "errors": self.errors,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": round(self.progress, 4),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchState":
        return cls(
            batch_id=data.get("batch_id", ""),
            session_id=data.get("session_id", ""),
            symbols=data.get("symbols", []),
            status=data.get("status", "pending"),
            completed=data.get("completed", []),
            failed=data.get("failed", []),
            errors=data.get("errors", {}),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )


class FetchStateRecovery:
    """Recovery manager for data fetching operations.
    
    Usage:
        recovery = FetchStateRecovery()
        
        # Start session
        session_id = recovery.start_session(
            operation="fetch_history",
            symbols=["000001.SZ", "000002.SZ"],
            interval="1m",
        )
        
        # Get session
        session = recovery.get_session(session_id)
        
        # Mark symbols as complete
        for symbol in session.pending_symbols:
            data = fetch(symbol)
            session.mark_complete(symbol)
            recovery.save_session(session)
    """
    
    def __init__(
        self,
        state_dir: Path | str | None = None,
        max_sessions: int = 10,
        retention_hours: int = 24,
    ) -> None:
        """Initialize fetch state recovery.
        
        Args:
            state_dir: Directory for state files
            max_sessions: Maximum sessions to retain
            retention_hours: Hours to retain completed sessions
        """
        self.state_dir = Path(state_dir) if state_dir else CONFIG.data_dir / "fetch_state"
        self.max_sessions = max_sessions
        self.retention_hours = retention_hours
        
        self._lock = threading.Lock()
        self._sessions: dict[str, FetchSession] = {}
        self._batches: dict[str, BatchState] = {}
        self._active_session_id: str | None = None
        
        self._init_directory()
        self._load_state()
    
    def _init_directory(self) -> None:
        """Initialize state directory."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "sessions").mkdir(exist_ok=True)
        (self.state_dir / "batches").mkdir(exist_ok=True)
    
    def _load_state(self) -> None:
        """Load state from disk."""
        sessions_dir = self.state_dir / "sessions"
        for session_path in sessions_dir.glob("*.json"):
            try:
                data = read_json(session_path)
                session = FetchSession.from_dict(data)
                self._sessions[session.session_id] = session
            except Exception as e:
                log.warning("Failed to load session %s: %s", session_path, e)
        
        # Load active session
        active_path = self.state_dir / "active_session.json"
        if active_path.exists():
            try:
                data = read_json(active_path)
                self._active_session_id = data.get("session_id")
            except Exception:
                pass
        
        log.debug("Loaded %d fetch sessions", len(self._sessions))
    
    def _save_session(self, session: FetchSession) -> None:
        """Save session to disk."""
        sessions_dir = self.state_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        
        session_path = sessions_dir / f"{session.session_id}.json"
        atomic_write_json(session_path, session.to_dict())
        
        # Update active session
        if session.status in ("running", "paused"):
            self._active_session_id = session.session_id
            atomic_write_json(
                self.state_dir / "active_session.json",
                {"session_id": session.session_id, "updated_at": session.updated_at},
            )
        
        self._sessions[session.session_id] = session
    
    def start_session(
        self,
        operation: str,
        symbols: list[str],
        interval: str = "1m",
        horizon: int = 0,
        lookback_bars: int = 0,
        extra_params: dict[str, Any] | None = None,
    ) -> str:
        """Start a new fetch session.
        
        Args:
            operation: Operation type (fetch_history, fetch_quote, etc.)
            symbols: List of symbols to fetch
            interval: Data interval
            horizon: Prediction horizon
            lookback_bars: Lookback bars for history
            extra_params: Additional parameters
        
        Returns:
            Session ID
        """
        with self._lock:
            now = datetime.now().isoformat()
            session_id = hashlib.sha256(
                f"{operation}_{now}_{','.join(sorted(symbols))}".encode()
            ).hexdigest()[:16]
            
            session = FetchSession(
                session_id=session_id,
                operation=operation,
                created_at=now,
                updated_at=now,
                status="running",
                total_symbols=len(symbols),
                pending_symbols=list(symbols),
                interval=interval,
                horizon=horizon,
                lookback_bars=lookback_bars,
                extra_params=extra_params or {},
                started_at=now,
            )
            
            self._save_session(session)
            self._active_session_id = session_id
            
            log.info(
                "Started fetch session %s: %d symbols, operation=%s",
                session_id, len(symbols), operation,
            )
            return session_id
    
    def get_session(self, session_id: str) -> FetchSession | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    def get_active_session(self) -> FetchSession | None:
        """Get the active session if any."""
        if not self._active_session_id:
            return None
        return self._sessions.get(self._active_session_id)
    
    def has_pending_session(self) -> bool:
        """Check if there's a pending session."""
        session = self.get_active_session()
        return (
            session is not None
            and session.status in ("running", "paused")
            and bool(session.pending_symbols)
        )
    
    def load_pending_session(self) -> FetchSession | None:
        """Load the pending session for resume."""
        session = self.get_active_session()
        if session and session.pending_symbols:
            log.info(
                "Found pending session: %d/%d completed, %d pending",
                session.completed_count, session.total_symbols, session.pending_count,
            )
            return session
        return None
    
    def mark_complete(
        self,
        session_id: str,
        symbol: str,
        save: bool = True,
    ) -> None:
        """Mark a symbol as completed."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.mark_complete(symbol)
                if save:
                    self._save_session(session)
    
    def mark_failed(
        self,
        session_id: str,
        symbol: str,
        error: str = "",
        save: bool = True,
    ) -> None:
        """Mark a symbol as failed."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.mark_failed(symbol, error)
                if save:
                    self._save_session(session)
    
    def retry_symbol(
        self,
        session_id: str,
        symbol: str,
        max_retries: int = 3,
    ) -> bool:
        """Retry a failed symbol."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            retry_count = session.retry_counts.get(symbol, 0)
            if retry_count >= max_retries:
                log.warning("Max retries reached for %s", symbol)
                return False
            
            return session.retry_symbol(symbol)
    
    def pause_session(self, session_id: str) -> bool:
        """Pause a running session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.status == "running":
                session.status = "paused"
                self._save_session(session)
                log.info("Session %s paused", session_id)
                return True
            return False
    
    def resume_session(self, session_id: str) -> bool:
        """Resume a paused session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.status == "paused":
                session.status = "running"
                session.started_at = datetime.now().isoformat()
                self._save_session(session)
                log.info("Session %s resumed", session_id)
                return True
            return False
    
    def complete_session(self, session_id: str) -> bool:
        """Mark a session as completed."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.status = "completed"
                session.completed_at = datetime.now().isoformat()
                self._save_session(session)
                log.info("Session %s completed", session_id)
                return True
            return False
    
    def fail_session(self, session_id: str, error: str = "") -> bool:
        """Mark a session as failed."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.status = "failed"
                session.last_error = error
                session.completed_at = datetime.now().isoformat()
                self._save_session(session)
                log.warning("Session %s failed: %s", session_id, error)
                return True
            return False
    
    def save_state(self) -> None:
        """Save all state to disk."""
        with self._lock:
            for session in self._sessions.values():
                self._save_session(session)
    
    def create_batch(
        self,
        session_id: str,
        symbols: list[str],
        batch_size: int = 50,
    ) -> list[BatchState]:
        """Create batches for a session.
        
        Args:
            session_id: Session ID
            symbols: Symbols to batch
            batch_size: Symbols per batch
        
        Returns:
            List of batch states
        """
        with self._lock:
            batches = []
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                batch_id = f"{session_id}_batch_{i // batch_size}"
                
                batch = BatchState(
                    batch_id=batch_id,
                    session_id=session_id,
                    symbols=batch_symbols,
                )
                
                batches.append(batch)
                self._batches[batch_id] = batch
            
            return batches
    
    def get_batch(self, batch_id: str) -> BatchState | None:
        """Get a batch by ID."""
        return self._batches.get(batch_id)
    
    def mark_batch_complete(self, batch_id: str, symbol: str) -> None:
        """Mark a symbol complete in a batch."""
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch:
                if symbol in batch.symbols and symbol not in batch.completed:
                    batch.completed.append(symbol)
                    if batch.started_at is None:
                        batch.started_at = datetime.now().isoformat()
                    if len(batch.completed) == len(batch.symbols):
                        batch.status = "completed"
                        batch.completed_at = datetime.now().isoformat()
    
    def cleanup_old_sessions(self) -> int:
        """Clean up old completed sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=self.retention_hours)
            cleaned = 0
            
            to_remove = []
            for session_id, session in self._sessions.items():
                if session.status == "completed" and session.completed_at:
                    completed = datetime.fromisoformat(session.completed_at)
                    if completed < cutoff:
                        to_remove.append(session_id)
            
            # Keep only max_sessions
            if len(self._sessions) - len(to_remove) > self.max_sessions:
                completed_sessions = [
                    (sid, s) for sid, s in self._sessions.items()
                    if s.status == "completed" and sid not in to_remove
                ]
                completed_sessions.sort(key=lambda x: x[1].completed_at or "")
                for sid, _ in completed_sessions[:len(self._sessions) - self.max_sessions]:
                    to_remove.append(sid)
            
            for session_id in to_remove:
                # Remove file
                session_path = self.state_dir / "sessions" / f"{session_id}.json"
                try:
                    if session_path.exists():
                        session_path.unlink()
                    del self._sessions[session_id]
                    cleaned += 1
                except Exception as e:
                    log.warning("Failed to cleanup session %s: %s", session_id, e)
            
            if cleaned > 0:
                log.info("Cleaned up %d old fetch sessions", cleaned)
            
            return cleaned
    
    def get_progress(self, session_id: str) -> dict[str, Any]:
        """Get session progress."""
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        return {
            "session_id": session.session_id,
            "status": session.status,
            "progress_percent": session.progress_percent,
            "completed": session.completed_count,
            "failed": session.failed_count,
            "pending": session.pending_count,
            "total": session.total_symbols,
            "last_symbol": session.last_symbol,
            "estimated_remaining_seconds": session.estimated_remaining_seconds,
        }
    
    def export_state(self, output_path: Path | str) -> bool:
        """Export all state to a file.
        
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
                "active_session_id": self._active_session_id,
                "sessions": [s.to_dict() for s in self._sessions.values()],
                "batches": [b.to_dict() for b in self._batches.values()],
            }
            
            atomic_write_json(output_path, export_data)
            log.info("Fetch state exported to %s", output_path)
            return True
        except Exception as e:
            log.error("Failed to export fetch state: %s", e)
            return False
    
    def import_state(self, input_path: Path | str) -> bool:
        """Import state from a file.
        
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
                self._active_session_id = data.get("active_session_id")
                
                for session_data in data.get("sessions", []):
                    session = FetchSession.from_dict(session_data)
                    self._sessions[session.session_id] = session
                
                for batch_data in data.get("batches", []):
                    batch = BatchState.from_dict(batch_data)
                    self._batches[batch.batch_id] = batch
            
            log.info("Fetch state imported from %s", input_path)
            return True
        except Exception as e:
            log.error("Failed to import fetch state: %s", e)
            return False


# Global instance
_fetch_recovery: FetchStateRecovery | None = None
_fetch_recovery_lock = threading.Lock()


def get_fetch_recovery() -> FetchStateRecovery:
    """Get or create the global fetch recovery instance."""
    global _fetch_recovery
    
    if _fetch_recovery is None:
        with _fetch_recovery_lock:
            if _fetch_recovery is None:
                _fetch_recovery = FetchStateRecovery()
    
    return _fetch_recovery


def reset_fetch_recovery() -> None:
    """Reset the global fetch recovery instance (for testing)."""
    global _fetch_recovery
    with _fetch_recovery_lock:
        _fetch_recovery = None
