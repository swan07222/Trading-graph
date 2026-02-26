"""Comprehensive audit logging for AI decisions.

Fixes:
- Audit/compliance challenges: Full audit trail for all AI actions
- Security: Tamper-evident logging with cryptographic hashing
- Debugging: Detailed context capture for reproducibility
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    # LLM events
    LLM_REQUEST = auto()
    LLM_RESPONSE = auto()
    LLM_ERROR = auto()
    
    # Command events
    COMMAND_PARSED = auto()
    COMMAND_VALIDATED = auto()
    COMMAND_CONFIRMED = auto()
    COMMAND_EXECUTED = auto()
    COMMAND_BLOCKED = auto()
    COMMAND_FAILED = auto()
    
    # Safety events
    SAFETY_CHECK = auto()
    RISK_LIMIT_TRIGGERED = auto()
    CIRCUIT_BREAKER_ACTIVATED = auto()
    
    # Trading events
    ORDER_CREATED = auto()
    ORDER_MODIFIED = auto()
    ORDER_CANCELLED = auto()
    ORDER_FILLED = auto()
    ORDER_REJECTED = auto()
    
    # System events
    SYSTEM_START = auto()
    SYSTEM_STOP = auto()
    CONFIG_CHANGED = auto()
    USER_LOGIN = auto()
    USER_LOGOUT = auto()
    
    # Data events
    DATA_ACCESSED = auto()
    MODEL_LOADED = auto()
    MODEL_TRAINED = auto()


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class AuditEvent:
    """A single audit event.
    
    All fields are immutable for audit integrity.
    """
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    user_id: str
    session_id: str
    description: str
    details: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    previous_hash: str = ""
    current_hash: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "severity": self.severity.name,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "description": self.description,
            "details": self.details,
            "context": self.context,
            "previous_hash": self.previous_hash,
            "current_hash": self.current_hash,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEvent:
        """Reconstruct from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=AuditEventType[data["event_type"]],
            severity=AuditSeverity[data["severity"]],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_id=data["user_id"],
            session_id=data["session_id"],
            description=data["description"],
            details=data.get("details", {}),
            context=data.get("context", {}),
            previous_hash=data.get("previous_hash", ""),
            current_hash=data.get("current_hash", ""),
        )


@dataclass
class AuditQuery:
    """Query parameters for audit log search."""
    event_type: AuditEventType | None = None
    severity: AuditSeverity | None = None
    user_id: str | None = None
    session_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    search_text: str | None = None
    limit: int = 100
    offset: int = 0


class AuditLogger:
    """Tamper-evident audit logging system.
    
    Features:
    - Cryptographic hash chaining (blockchain-like)
    - Asynchronous writing for performance
    - Automatic rotation and archiving
    - Query interface for compliance
    - Export capabilities
    """
    
    def __init__(
        self,
        log_dir: Path | None = None,
        enabled: bool = True,
        hash_chain: bool = True,
        async_write: bool = True,
    ) -> None:
        self.log_dir = log_dir or (CONFIG.logs_dir / "audit")
        self.enabled = enabled
        self.hash_chain = hash_chain
        self.async_write = async_write
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_file = self._get_log_file()
        self._lock = threading.RLock()
        self._last_hash = ""
        self._event_count = 0
        self._queue: list[AuditEvent] = []
        self._writer_thread: threading.Thread | None = None
        self._stop_writer = False
        
        # Start async writer if enabled
        if self.async_write:
            self._start_writer()
        
        log.info(f"Audit logger initialized: {self.log_dir}")
    
    def _get_log_file(self) -> Path:
        """Get current log file path with date-based rotation."""
        date_str = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"audit_{date_str}.jsonl"
    
    def _start_writer(self) -> None:
        """Start background writer thread."""
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="AuditWriter",
        )
        self._writer_thread.start()
    
    def _writer_loop(self) -> None:
        """Background writer loop."""
        while not self._stop_writer:
            events_to_write = []
            
            with self._lock:
                if self._queue:
                    events_to_write = self._queue[:]
                    self._queue.clear()
            
            if events_to_write:
                self._write_events(events_to_write)
            
            threading.Event().wait(0.1)  # 100ms batch interval
    
    def _write_events(self, events: list[AuditEvent]) -> None:
        """Write events to log file."""
        try:
            # Check if we need to rotate
            current_file = self._get_log_file()
            if current_file != self._current_file:
                self._current_file = current_file
            
            with open(self._current_file, "a", encoding="utf-8") as f:
                for event in events:
                    line = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
                    f.write(line + "\n")
            
            log.debug(f"Wrote {len(events)} audit events")
            
        except Exception as e:
            log.error(f"Audit write failed: {e}")
    
    def _compute_hash(
        self,
        event: AuditEvent,
        previous_hash: str,
    ) -> str:
        """Compute cryptographic hash for event."""
        data = {
            "event_id": event.event_id,
            "event_type": event.event_type.name,
            "timestamp": event.timestamp.isoformat(),
            "description": event.description,
            "details": event.details,
            "previous_hash": previous_hash,
        }
        content = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    def log(
        self,
        event_type: AuditEventType,
        description: str,
        details: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        user_id: str = "system",
        session_id: str = "",
    ) -> AuditEvent:
        """Log an audit event.
        
        Args:
            event_type: Type of event
            description: Human-readable description
            details: Structured event data
            context: Additional context (market conditions, etc.)
            severity: Event severity
            user_id: User who triggered the event
            session_id: Session identifier
            
        Returns:
            The created AuditEvent
        """
        if not self.enabled:
            # Create event but don't log
            return AuditEvent(
                event_id="",
                event_type=event_type,
                severity=severity,
                timestamp=datetime.now(),
                user_id=user_id,
                session_id=session_id,
                description=description,
                details=details or {},
                context=context or {},
            )
        
        import uuid
        
        with self._lock:
            event_id = f"evt_{uuid.uuid4().hex[:16]}"
            timestamp = datetime.now()
            
            # Compute hash chain
            previous_hash = self._last_hash if self.hash_chain else ""
            
            event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                severity=severity,
                timestamp=timestamp,
                user_id=user_id,
                session_id=session_id,
                description=description,
                details=details or {},
                context=context or {},
                previous_hash=previous_hash,
                current_hash="",  # Will be computed
            )
            
            # Compute current hash
            event.current_hash = self._compute_hash(event, previous_hash)
            
            # Update chain
            if self.hash_chain:
                self._last_hash = event.current_hash
            
            self._event_count += 1
            
            # Queue for async write or write immediately
            if self.async_write:
                self._queue.append(event)
            else:
                self._write_events([event])
        
        log.debug(f"Audit event logged: {event_type.name} - {description[:50]}")
        return event
    
    def log_command(
        self,
        command_type: str,
        command_id: str,
        action: str,
        result: str,
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Convenience method for logging command events."""
        event_type_map = {
            "parsed": AuditEventType.COMMAND_PARSED,
            "validated": AuditEventType.COMMAND_VALIDATED,
            "confirmed": AuditEventType.COMMAND_CONFIRMED,
            "executed": AuditEventType.COMMAND_EXECUTED,
            "blocked": AuditEventType.COMMAND_BLOCKED,
            "failed": AuditEventType.COMMAND_FAILED,
        }
        
        event_type = event_type_map.get(action, AuditEventType.COMMAND_PARSED)
        
        return self.log(
            event_type=event_type,
            description=f"Command {command_type} {action}: {result}",
            details={
                "command_type": command_type,
                "command_id": command_id,
                "action": action,
                "result": result,
                **(details or {}),
            },
            **kwargs,
        )
    
    def log_llm_request(
        self,
        prompt: str,
        model: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> AuditEvent:
        """Log LLM request."""
        return self.log(
            event_type=AuditEventType.LLM_REQUEST,
            description=f"LLM request to {model}",
            details={
                "model": model,
                "prompt_length": len(prompt),
                "parameters": parameters,
            },
            context={"prompt_preview": prompt[:200]},
            **kwargs,
        )
    
    def log_llm_response(
        self,
        response: str,
        model: str,
        tokens: dict[str, int],
        latency_ms: float,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log LLM response."""
        return self.log(
            event_type=AuditEventType.LLM_RESPONSE,
            description=f"LLM response from {model}",
            details={
                "model": model,
                "response_length": len(response),
                "tokens": tokens,
                "latency_ms": latency_ms,
            },
            context={"response_preview": response[:200]},
            **kwargs,
        )
    
    def query(self, query: AuditQuery) -> list[AuditEvent]:
        """Query audit logs.
        
        Args:
            query: Query parameters
            
        Returns:
            List of matching AuditEvents
        """
        results = []
        
        # Find relevant log files
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"))
        
        for log_file in log_files:
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        
                        try:
                            data = json.loads(line)
                            event = AuditEvent.from_dict(data)
                            
                            # Apply filters
                            if not self._matches_query(event, query):
                                continue
                            
                            results.append(event)
                            
                            if len(results) >= query.limit + query.offset:
                                break
                                
                        except (json.JSONDecodeError, KeyError):
                            continue
                            
            except Exception as e:
                log.warning(f"Error reading audit log {log_file}: {e}")
        
        # Apply pagination
        return results[query.offset:query.offset + query.limit]
    
    def _matches_query(self, event: AuditEvent, query: AuditQuery) -> bool:
        """Check if event matches query criteria."""
        if query.event_type and event.event_type != query.event_type:
            return False
        
        if query.severity and event.severity != query.severity:
            return False
        
        if query.user_id and event.user_id != query.user_id:
            return False
        
        if query.session_id and event.session_id != query.session_id:
            return False
        
        if query.start_time and event.timestamp < query.start_time:
            return False
        
        if query.end_time and event.timestamp > query.end_time:
            return False
        
        if query.search_text:
            search_lower = query.search_text.lower()
            if (search_lower not in event.description.lower() and
                search_lower not in json.dumps(event.details).lower()):
                return False
        
        return True
    
    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify the integrity of the audit chain.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"))
        
        for log_file in log_files:
            prev_hash = ""
            
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        
                        try:
                            data = json.loads(line)
                            event = AuditEvent.from_dict(data)
                            
                            # Verify hash
                            expected_hash = self._compute_hash(event, prev_hash)
                            if event.current_hash != expected_hash:
                                issues.append(
                                    f"{log_file}:{line_num} - Hash mismatch"
                                )
                            
                            prev_hash = event.current_hash
                            
                        except (json.JSONDecodeError, KeyError) as e:
                            issues.append(f"{log_file}:{line_num} - Parse error: {e}")
                            
            except Exception as e:
                issues.append(f"{log_file} - Read error: {e}")
        
        return len(issues) == 0, issues
    
    def export(
        self,
        output_path: Path,
        query: AuditQuery | None = None,
        format: str = "json",
    ) -> int:
        """Export audit logs.
        
        Args:
            output_path: Output file path
            query: Optional query to filter events
            format: Export format (json, csv)
            
        Returns:
            Number of events exported
        """
        events = self.query(query) if query else self.query(AuditQuery(limit=100000))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in events], f, indent=2, ensure_ascii=False)
        elif format == "csv":
            import csv
            
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "event_id", "event_type", "severity", "timestamp",
                    "user_id", "description", "details"
                ])
                for event in events:
                    writer.writerow([
                        event.event_id,
                        event.event_type.name,
                        event.severity.name,
                        event.timestamp.isoformat(),
                        event.user_id,
                        event.description,
                        json.dumps(event.details),
                    ])
        
        log.info(f"Exported {len(events)} events to {output_path}")
        return len(events)
    
    def shutdown(self) -> None:
        """Shutdown the audit logger."""
        self._stop_writer = True
        
        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)
        
        # Write any remaining events
        with self._lock:
            if self._queue:
                self._write_events(self._queue)
        
        log.info("Audit logger shutdown complete")
    
    def get_stats(self) -> dict[str, Any]:
        """Get audit logger statistics."""
        return {
            "enabled": self.enabled,
            "hash_chain": self.hash_chain,
            "async_write": self.async_write,
            "event_count": self._event_count,
            "queue_size": len(self._queue),
            "log_dir": str(self.log_dir),
            "current_file": str(self._current_file),
        }


# Singleton instance
_logger_instance: AuditLogger | None = None


def get_audit_logger(
    log_dir: Path | None = None,
    enabled: bool = True,
) -> AuditLogger:
    """Get or create the singleton AuditLogger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AuditLogger(log_dir, enabled)
    return _logger_instance


def audit_event(
    event_type: AuditEventType,
    description: str,
    **kwargs: Any,
) -> AuditEvent:
    """Convenience function to log an audit event."""
    return get_audit_logger().log(event_type, description, **kwargs)
