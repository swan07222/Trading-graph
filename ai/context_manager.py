"""Context management with conversation history persistence.

Fixes:
- Context window limits: Smart summarization and truncation
- Memory loss: Persistent conversation history
- Reproducibility: Full context capture for debugging
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class MessageRole(Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ConversationStatus(Enum):
    """Conversation lifecycle status."""
    ACTIVE = auto()
    PAUSED = auto()
    ARCHIVED = auto()
    CLOSED = auto()


@dataclass
class Message:
    """A single message in conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_id=data.get("message_id", ""),
            metadata=data.get("metadata", {}),
        )
    
    @property
    def token_estimate(self) -> int:
        """Estimate token count (rough approximation)."""
        # ~4 characters per token for English, ~2 for Chinese
        zh_chars = sum(1 for c in self.content if '\u4e00' <= c <= '\u9fff')
        other_chars = len(self.content) - zh_chars
        return (zh_chars // 2) + (other_chars // 4) + 10


@dataclass
class ConversationTurn:
    """A complete conversation turn (user + assistant)."""
    user_message: Message
    assistant_message: Message | None = None
    turn_id: str = ""
    command_id: str | None = None
    execution_result: dict[str, Any] | None = None
    duration_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_message": self.user_message.to_dict(),
            "assistant_message": self.assistant_message.to_dict() if self.assistant_message else None,
            "command_id": self.command_id,
            "execution_result": self.execution_result,
            "duration_ms": self.duration_ms,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationTurn:
        return cls(
            turn_id=data.get("turn_id", ""),
            user_message=Message.from_dict(data["user_message"]),
            assistant_message=Message.from_dict(data["assistant_message"]) if data.get("assistant_message") else None,
            command_id=data.get("command_id"),
            execution_result=data.get("execution_result"),
            duration_ms=data.get("duration_ms", 0.0),
        )


@dataclass
class ConversationSummary:
    """Compressed summary of conversation history."""
    summary_text: str
    key_points: list[str]
    entities_mentioned: list[str]
    commands_executed: list[str]
    time_range: tuple[datetime, datetime]
    turn_count: int
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_text": self.summary_text,
            "key_points": self.key_points,
            "entities_mentioned": self.entities_mentioned,
            "commands_executed": self.commands_executed,
            "time_range": (
                self.time_range[0].isoformat(),
                self.time_range[1].isoformat(),
            ),
            "turn_count": self.turn_count,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationSummary:
        """Deserialize summary payload with datetime normalization."""
        raw_range = data.get("time_range", ())
        if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
            start_raw, end_raw = raw_range
        else:
            now_iso = datetime.now().isoformat()
            start_raw, end_raw = now_iso, now_iso

        def _as_datetime(value: Any) -> datetime:
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    return datetime.now()
            return datetime.now()

        return cls(
            summary_text=str(data.get("summary_text", "")),
            key_points=list(data.get("key_points", [])),
            entities_mentioned=list(data.get("entities_mentioned", [])),
            commands_executed=list(data.get("commands_executed", [])),
            time_range=(_as_datetime(start_raw), _as_datetime(end_raw)),
            turn_count=int(data.get("turn_count", 0)),
            created_at=_as_datetime(data.get("created_at", datetime.now().isoformat())),
        )


@dataclass
class Conversation:
    """A complete conversation session."""
    conversation_id: str
    user_id: str
    session_id: str
    status: ConversationStatus = ConversationStatus.ACTIVE
    messages: list[Message] = field(default_factory=list)
    turns: list[ConversationTurn] = field(default_factory=list)
    summary: ConversationSummary | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a conversation turn."""
        self.turns.append(turn)
        if turn.user_message:
            self.messages.append(turn.user_message)
        if turn.assistant_message:
            self.messages.append(turn.assistant_message)
        self.updated_at = datetime.now()
    
    @property
    def total_tokens(self) -> int:
        """Estimate total tokens in conversation."""
        return sum(m.token_estimate for m in self.messages)
    
    @property
    def is_active(self) -> bool:
        """Check if conversation is active."""
        return self.status == ConversationStatus.ACTIVE
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "status": self.status.name,
            "messages": [m.to_dict() for m in self.messages],
            "turns": [t.to_dict() for t in self.turns],
            "summary": self.summary.to_dict() if self.summary else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "total_tokens": self.total_tokens,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conversation:
        return cls(
            conversation_id=data["conversation_id"],
            user_id=data["user_id"],
            session_id=data["session_id"],
            status=ConversationStatus[data["status"]],
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            turns=[ConversationTurn.from_dict(t) for t in data.get("turns", [])],
            summary=ConversationSummary.from_dict(data["summary"]) if data.get("summary") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


class ContextManager:
    """Manages conversation context with smart truncation and persistence.
    
    Features:
    - Automatic context window management
    - Conversation summarization
    - Persistent storage
    - Multi-session support
    - Context compression
    """
    
    def __init__(
        self,
        storage_dir: Path | None = None,
        max_context_window: int = 8192,
        max_conversations: int = 100,
        summary_threshold: int = 5000,
        auto_save: bool = True,
        ttl_hours: int = 24 * 7,  # 7 days
    ) -> None:
        self.storage_dir = storage_dir or (CONFIG.cache_dir / "conversations")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_context_window = max_context_window
        self.max_conversations = max_conversations
        self.summary_threshold = summary_threshold
        self.auto_save = auto_save
        self.ttl_hours = ttl_hours
        
        self._conversations: dict[str, Conversation] = {}
        self._active_session: str | None = None
        self._lock = threading.RLock()
        
        # Load existing conversations
        self._load_conversations()
        
        log.info(f"ContextManager initialized: {self.storage_dir}")
    
    def create_conversation(
        self,
        user_id: str = "default",
        session_id: str | None = None,
        system_prompt: str | None = None,
    ) -> Conversation:
        """Create a new conversation.
        
        Args:
            user_id: User identifier
            session_id: Optional session ID (auto-generated if None)
            system_prompt: Optional system prompt
            
        Returns:
            New Conversation instance
        """
        import uuid
        
        with self._lock:
            conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
            session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
            
            conversation = Conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                session_id=session_id,
            )
            
            # Add system prompt if provided
            if system_prompt:
                conversation.add_message(Message(
                    role=MessageRole.SYSTEM,
                    content=system_prompt,
                    message_id=f"msg_system_{uuid.uuid4().hex[:8]}",
                ))
            
            self._conversations[conversation_id] = conversation
            self._active_session = conversation_id
            
            # Enforce max conversations limit
            self._enforce_max_conversations()
            
            log.info(f"Created conversation: {conversation_id}")
            return conversation
    
    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID."""
        with self._lock:
            return self._conversations.get(conversation_id)
    
    def get_active_conversation(self) -> Conversation | None:
        """Get the currently active conversation."""
        with self._lock:
            if self._active_session:
                return self._conversations.get(self._active_session)
            return None
    
    def set_active(self, conversation_id: str) -> bool:
        """Set the active conversation."""
        with self._lock:
            if conversation_id in self._conversations:
                self._active_session = conversation_id
                return True
            return False
    
    def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message | None:
        """Add a message to a conversation.
        
        Args:
            conversation_id: Target conversation
            role: Message role
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Created Message or None if conversation not found
        """
        import uuid
        
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation:
                return None
            
            message = Message(
                role=role,
                content=content,
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                metadata=metadata or {},
            )
            
            conversation.add_message(message)
            
            # Check if summarization needed
            if conversation.total_tokens > self.summary_threshold:
                self._summarize_conversation(conversation)
            
            # Auto-save
            if self.auto_save:
                self._save_conversation(conversation)
            
            return message
    
    def add_turn(
        self,
        conversation_id: str,
        user_content: str,
        assistant_content: str | None = None,
        command_id: str | None = None,
        execution_result: dict[str, Any] | None = None,
    ) -> ConversationTurn | None:
        """Add a complete conversation turn.
        
        Args:
            conversation_id: Target conversation
            user_content: User message content
            assistant_content: Assistant response (optional)
            command_id: Associated command ID
            execution_result: Command execution result
            
        Returns:
            Created ConversationTurn or None if conversation not found
        """
        import uuid
        
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation:
                return None
            
            turn = ConversationTurn(
                turn_id=f"turn_{uuid.uuid4().hex[:8]}",
                user_message=Message(
                    role=MessageRole.USER,
                    content=user_content,
                    message_id=f"msg_user_{uuid.uuid4().hex[:8]}",
                ),
                assistant_message=Message(
                    role=MessageRole.ASSISTANT,
                    content=assistant_content or "",
                    message_id=f"msg_assistant_{uuid.uuid4().hex[:8]}",
                ) if assistant_content else None,
                command_id=command_id,
                execution_result=execution_result,
            )
            
            conversation.add_turn(turn)
            
            # Check if summarization needed
            if conversation.total_tokens > self.summary_threshold:
                self._summarize_conversation(conversation)
            
            # Auto-save
            if self.auto_save:
                self._save_conversation(conversation)
            
            return turn
    
    def get_context_for_llm(
        self,
        conversation_id: str,
        max_tokens: int | None = None,
        include_summary: bool = True,
    ) -> list[Message]:
        """Get context messages formatted for LLM.
        
        Implements smart truncation:
        1. Keep system messages
        2. Include summary if available
        3. Add recent messages up to token limit
        
        Args:
            conversation_id: Target conversation
            max_tokens: Override default max tokens
            include_summary: Include conversation summary
            
        Returns:
            List of messages for LLM context
        """
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation:
                return []
            
            max_tokens = max_tokens or self.max_context_window
            context_messages: list[Message] = []
            
            # 1. Add system message first
            for msg in conversation.messages:
                if msg.role == MessageRole.SYSTEM:
                    context_messages.append(msg)
                    break
            
            # 2. Add summary if available and requested
            if include_summary and conversation.summary:
                summary_msg = Message(
                    role=MessageRole.SYSTEM,
                    content=f"[Conversation Summary]\n{conversation.summary.summary_text}\n\nKey points: {', '.join(conversation.summary.key_points)}",
                )
                context_messages.append(summary_msg)
            
            # 3. Add recent messages (newest first, then reverse)
            recent_messages: list[Message] = []
            current_tokens = sum(m.token_estimate for m in context_messages)
            
            for msg in reversed(conversation.messages):
                if msg.role == MessageRole.SYSTEM:
                    continue  # Already added
                
                msg_tokens = msg.token_estimate
                if current_tokens + msg_tokens > max_tokens:
                    break
                
                recent_messages.append(msg)
                current_tokens += msg_tokens
            
            # Add in correct order (oldest first)
            context_messages.extend(reversed(recent_messages))
            
            return context_messages
    
    def _summarize_conversation(self, conversation: Conversation) -> None:
        """Create a summary of the conversation.
        
        This is a simple extractive summary.
        For better results, integrate with LLM.
        """
        # Extract key information
        commands_executed = []
        entities_mentioned = set()
        key_points = []
        
        for turn in conversation.turns:
            if turn.command_id:
                commands_executed.append(turn.command_id)
            
            # Extract stock codes (6-digit patterns)
            import re
            for content in [turn.user_message.content, turn.assistant_message.content if turn.assistant_message else ""]:
                stocks = re.findall(r'\b(\d{6})\b', content)
                entities_mentioned.update(stocks)
                
                # Extract first sentence as key point
                if '.' in content:
                    sentence = content.split('.')[0].strip()
                    if len(sentence) > 20 and len(key_points) < 10:
                        key_points.append(sentence)
        
        # Create summary text
        time_range = (
            conversation.created_at,
            conversation.updated_at,
        )
        
        summary = ConversationSummary(
            summary_text=f"Conversation with {len(conversation.turns)} turns. "
                        f"Executed {len(commands_executed)} commands. "
                        f"Discussed {len(entities_mentioned)} stocks.",
            key_points=key_points[:10],
            entities_mentioned=list(entities_mentioned),
            commands_executed=commands_executed,
            time_range=time_range,
            turn_count=len(conversation.turns),
        )
        
        conversation.summary = summary
        log.info(f"Summarized conversation: {conversation.conversation_id}")
    
    def _save_conversation(self, conversation: Conversation) -> None:
        """Save conversation to disk."""
        file_path = self.storage_dir / f"{conversation.conversation_id}.json"
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(conversation.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"Failed to save conversation: {e}")
    
    def _load_conversations(self) -> None:
        """Load conversations from disk."""
        try:
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, encoding="utf-8") as f:
                        data = json.load(f)
                        conversation = Conversation.from_dict(data)
                        
                        # Check TTL
                        age = datetime.now() - conversation.updated_at
                        if age < timedelta(hours=self.ttl_hours):
                            self._conversations[conversation.conversation_id] = conversation
                        else:
                            # Delete expired
                            file_path.unlink()
                            
                except Exception as e:
                    log.warning(f"Failed to load conversation {file_path}: {e}")
                    
        except Exception as e:
            log.warning(f"Failed to load conversations: {e}")
    
    def _enforce_max_conversations(self) -> None:
        """Remove oldest conversations if over limit."""
        if len(self._conversations) <= self.max_conversations:
            return
        
        # Sort by updated_at
        sorted_convs = sorted(
            self._conversations.values(),
            key=lambda c: c.updated_at,
        )
        
        # Remove oldest
        to_remove = len(self._conversations) - self.max_conversations
        for conv in sorted_convs[:to_remove]:
            del self._conversations[conv.conversation_id]
            file_path = self.storage_dir / f"{conv.conversation_id}.json"
            if file_path.exists():
                file_path.unlink()
    
    def close_conversation(self, conversation_id: str) -> bool:
        """Close a conversation."""
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation:
                return False
            
            conversation.status = ConversationStatus.CLOSED
            conversation.updated_at = datetime.now()
            
            if self.auto_save:
                self._save_conversation(conversation)
            
            if self._active_session == conversation_id:
                self._active_session = None
            
            log.info(f"Closed conversation: {conversation_id}")
            return True
    
    def list_conversations(
        self,
        user_id: str | None = None,
        status: ConversationStatus | None = None,
        limit: int = 20,
    ) -> list[Conversation]:
        """List conversations with optional filters."""
        with self._lock:
            result = list(self._conversations.values())
            
            if user_id:
                result = [c for c in result if c.user_id == user_id]
            
            if status:
                result = [c for c in result if c.status == status]
            
            # Sort by updated_at (newest first)
            result.sort(key=lambda c: c.updated_at, reverse=True)
            
            return result[:limit]
    
    def export_conversation(
        self,
        conversation_id: str,
        output_path: Path,
        format: str = "json",
    ) -> bool:
        """Export a conversation to file."""
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation:
                return False
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(conversation.to_dict(), f, ensure_ascii=False, indent=2)
            elif format == "txt":
                with open(output_path, "w", encoding="utf-8") as f:
                    for msg in conversation.messages:
                        f.write(f"[{msg.role.value}] {msg.content}\n\n")
            elif format == "md":
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"# Conversation: {conversation_id}\n\n")
                    for msg in conversation.messages:
                        role = msg.role.value.capitalize()
                        f.write(f"## {role}\n\n{msg.content}\n\n")
            
            log.info(f"Exported conversation to {output_path}")
            return True
    
    def get_stats(self) -> dict[str, Any]:
        """Get context manager statistics."""
        with self._lock:
            total_tokens = sum(c.total_tokens for c in self._conversations.values())
            active_count = sum(1 for c in self._conversations.values() if c.is_active)
            
            return {
                "total_conversations": len(self._conversations),
                "active_conversations": active_count,
                "total_tokens": total_tokens,
                "storage_dir": str(self.storage_dir),
                "max_context_window": self.max_context_window,
                "auto_save": self.auto_save,
            }
    
    def clear_all(self) -> None:
        """Clear all conversations."""
        with self._lock:
            self._conversations.clear()
            self._active_session = None
            
            # Clear disk storage
            for file_path in self.storage_dir.glob("*.json"):
                file_path.unlink()
            
            log.info("Cleared all conversations")


# Singleton instance
_manager_instance: ContextManager | None = None


def get_context_manager(
    storage_dir: Path | None = None,
    **kwargs: Any,
) -> ContextManager:
    """Get or create the singleton ContextManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ContextManager(storage_dir, **kwargs)
    return _manager_instance
