"""Local LLM Chat and Control System.

Production-grade local AI assistant with:
- Self-training from internet data (no pre-trained models)
- Deterministic command parsing for trading control
- Human-in-the-loop confirmations
- Comprehensive audit logging
- RAG-based knowledge grounding
- Prompt injection detection
- Latency-optimized async inference
"""

from .command_parser import CommandParser, ParsedCommand, CommandType
from .safety_validator import SafetyValidator, ValidationLevel
from .audit_logger import AuditLogger, AuditEvent
from .context_manager import ContextManager, ConversationTurn
from .rag_engine import RAGEngine, KnowledgeDocument
from .security import PromptGuard, SecurityLevel
from .local_llm import LLMBackend, LocalLLM, LocalLLMConfig, LLMResponse

__all__ = [
    "CommandParser",
    "ParsedCommand",
    "CommandType",
    "SafetyValidator",
    "ValidationLevel",
    "AuditLogger",
    "AuditEvent",
    "ContextManager",
    "ConversationTurn",
    "RAGEngine",
    "KnowledgeDocument",
    "PromptGuard",
    "SecurityLevel",
    "LLMBackend",
    "LocalLLM",
    "LocalLLMConfig",
    "LLMResponse",
]
