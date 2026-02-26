"""Local LLM Chat and Control System.

Production-grade local AI assistant with:
- Offline model support (llama.cpp, Ollama, vLLM)
- Deterministic command parsing for trading control
- Human-in-the-loop confirmations
- Comprehensive audit logging
- RAG-based knowledge grounding
- Prompt injection detection
- Latency-optimized async inference
"""

from .local_llm import LocalLLM, LocalLLMConfig
from .command_parser import CommandParser, ParsedCommand, CommandType
from .safety_validator import SafetyValidator, ValidationLevel
from .audit_logger import AuditLogger, AuditEvent
from .context_manager import ContextManager, ConversationTurn
from .rag_engine import RAGEngine, KnowledgeDocument
from .security import PromptGuard, SecurityLevel

__all__ = [
    "LocalLLM",
    "LocalLLMConfig",
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
]
