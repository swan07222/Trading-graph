"""Main AI controller integrating all components.

This is the central orchestrator for the local LLM chat and control system.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from config.settings import CONFIG
from utils.logger import get_logger

from .local_llm import LocalLLM, LocalLLMConfig, LLMResponse, get_llm, initialize_llm
from .command_parser import CommandParser, ParsedCommand, CommandType, get_parser
from .safety_validator import SafetyValidator, ValidationLevel, SafetyReport, get_validator
from .audit_logger import AuditLogger, AuditEventType, AuditSeverity, get_audit_logger
from .context_manager import ContextManager, MessageRole, get_context_manager
from .rag_engine import RAGEngine, DocumentSource, get_rag_engine
from .security import PromptGuard, SecurityLevel, get_prompt_guard

log = get_logger(__name__)


@dataclass
class ChatConfig:
    """Configuration for AI chat system."""
    # LLM settings
    llm_backend: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    llm_host: str = "127.0.0.1"
    llm_port: int = 11434
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # Security settings
    security_level: str = "high"
    require_confirmation: bool = True
    
    # Validation settings
    validation_level: str = "standard"
    
    # Context settings
    max_context_turns: int = 20
    enable_summarization: bool = True
    
    # RAG settings
    enable_rag: bool = True
    rag_top_k: int = 5
    
    # Audit settings
    enable_audit_log: bool = True
    
    # System prompt
    system_prompt: str = """You are an AI assistant for stock trading analysis.
You help users with:
- Market data queries
- Stock analysis and predictions
- Sentiment analysis
- Portfolio management
- Trading education

Always be honest about uncertainties. Never provide financial advice without disclaimers.
For trading commands, always confirm before execution."""


@dataclass
class ChatResponse:
    """Response from AI chat system."""
    response_text: str
    command: ParsedCommand | None = None
    safety_report: SafetyReport | None = None
    requires_confirmation: bool = False
    context_used: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    sources: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "response_text": self.response_text,
            "command": self.command.to_dict() if self.command else None,
            "safety_report": self.safety_report.to_dict() if self.safety_report else None,
            "requires_confirmation": self.requires_confirmation,
            "context_used": self.context_used,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "sources": self.sources,
        }


class AIController:
    """Main controller for local LLM chat and trading control.
    
    Orchestrates:
    - LLM inference
    - Command parsing
    - Safety validation
    - Context management
    - RAG retrieval
    - Security checks
    - Audit logging
    """
    
    def __init__(self, config: ChatConfig | None = None) -> None:
        self.config = config or ChatConfig()
        
        # Components (lazy initialized)
        self._llm: LocalLLM | None = None
        self._parser: CommandParser | None = None
        self._validator: SafetyValidator | None = None
        self._audit: AuditLogger | None = None
        self._context: ContextManager | None = None
        self._rag: RAGEngine | None = None
        self._security: PromptGuard | None = None
        
        # State
        self._initialized = False
        self._user_id = "default"
        self._session_id = ""
        
        log.info("AIController created")
    
    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return
        
        log.info("Initializing AIController...")
        
        # Initialize LLM
        llm_config = LocalLLMConfig(
            backend=self._get_backend_enum(self.config.llm_backend),
            model_name=self.config.llm_model,
            host=self.config.llm_host,
            port=self.config.llm_port,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        self._llm = get_llm(llm_config)
        await self._llm.initialize()
        
        # Initialize other components
        self._parser = get_parser()
        
        self._validator = get_validator(
            self._get_validation_level(self.config.validation_level)
        )
        
        self._audit = get_audit_logger(enabled=self.config.enable_audit_log)
        
        self._context = get_context_manager()
        
        self._rag = get_rag_engine() if self.config.enable_rag else None
        
        self._security = get_prompt_guard(
            self._get_security_level(self.config.security_level)
        )
        
        # Create initial conversation
        self._session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._context.create_conversation(
            user_id=self._user_id,
            session_id=self._session_id,
            system_prompt=self.config.system_prompt,
        )
        
        self._initialized = True
        log.info("AIController initialized")
    
    def _get_backend_enum(self, backend_str: str):
        """Convert backend string to enum."""
        from .local_llm import LLMBackend
        return LLMBackend(backend_str.lower())
    
    def _get_security_level(self, level_str: str):
        """Convert security level string to enum."""
        from .security import SecurityLevel
        return SecurityLevel[level_str.upper()]
    
    def _get_validation_level(self, level_str: str):
        """Convert validation level string to enum."""
        from .safety_validator import ValidationLevel
        return ValidationLevel[level_str.upper()]
    
    async def chat(
        self,
        message: str,
        user_id: str | None = None,
    ) -> ChatResponse:
        """Process a chat message.
        
        Args:
            message: User input message
            user_id: Optional user ID
            
        Returns:
            ChatResponse with AI response
        """
        import time
        start = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        if user_id:
            self._user_id = user_id
        
        # Security check
        security_report = self._security.analyze(message)
        if not security_report.is_safe:
            self._audit.log(
                event_type=AuditEventType.LLM_ERROR,
                description="Security check failed",
                details={"threats": [t.to_dict() for t in security_report.threats]},
                user_id=self._user_id,
                session_id=self._session_id,
            )
            return ChatResponse(
                response_text="⚠️ Your message triggered security filters. Please rephrase.",
                requires_confirmation=False,
            )
        
        # Get context for LLM
        conversation = self._context.get_active_conversation()
        context_messages = self._context.get_context_for_llm(
            conversation.conversation_id,
            include_summary=self.config.enable_summarization,
        )
        
        # RAG retrieval
        rag_context = ""
        sources = []
        if self._rag:
            rag_context = self._rag.get_context_for_query(
                message,
                top_k=self.config.rag_top_k,
            )
            sources = [r.document.metadata.get("source", "unknown") for r in self._rag.retrieve(message, top_k=1)]
        
        # Build prompt
        prompt = self._build_prompt(message, rag_context)
        
        # Log LLM request
        self._audit.log_llm_request(
            prompt=message,
            model=self.config.llm_model,
            parameters={"temperature": self.config.temperature},
            user_id=self._user_id,
            session_id=self._session_id,
        )
        
        # Generate response
        llm_response = await self._llm.generate(
            prompt=prompt,
            system_prompt=self.config.system_prompt,
        )
        
        # Log LLM response
        self._audit.log_llm_response(
            response=llm_response.content,
            model=llm_response.model,
            tokens={
                "prompt": llm_response.prompt_tokens,
                "completion": llm_response.completion_tokens,
                "total": llm_response.total_tokens,
            },
            latency_ms=llm_response.latency_ms,
            user_id=self._user_id,
            session_id=self._session_id,
        )
        
        # Parse command from response
        command = self._parser.parse(llm_response.content)
        
        # Validate command if trading-related
        safety_report = None
        requires_confirmation = False
        
        if command.command_type != CommandType.UNKNOWN:
            safety_report = self._validator.validate(command)
            requires_confirmation = command.requires_confirmation or safety_report.requires_approval
            
            # Log command
            self._audit.log_command(
                command_type=command.command_type.name,
                command_id=command.command_id,
                action="parsed",
                result="success" if safety_report.is_approved() else "blocked",
                user_id=self._user_id,
                session_id=self._session_id,
            )
        
        # Add to context
        self._context.add_turn(
            conversation.conversation_id,
            user_content=message,
            assistant_content=llm_response.content,
            command_id=command.command_id if command.command_type != CommandType.UNKNOWN else None,
        )
        
        latency_ms = (time.time() - start) * 1000
        
        return ChatResponse(
            response_text=llm_response.content,
            command=command if command.command_type != CommandType.UNKNOWN else None,
            safety_report=safety_report,
            requires_confirmation=requires_confirmation,
            context_used=rag_context,
            latency_ms=latency_ms,
            tokens_used=llm_response.total_tokens,
            sources=sources,
        )
    
    def _build_prompt(self, message: str, rag_context: str) -> str:
        """Build the final prompt for LLM."""
        prompt = message
        
        if rag_context:
            prompt = f"""Context information:
{rag_context}

Based on the above context, respond to: {message}"""
        
        return prompt
    
    async def chat_stream(
        self,
        message: str,
        user_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat response.
        
        Yields:
            Chunks of response text
        """
        if not self._initialized:
            await self.initialize()
        
        # Security check first
        security_report = self._security.analyze(message)
        if not security_report.is_safe:
            yield "⚠️ Your message triggered security filters."
            return
        
        # Get context
        conversation = self._context.get_active_conversation()
        
        # RAG retrieval
        rag_context = ""
        if self._rag:
            rag_context = self._rag.get_context_for_query(message)
        
        # Build prompt
        prompt = self._build_prompt(message, rag_context)
        
        # Stream response
        async for chunk in self._llm.generate_stream(
            prompt=prompt,
            system_prompt=self.config.system_prompt,
        ):
            yield chunk
        
        # Add to context (full message will be added after streaming)
        # Note: For streaming, we'd need to accumulate the full response
    
    async def execute_command(
        self,
        command: ParsedCommand,
        confirmation: bool = False,
    ) -> dict[str, Any]:
        """Execute a parsed command.
        
        Args:
            command: Parsed command to execute
            confirmation: User confirmation status
            
        Returns:
            Execution result
        """
        if not self._initialized:
            await self.initialize()
        
        # Check if confirmation required
        if command.requires_confirmation and not confirmation:
            return {
                "status": "pending_confirmation",
                "message": "This command requires confirmation. Please confirm to proceed.",
                "command": command.to_dict(),
            }
        
        # Validate command
        safety_report = self._validator.validate(command)
        
        if not safety_report.can_execute():
            self._audit.log_command(
                command_type=command.command_type.name,
                command_id=command.command_id,
                action="blocked",
                result="safety_check_failed",
                details={"reasons": safety_report.blocked_reasons},
                user_id=self._user_id,
                session_id=self._session_id,
            )
            return {
                "status": "blocked",
                "message": "Command blocked by safety checks",
                "reasons": safety_report.blocked_reasons,
            }
        
        # Execute based on command type
        result = await self._execute_command_impl(command)
        
        # Log execution
        self._audit.log_command(
            command_type=command.command_type.name,
            command_id=command.command_id,
            action="executed",
            result=result.get("status", "unknown"),
            user_id=self._user_id,
            session_id=self._session_id,
        )
        
        return result
    
    async def _execute_command_impl(
        self,
        command: ParsedCommand,
    ) -> dict[str, Any]:
        """Internal command execution."""
        cmd_type = command.command_type
        
        if cmd_type == CommandType.GET_QUOTE:
            return await self._exec_get_quote(command)
        elif cmd_type == CommandType.GET_PREDICTION:
            return await self._exec_get_prediction(command)
        elif cmd_type == CommandType.GET_SENTIMENT:
            return await self._exec_get_sentiment(command)
        elif cmd_type == CommandType.GET_PORTFOLIO:
            return await self._exec_get_portfolio(command)
        elif cmd_type == CommandType.HEALTH_CHECK:
            return await self._exec_health_check(command)
        else:
            return {
                "status": "not_implemented",
                "message": f"Command {cmd_type.name} not yet implemented",
            }
    
    async def _exec_get_quote(self, command: ParsedCommand) -> dict[str, Any]:
        """Execute GET_QUOTE command."""
        symbol = command.get_param("symbol")
        if not symbol:
            return {"status": "error", "message": "Symbol required"}
        
        # TODO: Integrate with actual data fetcher
        return {
            "status": "success",
            "symbol": symbol,
            "message": f"Quote lookup for {symbol} - integration pending",
        }
    
    async def _exec_get_prediction(self, command: ParsedCommand) -> dict[str, Any]:
        """Execute GET_PREDICTION command."""
        symbol = command.get_param("symbol")
        if not symbol:
            return {"status": "error", "message": "Symbol required"}
        
        # TODO: Integrate with predictor
        return {
            "status": "success",
            "symbol": symbol,
            "message": f"Prediction for {symbol} - integration pending",
        }
    
    async def _exec_get_sentiment(self, command: ParsedCommand) -> dict[str, Any]:
        """Execute GET_SENTIMENT command."""
        symbol = command.get_param("symbol")
        if not symbol:
            return {"status": "error", "message": "Symbol required"}
        
        # TODO: Integrate with sentiment analyzer
        return {
            "status": "success",
            "symbol": symbol,
            "message": f"Sentiment analysis for {symbol} - integration pending",
        }
    
    async def _exec_get_portfolio(self, command: ParsedCommand) -> dict[str, Any]:
        """Execute GET_PORTFOLIO command."""
        # TODO: Integrate with portfolio system
        return {
            "status": "success",
            "message": "Portfolio view - integration pending",
        }
    
    async def _exec_health_check(self, command: ParsedCommand) -> dict[str, Any]:
        """Execute HEALTH_CHECK command."""
        return {
            "status": "success",
            "components": {
                "llm": "initialized" if self._llm else "not_initialized",
                "parser": "initialized" if self._parser else "not_initialized",
                "validator": "initialized" if self._validator else "not_initialized",
                "audit": "initialized" if self._audit else "not_initialized",
                "context": "initialized" if self._context else "not_initialized",
                "rag": "initialized" if self._rag else "disabled",
                "security": "initialized" if self._security else "not_initialized",
            },
        }
    
    def add_market_data(
        self,
        symbol: str,
        data: dict[str, Any],
    ) -> None:
        """Add market data to RAG knowledge base."""
        if self._rag:
            self._rag.add_market_data(symbol, data)
    
    def add_news(
        self,
        title: str,
        content: str,
        source: str = "",
        symbols: list[str] | None = None,
    ) -> None:
        """Add news to RAG knowledge base."""
        if self._rag:
            self._rag.add_news(title, content, source, symbols=symbols)
    
    def add_sentiment(
        self,
        symbol: str,
        sentiment_score: float,
        analysis: str,
    ) -> None:
        """Add sentiment analysis to RAG knowledge base."""
        if self._rag:
            self._rag.add_sentiment(symbol, sentiment_score, analysis)
    
    def get_conversation_history(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get conversation history."""
        if not self._context:
            return []
        
        conversation = self._context.get_active_conversation()
        if not conversation:
            return []
        
        return [turn.to_dict() for turn in conversation.turns[-limit:]]
    
    def clear_conversation(self) -> None:
        """Clear current conversation."""
        if self._context:
            conversation = self._context.get_active_conversation()
            if conversation:
                self._context.close_conversation(conversation.conversation_id)
            
            # Create new conversation
            self._context.create_conversation(
                user_id=self._user_id,
                session_id=self._session_id,
                system_prompt=self.config.system_prompt,
            )
    
    def get_stats(self) -> dict[str, Any]:
        """Get system statistics."""
        stats = {
            "initialized": self._initialized,
            "user_id": self._user_id,
            "session_id": self._session_id,
        }
        
        if self._llm:
            stats["llm"] = self._llm.get_model_info()
        
        if self._context:
            stats["context"] = self._context.get_stats()
        
        if self._rag:
            stats["rag"] = self._rag.get_stats()
        
        if self._security:
            stats["security"] = self._security.get_stats()
        
        if self._audit:
            stats["audit"] = self._audit.get_stats()
        
        return stats
    
    async def shutdown(self) -> None:
        """Shutdown all components."""
        log.info("Shutting down AIController...")
        
        if self._llm:
            await self._llm.shutdown()
        
        if self._audit:
            self._audit.shutdown()
        
        if self._rag:
            self._rag.shutdown()
        
        self._initialized = False
        log.info("AIController shutdown complete")


# Singleton instance
_controller_instance: AIController | None = None


def get_ai_controller(config: ChatConfig | None = None) -> AIController:
    """Get or create the singleton AIController instance."""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = AIController(config)
    return _controller_instance


async def initialize_ai(config: ChatConfig | None = None) -> AIController:
    """Initialize the global AI controller."""
    controller = get_ai_controller(config)
    await controller.initialize()
    return controller
