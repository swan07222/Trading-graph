# ui/app_ai_ops_enhanced.py
"""Enhanced AI operations module with improved NLU, persistence, and safety.

Fixes implemented:
1. Intent classification with similarity scoring (not just keyword matching)
2. Persistent chat history with long-term memory retrieval
3. Async queue for parallel chat processing
4. Retry logic with exponential backoff and graceful degradation
5. Extensible command registry with plugin support
6. Confirmation dialogs and safety controls for trading actions
7. News/sentiment caching with fallback strategies
8. Optimized LLM training with incremental updates
"""

from __future__ import annotations

import hashlib
import heapq
import html
import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from config.settings import CONFIG, TradingMode
from ui.background_tasks import WorkerThread
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)
_UI_AI_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS

# =============================================================================
# CONFIGURATION
# =============================================================================

CHAT_HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chat_history")
CHAT_HISTORY_FILE = os.path.join(CHAT_HISTORY_DIR, "chat_history.json")
MEMORY_INDEX_FILE = os.path.join(CHAT_HISTORY_DIR, "memory_index.json")
MAX_CHAT_HISTORY = 500  # Increased from 250
MAX_LONG_TERM_MEMORIES = 1000
MEMORY_RETRIEVAL_TOP_K = 5
MAX_CONCURRENT_AI_QUERIES = 3  # Allow parallel processing
AI_QUERY_TIMEOUT = 180  # Reduced from 240 seconds
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2  # Exponential backoff base
SENTIMENT_CACHE_TTL = 300  # 5 minutes cache
LLM_TRAINING_MAX_ARTICLES = 200  # Reduced from 450
LLM_TRAINING_TIMEOUT = 1800
CONFIRMATION_TTL_SECONDS = 120.0

CONFIRMATION_REQUIRED_INTENTS = frozenset({
    "monitor_start",
    "monitor_stop",
    "scan_market",
    "watchlist_remove",
    "train_gm",
    "train_llm",
    "auto_train_gm",
    "auto_train_llm",
    "set_interval",
    "set_forecast",
    "set_lookback",
})

INTENT_PERMISSION_MAP: dict[str, str] = {
    "analyze_stock": "analyze",
    "refresh_sentiment": "analyze",
    "watchlist_add": "trade_paper",
    "watchlist_remove": "trade_paper",
    "monitor_start": "trade_paper",
    "monitor_stop": "trade_paper",
    "scan_market": "trade_paper",
    "set_interval": "trade_paper",
    "set_forecast": "trade_paper",
    "set_lookback": "trade_paper",
    "train_gm": "configure",
    "train_llm": "configure",
    "auto_train_gm": "configure",
    "auto_train_llm": "configure",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CommandSpec:
    """Specification for a registered command."""
    name: str
    patterns: list[str]
    handler: Callable
    description: str
    requires_confirmation: bool = False
    confirmation_message: str = ""
    category: str = "general"
    enabled: bool = True


@dataclass
class IntentMatch:
    """Result of intent classification."""
    intent: str
    confidence: float
    command: str | None
    entities: dict[str, Any]
    raw_score: float


@dataclass
class MemoryEntry:
    """Long-term memory entry for retrieval."""
    id: str
    text: str
    embedding_hash: str
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    importance: float = 1.0
    tags: list[str] = field(default_factory=list)


@dataclass
class ChatMessage:
    """Chat message with metadata."""
    timestamp: str
    sender: str
    role: str
    text: str
    level: str
    intent: str = ""
    entities: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# INTENT CLASSIFIER (Improved NLU)
# =============================================================================

class IntentClassifier:
    """Lightweight intent classifier using pattern similarity and fuzzy matching.
    
    Replaces simple keyword matching with:
    - Levenshtein distance for fuzzy pattern matching
    - Pattern scoring with multiple match criteria
    - Entity extraction with context awareness
    """
    
    def __init__(self) -> None:
        self._intent_patterns: dict[str, list[tuple[str, float]]] = {}
        self._entity_extractors: dict[str, Callable[[str], Any]] = {}
        self._register_default_patterns()
        
    def _register_default_patterns(self) -> None:
        """Register default intent patterns with scores."""
        self._intent_patterns = {
            "monitor_start": [
                ("start monitoring", 1.0),
                ("start monitor", 1.0),
                ("enable monitoring", 0.9),
                ("resume monitoring", 0.9),
                ("watch the market", 0.85),
                ("\u5f00\u59cb\u76d1\u63a7", 1.0),
                ("\u5f00\u542f\u76d1\u63a7", 1.0),
                ("\u6253\u5f00\u76d1\u63a7", 0.95),
                ("\u7ee7\u7eed\u76d1\u63a7", 0.9),
                ("\u76ef\u76d8", 0.9),
            ],
            "monitor_stop": [
                ("stop monitoring", 1.0),
                ("stop monitor", 1.0),
                ("pause monitoring", 0.95),
                ("disable monitoring", 0.9),
                ("\u505c\u6b62\u76d1\u63a7", 1.0),
                ("\u5173\u95ed\u76d1\u63a7", 1.0),
                ("\u6682\u505c\u76d1\u63a7", 0.95),
                ("\u53d6\u6d88\u76d1\u63a7", 0.9),
            ],
            "analyze_stock": [
                ("analyze", 0.85),
                ("analysis", 0.85),
                ("look at", 0.75),
                ("check", 0.7),
                ("review", 0.7),
                ("chart", 0.65),
                ("\u5206\u6790", 0.9),
                ("\u67e5\u770b", 0.85),
                ("\u56fe\u8868", 0.75),
            ],
            "watchlist_add": [
                ("add watchlist", 1.0),
                ("add to watchlist", 1.0),
                ("watch this", 0.85),
                ("follow", 0.8),
                ("\u52a0\u5165\u81ea\u9009", 1.0),
                ("\u6dfb\u52a0\u81ea\u9009", 1.0),
                ("\u5173\u6ce8", 0.9),
            ],
            "watchlist_remove": [
                ("remove watchlist", 1.0),
                ("remove from watchlist", 1.0),
                ("unfollow", 0.9),
                ("stop watching", 0.85),
                ("\u79fb\u9664\u81ea\u9009", 1.0),
                ("\u5220\u9664\u81ea\u9009", 1.0),
                ("\u53d6\u6d88\u5173\u6ce8", 0.9),
            ],
            "train_gm": [
                ("train gm", 1.0),
                ("train model", 0.95),
                ("\u8bad\u7ec3 gm", 1.0),
                ("\u8bad\u7ec3\u6a21\u578b", 0.9),
            ],
            "train_llm": [
                ("train llm", 1.0),
                ("train chat model", 0.95),
                ("fine tune llm", 0.9),
                ("\u8bad\u7ec3 llm", 1.0),
                ("\u8bad\u7ec3\u804a\u5929\u6a21\u578b", 1.0),
                ("\u5fae\u8c03 llm", 0.9),
            ],
            "auto_train_gm": [
                ("auto train gm", 1.0),
                ("auto learn", 0.9),
                ("\u81ea\u52a8\u8bad\u7ec3 gm", 1.0),
                ("\u7ee7\u7eed\u5b66\u4e60", 0.85),
            ],
            "auto_train_llm": [
                ("auto train llm", 1.0),
                ("auto llm", 0.9),
                ("train llm automatically", 0.9),
                ("\u81ea\u52a8\u8bad\u7ec3 llm", 1.0),
                ("\u81ea\u52a8\u8bad\u7ec3\u804a\u5929\u6a21\u578b", 1.0),
            ],
            "scan_market": [
                ("scan market", 1.0),
                ("scan for signal", 0.95),
                ("find opportunity", 0.9),
                ("\u626b\u63cf\u5e02\u573a", 1.0),
                ("\u626b\u5e02\u573a", 0.95),
                ("\u627e\u673a\u4f1a", 0.9),
            ],
            "refresh_sentiment": [
                ("refresh sentiment", 1.0),
                ("refresh news", 0.95),
                ("update sentiment", 0.9),
                ("\u5237\u65b0\u60c5\u7eea", 1.0),
                ("\u5237\u65b0\u65b0\u95fb", 1.0),
                ("\u5237\u65b0\u653f\u7b56", 0.95),
                ("\u66f4\u65b0\u60c5\u7eea", 0.95),
            ],
            "set_interval": [
                ("set interval", 1.0),
                ("timeframe", 0.95),
                ("switch to", 0.9),
                ("interval", 0.85),
                ("\u5468\u671f", 1.0),
                ("\u65f6\u95f4\u6846\u67b6", 0.95),
                ("\u5207\u6362\u5230", 0.95),
            ],
            "set_forecast": [
                ("set forecast", 1.0),
                ("forecast bars", 0.95),
                ("prediction bars", 0.95),
                ("horizon", 0.8),
                ("\u8bbe\u7f6e\u9884\u6d4b", 1.0),
                ("\u9884\u6d4b\u6b65\u6570", 0.95),
                ("\u524d\u77bb", 0.9),
            ],
            "set_lookback": [
                ("set lookback", 1.0),
                ("lookback", 0.95),
                ("history window", 0.9),
                ("history length", 0.9),
                ("\u8bbe\u7f6e\u56de\u770b", 1.0),
                ("\u56de\u770b", 0.95),
                ("\u56de\u6eaf", 0.95),
                ("\u5386\u53f2\u7a97\u53e3", 0.9),
            ],
            "status_query": [
                ("status", 1.0),
                ("current state", 1.0),
                ("what are you monitoring", 0.95),
                ("current settings", 0.9),
                ("\u5f53\u524d\u72b6\u6001", 1.0),
                ("\u73b0\u5728\u72b6\u6001", 0.95),
                ("\u53c2\u6570\u72b6\u6001", 0.9),
            ],
            "greeting": [
                ("hi", 1.0),
                ("hello", 1.0),
                ("hey", 1.0),
                ("\u4f60\u597d", 1.0),
                ("\u60a8\u597d", 1.0),
            ],
            "help": [
                ("help", 1.0),
                ("commands", 0.95),
                ("\u5e2e\u52a9", 1.0),
                ("\u547d\u4ee4", 0.9),
            ],
            "capability_query": [
                ("what can you do", 1.0),
                ("capability", 0.95),
                ("how can you help", 0.95),
                ("\u4f60\u80fd\u505a\u4ec0\u4e48", 1.0),
                ("\u53ef\u4ee5\u505a\u4ec0\u4e48", 1.0),
                ("\u600e\u4e48\u63a7\u5236", 0.9),
                ("\u5982\u4f55\u63a7\u5236", 0.9),
            ],
        }
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _similarity_score(self, text: str, pattern: str) -> float:
        """Calculate similarity score between text and pattern."""
        text_lower = text.lower()
        pattern_lower = pattern.lower()
        
        # Exact match
        if pattern_lower in text_lower:
            return 1.0
        
        # Fuzzy match with Levenshtein
        max_len = max(len(text_lower), len(pattern_lower))
        if max_len == 0:
            return 0.0
        
        distance = self._levenshtein_distance(text_lower, pattern_lower)
        similarity = 1.0 - (distance / max_len)
        
        # Boost score if pattern words are present
        pattern_words = pattern_lower.split()
        words_matched = sum(1 for w in pattern_words if w in text_lower)
        word_bonus = words_matched / len(pattern_words) * 0.2 if pattern_words else 0
        
        return min(1.0, similarity + word_bonus)
    
    def classify(self, text: str) -> IntentMatch:
        """Classify intent from text with confidence scoring."""
        text_lower = text.lower()
        best_intent = "unknown"
        best_confidence = 0.0
        best_command = None
        best_score = 0.0
        
        for intent, patterns in self._intent_patterns.items():
            for pattern, base_score in patterns:
                score = self._similarity_score(text_lower, pattern) * base_score
                
                if score > best_confidence:
                    best_confidence = score
                    best_intent = intent
                    best_score = score * base_score
        
        # Extract entities based on intent
        entities = self._extract_entities(text, best_intent)
        
        # Determine if confidence is high enough for direct execution
        if best_confidence >= 0.7:
            best_command = best_intent
        
        return IntentMatch(
            intent=best_intent,
            confidence=best_confidence,
            command=best_command,
            entities=entities,
            raw_score=best_score
        )
    
    def _extract_entities(self, text: str, intent: str) -> dict[str, Any]:
        """Extract entities from text based on intent."""
        _ = intent
        entities: dict[str, Any] = {}

        # Stock code extraction (6-digit codes)
        stock_match = re.search(r"\b(\d{6})\b", text)
        if stock_match:
            entities["stock_code"] = stock_match.group(1)

        # Interval extraction
        interval_match = re.search(
            r"\b(1m|5m|15m|30m|60m|1d|\d+\s*(m|min|mins|minute|minutes|h|hour|hours|d|day|days))\b",
            text.lower(),
        )
        if not interval_match:
            interval_match = re.search(
                r"(\d+)\s*(?:\u5206\u949f|\u5206|\u5c0f\u65f6|\u5929|\u65e5)",
                text,
            )
        if interval_match:
            entities["interval"] = self._normalize_interval(interval_match.group(1))

        # Number extraction for forecast/lookback
        number_match = re.search(r"(\d+)", text)
        if number_match:
            entities["number"] = int(number_match.group(1))

        return entities

    def _normalize_interval(self, interval: str) -> str:
        """Normalize interval string to standard format."""
        interval = interval.lower().strip()

        # Handle Chinese and word variants
        if any(tok in interval for tok in ("\u5206\u949f", "\u5206", "min", "minute")):
            num = re.search(r"(\d+)", interval)
            return f"{num.group(1)}m" if num else "1m"
        if any(tok in interval for tok in ("\u5c0f\u65f6", "hour")) or interval == "h":
            num = re.search(r"(\d+)", interval)
            hours = int(num.group(1)) if num else 1
            return f"{max(1, hours) * 60}m"
        if interval in ("d", "day", "days", "daily", "\u65e5", "\u5929"):
            return "1d"

        # Handle numeric minute fallback
        num = re.search(r"(\d+)", interval)
        if num:
            return f"{num.group(1)}m"

        return interval

# =============================================================================
# LONG-TERM MEMORY SYSTEM
# =============================================================================

class LongTermMemory:
    """Persistent memory system with retrieval capabilities."""
    
    def __init__(self) -> None:
        self._memories: dict[str, MemoryEntry] = {}
        self._lock = threading.Lock()
        self._access_queue: list[tuple[float, str]] = []  # Heap for LRU
        self._ensure_storage()
        self._load_memories()
    
    def _ensure_storage(self) -> None:
        """Ensure storage directory exists."""
        try:
            os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
        except _UI_AI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug(f"Failed to create chat history dir: {exc}")
    
    def _load_memories(self) -> None:
        """Load memories from disk."""
        try:
            if os.path.exists(MEMORY_INDEX_FILE):
                with open(MEMORY_INDEX_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for entry_data in data.get('memories', []):
                        entry = MemoryEntry(**entry_data)
                        self._memories[entry.id] = entry
                        heapq.heappush(
                            self._access_queue,
                            (-entry.last_accessed, entry.id)
                        )
        except _UI_AI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug(f"Failed to load memories: {exc}")
    
    def _save_memories(self) -> None:
        """Save memory index to disk."""
        try:
            data = {
                'memories': [
                    {
                        'id': m.id,
                        'text': m.text,
                        'embedding_hash': m.embedding_hash,
                        'timestamp': m.timestamp,
                        'access_count': m.access_count,
                        'last_accessed': m.last_accessed,
                        'importance': m.importance,
                        'tags': m.tags,
                    }
                    for m in self._memories.values()
                ]
            }
            with open(MEMORY_INDEX_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except _UI_AI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug(f"Failed to save memories: {exc}")
    
    def _hash_text(self, text: str) -> str:
        """Create hash for text deduplication."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def add_memory(
        self,
        text: str,
        tags: list[str] | None = None,
        importance: float = 1.0
    ) -> str:
        """Add a memory entry."""
        with self._lock:
            text_hash = self._hash_text(text)
            
            # Check for duplicates
            for entry in self._memories.values():
                if entry.embedding_hash == text_hash:
                    return entry.id
            
            # Create new memory
            memory_id = hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:16]
            now = time.time()
            
            entry = MemoryEntry(
                id=memory_id,
                text=text,
                embedding_hash=text_hash,
                timestamp=now,
                access_count=0,
                last_accessed=now,
                importance=importance,
                tags=tags or []
            )
            
            self._memories[memory_id] = entry
            heapq.heappush(self._access_queue, (-now, memory_id))
            
            # Prune if needed
            self._prune_memories()
            
            # Save periodically (every 10 additions)
            if len(self._memories) % 10 == 0:
                self._save_memories()
            
            return memory_id
    
    def retrieve(self, query: str, top_k: int = MEMORY_RETRIEVAL_TOP_K) -> list[MemoryEntry]:
        """Retrieve relevant memories for a query."""
        with self._lock:
            # Simple TF-IDF-like scoring based on word overlap
            query_words = set(query.lower().split())
            
            scored_memories = []
            for entry in self._memories.values():
                text_words = set(entry.text.lower().split())
                overlap = len(query_words & text_words)
                
                if overlap > 0:
                    # Score based on overlap, recency, and importance
                    recency_factor = 1.0 / (1.0 + (time.time() - entry.last_accessed) / 86400)
                    score = overlap * recency_factor * entry.importance
                    scored_memories.append((score, entry))
            
            # Sort by score and return top_k
            scored_memories.sort(key=lambda x: -x[0])
            
            # Update access counts
            for _, entry in scored_memories[:top_k]:
                entry.access_count += 1
                entry.last_accessed = time.time()
            
            self._save_memories()
            
            return [entry for _, entry in scored_memories[:top_k]]
    
    def _prune_memories(self) -> None:
        """Prune old/less important memories."""
        if len(self._memories) <= MAX_LONG_TERM_MEMORIES:
            return
        
        # Remove least important/least accessed
        to_remove = []
        for entry in self._memories.values():
            # Score for removal (lower = more likely to remove)
            removal_score = (
                entry.importance * 0.4 +
                (entry.access_count / max(1, time.time() - entry.timestamp)) * 0.6
            )
            to_remove.append((removal_score, entry.id))
        
        to_remove.sort(key=lambda x: x[0])
        
        for _, memory_id in to_remove[:len(self._memories) - MAX_LONG_TERM_MEMORIES]:
            del self._memories[memory_id]


# =============================================================================
# CHAT HISTORY PERSISTENCE
# =============================================================================

class PersistentChatHistory:
    """Persistent chat history with disk storage."""
    
    def __init__(self) -> None:
        self._messages: deque[ChatMessage] = deque(maxlen=MAX_CHAT_HISTORY)
        self._lock = threading.Lock()
        self._ensure_storage()
        self._load_history()
    
    def _ensure_storage(self) -> None:
        """Ensure storage directory exists."""
        try:
            os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
        except _UI_AI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug(f"Failed to create chat history dir: {exc}")
    
    def _load_history(self) -> None:
        """Load history from disk."""
        try:
            if os.path.exists(CHAT_HISTORY_FILE):
                with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for msg_data in data.get('messages', []):
                        msg = ChatMessage(**msg_data)
                        self._messages.append(msg)
        except _UI_AI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug(f"Failed to load chat history: {exc}")
    
    def _save_history(self) -> None:
        """Save history to disk."""
        try:
            data = {
                'messages': [
                    {
                        'timestamp': m.timestamp,
                        'sender': m.sender,
                        'role': m.role,
                        'text': m.text,
                        'level': m.level,
                        'intent': m.intent,
                        'entities': m.entities,
                    }
                    for m in self._messages
                ],
                'saved_at': datetime.now().isoformat()
            }
            with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except _UI_AI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug(f"Failed to save chat history: {exc}")
    
    def add_message(self, message: ChatMessage) -> None:
        """Add a message to history."""
        with self._lock:
            self._messages.append(message)
            # Save every 5 messages
            if len(self._messages) % 5 == 0:
                self._save_history()
    
    def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent messages."""
        with self._lock:
            messages = list(self._messages)[-limit:]
            return [
                {
                    'ts': m.timestamp,
                    'sender': m.sender,
                    'role': m.role,
                    'text': m.text,
                    'level': m.level,
                }
                for m in messages
            ]
    
    def get_context_for_query(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get contextual messages relevant to query."""
        # Simple relevance scoring
        query_words = set(query.lower().split())
        
        scored = []
        for msg in self._messages:
            msg_words = set(msg.text.lower().split())
            overlap = len(query_words & msg_words)
            if overlap > 0:
                scored.append((overlap, msg))
        
        scored.sort(key=lambda x: -x[0])
        
        return [
            {
                'ts': m.timestamp,
                'sender': m.sender,
                'role': m.role,
                'text': m.text,
                'level': m.level,
            }
            for _, m in scored[:limit]
        ]
    
    def clear(self) -> None:
        """Clear all history."""
        with self._lock:
            self._messages.clear()
            self._save_history()


# =============================================================================
# ASYNC QUERY PROCESSOR
# =============================================================================

class AsyncQueryProcessor:
    """Process AI queries asynchronously with queue management."""
    
    def __init__(self) -> None:
        self._queue: deque[tuple[str, Callable, Callable, Callable]] = deque()
        self._active_workers: dict[str, WorkerThread] = {}
        self._lock = threading.Lock()
        self._max_concurrent = MAX_CONCURRENT_AI_QUERIES
    
    def submit(
        self,
        query_id: str,
        work_fn: Callable[[], Any],
        done_fn: Callable[[Any], None],
        error_fn: Callable[[str], None]
    ) -> bool:
        """Submit a query for processing."""
        with self._lock:
            # Check if we can process immediately
            if len(self._active_workers) < self._max_concurrent:
                return self._start_worker(query_id, work_fn, done_fn, error_fn)
            
            # Queue for later
            self._queue.append((query_id, work_fn, done_fn, error_fn))
            return True
    
    def _start_worker(
        self,
        query_id: str,
        work_fn: Callable[[], Any],
        done_fn: Callable[[Any], None],
        error_fn: Callable[[str], None]
    ) -> bool:
        """Start a worker thread for the query."""
        try:
            worker = WorkerThread(work_fn, timeout_seconds=AI_QUERY_TIMEOUT)
            
            def on_done(result: Any) -> None:
                with self._lock:
                    self._active_workers.pop(query_id, None)
                    self._process_queue_locked()
                done_fn(result)
            
            def on_error(err: str) -> None:
                with self._lock:
                    self._active_workers.pop(query_id, None)
                    self._process_queue_locked()
                error_fn(err)
            
            worker.result.connect(on_done)
            worker.error.connect(on_error)
            
            self._active_workers[query_id] = worker
            worker.start()
            
            return True
        except _UI_AI_RECOVERABLE_EXCEPTIONS as exc:
            log.error(f"Failed to start worker: {exc}")
            return False
    
    def _process_queue(self) -> None:
        """Process queued queries."""
        with self._lock:
            self._process_queue_locked()

    def _process_queue_locked(self) -> None:
        """Process queued queries.

        Caller must hold ``self._lock``.
        """
        while self._queue and len(self._active_workers) < self._max_concurrent:
            query_id, work_fn, done_fn, error_fn = self._queue.popleft()
            self._start_worker(query_id, work_fn, done_fn, error_fn)
    
    def cancel(self, query_id: str) -> None:
        """Cancel a queued or running query."""
        with self._lock:
            # Remove from queue
            self._queue = deque(
                (qid, w, d, e) for qid, w, d, e in self._queue if qid != query_id
            )
            
            # Cancel running worker
            worker = self._active_workers.pop(query_id, None)
            if worker:
                try:
                    worker.terminate()
                except _UI_AI_RECOVERABLE_EXCEPTIONS:
                    pass


# =============================================================================
# COMMAND REGISTRY (Extensible)
# =============================================================================

class CommandRegistry:
    """Extensible command registry with plugin support."""
    
    def __init__(self) -> None:
        self._commands: dict[str, CommandSpec] = {}
        self._lock = threading.Lock()
        self._register_default_commands()
    
    def _register_default_commands(self) -> None:
        """Register default commands."""
        # Commands are registered dynamically in the enhanced ops
        pass
    
    def register(self, spec: CommandSpec) -> None:
        """Register a new command."""
        with self._lock:
            self._commands[spec.name] = spec
    
    def unregister(self, name: str) -> None:
        """Unregister a command."""
        with self._lock:
            self._commands.pop(name, None)
    
    def get_command(self, name: str) -> CommandSpec | None:
        """Get a command by name."""
        with self._lock:
            return self._commands.get(name)
    
    def list_commands(self, category: str | None = None) -> list[CommandSpec]:
        """List all commands, optionally filtered by category."""
        with self._lock:
            commands = list(self._commands.values())
            if category:
                commands = [c for c in commands if c.category == category]
            return commands
    
    def find_matching(self, text: str, classifier: IntentClassifier) -> list[CommandSpec]:
        """Find commands matching text intent."""
        intent_match = classifier.classify(text)
        
        if intent_match.confidence < 0.5:
            return []
        
        with self._lock:
            matching = []
            for cmd in self._commands.values():
                if not cmd.enabled:
                    continue
                
                # Check if command patterns match
                for pattern, _ in classifier._intent_patterns.get(cmd.name, []):
                    if pattern.lower() in text.lower():
                        matching.append(cmd)
                        break
            
            return matching


# =============================================================================
# SENTIMENT CACHE WITH FALLBACK
# =============================================================================

class SentimentCache:
    """Cache for sentiment/news data with fallback strategies."""
    
    def __init__(self) -> None:
        self._cache: dict[str, tuple[dict[str, Any], float]] = {}
        self._lock = threading.Lock()
        self._fallback_data: dict[str, Any] = {
            'overall': 0.0,
            'policy': 0.0,
            'market': 0.0,
            'confidence': 0.0,
        }
    
    def get(self, key: str) -> dict[str, Any] | None:
        """Get cached sentiment data."""
        with self._lock:
            if key not in self._cache:
                return None
            
            data, timestamp = self._cache[key]
            if time.time() - timestamp > SENTIMENT_CACHE_TTL:
                del self._cache[key]
                return None
            
            return data
    
    def set(self, key: str, data: dict[str, Any]) -> None:
        """Cache sentiment data."""
        with self._lock:
            self._cache[key] = (data, time.time())
    
    def get_with_fallback(self, key: str) -> dict[str, Any]:
        """Get cached data or fallback."""
        data = self.get(key)
        return data if data else self._fallback_data.copy()
    
    def clear_expired(self) -> None:
        """Clear expired cache entries."""
        with self._lock:
            now = time.time()
            expired = [
                k for k, (_, ts) in self._cache.items()
                if now - ts > SENTIMENT_CACHE_TTL
            ]
            for k in expired:
                del self._cache[k]


# =============================================================================
# ENHANCED AI OPS FUNCTIONS
# =============================================================================

# Global instances
_intent_classifier = IntentClassifier()
_long_term_memory = LongTermMemory()
_chat_history = PersistentChatHistory()
_query_processor = AsyncQueryProcessor()
_command_registry = CommandRegistry()
_sentiment_cache = SentimentCache()


def _confirmation_decision(text: str) -> str | None:
    t = str(text or "").strip().lower()
    if not t:
        return None
    yes_tokens = {
        "yes",
        "y",
        "confirm",
        "ok",
        "sure",
        "\u7ee7\u7eed",
        "\u786e\u8ba4",
        "\u662f",
        "\u597d\u7684",
    }
    no_tokens = {
        "no",
        "n",
        "cancel",
        "abort",
        "stop",
        "\u53d6\u6d88",
        "\u5426",
        "\u4e0d\u8981",
    }
    if t in yes_tokens:
        return "confirm"
    if t in no_tokens:
        return "cancel"
    return None


def _pending_confirmation(self: Any) -> dict[str, Any] | None:
    pending = getattr(self, "_chat_pending_confirmation", None)
    if not isinstance(pending, dict):
        return None
    expires_at = float(pending.get("expires_at", 0.0) or 0.0)
    if expires_at > 0.0 and time.time() > expires_at:
        setattr(self, "_chat_pending_confirmation", None)
        return None
    return pending


def _action_label(intent: str, entities: dict[str, Any], action_text: str = "") -> str:
    text = str(action_text or "").strip()
    if text:
        return text
    if intent == "set_interval":
        tok = str(entities.get("interval", "") or "").strip()
        return f"set interval {tok}" if tok else "set interval"
    if intent == "set_forecast":
        n = entities.get("number")
        return f"set forecast {n}" if n is not None else "set forecast"
    if intent == "set_lookback":
        n = entities.get("number")
        return f"set lookback {n}" if n is not None else "set lookback"
    if intent in {"analyze_stock", "watchlist_add", "watchlist_remove"}:
        code = str(entities.get("stock_code", "") or "").strip()
        return f"{intent} {code}".strip()
    return intent


def _authorize_chat_intent(self: Any, intent: str) -> tuple[bool, str]:
    permission = str(INTENT_PERMISSION_MAP.get(intent, "") or "").strip()
    if not permission:
        return True, ""

    try:
        from utils.security import get_access_control, get_audit_log
    except Exception:
        return False, "Authorization subsystem unavailable; action blocked."

    try:
        ac = get_access_control()
        _sync_access_identity(self, ac)
        allowed = bool(ac.check(permission))
        role = str(getattr(ac, "current_role", "unknown") or "unknown")
    except Exception as exc:
        return False, f"Authorization check failed: {exc}"

    try:
        audit = get_audit_log()
        if audit is not None:
            audit.log_access(f"chat_intent:{intent}", permission, allowed)
    except Exception:
        pass

    if allowed:
        return True, ""
    return (
        False,
        (
            f"Access denied for '{intent}'. Required permission='{permission}' "
            f"(current role='{role}')."
        ),
    )


def _sync_access_identity(self: Any, ac: Any) -> None:
    """Best-effort runtime role/user binding for chat authorization."""
    try:
        mode = getattr(CONFIG, "trading_mode", TradingMode.SIMULATION)
        mode_value = str(getattr(mode, "value", mode) or "").strip().lower()
    except Exception:
        mode_value = "simulation"

    desired_role = "live_trader" if mode_value == "live" else "trader"
    try:
        current_role = str(getattr(ac, "current_role", "") or "").strip()
        if current_role and current_role != desired_role:
            ac.set_role(desired_role)
    except Exception:
        pass

    desired_user = (
        str(os.getenv("TRADING_USER", "") or "").strip()
        or str(os.getenv("USERNAME", "") or "").strip()
        or str(os.getenv("USER", "") or "").strip()
        or "desktop_user"
    )
    try:
        current_user = str(getattr(ac, "_current_user", "") or "").strip()
        if desired_user and current_user != desired_user:
            ac.set_user(desired_user)
    except Exception:
        pass


def _queue_confirmation(
    self: Any,
    intent_match: IntentMatch,
    *,
    action_text: str = "",
) -> tuple[bool, str]:
    label = _action_label(intent_match.intent, intent_match.entities, action_text)
    setattr(
        self,
        "_chat_pending_confirmation",
        {
            "intent": str(intent_match.intent),
            "entities": dict(intent_match.entities or {}),
            "confidence": float(intent_match.confidence or 0.0),
            "action_text": str(action_text or ""),
            "created_at": float(time.time()),
            "expires_at": float(time.time() + CONFIRMATION_TTL_SECONDS),
        },
    )
    return (
        True,
        (
            f"Confirm action: {label}.\n"
            "Reply 'yes' or 'confirm' to proceed, 'cancel' to abort."
        ),
    )


def _handle_pending_confirmation(self: Any, text: str) -> tuple[bool, str]:
    pending = _pending_confirmation(self)
    if pending is None:
        return False, ""

    decision = _confirmation_decision(text)
    if decision is None:
        return True, "Pending action requires confirmation. Reply 'yes' or 'cancel'."

    if decision == "cancel":
        setattr(self, "_chat_pending_confirmation", None)
        return True, "Action cancelled."

    intent = str(pending.get("intent", "unknown") or "unknown")
    entities = dict(pending.get("entities") or {})
    confidence = float(pending.get("confidence", 0.0) or 0.0)
    action_text = str(pending.get("action_text", "") or "")
    setattr(self, "_chat_pending_confirmation", None)
    intent_match = IntentMatch(
        intent=intent,
        confidence=confidence,
        command=(intent if intent != "unknown" else None),
        entities=entities,
        raw_score=confidence,
    )
    handled, reply = _execute_ai_chat_command_enhanced(
        self,
        intent_match,
        action_override=action_text,
        confirmed=True,
    )
    if handled:
        return True, reply
    return True, f"Confirmed action '{_action_label(intent, entities, action_text)}' could not be executed."


def _append_ai_chat_message(
    self: Any,
    sender: str,
    message: str,
    *,
    role: str = "assistant",
    level: str = "info",
    intent: str = "",
    entities: dict[str, Any] | None = None,
) -> None:
    """Append message to chat view with persistence."""
    widget = getattr(self, "ai_chat_view", None)
    if widget is None:
        return

    colors = {
        "user": "#7cc7ff",
        "assistant": "#dbe4f3",
        "system": "#9aa6bf",
        "error": "#ff6b6b",
        "warning": "#f9c74f",
        "success": "#72f1b8",
        "info": "#dbe4f3",
    }
    safe_sender = html.escape(str(sender or "AI"))
    safe_text = html.escape(str(message or "")).replace("\n", "<br/>")
    ts = datetime.now().strftime("%H:%M:%S")
    role_color = colors.get(role, colors.get(level, "#dbe4f3"))
    body_color = colors.get(level, "#dbe4f3") if role == "system" else "#dbe4f3"

    show_in_panel = (
        str(role or "").strip().lower() == "assistant"
        or str(sender or "").strip().lower() in {"ai", "assistant"}
        or (
            str(role or "").strip().lower() == "system"
            and str(level or "").strip().lower() in {"warning", "error", "success"}
        )
    )

    if show_in_panel:
        widget.append(
            f'<span style="color:#7b88a5">[{ts}]</span> '
            f'<span style="color:{role_color};font-weight:600">{safe_sender}:</span> '
            f'<span style="color:{body_color}">{safe_text}</span>'
        )

    # Create and persist message
    msg = ChatMessage(
        timestamp=ts,
        sender=str(sender),
        role=str(role),
        text=str(message or ""),
        level=str(level),
        intent=intent,
        entities=entities or {},
    )
    _chat_history.add_message(msg)
    
    # Add to long-term memory for important messages
    if role in ("user", "assistant") and len(message) > 20:
        importance = 1.5 if "trade" in message.lower() or "buy" in message.lower() or "sell" in message.lower() else 1.0
        _long_term_memory.add_memory(message, importance=importance)

    if show_in_panel:
        try:
            sb = widget.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())
        except _UI_AI_RECOVERABLE_EXCEPTIONS:
            pass


def _on_ai_chat_send(self: Any) -> None:
    """Enhanced chat send with async processing and retry logic."""
    inp = getattr(self, "ai_chat_input", None)
    if inp is None:
        return
    text = str(inp.text() or "").strip()
    if not text:
        return
    inp.clear()

    # Resolve pending confirmation before normal intent routing.
    handled_confirm, confirm_reply = _handle_pending_confirmation(self, text)
    if handled_confirm:
        self._append_ai_chat_message("You", text, role="user", intent="confirmation_reply")
        self._append_ai_chat_message("AI", confirm_reply, role="assistant", intent="confirmation_reply")
        return

    # Classify intent first
    intent_match = _intent_classifier.classify(text)
    
    self._append_ai_chat_message(
        "You", text, role="user",
        intent=intent_match.intent,
        entities=intent_match.entities
    )

    # Fast path: high-confidence commands execute immediately
    if intent_match.confidence >= 0.8 and intent_match.command:
        try:
            handled, reply = _execute_ai_chat_command_enhanced(self, intent_match)
        except Exception as exc:
            self._append_ai_chat_message("System", f"Command failed: {exc}", role="system", level="error")
            return
        if handled:
            self._append_ai_chat_message("AI", reply, role="assistant", intent=intent_match.intent)
            return

    # Build context with long-term memory retrieval
    symbol = self._ui_norm(self.stock_input.text())
    interval = self._normalize_interval_token(self.interval_combo.currentText())
    forecast = int(self.forecast_spin.value())
    lookback = int(self.lookback_spin.value())
    monitor_on = bool(self.monitor and self.monitor.isRunning())
    
    state = {
        "symbol": symbol or "",
        "interval": interval,
        "forecast_bars": forecast,
        "lookback_bars": lookback,
        "monitoring": "on" if monitor_on else "off",
    }
    
    # Retrieve relevant memories
    memories = _long_term_memory.retrieve(text, top_k=MEMORY_RETRIEVAL_TOP_K)
    memory_context = "\n".join([f"- {m.text}" for m in memories]) if memories else ""
    
    # Get contextual chat history
    history = _chat_history.get_context_for_query(text, limit=15)

    # Refresh sentiment with cache
    if any(
        tok in str(text).lower()
        for tok in ("news", "policy", "sentiment", "\u65b0\u95fb", "\u653f\u7b56", "\u60c5\u7eea")
    ) and symbol:
        try:
            cached = _sentiment_cache.get(symbol)
            if not cached:
                self._refresh_news_policy_signal(symbol, force=False)
        except _UI_AI_RECOVERABLE_EXCEPTIONS:
            pass

    # Generate unique query ID
    query_id = hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:16]

    self._append_ai_chat_message(
        "System",
        "AI is thinking with local model context...",
        role="system",
        level="info",
    )

    def _work() -> dict[str, Any]:
        return _generate_ai_chat_reply_enhanced(
            self,
            prompt=text,
            symbol=symbol,
            app_state=state,
            history=history,
            memory_context=memory_context,
            intent_match=intent_match,
        )

    def _on_done(payload: Any) -> None:
        setattr(self, "_ai_retry_count", 0)
        if not isinstance(payload, dict):
            self._append_ai_chat_message("System", "AI reply failed (invalid payload).", role="system", level="error")
            return
        
        answer = str(payload.get("answer", "") or "").strip()
        action = str(payload.get("action", "") or "").strip()
        
        if action:
            try:
                handled2, action_msg = _execute_ai_chat_command_enhanced(self, intent_match, action_override=action)
            except Exception as exc:
                answer = f"{answer}\n\n[Action Error] {exc}"
            else:
                if handled2:
                    answer = f"{answer}\n\n[Action] {action_msg}"
                else:
                    answer = f"{answer}\n\n[Action Suggested] {action}"
        
        self._append_ai_chat_message("AI", answer or "No response.", role="assistant", intent=intent_match.intent)

    def _on_error(err: str) -> None:
        # Retry with exponential backoff
        attempts = getattr(self, '_ai_retry_count', 0)
        if attempts < MAX_RETRY_ATTEMPTS:
            delay = RETRY_BACKOFF_BASE ** attempts
            self._append_ai_chat_message(
                "System", 
                f"AI query failed, retrying in {delay}s... ({attempts + 1}/{MAX_RETRY_ATTEMPTS})", 
                role="system", 
                level="warning"
            )
            setattr(self, '_ai_retry_count', attempts + 1)
            
            # Schedule retry
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(int(delay * 1000), lambda: _on_ai_chat_send_retry(self, text))
        else:
            self._append_ai_chat_message(
                "System", 
                f"AI query failed after {MAX_RETRY_ATTEMPTS} attempts: {err}", 
                role="system", 
                level="error"
            )
            setattr(self, '_ai_retry_count', 0)

    # Submit to async processor
    queued = _query_processor.submit(query_id, _work, _on_done, _on_error)
    if not queued:
        self._append_ai_chat_message(
            "System",
            "Failed to queue AI request.",
            role="system",
            level="error",
        )


def _on_ai_chat_send_retry(self: Any, text: str) -> None:
    """Retry chat send."""
    # Reset input and re-send
    inp = getattr(self, "ai_chat_input", None)
    if inp:
        inp.setText(text)
        _on_ai_chat_send(self)


def _execute_ai_chat_command_enhanced(
    self: Any,
    intent_match: IntentMatch,
    action_override: str | None = None,
    confirmed: bool = False,
) -> tuple[bool, str]:
    """Enhanced command execution with safety controls."""
    intent = str(intent_match.intent or "unknown").strip()
    entities = dict(intent_match.entities or {})
    confidence = float(intent_match.confidence or 0.0)
    raw_action_text = str(action_override or "").strip()

    # If an override action is provided by the model, classify and execute that action.
    if raw_action_text:
        parsed = _intent_classifier.classify(raw_action_text)
        if parsed.intent != "unknown":
            intent = str(parsed.intent)
            entities = dict(parsed.entities or {})
            confidence = float(parsed.confidence or 0.0)
        else:
            # Fall back to raw text for weak matching checks.
            intent = raw_action_text.lower()

    low = str(intent or "").strip().lower()

    # Intent-level authorization.
    allowed, denial = _authorize_chat_intent(self, low)
    if not allowed:
        return True, denial

    # Confirmation gate for potentially risky actions.
    if (not confirmed) and low in CONFIRMATION_REQUIRED_INTENTS:
        return _queue_confirmation(
            self,
            IntentMatch(
                intent=low,
                confidence=confidence,
                command=low if low != "unknown" else None,
                entities=entities,
                raw_score=confidence,
            ),
            action_text=raw_action_text,
        )

    # Greeting
    if low == "greeting":
        return True, (
            "Hi. You can chat naturally and also control the app in plain language. "
            + _chat_state_summary_enhanced(self)
        )

    # Help
    if low == "help":
        return True, (
            "Local AI mode with improved understanding. Commands: "
            "analyze <code>, load <code>, start/stop monitoring, "
            "scan market, refresh sentiment, set interval <1m|5m|15m|30m|60m|1d>, "
            "set forecast <bars>, set lookback <bars>, add/remove watchlist <code>, "
            "train gm/llm, auto train gm/llm. "
            "Chinese: \u5206\u6790 <\u4ee3\u7801> / "
            "\u5f00\u59cb\u76d1\u63a7 / \u505c\u6b62\u76d1\u63a7 / "
            "\u5237\u65b0\u60c5\u7eea / \u5468\u671f 5m."
        )

    # Capability query
    if low == "capability_query":
        return True, (
            "I can chat about market/news/policy context and control the app with natural language. "
            "Examples: 'analyze 600519', 'watch this stock', 'switch to 15 minutes', "
            "'set forecast to 45', 'refresh sentiment', 'scan market'."
        )

    # Status query
    if low == "status_query":
        return True, _chat_state_summary_enhanced(self)

    # Monitor control
    if low == "monitor_stop":
        self.monitor_action.setChecked(False)
        self._stop_monitoring()
        return True, "Monitoring stopped."

    if low == "monitor_start":
        self.monitor_action.setChecked(True)
        self._start_monitoring()
        return True, "Monitoring started."

    # Market scan
    if low == "scan_market":
        self._scan_stocks()
        return True, "Market scan started."

    # Sentiment refresh
    if low == "refresh_sentiment":
        self._refresh_sentiment()
        symbol = self._ui_norm(self.stock_input.text())
        if symbol:
            self._refresh_news_policy_signal(symbol, force=True)
        return True, "Sentiment refresh started."

    # Interval change
    if low == "set_interval":
        token = entities.get("interval", "")
        if not token:
            return True, "Please specify an interval (1m, 5m, 15m, 30m, 60m, 1d)."
        allowed_tokens = {"1m", "5m", "15m", "30m", "60m", "1d"}
        if token not in allowed_tokens:
            return True, f"Unsupported interval '{token}'."
        self.interval_combo.setCurrentText(token)
        return True, f"Interval set to {token}."

    # Forecast setting
    if low == "set_forecast":
        bars = entities.get("number")
        if bars is None:
            return True, "Missing forecast bars value."
        bars = max(int(self.forecast_spin.minimum()), min(int(self.forecast_spin.maximum()), int(bars)))
        self.forecast_spin.setValue(bars)
        return True, f"Forecast set to {bars} bars."

    # Lookback setting
    if low == "set_lookback":
        bars = entities.get("number")
        if bars is None:
            return True, "Missing lookback bars value."
        bars = max(int(self.lookback_spin.minimum()), min(int(self.lookback_spin.maximum()), int(bars)))
        self.lookback_spin.setValue(bars)
        return True, f"Lookback set to {bars} bars."

    # Stock analysis
    if low == "analyze_stock":
        code = entities.get("stock_code") or self._ui_norm(self.stock_input.text())
        if not code:
            return True, "Please specify a stock code."
        self.stock_input.setText(code)
        self._analyze_stock()
        self._refresh_news_policy_signal(code, force=False)
        return True, f"Analyzing {code}."

    # Watchlist add
    if low == "watchlist_add":
        code = entities.get("stock_code") or self._ui_norm(self.stock_input.text())
        if not code:
            return True, "Please specify a stock code."
        self.stock_input.setText(code)
        self._add_to_watchlist()
        return True, f"Added {code} to watchlist."

    # Watchlist remove
    if low == "watchlist_remove":
        code = entities.get("stock_code")
        if not code:
            return True, "Please specify a stock code."
        row = self._watchlist_row_by_code.get(code)
        if row is not None:
            self.watchlist.selectRow(int(row))
            self._remove_from_watchlist()
            return True, f"Removed {code} from watchlist."
        return True, f"{code} is not in watchlist."

    # Training commands
    if low == "auto_train_gm":
        if hasattr(self, "_show_auto_learn"):
            self._show_auto_learn(auto_start=True)
        return True, "Auto Train GM panel opened and training started."

    if low == "auto_train_llm":
        self._auto_train_llm()
        return True, "Auto Train LLM panel opened."

    if low == "train_llm":
        self._start_llm_training()
        return True, "LLM training started."

    if low == "train_gm":
        self._start_training()
        return True, "Train GM dialog opened."

    return False, ""


def _chat_state_summary_enhanced(self: Any) -> str:
    """Enhanced state summary with memory info."""
    symbol = self._ui_norm(self.stock_input.text()) or "--"
    interval = self._normalize_interval_token(self.interval_combo.currentText())
    forecast = int(self.forecast_spin.value())
    lookback = int(self.lookback_spin.value())
    monitor_on = bool(self.monitor and self.monitor.isRunning())
    memory_count = len(_long_term_memory._memories)
    
    return (
        f"Current state: symbol={symbol}, interval={interval}, "
        f"forecast={forecast} bars, lookback={lookback} bars, "
        f"monitoring={'on' if monitor_on else 'off'}, "
        f"memories={memory_count}."
    )


def _generate_ai_chat_reply_enhanced(
    self: Any,
    *,
    prompt: str,
    symbol: str,
    app_state: dict[str, Any],
    history: list[dict[str, Any]],
    memory_context: str = "",
    intent_match: IntentMatch | None = None,
) -> dict[str, Any]:
    """Enhanced reply generation with memory context and local transformer chat."""
    try:
        from data.llm_sentiment import get_llm_analyzer

        analyzer = get_llm_analyzer()

        # Build enhanced state
        enhanced_state = {
            **app_state,
            'memory_context': memory_context,
            'intent': intent_match.intent if intent_match else '',
            'intent_confidence': intent_match.confidence if intent_match else 0.0,
        }

        # Use self-trained LLM for generation
        payload = analyzer.generate_response(
            prompt=str(prompt or ""),
            symbol=str(symbol or "") or None,
            app_state=enhanced_state,
            history=history,
        )

        return {
            "answer": str(payload.get("answer", "") or "").strip(),
            "action": str(payload.get("action", "") or "").strip(),
            "local_model_ready": bool(payload.get("local_model_ready", False)),
        }
    except Exception as exc:
        log.warning("Self-trained LLM chat failed: %s", exc)

        # Fallback with intent-based response
        if intent_match:
            fallback_response = (
                f"Chat not available: {exc}. "
                f"Detected intent: {intent_match.intent} (confidence: {intent_match.confidence:.2f}). "
                "Train the LLM first using 'Auto Train LLM' command."
            )
        else:
            fallback_response = (
                f"Chat not available: {exc}. "
                "Train the LLM first using 'Auto Train LLM' command."
            )
        
        return {
            "answer": fallback_response,
            "action": "",
            "local_model_ready": False,
        }


def _start_llm_training_enhanced(self: Any) -> None:
    """Enhanced LLM training with resource limits and incremental updates."""
    existing = self.workers.get("llm_train")
    if existing and existing.isRunning():
        self._append_ai_chat_message("System", "LLM training is already running.", role="system", level="warning")
        if hasattr(self, "log"):
            self.log("LLM training is already running.", "warning")
        return

    self._append_ai_chat_message(
        "System",
        (
            "Starting optimized LLM training "
            f"(max {LLM_TRAINING_MAX_ARTICLES} articles, includes self-chat model)..."
        ),
        role="system",
        level="info",
    )
    if hasattr(self, "log"):
        self.log("Starting optimized LLM training with self-chat model...", "info")

    def _work() -> dict[str, Any]:
        import os as _os
        from data.llm_sentiment import get_llm_analyzer
        from data.news_collector import get_collector

        analyzer = get_llm_analyzer()
        collector = get_collector()
        # Collect news first so chat model trains on freshest corpus
        articles = collector.collect_news(limit=LLM_TRAINING_MAX_ARTICLES, hours_back=72)
        # Use absolute path for chat history to avoid cwd-relative failures
        _base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        _chat_path = _os.path.join(_base, "data", "chat_history", "chat_history.json")
        chat_report = analyzer.train_chat_model(
            chat_history_path=_chat_path,
            max_steps=3000,
            epochs=3,
            training_profile="professional",
        )

        if articles:
            # Incremental training with smaller batch
            report = dict(analyzer.train(articles, max_samples=500) or {})
            report.setdefault("collected_articles", int(len(articles)))
            report.setdefault("new_articles", int(len(articles)))
            report.setdefault("collection_mode", "direct_news")
            report["chat_model_status"] = str(chat_report.get("status", "unknown") or "unknown")
            report["chat_model_steps"] = int(chat_report.get("steps", 0) or 0)
            report["chat_model_dir"] = str(chat_report.get("model_dir", "") or "")
            chat_note = str(chat_report.get("message", "") or "").strip()
            if chat_note:
                report["notes"] = f"{str(report.get('notes', '')).strip()} | chat={chat_note}".strip(" |")
            return report

        report = analyzer.auto_train_from_internet(
            hours_back=72,
            limit_per_query=80,
            max_samples=500,
            force_china_direct=False,
            only_new=False,
            min_new_articles=1,
            auto_related_search=True,
            allow_gm_bootstrap=False,
        )
        out = dict(report or {})
        out.setdefault("collection_mode", "auto_internet_fallback")
        out["chat_model_status"] = str(chat_report.get("status", "unknown") or "unknown")
        out["chat_model_steps"] = int(chat_report.get("steps", 0) or 0)
        out["chat_model_dir"] = str(chat_report.get("model_dir", "") or "")
        chat_note = str(chat_report.get("message", "") or "").strip()
        if chat_note:
            out["notes"] = f"{str(out.get('notes', '')).strip()} | chat={chat_note}".strip(" |")
        return out

    worker = WorkerThread(_work, timeout_seconds=LLM_TRAINING_TIMEOUT)
    # Register in workers dict BEFORE _track_worker to prevent race where
    # a concurrent check finds the key missing while thread is starting
    self.workers["llm_train"] = worker
    self._track_worker(worker)

    def _on_done(payload: Any) -> None:
        self.workers.pop("llm_train", None)
        if not isinstance(payload, dict):
            self._append_ai_chat_message("System", "LLM training failed (invalid payload).", role="system", level="error")
            if hasattr(self, "_refresh_model_training_statuses"):
                self._refresh_model_training_statuses()
            return
        
        status_text = str(payload.get("status", "unknown") or "unknown").strip()
        status = status_text.lower()
        
        if status in {"trained", "complete", "ok"}:
            level = "success"
        elif status in {"error", "failed"}:
            level = "error"
        else:
            level = "warning"
        
        msg = (
            f"LLM training complete: status={status_text}, "
            f"samples={payload.get('trained_samples', 0)}, "
            f"zh={payload.get('zh_samples', 0)}, en={payload.get('en_samples', 0)}, "
            f"arch={payload.get('training_architecture', 'hybrid_neural_network')}, "
            f"collected={payload.get('collected_articles', payload.get('new_articles', 0))}, "
            f"mode={payload.get('collection_mode', 'unknown')}, "
            f"chat_model={payload.get('chat_model_status', 'unknown')}."
        )
        
        notes = str(payload.get("notes", "") or "").strip()
        if notes:
            msg = f"{msg} notes={notes[:200]}"
        
        self._append_ai_chat_message("System", msg, role="system", level=level)
        if hasattr(self, "log"):
            self.log(msg, level)
        if hasattr(self, "_refresh_model_training_statuses"):
            self._refresh_model_training_statuses()

    def _on_error(err: str) -> None:
        self.workers.pop("llm_train", None)
        self._append_ai_chat_message("System", f"LLM training failed: {err}", role="system", level="error")
        if hasattr(self, "log"):
            self.log(f"LLM training failed: {err}", "error")
        if hasattr(self, "_refresh_model_training_statuses"):
            self._refresh_model_training_statuses()

    worker.result.connect(_on_done)
    worker.error.connect(_on_error)
    worker.start()


# =============================================================================
# COMPATIBILITY ALIASES (for app.py binding)
# =============================================================================
# These aliases ensure the enhanced module can be used as a drop-in replacement

# Core chat functions (already defined above)
# _append_ai_chat_message - defined
# _on_ai_chat_send - defined
# _generate_ai_chat_reply_enhanced - defined

# Alias _generate_ai_chat_reply to enhanced version
_generate_ai_chat_reply = _generate_ai_chat_reply_enhanced

# Alias _execute_ai_chat_command to enhanced version (needs wrapper)
def _execute_ai_chat_command(self: Any, prompt: str) -> tuple[bool, str]:
    """Compatibility wrapper for enhanced command execution."""
    intent_match = _intent_classifier.classify(prompt)
    return _execute_ai_chat_command_enhanced(self, intent_match)

# Alias _start_llm_training to enhanced version
_start_llm_training = _start_llm_training_enhanced

# Stub functions for compatibility with original API
def _handle_ai_chat_prompt(self: Any, prompt: str) -> str:
    """Handle chat prompt with intent classification."""
    intent_match = _intent_classifier.classify(prompt)
    
    # Fast path for high-confidence commands
    if intent_match.confidence >= 0.8:
        handled, reply = _execute_ai_chat_command_enhanced(self, intent_match)
        if handled:
            return reply
    
    return _build_ai_chat_response_enhanced(self, prompt, intent_match)


def _build_ai_chat_response_enhanced(
    self: Any, 
    prompt: str, 
    intent_match: IntentMatch | None = None
) -> str:
    """Build AI chat response with enhanced context."""
    symbol = self._ui_norm(self.stock_input.text())
    monitor_on = bool(self.monitor and self.monitor.isRunning())
    interval = self._normalize_interval_token(self.interval_combo.currentText())
    forecast = int(self.forecast_spin.value())
    lookback = int(self.lookback_spin.value())

    # Get sentiment with cache
    if hasattr(self, "_news_policy_signal_for"):
        sig = self._news_policy_signal_for(symbol if symbol else "__market__")
    else:
        sig = _sentiment_cache.get_with_fallback(symbol if symbol else "__market__")
    
    overall = float(sig.get("overall", 0.0) or 0.0)
    policy = float(sig.get("policy", 0.0) or 0.0)
    confidence = float(sig.get("confidence", 0.0) or 0.0)

    state = {
        "symbol": symbol or "",
        "interval": interval,
        "forecast_bars": forecast,
        "lookback_bars": lookback,
        "monitoring": "on" if monitor_on else "off",
        "news_policy_signal": {
            "overall": overall,
            "policy": policy,
            "confidence": confidence,
        },
    }

    # Retrieve memories for context
    memories = _long_term_memory.retrieve(prompt, top_k=3)
    memory_context = "\n".join([f"- {m.text}" for m in memories]) if memories else ""

    # Get contextual history
    history = _chat_history.get_context_for_query(prompt, limit=10)

    payload = _generate_ai_chat_reply_enhanced(
        self,
        prompt=str(prompt or ""),
        symbol=symbol,
        app_state=state,
        history=history,
        memory_context=memory_context,
        intent_match=intent_match,
    )
    
    answer = str(payload.get("answer", "") or "").strip()
    action = str(payload.get("action", "") or "").strip()
    
    if action:
        handled, action_msg = _execute_ai_chat_command_enhanced(self, intent_match or IntentMatch(
            intent="unknown", confidence=0.0, command=None, entities={}, raw_score=0.0
        ), action_override=action)
        if handled:
            answer = f"{answer}\n\n[Action] {action_msg}"
        else:
            answer = f"{answer}\n\n[Action Suggested] {action}"
    
    return answer or (
        f"State: symbol={symbol or '--'}, interval={interval}, forecast={forecast}, "
        f"lookback={lookback}, monitoring={'on' if monitor_on else 'off'}. "
        f"News-policy signal: overall={overall:+.2f}, policy={policy:+.2f}, confidence={confidence:.0%}. "
        "Use 'help' to see control commands."
    )


# Alias for compatibility
_build_ai_chat_response = _build_ai_chat_response_enhanced

# Auto-training functions
def _auto_train_llm(self: Any) -> None:
    """Open Auto Train LLM control panel (non-modal)."""
    if hasattr(self, "_show_llm_train_dialog"):
        dialog = self._show_llm_train_dialog(auto_start=False)
        if dialog is None and hasattr(self, "log"):
            self.log("Auto Train LLM dialog not available.", "error")
        elif hasattr(self, "log"):
            self.log("Auto Train LLM panel opened.", "info")

def _show_llm_train_dialog(self: Any, auto_start: bool = False) -> Any | None:
    """Show LLM train dialog."""
    try:
        from .llm_train_dialog import LLMTrainDialog
    except ImportError as exc:
        if hasattr(self, "log"):
            self.log(f"Auto Train LLM dialog not available: {exc}", "error")
        return None

    dialog = getattr(self, "_llm_train_dialog", None)
    if dialog is None:
        dialog = LLMTrainDialog(self)
        self._llm_train_dialog = dialog

        def _on_destroyed(*_args: object) -> None:
            self._llm_train_dialog = None

        if hasattr(dialog, "session_finished"):
            dialog.session_finished.connect(self._on_llm_training_session_finished)
        dialog.destroyed.connect(_on_destroyed)

    dialog.setModal(False)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    try:
        from PyQt6.QtCore import Qt

        dialog.setWindowState(
            (dialog.windowState() & ~Qt.WindowState.WindowMinimized)
            | Qt.WindowState.WindowActive
        )
    except Exception:
        pass
    if auto_start and hasattr(dialog, "start_or_resume_auto_train"):
        dialog.start_or_resume_auto_train()
    return dialog

def _on_llm_training_session_finished(self: Any, payload: dict[str, Any]) -> None:
    """Handle LLM training session completion."""
    data = dict(payload or {})
    status = str(data.get("status", "unknown") or "unknown").strip().lower()
    if status in {"ok", "trained", "complete"} and hasattr(self, "log"):
        self.log(
            (
                "Auto Train LLM completed: "
                f"collected={data.get('collected_articles', 0)}, "
                f"trained={data.get('trained_samples', 0)}, "
                f"arch={data.get('training_architecture', 'hybrid_neural_network')}"
            ),
            "success",
        )
    elif status == "stopped" and hasattr(self, "log"):
        self.log("Auto Train LLM stopped by user.", "warning")
    elif status in {"error", "failed"} and hasattr(self, "log"):
        self.log(f"Auto Train LLM failed: {data.get('error', 'unknown error')}", "error")

    if hasattr(self, "_refresh_model_training_statuses"):
        self._refresh_model_training_statuses()

def _refresh_model_training_statuses(self: Any) -> None:
    """Refresh both GM and LLM status labels shown in the left AI panel."""
    llm_status_widget = getattr(self, "llm_status", None)
    llm_info_widget = getattr(self, "llm_info", None)
    if llm_status_widget is None or llm_info_widget is None:
        return
    try:
        from data.llm_sentiment import get_llm_analyzer

        analyzer = get_llm_analyzer()
        status_payload = analyzer.get_training_status()
    except Exception as exc:
        llm_status_widget.setText("LLM Model: Error")
        llm_info_widget.setText(str(exc))
        return

    status = str(status_payload.get("status", "not_trained") or "not_trained").strip().lower()
    architecture = str(
        status_payload.get("training_architecture", "hybrid_neural_network")
        or "hybrid_neural_network"
    )
    trained_samples = int(status_payload.get("trained_samples", 0) or 0)
    finished_at = str(
        status_payload.get("finished_at", status_payload.get("saved_at", "")) or ""
    ).strip()
    finished_short = finished_at[:19].replace("T", " ") if finished_at else ""

    if status in {"trained", "complete", "ok"}:
        llm_status_widget.setText("LLM Model: Trained")
        llm_status_widget.setStyleSheet("color: #35b57c; font-weight: 700;")
    elif status in {"partial"}:
        llm_status_widget.setText("LLM Model: Partially Trained")
        llm_status_widget.setStyleSheet("color: #d8a03a; font-weight: 700;")
    elif status in {"stopped"}:
        llm_status_widget.setText("LLM Model: Stopped")
        llm_status_widget.setStyleSheet("color: #d8a03a; font-weight: 700;")
    elif status in {"error", "failed"}:
        llm_status_widget.setText("LLM Model: Error")
        llm_status_widget.setStyleSheet("color: #e5534b; font-weight: 700;")
    else:
        llm_status_widget.setText("LLM Model: Not trained")
        llm_status_widget.setStyleSheet("")

    info_parts = [architecture]
    if trained_samples > 0:
        info_parts.append(f"samples={trained_samples}")
    if finished_short:
        info_parts.append(f"last={finished_short}")
    llm_info_widget.setText(" | ".join(info_parts))


# News/Policy signal functions (use caching)
def _set_news_policy_signal(
    self: Any,
    symbol: str,
    signal_data: dict[str, Any],
) -> None:
    """Set news policy signal with caching."""
    # Store in app instance
    cache_attr = "_news_policy_signal_cache"
    if not hasattr(self, cache_attr):
        setattr(self, cache_attr, {})
    cache = getattr(self, cache_attr)
    cache[symbol] = signal_data
    
    # Also cache in global sentiment cache
    _sentiment_cache.set(symbol, signal_data)


def _news_policy_signal_for(self: Any, symbol: str) -> dict[str, Any]:
    """Get news policy signal for symbol with cache."""
    cache_attr = "_news_policy_signal_cache"
    if hasattr(self, cache_attr):
        cache = getattr(self, cache_attr)
        if symbol in cache:
            return cache[symbol]
    
    # Try global cache
    cached = _sentiment_cache.get(symbol)
    if cached:
        return cached
    
    return {
        "overall": 0.0,
        "policy": 0.0,
        "market": 0.0,
        "confidence": 0.0,
    }


def _refresh_news_policy_signal(
    self: Any,
    symbol: str,
    force: bool = False,
) -> None:
    """Refresh news policy signal with caching."""
    # Check cache first
    if not force:
        cached = _sentiment_cache.get(symbol)
        if cached:
            return
    
    # Refresh from source (delegate to original if available)
    try:
        from ui import app_ai_ops as _original
        if hasattr(_original, '_refresh_news_policy_signal'):
            _original._refresh_news_policy_signal(self, symbol, force)
            return
    except _UI_AI_RECOVERABLE_EXCEPTIONS:
        pass
    
    # Basic refresh - trigger sentiment refresh
    if hasattr(self, "_refresh_sentiment"):
        self._refresh_sentiment()


def _apply_enhanced_ops() -> None:
    """Apply enhanced operations as replacements."""
    # This is called from app.py to swap in enhanced versions
    # All functions are already aliased above
    pass


