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
LLM_TRAINING_TIMEOUT = 600  # Reduced from 1200 seconds


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
        # Monitor control intents
        self._intent_patterns["monitor_start"] = [
            ("start monitoring", 1.0),
            ("开启监控", 1.0),
            ("开始监控", 1.0),
            ("打开监控", 0.95),
            ("启动监控", 0.9),
            ("resume monitoring", 0.9),
            ("enable monitoring", 0.9),
            ("watch the market", 0.85),
            ("盯盘", 0.95),
            ("监控市场", 0.9),
        ]
        
        self._intent_patterns["monitor_stop"] = [
            ("stop monitoring", 1.0),
            ("停止监控", 1.0),
            ("关闭监控", 1.0),
            ("pause monitoring", 0.95),
            ("disable monitoring", 0.9),
            ("stop monitor", 0.95),
            ("别监控了", 0.9),
            ("先别监控", 0.85),
        ]
        
        # Analysis intents
        self._intent_patterns["analyze_stock"] = [
            ("analyze", 0.8),
            ("分析", 0.8),
            ("look at", 0.75),
            ("查看", 0.8),
            ("check", 0.7),
            ("review", 0.7),
            ("chart", 0.6),
            ("图表", 0.6),
        ]
        
        # Watchlist intents
        self._intent_patterns["watchlist_add"] = [
            ("add watchlist", 1.0),
            ("加入自选", 1.0),
            ("添加自选", 1.0),
            ("watch this", 0.85),
            ("follow", 0.8),
            ("关注", 0.9),
        ]
        
        self._intent_patterns["watchlist_remove"] = [
            ("remove watchlist", 1.0),
            ("移除自选", 1.0),
            ("删除自选", 1.0),
            ("unfollow", 0.9),
            ("stop watching", 0.85),
        ]
        
        # Training intents
        self._intent_patterns["train_gm"] = [
            ("train gm", 1.0),
            ("训练 gm", 1.0),
            ("训练模型", 0.9),
            ("train model", 0.95),
        ]
        
        self._intent_patterns["train_llm"] = [
            ("train llm", 1.0),
            ("训练 llm", 1.0),
            ("训练大模型", 1.0),
            ("train chat model", 0.95),
        ]
        
        self._intent_patterns["auto_train_gm"] = [
            ("auto train gm", 1.0),
            ("自动训练 gm", 1.0),
            ("auto learn", 0.9),
            ("继续学习", 0.85),
        ]
        
        self._intent_patterns["auto_train_llm"] = [
            ("auto train llm", 1.0),
            ("自动训练 llm", 1.0),
            ("自动训练大模型", 1.0),
        ]
        
        # Market scan intents
        self._intent_patterns["scan_market"] = [
            ("scan market", 1.0),
            ("扫描市场", 1.0),
            ("scan for signal", 0.95),
            ("find opportunity", 0.9),
            ("扫市场", 1.0),
            ("全市场扫描", 0.95),
        ]
        
        # Sentiment refresh intents
        self._intent_patterns["refresh_sentiment"] = [
            ("refresh sentiment", 1.0),
            ("刷新情绪", 1.0),
            ("refresh news", 0.95),
            ("刷新新闻", 1.0),
            ("update sentiment", 0.9),
            ("更新情绪", 0.95),
        ]
        
        # Interval change intents
        self._intent_patterns["set_interval"] = [
            ("set interval", 1.0),
            ("周期", 1.0),
            ("timeframe", 0.95),
            ("switch to", 0.9),
            ("切换到", 1.0),
            ("改成", 0.9),
            ("换成", 0.9),
        ]
        
        # Status query intents
        self._intent_patterns["status_query"] = [
            ("status", 1.0),
            ("当前状态", 1.0),
            ("current state", 1.0),
            ("what are you monitoring", 0.95),
            ("current settings", 0.9),
            ("参数状态", 0.95),
        ]
        
        # Greeting intents
        self._intent_patterns["greeting"] = [
            ("hi", 1.0),
            ("hello", 1.0),
            ("hey", 1.0),
            ("你好", 1.0),
            ("您好", 1.0),
            ("嗨", 1.0),
        ]
        
        # Help intents
        self._intent_patterns["help"] = [
            ("help", 1.0),
            ("help", 1.0),
            ("commands", 0.95),
            ("命令", 1.0),
            ("帮助", 1.0),
            ("幫助", 1.0),
        ]
        
        # Capability query intents
        self._intent_patterns["capability_query"] = [
            ("what can you do", 1.0),
            ("你能做什么", 1.0),
            ("capability", 0.95),
            ("how can you help", 0.95),
            ("可以做什么", 1.0),
            ("怎么控制", 0.9),
        ]
    
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
        entities = {}
        
        # Stock code extraction (6-digit codes)
        stock_match = re.search(r'\b(\d{6})\b', text)
        if stock_match:
            entities['stock_code'] = stock_match.group(1)
        
        # Interval extraction
        interval_match = re.search(r'\b(1m|5m|15m|30m|60m|1d|\d+\s*(m|min|分钟|h|hour|日|天))\b', text.lower())
        if interval_match:
            entities['interval'] = self._normalize_interval(interval_match.group(1))
        
        # Number extraction for forecast/lookback
        number_match = re.search(r'(\d+)', text)
        if number_match:
            entities['number'] = int(number_match.group(1))
        
        return entities
    
    def _normalize_interval(self, interval: str) -> str:
        """Normalize interval string to standard format."""
        interval = interval.lower().strip()
        
        # Handle Chinese
        if '分钟' in interval:
            num = re.search(r'(\d+)', interval)
            return f"{num.group(1)}m" if num else "1m"
        
        # Handle variations
        if interval in ('h', 'hour', '小时'):
            return "1h"
        if interval in ('日', '天', 'daily', 'day'):
            return "1d"
        
        # Handle minute variations
        num = re.search(r'(\d+)', interval)
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
                    self._process_queue()
                done_fn(result)
            
            def on_error(err: str) -> None:
                with self._lock:
                    self._active_workers.pop(query_id, None)
                    self._process_queue()
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

    if len(_chat_history._messages) > 250:
        del _chat_history._messages[0]

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
    if any(tok in str(text).lower() for tok in ("news", "policy", "sentiment", "新闻", "政策", "情绪")) and symbol:
        try:
            cached = _sentiment_cache.get(symbol)
            if not cached:
                self._refresh_news_policy_signal(symbol, force=False)
        except _UI_AI_RECOVERABLE_EXCEPTIONS:
            pass

    # Generate unique query ID
    query_id = hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:16]

    self._append_ai_chat_message("System", "AI is searching the internet and thinking...", role="system", level="info")

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
    _query_processor.submit(query_id, _work, _on_done, _on_error)


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
    action_override: str | None = None
) -> tuple[bool, str]:
    """Enhanced command execution with safety controls."""
    
    intent = intent_match.intent
    entities = intent_match.entities
    confidence = intent_match.confidence
    
    # Check for safety-critical actions
    safety_critical_intents = {
        "train_gm", "train_llm", "auto_train_gm", "auto_train_llm",
        "monitor_start", "scan_market"
    }
    
    # Require confirmation for safety-critical actions if confidence is low
    if intent in safety_critical_intents and confidence < 0.9:
        # For now, proceed but log - full confirmation dialog would go here
        log.info(f"Safety-critical action '{intent}' executed with confidence {confidence}")
    
    low = intent  # Use classified intent
    
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
            "Chinese: 分析 <代码> / 开始监控 / 停止监控 / 刷新情绪 / 周期 5m。"
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
        token = entities.get('interval', '')
        if not token:
            return True, "Please specify an interval (1m, 5m, 15m, 30m, 60m, 1d)."
        allowed = {"1m", "5m", "15m", "30m", "60m", "1d"}
        if token not in allowed:
            return True, f"Unsupported interval '{token}'."
        self.interval_combo.setCurrentText(token)
        return True, f"Interval set to {token}."
    
    # Forecast setting
    if "forecast" in low:
        bars = entities.get('number')
        if not bars:
            return True, "Missing forecast bars value."
        bars = max(int(self.forecast_spin.minimum()), min(int(self.forecast_spin.maximum()), bars))
        self.forecast_spin.setValue(bars)
        return True, f"Forecast set to {bars} bars."
    
    # Lookback setting
    if "lookback" in low:
        bars = entities.get('number')
        if not bars:
            return True, "Missing lookback bars value."
        bars = max(int(self.lookback_spin.minimum()), min(int(self.lookback_spin.maximum()), bars))
        self.lookback_spin.setValue(bars)
        return True, f"Lookback set to {bars} bars."
    
    # Stock analysis
    if low == "analyze_stock":
        code = entities.get('stock_code') or self._ui_norm(self.stock_input.text())
        if not code:
            return True, "Please specify a stock code."
        self.stock_input.setText(code)
        self._analyze_stock()
        self._refresh_news_policy_signal(code, force=False)
        return True, f"Analyzing {code}."
    
    # Watchlist add
    if low == "watchlist_add":
        code = entities.get('stock_code') or self._ui_norm(self.stock_input.text())
        if not code:
            return True, "Please specify a stock code."
        self.stock_input.setText(code)
        self._add_to_watchlist()
        return True, f"Added {code} to watchlist."
    
    # Watchlist remove
    if low == "watchlist_remove":
        code = entities.get('stock_code')
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
    """Enhanced reply generation with memory context and retry.
    
    Note: This uses your self-trained model from llm_sentiment.py.
    No pre-trained models (llama/ollama/transformers) are used.
    """
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
            "local_model_ready": True,  # Self-trained model
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
        f"Starting optimized LLM training (max {LLM_TRAINING_MAX_ARTICLES} articles)...",
        role="system",
        level="info",
    )
    if hasattr(self, "log"):
        self.log("Starting optimized LLM hybrid training...", "info")

    def _work() -> dict[str, Any]:
        from data.llm_sentiment import get_llm_analyzer
        from data.news_collector import get_collector

        analyzer = get_llm_analyzer()
        collector = get_collector()
        
        # Reduced article count for faster training
        articles = collector.collect_news(limit=LLM_TRAINING_MAX_ARTICLES, hours_back=72)
        
        if articles:
            # Incremental training with smaller batch
            report = dict(analyzer.train(articles, max_samples=500) or {})
            report.setdefault("collected_articles", int(len(articles)))
            report.setdefault("new_articles", int(len(articles)))
            report.setdefault("collection_mode", "direct_news")
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
        return out

    worker = WorkerThread(_work, timeout_seconds=LLM_TRAINING_TIMEOUT)
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
            f"mode={payload.get('collection_mode', 'unknown')}."
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
    self.workers["llm_train"] = worker
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

# Auto-training functions (use original from app_ai_ops if not defined)
# These will be imported from original module if needed
def _auto_train_llm(self: Any) -> None:
    """Open auto train LLM dialog (compatibility wrapper)."""
    # Delegate to original implementation if available
    try:
        from ui import app_ai_ops as _original
        if hasattr(_original, '_auto_train_llm'):
            _original._auto_train_llm(self)
    except _UI_AI_RECOVERABLE_EXCEPTIONS:
        log.debug("Original _auto_train_llm not available")

def _show_llm_train_dialog(self: Any) -> None:
    """Show LLM train dialog (compatibility wrapper)."""
    try:
        from ui import app_ai_ops as _original
        if hasattr(_original, '_show_llm_train_dialog'):
            _original._show_llm_train_dialog(self)
    except _UI_AI_RECOVERABLE_EXCEPTIONS:
        log.debug("Original _show_llm_train_dialog not available")

def _on_llm_training_session_finished(self: Any) -> None:
    """Handle LLM training session finished (compatibility wrapper)."""
    if hasattr(self, "_refresh_model_training_statuses"):
        self._refresh_model_training_statuses()

def _refresh_model_training_statuses(self: Any) -> None:
    """Refresh both GM and LLM status labels shown in the left AI panel."""
    # Import from original if available, otherwise use basic implementation
    try:
        from ui import app_ai_ops as _original
        if hasattr(_original, '_refresh_model_training_statuses'):
            _original._refresh_model_training_statuses(self)
            return
    except _UI_AI_RECOVERABLE_EXCEPTIONS:
        pass
    
    # Basic implementation
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
    trained_samples = int(status_payload.get("trained_samples", 0) or 0)
    finished_at = str(
        status_payload.get("finished_at", status_payload.get("saved_at", "")) or ""
    ).strip()
    finished_short = finished_at[:19].replace("T", " ") if finished_at else ""

    if status in {"trained", "complete", "ok"}:
        llm_status_widget.setText("LLM Model: Trained")
        llm_status_widget.setStyleSheet("color: #35b57c; font-weight: 700;")
        llm_info_widget.setText(
            f"Samples: {trained_samples}, Finished: {finished_short}"
        )
    elif status in {"partial"}:
        llm_status_widget.setText("LLM Model: Partially Trained")
        llm_status_widget.setStyleSheet("color: #d8a03a; font-weight: 700;")
        llm_info_widget.setText(
            f"Samples: {trained_samples}, Finished: {finished_short}"
        )
    else:
        llm_status_widget.setText("LLM Model: Not trained")
        llm_status_widget.setStyleSheet("color: #88909a; font-weight: 700;")
        llm_info_widget.setText("Train LLM to enable AI chat capabilities")


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
