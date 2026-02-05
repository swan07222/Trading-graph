"""
Event System - For decoupled, event-driven architecture
Score Target: 10/10
"""
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict
import threading
import queue
import time
from abc import ABC, abstractmethod


class EventType(Enum):
    """All event types in the system"""
    # Market Data
    TICK = auto()
    BAR = auto()
    QUOTE = auto()
    
    # Trading
    ORDER_SUBMITTED = auto()
    ORDER_ACCEPTED = auto()
    ORDER_REJECTED = auto()
    ORDER_FILLED = auto()
    ORDER_PARTIALLY_FILLED = auto()
    ORDER_CANCELLED = auto()
    
    # Portfolio
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_UPDATED = auto()
    
    # Risk
    RISK_BREACH = auto()
    CIRCUIT_BREAKER = auto()
    MARGIN_CALL = auto()
    
    # Signals
    SIGNAL_GENERATED = auto()
    PREDICTION_READY = auto()
    
    # System
    SYSTEM_START = auto()
    SYSTEM_STOP = auto()
    ERROR = auto()
    WARNING = auto()


@dataclass
class Event:
    """Base event class"""
    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class TickEvent(Event):
    """Market tick event"""
    symbol: str = ""
    price: float = 0.0
    volume: int = 0
    bid: float = 0.0
    ask: float = 0.0
    
    def __post_init__(self):
        self.type = EventType.TICK
        super().__post_init__()


@dataclass
class BarEvent(Event):
    """OHLCV bar event"""
    symbol: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    
    def __post_init__(self):
        self.type = EventType.BAR
        super().__post_init__()


@dataclass
class OrderEvent(Event):
    """Order lifecycle event"""
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: int = 0
    price: float = 0.0
    filled_qty: int = 0
    filled_price: float = 0.0
    message: str = ""


@dataclass
class SignalEvent(Event):
    """Trading signal event"""
    symbol: str = ""
    signal: str = ""  # BUY, SELL, HOLD
    strength: float = 0.0
    confidence: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    reasons: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.type = EventType.SIGNAL_GENERATED
        super().__post_init__()


@dataclass
class RiskEvent(Event):
    """Risk management event"""
    risk_type: str = ""
    current_value: float = 0.0
    limit_value: float = 0.0
    action_taken: str = ""


class EventHandler(ABC):
    """Abstract event handler"""
    
    @abstractmethod
    def handle(self, event: Event):
        pass


class EventBus:
    """
    Central event bus for publish-subscribe pattern
    Thread-safe with async support
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Event history for debugging
        self._history: List[Event] = []
        self._max_history = 1000
    
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """Subscribe to event type"""
        with self._lock:
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """Unsubscribe from event type"""
        with self._lock:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
    
    def publish(self, event: Event, async_: bool = True):
        """Publish event"""
        # Record in history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        if async_:
            self._queue.put(event)
        else:
            self._dispatch(event)
    
    def _dispatch(self, event: Event):
        """Dispatch event to subscribers"""
        with self._lock:
            handlers = self._subscribers.get(event.type, []).copy()
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Publish error event (sync to avoid recursion)
                error_event = Event(
                    type=EventType.ERROR,
                    data={'error': str(e), 'original_event': event}
                )
                self._dispatch_error(error_event)
    
    def _dispatch_error(self, event: Event):
        """Dispatch error without recursion risk"""
        with self._lock:
            handlers = self._subscribers.get(EventType.ERROR, []).copy()
        
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                pass  # Prevent infinite loop
    
    def start(self):
        """Start async event processing"""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
    
    def stop(self):
        """Stop async event processing"""
        self._running = False
        if self._worker_thread:
            self._queue.put(None)  # Sentinel
            self._worker_thread.join(timeout=5)
    
    def _worker(self):
        """Worker thread for async event processing"""
        while self._running:
            try:
                event = self._queue.get(timeout=0.1)
                if event is None:  # Sentinel
                    break
                self._dispatch(event)
            except queue.Empty:
                continue
            except Exception:
                continue
    
    def get_history(self, event_type: EventType = None, limit: int = 100) -> List[Event]:
        """Get event history"""
        if event_type:
            filtered = [e for e in self._history if e.type == event_type]
        else:
            filtered = self._history.copy()
        return filtered[-limit:]
    
    def clear_history(self):
        """Clear event history"""
        self._history.clear()


# Global event bus instance
EVENT_BUS = EventBus()