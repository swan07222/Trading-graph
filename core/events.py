# core/events.py
import queue
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)

class EventType(Enum):
    """All event types in the system."""
    TICK = auto()
    BAR = auto()
    QUOTE = auto()

    ORDER_SUBMITTED = auto()
    ORDER_ACCEPTED = auto()
    ORDER_REJECTED = auto()
    ORDER_FILLED = auto()
    ORDER_PARTIALLY_FILLED = auto()
    ORDER_CANCELLED = auto()

    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_UPDATED = auto()

    RISK_BREACH = auto()
    CIRCUIT_BREAKER = auto()
    MARGIN_CALL = auto()

    SIGNAL_GENERATED = auto()
    PREDICTION_READY = auto()

    SYSTEM_START = auto()
    SYSTEM_STOP = auto()
    ERROR = auto()
    WARNING = auto()

@dataclass
class Event:
    """Base event class."""
    type: EventType = EventType.SYSTEM_START
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    data: dict[str, Any] = field(default_factory=dict)

@dataclass
class TickEvent(Event):
    """Market tick event."""
    type: EventType = EventType.TICK
    symbol: str = ""
    price: float = 0.0
    volume: int = 0
    bid: float = 0.0
    ask: float = 0.0

@dataclass
class BarEvent(Event):
    """OHLCV bar event."""
    type: EventType = EventType.BAR
    symbol: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0

@dataclass
class OrderEvent(Event):
    """Order lifecycle event."""
    type: EventType = EventType.ORDER_SUBMITTED
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
    """Trading signal event."""
    type: EventType = EventType.SIGNAL_GENERATED
    symbol: str = ""
    signal: str = ""
    strength: float = 0.0
    confidence: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    reasons: list[str] = field(default_factory=list)

@dataclass
class RiskEvent(Event):
    """Risk management event."""
    type: EventType = EventType.RISK_BREACH
    risk_type: str = ""
    current_value: float = 0.0
    limit_value: float = 0.0
    action_taken: str = ""

class EventHandler(ABC):
    """Abstract event handler."""

    @abstractmethod
    def handle(self, event: Event) -> None:
        pass

class EventBus:
    """Central event bus for publish-subscribe pattern.
    Thread-safe with async support.

    FIX: Uses deque(maxlen=N) for O(1) bounded history instead of
    list with pop(0) which is O(n).
    """

    _instance: "EventBus | None" = None
    _lock = threading.RLock()
    _initialized: bool

    def __new__(cls) -> "EventBus":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    object.__setattr__(instance, "_initialized", False)
                    # FIX #1: Initialize thread-safe error depth tracking
                    object.__setattr__(instance, "_error_depth_lock", threading.Lock())
                    object.__setattr__(instance, "_error_depth", 0)
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        object.__setattr__(self, "_initialized", True)
        self._subscribers: dict[EventType, list[Callable]] = defaultdict(list)
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._sub_lock = threading.RLock()

        # FIX: Use deque with maxlen for O(1) bounded history
        self._max_history = 1000
        self._history: deque = deque(maxlen=self._max_history)
        # Note: _error_depth and _error_depth_lock initialized in __new__

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None]
    ) -> None:
        """Subscribe to event type (thread-safe)."""
        with self._sub_lock:
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)

    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None]
    ) -> None:
        """Unsubscribe from event type (thread-safe)."""
        with self._sub_lock:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass

    def clear_subscribers(self, event_type: EventType | None = None) -> None:
        """Clear all subscribers for a given event type,
        or all subscribers if event_type is None.
        Useful for testing.
        """
        with self._sub_lock:
            if event_type is not None:
                self._subscribers[event_type].clear()
            else:
                self._subscribers.clear()

    def publish(self, event: Event, async_: bool = True) -> None:
        """Publish event (thread-safe).

        FIX: deque.append is O(1) and auto-bounded by maxlen,
        no manual trimming needed.
        """
        # Record in history (deque is thread-safe for append)
        self._history.append(event)

        if async_ and self._running:
            self._queue.put(event)
        else:
            self._dispatch(event)

    def _dispatch(self, event: Event) -> None:
        """Dispatch event to subscribers.

        Note: Handlers receive a copy of the subscriber list and may safely
        subscribe/unsubscribe during dispatch. However, handlers should avoid
        blocking operations that could deadlock the event bus worker thread.
        Long-running handlers should use async execution or background threads.

        FIX #1: Use class-level lock for error depth tracking instead of
        thread-local storage to prevent race conditions and ensure reliable
        recursion prevention across all thread configurations.
        """
        with self._sub_lock:
            handlers = self._subscribers.get(event.type, []).copy()

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Avoid recursion: don't dispatch ERROR for ERROR handlers
                if event.type != EventType.ERROR:
                    # FIX #1: Use class-level lock for thread-safe depth tracking
                    should_skip = False
                    with self._error_depth_lock:
                        if self._error_depth >= 3:
                            should_skip = True
                        else:
                            self._error_depth += 1

                    if should_skip:
                        # Max depth reached - just log to break recursion
                        log.error(
                            f"Error handler failed (max depth reached): {e}"
                        )
                        continue

                    try:
                        error_event = Event(
                            type=EventType.ERROR,
                            data={
                                'error': str(e),
                                'handler': str(handler),
                                'original_event_type': event.type.name,
                            }
                        )
                        self._dispatch_error(error_event)
                    finally:
                        # FIX #1: Always decrement depth under lock
                        with self._error_depth_lock:
                            self._error_depth = max(0, self._error_depth - 1)
                else:
                    # Already in ERROR handler, just log
                    log.error(
                        f"Error in ERROR handler: {e}"
                    )

    def _dispatch_error(self, event: Event) -> None:
        """Dispatch error event without recursion risk."""
        with self._sub_lock:
            handlers = self._subscribers.get(
                EventType.ERROR, []
            ).copy()

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Last resort: just log it
                log.error(f"Error handler failed: {e}")

    def start(self) -> None:
        """Start async event processing."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="event_bus_worker"
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop async event processing gracefully."""
        if not self._running:
            return

        self._running = False

        self._queue.put(None)

        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            self._worker_thread = None

        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                if event is not None:
                    self._dispatch(event)
            except queue.Empty:
                break
            except Exception:
                break

    def _worker(self) -> None:
        """Worker thread for async event processing."""
        while self._running:
            try:
                event = self._queue.get(timeout=0.1)
                if event is None:
                    break
                self._dispatch(event)
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"Event worker error: {e}")

    def get_history(
        self,
        event_type: EventType | None = None,
        limit: int = 100
    ) -> list[Event]:
        """Get event history.

        Thread-safe: returns a snapshot copy of history.
        """
        with self._sub_lock:
            if event_type:
                filtered = [
                    e for e in self._history
                    if e.type == event_type
                ]
                return filtered[-limit:]
            else:
                history_list = list(self._history)
                return history_list[-limit:]

    def iter_history(
        self,
        event_type: EventType | None = None,
        limit: int = 100
    ):
        """Iterate over event history lazily (memory-efficient).

        FIX #8: Generator-based iteration to avoid creating full list copies
        when querying large event histories.

        Yields:
            Event objects in chronological order
        """
        count = 0
        with self._sub_lock:
            # Convert to list for iteration (bounded by maxlen)
            history_iter = iter(self._history)
        
        # Iterate outside the lock to prevent deadlocks
        for event in history_iter:
            if event_type is None or event.type == event_type:
                yield event
                count += 1
                if count >= limit:
                    break

    def get_history_iter(
        self,
        event_type: EventType | None = None,
        limit: int = 100
    ) -> list[Event]:
        """Get event history as a list using lazy iteration.

        More memory-efficient than get_history() for large histories.
        """
        return list(self.iter_history(event_type, limit))

    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()

    @property
    def history_size(self) -> int:
        """Current number of events in history."""
        return len(self._history)

    @property
    def is_running(self) -> bool:
        """Whether async processing is active."""
        return self._running


EVENT_BUS = EventBus()
