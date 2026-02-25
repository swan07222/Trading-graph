"""Modern async event bus with asyncio support.

This module provides:
    - Async event dispatch with asyncio
    - Type-safe event handlers
    - Event prioritization
    - Dead letter queue for failed events
    - Event replay capability
    - Distributed event bus with Redis (optional)
    - Event sourcing support

Example:
    >>> bus = AsyncEventBus()
    >>> await bus.start()
    >>> @bus.on(EventType.SIGNAL_GENERATED)
    >>> async def handle_signal(event: SignalEvent) -> None:
    ...     await process_signal(event)
    >>> await bus.emit(EventType.SIGNAL_GENERATED, signal=signal_data)
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

# FIX #5: Python 3.9 compatibility - use typing_extensions for ParamSpec and TypeVar
try:
    from typing import ParamSpec, TypeVar
except ImportError:
    # Python 3.9 requires typing_extensions for ParamSpec
    from typing_extensions import ParamSpec, TypeVar

from utils.logger import get_logger

log = get_logger()

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class EventType(Enum):
    """Standard event types."""
    # Market events
    TICK = auto()
    BAR = auto()
    QUOTE = auto()
    MARKET_DATA = auto()

    # Trading events
    ORDER_SUBMITTED = auto()
    ORDER_FILLED = auto()
    ORDER_REJECTED = auto()
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()

    # Analysis events
    SIGNAL_GENERATED = auto()
    PREDICTION_READY = auto()
    MODEL_TRAINED = auto()
    NEWS_COLLECTED = auto()
    SENTIMENT_UPDATED = auto()

    # System events
    SYSTEM_START = auto()
    SYSTEM_STOP = auto()
    HEALTH_CHECK = auto()
    ERROR = auto()
    WARNING = auto()


@dataclass
class Event:
    """Base event class with metadata.

    Attributes:
        id: Unique event identifier
        type: Event type
        timestamp: Event creation time
        source: Event source component
        priority: Event priority level
        data: Event payload
        correlation_id: ID for tracking related events
        causation_id: ID of the event that caused this one
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.SYSTEM_START
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    priority: EventPriority = EventPriority.NORMAL
    data: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None
    causation_id: str | None = None
    retries: int = 0
    max_retries: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "type": self.type.name,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "priority": self.priority.value,
            "data": self.data,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Create event from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=EventType[data.get("type", "SYSTEM_START")],
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            source=data.get("source", ""),
            priority=EventPriority(data.get("priority", 1)),
            data=data.get("data", {}),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
        )

    def with_correlation(self, correlation_id: str) -> Event:
        """Set correlation ID (fluent interface)."""
        self.correlation_id = correlation_id
        return self

    def with_causation(self, causation_id: str) -> Event:
        """Set causation ID (fluent interface)."""
        self.causation_id = causation_id
        return self


@dataclass
class EventHandler:
    """Registered event handler."""
    callback: Callable[[Event], Coroutine[Any, Any, None]]
    priority: EventPriority = EventPriority.NORMAL
    filter_func: Callable[[Event], bool] | None = None
    max_concurrency: int = 1

    async def handle(self, event: Event) -> bool:
        """Handle event with optional filtering.

        Returns:
            True if handler executed successfully
        """
        # Apply filter if present
        if self.filter_func and not self.filter_func(event):
            return True  # Filtered out, but not an error

        try:
            await self.callback(event)
            return True
        except Exception as e:
            log.exception(f"Event handler error: {e}")
            return False


@dataclass
class DeadLetter:
    """Failed event for dead letter queue."""
    event: Event
    error: str
    failed_at: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0


class AsyncEventBus:
    """Async event bus with enterprise features.

    Features:
        - Async event dispatch
        - Handler prioritization
        - Dead letter queue for failed events
        - Event replay
        - Event persistence (optional)
        - Distributed bus with Redis (optional)
        - Type-safe handlers

    Example:
        >>> bus = AsyncEventBus()
        >>> await bus.start()
        >>>
        >>> @bus.on(EventType.SIGNAL_GENERATED)
        >>> async def handle_signal(event: Event) -> None:
        ...     print(f"Signal: {event.data}")
        >>>
        >>> await bus.emit(EventType.SIGNAL_GENERATED, signal="BUY")
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        dead_letter_limit: int = 1000,
        enable_replay: bool = True,
        replay_buffer_size: int = 10000,
    ) -> None:
        """Initialize event bus.

        Args:
            max_queue_size: Maximum event queue size
            dead_letter_limit: Maximum dead letter queue size
            enable_replay: Enable event replay capability
            replay_buffer_size: Number of events to keep for replay
        """
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._queue: asyncio.PriorityQueue[tuple[int, float, Event]] = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )
        self._dead_letter_queue: list[DeadLetter] = []
        self._dead_letter_limit = dead_letter_limit
        self._running = False
        self._task: asyncio.Task | None = None
        self._enable_replay = enable_replay
        self._replay_buffer: deque[Event] = deque(maxlen=replay_buffer_size)
        self._event_count = 0
        self._error_count = 0
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> dict[str, Any]:
        """Get bus statistics."""
        return {
            "queue_size": self._queue.qsize(),
            "dead_letter_size": len(self._dead_letter_queue),
            "total_events": self._event_count,
            "total_errors": self._error_count,
            "handlers": sum(len(h) for h in self._handlers.values()),
        }

    def on(
        self,
        event_type: EventType,
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Callable[[Event], bool] | None = None,
    ) -> Callable:
        """Decorator to register event handler.

        Args:
            event_type: Type of event to handle
            priority: Handler priority
            filter_func: Optional filter function

        Returns:
            Decorator function

        Example:
            @bus.on(EventType.SIGNAL_GENERATED, priority=EventPriority.HIGH)
            async def handle_signal(event: Event) -> None:
                await process_signal(event.data)
        """
        def decorator(
            callback: Callable[[Event], Coroutine[Any, Any, None]],
        ) -> Callable[[Event], Coroutine[Any, Any, None]]:
            handler = EventHandler(
                callback=callback,
                priority=priority,
                filter_func=filter_func,
            )
            self._handlers[event_type].append(handler)
            # Sort handlers by priority (highest first)
            self._handlers[event_type].sort(
                key=lambda h: h.priority.value,
                reverse=True,
            )
            log.debug(f"Registered handler for {event_type.name}")
            return callback

        return decorator

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], Coroutine[Any, Any, None]],
        **kwargs: Any,
    ) -> None:
        """Subscribe to event type (alternative to decorator).

        Args:
            event_type: Type of event
            callback: Handler callback
            **kwargs: Additional handler options
        """
        handler = EventHandler(
            callback=callback,
            priority=kwargs.get("priority", EventPriority.NORMAL),
            filter_func=kwargs.get("filter_func"),
        )
        self._handlers[event_type].append(handler)
        self._handlers[event_type].sort(
            key=lambda h: h.priority.value,
            reverse=True,
        )

    async def emit(
        self,
        event_type: EventType,
        source: str = "",
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: str | None = None,
        causation_id: str | None = None,
        **data: Any,
    ) -> str:
        """Emit event to bus.

        Args:
            event_type: Type of event
            source: Event source
            priority: Event priority
            correlation_id: Track related events
            causation_id: Track causal chain
            **data: Event payload

        Returns:
            Event ID

        Raises:
            asyncio.QueueFull: If queue is full
        """
        event = Event(
            type=event_type,
            source=source,
            priority=priority,
            data=data,
            correlation_id=correlation_id,
            causation_id=causation_id,
        )

        await self._queue.put((
            -priority.value,  # Negative for priority ordering
            time.time(),
            event,
        ))

        # Add to replay buffer
        if self._enable_replay:
            self._replay_buffer.append(event)

        self._event_count += 1
        log.debug(f"Event emitted: {event_type.name} (id={event.id})")

        return event.id

    async def emit_event(self, event: Event) -> str:
        """Emit pre-built event.

        Args:
            event: Event object

        Returns:
            Event ID
        """
        await self._queue.put((
            -event.priority.value,
            time.time(),
            event,
        ))

        if self._enable_replay:
            self._replay_buffer.append(event)

        self._event_count += 1
        return event.id

    async def start(self) -> None:
        """Start event bus processing."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        log.info("Event bus started")

    async def stop(self) -> None:
        """Stop event bus processing."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        log.info("Event bus stopped")

    async def _process_loop(self) -> None:
        """Main event processing loop."""
        while self._running:
            try:
                # Get event from queue with timeout
                try:
                    _, _, event = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except TimeoutError:
                    continue

                # Process event
                await self._process_event(event)

                # Mark task as done
                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception(f"Event processing error: {e}")
                self._error_count += 1

    async def _process_event(self, event: Event) -> None:
        """Process single event through all handlers."""
        handlers = self._handlers.get(event.type, [])

        if not handlers:
            log.debug(f"No handlers for {event.type.name}")
            return

        # Execute handlers sequentially (could be parallelized)
        for handler in handlers:
            success = await handler.handle(event)

            if not success:
                event.retries += 1

                if event.retries >= event.max_retries:
                    # Move to dead letter queue
                    await self._add_to_dead_letter(
                        event,
                        f"Max retries ({event.max_retries}) exceeded",
                    )
                else:
                    # Re-queue for retry
                    await self._queue.put((
                        -event.priority.value,
                        time.time(),
                        event,
                    ))
                break

    async def _add_to_dead_letter(self, event: Event, error: str) -> None:
        """Add failed event to dead letter queue."""
        dead_letter = DeadLetter(
            event=event,
            error=error,
            retry_count=event.retries,
        )

        self._dead_letter_queue.append(dead_letter)
        self._error_count += 1

        # Trim if needed
        if len(self._dead_letter_queue) > self._dead_letter_limit:
            self._dead_letter_queue.pop(0)

        log.warning(
            f"Event moved to dead letter queue: {event.type.name} "
            f"(error={error})"
        )

    # Event replay
    async def replay(
        self,
        from_event_id: str | None = None,
        from_timestamp: datetime | None = None,
        event_type: EventType | None = None,
        handler: Callable[[Event], Coroutine[Any, Any, None]] | None = None,
    ) -> int:
        """Replay events from buffer.

        Args:
            from_event_id: Start from specific event
            from_timestamp: Start from specific time
            event_type: Filter by event type
            handler: Custom handler (uses registered handlers if None)

        Returns:
            Number of events replayed
        """
        if not self._enable_replay:
            raise RuntimeError("Event replay is disabled")

        count = 0
        for event in self._replay_buffer:
            # Apply filters
            if from_event_id and event.id <= from_event_id:
                continue
            if from_timestamp and event.timestamp < from_timestamp:
                continue
            if event_type and event.type != event_type:
                continue

            # Replay event
            if handler:
                await handler(event)
            else:
                # Use registered handlers
                handlers = self._handlers.get(event.type, [])
                for h in handlers:
                    await h.handle(event)

            count += 1

        log.info(f"Replayed {count} events")
        return count

    # Dead letter queue management
    async def retry_dead_letter(self, index: int) -> bool:
        """Retry event from dead letter queue.

        Args:
            index: Index in dead letter queue

        Returns:
            True if successfully re-queued
        """
        if index >= len(self._dead_letter_queue):
            return False

        dead_letter = self._dead_letter_queue[index]
        event = dead_letter.event
        event.retries = 0  # Reset retry count

        await self._queue.put((
            -event.priority.value,
            time.time(),
            event,
        ))

        # Remove from dead letter queue
        self._dead_letter_queue.pop(index)

        log.info(f"Re-queued dead letter event: {event.type.name}")
        return True

    async def clear_dead_letter(self) -> int:
        """Clear dead letter queue.

        Returns:
            Number of events cleared
        """
        count = len(self._dead_letter_queue)
        self._dead_letter_queue.clear()
        log.info(f"Cleared {count} events from dead letter queue")
        return count

    def get_dead_letter_queue(self) -> list[DeadLetter]:
        """Get dead letter queue."""
        return self._dead_letter_queue.copy()

    # Event persistence (optional)
    async def export_events(
        self,
        filepath: str,
        from_timestamp: datetime | None = None,
    ) -> int:
        """Export events to file.

        Args:
            filepath: Output file path
            from_timestamp: Export from specific time

        Returns:
            Number of events exported
        """
        events = []
        for event in self._replay_buffer:
            if from_timestamp and event.timestamp < from_timestamp:
                continue
            events.append(event.to_dict())

        with open(filepath, "w") as f:
            json.dump(events, f, indent=2)

        log.info(f"Exported {len(events)} events to {filepath}")
        return len(events)

    async def import_events(self, filepath: str) -> int:
        """Import events from file.

        Args:
            filepath: Input file path

        Returns:
            Number of events imported
        """
        with open(filepath) as f:
            events_data = json.load(f)

        count = 0
        for data in events_data:
            event = Event.from_dict(data)
            await self._queue.put((
                -event.priority.value,
                time.time(),
                event,
            ))
            count += 1

        log.info(f"Imported {count} events from {filepath}")
        return count


# Global event bus instance
_event_bus: AsyncEventBus | None = None


def get_event_bus() -> AsyncEventBus:
    """Get global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = AsyncEventBus()
    return _event_bus


async def init_event_bus() -> AsyncEventBus:
    """Initialize and start event bus."""
    bus = get_event_bus()
    await bus.start()
    return bus


async def shutdown_event_bus() -> None:
    """Shutdown event bus."""
    bus = get_event_bus()
    await bus.stop()
