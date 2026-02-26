# ADR 0004: Event-Driven Architecture

## Status
Accepted

## Date
2024-01-15  
Last Updated: 2026-02-25

## Context

The trading system has multiple components that need to communicate:
- UI needs real-time updates from data layer
- Prediction engine triggers chart updates
- News collector notifies sentiment analyzer
- Model training completion signals UI
- System health monitoring broadcasts status

We need a communication mechanism that:
- Decouples components
- Supports asynchronous messaging
- Enables event sourcing and replay
- Provides reliable delivery
- Thread-safe for multi-threaded environment

## Decision

Use an event-driven architecture with a central event bus:

### Event Bus Implementation

```python
# core/events.py
class EventBus:
    """Thread-safe event bus for component communication."""
    
    def __init__(self):
        self._subscribers = defaultdict(list)
        self._lock = threading.RLock()
    
    def emit(self, event_type: str, **data) -> None:
        """Publish event to all subscribers (thread-safe)."""
        with self._lock:
            for handler in self._subscribers[event_type]:
                handler(**data)
    
    def on(self, event_type: str, handler: Callable) -> None:
        """Subscribe handler to event type."""
        with self._lock:
            self._subscribers[event_type].append(handler)
    
    def off(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe handler from event type."""
        with self._lock:
            self._subscribers[event_type].remove(handler)
    
    def start(self) -> None:
        """Start event bus (initialize resources)."""
        pass
    
    def stop(self) -> None:
        """Stop event bus (cleanup resources)."""
        with self._lock:
            self._subscribers.clear()
```

### Core Events

| Event | Source | Consumers | Payload |
|-------|--------|-----------|---------|
| `EVENT_QUOTE_UPDATE` | DataFetcher | Chart, MarketWatch | symbol, price, volume |
| `EVENT_BAR_COMPLETE` | DataFetcher | Chart, Predictor | symbol, bar, interval |
| `EVENT_PREDICTION_READY` | Predictor | Chart, UI | symbol, prediction |
| `EVENT_SIGNAL_GENERATED` | Predictor | UI, Monitoring | symbol, signal, confidence |
| `EVENT_NEWS_COLLECTED` | NewsCollector | SentimentAnalyzer | articles |
| `EVENT_SENTIMENT_UPDATED` | SentimentAnalyzer | UI, Strategy | symbol, sentiment |
| `EVENT_MODEL_TRAINED` | Trainer | UI, Predictor | model_id, metrics |
| `EVENT_SYSTEM_START` | App | All | timestamp |
| `EVENT_SYSTEM_STOP` | App | All | timestamp |

### Event Flow

```
┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│ DataFetcher  │────▶│ Event Bus   │◀────│ Predictor    │
└──────────────┘     └──────┬──────┘     └──────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐
     │   Chart    │ │     UI     │ │ Monitoring │
     │  Widget    │ │            │ │            │
     └────────────┘ └────────────┘ └────────────┘
```

## Consequences

### Positive

- Loose coupling between components
- Easy to add new event handlers without modifying sources
- Supports event replay for debugging and testing
- Natural fit for real-time trading systems
- Thread-safe implementation for multi-threaded environment
- Enables async processing of events

### Negative

- Event flow can be harder to trace than direct calls
- Potential for event storms under high load
- Requires careful handler error handling (one handler failure shouldn't affect others)
- Memory leaks if handlers not properly unsubscribed

### Mitigation

- Implement event logging for debugging
- Use try/except in handlers to prevent cascading failures
- Add circuit breakers for event handlers under load
- Use weak references for handlers when appropriate
- Document all event types and payloads

## Implementation

### Emitting Events

```python
# data/fetcher.py
from core.events import EVENT_BUS, EVENT_QUOTE_UPDATE

def on_quote_received(self, symbol, price, volume):
    # Process quote...
    
    # Emit event
    EVENT_BUS.emit(
        EVENT_QUOTE_UPDATE,
        symbol=symbol,
        price=price,
        volume=volume,
        timestamp=datetime.now(),
    )
```

### Subscribing to Events

```python
# ui/app.py
from core.events import EVENT_BUS, EVENT_QUOTE_UPDATE

class TradingApp:
    def __init__(self):
        # Subscribe to events
        EVENT_BUS.on(EVENT_QUOTE_UPDATE, self.on_quote_update)
        EVENT_BUS.on(EVENT_PREDICTION_READY, self.on_prediction_update)
    
    def on_quote_update(self, symbol, price, volume, timestamp):
        # Update chart
        self.chart.update_quote(symbol, price, volume)
        
        # Update market watch
        self.market_watch.update_quote(symbol, price, volume)
```

### Event Handler Best Practices

```python
# ✅ Good - Handler with error handling
def on_quote_update(symbol, price, volume):
    try:
        # Process event
        update_chart(symbol, price)
    except Exception as e:
        log.exception(f"Error handling quote update: {e}")
        # Don't re-raise - prevent cascading failures

# ❌ Bad - Handler without error handling
def on_quote_update(symbol, price, volume):
    update_chart(symbol, price)  # Exception will affect other handlers
```

## Patterns

### Request-Reply Pattern

For synchronous communication:

```python
class RequestReplyBus(EventBus):
    def request(self, event_type: str, timeout: float = 5.0, **data) -> Any:
        """Send request and wait for reply."""
        future = Future()
        
        def handler(reply, **kwargs):
            future.set_result(reply)
        
        request_id = str(uuid.uuid4())
        self.on(f"{event_type}_reply_{request_id}", handler)
        
        self.emit(event_type, request_id=request_id, **data)
        
        try:
            return future.result(timeout=timeout)
        finally:
            self.off(f"{event_type}_reply_{request_id}", handler)
```

### Event Sourcing

For audit and replay:

```python
class EventStore:
    def __init__(self):
        self._events = []
    
    def store(self, event_type: str, data: dict):
        self._events.append({
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        })
    
    def replay(self, from_timestamp: str = None):
        for event in self._events:
            if from_timestamp and event["timestamp"] < from_timestamp:
                continue
            yield event
```

## References

- Event-Driven Architecture: https://en.wikipedia.org/wiki/Event-driven_architecture
- Publish-Subscribe Pattern: https://en.wikipedia.org/wiki/Publish%E2%80%93subscribe_pattern
- Python threading: https://docs.python.org/3/library/threading.html
