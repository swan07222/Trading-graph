# ADR 0004: Event-Driven Architecture

## Status
Accepted

## Date
2024-01-15

## Context
The trading system has multiple components that need to communicate:
- UI needs real-time updates from data layer
- Trading engine reacts to model signals
- Risk manager monitors all transactions
- Audit logger tracks all events

We need a communication mechanism that:
- Decouples components
- Supports asynchronous messaging
- Enables event sourcing and replay
- Provides reliable delivery

## Decision
Use an event-driven architecture with a central event bus:

### Event Bus Implementation

```python
class EventBus:
    def emit(self, event_type: str, **data) -> None:
        """Publish event to all subscribers"""
        
    def on(self, event_type: str, handler: Callable) -> None:
        """Subscribe handler to event type"""
        
    def off(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe handler from event type"""
```

### Core Events

```python
# Market data events
EVENT_QUOTE_UPDATE = "quote_update"
EVENT_BAR_COMPLETE = "bar_complete"

# Order events
EVENT_ORDER_SUBMITTED = "order_submitted"
EVENT_ORDER_FILLED = "order_filled"
EVENT_ORDER_CANCELLED = "order_cancelled"

# Signal events
EVENT_SIGNAL_GENERATED = "signal_generated"
EVENT_PREDICTION_READY = "prediction_ready"

# Risk events
EVENT_RISK_WARNING = "risk_warning"
EVENT_CIRCUIT_BREAKER = "circuit_breaker"

# System events
EVENT_SYSTEM_START = "system_start"
EVENT_SYSTEM_STOP = "system_stop"
```

## Consequences

### Positive
- Loose coupling between components
- Easy to add new event handlers
- Supports event replay for debugging
- Natural fit for real-time trading systems

### Negative
- Event flow can be harder to trace
- Potential for event storms under high load
- Requires careful handler error handling

### Mitigation
- Implement event logging and tracing
- Use priority queues for critical events
- Add circuit breakers for event handlers
