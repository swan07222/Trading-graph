# ADR 0005: Single-Node Design

## Status
Accepted

## Date
2024-01-15  
Last Updated: 2026-02-25

## Context

The trading system is designed for personal and small-team use with the following requirements:
- Simple deployment (single machine)
- No distributed coordination overhead
- Local data persistence
- Desktop UI integration
- Analysis-focused (not trading execution)

## Decision (Original)

Single-node architecture with:
- All components running on one machine
- SQLite for local persistence
- In-memory caching for performance
- File-based configuration
- Desktop UI (PyQt6) for interaction

## Decision (Extension - 2026-02-25)

Maintain single-node design with focus on analysis capabilities:

### Architecture Boundaries

**Included:**
- Data collection and processing
- ML model training and prediction
- News and sentiment analysis
- Backtesting and replay
- Desktop and web UI for visualization
- Recovery metrics and monitoring

**Excluded (by design):**
- Trading execution (OMS, broker integration)
- Portfolio management
- Risk management for live trading
- Multi-user coordination
- Distributed processing

### Component Layout

```
┌─────────────────────────────────────────────────────────┐
│                    Single Node                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Application Layer                    │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │  │
│  │  │   UI    │  │  CLI    │  │  Web Dashboard  │   │  │
│  │  │ (PyQt6) │  │         │  │    (FastAPI)    │   │  │
│  │  └─────────┘  └─────────┘  └─────────────────┘   │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Core Services                        │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │  │
│  │  │  Data   │  │ Models  │  │   Sentiment     │   │  │
│  │  │ Fetcher │  │(Informer│  │   (LLM-based)   │   │  │
│  │  │         │  │ TFT, etc│  │                 │   │  │
│  │  └─────────┘  └─────────┘  └─────────────────┘   │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Persistence                          │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │  │
│  │  │ SQLite  │  │ Session │  │   Model         │   │  │
│  │  │ (WAL)   │  │  Cache  │  │   Artifacts     │   │  │
│  │  └─────────┘  └─────────┘  └─────────────────┘   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Scaling Characteristics

| Dimension | Limit | Notes |
|-----------|-------|-------|
| **Data Volume** | ~100GB | SQLite + session cache |
| **Stocks Tracked** | ~5000 | All China A-shares |
| **Predictions/Second** | ~100 | Single GPU or CPU |
| **Concurrent Users** | 1-3 | Desktop UI + web dashboard |
| **Memory Usage** | 2-8GB | Configurable caching |

## Consequences

### Positive

- Simple deployment (single binary or script)
- No distributed coordination overhead
- Low latency (all components in-process)
- Easy backup (copy data directory)
- No network dependencies between components
- Suitable for personal and small-team use

### Negative

- Limited to single-machine resources
- No horizontal scaling
- Single point of failure
- Cannot distribute load across machines
- Limited concurrent user support

### Mitigation

- Use efficient algorithms (Informer O(L log L) vs Transformer O(L²))
- Enable GPU acceleration for ML inference
- Use Redis for external caching if needed
- Implement connection pooling for database
- Use async I/O for non-blocking operations

## Future Considerations

If institutional deployment is required:

### Potential Extensions

1. **PostgreSQL for Multi-User Support**
   - Replace SQLite with PostgreSQL
   - Add user authentication and authorization
   - Implement row-level security

2. **Redis for Distributed Caching**
   - External cache for shared state
   - Pub/sub for event distribution
   - Session management

3. **Message Queue for Event Streaming**
   - Kafka or RabbitMQ for event bus
   - Event sourcing for audit trail
   - Stream processing for real-time analytics

4. **Microservices for Independent Scaling**
   - Separate services for data, models, UI
   - API gateway for routing
   - Container orchestration (Kubernetes)

### Migration Path

```
Current (Single-Node)
    ↓
Add PostgreSQL (optional)
    ↓
Add Redis cache (optional)
    ↓
Extract services (data, models)
    ↓
Containerize components
    ↓
Orchestrate with Kubernetes
```

## Implementation Guidelines

### Resource Management

```python
# Use context managers for resources
with get_database() as db:
    bars = db.get_bars(symbol, start, end)

# Use async I/O for non-blocking operations
async def fetch_multiple(symbols):
    tasks = [fetch_symbol(s) for s in symbols]
    return await asyncio.gather(*tasks)

# Use generators for large datasets
def load_bars(symbol):
    for bar in database.query(symbol):
        yield bar
```

### Memory Management

```python
# Limit cache sizes
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_prediction(symbol, timestamp):
    ...

# Use weak references for event handlers
import weakref

def subscribe(handler):
    self._handlers.append(weakref.ref(handler))
```

### Configuration

```bash
# Single-node configuration
TRADING_MODE=simulation
TRADING_MODEL_DIR=models_saved
TRADING_DATA_DIR=data_storage
TRADING_LOG_DIR=logs

# Optional: External cache
TRADING_REDIS_ENABLED=1
TRADING_REDIS_HOST=localhost
TRADING_REDIS_PORT=6379

# Optional: PostgreSQL
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/trading
```

## References

- SQLite: https://sqlite.org/
- Redis: https://redis.io/
- PostgreSQL: https://postgresql.org/
- Twelve-Factor App: https://12factor.net/
