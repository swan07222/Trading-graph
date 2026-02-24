# ADR 0005: Single-Node Design

## Status
Accepted (Being Extended)

## Date
2024-01-15
Last Updated: 2026-02-23

## Context
The trading system is designed for personal and small-team use with the following requirements:
- Simple deployment (single machine)
- No distributed coordination overhead
- Local data persistence
- Desktop UI integration

## Decision (Original)
Single-node architecture with:
- All components running on one machine
- SQLite for local persistence
- In-memory caching for performance
- File-based configuration

## Decision (Extension - 2026-02-23)
Extend the system to support multi-asset trading while maintaining single-node simplicity:

### New Asset Classes

**Futures Support:**
- China Financial Futures Exchange (CFFEX)
- Shanghai Futures Exchange (SHFE)
- Zhengzhou Commodity Exchange (CZCE)
- Dalian Commodity Exchange (DCE)

**Options Support:**
- SSE 50 ETF Options
- SZSE 300 ETF Options
- Stock Options (individual)

### Extended Types

```python
class AssetType(Enum):
    STOCK = "stock"
    FUTURES = "futures"
    OPTIONS = "options"
    FOREX = "forex"
    CRYPTO = "crypto"

class FuturesContract:
    symbol: str           # e.g., "IF2603"
    underlying: str       # e.g., "IF" (CSI 300)
    expiry: date          # e.g., 2026-03-20
    multiplier: float     # e.g., 300
    tick_size: float      # e.g., 0.2
    margin_rate: float    # e.g., 0.12

class OptionsContract:
    symbol: str           # e.g., "10005001"
    underlying: str       # e.g., "000300"
    strike: float         # e.g., 4000
    expiry: date          # e.g., 2026-03-27
    option_type: str      # "call" or "put"
    multiplier: float     # e.g., 10000
```

### Multi-Asset OMS

Extended Order Management System to handle:
- Different margin requirements
- Contract rollover for futures
- Options Greeks calculation
- Multi-asset risk aggregation

## Consequences

### Positive
- Maintains deployment simplicity
- Adds significant capability without complexity
- Enables diversification strategies
- Better risk management through multiple asset classes

### Negative
- Still limited to single-node scaling
- More complex risk calculations
- Larger data storage requirements

### Future Considerations
For institutional deployment, consider:
- PostgreSQL for multi-user support
- Redis for distributed caching
- Message queue (Kafka/RabbitMQ) for event streaming
- Microservices architecture for independent scaling
