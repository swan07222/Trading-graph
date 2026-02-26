# ADR 0001: Use SQLite for Persistence

## Status
Accepted

## Date
2024-01-15  
Last Updated: 2026-02-25

## Context

The trading system requires persistent storage for:
- Historical market data (bars, quotes)
- Model artifacts metadata
- Session cache (per-symbol CSV compaction)
- Audit logs
- User preferences
- Recovery metrics

We need a storage solution that is:
- Simple to deploy (no separate database server)
- Reliable and ACID-compliant
- Suitable for single-user desktop application
- Easy to backup and migrate
- Support for concurrent reads

## Decision

Use SQLite as the primary persistence engine with the following characteristics:

### Configuration

- **WAL Mode**: Enable Write-Ahead Logging for concurrent reads without blocking writes
- **Single-file database**: Easy backup and portability
- **No external dependencies**: Built into Python standard library
- **Session cache**: Per-symbol CSV files with compaction for efficient storage

### Database Schema

**Market Data Table:**
```sql
CREATE TABLE IF NOT EXISTS market_data (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    amount REAL,
    PRIMARY KEY (symbol, timestamp)
);
```

**Session Cache Metadata:**
```sql
CREATE TABLE IF NOT EXISTS session_cache (
    symbol TEXT PRIMARY KEY,
    last_updated TIMESTAMP,
    row_count INTEGER,
    file_path TEXT,
    checksum TEXT
);
```

### Session Cache Strategy

- Per-symbol CSV files for efficient columnar access
- Compaction every N writes (default: 240)
- Retention: 45 days or 12,000 rows per symbol
- Max file size: 8 MB per symbol

## Consequences

### Positive

- Zero external database dependencies
- Simple deployment and backup (single file)
- Good performance for single-writer workloads
- ACID compliance for financial transactions
- WAL mode enables concurrent reads without blocking
- Session cache provides efficient per-symbol access

### Negative

- Not suitable for high-concurrency multi-user scenarios
- Limited to single-node deployment
- Write operations block other writes
- May require migration path for institutional scaling

### Mitigation

- Use `aiosqlite` for async operations to avoid blocking
- Implement connection pooling for efficient resource usage
- Use session cache for hot data to reduce database load
- For future multi-user needs, design abstraction layer to support PostgreSQL migration
- Implement periodic vacuum and optimization

## Implementation

```python
from data.database import get_database

db = get_database()

# Insert market data
db.insert_bars(symbol="600519", bars=df)

# Query historical data
bars = db.get_bars(symbol="600519", start=start_date, end=end_date)

# Session cache operations
from data.session_cache import SessionCache
cache = SessionCache()
cache.store_bars(symbol="600519", bars=df)
```

## References

- SQLite Documentation: https://sqlite.org/docs.html
- WAL Mode: https://sqlite.org/wal.html
- SQLAlchemy: https://sqlalchemy.org
