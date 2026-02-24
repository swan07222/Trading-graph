# ADR 0001: Use SQLite for Persistence

## Status
Accepted

## Date
2024-01-15

## Context
The trading system requires persistent storage for:
- Order management (OMS)
- Historical market data
- Model artifacts metadata
- Audit logs
- User preferences

We need a storage solution that is:
- Simple to deploy (no separate database server)
- Reliable and ACID-compliant
- Suitable for single-user desktop application
- Easy to backup and migrate

## Decision
Use SQLite as the primary persistence engine with the following characteristics:

- **WAL Mode**: Enable Write-Ahead Logging for concurrent reads
- **Single-file database**: Easy backup and portability
- **No external dependencies**: Built into Python standard library
- **Schema migrations**: Version-controlled schema changes

### Database Schema

**Orders Table:**
```sql
CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Fills Table:**
```sql
CREATE TABLE fills (
    id TEXT PRIMARY KEY,
    order_id TEXT REFERENCES orders(id),
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Market Data Table:**
```sql
CREATE TABLE market_data (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    PRIMARY KEY (symbol, timestamp)
);
```

## Consequences

### Positive
- Zero external database dependencies
- Simple deployment and backup
- Good performance for single-writer workloads
- ACID compliance for financial transactions

### Negative
- Not suitable for high-concurrency multi-user scenarios
- Limited to single-node deployment
- May require migration path for institutional scaling

### Mitigation
- For future multi-user needs, design abstraction layer to support PostgreSQL migration
- Use connection pooling and proper locking strategies
- Implement periodic vacuum and optimization
