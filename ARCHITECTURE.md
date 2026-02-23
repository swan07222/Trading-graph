# Trading Graph - Architecture Documentation

## Overview

Trading Graph is a professional-grade AI-assisted stock trading system for China A-shares. It features a modular architecture with clear separation of concerns, event-driven communication, and production-ready operational controls.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    PyQt6 Desktop Application                      │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │    │
│  │  │  Chart   │  │Watchlist │  │ Portfolio│  │  Auto-Trade Panel│ │    │
│  │  │  Panel   │  │  Table   │  │  Panel   │  │                  │ │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (Event Bus)
┌─────────────────────────────────────────────────────────────────────────┐
│                           BUSINESS LOGIC LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │    Trading   │  │     Risk     │  │    Models    │                  │
│  │    Engine    │  │  Management  │  │    (ML/AI)   │                  │
│  │              │  │              │  │              │                  │
│  │  - OMS       │  │  - VaR       │  │  - Ensemble  │                  │
│  │  - Executor  │  │  - Limits    │  │  - Forecaster│                  │
│  │  - Broker    │  │  - Kill Sw.  │  │  - AutoLearn │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │   Fetcher    │  │   Database   │  │    Cache     │                  │
│  │              │  │              │  │              │                  │
│  │  - Tencent   │  │  - SQLite    │  │  - LRU       │                  │
│  │  - AkShare   │  │  - WAL Mode  │  │  - Tiered    │                  │
│  │  - Sina      │  │  - Sanitize  │  │  - Session   │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           INFRASTRUCTURE LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │   Security   │  │    Audit     │  │   Metrics    │                  │
│  │              │  │              │  │              │                  │
│  │  - Fernet    │  │  - Hash Chain│  │  - HTTP      │                  │
│  │  - 2FA       │  │  - Tamper    │  │  - Prometheus│                  │
│  │  - Key Mgmt  │  │  - Evidence  │  │  - Health    │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### Core Modules (`core/`)

| Module | Responsibility | Key Classes |
|--------|----------------|-------------|
| `types.py` | Domain type definitions | `Order`, `Fill`, `Position`, `Account`, `OrderSide`, `OrderType` |
| `events.py` | Event bus implementation | `EventBus`, `Event`, `BarEvent`, `SignalEvent` |
| `exceptions.py` | Custom exceptions | `TradingError`, `OrderError`, `RiskError`, `SecurityError` |
| `constants.py` | Market constants | `HOLIDAYS`, `LIMIT_PCT`, `LOT_SIZE`, `TRADING_HOURS` |
| `symbols.py` | Symbol parsing | `parse_instrument()`, `instrument_key()` |
| `network.py` | Network utilities | `NetworkEnv`, `get_network_env()` |

### Data Modules (`data/`)

| Module | Responsibility | Key Classes |
|--------|----------------|-------------|
| `fetcher.py` | Multi-source data fetching | `DataFetcher`, `DataSource` |
| `database.py` | SQLite persistence | `MarketDatabase`, sanitization functions |
| `cache.py` | Multi-tier caching | `LRUCache`, `DiskCache`, `TieredCache` |
| `processor.py` | Data processing | `DataProcessor`, `RealtimeBuffer`, `create_labels()` |
| `features.py` | Feature engineering | `FeatureEngine` (57 technical indicators) |
| `session_cache.py` | Session-level caching | `SessionBarCache` |

### Model Modules (`models/`)

| Module | Responsibility | Key Classes |
|--------|----------------|-------------|
| `ensemble.py` | Model ensemble | `EnsembleModel`, model voting |
| `networks.py` | Neural network architectures | `LSTMNet`, `GRUNet`, `TCN`, `TransformerNet` |
| `trainer.py` | Model training | `Trainer`, training pipeline |
| `predictor.py` | Prediction inference | `Predictor`, confidence scoring |
| `auto_learner.py` | AutoML | `AutoLearner`, threshold tuning |

### Trading Modules (`trading/`)

| Module | Responsibility | Key Classes |
|--------|----------------|-------------|
| `oms.py` | Order management | `OrderManagementSystem`, T+1 settlement |
| `executor.py` | Order execution | `ExecutionEngine`, smart routing |
| `broker.py` | Broker abstraction | `BrokerInterface`, `SimulatorBroker` |
| `risk.py` | Risk management | `RiskManager`, VaR, circuit breakers |
| `kill_switch.py` | Emergency stop | `KillSwitch`, auto-pause triggers |
| `portfolio.py` | Portfolio tracking | `Portfolio`, P&L calculation |

### UI Modules (`ui/`)

| Module | Responsibility | Key Classes |
|--------|----------------|-------------|
| `app.py` | Main application | `MainApp`, lifecycle management |
| `charts.py` | Chart rendering | `ChartWidget`, candlestick rendering |
| `widgets.py` | Custom widgets | `SignalTable`, `PortfolioTable` |
| `dialogs.py` | Modal dialogs | `OrderDialog`, `SettingsDialog` |

### Utility Modules (`utils/`)

| Module | Responsibility | Key Classes |
|--------|----------------|-------------|
| `security.py` | Encryption & auth | `SecureStorage`, `AuditLog`, `TradingSecurityError` |
| `metrics.py` | Performance metrics | `ProcessMetrics`, CPU/memory tracking |
| `metrics_http.py` | Metrics server | HTTP server for `/metrics`, `/healthz` |
| `logger.py` | Logging setup | `get_logger()`, rotating file handlers |
| `helpers.py` | General utilities | Type conversion, timestamp utilities |

## Data Flow

### Real-Time Quote Flow
```
Market Data Source (Tencent/AkShare)
         │
         ▼
   Feed Manager (data/feeds.py)
         │
         ├───► Session Cache (real-time bars)
         │
         └───► UI Application
                  │
                  ├───► Chart Update
                  ├───► Watchlist Update
                  └───► Signal Detection
```

### Order Execution Flow
```
User/UI Request
      │
      ▼
Order Validation (Risk Manager)
      │
      ├───► Pre-trade Checks (limits, concentration)
      │
      ▼
Order Submission (OMS)
      │
      ├───► Order ID Assignment
      ├───► Database Persistence
      │
      ▼
Broker Execution
      │
      ├───► Simulator / Live Broker
      ├───► Fill Callback
      │
      ▼
Fill Processing (OMS)
      │
      ├───► Position Update
      ├───► Cash Update
      ├───► T+1 Settlement Tracking
      │
      ▼
Portfolio & Risk Update
```

### ML Prediction Flow
```
Feature Request
      │
      ▼
Data Loading (History + Real-time)
      │
      ▼
Feature Engineering (57 indicators)
      │
      ▼
Model Ensemble (5 models)
      │
      ├───► LSTM Prediction
      ├───► GRU Prediction
      ├───► TCN Prediction
      ├───► Transformer Prediction
      └───► Hybrid Prediction
      │
      ▼
Confidence-Weighted Voting
      │
      ▼
Signal Generation (BUY/HOLD/SELL)
      │
      ▼
Drift Detection Guard
      │
      ▼
Prediction Output
```

## Configuration System

### Configuration Hierarchy
```
1. Default Values (config/settings.py)
         │
         ▼
2. Environment Variables (TRADING_*)
         │
         ▼
3. User Config File (~/.trading_graph/config.json)
         │
         ▼
4. Runtime Overrides (CLI args, UI settings)
```

### Key Configuration Sections

```python
# Data Configuration
data:
  cache_ttl_hours: 24
  max_history_days: 500
  truth_preserving_cleaning: true
  
# Model Configuration  
model:
  ensemble_size: 5
  min_confidence: 0.55
  drift_threshold: 0.15
  
# Risk Configuration
risk:
  max_position_pct: 20.0
  daily_loss_limit: 5000.0
  var_confidence: 0.95
  
# Trading Configuration
trading:
  commission: 0.0003
  lot_size: 100
  mode: "simulation"  # or "live"
```

## Security Architecture

### Encryption
- **Credential Storage**: Fernet (symmetric encryption)
- **Key Management**: External key via `TRADING_SECURE_MASTER_KEY` or local key file
- **Audit Log**: SHA-256 hash chain for tamper evidence

### Access Control
- **2FA Support**: TOTP-based two-factor authentication
- **Role-Based Access**: User/Admin roles with permission levels
- **Live Trading Lock**: Prevents unauthorized live trading

### Audit Trail
```python
# Each audit entry includes:
{
    "timestamp": "2026-02-23T10:30:00Z",
    "event": "ORDER_SUBMIT",
    "user": "trader_001",
    "details": {...},
    "prev_hash": "abc123...",  # Hash chain integrity
    "signature": "def456..."   # Digital signature
}
```

## Performance Optimizations

### Caching Strategy
| Cache Type | TTL | Size Limit | Use Case |
|------------|-----|------------|----------|
| LRU Memory | Session | 500 MB | Frequently accessed data |
| Disk Cache | 24 hours | 2 GB | Historical bars |
| Session Cache | Trading session | 100 MB | Real-time bars |
| Quote Cache | 30 seconds | N/A | Price quotes |
| Trained Stock | 5 minutes | N/A | Model metadata |

### Threading Model
```
Main UI Thread (PyQt6)
      │
      ├───► Worker Threads (analysis, training)
      │
      ├───► Feed Thread (market data)
      │
      ├───► Executor Thread (order execution)
      │
      └───► Metrics Thread (monitoring)
```

### Database Optimization
- **WAL Mode**: Concurrent reads without blocking writes
- **Connection Pooling**: Per-thread connections with cleanup
- **Batch Operations**: Bulk inserts for historical data
- **Indexing**: Optimized queries for time-series data

## Testing Strategy

### Test Pyramid
```
           /\
          /  \    E2E Tests (5%)
         /----\   Integration Tests (25%)
        /      \  Unit Tests (70%)
       /________\
```

### Coverage Requirements
- **Overall**: 85% minimum
- **Critical Paths**: 95%+ (OMS, Risk, Security)
- **UI Components**: 70%+

### Test Categories
1. **Unit Tests**: Individual function/class testing
2. **Integration Tests**: Component interaction testing
3. **Recovery Tests**: Crash recovery, failover scenarios
4. **Security Tests**: Encryption, access control validation
5. **Performance Tests**: Load testing, memory leak detection

## Monitoring & Observability

### Metrics Endpoints
| Endpoint | Description |
|----------|-------------|
| `/metrics` | Prometheus-compatible metrics |
| `/healthz` | Health check (healthy/degraded/unhealthy) |
| `/api/v1/dashboard` | JSON dashboard data |

### Key Metrics
- **System**: CPU%, Memory MB, Thread count
- **Trading**: Orders submitted, fills, rejection rate
- **Risk**: VaR, drawdown, position concentration
- **Model**: Prediction count, confidence distribution, drift score
- **Data**: Fetch latency, cache hit rate, source health

### Logging Levels
| Level | Use Case |
|-------|----------|
| DEBUG | Detailed diagnostic information |
| INFO | Normal operational messages |
| WARNING | Recoverable errors, degraded mode |
| ERROR | Non-recoverable errors |
| CRITICAL | System-wide failures |

## Deployment Architecture

### Single-Node Deployment
```
┌─────────────────────────────────────┐
│         Trading Workstation          │
│  ┌───────────────────────────────┐  │
│  │   Trading Graph Application    │  │
│  │                               │  │
│  │  ┌─────┐ ┌─────┐ ┌─────┐     │  │
│  │  │ UI  │ │ ML  │ │ OMS │     │  │
│  │  └─────┘ └─────┘ └─────┘     │  │
│  │                               │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │   SQLite Database       │  │  │
│  │  └─────────────────────────┘  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

### High Availability (Future)
```
┌─────────────────┐         ┌─────────────────┐
│   Primary Node  │◄───────►│  Secondary Node │
│                 │  Lease  │                 │
│  - Active       │         │  - Standby      │
└─────────────────┘         └─────────────────┘
         │                           │
         └───────────┬───────────────┘
                     │
              ┌──────▼──────┐
              │  Shared DB  │
              │  (PostgreSQL│
              └─────────────┘
```

## Extension Points

### Adding New Data Sources
1. Implement `DataSource` interface
2. Add to fetcher source registry
3. Configure health check parameters
4. Update source selection logic

### Adding New Strategies
1. Create strategy class in `strategies/`
2. Implement `generate_signals()` method
3. Register in `enabled.json`
4. Add backtest configuration

### Adding New Models
1. Implement model architecture in `models/networks.py`
2. Add to ensemble configuration
3. Update model metadata tracking
4. Add drift detection parameters

## Troubleshooting

### Common Issues

| Issue | Symptom | Resolution |
|-------|---------|------------|
| Data fetch timeout | "No market quote" errors | Check network, increase `TRADING_FETCH_TIMEOUT` |
| Model drift | Low confidence predictions | Retrain models, check `drift_threshold` config |
| OMS state mismatch | Fill duplication | Run `--recovery-drill`, verify DB integrity |
| UI freezing | Unresponsive interface | Check worker thread timeouts, reduce data volume |

### Diagnostic Commands
```bash
# Health check
python main.py --health --health-strict

# System diagnostics
python main.py --doctor --doctor-strict

# Recovery drill
python main.py --recovery-drill

# Backtest validation
python main.py --backtest
```

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| 0.1.0 | 2026-01 | Initial release |
| 0.2.0 | 2026-02 | Ensemble models, auto-trading |
| 0.3.0 | 2026-03 | Risk management enhancements |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

Proprietary - All rights reserved.
