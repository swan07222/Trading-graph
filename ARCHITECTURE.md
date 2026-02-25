# Trading Graph Architecture

## Overview

Trading Graph is a desktop AI trading system for China A-shares with multi-source market data, model training, and auto-trade execution.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         UI Layer (PyQt6)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │ Main Window │  │ Chart Widget│  │ Order Panel │  │  Widgets   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Application Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │AppController│  │ Background  │  │  Analysis   │  │  Dialogs   │  │
│  │             │  │    Tasks    │  │     Ops     │  │            │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Data Layer     │      │   Model Layer    │      │  Trading Layer   │
│ ┌──────────────┐ │      │ ┌──────────────┐ │      │ ┌──────────────┐ │
│ │ DataFetcher  │ │      │ │ Trainer      │ │      │ │ OMS          │ │
│ │ Sources      │ │      │ │ Predictor    │ │      │ │ Risk Manager │ │
│ │ Cache        │ │      │ │ AutoLearner  │ │      │ │ Portfolio    │ │
│ │ Database     │ │      │ │ Ensemble     │ │      │ │ Broker       │ │
│ └──────────────┘ │      │ └──────────────┘ │      │ └──────────────┘ │
└──────────────────┘      └──────────────────┘      └──────────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Core Layer                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │ Events      │  │ Types       │  │ Constants   │  │ Network    │  │
│  │ (Event Bus) │  │(Dataclasses)│  │ (Config)    │  │ (China)    │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Infrastructure                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │ SQLite      │  │ File System │  │ Security    │  │ Logging    │  │
│  │(Persistence)│  │ (Cache)     │  │ (Crypto)    │  │ (Audit)    │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
trading-graph/
├── analysis/          # Backtest, replay, sentiment, strategy engine
├── config/            # Settings, runtime environment
├── core/              # Core types, events, constants, network
├── data/              # Data fetching, caching, processing
├── models/            # ML models, training, prediction
├── strategies/        # Trading strategies
├── trading/           # OMS, risk, portfolio, broker integration
├── ui/                # PyQt6 application and widgets
├── utils/             # Utilities, security, metrics
├── tests/             # Test suite
└── docs/              # Documentation
```

## Data Flow

### Real-Time Data Pipeline

```
Market Data Sources → DataFetcher → Cache → Database → UI/Models
     (Tencent)          │           │        │          │
     (AkShare)          ▼           ▼        ▼          ▼
     (Sina)      [Source Selection] [Persistence] [Real-time Update]
     (Yahoo)           │
                       ▼
                [Health Scoring]
                       │
                       ▼
                [Failover Logic]
```

### Model Training Pipeline

```
Historical Data → Feature Engineering → Model Training → Model Persistence
     │                  │                    │                │
     ▼                  ▼                    ▼                ▼
[Database]      [Technical Indicators] [LSTM/GRU/TCN]   [.pt files]
                [Statistical Features] [Ensemble]       [.pkl scalers]
```

### Trade Execution Flow

```
Signal Generation → Risk Check → Order Submission → Broker → Fill Report
     │                  │              │             │          │
     ▼                  ▼              ▼             ▼          ▼
[Model/Strategy]  [Risk Manager]  [OMS]        [Broker API] [Database]
```

## Key Components

### DataFetcher (`data/fetcher.py`)

Multi-source data fetcher with automatic failover:
- **Primary**: Tencent (real-time quotes)
- **Secondary**: AkShare/EastMoney (intraday history)
- **Tertiary**: Sina, Yahoo (fallbacks)

### OrderManagementSystem (`trading/oms.py`)

Order management with SQLite persistence:
- Order lifecycle tracking
- Fill reconciliation
- Position management
- P&L calculation

### RiskManager (`trading/risk.py`)

Real-time risk monitoring:
- Position limits
- Daily loss limits
- Drawdown controls
- Circuit breaker

### AutoTrader (`trading/auto_trader.py`)

Automated trading execution:
- Signal processing
- Confidence thresholds
- Cooldown management
- Approval workflows (SEMI_AUTO mode)

## Event System

Event-driven architecture using `core/events.py`:

```python
# Event types
EVENT_ORDER_SUBMITTED = "order_submitted"
EVENT_ORDER_FILLED = "order_filled"
EVENT_SIGNAL_GENERATED = "signal_generated"
EVENT_RISK_WARNING = "risk_warning"
EVENT_CIRCUIT_BREAKER = "circuit_breaker"

# Usage
EVENT_BUS.emit(EVENT_ORDER_SUBMITTED, order=order)
EVENT_BUS.on(EVENT_ORDER_FILLED, handler=on_fill)
```

## Configuration

Environment-based configuration via `config/settings.py`:

```bash
# Trading mode
TRADING_MODE=simulation  # simulation | live

# Risk limits
TRADING_MAX_POSITION_PCT=15
TRADING_MAX_DAILY_LOSS_PCT=3

# Network (China optimization)
TRADING_VPN=1
TRADING_PROXY_URL=http://127.0.0.1:7890

# Auto-trade
TRADING_AUTO_TRADE_MODE=manual  # manual | semi_auto | auto
TRADING_MIN_CONFIDENCE=0.70
```

## Security

### Credential Storage

Encrypted credential storage using `cryptography.fernet`:
- Master key stored in environment or secure file
- Credentials encrypted at rest
- Audit logging for all access

### Audit Logging

Comprehensive audit trail:
```python
audit.log(
    event="ORDER_SUBMIT",
    user="trader_001",
    details={"order_id": "ORD-123", "symbol": "600519"},
)
```

## Performance Considerations

### Caching Strategy

- **Session cache**: In-memory LRU cache for real-time data
- **Disk cache**: SQLite for historical data
- **Model cache**: Pre-loaded models in memory

### Database Optimization

- WAL mode for concurrent reads
- Indexed queries on symbol/timestamp
- Periodic vacuum and cleanup

## Extension Points

### Adding New Data Sources

1. Create source class in `data/fetcher_sources.py`
2. Implement `fetch_quote()` and `fetch_history()`
3. Add to source rotation in `DataFetcher`

### Adding New Strategies

1. Create strategy class in `strategies/`
2. Implement `generate_signal()` method
3. Register in strategy marketplace

### Adding New Models

1. Create model class in `models/`
2. Implement `train()` and `predict()` methods
3. Add to ensemble in `models/ensemble.py`

## Architecture Decisions

See [docs/adr/](docs/adr/) for Architecture Decision Records.

## Future Enhancements

- [ ] Multi-asset support (futures, options)
- [ ] Cloud backup integration
- [ ] Distributed deployment support
- [ ] Model explainability (SHAP/LIME)
- [ ] 2FA authentication
- [ ] FIX protocol integration
