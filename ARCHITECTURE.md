# Trading Graph Architecture

## Overview

Trading Graph is a desktop AI trading **analysis** system for China A-shares with multi-source market data, news/policy collection, sentiment analysis, and model training.

**Note**: Trading execution components (portfolio management, risk management, OMS, broker integration, auto-trading) have been removed. This system focuses on **analysis and prediction** only.

## Code Statistics

- **Total Python Files**: 246
- **Total Lines of Code**: ~100,000

### Breakdown by Module
| Module | Files | Lines | Description |
|--------|-------|-------|-------------|
| `data/` | 30 | ~25,000 | Data fetching, news collection, sentiment analysis |
| `models/` | 22 | ~20,000 | ML models, training, prediction, news-based training |
| `ui/` | 18 | ~15,000 | PyQt6 application and widgets |
| `tests/` | 80+ | ~25,000 | Test suite |
| `analysis/` | 8 | ~6,000 | Backtest, replay, strategy engine |
| `utils/` | 20 | ~8,000 | Utilities, security, metrics |
| `core/` | 9 | ~4,000 | Core types, events, constants |
| `config/` | 4 | ~1,800 | Settings, runtime environment |
| `strategies/` | 15 | ~3,000 | Trading strategies |
| `scripts/` | 15 | ~3,000 | Utility scripts |

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         UI Layer (PyQt6)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ Main Window │  │Chart Widget │  │ Sentiment Analysis Panel    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Application Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │     App     │  │ Background  │  │  News/Sentiment Analysis    │  │
│  │  Controller │  │    Tasks    │  │                             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Data Layer     │      │   Model Layer    │      │  Sentiment Layer │
│ ┌──────────────┐ │      │ ┌──────────────┐ │      │ ┌──────────────┐ │
│ │ DataFetcher  │ │      │ │ Trainer      │ │      │ │ News         │ │
│ │ NewsCollector│ │      │ │ Predictor    │ │      │ │ Collector    │ │
│ │ Sources      │ │      │ │ NewsTrainer  │ │      │ │ Sentiment    │ │
│ │ Cache        │ │      │ │ AutoLearner  │ │      │ │ Analyzer     │ │
│ │ Database     │ │      │ │ Ensemble     │ │      │ │ EntityExtract│ │
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
├── analysis/          # Backtest, replay, strategy engine
├── config/            # Settings, runtime environment
├── core/              # Core types, events, constants, network
├── data/              # Data fetching, news collection, sentiment analysis
├── models/            # ML models, training, prediction (including news-based)
├── strategies/        # Trading strategies
├── trading/           # Sentiment analysis module only (execution removed)
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

### News Collection Pipeline

```
                    ┌─────────────────┐
                    │  VPN Detection  │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ China Sources│  │International │  │   Fallback   │
    │   (VPN Off)  │  │Sources (VPN) │  │   Sources    │
    │  - Jin10     │  │  - Reuters   │  │  - Cache     │
    │  - EastMoney │  │  - Bloomberg │  │  - Local DB  │
    │  - Sina      │  │  - Yahoo     │  │              │
    └──────┬───────┘  └──────┬───────┘  └──────────────┘
           │                 │
           └────────┬────────┘
                    ▼
           ┌────────────────┐
           │  Normalization │
           │  Deduplication │
           │   Scoring      │
           └────────┬───────┘
                    ▼
           ┌────────────────┐
           │ Sentiment      │
           │ Analysis       │
           └────────┬───────┘
                    ▼
           ┌────────────────┐
           │ Trading Signal │
           │ Generation     │
           └────────────────┘
```

### Model Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Traditional Training                     │
│  Historical Data → Features → LSTM/GRU/TCN → Prediction     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    News-Based Training                      │
│  News Articles → Transformer Encoder ─┐                     │
│  Sentiment Scores → MLP ──────────────┼→ Fusion → Signal    │
│  Price Data → LSTM ───────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### DataFetcher (`data/fetcher.py`)

Multi-source data fetcher with automatic failover:
- **Primary**: Tencent (real-time quotes)
- **Secondary**: AkShare/EastMoney (intraday history)
- **Tertiary**: Sina, Yahoo (fallbacks)

### NewsCollector (`data/news_collector.py`)

VPN-aware news collection system:
- **Chinese Sources** (VPN Off): Jin10, EastMoney, Sina Finance, Xueqiu, Caixin, CSRC
- **International Sources** (VPN On): Reuters, Bloomberg, Yahoo Finance, MarketWatch, CNBC
- Auto-detect network environment
- Multi-source aggregation with health scoring
- Content parsing and normalization

### SentimentAnalyzer (`data/sentiment_analyzer.py`)

Multi-factor sentiment analysis:
- General sentiment scoring (-1.0 to 1.0)
- Policy impact assessment
- Market sentiment detection
- Entity extraction (companies, policies, people)
- Trading signal generation

### NewsTrainer (`models/news_trainer.py`)

News-based model training:
- **NewsEncoder**: Transformer-based text encoder
- **Sentiment Fusion**: Combines sentiment scores with encoded text
- **Price Encoder**: LSTM for historical price patterns
- **Fusion Layer**: Integrates news and price features
- **Prediction Head**: Outputs trading signals and confidence

## Event System

Event-driven architecture using `core/events.py`:

```python
# Event types (analysis-focused)
EVENT_SIGNAL_GENERATED = "signal_generated"
EVENT_NEWS_COLLECTED = "news_collected"
EVENT_SENTIMENT_UPDATED = "sentiment_updated"
EVENT_MODEL_TRAINED = "model_trained"

# Usage
EVENT_BUS.emit(EVENT_SIGNAL_GENERATED, signal=signal)
EVENT_BUS.on(EVENT_SENTIMENT_UPDATED, handler=on_sentiment_change)
```

## Configuration

Environment-based configuration via `config/settings.py`:

```bash
# Network (China optimization)
TRADING_VPN=1
TRADING_PROXY_URL=http://127.0.0.1:7890
TRADING_CHINA_DIRECT=1

# News collection
TRADING_NEWS_LIMIT=100
TRADING_NEWS_HOURS_BACK=24

# Sentiment analysis
TRADING_SENTIMENT_CONFIDENCE_THRESHOLD=0.3
TRADING_SENTIMENT_REFRESH_INTERVAL=30

# Model training
TRADING_TRAINING_EPOCHS=100
TRADING_BATCH_SIZE=32
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
    event="MODEL_TRAIN",
    user="analyst_001",
    details={"model": "news_model", "epochs": 50},
)
```

## Performance Considerations

### Caching Strategy

- **Session cache**: In-memory LRU cache for real-time data
- **News cache**: SQLite for collected articles
- **Model cache**: Pre-loaded models in memory
- **Sentiment cache**: Cached sentiment scores with TTL

### Database Optimization

- WAL mode for concurrent reads
- Indexed queries on symbol/timestamp
- Periodic vacuum and cleanup

## Extension Points

### Adding New News Sources

1. Create source class in `data/news_collector.py`
2. Implement `fetch_news()` method
3. Add to source rotation based on VPN mode

### Adding New Sentiment Features

1. Add lexicon entries to `SentimentAnalyzer`
2. Implement new scoring method
3. Update fusion weights in analysis

### Adding New Models

1. Create model class in `models/`
2. Implement `train()` and `predict()` methods
3. Add to ensemble in `models/ensemble.py`

## Architecture Decisions

See [docs/adr/](docs/adr/) for Architecture Decision Records.

## Future Enhancements

- [ ] Enhanced NLP for Chinese policy documents
- [ ] Real-time news streaming
- [ ] Multi-modal sentiment (text + social media)
- [ ] Cloud backup integration
- [ ] Distributed deployment support
- [ ] Model explainability (SHAP/LIME)
- [ ] 2FA authentication
