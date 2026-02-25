# Architecture Overview

Trading Graph is a desktop AI analysis system for China A-shares with multi-source data, news/sentiment analysis, and model training.

---

## System Stats

| Metric | Value |
|--------|-------|
| Python Files | 246 |
| Lines of Code | ~100,000 |
| Test Coverage | 85%+ |

### Module Breakdown

| Module | Files | Lines | Purpose |
|--------|-------|-------|---------|
| `data/` | 30 | ~25,000 | Data fetching, news, sentiment |
| `models/` | 22 | ~20,000 | ML models, training, prediction |
| `tests/` | 80+ | ~25,000 | Test suite |
| `ui/` | 18 | ~15,000 | PyQt6 application |
| `analysis/` | 8 | ~6,000 | Backtest, replay, strategy |
| `utils/` | 20 | ~8,000 | Utilities, security, metrics |
| `core/` | 9 | ~4,000 | Types, events, constants |
| `config/` | 4 | ~1,800 | Settings, environment |

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UI Layer (PyQt6 / Web)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Main Window  │  │ Chart Widget │  │ Sentiment Panel  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │App Controller│  │ Background   │  │ News/Sentiment   │   │
│  │              │  │ Tasks        │  │ Analysis         │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   Data Layer     │ │   Model Layer    │ │  Sentiment Layer │
│ ┌──────────────┐ │ │ ┌──────────────┐ │ │ ┌──────────────┐ │
│ │ DataFetcher  │ │ │ │ Trainer      │ │ │ │ Collector    │ │
│ │ NewsCollector│ │ │ │ Predictor    │ │ │ │ Analyzer     │ │
│ │ Cache        │ │ │ │ NewsTrainer  │ │ │ │ EntityExtract│ │
│ │ Database     │ │ │ │ Ensemble     │ │ │ │ SignalGen    │ │
│ └──────────────┘ │ │ └──────────────┘ │ │ └──────────────┘ │
└──────────────────┘ └──────────────────┘ └──────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Events       │  │ Types        │  │ Network (China)  │   │
│  │ (Event Bus)  │  │ (Dataclasses)│  │ (VPN-aware)      │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Infrastructure                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ SQLite       │  │ File System  │  │ Security         │   │
│  │ (Persistence)│  │ (Cache)      │  │ (Encryption)     │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Pipelines

### Real-Time Market Data

```
┌─────────────────────────────────────────────────────────────┐
│  Sources: Tencent, AkShare, Sina, Yahoo                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Source Selection│
                    │ (Health Scoring)│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Primary  │  │Secondary │  │ Fallback │
        │ Tencent  │  │ AkShare  │  │  Sina    │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                   ┌───────────────┐
                   │ Failover Logic│
                   └───────┬───────┘
                           ▼
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        ┌────────┐   ┌────────┐   ┌────────┐
        │ Cache  │   │Database│   │   UI   │
        └────────┘   └────────┘   └────────┘
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
    │ • Jin10      │  │ • Reuters    │  │ • Cache      │
    │ • EastMoney  │  │ • Bloomberg  │  │ • Local DB   │
    │ • Sina       │  │ • Yahoo      │  │              │
    │ • Xueqiu     │  │ • MarketWatch│  │              │
    │ • Caixin     │  │ • CNBC       │  │              │
    │ • CSRC       │  │              │  │              │
    └──────┬───────┘  └──────┬───────┘  └──────────────┘
           │                 │
           └────────┬────────┘
                    ▼
           ┌────────────────┐
           │ Normalization  │
           │ Deduplication  │
           │ Scoring        │
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
│              Traditional Training Pipeline                  │
│                                                             │
│  Historical Data → Feature Engineering → LSTM/GRU/TCN       │
│                                              ↓              │
│                                      Prediction Output      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              News-Based Training Pipeline                   │
│                                                             │
│  News Articles ──→ Transformer Encoder ──┐                  │
│  Sentiment Scores → MLP Encoder ─────────┼→ Fusion Layer    │
│  Price Data ─────→ LSTM Encoder ─────────┘     ↓            │
│                                      Prediction + Confidence│
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### DataFetcher (`data/fetcher.py`)

Multi-source data fetcher with automatic failover.

| Priority | Source | Use Case |
|----------|--------|----------|
| Primary | Tencent | Real-time quotes |
| Secondary | AkShare | Intraday history |
| Tertiary | Sina, Yahoo | Fallback |

### NewsCollector (`data/news_collector.py`)

VPN-aware news collection with auto source selection.

**Features:**
- Network environment auto-detection
- Multi-source aggregation
- Content normalization and deduplication
- Health scoring and failover

### SentimentAnalyzer (`data/sentiment_analyzer.py`)

Multi-factor sentiment analysis engine.

**Capabilities:**
- General sentiment scoring (-1.0 to 1.0)
- Policy impact assessment
- Market sentiment detection
- Entity extraction (companies, policies, people)
- Trading signal generation

### NewsTrainer (`models/news_trainer.py`)

News-based model training with multi-modal fusion.

| Component | Architecture | Purpose |
|-----------|--------------|---------|
| NewsEncoder | Transformer | Text encoding |
| SentimentFusion | MLP | Sentiment integration |
| PriceEncoder | LSTM | Price pattern learning |
| FusionLayer | Attention | Multi-modal fusion |
| PredictionHead | Dense | Signal + confidence |

---

## Event System

Event-driven architecture via `core/events.py`.

### Event Types

| Event | Description |
|-------|-------------|
| `EVENT_SIGNAL_GENERATED` | Trading signal created |
| `EVENT_NEWS_COLLECTED` | News articles collected |
| `EVENT_SENTIMENT_UPDATED` | Sentiment scores updated |
| `EVENT_MODEL_TRAINED` | Model training completed |

### Usage Example

```python
from core.events import EVENT_BUS, EVENT_SIGNAL_GENERATED

# Emit event
EVENT_BUS.emit(EVENT_SIGNAL_GENERATED, signal=signal)

# Subscribe to event
EVENT_BUS.on(EVENT_SIGNAL_GENERATED, handler=on_signal)
```

---

## Configuration

Environment-based configuration via `config/settings.py`.

### Network Settings

```bash
TRADING_VPN=1                              # Enable VPN mode
TRADING_PROXY_URL=http://127.0.0.1:7890   # Proxy URL
TRADING_CHINA_DIRECT=1                     # China direct mode
```

### News Collection

```bash
TRADING_NEWS_LIMIT=100                     # Articles per source
TRADING_NEWS_HOURS_BACK=24                 # Time window
```

### Sentiment Analysis

```bash
TRADING_SENTIMENT_CONFIDENCE_THRESHOLD=0.3
TRADING_SENTIMENT_REFRESH_INTERVAL=30
```

### Model Training

```bash
TRADING_TRAINING_EPOCHS=100
TRADING_BATCH_SIZE=32
```

---

## Security

### Credential Storage

Encrypted storage using `cryptography.fernet`:
- Master key in environment or secure file
- Credentials encrypted at rest
- Audit logging on all access

### Audit Logging

```python
from utils.security import get_audit_log

audit = get_audit_log()
audit.log(
    event="MODEL_TRAIN",
    user="analyst_001",
    details={"model": "news_model", "epochs": 50},
)
```

---

## Performance

### Caching Strategy

| Cache Type | Storage | TTL |
|------------|---------|-----|
| Session | In-memory LRU | Session lifetime |
| News | SQLite | 24 hours |
| Models | In-memory | Until retrain |
| Sentiment | SQLite + Memory | 1 hour |

### Database Optimization

- WAL mode for concurrent reads
- Indexed queries on symbol/timestamp
- Periodic vacuum and cleanup

---

## Extension Points

### Adding News Sources

1. Create source class in `data/news_collector.py`
2. Implement `fetch_news()` method
3. Register in source rotation (VPN on/off)

### Adding Sentiment Features

1. Add lexicon entries to `SentimentAnalyzer`
2. Implement scoring method
3. Update fusion weights

### Adding ML Models

1. Create model class in `models/`
2. Implement `train()` and `predict()`
3. Register in `models/ensemble.py`

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| SQLite | Simple, embedded, no external dependencies |
| PyQt6 | Mature desktop framework, rich widgets |
| PyTorch | Flexible ML framework, active community |
| Event-driven | Loose coupling, extensibility |
| Single-node | Personal/small-team use case |

---

## Future Enhancements

- [ ] Enhanced NLP for Chinese policy documents
- [ ] Real-time news streaming
- [ ] Multi-modal sentiment (text + social)
- [ ] Cloud backup integration
- [ ] Model explainability (SHAP/LIME)
- [ ] 2FA authentication
