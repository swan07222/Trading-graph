# Architecture Overview

Trading Graph is a desktop AI analysis system for China A-shares with multi-source data, LLM-powered sentiment analysis, and modern deep learning models (Informer, TFT, N-BEATS, TSMixer).

---

## System Stats

| Metric | Value |
|--------|-------|
| Python Files | ~200 |
| Lines of Code | ~80,000 |
| Test Coverage | 40%+ |

### Module Breakdown

| Module | Files | Lines | Purpose |
|--------|-------|-------|---------|
| `data/` | ~40 | ~20,000 | Data fetching, news collection, LLM sentiment |
| `models/` | ~25 | ~18,000 | GM ensemble (Informer, TFT, N-BEATS, TSMixer), LLM trainer |
| `ui/` | ~20 | ~15,000 | PyQt6 application |
| `tests/` | ~20 | ~10,000 | Test suite |
| `utils/` | ~25 | ~8,000 | Utilities, security, metrics |
| `analysis/` | ~10 | ~6,000 | Backtest, replay, strategy |
| `core/` | ~10 | ~4,000 | Types, events, constants, symbols |
| `config/` | ~4 | ~2,000 | Settings, environment |

### Model Storage

| Directory | Contents |
|-----------|----------|
| `models_saved/GM/` | Guessing Model artifacts (ensemble weights, forecasters, scalers) |
| `models_saved/LLM/` | LLM sentiment models (transformer weights, embeddings, classifiers) |

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
│ │ DataFetcher  │ │ │ │ GM Trainer   │ │ │ │ LLM Analyzer │ │
│ │ NewsCollector│ │ │ │ (Informer,   │ │ │ │ (Transformer)│ │
│ │ Cache        │ │ │ │  TFT, N-BEATS,│ │ │ │ Sentiment    │ │
│ │ Database     │ │ │ │  TSMixer)    │ │ │ │ Collector    │ │
│ │              │ │ │ │ Predictor    │ │ │ │ EntityExtract│ │
│ │              │ │ │ │ Ensemble     │ │ │ │ SignalGen    │ │
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
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ Redis        │  │ Prometheus   │                         │
│  │ (Optional)   │  │ Metrics      │                         │
│  └──────────────┘  └──────────────┘                         │
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
    │ • EastMoney  │  │ • Reuters    │  │ • Cache      │
    │ • Sina       │  │ • Bloomberg  │  │ • Local DB   │
    │ • Xueqiu     │  │ • Yahoo      │  │              │
    │ • Caixin     │  │ • MarketWatch│  │              │
    │ • CSRC       │  │ • CNBC       │  │              │
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
           │ LLM Sentiment  │
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
│         Hybrid Model Training Pipeline                      │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  GM Training    │    │  LLM Training   │                │
│  │  (Price)        │    │  (Sentiment)    │                │
│  │                 │    │                 │                │
│  │  Historical     │    │  News Articles  │                │
│  │  Data → Features│    │  → Embeddings   │                │
│  │                 │    │                 │                │
│  │  Ensemble:      │    │  Transformer +  │                │
│  │  - Informer     │    │  MLP Classifier │                │
│  │  - TFT          │    │                 │                │
│  │  - N-BEATS      │    │  Bilingual      │                │
│  │  - TSMixer      │    │  (zh/en)        │                │
│  │                 │    │                 │                │
│  │  models_saved/  │    │  models_saved/  │                │
│  │  GM/            │    │  LLM/           │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                      │                          │
│           └──────────┬───────────┘                          │
│                      ▼                                      │
│           ┌───────────────────┐                             │
│           │   Prediction      │                             │
│           │   (GM + LLM)      │                             │
│           └───────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### DataFetcher (`data/fetcher.py`)

Multi-source data fetcher with automatic failover and quality validation.

| Priority | Source | Use Case |
|----------|--------|----------|
| Primary | Tencent | Real-time quotes |
| Secondary | AkShare | Intraday history |
| Tertiary | Sina, Yahoo | Fallback |

**Features:**
- VPN-aware routing
- Multi-source reconciliation
- Data quality validation
- Session caching with compaction
- History flow with depth retry

### NewsCollector (`data/news_collector.py`)

VPN-aware news collection with auto source selection.

**Features:**
- Network environment auto-detection
- Multi-source aggregation
- Content normalization and deduplication
- Health scoring and failover
- Category classification (policy, market, company, economic, regulatory)

### LLM Sentiment Analyzer (`data/llm_sentiment.py`)

Bilingual (Chinese/English) LLM sentiment analyzer with auto-training capability.

**Capabilities:**
- Transformer-based sentiment classification
- Auto-training on collected news data
- Policy impact assessment
- Market sentiment detection
- Entity extraction (companies, policies, people, sectors)
- Trading signal generation
- Hybrid fallback: sklearn MLP when transformer unavailable
- Model storage: `models_saved/LLM/`

### Guessing Model Ensemble (`models/ensemble.py`)

Ensemble of modern architectures for price prediction with calibrated weighted voting.

| Model | Architecture | Purpose |
|-------|--------------|---------|
| **Informer** | Probabilistic attention | O(L log L) complexity for long sequences |
| **TFT** | Temporal Fusion Transformer | Interpretable multi-horizon predictions |
| **N-BEATS** | Neural basis expansion | Trend and seasonality decomposition |
| **TSMixer** | All-MLP | Efficient time series mixing |

**Features:**
- Weighted voting based on validation performance
- Uncertainty quantification
- Prediction calibration
- Walk-forward validation
- Drift detection
- Model storage: `models_saved/GM/`

### Predictor (`models/predictor.py`)

Real-time prediction engine with multi-interval support combining GM and LLM outputs.

**Features:**
- Multi-interval support (1m, 5m, 15m, 30m, 1h, 1d)
- Prediction caching with configurable TTL
- News sentiment integration from LLM
- Position sizing based on Kelly criterion
- Trading levels (support/resistance)
- Forecast horizon prediction

---

## Event System

Event-driven architecture via `core/events.py`.

### Event Types

| Event | Description |
|-------|-------------|
| `EVENT_QUOTE_UPDATE` | Real-time quote received |
| `EVENT_BAR_COMPLETE` | Candlestick bar completed |
| `EVENT_SIGNAL_GENERATED` | Trading signal created |
| `EVENT_PREDICTION_READY` | Prediction ready for display |
| `EVENT_NEWS_COLLECTED` | News articles collected |
| `EVENT_SENTIMENT_UPDATED` | Sentiment scores updated |
| `EVENT_MODEL_TRAINED` | Model training completed |
| `EVENT_SYSTEM_START` | System started |
| `EVENT_SYSTEM_STOP` | System stopped |

### Usage Example

```python
from core.events import EVENT_BUS, EVENT_SIGNAL_GENERATED

# Emit event
EVENT_BUS.emit(EVENT_SIGNAL_GENERATED, signal=signal)

# Subscribe to event
EVENT_BUS.on(EVENT_SIGNAL_GENERATED, handler=on_signal)

# Start/Stop event bus
EVENT_BUS.start()
EVENT_BUS.stop()
```

---

## Configuration

Environment-based configuration via `config/settings.py`.

### Network Settings

```bash
TRADING_VPN=1                              # Enable VPN mode
TRADING_PROXY_URL=http://127.0.0.1:7890   # Proxy URL
TRADING_CHINA_DIRECT=1                     # China direct mode
TRADING_CONNECTION_TIMEOUT=30              # Connection timeout (seconds)
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
TRADING_SEQUENCE_LENGTH=60
```

### Metrics & Monitoring

```bash
TRADING_METRICS_ENABLED=1                  # Enable metrics server
TRADING_METRICS_PORT=8000                  # Metrics server port
TRADING_METRICS_HOST=127.0.0.1             # Metrics server host
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
    details={"model": "ensemble", "epochs": 100},
)
```

---

## Performance

### Caching Strategy

| Cache Type | Storage | TTL |
|------------|---------|-----|
| Session | SQLite (per-symbol CSV) | 45 days |
| Real-time quotes | In-memory LRU | 5 seconds |
| Predictions | In-memory | 5 seconds (1.2s realtime) |
| News sentiment | In-memory | 45-180 seconds |
| Models | In-memory | Until retrain |

### Database Optimization

- WAL mode for concurrent reads
- Indexed queries on symbol/timestamp
- Periodic vacuum and cleanup
- Session cache compaction

---

## Monitoring

### Prometheus Metrics (`utils/metrics.py`)

- Counters, gauges, histograms
- Latency tracking
- Error rate monitoring
- Resource utilization

### Health Checks

```bash
# Basic health check
python main.py --health

# Full diagnostics
python main.py --doctor

# Strict mode (fails on any issue)
python main.py --doctor --doctor-strict
```

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

1. Create model class in `models/networks.py`
2. Implement `train()` and `predict()`
3. Register in `models/ensemble.py`

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| SQLite | Simple, embedded, no external dependencies |
| PyQt6 | Mature desktop framework, rich widgets, high-performance charts |
| PyTorch | Flexible ML framework, GPU acceleration, active community |
| Event-driven | Loose coupling, extensibility, natural fit for trading |
| Single-node | Personal/small-team use case, simple deployment |
| Modern architectures | Informer/TFT/N-BEATS/TSMixer outperform legacy LSTM/GRU/TCN |

---

## Future Enhancements

- [ ] Enhanced NLP for Chinese policy documents
- [x] Real-time news streaming (WebSocket service + desktop realtime panel)
- [ ] Multi-modal sentiment (text + social)
- [ ] Cloud backup integration
- [ ] Model explainability dashboard UI (SHAP/LIME backend already implemented)
- [ ] 2FA authentication
- [ ] Distributed training support (Ray, DeepSpeed)
