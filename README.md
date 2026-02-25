# Trading Graph

Desktop AI trading analysis system for China A-shares with:
- **News and Policy Analysis**: Multi-source news collection with VPN-aware routing
- **Sentiment Analysis**: AI-powered sentiment analysis for market prediction
- **China network optimization** (auto failover, proxy support, DNS optimization)
- **Enhanced sentiment analysis** (Jin10, EastMoney, Sina, Xueqiu, Reuters, Yahoo Finance)
- Model training and prediction with **explainability**
- **News-based model training** for understanding policy and market sentiment
- PyQt real-time charting and operations UI

## Key Features

### 1. Stock Search & Discovery
- **Search all stocks**: Discovers all China A-share stocks (SSE, SZSE, BSE)
- Multiple discovery sources: AkShare, Tencent, CSI index constituents
- Smart scoring based on market cap, volume, and index membership
- Supports 600/601/603/605, 688, 000/001/002/003, 300/301, 83/87/43 prefixes

### 2. News & Policy Data Collection
- **VPN-aware routing**: Auto-detects network environment
  - **VPN Off (China Direct)**: Jin10, EastMoney, Sina Finance, Xueqiu, Caixin, CSRC
  - **VPN On (International)**: Reuters, Bloomberg, Yahoo Finance, MarketWatch, CNBC
- **Policy keyword detection**: Automatically identifies policy/regulatory news
- **Multi-language support**: Chinese and English news processing
- **Real-time collection**: Continuous news monitoring with configurable intervals

### 3. Sentiment Analysis
- **Multi-factor scoring**: General sentiment, policy impact, market sentiment
- **Entity extraction**: Identifies companies, policies, and key figures
- **Trading signals**: Converts sentiment to actionable trading signals
- **Temporal analysis**: Tracks sentiment trends over time

### 4. News-Based Model Training
- **Multi-modal architecture**: Combines news embeddings, sentiment features, and price data
- **Transformer encoder**: Processes news text for semantic understanding
- **LSTM price encoder**: Captures temporal price patterns
- **Fusion model**: Integrates news and price signals for prediction
- **Train command**: `python main.py --train-news --epochs 50`

### 5. Model Training & Explainability
- **Train on all stocks**: `python main.py --train --epochs 100`
- **Train on specific stock**: `python main.py --train-stock 600519 --epochs 100`
- Auto-learning with continuous improvement
- Ensemble models (LSTM, GRU, TCN, Transformer, Hybrid)
- **Explainability** for model predictions
- **Uncertainty estimation** with Monte Carlo Dropout
- **Conformal Prediction** for valid confidence intervals

### 6. China Network Support
- **Fully optimized for mainland China network conditions**
- ✅ 5+ Chinese financial data providers with auto-failover
- ✅ Proxy support (HTTP/SOCKS5) for VPN users
- ✅ China-optimized DNS resolution (114DNS, AliDNS, DNSPod)
- ✅ Extended timeouts for Great Firewall conditions
- ✅ Network diagnostics: `python -m utils.china_diagnostics`

### 7. Real-Time Charting
- **Live candlestick updates** with real-time price feeds
- **AI prediction overlay** (dashed cyan line)
- **Uncertainty bands** (dotted yellow lines)
- Technical indicators (SMA, EMA, Bollinger Bands, VWAP)
- Interactive hover tooltips with OHLCV data

## Scope

This project is desktop-first and single-node. It is suitable for personal and small-team workflows, not full institutional deployment.

**Note**: Trading execution components (portfolio management, risk management, OMS, broker integration, auto-trading) have been removed. This system focuses on **analysis and prediction** only.

Tooling is Python-only (`pyproject.toml` + `pip` requirements); no Node/NPM step is required.

## Quick Start

Install:

```bash
pip install -r requirements.txt
```

Desktop/UI + ML + CN/VPN data providers:

```bash
pip install -r requirements-desktop.txt
```

Full optional stack (desktop + NLP extras):

```bash
pip install -r requirements-all.txt
```

Run UI:

```bash
python main.py
```

## Useful Commands

### News Collection
```bash
# Collect news from web sources
python main.py --collect-news

# Analyze sentiment for a specific stock
python main.py --analyze-sentiment 600519
```

### Model Training
```bash
# Train news-based model
python main.py --train-news --epochs 50

# Train traditional model
python main.py --train --epochs 100

# Auto-learn
python main.py --auto-learn --max-stocks 50
```

### Prediction
```bash
# Predict stock movement
python main.py --predict 600519
```

### Backtest
```bash
# Run backtest
python main.py --backtest

# Optimize backtest parameters
python main.py --backtest-optimize
```

### Diagnostics
```bash
# System health
python main.py --health

# System diagnostics
python main.py --doctor
```

## Network Configuration

### China Direct Mode (VPN Off)
```bash
export TRADING_CHINA_DIRECT=1
python main.py --collect-news
```

### VPN Mode (International Sources)
```bash
export TRADING_VPN=1
export TRADING_PROXY_URL=http://127.0.0.1:7890
python main.py --collect-news
```

## Data Sources

### Chinese Sources (VPN Off)
- Jin10 (财经快讯)
- EastMoney (东方财富网)
- Sina Finance
- Xueqiu (雪球)
- Caixin (财新)
- CSRC (中国证监会)

### International Sources (VPN On)
- Reuters
- Bloomberg
- Yahoo Finance
- MarketWatch
- CNBC

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UI Layer (PyQt6)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Main Window │  │Chart Widget │  │ Sentiment Analysis  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │     App     │  │ Background  │  │  News Collection    │  │
│  │  Controller │  │    Tasks    │  │  Sentiment Analysis │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   Data Layer     │ │   Model Layer    │ │   Sentiment      │
│ ┌──────────────┐ │ │ ┌──────────────┐ │ │ ┌──────────────┐ │
│ │ DataFetcher  │ │ │ │ Trainer      │ │ │ │ Analyzer     │ │
│ │ NewsCollector│ │ │ │ Predictor    │ │ │ │ Entity Extract││
│ │ Cache        │ │ │ │ News Trainer │ │ │ │ Policy Detect│ │
│ │ Database     │ │ │ │ Ensemble     │ │ │ │ Signal Gen   │ │
│ └──────────────┘ │ │ └──────────────┘ │ │ └──────────────┘ │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

## Validation

Run tests:

```bash
pytest -q
```

Lint:

```bash
ruff check .
```

## Main Directories

- `data/`: Data fetch, news collection, sentiment analysis
- `models/`: Model training/prediction (traditional + news-based)
- `ui/`: PyQt application and chart rendering
- `analysis/`: Backtest, replay modules
- `config/`: Settings, runtime environment
- `core/`: Core types, events, constants
- `utils/`: Utilities, security, metrics

## Safety Note

This is a decision-support framework, not a guaranteed-profit system. Use for research and analysis purposes only.
