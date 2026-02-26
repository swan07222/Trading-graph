# Trading Graph 2.0

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **AI-powered stock analysis platform for China A-shares.** Built for analysts, researchers, and quantitative traders.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Commands Reference](#commands-reference)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Testing & Quality](#testing--quality)
- [Important Notices](#important-notices)

---

## Overview

Trading Graph is a comprehensive AI-driven analysis framework for China's A-share market (SSE, SZSE, BSE). It combines multi-source data collection, LLM-powered sentiment analysis, and modern deep learning models to provide actionable trading insights.

**What's New in 2.0:**

| Area | Improvements |
|------|--------------|
| **Performance** | Async I/O, Redis caching, PostgreSQL support |
| **ML Models** | Informer, TFT, N-BEATS, TSMixer architectures |
| **Developer UX** | Full type hints, FastAPI metrics endpoint, WebSocket streaming |
| **China Support** | VPN-aware routing, 5+ local data providers, optimized timeouts |
| **Monitoring** | Prometheus-compatible metrics, health dashboards |

---

## Key Features

### 🔍 Stock Discovery

Comprehensive screening of all China A-share stocks with intelligent scoring based on:
- Market capitalization
- Trading volume
- Index membership (CSI 300, SSE 50, etc.)
- Liquidity metrics

### 📰 News Collection

VPN-aware news aggregation with automatic source selection:

| Region | Sources |
|--------|---------|
| **China Direct** | EastMoney, Sina Finance, Xueqiu, Caixin, CSRC |
| **International** | Reuters, Bloomberg, Yahoo Finance, MarketWatch, CNBC |

Features:
- Network environment auto-detection (VPN vs China direct)
- Multi-source aggregation and deduplication
- Content normalization
- Health scoring and automatic failover

### 🧠 Sentiment Analysis

Multi-factor sentiment engine with:
- General sentiment scoring (-1.0 to 1.0)
- Policy impact assessment
- Market sentiment detection
- Entity extraction (companies, policies, people, industries)
- LLM-powered sentiment analysis using transformer models
- Trading signal generation

### 📊 Model Training

**Hybrid Model Architecture:**

The system uses a hybrid approach combining traditional time series models with LLM-based sentiment analysis:

| Model Type | Architecture | Storage | Purpose |
|------------|--------------|---------|---------|
| **Guessing Model (GM)** | Informer + TFT + N-BEATS + TSMixer ensemble | `models_saved/GM/` | Price movement prediction |
| **LLM Sentiment** | Transformer-based bilingual analyzer | `models_saved/LLM/` | News/policy sentiment analysis |

**Guessing Model (GM):**
- Ensemble of modern architectures (Informer, TFT, N-BEATS, TSMixer)
- Weighted voting based on validation performance
- Multi-interval support (1m, 5m, 15m, 30m, 1h, 1d)
- Uncertainty quantification with confidence bands

**LLM Sentiment Model:**
- Bilingual (Chinese/English) sentiment analysis
- Auto-training on collected news data
- Policy impact detection
- Entity extraction (companies, policies, people)
- Hybrid approach: transformer + sklearn MLP fallback

**Explainability:**
- SHAP values for GM predictions
- Feature importance analysis
- Confidence calibration
- Walk-forward validation

### 📈 Real-Time Charts

Live candlestick visualization with:
- AI predictions overlay
- Uncertainty bands
- Technical indicators (SMA, EMA, Bollinger, VWAP)
- Real-time data streaming via WebSocket
- PyQt6 high-performance rendering

### 🛡️ Monitoring

Built-in observability features:
- Health status dashboard
- Prometheus-compatible metrics export
- Institutional readiness checks

---

## Quick Start

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Installation

```bash
# Core dependencies (minimum installation)
pip install -r requirements.txt

# With web dashboard
pip install -r requirements-web.txt

# Full stack (includes NLP, GUI, and all features)
pip install -r requirements-all.txt

# Development setup
pip install -r requirements-dev.txt
```

### Run the Application

```bash
# Desktop UI (PyQt6)
python main.py

# CLI mode
python main.py --cli

# Web Dashboard (FastAPI)
python -m ui.web_dashboard
# Access at: http://localhost:8000
```

### Optional: Enhanced Infrastructure

```bash
# Redis (caching layer)
docker run -d -p 6379:6379 redis:latest

# PostgreSQL (production storage)
docker run -d -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  postgres:15
```

---

## Commands Reference

### News & Sentiment

```bash
# Collect news from all sources
python main.py --collect-news

# Analyze sentiment for a specific stock (by code)
python main.py --analyze-sentiment 600519
```

### Model Training

```bash
# Train all models (Informer, TFT, N-BEATS, TSMixer)
python main.py --train --epochs 100

# Train news-based model only
python main.py --train-news --epochs 50

# Auto-learning across multiple stocks
python main.py --auto-learn --max-stocks 50 --continuous
```

### Prediction & Analysis

```bash
# Predict price movement for a stock
python main.py --predict 600519

# Run backtest
python main.py --backtest

# Optimize backtest parameters
python main.py --backtest-optimize
```

### Diagnostics & Health

```bash
# System health check
python main.py --health

# Full system diagnostics
python main.py --doctor

# Strict mode (fails on any issue)
python main.py --doctor --doctor-strict

# Network diagnostics (China-specific)
python -m utils.china_diagnostics
```

### Market Replay

```bash
# Replay historical market data
python main.py --replay-file data/replay_2024.csv --replay-speed 20
```

---

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# ================================================================
# TRADING MODE
# ================================================================
TRADING_MODE=simulation              # simulation | live
TRADING_INITIAL_CAPITAL=100000

# ================================================================
# RISK LIMITS
# ================================================================
TRADING_MAX_POSITION_PCT=15          # Max position % of portfolio
TRADING_MAX_DAILY_LOSS_PCT=3         # Max daily loss %
TRADING_MAX_DRAWDOWN_PCT=15          # Circuit breaker threshold
TRADING_MAX_POSITIONS=10             # Max concurrent positions

# ================================================================
# AUTO-TRADE SETTINGS
# ================================================================
TRADING_AUTO_TRADE_MODE=manual       # manual | semi_auto | auto
TRADING_MIN_CONFIDENCE=0.70          # Min confidence for auto-trades
TRADING_MAX_TRADES_PER_STOCK=3
TRADING_TRADE_COOLDOWN=300           # Seconds between trades

# ================================================================
# DATA SOURCES
# ================================================================
TRADING_PRIMARY_SOURCE=tencent       # tencent | akshare | sina | yahoo
TRADING_AKSHARE_ENABLED=1

# ================================================================
# CHINA NETWORK SETTINGS
# ================================================================
TRADING_VPN=0                        # Enable VPN mode
TRADING_CHINA_DIRECT=0               # Force China direct mode
# TRADING_PROXY_URL=http://127.0.0.1:7890
TRADING_CONNECTION_TIMEOUT=30

# ================================================================
# MODEL SETTINGS
# ================================================================
TRADING_MODEL_DIR=models_saved        # Base model directory
                                      # GM models: models_saved/GM/
                                      # LLM models: models_saved/LLM/
TRADING_DEFAULT_EPOCHS=100

# ================================================================
# UI SETTINGS
# ================================================================
TRADING_CHART_THEME=dark
TRADING_REALTIME_CHARTS=1
TRADING_CHART_UPDATE_INTERVAL=3000

# ================================================================
# LOGGING & SECURITY
# ================================================================
TRADING_LOG_LEVEL=INFO
TRADING_AUDIT_LOG=1

# ================================================================
# METRICS & MONITORING
# ================================================================
TRADING_METRICS_ENABLED=0
TRADING_METRICS_PORT=8000
TRADING_METRICS_HOST=127.0.0.1
```

### China Direct Mode

For users within mainland China:

```bash
export TRADING_CHINA_DIRECT=1
```

This enables:
- Direct access to Chinese data sources
- Optimized routing for mainland networks
- Reduced latency

### VPN Mode (International)

For users outside China or needing international sources:

```bash
export TRADING_VPN=1
export TRADING_PROXY_URL=http://127.0.0.1:7890
```

This enables:
- VPN-aware routing
- International source selection
- Automatic failover

---

## Project Structure

```
trading-graph/
├── analysis/          # Backtest, replay, strategy engine, technical analysis
├── config/            # Settings, environment configuration
├── core/              # Types, events, constants, symbols
├── data/              # Data fetching, news collection, sentiment analysis
├── models/            # ML models (Informer, TFT, N-BEATS, TSMixer), training, prediction
├── models_saved/      # Trained model artifacts
│   ├── GM/            # Guessing Model (price prediction ensemble)
│   └── LLM/           # LLM sentiment models (bilingual analyzer)
├── strategies/        # Trading strategies
├── trading/           # Execution engine (disabled in analysis-only mode)
├── ui/                # PyQt6 desktop & web dashboard
├── utils/             # Utilities, security, metrics
├── tests/             # Test suite
├── scripts/           # Utility scripts
├── docs/              # Documentation
└── main.py            # Entry point
```

---

## Testing & Quality

### Run Tests

```bash
# All tests with coverage
pytest

# Specific test file
pytest tests/test_predictor.py -v

# With HTML coverage report
pytest --cov=trading_graph --cov-report=html
```

### Code Quality

```bash
# Lint code
ruff check .

# Format code
ruff format .

# Type checking
mypy trading_graph/
```

### Coverage Requirements

| Module Type | Minimum Coverage |
|-------------|------------------|
| Overall | 40%+ |
| Critical (Data, Models) | 60%+ |

---

## API Endpoints

### Metrics Server (when enabled via `TRADING_METRICS_ENABLED=1`)

| Endpoint | Description |
|----------|-------------|
| `GET /metrics` | Prometheus-compatible metrics |
| `GET /healthz` | Health check |
| `GET /api/v1/dashboard` | Dashboard data |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ws/quotes` | Real-time quote streaming |
| `/ws/signals` | Trading signal updates |
| `/ws/health` | System health updates |

---

## Important Notices

### Scope

**Desktop-first, single-node deployment.** Designed for personal and small-team research workflows.

> ⚠️ **Analysis Framework Only**: This system provides **analysis and prediction capabilities only**. Trading execution components have been intentionally excluded from this distribution.

### Safety Notice

> This system provides **decision support for research purposes only**. It is not a guaranteed-profit system and should not be considered financial advice. All trading decisions are made at your own risk.

### Limitations

- **Single-node architecture**: Not designed for distributed deployment
- **Desktop UI**: Requires local installation (PyQt6)
- **Data sources**: Relies on third-party data providers (not direct exchange feeds)
- **Model confidence**: Predictions are probabilistic, not deterministic

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture overview |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [docs/OPERATIONS_PLAYBOOK.md](docs/OPERATIONS_PLAYBOOK.md) | Deployment and operations guide |
| [docs/adr/](docs/adr/) | Architecture decision records |

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Support

- **Issues**: Open a GitHub issue for bugs
- **Discussions**: GitHub Discussions for questions
- **Documentation**: See `docs/` directory for detailed guides
