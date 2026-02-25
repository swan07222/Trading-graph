# Trading Graph 2.0

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

AI-powered stock analysis platform for China A-shares. Built for analysts and researchers.

---

## What's New in 2.0

| Area | Improvements |
|------|--------------|
| **Performance** | Async I/O, Redis caching, PostgreSQL support |
| **ML Models** | Informer, TFT, N-BEATS, TSMixer architectures |
| **Developer UX** | Full type hints, FastAPI dashboard, WebSocket streaming |
| **China Support** | 5+ local data providers, proxy support, optimized timeouts |

---

## Features

### 🔍 Stock Discovery
Search and discover all China A-share stocks (SSE, SZSE, BSE) with smart scoring based on market cap, volume, and index membership.

### 📰 News Collection
VPN-aware routing with automatic source selection:
- **China Direct**: Jin10, EastMoney, Sina, Xueqiu, Caixin, CSRC
- **International**: Reuters, Bloomberg, Yahoo Finance, MarketWatch, CNBC

### 🧠 Sentiment Analysis
Multi-factor scoring with entity extraction, policy detection, and trading signal generation.

### 📊 Model Training
- Traditional: LSTM, GRU, TCN, Transformer, Hybrid ensemble
- News-based: Transformer encoder + sentiment fusion + price patterns
- Explainability with SHAP values and uncertainty estimation

### 📈 Real-Time Charts
Live candlestick charts with AI predictions, uncertainty bands, and technical indicators (SMA, EMA, Bollinger, VWAP).

---

## Quick Start

### Installation

```bash
# Core dependencies
pip install -r requirements.txt

# With web dashboard (recommended)
pip install -r requirements-web.txt

# Full stack (includes NLP)
pip install -r requirements-all.txt
```

### Run

```bash
# Desktop UI
python main.py

# Web Dashboard
python -m ui.web_dashboard
# Access: http://localhost:8000
```

### Optional: Redis & PostgreSQL

```bash
# Redis (caching)
docker run -d -p 6379:6379 redis:latest

# PostgreSQL (production storage)
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:15
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/trading"
```

---

## Commands

### News & Sentiment
```bash
python main.py --collect-news                    # Collect news
python main.py --analyze-sentiment 600519        # Analyze specific stock
```

### Training
```bash
python main.py --train --epochs 100              # Train all models
python main.py --train-news --epochs 50          # Train news model
python main.py --train-stock 600519 --epochs 50  # Train on specific stock
```

### Prediction & Analysis
```bash
python main.py --predict 600519                  # Predict movement
python main.py --backtest                        # Run backtest
python main.py --auto-learn --max-stocks 50      # Auto-learning
```

### Diagnostics
```bash
python main.py --health                          # System health
python main.py --doctor                          # Full diagnostics
python -m utils.china_diagnostics                # Network diagnostics
```

---

## Network Configuration

### China Direct Mode
```bash
export TRADING_CHINA_DIRECT=1
```

### VPN Mode (International)
```bash
export TRADING_VPN=1
export TRADING_PROXY_URL=http://127.0.0.1:7890
```

---

## Project Structure

```
trading-graph/
├── data/          # Data fetching, news collection, sentiment
├── models/        # ML models, training, prediction
├── ui/            # PyQt6 desktop & web dashboard
├── analysis/      # Backtest, replay, strategy engine
├── config/        # Settings, environment config
├── core/          # Types, events, constants
├── utils/         # Utilities, security, metrics
└── tests/         # Test suite
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard home |
| `GET /api/stocks` | List all stocks |
| `GET /api/stocks/{symbol}` | Stock details |
| `POST /api/predict` | Generate prediction |
| `GET /api/news` | Recent news |
| `GET /api/sentiment/{symbol}` | Sentiment analysis |
| `GET /docs` | Swagger API docs |

---

## Testing & Linting

```bash
# Run tests
pytest -q

# Lint code
ruff check .

# Type checking
mypy trading_graph/
```

---

## Scope

**Desktop-first, single-node deployment.** Designed for personal and small-team research workflows.

> ⚠️ **Note**: This is an **analysis and prediction framework only**. Trading execution components (portfolio management, risk management, OMS, broker integration) have been removed.

---

## Safety Notice

This system provides decision support for research purposes only. It is not a guaranteed-profit system. Use at your own risk.

---

## License

MIT License - See LICENSE file for details.
