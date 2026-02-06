# AI Stock Trading System (A-Share Focus) — Walk-Forward Backtest + Live/Paper Execution

A production-oriented Python trading system with:
- Feature engineering (strictly causal indicators)
- Supervised ML ensemble (LSTM / GRU / TCN / Transformer / Hybrid)
- Walk-forward backtesting (time-aligned multi-stock returns)
- Realistic A-share constraints (T+1, lot size, price limits)
- OMS with SQLite persistence (WAL), crash recovery, fill idempotency
- Risk manager + circuit breakers + kill switch
- Desktop GUI (PyQt6) for monitoring, analysis, and execution

> **Disclaimer:** This project is for research/education. Live trading involves risk. Use at your own responsibility.

---

## Features

### Research & Modeling
- Strictly causal features (no look-ahead)
- Labeling by configurable horizon and thresholds
- RobustScaler fitted on training data only
- Ensemble model with weighted logits + temperature calibration
- Leakage tests (scaler/labels/sequence consistency)

### Backtesting
- Walk-forward folds with embargo
- Time-aligned daily returns across multiple symbols
- Buy-and-hold benchmark (compounded)
- Trade counting by entries (not bars)
- Commission, stamp tax, slippage modeling
- A-share price limit checks

### Trading (Paper/Live)
- Broker abstraction (Simulator / easytrader-based brokers)
- OMS as single source of truth (SQLite)
- Fill sync loop (broker.get_fills()) — no fabricated fills
- Risk manager with:
  - position limits
  - daily loss & drawdown limits
  - quote staleness checks
  - rate limits
- Kill switch + circuit breakers with persisted state

### Observability & Compliance
- Structured logging (UTF-8 safe)
- Audit logs (JSONL gzip)
- Alerts: log/desktop/email/webhook (configurable)
- Metrics registry (counters/gauges/histograms)

---

## Project Structure

- `analysis/`  
  - `technical.py` indicators + signal scoring  
  - `sentiment.py` news scraping + keyword sentiment  
  - `backtest.py` walk-forward backtester

- `data/`  
  - `fetcher.py` multi-source OHLCV + realtime quotes  
  - `features.py` causal feature engine  
  - `processor.py` labeling, scaling, sequences  
  - `feeds.py` realtime polling feed + bar aggregation  
  - `database.py` local market DB  
  - `cache.py` tiered cache

- `models/`  
  - `ensemble.py` ensemble training + calibrated inference  
  - `networks.py` model architectures  
  - `trainer.py` full training pipeline  
  - `predictor.py` inference + trade levels + position sizing

- `trading/`  
  - `oms.py` persistent OMS  
  - `executor.py` execution engine + sync loops  
  - `risk.py` risk manager  
  - `kill_switch.py` kill switch  
  - `alerts.py` alert manager  
  - `health.py` health monitor

- `ui/`  
  - `app.py` desktop GUI  
  - widgets/charts/dialogs

---

## Requirements

Python 3.9+ recommended.

Core dependencies:
- numpy, pandas, scikit-learn
- torch
- ta
- PyQt6
- psutil
- requests, bs4

Optional:
- akshare (A-share data)
- yfinance (fallback data)
- cryptography (secure storage)
- easytrader (live broker integration)

---

## Installation

```bash
git clone <your-repo>
cd <your-repo>
python -m venv .venv
# activate venv...
pip install -r requirements.txt