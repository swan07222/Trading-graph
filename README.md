# AI Stock Trading System (A-Share Focus)

A modular AI-assisted stock trading system with:
- Feature engineering (causal, no look-ahead bias)
- Deep learning ensemble (LSTM / GRU / TCN / Transformer / Hybrid)
- Walk-forward backtesting with embargo
- Real-time monitoring (polling feed) + GUI (PyQt6)
- Production-style trading stack: OMS + Risk Manager + Kill Switch + Alerts
- SQLite persistence + audit logging

> Disclaimer: This project is for research/education. Trading involves significant risk. No guarantee of profitability.

---

## Features

### Data Layer
- Multi-source historical + realtime fetching with fallback
  - AkShare (primary for China A-shares)
  - Yahoo Finance (fallback)
- Tiered caching (L1 memory + L2 disk + L3 compressed disk)
- Local market database (SQLite) for bars, features, predictions

### Feature Engineering
- Technical indicators built using only past data (strictly causal)
- RobustScaler normalization fitted **only on training data**
- Causal rolling features (no centered windows)

### Models (AI)
- Ensemble model with weighted logits aggregation
- Probability calibration via temperature scaling
- Batched predictions (recommended)
- Predicts 3 classes: DOWN / NEUTRAL / UP over `PREDICTION_HORIZON`

### Backtesting
- Walk-forward backtesting with:
  - Train/test folds
  - Embargo gap to reduce leakage
  - Time-aligned daily returns across stocks
  - Benchmark buy-and-hold compounded returns
  - Trade counting by entries (not bars)
- A-share constraints supported in simulation/backtest (limit rules partially implemented)

### Trading Stack
- Broker interface:
  - Paper trading simulator (slippage/commission, T+1)
  - easytrader-based connectors (TongHuaShun and universal broker modes)
- OMS (Order Management System):
  - SQLite WAL persistence
  - Order state machine + crash recovery
  - Fill tracking + reconciliation
  - T+1 pending settlement tracking
- Risk Manager:
  - Daily loss limit, drawdown limit
  - Concentration / exposure checks
  - Quote staleness checks
  - VaR / Expected Shortfall (historical simulation)
- Kill Switch + Circuit Breakers
- Alerts: log/desktop/email/webhook (configurable)

### UI
- Professional PyQt6 desktop app:
  - Watchlist monitoring
  - Signal panel and chart widgets
  - Portfolio and positions table
  - Training / backtest dialogs

---

## Repository Layout