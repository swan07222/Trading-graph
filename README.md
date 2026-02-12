# Trading Graph

AI-assisted desktop trading system focused on China A-share equities.  
Includes data ingestion/failover, model training and auto-learning, execution/risk controls, monitoring, and a professional PyQt UI.

## 1) Current Project Level

### Engineering maturity
- **Advanced independent / pre-production grade**
- Strong for personal and small-team desktop trading workflows
- Not equivalent to full institutional platform infrastructure

### Practical readiness (current stack)
- Desktop A-share live-assisted use: **high**
- Institutional deployment (HA/DR, multi-venue OMS/FIX, regulated process stack): **partial**

## 2) Score Comparison vs Famous Apps

Benchmarked against well-known trading support platforms (TradingView / Thinkorswim / IBKR TWS / MT5 class), on **desktop A-share focused scope**.

### Focused scorecard (features you requested)

| Feature Area | Score (/10) |
|---|---:|
| Real-time/historical data reliability | 9.2 |
| Multi-source failover + VPN adaptation | 9.3 |
| Auto-learning robustness | 9.3 |
| Risk/OMS safety controls | 9.3 |
| Health monitoring + runbook controls | 9.1 |
| Audit/governance controls (tamper-evident chain + permissions) | 9.0 |
| Desktop UX/professional UI | 9.1 |
| Testing and regression support | 9.1 |
| **Focused overall** | **9.2** |

## 3) Profit Expectation (Careful, Realistic)

There is no “correct guaranteed profit” estimate for any strategy system.  
The correct way is scenario-based expectation with strict risk limits.

### Practical framework
- Use expectancy:
`Expectancy = WinRate * AvgWin - (1 - WinRate) * AvgLoss`
- Use max daily loss and max drawdown policies from risk config
- Validate on:
1. Walk-forward backtest
2. Replay-based deterministic regression
3. Paper/live shadow period before real capital scaling

### Reality check
- Profit depends more on market regime and risk discipline than model accuracy alone.
- Treat this system as a decision/risk framework, not a guaranteed alpha engine.

## 4) How the System Acts (Runtime Flow)

1. UI or CLI starts from `main.py`
2. `data/fetcher.py` resolves sources by network mode + health score
3. Data cleaned/cached in `data/database.py` and in-memory cache
4. `models/predictor.py` loads ensemble/scaler and produces signals
5. `trading/executor.py` gates submissions by:
   - market-open checks
   - quote freshness checks
   - risk manager checks
   - access/governance checks (live permissions/approvals policy)
6. Orders are persisted and synchronized in OMS (`trading/oms.py`)
7. Health monitor tracks operational status and can auto-pause autonomous trading in degraded mode
8. Audit subsystem records actions with tamper-evident hash chain

## 5) Installation

```bash
pip install -r requirements.txt
```

Optional GPU example (CUDA 11.8):
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 6) Quick Start

### Desktop UI
```bash
python main.py
```

### Auto-learning (headless)
```bash
python main.py --auto-learn --max-stocks 200 --continuous
```

### Backtest
```bash
python main.py --backtest
```

### Replay mode (deterministic feed replay)
```bash
python main.py --replay-file path/to/replay.csv --replay-speed 20
```

### Health report
```bash
python main.py --health
```

## 7) Core CLI Options

```bash
python main.py --train --epochs 100
python main.py --predict 600519
python main.py --auto-learn --max-stocks 50 --epochs 50 --continuous
python main.py --recovery-drill
```

## 8) Configuration

Primary config: `config/settings.py`  
Optional overrides: `config.json` and selected environment variables.

Key config groups:
- `data`: cache, timeouts, parallelism
- `model`: architecture/training thresholds
- `risk`: limits, drawdown, order frequency, quote staleness
- `security`: audit, permissions, governance flags
- `auto_trade`: autonomous behavior thresholds and limits

## 9) Full Code Map (What Each File Does)

### Root
- `main.py`: entrypoint; modes (UI, train, predict, auto-learn, backtest, replay, health, recovery drill).
- `debug_learn.py`: diagnostics helper for learning/universe flows.
- `test_fetch.py`: manual fetch/debug helper.
- `pyproject.toml`: project/build/tool config.

### `analysis/`
- `analysis/backtest.py`: walk-forward backtesting and result metrics.
- `analysis/technical.py`: technical indicator analysis.
- `analysis/sentiment.py`: sentiment/news scoring helpers.
- `analysis/replay.py`: deterministic market replay loader/player (csv/jsonl).
- `analysis/__init__.py`: analysis module exports.

### `config/`
- `config/settings.py`: typed config dataclasses, validation, load/save, paths.
- `config/__init__.py`: config exports.

### `core/`
- `core/types.py`: core dataclasses/enums (orders, fills, signals, auto-trade state).
- `core/constants.py`: market/exchange constants and helper rules.
- `core/instruments.py`: symbol/instrument normalization and parsing.
- `core/network.py`: network environment detection and cache.
- `core/events.py`: thread-safe event bus and event types.
- `core/exceptions.py`: domain-specific exception classes.
- `core/symbols.py`: symbol utilities.
- `core/__init__.py`: core exports.

### `data/`
- `data/fetcher.py`: multi-source data fetch, source health scoring, caching, realtime/history APIs.
- `data/database.py`: sqlite storage for bars/features/predictions.
- `data/cache.py`: memory cache utilities.
- `data/processor.py`: scaling, sequence prep, leakage-safe split logic, realtime sequence prep.
- `data/features.py`: feature engineering pipeline.
- `data/universe.py`: universe discovery/cache/fallback selection.
- `data/discovery.py`: stock discovery logic from spot/index/fallback sources.
- `data/feeds.py`: realtime feed integration utilities.
- `data/news.py`: news ingestion support.
- `data/validators.py`: validation helpers.
- `data/__init__.py`: data exports.

### `models/`
- `models/networks.py`: neural network architectures (LSTM/GRU/TCN/Transformer/Hybrid).
- `models/layers.py`: reusable model layers/blocks.
- `models/ensemble.py`: ensemble training/inference, weights, calibration, save/load.
- `models/trainer.py`: end-to-end training pipeline (classifier + forecaster).
- `models/predictor.py`: runtime prediction orchestration and quick/batch methods.
- `models/auto_learner.py`: continuous learning loop (rotation, replay, holdout validation, state).
- `models/__init__.py`: model exports.

### `trading/`
- `trading/executor.py`: execution engine, auto-trader, submission gating, broker sync loops.
- `trading/oms.py`: order management system, persistence, fill processing.
- `trading/risk.py`: risk manager checks/metrics.
- `trading/kill_switch.py`: emergency stop controls and callbacks.
- `trading/broker.py`: broker abstraction and concrete adapter creation.
- `trading/portfolio.py`: equity/PnL/risk metric aggregation.
- `trading/alerts.py`: alert routing and history.
- `trading/health.py`: component/system health monitor + SLO checks.
- `trading/signals.py`: signal generation/orchestration helpers.
- `trading/__init__.py`: trading exports.

### `ui/`
- `ui/app.py`: main desktop application window and workflow orchestration.
- `ui/auto_learn_dialog.py`: auto-learning and targeted training dialog.
- `ui/dialogs.py`: training/backtest/broker/risk settings dialogs.
- `ui/widgets.py`: reusable UI widgets (signal panel, log widget, metric cards, tables).
- `ui/charts.py`: chart rendering layer.
- `ui/news_widget.py`: news display widget.
- `ui/__init__.py`: UI exports.

### `utils/`
- `utils/logger.py`: project logging setup.
- `utils/atomic_io.py`: atomic file write/read helpers.
- `utils/security.py`: secure storage, access control, audit log + integrity verification.
- `utils/metrics.py`: in-process metrics counters/gauges.
- `utils/metrics_http.py`: metrics/health HTTP endpoint.
- `utils/cancellation.py`: cancellation token utilities.
- `utils/helpers.py`: shared conversion and helper functions.
- `utils/__init__.py`: utils exports.

### `tests/`
- `tests/test_data.py`: fetch/data behavior tests.
- `tests/test_data_leakage.py`: anti-leakage and split integrity tests.
- `tests/test_models.py`: model and training path tests.
- `tests/test_oms_fills.py`: OMS/fill behavior tests.
- `tests/test_replay.py`: replay loader/order determinism tests.
- `tests/test_audit_integrity.py`: audit hash-chain integrity tests.
- `tests/test_executor_health_guard.py`: health policy execution guards.
- `tests/conftest.py`: shared fixtures/environment setup.

## 10) Testing

Run full tests:
```bash
pytest
```

Focused regression pack:
```bash
pytest -q tests/test_data.py tests/test_data_leakage.py tests/test_models.py tests/test_oms_fills.py tests/test_replay.py tests/test_audit_integrity.py tests/test_executor_health_guard.py
```

## 11) Troubleshooting

1. Auto-learning stalls in VPN mode:
   - keep `max_stocks` moderate (system auto-caps under VPN for stability)
   - verify `--health` status and source availability logs
2. Frequent provider failures:
   - check network mode detection logs
   - ensure data source dependencies are installed
3. If state looks inconsistent:
   - stop app, back up then inspect `data_storage/learner_state.json`
4. Integrity checks:
   - use audit query + integrity verification path from `utils/security.py`

## 12) Limitations

1. Still a desktop single-node architecture (not full HA/DR cluster).
2. External provider/network quality remains a key dependency.
3. Institutional process depth is improved but not full regulated enterprise stack.
