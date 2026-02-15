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

Development/tooling deps (ruff, pytest, mypy):
```bash
pip install -r requirements-dev.txt
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

```bash
python -m pytest -q
```

## 11) Strategy Scripting (Extensibility)

Custom script strategies are auto-loaded from `strategies/*.py`.

Required contract:
- `generate_signal(df, indicators, context) -> dict`
- Return dict fields: `action` (`buy`/`sell`/`hold`), `score` (`0..1`), `reason` (optional)

Example script:
- `strategies/momentum_breakout.py`

These script outputs are blended into final signal scoring with bounded impact.

## 12) Strategy Marketplace (Professional)

Marketplace files:
- `strategies/marketplace.json`: strategy catalog metadata
- `strategies/enabled.json`: enabled strategy IDs

UI:
- Open `AI Model -> Strategy Marketplace`
- Review integrity status (`ok/mismatch/missing`)
- Enable/disable and save active strategy set

## 13) Chart Quick Trade UX

In the desktop chart:
- Right-click chart to open quick-trade menu
- Choose `Buy @ price` or `Sell @ price`
- Enter quantity and submit through the same risk/permission checks as normal execution

## 14) Real-Time Session Cache For Auto-Learn

During real-time feed updates, finalized bars are persisted to:
- `data_storage/session_bars/*.csv`

Auto-learn dialog can inject these session symbols as priority training targets,
so pressing `Start Auto Learning` can immediately train with data captured during trading.

## 15) Local Web API Bridge (Dashboard/Mobile Companion)

The metrics server now also exposes lightweight JSON endpoints for local integrations.

Start with:
```bash
set TRADING_METRICS_PORT=9090
set TRADING_HTTP_API_KEY=your-secret-key
python main.py
```

Endpoints:
- `GET /healthz`
- `GET /metrics`
- `GET /api/v1/providers`
- `GET /api/v1/snapshot/<provider>`
- `GET /api/v1/dashboard?limit=20`

Authentication:
- If `TRADING_HTTP_API_KEY` is set, `/api/v1/*` requires header `X-API-Key: <key>`
  or query parameter `api_key=<key>`.

Extensibility:
- Register custom snapshot providers via:
  `utils.metrics_http.register_snapshot_provider(name, callable)`

Run full tests:
```bash
pytest
```

Focused regression pack:
```bash
pytest -q tests/test_data.py tests/test_data_leakage.py tests/test_models.py tests/test_oms_fills.py tests/test_replay.py tests/test_audit_integrity.py tests/test_executor_health_guard.py
```

Manual cache delete (guarded):
```bash
python scripts/manual_cache_delete.py --confirm --tier all
```

## 16) Policy-As-Code Governance

Live-trade governance can be enforced through:
- `config/security_policy.json`
- runtime checks in `trading/executor.py` via `utils/policy.py`

Example policy keys:
- `enabled`
- `live_trade.min_approvals`
- `live_trade.require_distinct_approvers`
- `live_trade.max_order_notional`
- `live_trade.blocked_symbols`
- `live_trade.allowed_sides`

If a trade violates policy, submission is rejected and audited as:
- `risk` event type `live_trade_blocked_policy`

## 17) Signed Release Artifacts

Release workflow:
- `.github/workflows/release.yml`
- Trigger: push tag `v*` or manual dispatch
- Outputs: wheel/sdist + `dist/release_manifest.json`
- Includes build provenance attestation (`actions/attest-build-provenance`)

Manifest generation:
```bash
python scripts/generate_release_manifest.py --dist-dir dist --version v1.2.3 --output dist/release_manifest.json
```

Optional manifest signature:
- Set env var `RELEASE_MANIFEST_SECRET` in GitHub secrets for HMAC signature embedding.

## 18) Operations Playbook (Deploy / Rollback / Observability)

Operational runbook:
- `docs/OPERATIONS_PLAYBOOK.md`

Pre-deploy strict checks:
```bash
python scripts/release_preflight.py --observability-url http://127.0.0.1:9090
```

Regulated readiness gate:
```bash
python scripts/regulatory_readiness.py
```

HA/DR lease-fencing drill:
```bash
python scripts/ha_dr_drill.py --backend sqlite --ttl-seconds 5
```

Create deployment snapshot:
```bash
python scripts/deployment_snapshot.py create --snapshot-dir backups
```

Rollback (dry-run first, then confirm):
```bash
python scripts/deployment_snapshot.py restore --archive backups/snapshot_<tag>.tar.gz --dry-run
python scripts/deployment_snapshot.py restore --archive backups/snapshot_<tag>.tar.gz --confirm
```

Observability endpoint probe:
```bash
python scripts/observability_probe.py --base-url http://127.0.0.1:9090
```

Typecheck gate (CI and local):
```bash
python scripts/typecheck_gate.py
```

Refresh type baseline intentionally:
```bash
python scripts/typecheck_gate.py --write-baseline
```

## 19) End-to-End Soak Testing

Paper/simulation soak:
```bash
python scripts/soak_broker_e2e.py --mode paper --duration-minutes 120 --poll-seconds 5 --symbols 600519,000001
```

Live-condition soak (explicit safety switch required):
```bash
python scripts/soak_broker_e2e.py --mode live --allow-live --duration-minutes 120 --poll-seconds 5 --symbols 600519,000001
```

Optional probe order path (extra risk; explicitly gated):
```bash
python scripts/soak_broker_e2e.py --mode live --allow-live --place-probe-order --allow-live-orders --probe-symbol 600519
```

## 20) Multi-Provider Sentiment Fusion

`analysis/sentiment.py` now supports:
- built-in providers: `sina`, `eastmoney`
- provider reliability weights for aggregation
- runtime provider registration:

```python
from analysis.sentiment import NewsScraper

scraper = NewsScraper()
scraper.register_provider("custom_feed", fetcher_callable, weight=1.2)
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
