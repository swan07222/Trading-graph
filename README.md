# Trading Graph

AI-assisted desktop trading system focused on China A-share equities.

It combines:
- multi-source market data ingestion with failover
- model training and auto-learning
- execution and risk controls
- health monitoring and audit trails
- a PyQt desktop UI

## Project Status

- Maturity: advanced independent / pre-production grade
- Best fit: personal and small-team desktop trading workflows
- Institutional stack parity: partial (not full enterprise HA/DR + regulated ops stack)

## Key Features

- Unified desktop workflow: data, models, risk, execution, and monitoring in one app.
- Data resilience: source health scoring, failover routing, VPN-aware behavior.
- Risk-first execution: quote staleness checks, pre-trade controls, kill switch, policy checks.
- Replay and backtest support for safer strategy validation.
- Auto-learning pipeline with session cache integration.
- Tamper-evident audit chain and governance hooks for live controls.

## Runtime Flow

1. Entry via `main.py` (UI or CLI mode)
2. `data/fetcher.py` selects sources by network mode + source health
3. Data is cleaned and stored in cache/SQLite (`data/database.py`)
4. `models/predictor.py` generates model outputs/signals
5. `trading/executor.py` enforces market/risk/governance gates
6. `trading/oms.py` persists and reconciles order lifecycle
7. `trading/health.py` tracks runtime health and degraded-mode actions
8. `utils/security.py` records audited security-sensitive events

## Installation

Core:

```bash
pip install -r requirements.txt
```

Development tools:

```bash
pip install -r requirements-dev.txt
```

Optional GPU example (CUDA 11.8):

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

Desktop UI:

```bash
python main.py
```

Auto-learning (headless):

```bash
python main.py --auto-learn --max-stocks 200 --continuous
```

Backtest:

```bash
python main.py --backtest
```

Replay:

```bash
python main.py --replay-file path/to/replay.csv --replay-speed 20
```

Health report:

```bash
python main.py --health
```

## Common CLI Commands

```bash
python main.py --train --epochs 100
python main.py --predict 600519
python main.py --auto-learn --max-stocks 50 --epochs 50 --continuous
python main.py --recovery-drill
```

## Configuration

Primary config:
- `config/settings.py`

Optional overrides:
- `config.json`
- selected environment variables

Main config groups:
- `data`
- `model`
- `risk`
- `security`
- `auto_trade`

## Testing

Run all tests:

```bash
pytest -q
```

Focused regression pack:

```bash
pytest -q tests/test_data.py tests/test_data_leakage.py tests/test_models.py tests/test_oms_fills.py tests/test_replay.py tests/test_audit_integrity.py tests/test_executor_health_guard.py
```

Typecheck gate:

```bash
python scripts/typecheck_gate.py
```

## Strategy Extensibility

Custom strategies are loaded from `strategies/*.py`.

Required function contract:
- `generate_signal(df, indicators, context) -> dict`
- return fields: `action` (`buy`/`sell`/`hold`), `score` (`0..1`), optional `reason`

Reference example:
- `strategies/momentum_breakout.py`

Marketplace files:
- `strategies/marketplace.json`
- `strategies/enabled.json`

## Operations

Playbook:
- `docs/OPERATIONS_PLAYBOOK.md`

Preflight checks:

```bash
python scripts/release_preflight.py --observability-url http://127.0.0.1:9090
```

Regulatory readiness:

```bash
python scripts/regulatory_readiness.py
```

HA/DR lease drill:

```bash
python scripts/ha_dr_drill.py --backend sqlite --ttl-seconds 5
```

Deployment snapshot:

```bash
python scripts/deployment_snapshot.py create --snapshot-dir backups
```

Rollback dry-run then confirm:

```bash
python scripts/deployment_snapshot.py restore --archive backups/snapshot_<tag>.tar.gz --dry-run
python scripts/deployment_snapshot.py restore --archive backups/snapshot_<tag>.tar.gz --confirm
```

## Known Disadvantages

1. Desktop-first, single-node architecture: this is strong for local control but weaker for cloud-native scaling and team collaboration.
2. External data-provider and network dependency: quote quality and timeliness still depend on third-party endpoints and connectivity conditions.

## Repository Layout (Top-Level)

- `analysis/`: backtest, replay, sentiment, strategy engine helpers
- `config/`: typed settings and policy config
- `core/`: shared types, symbols, network/env, constants
- `data/`: fetch, clean, cache, store, validate, discover
- `models/`: networks, ensemble, trainer, predictor, auto-learner
- `trading/`: broker, executor, OMS, risk, health, alerts
- `ui/`: PyQt application and widgets
- `utils/`: logging, metrics, security, helper utilities
- `tests/`: regression and integration test coverage

## Safety Note

This system is a decision and risk framework, not a guaranteed-profit engine.
Always validate strategies with backtest + replay + paper/shadow operation before scaling real capital.
