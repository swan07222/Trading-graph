# Trading Graph

Desktop AI trading system for China A-shares with:
- multi-source market data (Tencent, AkShare/EastMoney, Sina, Yahoo fallback)
- model training and prediction
- auto-trade execution and risk controls
- PyQt real-time charting and operations UI

## Scope

This project is desktop-first and single-node. It is suitable for personal and small-team workflows, not full institutional deployment.
Tooling is Python-only (`pyproject.toml` + `pip` requirements); no Node/NPM step is required.

## Key Capabilities

- Robust history/realtime data fetch with source health scoring and network-aware routing
- Daily history quorum checks before persisting internet data to local DB
- Session cache + SQLite persistence with cleanup/sanitization guards
- Live signal monitor + auto-trade policy controls
- Replay/backtest utilities and operations scripts

## Data Source Policy

For CN equities:
- Realtime: Tencent primary, then controlled fallbacks (`spot_cache`, recent last-good, local last close)
- Intraday history: best-source selection by quality score, stale-bar detection, cross-validation
- Daily history: multi-source consensus merge (Tencent/AkShare/Sina when available) + quorum gate before DB write

`AkShareSource` availability depends on EastMoney reachability (`env.eastmoney_ok`) and China-direct network conditions.

Emergency endpoint override controls (no code change needed):
- `TRADING_SINA_KLINE_ENDPOINTS`
- `TRADING_TENCENT_BATCH_ENDPOINTS`
- `TRADING_TENCENT_DAILY_ENDPOINTS`

For multiple endpoints, separate entries with `;`.

## Candle Rendering Pipeline

1. Load bars from fetcher/database/session cache
2. Normalize interval and bucket timestamps
3. Sanitize OHLC shape and scale
4. Drop mixed-interval / malformed bars
5. Render candles + overlays + forecast

## Recent Reliability Fixes

- Fixed history loading to use native `1d/1wk/1mo` fetch intervals (instead of forcing all chart history through `1m` resampling)
- Tightened render-side intraday guardrails to block oversized outlier candles
- Overlays are now computed from the same filtered candles that are actually rendered
- Fixed chart viewport bug where X-range always started at `0`
- Removed duplicate tick session-cache persistence path
- Fixed pending-approval button sizing in UI action table

## Why Candles Can Display Incorrectly

Common root causes:
- mixed intervals merged into one chart window
- malformed intraday OHLC from provider partial rows
- scale mismatch (for example provider rows in wrong magnitude)
- stale/flat bars dominating when network/source is degraded
- UI loading daily/weekly/monthly from truncated minute windows

Current code addresses these with:
- interval filtering and bucket normalization
- OHLC sanitization and outlier-drop guards
- source quality scoring and consensus merge
- fallback layering with explicit source tagging
- native interval fetch for higher-timeframe charts

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

Live trading profile (includes broker connector dependency):

```bash
pip install -r requirements-live.txt
```

Run UI:

```bash
python main.py
```

## Useful Commands

Train:

```bash
python main.py --train --epochs 100
```

Predict:

```bash
python main.py --predict 600519
```

Auto-learn:

```bash
python main.py --auto-learn --max-stocks 50 --continuous
```

Backtest:

```bash
python main.py --backtest
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

Type gate:

```bash
python scripts/typecheck_gate.py
```

Strict type gate:

```bash
python scripts/typecheck_strict_gate.py
```

Live readiness doctor:

```bash
python main.py --doctor --doctor-live --doctor-strict
```

## Main Directories

- `data/`: data fetch, cache, persistence, validation
- `models/`: model training/prediction/auto-learning
- `trading/`: execution, risk, OMS, health
- `ui/`: PyQt application and chart rendering
- `analysis/`: replay, backtest, strategy/sentiment modules
- `tests/`: regression and integration coverage

## Safety Note

This is a decision-support and execution framework, not a guaranteed-profit system. Use paper/simulation and replay/backtest validation before scaling real capital.
