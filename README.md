# Trading Graph

Desktop AI trading system for China A-shares with:
- multi-source market data (Tencent, AkShare/EastMoney, Sina, Yahoo fallback)
- **China network optimization** (auto failover, proxy support, DNS optimization)
- **Enhanced sentiment analysis** (Jin10, Xueqiu, EastMoney, Sina)
- model training and prediction with **explainability**
- **Multi-asset support** (stocks, futures, options, forex, crypto)
- auto-trade execution and risk controls
- **2FA authentication** for security
- **Cloud backup** support (S3, Azure, GCS)
- PyQt real-time charting and operations UI

## Key Features

### 1. Stock Search & Discovery
- **Search all stocks**: Discovers all China A-share stocks (SSE, SZSE, BSE)
- Multiple discovery sources: AkShare, Tencent, CSI index constituents
- Smart scoring based on market cap, volume, and index membership
- Supports 600/601/603/605, 688, 000/001/002/003, 300/301, 83/87/43 prefixes

### 2. Model Training & Explainability
- **Train on all stocks**: `python main.py --train --epochs 100`
- **Train on specific stock**: `python main.py --train-stock 600519 --epochs 100`
- Auto-learning with continuous improvement
- Ensemble models (LSTM, GRU, TCN, Transformer, Hybrid)
- **NEW: SHAP/LIME explainability** for model predictions
- **NEW: Monte Carlo Dropout** for uncertainty estimation
- **NEW: Conformal Prediction** for valid confidence intervals

### 3. Multi-Asset Trading
- **Stocks**: China A-shares (SSE, SZSE, BSE)
- **Futures**: CFFEX, SHFE, CZCE, DCE contracts
- **Options**: SSE 50 ETF, SZSE 300 ETF options
- **Forex**: Major currency pairs
- **Crypto**: Bitcoin, Ethereum support

### 4. China Network Support
- **Fully optimized for mainland China network conditions**
- ✅ 5+ Chinese financial data providers with auto-failover
- ✅ Proxy support (HTTP/SOCKS5) for VPN users
- ✅ China-optimized DNS resolution (114DNS, AliDNS, DNSPod)
- ✅ Extended timeouts for Great Firewall conditions
- ✅ Network diagnostics: `python -m utils.china_diagnostics`

### 5. Enhanced Prediction & Uncertainty
- **Calibrated confidence scores** (confidence = actual accuracy)
- **Uncertainty decomposition** (epistemic vs aleatoric)
- **Dynamic confidence thresholds** by market regime
- **Prediction intervals** with 90% coverage guarantee
- **Ensemble disagreement** as uncertainty metric

### 6. Security & Authentication
- **2FA support** (TOTP with Google Authenticator, Authy)
- **Backup codes** for account recovery
- **Rate limiting** for brute-force protection
- **Encrypted credential storage** with audit logging

### 7. Cloud Backup & Recovery
- **S3/Azure/GCS backup** support
- **Automated scheduled backups**
- **Point-in-time recovery**
- **30-day retention** (configurable)

### 8. Real-Time Charting
- **Live candlestick updates** with real-time price feeds
- **AI prediction overlay** (dashed cyan line)
- **Uncertainty bands** (dotted yellow lines)
- Technical indicators (SMA, EMA, Bollinger Bands, VWAP)
- Interactive hover tooltips with OHLCV data

## Scope

This project is desktop-first and single-node. It is suitable for personal and small-team workflows, not full institutional deployment.
Tooling is Python-only (`pyproject.toml` + `pip` requirements); no Node/NPM step is required.
Runtime singleton behavior can be disabled with `TRADING_DISABLE_SINGLETONS=1` to create isolated in-process instances for testing/multi-run tooling.

## China Network Support

**Fully optimized for mainland China network conditions:**

- ✅ 5 Chinese financial data providers with auto-failover
- ✅ Proxy support (HTTP/SOCKS5) for VPN users
- ✅ China-optimized DNS resolution
- ✅ Connection pooling for Chinese ISPs
- ✅ Extended timeouts for Great Firewall conditions
- ✅ Network diagnostics utility

**Quick setup for China:**

```bash
# Run network diagnostics
python -m utils.china_diagnostics

# Configure proxy (if using VPN)
export TRADING_PROXY_URL=http://127.0.0.1:7890

# Force VPN mode
export TRADING_VPN=1

# Or force China direct mode
export TRADING_CHINA_DIRECT=1
```

See [docs/CHINA_NETWORK.md](docs/CHINA_NETWORK.md) for detailed guide.

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

Provider/fallback policy controls:
- `TRADING_ENABLED_SOURCES` overrides provider set/order (comma/semicolon list, e.g. `yahoo,tencent`).
- `TRADING_STRICT_REALTIME_QUOTES=1` disables quote fallback layers (spot cache, last-good, DB last-close). Live mode defaults to strict realtime.
- `TRADING_ALLOW_LAST_CLOSE_FALLBACK=1` opt-in for DB last-close fallback (disabled by default).
- `TRADING_ALLOW_STALE_REALTIME_FALLBACK=1` allows stale fallback quotes to pass through as delayed (disabled by default).
- `TRADING_INTRADAY_SESSION_POLICY=none` disables CN-only intraday session clipping for non-CN markets.
- `TRADING_FETCHER_SCOPE=thread|process` controls singleton isolation (`thread` default to avoid cross-thread mutable-cache coupling).

Secure storage controls:
- `TRADING_SECURE_MASTER_KEY` uses an external Fernet key from env (no local key file write).
- `TRADING_SECURE_KEY_PATH` overrides local key-file path (default is outside `data_storage/`).
- `TRADING_SECURE_STORAGE_PATH` overrides encrypted credential store path.
- `TRADING_LOCK_ACCESS_IDENTITY=1` locks runtime role/user mutations in access control.

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
