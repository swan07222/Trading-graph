# Operations Playbook

Deployment, rollback, and runtime observability for Trading Graph.

---

## Table of Contents

- [Pre-Deploy Checks](#1-pre-deploy-checks)
- [Deployment](#2-deployment)
- [Rollback](#3-rollback)
- [Health Monitoring](#4-health-monitoring)
- [Incident Response](#5-incident-response)

---

## 1. Pre-Deploy Checks

Run full preflight checks before every deployment:

```bash
# System health check
python main.py --health

# Full diagnostics
python main.py --doctor

# Strict mode (fails on any issue)
python main.py --doctor --doctor-strict

# Type checking
python scripts/typecheck_gate.py

# Run tests
pytest -q
```

### Health Gate Checks

The `--doctor-strict` flag validates:

| Check | Description |
|-------|-------------|
| **Dependencies** | Required Python packages installed |
| **Paths** | Data, model, log directories exist and writable |
| **Models** | Ensemble and forecaster artifacts present |
| **Config** | Configuration validation passes |
| **Institutional Readiness** | Required controls enabled |

### Type Gate

```bash
# Run type checker
python scripts/typecheck_gate.py

# Update baseline if legacy issues exist intentionally
python scripts/typecheck_gate.py --write-baseline
```

---

## 2. Deployment

### Standard Deployment

```bash
# 1. Stop existing process
# (If running as service)
sudo systemctl stop trading-graph

# 2. Backup current state
python scripts/deployment_snapshot.py create --snapshot-dir backups

# 3. Deploy new version
git pull origin main
pip install -r requirements.txt

# 4. Run post-deploy checks
python main.py --health
python main.py --doctor

# 5. Start service
sudo systemctl start trading-graph

# 6. Verify health
curl http://localhost:8000/healthz
```

### Metrics Server

Enable metrics server for monitoring:

```bash
# In .env
TRADING_METRICS_ENABLED=1
TRADING_METRICS_PORT=8000
TRADING_METRICS_HOST=127.0.0.1
```

Access endpoints:
- `GET /metrics` - Prometheus-compatible metrics
- `GET /healthz` - Health check
- `GET /api/v1/dashboard` - Dashboard data

---

## 3. Rollback

### Create Snapshot Before Deploy

```bash
python scripts/deployment_snapshot.py create \
  --snapshot-dir backups \
  --include-configs \
  --include-models
```

### Dry-Run Rollback

```bash
python scripts/deployment_snapshot.py restore \
  --archive backups/snapshot_<tag>.tar.gz \
  --dry-run
```

### Execute Rollback

```bash
python scripts/deployment_snapshot.py restore \
  --archive backups/snapshot_<tag>.tar.gz \
  --confirm
```

### Post-Rollback Verification

```bash
# 1. Run health checks
python main.py --health

# 2. Run diagnostics
python main.py --doctor

# 3. Verify metrics endpoint
curl http://localhost:8000/healthz

# 4. Test prediction
python main.py --predict 600519
```

---

## 4. Health Monitoring

### Runtime Health Checks

```bash
# Basic health (always returns healthy for analysis-only mode)
python main.py --health

# Full diagnostics with strict mode
python main.py --doctor --doctor-strict
```

### Metrics Endpoints

When `TRADING_METRICS_ENABLED=1`:

| Endpoint | Description |
|----------|-------------|
| `GET /metrics` | Prometheus-compatible metrics (counters, gauges, histograms) |
| `GET /healthz` | Simple health check (HTTP 200 = healthy) |
| `GET /api/v1/dashboard` | Dashboard data with system stats |

### Recovery Metrics

The system tracks recovery operations via `utils/recovery_metrics.py`:

```python
from utils.recovery_metrics import get_recovery_metrics

metrics = get_recovery_metrics()

# Record operation
metrics.record_operation(
    operation="fetch_data",
    success=True,
    duration_seconds=1.5,
    attempts=2,
)

# Get summary
summary = metrics.get_summary()

# Export metrics
export = metrics.export_metrics(output_path="metrics.json")
```

### Key Metrics to Monitor

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `data_fetch_errors` | Data fetch failures | > 5 per minute |
| `prediction_latency_seconds` | Prediction response time | p99 > 2s |
| `model_load_errors` | Model loading failures | > 0 |
| `cache_hit_ratio` | Cache effectiveness | < 0.5 |
| `consecutive_failures` | Consecutive operation failures | > 3 |

---

## 5. Incident Response

### Data Fetch Failures

**Symptoms:**
- Missing price data
- Stale quotes
- Multiple source failures

**Response:**
```bash
# 1. Check network connectivity
python -m utils.china_diagnostics

# 2. Verify source health
# Check logs for source-specific errors

# 3. Enable VPN mode if in China
export TRADING_VPN=1
export TRADING_PROXY_URL=http://127.0.0.1:7890

# 4. Restart data fetcher
# (Restart application)
```

### Model Prediction Failures

**Symptoms:**
- Prediction errors
- Missing model artifacts
- Low confidence scores

**Response:**
```bash
# 1. Check model files
ls -la models_saved/

# 2. Verify model integrity
python main.py --doctor

# 3. Retrain if needed
python main.py --train --epochs 100

# 4. Fallback to cached predictions
# (Automatic - cache TTL is 5 seconds)
```

### Memory Issues

**Symptoms:**
- Slow performance
- Out of memory errors
- Application crashes

**Response:**
```bash
# 1. Clear caches
# (Restart application - caches are in-memory)

# 2. Reduce cache sizes in config
# Edit .env: reduce TRADING_MAX_MEMORY_CACHE_MB

# 3. Enable Redis for external caching
docker run -d -p 6379:6379 redis:latest
```

### Network/VPN Issues

**Symptoms:**
- Connection timeouts
- Source unreachable
- High latency

**Response:**
```bash
# 1. Test network connectivity
curl -v https://www.tencent.com
curl -v https://www.akshare.xyz

# 2. Check proxy configuration
echo $TRADING_PROXY_URL

# 3. Switch source priority
# Edit .env: change TRADING_PRIMARY_SOURCE

# 4. Enable China direct mode if applicable
export TRADING_CHINA_DIRECT=1
```

---

## 6. Maintenance Tasks

### Daily

```bash
# Check health
python main.py --health

# Review logs
tail -f logs/trading.log | grep -E "ERROR|WARNING"
```

### Weekly

```bash
# Run full diagnostics
python main.py --doctor

# Clean old logs
find logs/ -name "*.log" -mtime +7 -delete

# Backup models
cp -r models_saved/ backups/models_$(date +%Y%m%d)
```

### Monthly

```bash
# Retrain models with new data
python main.py --train --epochs 100

# Review and update dependencies
pip list --outdated
pip install --upgrade -r requirements.txt

# Clean old metrics
# (Automatic - retention is 24 hours by default)
```

---

## 7. Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_MODE` | simulation | simulation \| live |
| `TRADING_MODEL_DIR` | models_saved | Model artifacts directory |
| `TRADING_LOG_LEVEL` | INFO | DEBUG \| INFO \| WARNING \| ERROR |
| `TRADING_METRICS_ENABLED` | 0 | Enable metrics server |
| `TRADING_METRICS_PORT` | 8000 | Metrics server port |
| `TRADING_VPN` | 0 | Enable VPN mode |
| `TRADING_CHINA_DIRECT` | 0 | China direct mode |
| `TRADING_CONNECTION_TIMEOUT` | 30 | Connection timeout (seconds) |

### File Locations

| Path | Purpose |
|------|---------|
| `data_storage/` | Market data, cached bars |
| `models_saved/` | Trained model artifacts |
| `logs/` | Application logs |
| `cache/` | Temporary cache files |
| `audit/` | Audit logs |

---

## 8. Support Contacts

| Issue Type | Action |
|------------|--------|
| Bug report | Open GitHub issue |
| Feature request | Open GitHub issue |
| Question | GitHub Discussions |
| Security issue | Contact maintainers directly |

---

## Appendix: Health Check Output

### Healthy System

```json
{
  "status": "healthy",
  "execution_enabled": false,
  "can_trade": false,
  "analysis_ready": true,
  "note": "Trading execution disabled"
}
```

### Doctor Report

```json
{
  "timestamp": "2026-02-25T10:00:00",
  "dependencies": {
    "psutil": true,
    "numpy": true,
    "pandas": true,
    "sklearn": true,
    "torch": true
  },
  "paths": {
    "data_dir": {"exists": true, "writable": true},
    "model_dir": {"exists": true, "writable": true},
    "log_dir": {"exists": true, "writable": true}
  },
  "models": {
    "ensembles": 5,
    "forecasters": 3,
    "scalers": 8
  },
  "institutional_readiness": {
    "pass": true
  }
}
```
