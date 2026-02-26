# Production Enhancements

Production-grade features for Trading Graph deployment.

---

## Overview

Trading Graph includes several production enhancements for robust deployment:

| Module | File | Purpose |
|--------|------|---------|
| Recovery Metrics | `utils/recovery_metrics.py` | Operation tracking, success/failure metrics |
| Prometheus Metrics | `utils/metrics.py` | Prometheus-compatible metrics export |
| Metrics HTTP Server | `utils/metrics_http.py` | HTTP endpoint for metrics |
| Institutional Readiness | `utils/institutional.py` | Institutional control checks |
| Recovery Manager | `utils/recovery_manager.py` | Recovery operation coordination |
| Confidence Calibration | `models/confidence_calibration.py` | Calibrated confidence scores |
| Regime Detection | `models/regime_detection.py` | Market regime detection |
| Realistic Backtest | `analysis/realistic_backtest.py` | Market impact modeling |

---

## 1. Recovery Metrics

**File:** `utils/recovery_metrics.py`

### Purpose

Track recovery operations across the system with:
- Success/failure tracking per operation type
- Performance monitoring (latency, attempts)
- Health status dashboard
- Disk persistence for metrics

### Usage

```python
from utils.recovery_metrics import RecoveryMetrics

metrics = RecoveryMetrics(
    metrics_dir="logs/recovery",
    retention_hours=24,
    max_records=10000,
)

# Record operation
metrics.record_operation(
    operation="fetch_data",
    success=True,
    duration_seconds=1.5,
    attempts=2,
)

# Get metrics
summary = metrics.get_summary()
health = metrics.get_health()

# Export for monitoring
export = metrics.export_metrics(output_path="metrics.json")
```

### Data Classes

| Class | Purpose |
|-------|---------|
| `OperationRecord` | Single operation record |
| `OperationMetrics` | Aggregated metrics per operation |
| `RecoveryHealth` | Overall health status |
| `RecoveryMetrics` | Main metrics collector |

### Health Status Levels

| Status | Condition |
|--------|-----------|
| `healthy` | Success rate > 80%, consecutive failures < 3 |
| `degraded` | Success rate 50-80% OR consecutive failures 3-4 |
| `unhealthy` | Success rate < 50% OR consecutive failures >= 5 |

---

## 2. Prometheus Metrics

**File:** `utils/metrics.py`

### Purpose

Export Prometheus-compatible metrics for monitoring:
- Counters (cumulative values)
- Gauges (current values)
- Histograms (distribution of values)

### Usage

```python
from utils.metrics import MetricsRegistry

registry = MetricsRegistry()

# Counter
registry.inc_counter("requests_total", value=1)
registry.inc_counter("errors_total", labels={"type": "timeout"})

# Gauge
registry.set_gauge("active_connections", value=10)
registry.set_gauge("queue_size", value=5)

# Histogram
registry.observe_histogram(
    "request_latency_seconds",
    value=0.125,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# Export in Prometheus format
metrics_text = registry.export_prometheus()
```

### Metric Types

| Type | Use Case | Example |
|------|----------|---------|
| Counter | Cumulative counts | Total requests, errors |
| Gauge | Current values | Active connections, queue size |
| Histogram | Distributions | Latency, request size |

---

## 3. Metrics HTTP Server

**File:** `utils/metrics_http.py`

### Purpose

Serve metrics via HTTP endpoints for monitoring systems.

### Configuration

```bash
# In .env
TRADING_METRICS_ENABLED=1
TRADING_METRICS_PORT=8000
TRADING_METRICS_HOST=127.0.0.1
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/metrics` | GET | Prometheus-compatible metrics |
| `/healthz` | GET | Simple health check (HTTP 200 = healthy) |
| `/api/v1/dashboard` | GET | Dashboard data with system stats |

### Usage

```python
from utils.metrics_http import serve

# Start server
server = serve(port=8000, host="127.0.0.1")

# Access metrics
# curl http://127.0.0.1:8000/metrics

# Stop server
server.stop()
```

---

## 4. Institutional Readiness

**File:** `utils/institutional.py`

### Purpose

Validate institutional readiness controls:
- Audit logging enabled
- Risk controls configured
- Compliance checks passed
- Security measures active

### Usage

```python
from utils.institutional import collect_institutional_readiness

report = collect_institutional_readiness()

if report.get("pass", False):
    print("Institutional readiness: PASS")
else:
    print(f"Failed controls: {report.get('failed_required_controls', [])}")
```

### Required Controls

| Control | Description |
|---------|-------------|
| Audit logging | All actions logged with timestamps |
| Risk limits | Position limits, daily loss limits configured |
| Security | Encrypted credential storage |
| Health checks | System health monitoring active |

---

## 5. Confidence Calibration

**File:** `models/confidence_calibration.py`

### Purpose

Calibrate model confidence scores using:
- Temperature scaling
- Isotonic regression
- Monte Carlo dropout uncertainty

### Usage

```python
from models.confidence_calibration import ConfidenceCalibrator

calibrator = ConfidenceCalibrator(n_buckets=10)

# Record predictions for calibration
calibrator.record_prediction(
    confidence=0.75,
    correct=True,
)

# Get calibration report
report = calibrator.get_calibration_report()
print(f"Expected Calibration Error: {report['expected_calibration_error']:.3f}")
print(f"Brier Score: {report['brier_score']:.3f}")
```

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Expected Calibration Error (ECE) | Weighted gap between confidence and accuracy | < 0.05 |
| Brier Score | Mean squared error of predictions | < 0.25 |
| Calibration Slope | Slope of calibration curve | ~1.0 |

---

## 6. Regime Detection

**File:** `models/regime_detection.py`

### Purpose

Detect market regime changes for adaptive model behavior:
- Bull/Bear market detection
- Volatility regime identification
- Crisis detection
- Adaptive retraining triggers

### Usage

```python
from models.regime_detection import RegimeDetector, MarketRegime

detector = RegimeDetector(lookback_days=20)

# Update with price data
detector.update_price("600519", datetime.now(), 1800.0, volume=1e6)

# Get current regime
metrics = detector.detect_regime("600519")
print(f"Regime: {metrics.regime.value}")
print(f"Volatility: {metrics.volatility_level}")
```

### Market Regimes

| Regime | Description |
|--------|-------------|
| `BULL_LOW_VOL` | Bull market, low volatility |
| `BULL_HIGH_VOL` | Bull market, high volatility |
| `BEAR_LOW_VOL` | Bear market, low volatility |
| `BEAR_HIGH_VOL` | Bear market, high volatility |
| `SIDEWAYS` | Range-bound market |
| `CRISIS` | Extreme volatility, crash |
| `TRANSITION` | Regime change in progress |

---

## 7. Realistic Backtest

**File:** `analysis/realistic_backtest.py`

### Purpose

Model realistic trading conditions:
- Market impact (square-root law)
- Transaction costs (commission, stamp duty)
- Slippage estimation
- Fill probability modeling

### Usage

```python
from analysis.realistic_backtest import (
    RealisticBacktestEngine,
    BacktestConfig,
)

config = BacktestConfig(
    initial_capital=1e6,
    impact_coefficient=0.15,
    base_spread_bps=10.0,
    commission_rate=0.00025,
    stamp_duty_rate=0.001,
)

engine = RealisticBacktestEngine(config)

# Simulate fill
fill = engine.simulate_fill(
    order_id="order_001",
    symbol="600519",
    side="buy",
    quantity=100,
    price=1800.0,
    daily_volume=1e6,
)

print(f"Fill price: {fill.fill_price:.2f}")
print(f"Market impact: {fill.market_impact_bps:.1f} bps")
print(f"Transaction cost: {fill.transaction_cost:.2f} CNY")
```

### Transaction Costs (China A-Shares)

| Cost | Rate | Side |
|------|------|------|
| Commission | 0.025% (min 5 CNY) | Both |
| Stamp Duty | 0.1% | Sell only |
| Transfer Fee | 0.002% | Both |
| Exchange Fee | 0.00487% | Both |

---

## Integration Example

```python
from utils.recovery_metrics import get_recovery_metrics
from utils.metrics import MetricsRegistry
from models.confidence_calibration import ConfidenceCalibrator
from models.regime_detection import RegimeDetector

class ProductionSystem:
    def __init__(self):
        self.recovery_metrics = get_recovery_metrics()
        self.metrics_registry = MetricsRegistry()
        self.calibrator = ConfidenceCalibrator()
        self.regime_detector = RegimeDetector()

    def run_prediction(self, symbol: str) -> dict:
        start_time = time.time()
        success = False

        try:
            # Get regime
            regime = self.regime_detector.detect_regime(symbol)

            # Run prediction
            prediction = self.predictor.predict(symbol)

            # Record for calibration
            self.calibrator.record_prediction(
                confidence=prediction.confidence,
                correct=None,  # Will be known later
            )

            # Record recovery metrics
            duration = time.time() - start_time
            self.recovery_metrics.record_operation(
                operation="predict",
                success=True,
                duration_seconds=duration,
            )

            # Update Prometheus metrics
            self.metrics_registry.inc_counter("predictions_total")
            self.metrics_registry.observe_histogram(
                "prediction_latency_seconds",
                value=duration,
            )

            success = True
            return prediction

        finally:
            if not success:
                self.recovery_metrics.record_operation(
                    operation="predict",
                    success=False,
                    duration_seconds=time.time() - start_time,
                )
                self.metrics_registry.inc_counter(
                    "errors_total",
                    labels={"source": "prediction"},
                )
```

---

## Monitoring Dashboard

### Key Metrics to Track

| Metric | Type | Alert Threshold |
|--------|------|-----------------|
| `predictions_total` | Counter | Sudden drop |
| `errors_total` | Counter | > 5 per minute |
| `prediction_latency_seconds` | Histogram | p99 > 2s |
| `consecutive_failures` | Gauge | > 3 |
| `success_rate_24h` | Gauge | < 0.8 |
| `calibration_error` | Gauge | > 0.05 |

### Grafana Dashboard Example

```json
{
  "panels": [
    {
      "title": "Prediction Rate",
      "targets": [{"expr": "rate(predictions_total[5m])"}]
    },
    {
      "title": "Error Rate",
      "targets": [{"expr": "rate(errors_total[5m])"}]
    },
    {
      "title": "Latency Histogram",
      "targets": [{"expr": "histogram_quantile(0.99, prediction_latency_seconds_bucket)"}]
    },
    {
      "title": "Health Status",
      "targets": [{"expr": "health_status"}]
    }
  ]
}
```

---

## Best Practices

1. **Enable metrics in production**: Set `TRADING_METRICS_ENABLED=1`
2. **Monitor recovery health**: Check `get_health()` regularly
3. **Calibrate confidence**: Record predictions for calibration
4. **Detect regime changes**: Adapt behavior based on market regime
5. **Use realistic backtest**: Model transaction costs and market impact

---

## Troubleshooting

### Metrics Not Exporting

```bash
# Check if server is running
curl http://localhost:8000/metrics

# Check logs for errors
tail -f logs/trading.log | grep metrics
```

### High Error Rate

```bash
# Check recovery metrics
python -c "from utils.recovery_metrics import get_recovery_metrics; print(get_recovery_metrics().get_summary())"

# Run diagnostics
python main.py --doctor --doctor-strict
```

### Calibration Drift

```python
from models.confidence_calibration import ConfidenceCalibrator

calibrator = ConfidenceCalibrator()
report = calibrator.get_calibration_report()

if report['expected_calibration_error'] > 0.05:
    print("Recalibration needed")
    calibrator.recalibrate()
```
