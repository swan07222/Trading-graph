# Production Trading Enhancements

> Deprecated for this build: this document describes legacy live-execution modules
> that are not part of the current analysis-only product scope.
> Use `docs/OPERATIONS_PLAYBOOK.md` and `README.md` for current workflows.

This document describes the production-grade enhancements added to address the disadvantages of using this project for real-time trading.

## Overview

The following modules have been added to make the trading system more robust for production use:

| Module | File | Purpose |
|--------|------|---------|
| Execution Risk | `trading/execution_risk.py` | Smart order routing, slippage control, execution validation |
| Circuit Breaker | `trading/circuit_breaker.py` | Multi-level circuit breakers, kill switch, compliance |
| Regime Detection | `models/regime_detection.py` | Market regime detection, adaptive retraining |
| Data Quality | `data/quality_monitor.py` | Multi-source validation, provider health monitoring |
| Confidence Calibration | `models/confidence_calibration.py` | Calibrated confidence scores, uncertainty bands |
| Realistic Backtest | `analysis/realistic_backtest.py` | Market impact modeling, transaction costs |
| Production Health | `trading/production_health.py` | System health monitoring, production gate |

---

## 1. Execution Risk Mitigation

**File:** `trading/execution_risk.py`

### Problem Addressed
- Retail-grade broker connector risk
- Slippage uncertainty (up to 5% in extreme cases)
- No direct market access (DMA)
- Execution not guaranteed at predicted prices

### Key Features

```python
from trading.execution_risk import (
    SmartOrderRouter,
    ExecutionValidator,
    ExecutionConfig,
    create_execution_pipeline,
)

# Create execution pipeline
config = ExecutionConfig(
    max_slippage_bps=20.0,  # Max 20 bps slippage
    max_participation_rate=0.05,  # Max 5% of daily volume
    order_split_threshold=100000.0,  # Split orders > 100k
)

router, validator = create_execution_pipeline(config)

# Estimate slippage before order
estimate = router.estimate_slippage(
    symbol="600519",
    side=OrderSide.BUY,
    quantity=100,
    price=1800.0,
    daily_volume=1e6,
    volatility=0.30,
    bid_ask_spread=0.001,
)

print(f"Estimated slippage: {estimate.total_estimated_bps:.1f} bps")
print(f"Worst case: {estimate.worst_case_bps:.1f} bps")

# Validate before submission
is_valid, reason = validator.validate_pre_trade(
    order=my_order,
    current_price=1800.0,
    quote_age_ms=100,
)

if not is_valid:
    print(f"Order rejected: {reason}")
```

### Classes

- **SmartOrderRouter**: Smart order routing with slippage estimation
- **ExecutionValidator**: Pre-trade and post-trade validation
- **ExecutionReport**: Post-trade execution quality analysis
- **SlippageEstimate**: Real-time slippage estimation with confidence bounds

---

## 2. Circuit Breaker & Kill Switch

**File:** `trading/circuit_breaker.py`

### Problem Addressed
- Desktop application = single point of failure
- No redundancy/failover for production trading
- Network issues can disrupt data feeds
- No regulatory compliance for automated trading

### Key Features

```python
from trading.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    RedundantHealthMonitor,
    ComplianceChecker,
    TradingState,
)

# Configure circuit breakers
config = CircuitBreakerConfig(
    max_daily_loss_pct=3.0,  # Halt if down 3% in a day
    max_weekly_loss_pct=7.0,  # Halt if down 7% in a week
    max_drawdown_pct=10.0,  # Reduce risk at 10% drawdown
    max_data_latency_ms=5000,  # Halt if data > 5s old
    max_consecutive_failures=5,  # Halt after 5 data failures
)

# Create circuit breaker
cb = CircuitBreaker(config)

# Update risk metrics
cb.update_metrics(
    daily_pnl_pct=-1.5,
    drawdown_pct=5.0,
    data_latency_ms=200,
    total_exposure_pct=0.60,
)

# Check if trading is allowed
if cb.is_trading_halted:
    print("Trading halted by circuit breaker!")
    print(f"State: {cb.current_state}")

# Record order (checks rate limits)
if cb.record_order(order_value_pct=0.02):
    submit_order(my_order)

# Compliance checking
compliance = ComplianceChecker()
is_compliant, reason = compliance.validate_order(
    symbol="600519",
    side="buy",
    quantity=100,
    price=1800.0,
    current_price=1795.0,
)
```

### Circuit Breaker Levels

| Level | Action |
|-------|--------|
| WARNING | Log warning, continue trading |
| REDUCE_RISK | Reduce position sizes |
| HALT_NEW_POSITIONS | Stop opening new positions |
| HALT_ALL_TRADING | Stop all trading |
| EMERGENCY_CLOSE | Close all positions immediately |

---

## 3. Model Regime Detection

**File:** `models/regime_detection.py`

### Problem Addressed
- ML models can overfit historical data
- Market regime changes can invalidate trained models
- Requires continuous retraining
- No guarantee models adapt quickly to black swan events

### Key Features

```python
from models.regime_detection import (
    RegimeDetector,
    AdaptiveRetrainer,
    RegimeAwareEnsemble,
    MarketRegime,
)

# Detect market regime
detector = RegimeDetector(lookback_days=20)

# Update with price data
detector.update_price("600519", datetime.now(), 1800.0, volume=1e6)

# Get current regime
metrics = detector.detect_regime("600519")
print(f"Regime: {metrics.regime.value}")
print(f"Confidence: {metrics.confidence:.1%}")
print(f"Volatility: {metrics.volatility_level:.1%}")

# Adaptive retraining
retrainer = AdaptiveRetrainer(detector)

# Record predictions for performance tracking
retrainer.record_prediction(
    model_id="ensemble_v1",
    symbol="600519",
    predicted_signal="buy",
    actual_signal="buy",
    confidence=0.75,
    error_pct=0.02,
    regime=metrics.regime,
)

# Check if retrain needed
should_retrain, reason = retrainer.should_retrain(
    model_id="ensemble_v1",
    symbol="600519",
    current_regime=metrics.regime,
)

if should_retrain:
    print(f"Retraining triggered: {reason}")
    # Trigger retraining...

# Black swan response
if metrics.regime == MarketRegime.CRISIS:
    retrainer.trigger_black_swan_retrain()
```

### Market Regimes

| Regime | Description |
|--------|-------------|
| BULL_LOW_VOL | Bull market, low volatility |
| BULL_HIGH_VOL | Bull market, high volatility |
| BEAR_LOW_VOL | Bear market, low volatility |
| BEAR_HIGH_VOL | Bear market, high volatility |
| SIDEWAYS | Range-bound market |
| CRISIS | Extreme volatility, crash |
| TRANSITION | Regime change in progress |

---

## 4. Data Quality Monitoring

**File:** `data/quality_monitor.py`

### Problem Addressed
- Free data providers - not professional-grade
- Auto-failover suggests unreliable connections
- No direct exchange feed connections
- Data delays possible

### Key Features

```python
from data.quality_monitor import (
    DataQualityMonitor,
    ProviderHealthMonitor,
    MultiSourceValidator,
    DataQualityConfig,
    create_data_quality_pipeline,
)

# Create data quality pipeline
config = DataQualityConfig(
    stale_threshold_seconds=60.0,
    max_price_jump_pct=0.10,
    provider_failure_threshold=5,
)

quality_monitor, provider_monitor, validator = create_data_quality_pipeline(config)

# Validate incoming data
report = quality_monitor.validate_data(
    symbol="600519",
    source="tencent",
    price=1800.0,
    volume=1e6,
    data_timestamp=datetime.now(),
)

print(f"Quality: {report.quality.value}")
print(f"Score: {report.quality_score:.1f}/100")
print(f"Latency: {report.latency_ms:.0f}ms")

if report.is_stale:
    print("WARNING: Data is stale!")

# Track provider health
provider_monitor.record_request(
    provider_id="tencent",
    success=True,
    latency_ms=150,
)

# Get best provider
best = provider_monitor.get_best_provider()
print(f"Best data provider: {best}")

# Multi-source validation
validator.record_price("600519", "tencent", 1800.0, datetime.now())
validator.record_price("600519", "akshare", 1799.5, datetime.now())

is_valid, reason, consensus = validator.validate_consensus(
    symbol="600519",
    source="tencent",
    price=1800.0,
)

if not is_valid:
    print(f"Price outlier detected: {reason}")
```

### Data Quality Levels

| Quality | Description |
|---------|-------------|
| EXCELLENT | < 100ms latency, all validations pass |
| GOOD | < 500ms latency, minor issues |
| FAIR | < 2000ms latency, some issues |
| POOR | > 2000ms latency, multiple issues |
| STALE | Data too old |
| INVALID | Failed validation |

---

## 5. Confidence Calibration

**File:** `models/confidence_calibration.py`

### Problem Addressed
- Only 70%+ confidence threshold - meaning 30%+ could be wrong
- Relies on historical patterns
- Model predictions are probabilistic, not deterministic

### Key Features

```python
from models.confidence_calibration import (
    ConfidenceCalibrator,
    UncertaintyEstimator,
    DynamicThresholdManager,
    create_calibrated_prediction,
)

# Calibrate confidence scores
calibrator = ConfidenceCalibrator(n_buckets=10)

# Record predictions for calibration
prediction = create_calibrated_prediction(
    symbol="600519",
    signal="buy",
    raw_confidence=0.75,
    current_price=1800.0,
    ensemble_predictions=[1820, 1815, 1825, 1810, 1830],
    regime="bull_low_vol",
    calibrator=calibrator,
)

print(f"Raw confidence: {prediction.raw_confidence:.1%}")
print(f"Calibrated confidence: {prediction.calibrated_confidence:.1%}")
print(f"Uncertainty: {prediction.uncertainty:.1%}")
print(f"Prediction range: {prediction.prediction_interval_lower:.2f} - {prediction.prediction_interval_upper:.2f}")
print(f"Reliable: {prediction.is_reliable}")

# Dynamic thresholds
threshold_mgr = DynamicThresholdManager(base_threshold=0.70)

# Update thresholds based on regime
threshold_mgr.update_regime_threshold(
    regime="bull_low_vol",
    volatility=0.20,
    recent_accuracy=0.65,
)

# Check if should trade
should_trade, reason = threshold_mgr.should_trade(
    confidence=0.72,
    symbol="600519",
    regime="bull_low_vol",
    uncertainty=0.15,
)

if should_trade:
    print(f"Trade approved: {reason}")
else:
    print(f"Trade rejected: {reason}")

# Get calibration report
report = calibrator.get_calibration_report()
print(f"Expected Calibration Error: {report['expected_calibration_error']:.3f}")
```

### Confidence Levels

| Level | Range | Action |
|-------|-------|--------|
| VERY_LOW | 0-40% | Do not trade |
| LOW | 40-55% | Paper trade only |
| MEDIUM | 55-70% | Small size |
| HIGH | 70-85% | Normal size |
| VERY_HIGH | 85-100% | Can increase size |

---

## 6. Realistic Backtest

**File:** `analysis/realistic_backtest.py`

### Problem Addressed
- Backtest results don't guarantee live performance
- Simulation assumes perfect execution
- Real market impact not fully captured

### Key Features

```python
from analysis.realistic_backtest import (
    RealisticBacktestEngine,
    BacktestConfig,
    MarketImpactModel,
    TransactionCostModel,
    create_realistic_backtest_engine,
)

# Create realistic backtest engine
config = BacktestConfig(
    initial_capital=1e6,
    impact_coefficient=0.15,  # Square-root law coefficient
    base_spread_bps=10.0,
    commission_rate=0.00025,  # 0.025%
    stamp_duty_rate=0.001,  # 0.1% on sells
    max_participation_rate=0.05,  # Max 5% of daily volume
)

engine = create_realistic_backtest_engine(config)

# Set daily volume for symbol
engine.set_daily_volume("600519", daily_volume=1e6)

# Simulate realistic fill
from analysis.realistic_backtest import OrderType

fill = engine.simulate_fill(
    order_id="order_001",
    symbol="600519",
    order_type=OrderType.MARKET,
    side="buy",
    quantity=100,
    price=1800.0,
    daily_volume=1e6,
)

print(f"Requested: {fill.requested_qty} @ {fill.requested_price:.2f}")
print(f"Filled: {fill.filled_qty} @ {fill.fill_price:.2f}")
print(f"Fill type: {fill.fill_type.value}")
print(f"Market impact: {fill.market_impact_bps:.1f} bps")
print(f"Slippage: {fill.slippage_bps:.1f} bps")
print(f"Transaction cost: {fill.transaction_cost:.2f} CNY")
print(f"Total cost: {fill.total_cost_bps:.1f} bps")

# Get execution statistics
stats = engine.get_execution_statistics()
print(f"Fill rate: {stats['fill_rate']:.1%}")
print(f"Avg total cost: {stats['avg_total_cost_bps']:.1f} bps")
```

### Transaction Costs (China A-Shares)

| Cost | Rate | Side |
|------|------|------|
| Commission | 0.025% (min 5 CNY) | Both |
| Stamp Duty | 0.1% | Sell only |
| Transfer Fee | 0.002% | Both |
| Exchange Fee | 0.00487% | Both |

---

## 7. Production Health Checks

**File:** `trading/production_health.py`

### Problem Addressed
- Desktop application = single point of failure
- No redundancy/failover
- Network issues disrupt data feeds
- No institutional-grade infrastructure

### Key Features

```python
from trading.production_health import (
    ProductionHealthMonitor,
    ProductionGate,
    create_production_monitor,
    create_production_gate,
    ReadinessLevel,
)

# Create and start health monitor
monitor = create_production_monitor()
monitor.start()

# Get current health
health = monitor.get_health()
print(f"Status: {health.status.value}")
print(f"Readiness: {health.readiness.value}")
print(f"Uptime: {health.uptime_seconds:.0f}s")

for check in health.checks:
    print(f"  {check.name}: {check.status.value} - {check.message}")

# Production gate
gate = create_production_gate()
gate.run_checks()

if gate.passes(require_all=True):
    print("✓ Production gate PASSED - Ready for live trading")
else:
    print("✗ Production gate FAILED")
    report = gate.get_report()
    print(f"Failed checks: {report['failed_required']}")

# Check readiness level
if health.readiness == ReadinessLevel.FULL_LIVE:
    print("Full production ready")
elif health.readiness == ReadinessLevel.LIMITED_LIVE:
    print("Limited live trading (small sizes)")
elif health.readiness == ReadinessLevel.PAPER_ONLY:
    print("Paper trading only")
else:
    print("NOT READY - Critical failures")
```

### Readiness Levels

| Level | Description |
|-------|-------------|
| NOT_READY | Critical failures - do not trade |
| PAPER_ONLY | Can paper trade, not live |
| LIMITED_LIVE | Small size live trading |
| FULL_LIVE | Full production ready |

### Health Checks

- **network_connectivity**: Ping critical hosts
- **data_feed_health**: Data source availability
- **model_readiness**: Model files present
- **risk_system_status**: OMS, kill switch, circuit breaker
- **database_connectivity**: Database accessible
- **system_resources**: CPU, memory, disk OK

---

## Integration Example

Here's how to integrate all enhancements into a production trading loop:

```python
from trading.execution_risk import create_execution_pipeline, ExecutionConfig
from trading.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from trading.production_health import create_production_gate, ReadinessLevel
from models.regime_detection import RegimeDetector, AdaptiveRetrainer
from models.confidence_calibration import ConfidenceCalibrator, DynamicThresholdManager
from data.quality_monitor import create_data_quality_pipeline
from analysis.realistic_backtest import create_realistic_backtest_engine

class ProductionTradingSystem:
    def __init__(self):
        # Initialize all components
        self.exec_router, self.exec_validator = create_execution_pipeline()
        self.circuit_breaker = CircuitBreaker(CircuitBreakerConfig())
        self.production_gate = create_production_gate()
        self.regime_detector = RegimeDetector()
        self.retrainer = AdaptiveRetrainer(self.regime_detector)
        self.calibrator = ConfidenceCalibrator()
        self.threshold_mgr = DynamicThresholdManager()
        self.quality_monitor, self.provider_monitor, self.validator = create_data_quality_pipeline()
        self.backtest_engine = create_realistic_backtest_engine()

    def should_trade(self, prediction) -> bool:
        """Check all gates before trading."""
        # 1. Production gate
        self.production_gate.run_checks()
        if not self.production_gate.passes():
            return False

        # 2. Circuit breaker
        if self.circuit_breaker.is_trading_halted:
            return False

        # 3. Data quality
        if not self.quality_monitor.is_data_fresh(prediction.symbol):
            return False

        # 4. Confidence threshold
        should_trade, _ = self.threshold_mgr.should_trade(
            confidence=prediction.calibrated_confidence,
            symbol=prediction.symbol,
            regime=self.regime_detector.detect_regime().regime.value,
        )
        return should_trade

    def execute_trade(self, order, current_price) -> bool:
        """Execute trade with all risk controls."""
        # 1. Pre-trade validation
        is_valid, reason = self.exec_validator.validate_pre_trade(
            order, current_price, quote_age_ms=100
        )
        if not is_valid:
            log.warning(f"Order rejected: {reason}")
            return False

        # 2. Circuit breaker order check
        order_value_pct = (order.quantity * order.price) / self.capital
        if not self.circuit_breaker.record_order(order_value_pct):
            log.warning("Order rejected by circuit breaker")
            return False

        # 3. Slippage estimation
        estimate = self.exec_router.estimate_slippage(
            order.symbol, order.side, order.quantity,
            order.price, daily_volume=1e6, volatility=0.3
        )
        if estimate.worst_case_bps > 30:
            log.warning(f"Slippage too high: {estimate.worst_case_bps:.1f} bps")
            return False

        # 4. Submit order (via broker)
        fill = self.submit_order_to_broker(order)

        # 5. Record execution quality
        if fill:
            self.exec_router.record_execution(
                order, fill, current_price, fill_time_ms=50
            )

        return True

    def run(self):
        """Main trading loop."""
        while True:
            try:
                # Update health
                self.circuit_breaker.update_heartbeat()

                # Get prediction
                prediction = self.get_prediction()

                # Calibrate confidence
                calibrated = create_calibrated_prediction(
                    symbol=prediction.symbol,
                    signal=prediction.signal,
                    raw_confidence=prediction.confidence,
                    current_price=prediction.price,
                    ensemble_predictions=prediction.ensemble_preds,
                    regime=self.regime_detector.detect_regime().regime.value,
                    calibrator=self.calibrator,
                )

                # Check if should trade
                if self.should_trade(calibrated):
                    self.execute_trade(create_order(calibrated), prediction.price)

                # Record for calibration
                self.calibrator.record_prediction(calibrated)

                time.sleep(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                log.exception(f"Trading loop error: {e}")
                self.circuit_breaker.record_data_failure()
```

---

## Summary of Improvements

| Disadvantage | Solution |
|--------------|----------|
| Retail-grade execution | Smart order routing, slippage estimation, execution validation |
| 5% slippage risk | Real-time slippage modeling, worst-case bounds |
| Single point of failure | Redundant health monitoring, circuit breakers |
| Network disruptions | Data quality monitoring, provider health tracking |
| 70% confidence = 30% wrong | Confidence calibration, uncertainty bands |
| Model overfitting | Regime detection, adaptive retraining |
| Backtest ≠ Live | Realistic market impact, transaction cost modeling |
| No compliance | T+1 settlement checks, price limit validation |

## Next Steps

1. **Paper Trading**: Test all enhancements in paper trading mode first
2. **Calibration Period**: Run for 2-4 weeks to collect calibration data
3. **Threshold Tuning**: Adjust thresholds based on observed performance
4. **Gradual Rollout**: Start with small sizes, scale up as confidence grows

## Disclaimer

These enhancements improve robustness but do not guarantee profits or eliminate risk. Always:
- Start with paper trading
- Use appropriate position sizing
- Monitor system health continuously
- Be prepared to intervene manually
