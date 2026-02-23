"""Tests for Prometheus metrics exporter."""
import pytest
from utils.metrics_prometheus import (
    MetricsRegistry, MetricSample,
    TradingMetrics, TRADING_METRICS,
    record_order, record_fill, record_prediction,
    get_trading_metrics,
)


def test_metrics_registry_gauge():
    """Test gauge metric recording."""
    registry = MetricsRegistry()
    registry.gauge("test_gauge", 42.0, labels={"host": "server1"}, help_text="Test gauge")
    
    output = registry.generate_prometheus_output()
    assert "# HELP test_gauge Test gauge" in output
    assert "# TYPE test_gauge gauge" in output
    assert 'test_gauge{host="server1"} 42.0' in output


def test_metrics_registry_counter():
    """Test counter metric recording."""
    registry = MetricsRegistry()
    registry.counter("test_counter", 1.0, labels={"method": "GET"}, help_text="Test counter")
    registry.counter("test_counter", 1.0, labels={"method": "GET"})
    
    output = registry.generate_prometheus_output()
    # HELP text is only added on first registration, check TYPE and value
    assert "# TYPE test_counter counter" in output
    assert 'test_counter{method="GET"} 2.0' in output


def test_metrics_registry_clear():
    """Test registry clear."""
    registry = MetricsRegistry()
    registry.gauge("test", 1.0)
    registry.clear()
    
    output = registry.generate_prometheus_output()
    assert output == ""


def test_trading_metrics_record_order():
    """Test TradingMetrics order recording."""
    metrics = TradingMetrics()
    assert metrics.orders_submitted == 0
    
    metrics.record_order("FILLED")
    assert metrics.orders_submitted == 1
    assert metrics.orders_filled == 1
    
    metrics.record_order("REJECTED")
    assert metrics.orders_submitted == 2
    assert metrics.orders_rejected == 1


def test_trading_metrics_record_fill():
    """Test TradingMetrics fill recording."""
    metrics = TradingMetrics()
    metrics.record_fill(1000.0, 3.0)
    
    assert metrics.total_fill_value == 1000.0
    assert metrics.total_commission == 3.0


def test_trading_metrics_record_prediction():
    """Test TradingMetrics prediction recording."""
    metrics = TradingMetrics()
    metrics.record_prediction(0.8, 0.1)
    
    assert metrics.predictions_made == 1
    assert metrics.avg_confidence == 0.8


def test_trading_metrics_record_fetch():
    """Test TradingMetrics fetch recording."""
    metrics = TradingMetrics()
    metrics.record_fetch(50.0, True)
    metrics.record_fetch(60.0, False)
    
    assert len(metrics.fetch_latency_ms) == 2
    assert metrics.cache_hits == 1
    assert metrics.cache_misses == 1


def test_trading_metrics_export():
    """Test TradingMetrics export to registry."""
    metrics = TradingMetrics()
    metrics.record_order("FILLED")
    metrics.record_fill(1000.0, 3.0)
    
    registry = MetricsRegistry()
    metrics.export_to_registry(registry)
    
    output = registry.generate_prometheus_output()
    assert "trading_orders_submitted_total" in output
    assert "trading_fill_value_total" in output


def test_get_trading_metrics():
    """Test get_trading_metrics function."""
    metrics = get_trading_metrics()
    assert isinstance(metrics, TradingMetrics)


def test_convenience_functions():
    """Test convenience functions."""
    record_order("FILLED")
    record_fill(500.0, 1.5)
    record_prediction(0.7, 0.05)
    
    metrics = get_trading_metrics()
    assert metrics.orders_filled >= 1


def test_metric_sample():
    """Test MetricSample dataclass."""
    sample = MetricSample(
        name="test",
        value=100.0,
        labels={"env": "prod"},
        help_text="Test metric",
        metric_type="gauge",
    )
    assert sample.name == "test"
    assert sample.value == 100.0
    assert sample.labels == {"env": "prod"}
    assert sample.help_text == "Test metric"
    assert sample.metric_type == "gauge"


def test_metrics_registry_multiple_labels():
    """Test registry with multiple label combinations."""
    registry = MetricsRegistry()
    registry.gauge("multi_label", 1.0, labels={"a": "1", "b": "2"})
    registry.gauge("multi_label", 2.0, labels={"a": "1", "b": "3"})
    
    output = registry.generate_prometheus_output()
    assert 'multi_label{a="1",b="2"} 1.0' in output
    assert 'multi_label{a="1",b="3"} 2.0' in output


def test_trading_metrics_update_pnl():
    """Test P&L update."""
    metrics = get_trading_metrics()
    metrics.realized_pnl = 1000.0
    metrics.unrealized_pnl = 500.0
    
    assert metrics.realized_pnl == 1000.0
    assert metrics.unrealized_pnl == 500.0


def test_trading_metrics_update_risk():
    """Test risk metrics update."""
    metrics = get_trading_metrics()
    metrics.current_var = 5000.0
    metrics.max_drawdown = 2000.0
    metrics.position_concentration = 0.25
    
    assert metrics.current_var == 5000.0
    assert metrics.max_drawdown == 2000.0
    assert metrics.position_concentration == 0.25
