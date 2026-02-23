"""Prometheus metrics exporter for Trading Graph.

Provides Prometheus-compatible metrics for monitoring and alerting.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class MetricSample:
    """A single metric sample."""

    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    help_text: str = ""
    metric_type: str = "gauge"  # gauge, counter, histogram


class MetricsRegistry:
    """Thread-safe metrics registry.

    Example:
        registry = MetricsRegistry()
        registry.gauge("memory_mb", 512.5, labels={"host": "server1"})
        registry.counter("orders_submitted", labels={"side": "buy"})
        print(registry.generate_prometheus_output())
    """

    def __init__(self) -> None:
        self._metrics: dict[str, list[MetricSample]] = {}
        self._lock = threading.RLock()
        self._help_texts: dict[str, str] = {}
        self._metric_types: dict[str, str] = {}

    def _key(self, name: str, labels: dict[str, str]) -> str:
        """Generate unique key for metric + labels."""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}" if label_str else name

    def gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        """Record a gauge metric value.

        Gauges represent a single numerical value that can arbitrarily go up
        and down (e.g., memory usage, temperature).

        Args:
            name: Metric name
            value: Current value
            labels: Optional labels for filtering
            help_text: Description of the metric
        """
        labels = labels or {}
        with self._lock:
            self._help_texts[name] = help_text
            self._metric_types[name] = "gauge"

            if name not in self._metrics:
                self._metrics[name] = []

            key = self._key(name, labels)
            sample = MetricSample(
                name=name,
                value=float(value),
                labels=labels,
                help_text=help_text,
                metric_type="gauge",
            )

            # Replace existing sample with same labels
            self._metrics[name] = [
                s for s in self._metrics[name] if self._key(s.name, s.labels) != key
            ]
            self._metrics[name].append(sample)

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        """Increment a counter metric.

        Counters only go up (e.g., number of requests, errors).

        Args:
            name: Metric name
            value: Amount to increment (default 1.0)
            labels: Optional labels for filtering
            help_text: Description of the metric
        """
        labels = labels or {}
        with self._lock:
            self._help_texts[name] = help_text
            self._metric_types[name] = "counter"

            if name not in self._metrics:
                self._metrics[name] = []

            key = self._key(name, labels)
            existing = [s for s in self._metrics[name] if self._key(s.name, s.labels) == key]

            if existing:
                # Increment existing counter
                current = existing[0]
                new_sample = MetricSample(
                    name=name,
                    value=current.value + float(value),
                    labels=labels,
                    help_text=help_text,
                    metric_type="counter",
                )
                self._metrics[name] = [
                    s for s in self._metrics[name] if self._key(s.name, s.labels) != key
                ]
                self._metrics[name].append(new_sample)
            else:
                # Create new counter
                self._metrics[name].append(
                    MetricSample(
                        name=name,
                        value=float(value),
                        labels=labels,
                        help_text=help_text,
                        metric_type="counter",
                    )
                )

    def histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        buckets: tuple[float, ...] = (
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
        ),
        help_text: str = "",
    ) -> None:
        """Record a histogram metric value.

        Histograms track distribution of values (e.g., request latency).

        Args:
            name: Metric name
            value: Value to record
            labels: Optional labels for filtering
            buckets: Bucket boundaries
            help_text: Description of the metric
        """
        labels = labels or {}
        with self._lock:
            self._help_texts[name] = help_text
            self._metric_types[name] = "histogram"

            # Record bucket counts
            for bucket in buckets:
                bucket_name = f"{name}_bucket"
                bucket_labels = {**labels, "le": str(bucket)}
                if value <= bucket:
                    self.counter(bucket_name, value, bucket_labels, help_text)

            # Record sum and count
            self.counter(f"{name}_sum", value, labels, help_text)
            self.counter(f"{name}_count", 1, labels, help_text)

    def generate_prometheus_output(self) -> str:
        """Generate Prometheus text format output.

        Returns:
            Prometheus-formatted metrics string
        """
        lines: list[str] = []

        with self._lock:
            for name, samples in sorted(self._metrics.items()):
                if not samples:
                    continue

                metric_type = self._metric_types.get(name, "gauge")
                help_text = self._help_texts.get(name, "")

                # Add HELP line
                if help_text:
                    lines.append(f"# HELP {name} {help_text}")

                # Add TYPE line
                lines.append(f"# TYPE {name} {metric_type}")

                # Add samples
                for sample in sorted(samples, key=lambda s: s.timestamp, reverse=True):
                    labels_str = ",".join(f'{k}="{v}"' for k, v in sorted(sample.labels.items()))
                    if labels_str:
                        lines.append(f"{name}{{{labels_str}}} {sample.value}")
                    else:
                        lines.append(f"{name} {sample.value}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()


# Global metrics registry
METRICS_REGISTRY = MetricsRegistry()


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics endpoint."""

    registry: MetricsRegistry = METRICS_REGISTRY

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging."""
        pass

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            output = self.registry.generate_prometheus_output()
            self.wfile.write(output.encode("utf-8"))
        elif self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"healthy"}')
        else:
            self.send_response(404)
            self.end_headers()


def start_metrics_server(
    port: int = 9090,
    host: str = "127.0.0.1",
    registry: MetricsRegistry | None = None,
) -> HTTPServer:
    """Start Prometheus metrics HTTP server.

    Args:
        port: Port to listen on
        host: Host to bind to
        registry: Metrics registry (uses global if None)

    Returns:
        HTTPServer instance

    Example:
        server = start_metrics_server(port=9090)
        try:
            server.serve_forever()
        finally:
            server.shutdown()
    """
    if registry:
        MetricsHandler.registry = registry

    server = HTTPServer((host, port), MetricsHandler)
    log.info("Metrics server started at http://%s:%d/metrics", host, port)
    return server


# Trading-specific metrics collectors


@dataclass
class TradingMetrics:
    """Trading-specific metrics collection."""

    # Order metrics
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    orders_cancelled: int = 0

    # Fill metrics
    total_fill_value: float = 0.0
    total_commission: float = 0.0

    # Performance metrics
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Risk metrics
    current_var: float = 0.0
    max_drawdown: float = 0.0
    position_concentration: float = 0.0

    # Model metrics
    predictions_made: int = 0
    avg_confidence: float = 0.0
    drift_score: float = 0.0

    # Data metrics
    fetch_latency_ms: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    cache_hits: int = 0
    cache_misses: int = 0

    def record_order(self, status: str) -> None:
        """Record an order submission."""
        self.orders_submitted += 1
        if status == "FILLED":
            self.orders_filled += 1
        elif status == "REJECTED":
            self.orders_rejected += 1
        elif status == "CANCELLED":
            self.orders_cancelled += 1

    def record_fill(self, value: float, commission: float) -> None:
        """Record a fill."""
        self.total_fill_value += value
        self.total_commission += commission

    def record_prediction(self, confidence: float, drift: float) -> None:
        """Record a prediction."""
        self.predictions_made += 1
        # Running average
        n = self.predictions_made
        self.avg_confidence = ((self.avg_confidence * (n - 1)) + confidence) / n
        self.drift_score = drift

    def record_fetch(self, latency_ms: float, cache_hit: bool) -> None:
        """Record a data fetch."""
        self.fetch_latency_ms.append(latency_ms)
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def export_to_registry(self, registry: MetricsRegistry) -> None:
        """Export metrics to Prometheus registry."""
        # Order metrics
        registry.counter(
            "trading_orders_submitted_total",
            value=self.orders_submitted,
            help_text="Total number of orders submitted",
        )
        registry.counter(
            "trading_orders_filled_total",
            value=self.orders_filled,
            help_text="Total number of orders filled",
        )
        registry.counter(
            "trading_orders_rejected_total",
            value=self.orders_rejected,
            help_text="Total number of orders rejected",
        )
        registry.counter(
            "trading_orders_cancelled_total",
            value=self.orders_cancelled,
            help_text="Total number of orders cancelled",
        )

        # Fill metrics
        registry.gauge(
            "trading_fill_value_total",
            self.total_fill_value,
            help_text="Total fill value in CNY",
        )
        registry.gauge(
            "trading_commission_total",
            self.total_commission,
            help_text="Total commission paid in CNY",
        )

        # Performance metrics
        registry.gauge(
            "trading_pnl_realized",
            self.realized_pnl,
            help_text="Realized P&L in CNY",
        )
        registry.gauge(
            "trading_pnl_unrealized",
            self.unrealized_pnl,
            help_text="Unrealized P&L in CNY",
        )

        # Risk metrics
        registry.gauge(
            "trading_var_95",
            self.current_var,
            help_text="Current 95% VaR in CNY",
        )
        registry.gauge(
            "trading_max_drawdown",
            self.max_drawdown,
            help_text="Maximum drawdown in CNY",
        )
        registry.gauge(
            "trading_position_concentration",
            self.position_concentration,
            help_text="Largest position as % of portfolio",
        )

        # Model metrics
        registry.counter(
            "ml_predictions_total",
            value=self.predictions_made,
            help_text="Total predictions made",
        )
        registry.gauge(
            "ml_avg_confidence",
            self.avg_confidence,
            help_text="Average prediction confidence",
        )
        registry.gauge(
            "ml_drift_score",
            self.drift_score,
            help_text="Current model drift score",
        )

        # Data metrics
        if self.fetch_latency_ms:
            avg_latency = sum(self.fetch_latency_ms) / len(self.fetch_latency_ms)
            registry.gauge(
                "data_fetch_latency_ms",
                avg_latency,
                help_text="Average fetch latency in milliseconds",
            )
        cache_total = self.cache_hits + self.cache_misses
        if cache_total > 0:
            hit_rate = self.cache_hits / cache_total
            registry.gauge(
                "data_cache_hit_rate",
                hit_rate,
                help_text="Cache hit rate (0-1)",
            )


# Global trading metrics instance
TRADING_METRICS = TradingMetrics()


def get_trading_metrics() -> TradingMetrics:
    """Get global trading metrics instance."""
    return TRADING_METRICS


def export_trading_metrics() -> None:
    """Export trading metrics to Prometheus registry."""
    TRADING_METRICS.export_to_registry(METRICS_REGISTRY)


# Background metrics exporter


class MetricsExporter:
    """Background task to periodically export metrics."""

    def __init__(
        self,
        interval_seconds: float = 60.0,
        registry: MetricsRegistry | None = None,
    ) -> None:
        self.interval = interval_seconds
        self.registry = registry or METRICS_REGISTRY
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _export_loop(self) -> None:
        """Main export loop."""
        while not self._stop_event.is_set():
            try:
                export_trading_metrics()
            except Exception as e:
                log.error("Metrics export failed: %s", e)

            # Wait for next interval
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        """Start background exporter."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._export_loop,
            daemon=True,
            name="metrics-exporter",
        )
        self._thread.start()
        log.info("Metrics exporter started (interval: %.1fs)", self.interval)

    def stop(self) -> None:
        """Stop background exporter."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            log.info("Metrics exporter stopped")


# Convenience functions for common metrics


def record_order(status: str) -> None:
    """Record an order submission."""
    TRADING_METRICS.record_order(status)


def record_fill(value: float, commission: float) -> None:
    """Record a fill."""
    TRADING_METRICS.record_fill(value, commission)


def record_prediction(confidence: float, drift: float) -> None:
    """Record a prediction."""
    TRADING_METRICS.record_prediction(confidence, drift)


def record_fetch(latency_ms: float, cache_hit: bool) -> None:
    """Record a data fetch."""
    TRADING_METRICS.record_fetch(latency_ms, cache_hit)


def update_pnl(realized: float, unrealized: float) -> None:
    """Update P&L metrics."""
    TRADING_METRICS.realized_pnl = realized
    TRADING_METRICS.unrealized_pnl = unrealized


def update_risk(var: float, drawdown: float, concentration: float) -> None:
    """Update risk metrics."""
    TRADING_METRICS.current_var = var
    TRADING_METRICS.max_drawdown = drawdown
    TRADING_METRICS.position_concentration = concentration
