# utils/metrics.py
from __future__ import annotations

import re
import threading
import time
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)

_VALID_METRIC_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_VALID_LABEL_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Default histogram buckets (latency-oriented)
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)

def _sanitize_name(name: str) -> str:
    """
    FIX #2: Sanitize metric name for Prometheus compatibility.
    Replaces invalid characters with underscores.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_:]", "_", name)
    # Ensure doesn't start with digit
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    if not sanitized:
        sanitized = "_unnamed"
    return sanitized

def _validate_labels(labels: dict[str, str] | None) -> None:
    """Validate label names and values."""
    if labels is None:
        return
    for key, value in labels.items():
        if not _VALID_LABEL_NAME.match(key):
            raise ValueError(
                f"Invalid label name {key!r} — must match {_VALID_LABEL_NAME.pattern}"
            )
        if not isinstance(value, str):
            raise TypeError(f"Label value for {key!r} must be a string, got {type(value).__name__}")

class MetricsRegistry:
    """
    Prometheus-compatible metrics registry.

    Supports counters, gauges, and histograms with proper exposition format.
    """

    def __init__(self, max_keys: int = 10000) -> None:
        self._lock = threading.RLock()
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._histogram_buckets: dict[str, tuple[float, ...]] = {}
        self._max_keys = max_keys
        self._max_observations = 1000

        self._metric_types: dict[str, str] = {}
        self._metric_help: dict[str, str] = {}

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        """Increment a counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented (value >= 0)")

        _validate_labels(labels)
        name = _sanitize_name(name)
        key = self._make_key(name, labels)

        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value
            self._metric_types[name] = "counter"
            if help_text:
                self._metric_help[name] = help_text
            self._check_key_limit()

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        """Set a gauge value."""
        _validate_labels(labels)
        name = _sanitize_name(name)
        key = self._make_key(name, labels)

        with self._lock:
            self._gauges[key] = value
            self._metric_types[name] = "gauge"
            if help_text:
                self._metric_help[name] = help_text
            self._check_key_limit()

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        buckets: tuple[float, ...] | None = None,
        help_text: str = "",
    ) -> None:
        """Observe a value for histogram."""
        _validate_labels(labels)
        name = _sanitize_name(name)
        key = self._make_key(name, labels)

        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
                self._histogram_buckets[key] = buckets or DEFAULT_BUCKETS

            self._histograms[key].append(value)

            if len(self._histograms[key]) > self._max_observations:
                self._histograms[key] = self._histograms[key][
                    -self._max_observations:
                ]

            self._metric_types[name] = "histogram"
            if help_text:
                self._metric_help[name] = help_text
            self._check_key_limit()

    def _make_key(self, name: str, labels: dict[str, str] | None = None) -> str:
        """Build Prometheus-format key: metric_name{label="value",...}"""
        if not labels:
            return name
        label_str = ",".join(
            f'{k}="{v}"' for k, v in sorted(labels.items())
        )
        return f"{name}{{{label_str}}}"

    def _parse_key(self, key: str) -> tuple[str, str]:
        """Split key into (base_name, labels_string)."""
        if "{" in key:
            idx = key.index("{")
            return key[:idx], key[idx:]
        return key, ""

    def _check_key_limit(self) -> None:
        """FIX #9: Evict oldest keys if we exceed max_keys."""
        total = len(self._counters) + len(self._gauges) + len(self._histograms)
        if total <= self._max_keys:
            return

        # Evict from histograms first (they use the most memory)
        while (
            self._histograms
            and len(self._counters) + len(self._gauges) + len(self._histograms)
            > self._max_keys
        ):
            oldest_key = next(iter(self._histograms))
            del self._histograms[oldest_key]
            self._histogram_buckets.pop(oldest_key, None)
            log.debug("Evicted histogram key: %s", oldest_key)

    def reset(self) -> None:
        """FIX #7: Clear all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._histogram_buckets.clear()
            self._metric_types.clear()
            self._metric_help.clear()

    def remove(self, name: str, labels: dict[str, str] | None = None) -> None:
        """FIX #7: Remove a specific metric."""
        name = _sanitize_name(name)
        key = self._make_key(name, labels)
        with self._lock:
            self._counters.pop(key, None)
            self._gauges.pop(key, None)
            self._histograms.pop(key, None)
            self._histogram_buckets.pop(key, None)

    def get_all(self) -> dict[str, Any]:
        """Get all metrics as a dict."""
        with self._lock:
            histogram_summary: dict[str, dict[str, float | int]] = {}
            result: dict[str, Any] = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": histogram_summary,
            }
            for k, v in self._histograms.items():
                histogram_summary[k] = {
                    "count": len(v),
                    "sum": sum(v),
                    "avg": sum(v) / len(v) if v else 0,
                }
            return result

    def to_prometheus(self) -> str:
        """
        FIX #3, #4: Export in valid Prometheus exposition format.
        Includes # TYPE and # HELP annotations.
        Histograms emit proper _bucket, _count, _sum lines.
        """
        lines: list[str] = []
        emitted_types: set = set()

        with self._lock:
            # --- Counters ---
            for key, value in sorted(self._counters.items()):
                base, _ = self._parse_key(key)
                if base not in emitted_types:
                    help_text = self._metric_help.get(base, "")
                    if help_text:
                        lines.append(f"# HELP {base} {help_text}")
                    lines.append(f"# TYPE {base} counter")
                    emitted_types.add(base)
                lines.append(f"{key} {float(value)}")

            # --- Gauges ---
            for key, value in sorted(self._gauges.items()):
                base, _ = self._parse_key(key)
                if base not in emitted_types:
                    help_text = self._metric_help.get(base, "")
                    if help_text:
                        lines.append(f"# HELP {base} {help_text}")
                    lines.append(f"# TYPE {base} gauge")
                    emitted_types.add(base)
                lines.append(f"{key} {float(value)}")

            # --- Histograms ---
            for key, obs in sorted(self._histograms.items()):
                base, labels = self._parse_key(key)
                buckets = self._histogram_buckets.get(key, DEFAULT_BUCKETS)

                if base not in emitted_types:
                    help_text = self._metric_help.get(base, "")
                    if help_text:
                        lines.append(f"# HELP {base} {help_text}")
                    lines.append(f"# TYPE {base} histogram")
                    emitted_types.add(base)

                count = len(obs)
                total = float(sum(obs)) if obs else 0.0

                # FIX #4: Emit proper _bucket lines
                # Strip existing labels for re-composition
                existing_labels = ""
                if labels:
                    # Remove surrounding { }
                    existing_labels = labels[1:-1]

                for bound in sorted(buckets):
                    bucket_count = sum(1 for o in obs if o <= bound)
                    if existing_labels:
                        label_str = f"{{{existing_labels},le=\"{bound}\"}}"
                    else:
                        label_str = f"{{le=\"{bound}\"}}"
                    lines.append(f"{base}_bucket{label_str} {float(bucket_count)}")

                # +Inf bucket
                if existing_labels:
                    inf_label = f"{{{existing_labels},le=\"+Inf\"}}"
                else:
                    inf_label = '{le="+Inf"}'
                lines.append(f"{base}_bucket{inf_label} {float(count)}")

                # _count and _sum
                lines.append(f"{base}_count{labels} {float(count)}")
                lines.append(f"{base}_sum{labels} {float(total)}")

        return "\n".join(lines) + ("\n" if lines else "")

_metrics = MetricsRegistry()

def get_metrics() -> MetricsRegistry:
    """Get the global metrics registry."""
    return _metrics

def inc_counter(
    name: str,
    value: float = 1.0,
    labels: dict[str, str] | None = None,
) -> None:
    _metrics.inc_counter(name, value, labels)

def set_gauge(
    name: str,
    value: float,
    labels: dict[str, str] | None = None,
    help_text: str = "",
) -> None:
    _metrics.set_gauge(name, value, labels, help_text=help_text)

def observe(
    name: str,
    value: float,
    labels: dict[str, str] | None = None,
) -> None:
    _metrics.observe_histogram(name, value, labels)

_process_metrics_lock = threading.Lock()
_process_metrics_started = False

def start_process_metrics(interval_seconds: float = 5.0) -> None:
    """
    Background gauges: cpu/mem/threads. Safe to call multiple times.

    FIX #5: Thread-safe guard.
    FIX #6: Warmup cpu_percent on first call.
    """
    global _process_metrics_started

    # FIX #5: Proper lock
    with _process_metrics_lock:
        if _process_metrics_started:
            return
        _process_metrics_started = True

    try:
        import os

        import psutil
    except ImportError:
        log.debug("psutil not available — process metrics disabled")
        return

    proc = psutil.Process(os.getpid())

    # FIX #6: First call always returns 0.0 — discard it
    try:
        proc.cpu_percent(interval=None)
    except Exception:
        pass

    def loop() -> None:
        while True:
            try:
                cpu = float(proc.cpu_percent(interval=None))
                set_gauge(
                    "process_cpu_percent", cpu,
                    help_text="Process CPU usage percent",
                )
                mem = proc.memory_info()
                set_gauge(
                    "process_memory_rss_bytes",
                    float(mem.rss),
                    help_text="Process resident set size in bytes",
                )
                set_gauge(
                    "process_threads",
                    float(proc.num_threads()),
                    help_text="Number of process threads",
                )
            except Exception as e:
                log.debug("Process metrics collection failed: %s", e)
            time.sleep(float(interval_seconds))

    t = threading.Thread(target=loop, daemon=True, name="process_metrics")
    t.start()
