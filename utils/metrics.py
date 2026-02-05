"""
Prometheus-style Metrics Export
"""
import threading
import time
from typing import Dict
from dataclasses import dataclass, field
from datetime import datetime
import json

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Metric:
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsRegistry:
    """Simple metrics registry for observability"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = {}
    
    def inc_counter(self, name: str, value: float = 1.0, labels: Dict = None):
        """Increment a counter"""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value
    
    def set_gauge(self, name: str, value: float, labels: Dict = None):
        """Set a gauge value"""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value
    
    def observe_histogram(self, name: str, value: float, labels: Dict = None):
        """Observe a value for histogram"""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)
            # Keep last 1000 observations
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
    
    def _make_key(self, name: str, labels: Dict = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_all(self) -> Dict:
        """Get all metrics"""
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {k: {'count': len(v), 'sum': sum(v), 'avg': sum(v)/len(v) if v else 0}
                              for k, v in self._histograms.items()}
            }
    
    def to_prometheus(self) -> str:
        """Export in Prometheus format"""
        lines = []
        with self._lock:
            for name, value in self._counters.items():
                lines.append(f"{name} {value}")
            for name, value in self._gauges.items():
                lines.append(f"{name} {value}")
        return "\n".join(lines)


# Global registry
_metrics = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    return _metrics


# Convenience functions
def inc_counter(name: str, value: float = 1.0, labels: Dict = None):
    _metrics.inc_counter(name, value, labels)

def set_gauge(name: str, value: float, labels: Dict = None):
    _metrics.set_gauge(name, value, labels)

def observe(name: str, value: float, labels: Dict = None):
    _metrics.observe_histogram(name, value, labels)