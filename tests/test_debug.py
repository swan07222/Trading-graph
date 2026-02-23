"""Tests for debugging and profiling utilities."""
import time

import pytest

from utils.debug import (
    TimingContext,
    TimingStats,
    diagnose_performance,
    memory_tracker,
    profile_context,
    profile_function,
    slow_call_threshold,
)


def test_timing_stats():
    """Test TimingStats class."""
    stats = TimingStats()
    assert stats.count == 0
    assert stats.total == 0.0
    assert stats.min == float("inf")
    assert stats.max == 0.0
    
    stats.record(0.1)
    assert stats.count == 1
    assert stats.total == pytest.approx(0.1, rel=1e-9)
    assert stats.min == 0.1
    assert stats.max == 0.1
    assert stats.avg == pytest.approx(0.1, rel=1e-9)
    
    stats.record(0.2)
    assert stats.count == 2
    assert stats.total == pytest.approx(0.3, rel=1e-9)
    assert stats.avg == pytest.approx(0.15, rel=1e-9)


def test_timing_context():
    """Test TimingContext."""
    with TimingContext("test_block", log_results=False) as ctx:
        time.sleep(0.01)
    
    assert ctx.elapsed >= 0.01
    stats = TimingContext.get_stats("test_block")
    assert stats is not None
    assert stats.count >= 1


def test_timing_context_multiple():
    """Test TimingContext with multiple uses."""
    for _ in range(3):
        with TimingContext("multi_test", log_results=False):
            time.sleep(0.001)
    
    stats = TimingContext.get_stats("multi_test")
    assert stats is not None
    assert stats.count == 3


def test_diagnose_performance():
    """Test diagnose_performance."""
    results = diagnose_performance()
    assert "timestamp" in results
    assert "memory_mb" in results
    assert "cpu_percent" in results
    assert "thread_count" in results
    assert results["memory_mb"] > 0
    assert results["thread_count"] >= 1


def test_profile_function():
    """Test profile_function decorator."""
    @profile_function(enable_memory=False)
    def test_func():
        time.sleep(0.01)
        return 42
    
    # Note: This test exercises the decorator but profiling output format varies by platform
    result = test_func()
    assert result == 42


@pytest.mark.skip(reason="Profile output format varies by platform")
def test_profile_function_detailed():
    """Test profile_function decorator detailed output."""
    # Detailed profiling test skipped due to platform variations
    pass


def test_profile_context():
    """Test profile_context."""
    with profile_context("test_profile", enable_memory=False, log_level="debug"):
        time.sleep(0.01)


def test_slow_call_threshold():
    """Test slow_call_threshold decorator."""
    call_count = [0]
    
    @slow_call_threshold(threshold_ms=100.0)
    def slow_func():
        call_count[0] += 1
        time.sleep(0.01)
        return "done"
    
    result = slow_func()
    assert result == "done"
    assert call_count[0] == 1


def test_memory_tracker():
    """Test memory_tracker context."""
    with memory_tracker("test_memory"):
        data = list(range(1000))
        assert len(data) == 1000


def test_timing_stats_str():
    """Test TimingStats string representation."""
    stats = TimingStats()
    stats.record(0.1)
    stats.record(0.2)
    stats.record(0.3)
    
    str_repr = str(stats)
    assert "count=3" in str_repr
    assert "avg=" in str_repr
