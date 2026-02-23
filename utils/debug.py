"""
Debugging and profiling utilities for Trading Graph.

This module provides comprehensive debugging, profiling, and diagnostic tools
for performance analysis and troubleshooting.
"""
from __future__ import annotations

import cProfile
import functools
import linecache
import os
import pstats
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ProfileResult:
    """Results from profiling a function."""

    func_name: str
    total_time: float
    call_count: int
    time_per_call: float
    top_callers: list[tuple[str, float]] = field(default_factory=list)
    memory_allocated_mb: float = 0.0

    def __str__(self) -> str:
        return (
            f"ProfileResult({self.func_name}): "
            f"{self.call_count} calls, "
            f"{self.total_time*1000:.2f}ms total, "
            f"{self.time_per_call*1000:.3f}ms/call, "
            f"{self.memory_allocated_mb:.2f}MB"
        )


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    timestamp: datetime
    current_mb: float
    peak_mb: float
    top_allocations: list[tuple[str, int]] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"MemorySnapshot: {self.current_mb:.1f}MB current, "
            f"{self.peak_mb:.1f}MB peak"
        )


def profile_function(
    enable_memory: bool = False,
    top_n: int = 10,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to profile function execution.

    Args:
        enable_memory: Whether to track memory allocation
        top_n: Number of top callers to include in results

    Returns:
        Decorated function with profiling

    Example:
        @profile_function(enable_memory=True)
        def expensive_operation(data: list) -> list:
            return sorted(data)
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Start profiling
            profiler = cProfile.Profile()
            profiler.enable()

            # Start memory tracking if enabled
            if enable_memory:
                tracemalloc.start()
                start_snapshot = tracemalloc.take_snapshot()

            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
            finally:
                # Stop profiling
                profiler.disable()
                elapsed = time.perf_counter() - start_time

                # Process results
                stream = StringIO()
                stats = pstats.Stats(profiler, stream=stream)
                stats.sort_stats("cumulative")
                stats.print_stats(top_n)

                # Extract top callers - pstats format: (ncalls, tottime, cumtime, callers_dict)
                top_callers: list[tuple[str, float]] = []
                for func_key, stats_tuple in list(stats.stats.items())[:top_n]:
                    if isinstance(stats_tuple, tuple) and len(stats_tuple) >= 3:
                        tottime_val = stats_tuple[2] if isinstance(stats_tuple[2], (int, float)) else 0.0
                        top_callers.append((str(func_key), float(tottime_val)))

                # Memory tracking
                memory_mb = 0.0
                if enable_memory and start_snapshot:
                    end_snapshot = tracemalloc.take_snapshot()
                    top_stats = end_snapshot.compare_to(
                        start_snapshot, "lineno"
                    )
                    memory_mb = sum(stat.size_diff for stat in top_stats[:10]) / (
                        1024 * 1024
                    )
                    tracemalloc.stop()

                # Log results - use first caller's ncalls if available
                call_count = 1
                if top_callers:
                    # Try to get call count from stats
                    for func_key, stats_tuple in list(stats.stats.items())[:1]:
                        if isinstance(stats_tuple, tuple) and len(stats_tuple) >= 1:
                            call_count = int(stats_tuple[0])
                            break
                
                result_obj = ProfileResult(
                    func_name=func.__name__,
                    total_time=elapsed,
                    call_count=call_count,
                    time_per_call=elapsed / max(1, call_count),
                    top_callers=top_callers,
                    memory_allocated_mb=memory_mb,
                )

                log.info("Profile: %s", result_obj)
                if top_callers:
                    log.debug("Top callers:")
                    for name, t in top_callers[:5]:
                        log.debug("  %s: %.3fms", name, t * 1000)

            return result

        return wrapper

    return decorator


@contextmanager
def profile_context(
    name: str = "block",
    enable_memory: bool = False,
    log_level: str = "info",
):
    """
    Context manager for profiling code blocks.

    Args:
        name: Name for this profiling block
        enable_memory: Whether to track memory allocation
        log_level: Logging level for results

    Yields:
        None

    Example:
        with profile_context("data_loading", enable_memory=True):
            data = load_large_dataset()
    """
    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # Start memory tracking
    start_memory = 0.0
    if enable_memory:
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]

    start_time = time.perf_counter()

    try:
        yield
    finally:
        # Calculate elapsed time
        elapsed = time.perf_counter() - start_time

        # Stop profiling
        profiler.disable()

        # Calculate memory
        memory_diff = 0.0
        if enable_memory:
            current, peak = tracemalloc.get_traced_memory()
            memory_diff = (current - start_memory) / (1024 * 1024)
            tracemalloc.stop()

        # Log results
        log_func = getattr(log, log_level.lower(), log.info)
        log_func(
            "Profile[%s]: %.3fms, memory: %.2fMB",
            name,
            elapsed * 1000,
            memory_diff,
        )


def get_memory_snapshot(top_n: int = 10) -> MemorySnapshot:
    """
    Take a snapshot of current memory usage.

    Args:
        top_n: Number of top allocations to include

    Returns:
        MemorySnapshot with current usage details
    """
    current, peak = tracemalloc.get_traced_memory()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")[:top_n]

    top_allocations: list[tuple[str, int]] = []
    for stat in top_stats:
        frame = stat.traceback[0]
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            top_allocations.append((f"{frame.filename}:{frame.lineno}", stat.size))

    return MemorySnapshot(
        timestamp=datetime.now(),
        current_mb=current / (1024 * 1024),
        peak_mb=peak / (1024 * 1024),
        top_allocations=top_allocations,
    )


@contextmanager
def memory_tracker(label: str = "block"):
    """
    Context manager for tracking memory changes.

    Args:
        label: Label for this tracking block

    Yields:
        None

    Example:
        with memory_tracker("data_processing"):
            process_large_data()
    """
    tracemalloc.start()
    start_current, start_peak = tracemalloc.get_traced_memory()

    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")[:5]

        log.info(
            "Memory[%s]: %.1fMB → %.1fMB (Δ%.1fMB), peak: %.1fMB",
            label,
            start_current / (1024 * 1024),
            current / (1024 * 1024),
            (current - start_current) / (1024 * 1024),
            peak / (1024 * 1024),
        )

        if top_stats:
            log.debug("Top allocations:")
            for stat in top_stats:
                log.debug("  %s: %.1fKB", stat.traceback[0], stat.size / 1024)

        tracemalloc.stop()


def trace_calls(frame: Any, event: str, arg: Any) -> Any:
    """
    Trace function call entries and exits.

    Use with sys.settrace() for detailed call tracing.

    Example:
        sys.settrace(trace_calls)
        # ... code to trace ...
        sys.settrace(None)
    """
    if event == "call":
        code = frame.f_code
        log.debug(
            "CALL: %s:%s (%s)",
            code.co_filename,
            code.co_name,
            frame.f_lineno,
        )
    elif event == "return":
        code = frame.f_code
        log.debug("RETURN: %s:%s", code.co_filename, code.co_name)
    return trace_calls


@contextmanager
def trace_execution(
    log_file: Path | None = None,
    trace_returns: bool = False,
):
    """
    Context manager for tracing function calls.

    Args:
        log_file: Optional file to write trace logs
        trace_returns: Whether to trace return values

    Yields:
        None

    Example:
        with trace_execution(log_file=Path("trace.log")):
            run_complex_operation()
    """
    old_trace = sys.gettrace()

    # Setup log file if provided
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = log.FileHandler(log_file)
        log.getLogger().addHandler(handler)

    try:
        sys.settrace(trace_calls)
        yield
    finally:
        sys.settrace(old_trace)

        # Cleanup log file handler
        if log_file:
            for handler in log.getLogger().handlers[:]:
                if isinstance(handler, log.FileHandler):
                    handler.close()
                    log.getLogger().removeHandler(handler)


@dataclass
class TimingStats:
    """Statistics for timing measurements."""

    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = 0.0
    last: float = 0.0

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)

    def record(self, elapsed: float) -> None:
        """Record a timing measurement."""
        self.count += 1
        self.total += elapsed
        self.min = min(self.min, elapsed)
        self.max = max(self.max, elapsed)
        self.last = elapsed

    def __str__(self) -> str:
        return (
            f"TimingStats(count={self.count}, avg={self.avg*1000:.2f}ms, "
            f"min={self.min*1000:.2f}ms, max={self.max*1000:.2f}ms)"
        )


class TimingContext:
    """Context manager for timing code blocks with statistics."""

    _stats: dict[str, TimingStats] = {}

    def __init__(self, name: str, log_results: bool = True):
        self.name = name
        self.log_results = log_results
        self.elapsed = 0.0

    def __enter__(self) -> "TimingContext":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self.start

        # Update stats
        if self.name not in self._stats:
            self._stats[self.name] = TimingStats()
        self._stats[self.name].record(self.elapsed)

        # Log if requested
        if self.log_results:
            log.debug(
                "Timing[%s]: %.3fms (avg: %.3fms, count: %d)",
                self.name,
                self.elapsed * 1000,
                self._stats[self.name].avg * 1000,
                self._stats[self.name].count,
            )

    @classmethod
    def get_stats(cls, name: str) -> TimingStats | None:
        """Get timing statistics for a named block."""
        return cls._stats.get(name)

    @classmethod
    def print_all_stats(cls) -> None:
        """Print all collected timing statistics."""
        if not cls._stats:
            log.info("No timing stats collected")
            return

        log.info("Timing Statistics:")
        log.info("%-30s %10s %10s %10s %10s", "Name", "Count", "Avg(ms)", "Min(ms)", "Max(ms)")
        log.info("-" * 80)
        for name, stats in sorted(cls._stats.items(), key=lambda x: x[1].total, reverse=True):
            log.info(
                "%-30s %10d %10.3f %10.3f %10.3f",
                name,
                stats.count,
                stats.avg * 1000,
                stats.min * 1000,
                stats.max * 1000,
            )


def slow_call_threshold(threshold_ms: float = 100.0) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to log calls that exceed a time threshold.

    Args:
        threshold_ms: Threshold in milliseconds

    Returns:
        Decorated function

    Example:
        @slow_call_threshold(threshold_ms=50.0)
        def potentially_slow_operation():
            pass
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                if elapsed_ms > threshold_ms:
                    log.warning(
                        "Slow call: %s took %.1fms (threshold: %.1fms)",
                        func.__name__,
                        elapsed_ms,
                        threshold_ms,
                    )

        return wrapper

    return decorator


def diagnose_performance() -> dict[str, Any]:
    """
    Run performance diagnostics and return results.

    Returns:
        Dictionary with diagnostic results

    Example:
        results = diagnose_performance()
        print(f"Memory: {results['memory_mb']}MB")
        print(f"Threads: {results['thread_count']}")
    """
    import threading

    import psutil

    process = psutil.Process(os.getpid())

    # Memory
    mem_info = process.memory_info()
    memory_mb = mem_info.rss / (1024 * 1024)

    # CPU
    cpu_percent = process.cpu_percent(interval=0.1)

    # Threads
    thread_count = threading.active_count()

    # File descriptors (Unix only)
    try:
        fd_count = process.num_fds()
    except (AttributeError, psutil.AccessDenied):
        fd_count = 0

    return {
        "timestamp": datetime.now().isoformat(),
        "memory_mb": memory_mb,
        "memory_percent": process.memory_percent(),
        "cpu_percent": cpu_percent,
        "thread_count": thread_count,
        "fd_count": fd_count,
        "pid": process.pid,
    }


def print_performance_report() -> None:
    """Print a formatted performance report to logs."""
    results = diagnose_performance()

    log.info("=" * 60)
    log.info("PERFORMANCE REPORT")
    log.info("=" * 60)
    log.info("Timestamp: %s", results["timestamp"])
    log.info("PID: %d", results["pid"])
    log.info("-" * 60)
    log.info("Memory: %.1f MB (%.1f%%)", results["memory_mb"], results["memory_percent"])
    log.info("CPU: %.1f%%", results["cpu_percent"])
    log.info("Threads: %d", results["thread_count"])
    log.info("File Descriptors: %d", results["fd_count"])
    log.info("=" * 60)

    # Print timing stats if any
    TimingContext.print_all_stats()
