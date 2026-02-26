# core/network.py
"""China-only network detection module.

This module provides network environment detection optimized for mainland China users.
All data sources are China-accessible endpoints (no VPN required).
"""

import inspect
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import requests

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class NetworkEnv:
    """Current network environment snapshot for China-only mode."""
    is_china_direct: bool = True  # Always True for China-only mode
    
    # Endpoint reachability (cached) - China-accessible endpoints only
    eastmoney_ok: bool = False  # AkShare backend
    tencent_ok: bool = False  # Tencent quotes
    baidu_ok: bool = False  # Baidu accessibility (China network indicator)
    sina_ok: bool = False  # Sina Finance accessibility
    
    detected_at: datetime | None = None
    detection_method: str = ""
    latency_ms: float = 0.0


class NetworkDetector:
    """Singleton that probes China-accessible endpoints.

    Results are cached for `ttl_seconds` to avoid repeated probing.
    Thread-safe.
    """

    _instance = None
    _lock = threading.Lock()
    _initialized: bool

    def __new__(cls) -> "NetworkDetector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._env: NetworkEnv | None = None
        self._env_time: float = 0.0
        self._ttl: float = 120.0  # Re-detect every 2 minutes
        self._probe_lock = threading.Lock()

    def _has_fresh_cache_unlocked(self, now: float | None = None) -> bool:
        """Check cache freshness. Caller should hold _probe_lock."""
        if self._env is None:
            return False
        t_now = float(time.time() if now is None else now)
        return (t_now - float(self._env_time)) < float(self._ttl)

    def get_env(self, force_refresh: bool = False) -> NetworkEnv:
        """Get current network environment (cached)."""
        with self._probe_lock:
            if not force_refresh and self._has_fresh_cache_unlocked():
                return self._env  # type: ignore[return-value]
            prev_env = self._env

        env = self._run_detect(prev_env)

        with self._probe_lock:
            if force_refresh:
                self._env = env
                self._env_time = time.time()
                return self._env  # type: ignore[return-value]

            if not self._has_fresh_cache_unlocked():
                self._env = env
                self._env_time = time.time()
            return self._env  # type: ignore[return-value]

    def _run_detect(self, prev_env: NetworkEnv | None) -> NetworkEnv:
        """Call detector while supporting legacy/monkeypatched zero-arg callables."""
        detect_fn = self._detect
        try:
            n_params = len(inspect.signature(detect_fn).parameters)
        except (TypeError, ValueError):
            n_params = 1
        if n_params <= 0:
            return detect_fn()  # type: ignore[misc]
        return detect_fn(prev_env)

    def invalidate(self) -> None:
        """Force re-detection on next call."""
        with self._probe_lock:
            self._env = None
            self._env_time = 0.0

    def peek_env(self) -> NetworkEnv | None:
        """Return cached environment without probing."""
        with self._probe_lock:
            if not self._has_fresh_cache_unlocked():
                return None
            return self._env

    def _detect(self, prev_env: NetworkEnv | None = None) -> NetworkEnv:
        """Probe China-accessible endpoints concurrently.
        
        FIX China Network 2026-02-26:
        - Only China-accessible endpoints (EastMoney, Tencent, Baidu, Sina)
        - Reduced timeouts for faster detection
        - Always assumes China direct mode
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        env = NetworkEnv(detected_at=datetime.now())
        start = time.time()

        # China-accessible endpoints only
        probes = {
            "tencent_ok": ("https://qt.gtimg.cn/q=sh600519", 4),
            "eastmoney_ok": (
                "https://82.push2.eastmoney.com/api/qt/clist/get"
                "?pn=1&pz=1&fields=f2&fid=f3&fs=m:0+t:6",
                4,
            ),
            "baidu_ok": ("https://www.baidu.com", 4),
            "sina_ok": ("https://finance.sina.com.cn", 4),
        }

        default_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        def run_probe(url: str, timeout: float | int) -> bool:
            try:
                effective_timeout = min(timeout, 4.0)
                r = requests.get(url, timeout=effective_timeout, headers=default_headers)
                return r.status_code == 200
            except (requests.Timeout, requests.ConnectionError):
                return False
            except Exception:
                return False

        OVERALL_TIMEOUT = 12
        futures = []
        try:
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="network_probe") as ex:
                fut_map = {
                    ex.submit(run_probe, url, to): k
                    for k, (url, to) in probes.items()
                }
                futures = list(fut_map.keys())
                for fut in as_completed(fut_map, timeout=OVERALL_TIMEOUT):
                    k = fut_map[fut]
                    try:
                        setattr(env, k, bool(fut.result()))
                    except Exception:
                        setattr(env, k, False)
        except FuturesTimeoutError:
            log.warning(
                f"Network detection timed out after {OVERALL_TIMEOUT}s. "
                "Using partial results."
            )
            for fut in futures:
                if not fut.done():
                    fut.cancel()
        except Exception as e:
            log.error(f"Network detection failed: {e}")
            for fut in futures:
                if not fut.done():
                    fut.cancel()

        # China direct mode: Check if China endpoints are accessible
        china_endpoints_ok = env.eastmoney_ok or env.tencent_ok or env.sina_ok
        
        if china_endpoints_ok:
            env.is_china_direct = True
            env.detection_method = "china_endpoints_ok"
        else:
            # All China endpoints failed - use previous or default
            if prev_env is not None:
                env.is_china_direct = bool(prev_env.is_china_direct)
                env.detection_method = "all_failed_keep_previous"
            else:
                env.is_china_direct = True
                env.detection_method = "default_china_direct"

        env.latency_ms = (time.time() - start) * 1000
        log.info(
            f"Network detected: CHINA_DIRECT "
            f"({env.detection_method}) "
            f"[eastmoney={'OK' if env.eastmoney_ok else 'FAIL'}, "
            f"tencent={'OK' if env.tencent_ok else 'FAIL'}, "
            f"baidu={'OK' if env.baidu_ok else 'FAIL'}] "
            f"({env.latency_ms:.0f}ms)"
        )
        return env


# Module-level convenience functions
_detector = NetworkDetector()


def get_network_env(force_refresh: bool = False) -> NetworkEnv:
    """Get current network environment."""
    return _detector.get_env(force_refresh=force_refresh)


def peek_network_env() -> NetworkEnv | None:
    """Get cached network environment without triggering probe."""
    return _detector.peek_env()


def invalidate_network_cache() -> None:
    """Force re-detection on next call."""
    _detector.invalidate()


def is_china_direct() -> bool:
    """Quick check: are we on a direct China connection? (Always True)"""
    return get_network_env().is_china_direct
