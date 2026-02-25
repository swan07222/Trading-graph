# core/network.py

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
    """Current network environment snapshot."""
    is_china_direct: bool = True          # True = China IP, no VPN
    is_vpn_active: bool = False           # True = foreign IP (Astrill ON)

    # Endpoint reachability (cached)
    eastmoney_ok: bool = False            # AkShare backend
    tencent_ok: bool = False              # Tencent quotes
    yahoo_ok: bool = False                # Yahoo Finance
    csindex_ok: bool = False              # CSIndex constituents

    detected_at: datetime | None = None
    detection_method: str = ""
    latency_ms: float = 0.0


class NetworkDetector:
    """Singleton that probes endpoints to determine network environment.

    Results are cached for `ttl_seconds` to avoid repeated probing.
    Thread-safe.

    FIX Bug 11: Releases lock during slow HTTP probes so other threads
    can read cached results without blocking for ~10s.
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
        self._detecting = threading.Event()  # Signals when detection is in progress

    def _has_fresh_cache_unlocked(self, now: float | None = None) -> bool:
        """Check cache freshness. Caller should hold _probe_lock."""
        if self._env is None:
            return False
        t_now = float(time.time() if now is None else now)
        return (t_now - float(self._env_time)) < float(self._ttl)

    def get_env(self, force_refresh: bool = False) -> NetworkEnv:
        """Get current network environment (cached).

        FIX Bug 11: The lock is released during the slow HTTP probing
        phase so that other threads can still read the (stale but usable)
        cached environment instead of blocking for ~10 seconds.
        """
        with self._probe_lock:
            if not force_refresh and self._has_fresh_cache_unlocked():
                return self._env  # type: ignore[return-value]

            # Snapshot previous env before releasing lock
            prev_env = self._env

        # Detect WITHOUT holding the lock — other threads can still read cache
        env = self._run_detect(prev_env)

        with self._probe_lock:
            if force_refresh:
                self._env = env
                self._env_time = time.time()
                return self._env  # type: ignore[return-value]

            # Another thread may have updated while we were probing;
            # only write if our result is newer
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
        """Return cached environment without probing.
        Returns None when cache is missing/stale.
        """
        with self._probe_lock:
            if not self._has_fresh_cache_unlocked():
                return None
            return self._env

    def _detect(self, prev_env: NetworkEnv | None = None) -> NetworkEnv:
        """Probe endpoints concurrently and determine network environment.

        Args:
            prev_env: Previous environment snapshot for fallback logic.
            
        FIX #12: Added overall timeout for network detection to prevent
        application startup hangs if ThreadPoolExecutor doesn't clean up properly.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

        from config.runtime_env import env_flag

        env = NetworkEnv(detected_at=datetime.now())
        start = time.time()

        force_vpn = env_flag("TRADING_VPN")
        force_china_direct = env_flag("TRADING_CHINA_DIRECT")

        probes = {
            "tencent_ok": ("https://qt.gtimg.cn/q=sh600519", 5),
            "eastmoney_ok": (
                "https://82.push2.eastmoney.com/api/qt/clist/get"
                "?pn=1&pz=1&fields=f2&fid=f3&fs=m:0+t:6",
                5,
            ),
            "yahoo_ok": (
                "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=1d",
                6,
            ),
            "csindex_ok": ("https://www.csindex.com.cn/", 5),
        }

        # Avoid sharing requests.Session across threads.
        # requests.Session is not thread-safe for concurrent get() calls.
        default_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            )
        }

        def run_probe(url: str, timeout: float | int) -> bool:
            try:
                # FIX #13: Add per-probe timeout cap to prevent single slow probe from delaying detection
                effective_timeout = min(timeout, 5.0)  # Max 5 seconds per probe
                r = requests.get(url, timeout=effective_timeout, headers=default_headers)
                return r.status_code == 200
            except requests.Timeout:
                # Explicitly handle timeout for better logging
                return False
            except Exception:
                return False

        # FIX #12: Add overall timeout for network detection
        OVERALL_TIMEOUT = 20  # seconds
        futures = []
        try:
            with ThreadPoolExecutor(max_workers=4) as ex:
                fut_map = {
                    ex.submit(run_probe, url, to): k
                    for k, (url, to) in probes.items()
                }
                futures = list(fut_map.keys())
                # FIX #12: Use overall timeout to prevent hangs
                for fut in as_completed(fut_map, timeout=OVERALL_TIMEOUT):
                    k = fut_map[fut]
                    try:
                        setattr(env, k, bool(fut.result()))
                    except Exception:
                        setattr(env, k, False)
        except FuturesTimeoutError:
            log.warning(
                f"Network detection timed out after {OVERALL_TIMEOUT}s. "
                "Using fallback logic."
            )
            # Cancel all pending futures to clean up resources
            for fut in futures:
                fut.cancel()
            # All probes failed, use fallback
            env.tencent_ok = False
            env.eastmoney_ok = False
            env.yahoo_ok = False
            env.csindex_ok = False
        except Exception as e:
            log.debug(f"Network detection thread pool error: {e}")
            # Cancel all pending futures on error
            for fut in futures:
                try:
                    fut.cancel()
                except Exception:
                    pass

        if env.eastmoney_ok and not env.yahoo_ok:
            env.is_china_direct = True
            env.is_vpn_active = False
            env.detection_method = "eastmoney_ok+yahoo_blocked"
        elif env.yahoo_ok and not env.eastmoney_ok:
            env.is_china_direct = False
            env.is_vpn_active = True
            env.detection_method = "yahoo_ok+eastmoney_blocked"
        elif env.eastmoney_ok and env.yahoo_ok:
            env.is_china_direct = True
            env.is_vpn_active = False
            env.detection_method = "both_ok_prefer_domestic"
        else:
            # Both major probes failed (often transient/rate-limit).
            # Keep previous mode if available to avoid flip-flopping.
            if prev_env is not None:
                env.is_china_direct = bool(prev_env.is_china_direct)
                env.is_vpn_active = bool(prev_env.is_vpn_active)
                env.detection_method = "both_failed_keep_previous"
            else:
                # First detection fallback: Tencent is a weak China hint.
                env.is_china_direct = bool(env.tencent_ok)
                env.is_vpn_active = False
                env.detection_method = (
                    "both_failed_tencent_ok" if env.tencent_ok
                    else "both_failed_all_down"
                )

        # Environment overrides (useful for China + VPN)
        if force_vpn is True:
            env.is_vpn_active = True
            env.is_china_direct = False
            env.detection_method = "env_force_vpn"
            # [DBG] VPN forced diagnostic
            log.info("[DBG] VPN mode FORCED via TRADING_VPN environment variable")
        elif force_china_direct is True:
            env.is_china_direct = True
            env.is_vpn_active = False
            env.detection_method = "env_force_china_direct"
            # [DBG] China direct forced diagnostic
            log.info("[DBG] China direct mode FORCED via TRADING_CHINA_DIRECT environment variable")

        env.latency_ms = (time.time() - start) * 1000
        log.info(
            f"Network detected: "
            f"{'CHINA_DIRECT' if env.is_china_direct else 'VPN_FOREIGN'} "
            f"({env.detection_method}) "
            f"[eastmoney={'OK' if env.eastmoney_ok else 'FAIL'}, "
            f"tencent={'OK' if env.tencent_ok else 'FAIL'}, "
            f"yahoo={'OK' if env.yahoo_ok else 'FAIL'}] "
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
    """Quick check: are we on a direct China connection?"""
    return get_network_env().is_china_direct


def is_vpn_active() -> bool:
    """Quick check: is VPN routing traffic abroad?"""
    return get_network_env().is_vpn_active
