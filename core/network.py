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
    # FIX China 2026-02-26: Replaced yahoo_ok with baidu_ok for China compatibility
    yahoo_ok: bool = False                # Backward compat (mapped to baidu_ok)
    baidu_ok: bool = False                # Baidu accessibility (China network indicator)
    sina_ok: bool = False                 # Sina Finance accessibility
    csindex_ok: bool = False              # Backward compat (mapped to sina_ok)

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
        
        FIX China Network 2026-02-26: 
        - Removed Yahoo Finance probe (blocked in China)
        - Added Baidu probe for China direct detection
        - Added Sina Finance probe as secondary China endpoint
        - Reduced timeouts for faster detection in China network
        - Added more China-friendly fallback logic
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        from config.runtime_env import env_flag

        env = NetworkEnv(detected_at=datetime.now())
        start = time.time()

        force_vpn = env_flag("TRADING_VPN")
        force_china_direct = env_flag("TRADING_CHINA_DIRECT")

        # FIX China Network: Use China-accessible endpoints for probing
        # Yahoo Finance removed (blocked in China), added Baidu and Sina
        probes = {
            "tencent_ok": ("https://qt.gtimg.cn/q=sh600519", 4),
            "eastmoney_ok": (
                "https://82.push2.eastmoney.com/api/qt/clist/get"
                "?pn=1&pz=1&fields=f2&fid=f3&fs=m:0+t:6",
                4,
            ),
            # FIX China: Replace Yahoo with Baidu for foreign IP detection
            "baidu_ok": (
                "https://www.baidu.com",
                4,
            ),
            # FIX China: Add Sina Finance as secondary probe
            "sina_ok": (
                "https://finance.sina.com.cn",
                4,
            ),
        }

        # Avoid sharing requests.Session across threads.
        # requests.Session is not thread-safe for concurrent get() calls.
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
                # FIX #13 + China: Add per-probe timeout cap
                effective_timeout = min(timeout, 4.0)  # Max 4 seconds per probe for China
                r = requests.get(url, timeout=effective_timeout, headers=default_headers)
                return r.status_code == 200
            except requests.Timeout:
                # Explicitly handle timeout for better logging
                log.debug("Probe timeout: %s", url)
                return False
            except requests.ConnectionError:
                # FIX China: Common in GFW scenarios
                log.debug("Probe connection error: %s", url)
                return False
            except Exception as e:
                log.debug("Probe failed %s: %s", url, type(e).__name__)
                return False

        # FIX China: Reduce overall timeout for faster startup
        OVERALL_TIMEOUT = 12  # seconds (reduced from 20s for China network)
        futures = []
        try:
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="network_probe") as ex:
                fut_map = {
                    ex.submit(run_probe, url, to): k
                    for k, (url, to) in probes.items()
                }
                futures = list(fut_map.keys())
                # FIX #12 + China: Use overall timeout
                for fut in as_completed(fut_map, timeout=OVERALL_TIMEOUT):
                    k = fut_map[fut]
                    try:
                        setattr(env, k, bool(fut.result()))
                    except Exception:
                        setattr(env, k, False)
        except FuturesTimeoutError:
            # FIX #10 + China: Log timeout and continue with partial results
            log.warning(
                f"Network detection timed out after {OVERALL_TIMEOUT}s. "
                "Using partial results."
            )
            # Cancel pending futures to clean up resources
            for fut in futures:
                if not fut.done():
                    fut.cancel()
        except Exception as e:
            # FIX #10 + China: Catch all exceptions
            log.error(f"Network detection failed: {e}")
            # Cancel pending futures to clean up resources
            for fut in futures:
                if not fut.done():
                    fut.cancel()

        # FIX China: Map baidu_ok/sina_ok to yahoo_ok/csindex_ok for backward compatibility
        # This allows existing code to work without changes
        env.yahoo_ok = env.baidu_ok  # Treat Baidu access as "foreign IP OK" proxy
        env.csindex_ok = env.sina_ok  # Treat Sina as secondary China probe

        # Handle case where all probes failed
        china_endpoints_ok = env.eastmoney_ok or env.tencent_ok or env.sina_ok
        if not any([china_endpoints_ok, env.baidu_ok]):
            log.warning(
                "All network probes failed. Using fallback logic."
            )

        # FIX China: Updated detection logic for China network reality
        # China direct: Can access EastMoney/Tencent/Sina but NOT Baidu (via GFW)
        # VPN mode: Can access Baidu AND China endpoints (via foreign routing)
        if china_endpoints_ok and not env.baidu_ok:
            env.is_china_direct = True
            env.is_vpn_active = False
            env.detection_method = "china_endpoints_ok+baidu_blocked"
        elif env.baidu_ok and china_endpoints_ok:
            # Both accessible - likely VPN or good international connection
            env.is_china_direct = False
            env.is_vpn_active = True
            env.detection_method = "both_ok_vpn_mode"
        elif env.baidu_ok and not china_endpoints_ok:
            # Only Baidu accessible - unusual, treat as VPN with China endpoint issues
            env.is_china_direct = False
            env.is_vpn_active = True
            env.detection_method = "baidu_ok+china_endpoints_blocked"
        else:
            # All China endpoints failed
            # Keep previous mode if available to avoid flip-flopping
            if prev_env is not None:
                env.is_china_direct = bool(prev_env.is_china_direct)
                env.is_vpn_active = bool(prev_env.is_vpn_active)
                env.detection_method = "all_failed_keep_previous"
            else:
                # First detection fallback: Use environment variables or default to China direct
                if force_vpn:
                    env.is_china_direct = False
                    env.is_vpn_active = True
                elif force_china_direct:
                    env.is_china_direct = True
                    env.is_vpn_active = False
                else:
                    # Default to China direct for mainland users
                    env.is_china_direct = True
                    env.is_vpn_active = False
                env.detection_method = "env_override_or_default_china"

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
