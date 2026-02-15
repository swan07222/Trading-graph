# core/network.py

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import requests

from utils.logger import get_logger

log = get_logger(__name__)

@dataclass
class NetworkEnv:
    """Current network environment snapshot"""
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
    """
    Singleton that probes endpoints to determine network environment.

    Results are cached for `ttl_seconds` to avoid repeated probing.
    Thread-safe.
    """

    _instance = None
    _lock = threading.Lock()
    _initialized: bool

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
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

            env = self._detect()
            self._env = env
            self._env_time = time.time()
            return env

    def invalidate(self):
        """Force re-detection on next call."""
        with self._probe_lock:
            self._env = None
            self._env_time = 0.0

    def peek_env(self) -> NetworkEnv | None:
        """
        Return cached environment without probing.
        Returns None when cache is missing/stale.
        """
        with self._probe_lock:
            if not self._has_fresh_cache_unlocked():
                return None
            return self._env

    def _detect(self) -> NetworkEnv:
        """Probe endpoints concurrently and determine network environment."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        env = NetworkEnv(detected_at=datetime.now())
        start = time.time()

        def _env_bool(key: str) -> bool | None:
            val = os.environ.get(key)
            if val is None:
                return None
            return str(val).strip().lower() in ("1", "true", "yes", "on")

        force_vpn = _env_bool("TRADING_VPN")
        force_china_direct = _env_bool("TRADING_CHINA_DIRECT")

        probes = {
            "tencent_ok": ("https://qt.gtimg.cn/q=sh600519", 3),
            "eastmoney_ok": ("https://82.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1&fields=f2&fid=f3&fs=m:0+t:6", 3),
            "yahoo_ok": ("https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=1d", 4),
            "csindex_ok": ("https://www.csindex.com.cn/", 3),
        }

        # Avoid sharing requests.Session across threads.
        # requests.Session is not thread-safe for concurrent get() calls.
        default_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            )
        }

        def run_probe(url, timeout):
            try:
                r = requests.get(url, timeout=timeout, headers=default_headers)
                return r.status_code == 200
            except Exception:
                return False

        try:
            with ThreadPoolExecutor(max_workers=4) as ex:
                fut_map = {
                    ex.submit(run_probe, url, to): k
                    for k, (url, to) in probes.items()
                }
                for fut in as_completed(fut_map, timeout=10):
                    k = fut_map[fut]
                    try:
                        setattr(env, k, bool(fut.result()))
                    except Exception:
                        setattr(env, k, False)
        except Exception as e:
            log.debug(f"Network detection thread pool error: {e}")
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
            prev = self._env
            if prev is not None:
                env.is_china_direct = bool(prev.is_china_direct)
                env.is_vpn_active = bool(prev.is_vpn_active)
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
        elif force_china_direct is True:
            env.is_china_direct = True
            env.is_vpn_active = False
            env.detection_method = "env_force_china_direct"

        env.latency_ms = (time.time() - start) * 1000
        log.info(
            f"Network detected: {'CHINA_DIRECT' if env.is_china_direct else 'VPN_FOREIGN'} "
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

def invalidate_network_cache():
    """Force re-detection on next call."""
    _detector.invalidate()

def is_china_direct() -> bool:
    """Quick check: are we on a direct China connection?"""
    return get_network_env().is_china_direct

def is_vpn_active() -> bool:
    """Quick check: is VPN routing traffic abroad?"""
    return get_network_env().is_vpn_active
