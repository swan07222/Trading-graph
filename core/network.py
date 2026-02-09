# core/network.py
"""
Network Environment Detection & Routing

Detects whether we're on a direct China IP or routing through VPN (foreign IP).
All data sources check this to decide which backends to use.

Usage:
    from core.network import get_network_env, NetworkEnv

    env = get_network_env()
    if env.is_china_direct:
        # Use AkShare, Tencent
    else:
        # Use Yahoo Finance
"""
import time
import threading
import requests
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class NetworkEnv:
    """Current network environment snapshot"""
    # Core detection
    is_china_direct: bool = True          # True = China IP, no VPN
    is_vpn_active: bool = False           # True = foreign IP (Astrill ON)

    # Endpoint reachability (cached)
    eastmoney_ok: bool = False            # AkShare backend
    tencent_ok: bool = False              # Tencent quotes
    yahoo_ok: bool = False                # Yahoo Finance
    csindex_ok: bool = False              # CSIndex constituents

    # Metadata
    detected_at: Optional[datetime] = None
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
        self._env: Optional[NetworkEnv] = None
        self._env_time: float = 0.0
        self._ttl: float = 120.0  # Re-detect every 2 minutes
        self._probe_lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_env(self, force_refresh: bool = False) -> NetworkEnv:
        """Get current network environment (cached)."""
        now = time.time()

        if not force_refresh and self._env is not None and (now - self._env_time) < self._ttl:
            return self._env

        with self._probe_lock:
            # Double-check after acquiring lock
            if not force_refresh and self._env is not None and (now - self._env_time) < self._ttl:
                return self._env

            env = self._detect()
            self._env = env
            self._env_time = time.time()
            return env

    def invalidate(self):
        """Force re-detection on next call."""
        self._env = None
        self._env_time = 0.0

    def _detect(self) -> NetworkEnv:
        """Probe endpoints and determine network environment."""
        env = NetworkEnv(detected_at=datetime.now())
        start = time.time()

        # Probe 1: Tencent (domestic, always works from China, may work from abroad too)
        env.tencent_ok = self._probe("https://qt.gtimg.cn/q=sh600519", timeout=3)

        # Probe 2: Eastmoney (domestic, BLOCKS foreign IPs)
        env.eastmoney_ok = self._probe(
            "https://82.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1&fields=f2&fid=f3&fs=m:0+t:6",
            timeout=3
        )

        # Probe 3: Yahoo Finance (international, blocked in China without VPN)
        env.yahoo_ok = self._probe(
            "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=1d",
            timeout=4
        )

        # Probe 4: CSIndex (domestic, sometimes blocks foreign IPs)
        env.csindex_ok = self._probe("https://www.csindex.com.cn/", timeout=3)

        # Determine environment
        if env.eastmoney_ok and not env.yahoo_ok:
            # Classic China direct: eastmoney works, Yahoo blocked
            env.is_china_direct = True
            env.is_vpn_active = False
            env.detection_method = "eastmoney_ok+yahoo_blocked"

        elif env.yahoo_ok and not env.eastmoney_ok:
            # VPN active: Yahoo works, eastmoney blocked (foreign IP)
            env.is_china_direct = False
            env.is_vpn_active = True
            env.detection_method = "yahoo_ok+eastmoney_blocked"

        elif env.eastmoney_ok and env.yahoo_ok:
            # Both work: could be smart-route VPN or corporate network
            # Prefer AkShare since it has better CN data
            env.is_china_direct = True
            env.is_vpn_active = False
            env.detection_method = "both_ok_prefer_domestic"

        elif not env.eastmoney_ok and not env.yahoo_ok:
            # Nothing works: network issue
            # Fall back to Tencent if available
            env.is_china_direct = env.tencent_ok
            env.is_vpn_active = False
            env.detection_method = "both_failed_fallback"

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

    def _probe(self, url: str, timeout: float = 3) -> bool:
        """Quick HTTP probe - returns True if endpoint responds with 200."""
        try:
            r = self._session.get(url, timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False


# Module-level convenience functions

_detector = NetworkDetector()


def get_network_env(force_refresh: bool = False) -> NetworkEnv:
    """Get current network environment."""
    return _detector.get_env(force_refresh=force_refresh)


def invalidate_network_cache():
    """Force re-detection on next call."""
    _detector.invalidate()


def is_china_direct() -> bool:
    """Quick check: are we on a direct China connection?"""
    return get_network_env().is_china_direct


def is_vpn_active() -> bool:
    """Quick check: is VPN routing traffic abroad?"""
    return get_network_env().is_vpn_active