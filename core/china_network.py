"""China Network Optimization Module.

Provides optimized connectivity for mainland China users with:
- Multiple Chinese CDN endpoints
- Connection pooling with keep-alive
- Optimized timeouts for China ISP conditions
- Network quality scoring and automatic failover
- Chinese DNS resolvers
"""
from __future__ import annotations

import os
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.exceptions import DataFetchError
from utils.logger import get_logger

log = get_logger(__name__)


# ==================== CHINA ENDPOINTS ====================

CHINA_ENDPOINTS = {
    "eastmoney": [
        "https://82.push2.eastmoney.com",
        "https://push2.eastmoney.com",
        "https://np-anotice-stock.eastmoney.com",
        "https://api.fund.eastmoney.com",
    ],
    "sina": [
        "https://feed.mix.sina.com.cn",
        "https://hq.sinajs.cn",
        "https://finance.sina.com.cn",
    ],
    "tencent": [
        "https://qt.gtimg.cn",
        "https://web.ifzq.gtimg.cn",
        "https://gu.qq.com",
    ],
    "akshare": [
        "https://www.akshare.xyz",
        "https://api.akshare.cn",
    ],
    "jin10": [
        "https://api.jin10.com",
        "https://www.jin10.com",
    ],
    "xueqiu": [
        "https://xueqiu.com",
        "https://stock.xueqiu.com",
    ],
    "csindex": [
        "https://www.csindex.com.cn",
    ],
    "sse": [
        "http://www.sse.com.cn",
    ],
    "szse": [
        "http://www.szse.cn",
    ],
}

# Chinese DNS resolvers (faster than 8.8.8.8 in China)
CHINA_DNS_SERVERS = [
    ("114.114.114.114", 53),
    ("114.114.115.115", 53),
    ("223.5.5.5", 53),
    ("223.6.6.6", 53),
    ("119.29.29.29", 53),
    ("182.254.116.100", 53),
    ("1.2.4.8", 53),
]

DOH_ENDPOINTS = [
    "https://dns.alidns.com/dns-query",
    "https://doh.pub/dns-query",
]


@dataclass
class EndpointQuality:
    """Quality metrics for a network endpoint."""
    endpoint: str
    latency_ms: float = 0.0
    success_rate: float = 1.0
    last_success: datetime | None = None
    last_failure: datetime | None = None
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    quality_score: float = 1.0
    is_available: bool = True

    def update(self, success: bool, latency_ms: float = 0.0) -> None:
        """Update quality metrics after a request."""
        self.total_requests += 1
        now = datetime.now()

        if success:
            self.successful_requests += 1
            self.last_success = now
            self.consecutive_failures = 0
            self.latency_ms = latency_ms
        else:
            self.last_failure = now
            self.consecutive_failures += 1

        if self.total_requests > 0:
            recent_weight = min(1.0, self.total_requests / 20)
            old_rate = self.success_rate
            new_rate = self.successful_requests / self.total_requests
            self.success_rate = (1 - recent_weight) * old_rate + recent_weight * new_rate

        self._calculate_quality_score()
        self.is_available = self.consecutive_failures < 5

    def _calculate_quality_score(self) -> None:
        """Calculate overall quality score (0-1)."""
        latency_score = max(0, 1 - (self.latency_ms / 500))
        success_score = self.success_rate
        recency_score = 0.0
        if self.last_success:
            minutes_since_success = (datetime.now() - self.last_success).total_seconds() / 60
            recency_score = max(0, 1 - (minutes_since_success / 30))
        failure_penalty = min(0.5, self.consecutive_failures * 0.1)
        self.quality_score = max(0, min(1,
            latency_score * 0.3 + success_score * 0.5 + recency_score * 0.2 - failure_penalty
        ))


class ChinaNetworkOptimizer:
    """Singleton optimizer for China network connectivity."""

    _instance: ChinaNetworkOptimizer | None = None
    _lock = threading.Lock()
    _initialized: bool

    def __new__(cls) -> ChinaNetworkOptimizer:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
                    object.__setattr__(instance, "_initialized", False)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        object.__setattr__(self, "_initialized", True)

        self._endpoint_quality: dict[str, dict[str, EndpointQuality]] = {}
        self._quality_lock = threading.Lock()
        self._sessions: dict[str, requests.Session] = {}
        self._session_lock = threading.Lock()
        self._dns_cache: dict[str, tuple[str, float]] = {}
        self._dns_cache_ttl = 300.0

        self._init_quality_tracking()

    def _init_quality_tracking(self) -> None:
        """Initialize quality tracking for all endpoints."""
        for provider, endpoints in CHINA_ENDPOINTS.items():
            self._endpoint_quality[provider] = {}
            for endpoint in endpoints:
                self._endpoint_quality[provider][endpoint] = EndpointQuality(endpoint=endpoint)

    def _get_session(self, provider: str) -> requests.Session:
        """Get or create a cached session with connection pooling."""
        with self._session_lock:
            if provider in self._sessions:
                return self._sessions[provider]

            session = requests.Session()

            adapter = ChinaHTTPAdapter(
                pool_connections=10,
                pool_maxsize=50,
                pool_block=False,
                max_retries=Retry(
                    total=3,
                    backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET", "POST", "HEAD"],
                ),
            )

            session.mount("https://", adapter)
            session.mount("http://", adapter)

            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            })

            self._sessions[provider] = session
            return session

    def get_best_endpoint(self, provider: str) -> str:
        """Get the best quality endpoint for a provider."""
        with self._quality_lock:
            if provider not in self._endpoint_quality:
                endpoints = CHINA_ENDPOINTS.get(provider, [])
                return endpoints[0] if endpoints else ""

            endpoints = self._endpoint_quality[provider]
            available = [(url, q) for url, q in endpoints.items() if q.is_available]

            if not available:
                endpoints_list = CHINA_ENDPOINTS.get(provider, [])
                return endpoints_list[0] if endpoints_list else ""

            available.sort(key=lambda x: x[1].quality_score, reverse=True)
            return available[0][0]

    def is_provider_available(self, provider: str) -> bool:
        """Check if a provider has at least one available endpoint."""
        with self._quality_lock:
            if provider not in self._endpoint_quality:
                return False
            return any(q.is_available for q in self._endpoint_quality[provider].values())

    def update_endpoint_quality(
        self,
        provider: str,
        endpoint: str,
        success: bool,
        latency_ms: float = 0.0,
    ) -> None:
        """Update quality metrics for an endpoint."""
        with self._quality_lock:
            if provider not in self._endpoint_quality:
                self._endpoint_quality[provider] = {}

            if endpoint not in self._endpoint_quality[provider]:
                self._endpoint_quality[provider][endpoint] = EndpointQuality(endpoint=endpoint)

            self._endpoint_quality[provider][endpoint].update(success, latency_ms)

    def get_endpoint_quality_report(self, provider: str | None = None) -> dict[str, Any]:
        """Get quality report for endpoints."""
        with self._quality_lock:
            report = {}
            providers = [provider] if provider else list(CHINA_ENDPOINTS.keys())

            for prov in providers:
                if prov not in self._endpoint_quality:
                    continue

                endpoints_data = []
                for url, quality in self._endpoint_quality[prov].items():
                    endpoints_data.append({
                        "endpoint": url,
                        "latency_ms": round(quality.latency_ms, 2),
                        "success_rate": round(quality.success_rate, 4),
                        "quality_score": round(quality.quality_score, 4),
                        "is_available": quality.is_available,
                        "consecutive_failures": quality.consecutive_failures,
                        "total_requests": quality.total_requests,
                    })

                endpoints_data.sort(key=lambda x: float(x["quality_score"]), reverse=True)

                report[prov] = {
                    "endpoints": endpoints_data,
                    "best_endpoint": endpoints_data[0]["endpoint"] if endpoints_data else None,
                    "best_quality_score": endpoints_data[0]["quality_score"] if endpoints_data else 0,
                }

            return report

    def test_endpoint(self, url: str, timeout: float = 5.0) -> tuple[bool, float]:
        """Test endpoint connectivity."""
        start = time.time()
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            latency_ms = (time.time() - start) * 1000
            success = response.status_code in (200, 301, 302)
            return success, latency_ms
        except Exception:
            latency_ms = (time.time() - start) * 1000
            return False, latency_ms

    def run_endpoint_probe(self, provider: str) -> None:
        """Run connectivity probe for all endpoints of a provider."""
        endpoints = CHINA_ENDPOINTS.get(provider, [])
        if not endpoints:
            return

        from concurrent.futures import Future, ThreadPoolExecutor, as_completed

        max_workers = min(5, max(1, len(endpoints)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[Future, str] = {
                executor.submit(lambda ep: self.test_endpoint(ep), ep): ep
                for ep in endpoints
            }

            for future in as_completed(futures):
                endpoint = futures[future]
                try:
                    success, latency = future.result(timeout=10.0)
                    self.update_endpoint_quality(provider, endpoint, success, latency)
                except Exception:
                    self.update_endpoint_quality(provider, endpoint, False, 10000.0)

    def resolve_dns(self, hostname: str) -> str | None:
        """Resolve DNS using Chinese DNS servers."""
        now = time.time()
        if hostname in self._dns_cache:
            ip, cached_at = self._dns_cache[hostname]
            if now - cached_at < self._dns_cache_ttl:
                return ip

        try:
            ip = socket.gethostbyname(hostname)
            self._dns_cache[hostname] = (ip, now)
            return ip
        except socket.gaierror:
            pass

        for doh_url in DOH_ENDPOINTS:
            try:
                import json as json_module
                response = requests.get(
                    doh_url,
                    params={"name": hostname, "type": "A"},
                    headers={"Accept": "application/dns-json"},
                    timeout=3.0,
                )
                if response.status_code == 200:
                    data = json_module.loads(response.text)
                    if data.get("Status") == 0 and "Answer" in data:
                        for answer in data["Answer"]:
                            if answer.get("type") == 1:
                                ip = answer.get("data")
                                if ip:
                                    self._dns_cache[hostname] = (ip, now)
                                    return ip
            except Exception:
                continue

        return None

    def get_network_status(self) -> dict[str, Any]:
        """Get comprehensive network status report."""
        status: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "providers": {},
            "overall_quality": 0.0,
        }

        quality_scores: list[float] = []
        for provider in CHINA_ENDPOINTS.keys():
            report = self.get_endpoint_quality_report(provider)
            if provider in report:
                provider_data = report[provider]
                status["providers"][provider] = {
                    "best_endpoint": provider_data["best_endpoint"],
                    "quality_score": provider_data["best_quality_score"],
                    "available_endpoints": sum(
                        1 for ep in self._endpoint_quality.get(provider, {}).values()
                        if ep.is_available
                    ),
                }
                quality_scores.append(provider_data["best_quality_score"])

        if quality_scores:
            status["overall_quality"] = sum(quality_scores) / len(quality_scores)

        return status


class ChinaHTTPAdapter(HTTPAdapter):
    """HTTP adapter optimized for China network conditions."""

    def __init__(
        self,
        pool_connections: int = 10,
        pool_maxsize: int = 50,
        pool_block: bool = False,
        max_retries: Retry | None = None,
    ) -> None:
        self._socket_options = [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60),
            (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),
            (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),
            (socket.SOL_TCP, socket.TCP_NODELAY, 1),
        ]
        super().__init__(pool_connections, pool_maxsize, pool_block, max_retries)

    def init_poolmanager(
        self,
        connections: int,
        maxsize: int,
        block: bool = False,
        **pool_kwargs: Any,
    ) -> None:
        pool_kwargs["maxsize"] = maxsize
        pool_kwargs["block"] = block
        existing_options = pool_kwargs.get("socket_options", [])
        pool_kwargs["socket_options"] = existing_options + self._socket_options

        ssl_context = ssl.create_default_context()
        ssl_context.set_ciphers(
            "ECDHE+AESGCM:DHE+AESGCM:ECDHE+CHACHA20:DHE+CHACHA20:"
            "ECDHE+AES:DHE+AES:!aNULL:!eNULL:!aDSS:!SHA1:!AESCCM"
        )
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        ssl_context.options |= (
            ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 |
            ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
        )
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        pool_kwargs["ssl_context"] = ssl_context

        super().init_poolmanager(connections, maxsize, block, **pool_kwargs)


# ==================== CONVENIENCE FUNCTIONS ====================

_optimizer: ChinaNetworkOptimizer | None = None


def get_optimizer() -> ChinaNetworkOptimizer:
    """Get the China network optimizer singleton."""
    global _optimizer
    if _optimizer is None:
        _optimizer = ChinaNetworkOptimizer()
    return _optimizer


def get_best_endpoint(provider: str) -> str:
    """Get the best quality endpoint for a provider."""
    return get_optimizer().get_best_endpoint(provider)


def update_endpoint_quality(
    provider: str,
    endpoint: str,
    success: bool,
    latency_ms: float = 0.0,
) -> None:
    """Update quality metrics for an endpoint."""
    get_optimizer().update_endpoint_quality(provider, endpoint, success, latency_ms)


def get_optimized_session(provider: str) -> requests.Session:
    """Get an optimized requests session for a provider."""
    return get_optimizer()._get_session(provider)


def get_network_status() -> dict[str, Any]:
    """Get comprehensive network status report."""
    return get_optimizer().get_network_status()
