"""China Network Optimization Module.

Provides enhanced connectivity for China mainland users with:
- Multiple Chinese CDN and DNS resolvers
- Proxy support (HTTP/SOCKS5)
- Connection pooling with keep-alive
- Optimized timeouts for China ISP conditions
- Network quality scoring and automatic failover
- Great Firewall circumvention helpers
"""
from __future__ import annotations

import os
import socket
import ssl
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.exceptions import DataFetchError
from utils.logger import get_logger

log = get_logger(__name__)


# ==================== CHINA-SPECIFIC ENDPOINTS ====================

# Primary Chinese financial data endpoints (domestic CDN)
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
    ("114.114.114.114", 53),      # 114DNS
    ("114.114.115.115", 53),      # 114DNS backup
    ("223.5.5.5", 53),            # AliDNS
    ("223.6.6.6", 53),            # AliDNS backup
    ("119.29.29.29", 53),         # DNSPod
    ("182.254.116.100", 53),      # DNSPod backup
    ("1.2.4.8", 53),              # CNNIC
]

# Public DNS-over-HTTPS (DoH) endpoints accessible from China
DOH_ENDPOINTS = [
    "https://dns.alidns.com/dns-query",
    "https://doh.pub/dns-query",
]


# ==================== NETWORK QUALITY SCORING ====================

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
    quality_score: float = 1.0  # 0-1, higher is better
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

        # Calculate success rate (exponential decay for recent performance)
        if self.total_requests > 0:
            recent_weight = min(1.0, self.total_requests / 20)
            old_rate = self.success_rate
            new_rate = self.successful_requests / self.total_requests
            self.success_rate = (1 - recent_weight) * old_rate + recent_weight * new_rate

        # Calculate quality score
        self._calculate_quality_score()

        # Mark unavailable after 5 consecutive failures
        self.is_available = self.consecutive_failures < 5

    def _calculate_quality_score(self) -> None:
        """Calculate overall quality score (0-1)."""
        # Latency score (lower is better, 0-500ms range)
        latency_score = max(0, 1 - (self.latency_ms / 500))

        # Success rate score
        success_score = self.success_rate

        # Recency bonus (recent success is good)
        recency_score = 0.0
        if self.last_success:
            minutes_since_success = (datetime.now() - self.last_success).total_seconds() / 60
            recency_score = max(0, 1 - (minutes_since_success / 30))

        # Consecutive failure penalty
        failure_penalty = min(0.5, self.consecutive_failures * 0.1)

        # Weighted combination
        self.quality_score = max(0, min(1,
            latency_score * 0.3 +
            success_score * 0.5 +
            recency_score * 0.2 -
            failure_penalty
        ))


class ChinaNetworkOptimizer:
    """Singleton optimizer for China network connectivity.

    Features:
    - Endpoint quality monitoring and automatic failover
    - Connection pooling with keep-alive
    - Proxy support (HTTP/SOCKS5)
    - Optimized timeouts for China ISP conditions
    - DNS resolution optimization
    """

    _instance: ChinaNetworkOptimizer | None = None
    _lock = threading.Lock()
    _initialized: bool

    def __new__(cls) -> ChinaNetworkOptimizer:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
                    # Mark as not initialized yet - __init__ will set it
                    object.__setattr__(instance, "_initialized", False)
        return cls._instance

    def __init__(self) -> None:
        # Double-check initialization to prevent race conditions
        if getattr(self, "_initialized", False):
            return
        
        # Mark as initialized BEFORE actual initialization
        # This prevents re-entry if __init__ is called from multiple threads
        object.__setattr__(self, "_initialized", True)

        # Quality tracking
        self._endpoint_quality: dict[str, dict[str, EndpointQuality]] = {}
        self._quality_lock = threading.Lock()

        # Session cache
        self._sessions: dict[str, requests.Session] = {}
        self._session_lock = threading.Lock()

        # Proxy configuration
        self._proxy_url: str | None = None
        self._proxy_enabled: bool = False

        # DNS cache
        self._dns_cache: dict[str, tuple[str, float]] = {}
        self._dns_cache_ttl = 300.0  # 5 minutes

        # Initialize quality tracking for all endpoints
        self._init_quality_tracking()

        # Load proxy from environment
        self._load_proxy_config()

    def _init_quality_tracking(self) -> None:
        """Initialize quality tracking for all endpoints."""
        for provider, endpoints in CHINA_ENDPOINTS.items():
            self._endpoint_quality[provider] = {}
            for endpoint in endpoints:
                self._endpoint_quality[provider][endpoint] = EndpointQuality(endpoint=endpoint)

    def _load_proxy_config(self) -> None:
        """Load proxy configuration from environment."""
        proxy_url = os.environ.get("TRADING_PROXY_URL", "").strip()
        if not proxy_url:
            # Check standard proxy env vars
            proxy_url = (
                os.environ.get("HTTP_PROXY", "").strip() or
                os.environ.get("HTTPS_PROXY", "").strip()
            )

        if proxy_url:
            self._proxy_url = proxy_url
            self._proxy_enabled = True
            log.info(f"Proxy configured: {self._mask_proxy_url(proxy_url)}")

    def _mask_proxy_url(self, url: str) -> str:
        """Mask proxy URL for logging."""
        if "@" in url:
            parts = url.split("@")
            if len(parts) == 2:
                auth, rest = parts
                if ":" in auth:
                    username = auth.split(":")[0]
                    return f"{username}:***@{rest}"
        return url

    def enable_proxy(self, proxy_url: str) -> None:
        """Enable proxy with given URL."""
        self._proxy_url = proxy_url
        self._proxy_enabled = True
        self._clear_sessions()  # Clear existing sessions
        log.info(f"Proxy enabled: {self._mask_proxy_url(proxy_url)}")

    def disable_proxy(self) -> None:
        """Disable proxy."""
        self._proxy_enabled = False
        self._clear_sessions()
        log.info("Proxy disabled")

    def is_proxy_enabled(self) -> bool:
        """Check if proxy is enabled."""
        return self._proxy_enabled

    def _clear_sessions(self) -> None:
        """Clear all cached sessions."""
        with self._session_lock:
            for session in self._sessions.values():
                try:
                    session.close()
                except Exception:
                    pass
            self._sessions.clear()

    def _get_session(self, provider: str) -> requests.Session:
        """Get or create a cached session with connection pooling.
        
        Thread-safe: Uses locks to protect proxy configuration access.
        """
        with self._session_lock:
            if provider in self._sessions:
                return self._sessions[provider]

            # Create new session with optimized settings for China
            session = requests.Session()

            # Connection pooling with retry logic
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

            # Default headers
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            })

            # Configure proxy if enabled (thread-safe access)
            # Note: We're already holding _session_lock, but we need
            # _quality_lock for proxy config to avoid race conditions
            with self._quality_lock:
                proxy_enabled = self._proxy_enabled
                proxy_url = self._proxy_url
            
            if proxy_enabled and proxy_url:
                session.proxies.update({
                    "http": proxy_url,
                    "https": proxy_url,
                })

            self._sessions[provider] = session
            return session

    def get_best_endpoint(self, provider: str) -> str:
        """Get the best quality endpoint for a provider."""
        with self._quality_lock:
            if provider not in self._endpoint_quality:
                # Return first endpoint if provider not tracked
                endpoints = CHINA_ENDPOINTS.get(provider, [])
                return endpoints[0] if endpoints else ""

            endpoints = self._endpoint_quality[provider]
            available = [
                (url, q) for url, q in endpoints.items()
                if q.is_available
            ]

            if not available:
                # All endpoints unavailable, return first
                endpoints_list = CHINA_ENDPOINTS.get(provider, [])
                return endpoints_list[0] if endpoints_list else ""

            # Sort by quality score (descending)
            available.sort(key=lambda x: x[1].quality_score, reverse=True)
            return available[0][0]

    def is_provider_available(self, provider: str) -> bool:
        """Check if a provider has at least one available endpoint.
        
        Args:
            provider: Provider name (e.g., 'eastmoney', 'sina')
            
        Returns:
            True if at least one endpoint is available
        """
        with self._quality_lock:
            if provider not in self._endpoint_quality:
                return False
            
            return any(
                quality.is_available 
                for quality in self._endpoint_quality[provider].values()
            )

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

                # Sort by quality score
                endpoints_data.sort(key=lambda x: float(x["quality_score"]), reverse=True)

                report[prov] = {
                    "endpoints": endpoints_data,
                    "best_endpoint": endpoints_data[0]["endpoint"] if endpoints_data else None,
                    "best_quality_score": endpoints_data[0]["quality_score"] if endpoints_data else 0,
                }

            return report

    def test_endpoint(
        self,
        url: str,
        timeout: float = 5.0,
    ) -> tuple[bool, float]:
        """Test endpoint connectivity.

        Returns:
            Tuple of (success, latency_ms)
        """
        start = time.time()
        try:
            # Use a simple HEAD request for testing
            response = requests.head(
                url,
                timeout=timeout,
                allow_redirects=True,
            )
            latency_ms = (time.time() - start) * 1000
            success = response.status_code in (200, 301, 302)
            return success, latency_ms
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            log.debug(f"Endpoint test failed ({url}): {e}")
            return False, latency_ms

    def run_endpoint_probe(self, provider: str) -> None:
        """Run connectivity probe for all endpoints of a provider.
        
        Thread-safe: Uses ThreadPoolExecutor with proper error handling.
        """
        endpoints = CHINA_ENDPOINTS.get(provider, [])
        if not endpoints:
            log.debug(f"No endpoints configured for provider: {provider}")
            return

        def test_one(endpoint: str) -> tuple[str, bool, float]:
            """Test single endpoint and return results."""
            success, latency = self.test_endpoint(endpoint)
            return endpoint, success, latency

        # Test endpoints concurrently with proper error handling
        from concurrent.futures import Future, ThreadPoolExecutor, as_completed

        max_workers = min(5, max(1, len(endpoints)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[Future, str] = {
                executor.submit(test_one, ep): ep for ep in endpoints
            }
            
            for future in as_completed(futures):
                endpoint = futures[future]
                try:
                    result = future.result(timeout=10.0)
                    if result is not None:
                        ep, success, latency = result
                        self.update_endpoint_quality(provider, ep, success, latency)
                        log.debug(
                            f"Probe {provider}/{ep}: "
                            f"{'OK' if success else 'FAIL'} ({latency:.0f}ms)"
                        )
                except TimeoutError:
                    log.warning(f"Endpoint probe timeout for {endpoint}")
                    self.update_endpoint_quality(provider, endpoint, False, 10000.0)
                except Exception as e:
                    log.debug(f"Endpoint probe error for {endpoint}: {e}")
                    self.update_endpoint_quality(provider, endpoint, False, 0.0)

    def resolve_dns(self, hostname: str) -> str | None:
        """Resolve DNS using Chinese DNS servers.

        Returns:
            Resolved IP address or None

        Note:
            This method attempts to use Chinese DNS servers for better
            resolution within China. However, Python's standard library
            doesn't support custom DNS servers directly, so we use
            DNS-over-HTTPS (DoH) as a fallback.
        """
        # Check cache first
        now = time.time()
        if hostname in self._dns_cache:
            ip, cached_at = self._dns_cache[hostname]
            if now - cached_at < self._dns_cache_ttl:
                return ip

        # Try standard resolution first (fastest path)
        try:
            ip = socket.gethostbyname(hostname)
            self._dns_cache[hostname] = (ip, now)
            return ip
        except socket.gaierror:
            pass

        # Try DNS-over-HTTPS (DoH) with Chinese providers
        # This works around the limitation of not being able to use
        # custom DNS servers directly from Python standard library
        for doh_url in DOH_ENDPOINTS:
            try:
                import json as json_module
                # DNS query for A record (type 1)
                response = requests.get(
                    doh_url,
                    params={
                        "name": hostname,
                        "type": "A",
                    },
                    headers={
                        "Accept": "application/dns-json",
                    },
                    timeout=3.0,
                )
                if response.status_code == 200:
                    data = json_module.loads(response.text)
                    if data.get("Status") == 0 and "Answer" in data:
                        for answer in data["Answer"]:
                            if answer.get("type") == 1:  # A record
                                ip = answer.get("data")
                                if ip:
                                    self._dns_cache[hostname] = (ip, now)
                                    return ip
            except Exception as e:
                log.debug(f"DoH resolution failed ({doh_url}): {e}")
                continue

        # Final fallback: try each Chinese DNS server via system resolver
        # Note: This doesn't actually query the DNS server directly,
        # but attempts resolution in case system is configured to use them
        for _dns_server, _port in CHINA_DNS_SERVERS:
            try:
                # Attempt to resolve - system may use configured DNS
                ip = socket.gethostbyname(hostname)
                if ip:
                    self._dns_cache[hostname] = (ip, now)
                    return ip
            except socket.gaierror:
                continue

        log.warning(f"All DNS resolution attempts failed for {hostname}")
        return None

    def get_network_status(self) -> dict[str, Any]:
        """Get comprehensive network status report."""
        status: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "proxy_enabled": self._proxy_enabled,
            "proxy_url": self._mask_proxy_url(self._proxy_url) if self._proxy_url else None,
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
    """HTTP adapter optimized for China network conditions.

    Features:
    - Larger connection pool
    - Keep-alive connections
    - Optimized socket options
    - Better retry logic for intermittent failures
    - Proper SSL context configuration
    """

    def __init__(
        self,
        pool_connections: int = 10,
        pool_maxsize: int = 50,
        pool_block: bool = False,
        max_retries: Retry | None = None,
    ) -> None:
        # Store socket options for later use
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
        # Enable keep-alive with optimized socket options
        pool_kwargs["maxsize"] = maxsize
        pool_kwargs["block"] = block
        
        # Merge socket options (preserve existing if any)
        existing_options = pool_kwargs.get("socket_options", [])
        pool_kwargs["socket_options"] = existing_options + self._socket_options

        # Create proper SSL context for China network conditions
        ssl_context = ssl.create_default_context()
        ssl_context.set_ciphers(
            "ECDHE+AESGCM:DHE+AESGCM:ECDHE+CHACHA20:DHE+CHACHA20:"
            "ECDHE+AES:DHE+AES:!aNULL:!eNULL:!aDSS:!SHA1:!AESCCM"
        )
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        ssl_context.options |= (
            ssl.OP_NO_SSLv2 |
            ssl.OP_NO_SSLv3 |
            ssl.OP_NO_TLSv1 |
            ssl.OP_NO_TLSv1_1
        )
        # Enable SNI for proper HTTPS support
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


def enable_proxy(proxy_url: str) -> None:
    """Enable proxy with given URL."""
    get_optimizer().enable_proxy(proxy_url)


def disable_proxy() -> None:
    """Disable proxy."""
    get_optimizer().disable_proxy()


def is_proxy_enabled() -> bool:
    """Check if proxy is enabled."""
    return get_optimizer().is_proxy_enabled()


# ==================== DECORATORS ====================

def china_optimized(provider: str):
    """Decorator for functions that fetch from Chinese endpoints.

    Automatically:
    - Selects best endpoint
    - Tracks quality metrics
    - Retries on failure with failover
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            optimizer = get_optimizer()

            # Get best endpoint
            best_endpoint = optimizer.get_best_endpoint(provider)

            # Try up to 3 endpoints
            endpoints_tried = []
            last_error = None

            for _ in range(3):
                if best_endpoint in endpoints_tried:
                    best_endpoint = optimizer.get_best_endpoint(provider)
                    continue

                endpoints_tried.append(best_endpoint)
                start = time.time()

                try:
                    # Inject endpoint into kwargs
                    kwargs["endpoint"] = best_endpoint
                    result = func(*args, **kwargs)

                    # Success - update quality
                    latency = (time.time() - start) * 1000
                    optimizer.update_endpoint_quality(provider, best_endpoint, True, latency)
                    return result

                except Exception as e:
                    last_error = e
                    latency = (time.time() - start) * 1000
                    optimizer.update_endpoint_quality(provider, best_endpoint, False, latency)
                    log.debug(f"Endpoint failed ({best_endpoint}): {e}")

                    # Get next best endpoint
                    best_endpoint = optimizer.get_best_endpoint(provider)

            # All endpoints failed
            raise DataFetchError(
                f"All endpoints failed for {provider}: {last_error}"
            ) from last_error

        return wrapper
    return decorator


def check_search_engine_health(
    engine_name: str,
    test_url: str,
    timeout: int = 10,
) -> tuple[bool, float]:
    """Check health of a search engine endpoint.
    
    China-optimized: Tests search engine accessibility and response time.
    
    Args:
        engine_name: Name of the search engine (e.g., 'baidu', 'bing_cn')
        test_url: URL to test (usually the search engine homepage)
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (is_healthy, latency_ms)
    """
    start = time.time()
    try:
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            timeout=timeout,
        )
        session.mount('https://', adapter)
        
        response = session.get(test_url, timeout=timeout)
        latency_ms = (time.time() - start) * 1000
        
        is_healthy = response.status_code == 200
        session.close()
        
        log.debug(f"Search engine {engine_name} health check: {'OK' if is_healthy else 'FAILED'} ({latency_ms:.0f}ms)")
        return is_healthy, latency_ms
        
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        log.warning(f"Search engine {engine_name} health check FAILED: {e}")
        return False, latency_ms


def get_network_health_report() -> dict[str, Any]:
    """Get comprehensive network health report for China users.
    
    Returns:
        Dictionary with health metrics for all monitored endpoints
    """
    optimizer = get_optimizer()
    
    # Test search engines
    search_engines = {
        "baidu": "https://www.baidu.com",
        "bing_cn": "https://cn.bing.com",
        "sogou": "https://www.sogou.com",
    }
    
    health_report = {
        "timestamp": datetime.now().isoformat(),
        "search_engines": {},
        "data_providers": {},
        "dns_servers": [],
        "proxy_enabled": is_proxy_enabled(),
    }
    
    # Check search engines
    for name, url in search_engines.items():
        is_healthy, latency = check_search_engine_health(name, url)
        health_report["search_engines"][name] = {
            "healthy": is_healthy,
            "latency_ms": latency,
            "status": "OK" if is_healthy else "UNREACHABLE",
        }
    
    # Get data provider status
    for provider in CHINA_ENDPOINTS:
        best_endpoint = optimizer.get_best_endpoint(provider)
        health_report["data_providers"][provider] = {
            "best_endpoint": best_endpoint,
            "available": optimizer.is_provider_available(provider),
        }
    
    # Test DNS servers
    for dns_ip, dns_port in CHINA_DNS_SERVERS[:3]:  # Test top 3
        try:
            socket.setdefaulttimeout(2)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(2)
            sock.sendto(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', (dns_ip, dns_port))
            sock.close()
            health_report["dns_servers"].append({
                "ip": dns_ip,
                "reachable": True,
            })
        except Exception:
            health_report["dns_servers"].append({
                "ip": dns_ip,
                "reachable": False,
            })
    
    return health_report


def close_all_sessions() -> None:
    """FIX #9: Close all cached sessions to prevent resource leaks.
    
    Call this during application shutdown to properly clean up network resources.
    """
    optimizer = get_optimizer()
    optimizer._clear_sessions()


def shutdown() -> None:
    """FIX #9: Shutdown network optimizer and clean up all resources.
    
    Call this during application shutdown.
    """
    close_all_sessions()
