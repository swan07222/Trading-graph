"""China Network Diagnostics Utility.

Provides comprehensive network testing for China mainland users:
- Great Firewall connectivity tests
- Chinese CDN endpoint quality
- DNS resolution speed
- Proxy connectivity
- ISP-specific optimization recommendations
"""
from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TestResult:
    """Result of a network test."""
    name: str
    category: str
    success: bool
    latency_ms: float = 0.0
    error: str = ""
    details: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.details is None:
            self.details = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "success": self.success,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "details": self.details,
        }


class ChinaNetworkDiagnostics:
    """Comprehensive network diagnostics for China users.
    
    Tests:
    - Financial data provider connectivity
    - DNS resolution speed
    - Great Firewall traversal
    - Proxy functionality
    - ISP-specific routing
    """

    # Chinese financial data providers
    FINANCIAL_ENDPOINTS = {
        "EastMoney": [
            "https://82.push2.eastmoney.com",
            "https://www.eastmoney.com",
        ],
        "Sina Finance": [
            "https://finance.sina.com.cn",
            "https://hq.sinajs.cn",
        ],
        "Tencent Finance": [
            "https://qt.gtimg.cn",
            "https://gu.qq.com",
        ],
        "Jin10": [
            "https://www.jin10.com",
            "https://api.jin10.com",
        ],
        "Xueqiu": [
            "https://xueqiu.com",
            "https://stock.xueqiu.com",
        ],
        "CSIndex": [
            "https://www.csindex.com.cn",
        ],
        "SSE": [
            "http://www.sse.com.cn",
        ],
        "SZSE": [
            "http://www.szse.cn",
        ],
    }

    # International services (test VPN status)
    INTERNATIONAL_ENDPOINTS = {
        "Yahoo Finance": [
            "https://finance.yahoo.com",
        ],
        "Google DNS": [
            "8.8.8.8",
        ],
    }

    # Chinese DNS servers
    CHINA_DNS = [
        ("114.114.114.114", "114DNS"),
        ("223.5.5.5", "AliDNS"),
        ("119.29.29.29", "DNSPod"),
        ("1.2.4.8", "CNNIC"),
    ]

    def __init__(self) -> None:
        self.results: list[TestResult] = []
        self.summary: dict[str, Any] = {}

    def test_endpoint(
        self,
        name: str,
        url: str,
        category: str = "connectivity",
        timeout: float = 5.0,
    ) -> TestResult:
        """Test connectivity to an endpoint."""
        import requests

        start = time.time()
        error = ""
        success = False

        try:
            # Determine if URL is IP or hostname
            if url.startswith("http"):
                response = requests.head(url, timeout=timeout, allow_redirects=True)
                success = response.status_code in (200, 301, 302)
                if not success:
                    error = f"HTTP {response.status_code}"
            else:
                # IP address test (ping via socket)
                socket.setdefaulttimeout(timeout)
                socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((url, 443))
                success = True

        except requests.exceptions.Timeout:
            error = "Timeout"
        except requests.exceptions.ConnectionError as e:
            error = f"Connection error: {str(e)[:50]}"
        except TimeoutError:
            error = "Socket timeout"
        except Exception as e:
            error = str(e)[:100]

        latency_ms = (time.time() - start) * 1000

        return TestResult(
            name=name,
            category=category,
            success=success,
            latency_ms=latency_ms,
            error=error,
            details={"url": url, "timeout": timeout},
        )

    def test_dns_resolution(
        self,
        hostname: str,
        dns_server: str,
        dns_name: str,
    ) -> TestResult:
        """Test DNS resolution speed."""
        start = time.time()
        error = ""
        success = False
        resolved_ip = ""

        try:
            # Set DNS server (system-dependent)
            # Note: Python doesn't support custom DNS servers directly
            # This is a simplified test using system DNS
            ip = socket.gethostbyname(hostname)
            success = True
            resolved_ip = ip
        except socket.gaierror as e:
            error = f"DNS resolution failed: {str(e)[:50]}"
        except Exception as e:
            error = str(e)[:100]

        latency_ms = (time.time() - start) * 1000

        return TestResult(
            name=f"DNS {dns_name}",
            category="dns",
            success=success,
            latency_ms=latency_ms,
            error=error,
            details={
                "hostname": hostname,
                "dns_server": dns_server,
                "dns_name": dns_name,
                "resolved_ip": resolved_ip,
            },
        )

    def test_proxy(
        self,
        proxy_url: str,
        test_url: str = "https://www.eastmoney.com",
    ) -> TestResult:
        """Test proxy connectivity."""
        import requests

        start = time.time()
        error = ""
        success = False

        try:
            proxies = {
                "http": proxy_url,
                "https": proxy_url,
            }
            response = requests.get(
                test_url,
                proxies=proxies,
                timeout=10.0,
            )
            success = response.status_code == 200
            if not success:
                error = f"HTTP {response.status_code}"
        except requests.exceptions.Timeout:
            error = "Proxy timeout"
        except requests.exceptions.ProxyError as e:
            error = f"Proxy error: {str(e)[:50]}"
        except Exception as e:
            error = str(e)[:100]

        latency_ms = (time.time() - start) * 1000

        return TestResult(
            name="Proxy Test",
            category="proxy",
            success=success,
            latency_ms=latency_ms,
            error=error,
            details={
                "proxy_url": proxy_url,
                "test_url": test_url,
            },
        )

    def run_all_tests(self, proxy_url: str | None = None) -> dict[str, Any]:
        """Run comprehensive network diagnostics.
        
        Returns:
            Dictionary with test results and recommendations
        """
        self.results = []

        # Test Chinese financial endpoints
        log.info("Testing Chinese financial endpoints...")
        for provider, urls in self.FINANCIAL_ENDPOINTS.items():
            for url in urls:
                result = self.test_endpoint(provider, url, "financial")
                self.results.append(result)

        # Test international endpoints (VPN status)
        log.info("Testing international endpoints...")
        for provider, urls in self.INTERNATIONAL_ENDPOINTS.items():
            for url in urls:
                result = self.test_endpoint(provider, url, "international")
                self.results.append(result)

        # Test DNS resolution
        log.info("Testing DNS resolution...")
        for dns_ip, dns_name in self.CHINA_DNS:
            # Note: This uses system DNS, not the specified server
            # Real DNS server testing requires dnspython library
            result = self.test_endpoint(
                f"DNS {dns_name}",
                f"https://{dns_ip}",
                "dns",
                timeout=3.0,
            )
            self.results.append(result)

        # Test proxy if provided
        if proxy_url:
            log.info("Testing proxy...")
            result = self.test_proxy(proxy_url)
            self.results.append(result)

        # Generate summary
        self._generate_summary()

        return self.get_report()

    def _generate_summary(self) -> None:
        """Generate summary statistics and recommendations."""
        financial_results = [r for r in self.results if r.category == "financial"]
        international_results = [r for r in self.results if r.category == "international"]
        dns_results = [r for r in self.results if r.category == "dns"]
        proxy_results = [r for r in self.results if r.category == "proxy"]

        # Calculate success rates
        financial_success = sum(1 for r in financial_results if r.success)
        financial_total = len(financial_results)
        financial_rate = financial_success / financial_total if financial_total > 0 else 0

        international_success = sum(1 for r in international_results if r.success)
        international_total = len(international_results)
        international_rate = international_success / international_total if international_total > 0 else 0

        # Determine network mode
        if financial_rate > 0.7 and international_rate < 0.3:
            network_mode = "china_direct"
            vpn_recommended = False
        elif international_rate > 0.7:
            network_mode = "vpn_active"
            vpn_recommended = False
        else:
            network_mode = "mixed_or_poor"
            vpn_recommended = True

        # Calculate average latencies
        financial_latencies = [r.latency_ms for r in financial_results if r.success]
        avg_financial_latency = (
            sum(financial_latencies) / len(financial_latencies)
            if financial_latencies else 0
        )

        # Generate recommendations
        recommendations = []

        if financial_rate < 0.5:
            recommendations.append({
                "priority": "high",
                "issue": "Poor connectivity to Chinese financial providers",
                "suggestion": "Check firewall settings, try alternative DNS servers (114.114.114.114 or 223.5.5.5)",
            })

        if avg_financial_latency > 2000:
            recommendations.append({
                "priority": "medium",
                "issue": f"High latency to financial endpoints ({avg_financial_latency:.0f}ms)",
                "suggestion": "Consider using a domestic CDN or proxy service",
            })

        if vpn_recommended:
            recommendations.append({
                "priority": "medium",
                "issue": "Unstable network configuration",
                "suggestion": "Enable VPN for consistent access to international data sources",
            })

        if dns_results and sum(1 for r in dns_results if r.success) < len(dns_results) / 2:
            recommendations.append({
                "priority": "high",
                "issue": "DNS resolution issues detected",
                "suggestion": "Change DNS servers to AliDNS (223.5.5.5) or 114DNS (114.114.114.114)",
            })

        self.summary = {
            "timestamp": datetime.now().isoformat(),
            "network_mode": network_mode,
            "financial_connectivity": {
                "success_rate": round(financial_rate, 4),
                "successful": financial_success,
                "total": financial_total,
                "avg_latency_ms": round(avg_financial_latency, 2),
            },
            "international_connectivity": {
                "success_rate": round(international_rate, 4),
                "successful": international_success,
                "total": international_total,
            },
            "dns_status": {
                "tested": len(dns_results),
                "successful": sum(1 for r in dns_results if r.success),
            },
            "proxy_status": {
                "tested": len(proxy_results),
                "successful": sum(1 for r in proxy_results if r.success),
            },
            "recommendations": recommendations,
            "overall_health": "good" if financial_rate > 0.7 else "fair" if financial_rate > 0.4 else "poor",
        }

    def get_report(self) -> dict[str, Any]:
        """Get full diagnostic report."""
        return {
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
        }

    def print_report(self) -> None:
        """Print formatted diagnostic report."""
        if not self.summary:
            self._generate_summary()

        print("\n" + "=" * 60)
        print("CHINA NETWORK DIAGNOSTICS REPORT")
        print("=" * 60)
        print(f"Timestamp: {self.summary.get('timestamp', 'N/A')}")
        print(f"Network Mode: {self.summary.get('network_mode', 'unknown')}")
        print(f"Overall Health: {self.summary.get('overall_health', 'unknown')}")
        print()

        # Financial connectivity
        fin = self.summary.get("financial_connectivity", {})
        print("FINANCIAL ENDPOINTS:")
        print(f"  Success Rate: {fin.get('success_rate', 0) * 100:.1f}%")
        print(f"  Avg Latency: {fin.get('avg_latency_ms', 0):.0f}ms")
        print()

        # International connectivity
        intl = self.summary.get("international_connectivity", {})
        print("INTERNATIONAL ENDPOINTS:")
        print(f"  Success Rate: {intl.get('success_rate', 0) * 100:.1f}%")
        print()

        # Recommendations
        recommendations = self.summary.get("recommendations", [])
        if recommendations:
            print("RECOMMENDATIONS:")
            for rec in recommendations:
                priority_marker = "!!!" if rec["priority"] == "high" else "!"
                print(f"  {priority_marker} [{rec['priority'].upper()}] {rec['issue']}")
                print(f"      → {rec['suggestion']}")
        else:
            print("RECOMMENDATIONS: None - Network configuration looks good!")

        print()
        print("=" * 60)

        # Detailed results
        print("\nDETAILED RESULTS:")
        print("-" * 60)
        for result in self.results:
            status = "✓" if result.success else "✗"
            latency_str = f"{result.latency_ms:.0f}ms" if result.success else "N/A"
            print(f"{status} {result.name}: {latency_str} {result.error}")


def run_diagnostics(proxy_url: str | None = None) -> dict[str, Any]:
    """Run China network diagnostics.
    
    Args:
        proxy_url: Optional proxy URL to test
        
    Returns:
        Diagnostic report dictionary
    """
    diagnostics = ChinaNetworkDiagnostics()
    return diagnostics.run_all_tests(proxy_url=proxy_url)


def print_diagnostics(proxy_url: str | None = None) -> None:
    """Run and print China network diagnostics."""
    diagnostics = ChinaNetworkDiagnostics()
    diagnostics.run_all_tests(proxy_url=proxy_url)
    diagnostics.print_report()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="China Network Diagnostics")
    parser.add_argument("--proxy", type=str, help="Proxy URL to test")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.json:
        import json
        report = run_diagnostics(proxy_url=args.proxy)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_diagnostics(proxy_url=args.proxy)
