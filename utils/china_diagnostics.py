"""China Network Diagnostics Utility.

Provides network testing for mainland China users:
- Chinese financial endpoint connectivity
- DNS resolution checks
- Optional proxy connectivity checks
- Actionable recommendations
"""

from __future__ import annotations

import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TestResult:
    """Result of a single network check."""

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
    """Comprehensive network diagnostics for China users."""

    FINANCIAL_ENDPOINTS: dict[str, list[str]] = {
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

    CHINA_DNS: list[tuple[str, str]] = [
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
        """Test endpoint connectivity."""
        import requests

        start = time.time()
        success = False
        error = ""

        try:
            if url.startswith("http"):
                response = requests.head(url, timeout=timeout, allow_redirects=True)
                if response.status_code in (200, 301, 302):
                    success = True
                elif response.status_code in (403, 405, 429):
                    # A few providers reject HEAD; fallback to lightweight GET.
                    with requests.get(
                        url,
                        timeout=timeout,
                        allow_redirects=True,
                        stream=True,
                    ) as get_response:
                        success = 200 <= get_response.status_code < 400
                        if not success:
                            error = f"HTTP {get_response.status_code}"
                else:
                    error = f"HTTP {response.status_code}"
            else:
                with socket.create_connection((url, 443), timeout=timeout):
                    success = True
        except requests.exceptions.Timeout:
            error = "Timeout"
        except requests.exceptions.ConnectionError as exc:
            error = f"Connection error: {str(exc)[:50]}"
        except TimeoutError:
            error = "Socket timeout"
        except Exception as exc:  # pragma: no cover - defensive fallback
            error = str(exc)[:100]

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
        success = False
        resolved_ip = ""
        error = ""

        try:
            resolved_ip = self._resolve_dns(hostname, dns_server)
            success = bool(resolved_ip)
            if not success:
                error = "No A record returned"
        except socket.gaierror as exc:
            error = f"DNS resolution failed: {str(exc)[:50]}"
        except Exception as exc:  # pragma: no cover - defensive fallback
            error = str(exc)[:100]

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

    def _resolve_dns(self, hostname: str, dns_server: str) -> str:
        """Resolve hostname via dnspython if available, else system resolver."""
        try:
            import dns.resolver  # type: ignore[import-untyped]
        except Exception:
            return socket.gethostbyname(hostname)

        resolver = dns.resolver.Resolver(configure=False)
        resolver.nameservers = [dns_server]
        resolver.timeout = 3.0
        resolver.lifetime = 3.0
        answer = resolver.resolve(hostname, "A")
        if not answer:
            return ""
        return answer[0].to_text()

    def test_proxy(
        self,
        proxy_url: str,
        test_url: str = "https://www.eastmoney.com",
    ) -> TestResult:
        """Test proxy connectivity."""
        import requests

        start = time.time()
        success = False
        error = ""

        try:
            proxies = {"http": proxy_url, "https": proxy_url}
            response = requests.get(test_url, proxies=proxies, timeout=10.0)
            success = response.status_code == 200
            if not success:
                error = f"HTTP {response.status_code}"
        except requests.exceptions.Timeout:
            error = "Proxy timeout"
        except requests.exceptions.ProxyError as exc:
            error = f"Proxy error: {str(exc)[:50]}"
        except Exception as exc:  # pragma: no cover - defensive fallback
            error = str(exc)[:100]

        latency_ms = (time.time() - start) * 1000
        return TestResult(
            name="Proxy Test",
            category="proxy",
            success=success,
            latency_ms=latency_ms,
            error=error,
            details={"proxy_url": proxy_url, "test_url": test_url},
        )

    def run_all_tests(self, proxy_url: str | None = None) -> dict[str, Any]:
        """Run all diagnostics and return a full report."""
        self.results = []

        log.info("Testing Chinese financial endpoints...")
        for provider, urls in self.FINANCIAL_ENDPOINTS.items():
            for url in urls:
                self.results.append(self.test_endpoint(provider, url, "financial"))

        log.info("Testing DNS resolution...")
        for dns_ip, dns_name in self.CHINA_DNS:
            self.results.append(
                self.test_dns_resolution(
                    hostname="www.eastmoney.com",
                    dns_server=dns_ip,
                    dns_name=dns_name,
                )
            )

        if proxy_url:
            log.info("Testing proxy...")
            self.results.append(self.test_proxy(proxy_url))

        self._generate_summary()
        return self.get_report()

    def _generate_summary(self) -> None:
        """Build summary statistics and recommendations."""
        financial_results = [r for r in self.results if r.category == "financial"]
        dns_results = [r for r in self.results if r.category == "dns"]
        proxy_results = [r for r in self.results if r.category == "proxy"]

        financial_success = sum(1 for r in financial_results if r.success)
        financial_total = len(financial_results)
        financial_rate = financial_success / financial_total if financial_total else 0.0

        financial_latencies = [r.latency_ms for r in financial_results if r.success]
        avg_financial_latency = (
            sum(financial_latencies) / len(financial_latencies)
            if financial_latencies
            else 0.0
        )

        recommendations: list[dict[str, str]] = []
        if financial_rate < 0.5:
            recommendations.append(
                {
                    "priority": "high",
                    "issue": "Poor connectivity to Chinese financial providers",
                    "suggestion": (
                        "Check firewall settings and try AliDNS (223.5.5.5) "
                        "or 114DNS (114.114.114.114)."
                    ),
                }
            )
        if avg_financial_latency > 2000:
            recommendations.append(
                {
                    "priority": "medium",
                    "issue": f"High latency to financial endpoints ({avg_financial_latency:.0f}ms)",
                    "suggestion": "Consider a domestic proxy/CDN and reduce concurrent requests.",
                }
            )
        if dns_results and sum(1 for r in dns_results if r.success) < (len(dns_results) / 2):
            recommendations.append(
                {
                    "priority": "high",
                    "issue": "DNS resolution issues detected",
                    "suggestion": "Switch DNS servers to AliDNS or 114DNS.",
                }
            )

        if financial_rate > 0.7:
            overall = "good"
        elif financial_rate > 0.4:
            overall = "fair"
        else:
            overall = "poor"

        self.summary = {
            "timestamp": datetime.now().isoformat(),
            "network_mode": "china_direct",
            "financial_connectivity": {
                "success_rate": round(financial_rate, 4),
                "successful": financial_success,
                "total": financial_total,
                "avg_latency_ms": round(avg_financial_latency, 2),
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
            "overall_health": overall,
        }

    def get_report(self) -> dict[str, Any]:
        """Return the full diagnostics report."""
        return {
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
        }

    @staticmethod
    def _safe_terminal_text(text: str) -> str:
        """Normalize text so it can be printed on non-UTF terminals."""
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            text.encode(encoding)
            return text
        except UnicodeEncodeError:
            return text.encode(encoding, errors="replace").decode(encoding, errors="replace")

    def _print_line(self, text: str = "") -> None:
        print(self._safe_terminal_text(text))

    def print_report(self) -> None:
        """Print a formatted diagnostics report."""
        if not self.summary:
            self._generate_summary()

        self._print_line("\n" + "=" * 60)
        self._print_line("CHINA NETWORK DIAGNOSTICS REPORT")
        self._print_line("=" * 60)
        self._print_line(f"Timestamp: {self.summary.get('timestamp', 'N/A')}")
        self._print_line(f"Network Mode: {self.summary.get('network_mode', 'unknown')}")
        self._print_line(f"Overall Health: {self.summary.get('overall_health', 'unknown')}")
        self._print_line()

        fin = self.summary.get("financial_connectivity", {})
        self._print_line("FINANCIAL ENDPOINTS:")
        self._print_line(f"  Success Rate: {fin.get('success_rate', 0) * 100:.1f}%")
        self._print_line(f"  Avg Latency: {fin.get('avg_latency_ms', 0):.0f}ms")
        self._print_line()

        recommendations = self.summary.get("recommendations", [])
        if recommendations:
            self._print_line("RECOMMENDATIONS:")
            for rec in recommendations:
                self._print_line(f"  [{rec['priority'].upper()}] {rec['issue']}")
                self._print_line(f"      -> {rec['suggestion']}")
        else:
            self._print_line("RECOMMENDATIONS: None - Network configuration looks good.")

        self._print_line()
        self._print_line("=" * 60)

        self._print_line("\nDETAILED RESULTS:")
        self._print_line("-" * 60)
        for result in self.results:
            status = "[OK]" if result.success else "[FAIL]"
            latency_str = f"{result.latency_ms:.0f}ms" if result.success else "N/A"
            self._print_line(f"{status} {result.name}: {latency_str} {result.error}")


def run_diagnostics(proxy_url: str | None = None) -> dict[str, Any]:
    """Run China network diagnostics."""
    diagnostics = ChinaNetworkDiagnostics()
    return diagnostics.run_all_tests(proxy_url=proxy_url)


def print_diagnostics(proxy_url: str | None = None) -> None:
    """Run China diagnostics and print formatted output."""
    diagnostics = ChinaNetworkDiagnostics()
    diagnostics.run_all_tests(proxy_url=proxy_url)
    diagnostics.print_report()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="China Network Diagnostics")
    parser.add_argument("--proxy", type=str, help="Proxy URL to test")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.json:
        report = run_diagnostics(proxy_url=args.proxy)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_diagnostics(proxy_url=args.proxy)
