from __future__ import annotations

import io
import sys

import requests

from utils.china_diagnostics import ChinaNetworkDiagnostics
from utils.china_diagnostics import TestResult as DiagnosticsResult


def test_run_all_tests_uses_dns_resolution(monkeypatch) -> None:
    diagnostics = ChinaNetworkDiagnostics()

    endpoint_calls: list[tuple[str, str, str]] = []
    dns_calls: list[tuple[str, str, str]] = []

    def _fake_endpoint(
        name: str,
        url: str,
        category: str = "connectivity",
        timeout: float = 5.0,
    ) -> DiagnosticsResult:
        _ = timeout
        endpoint_calls.append((name, url, category))
        return DiagnosticsResult(name=name, category=category, success=True, latency_ms=1.0)

    def _fake_dns(hostname: str, dns_server: str, dns_name: str) -> DiagnosticsResult:
        dns_calls.append((hostname, dns_server, dns_name))
        return DiagnosticsResult(
            name=f"DNS {dns_name}",
            category="dns",
            success=True,
            latency_ms=1.0,
        )

    monkeypatch.setattr(diagnostics, "test_endpoint", _fake_endpoint)
    monkeypatch.setattr(diagnostics, "test_dns_resolution", _fake_dns)

    report = diagnostics.run_all_tests()

    assert report["summary"]["dns_status"]["tested"] == len(diagnostics.CHINA_DNS)
    assert len(dns_calls) == len(diagnostics.CHINA_DNS)
    assert all(hostname == "www.eastmoney.com" for hostname, _, _ in dns_calls)
    assert endpoint_calls
    assert all(category == "financial" for _, _, category in endpoint_calls)


def test_test_endpoint_falls_back_to_get_when_head_is_blocked(monkeypatch) -> None:
    diagnostics = ChinaNetworkDiagnostics()

    class _Resp:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            _ = exc_type, exc, tb
            return False

    monkeypatch.setattr(requests, "head", lambda *args, **kwargs: _Resp(405))
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: _Resp(200))

    result = diagnostics.test_endpoint("EastMoney", "https://www.eastmoney.com")
    assert result.success is True
    assert result.error == ""


def test_print_report_is_safe_on_ascii_only_stdout(monkeypatch) -> None:
    diagnostics = ChinaNetworkDiagnostics()
    diagnostics.summary = {
        "timestamp": "2026-02-27T00:00:00",
        "network_mode": "china_direct",
        "overall_health": "fair",
        "financial_connectivity": {"success_rate": 0.5, "avg_latency_ms": 2500.0},
        "recommendations": [
            {
                "priority": "high",
                "issue": "DNS \u95ee\u9898",
                "suggestion": "\u8bf7\u66f4\u6362DNS\u670d\u52a1\u5668",
            }
        ],
    }
    diagnostics.results = [
        DiagnosticsResult(
            name="EastMoney",
            category="financial",
            success=False,
            error="\u8fde\u63a5\u5931\u8d25",
        )
    ]

    class _AsciiOnlyStdout(io.StringIO):
        @property
        def encoding(self) -> str:
            return "ascii"

        def write(self, s: str) -> int:
            s.encode("ascii")
            return super().write(s)

    fake_stdout = _AsciiOnlyStdout()
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    diagnostics.print_report()

    output = fake_stdout.getvalue()
    assert "[FAIL] EastMoney: N/A" in output
    assert "RECOMMENDATIONS:" in output
