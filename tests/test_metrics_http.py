import json
import urllib.error
import urllib.request

from utils import metrics_http


def _fetch_json(url: str, headers: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=5) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


def test_metrics_http_snapshot_provider():
    metrics_http.unregister_snapshot_provider("unit_test")
    metrics_http.register_snapshot_provider("unit_test", lambda: {"value": 42})

    server = metrics_http.MetricsServer(port=0, host="127.0.0.1")
    server.start()
    try:
        status, providers = _fetch_json(f"{server.url}/api/v1/providers")
        assert status == 200
        assert "unit_test" in providers["providers"]

        status, snap = _fetch_json(f"{server.url}/api/v1/snapshot/unit_test")
        assert status == 200
        assert snap["provider"] == "unit_test"
        assert snap["snapshot"]["value"] == 42
    finally:
        server.stop()
        metrics_http.unregister_snapshot_provider("unit_test")


def test_metrics_http_api_key_guard(monkeypatch):
    metrics_http.unregister_snapshot_provider("secure_test")
    metrics_http.register_snapshot_provider("secure_test", lambda: {"ok": True})
    monkeypatch.setenv("TRADING_HTTP_API_KEY", "secret-key")

    server = metrics_http.MetricsServer(port=0, host="127.0.0.1")
    server.start()
    try:
        req = urllib.request.Request(f"{server.url}/api/v1/providers")
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("Expected unauthorized response")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        status, providers = _fetch_json(
            f"{server.url}/api/v1/providers",
            headers={"X-API-Key": "secret-key"},
        )
        assert status == 200
        assert "secure_test" in providers["providers"]
    finally:
        server.stop()
        metrics_http.unregister_snapshot_provider("secure_test")


def test_metrics_http_operational_snapshots():
    server = metrics_http.MetricsServer(port=0, host="127.0.0.1")
    server.start()
    try:
        status, alerts = _fetch_json(f"{server.url}/api/v1/alerts/stats")
        assert status == 200
        assert "snapshot" in alerts
        assert alerts["snapshot"]["status"] in {"ok", "unavailable"}

        status, policy = _fetch_json(f"{server.url}/api/v1/governance/policy")
        assert status == 200
        assert "snapshot" in policy
        assert policy["snapshot"]["status"] in {"ok", "unavailable"}

        status, marketplace = _fetch_json(f"{server.url}/api/v1/strategy/marketplace")
        assert status == 200
        assert "snapshot" in marketplace
        assert marketplace["snapshot"]["status"] in {"ok", "unavailable"}
    finally:
        server.stop()


def test_metrics_http_extended_telemetry_endpoints(monkeypatch):
    monkeypatch.setattr(
        metrics_http,
        "_build_sentiment_snapshot",
        lambda stock_code=None, hours_lookback=24: {  # noqa: ARG005
            "status": "ok",
            "snapshot": {"scope": stock_code or "market", "hours": hours_lookback},
        },
    )

    server = metrics_http.MetricsServer(port=0, host="127.0.0.1")
    server.start()
    try:
        status, api_index = _fetch_json(f"{server.url}/api/v1")
        assert status == 200
        assert "/api/v1/execution" in api_index["routes"]
        assert "/api/v1/risk/metrics" in api_index["routes"]
        assert "/api/v1/sentiment" in api_index["routes"]
        assert "/api/v1/data/cache" in api_index["routes"]
        assert "/api/v1/data/feeds" in api_index["routes"]
        assert "/api/v1/learning/status" in api_index["routes"]
        assert "/api/v1/scanner/status" in api_index["routes"]

        status, execution = _fetch_json(f"{server.url}/api/v1/execution")
        assert status == 200
        assert "snapshot" in execution
        assert execution["snapshot"]["status"] in {"ok", "unavailable"}

        status, execution_quality = _fetch_json(f"{server.url}/api/v1/execution/quality")
        assert status == 200
        assert "snapshot" in execution_quality
        assert execution_quality["snapshot"]["status"] in {"ok", "unavailable"}

        status, risk = _fetch_json(f"{server.url}/api/v1/risk/metrics")
        assert status == 200
        assert "snapshot" in risk
        assert risk["snapshot"]["status"] in {"ok", "unavailable"}

        status, health = _fetch_json(f"{server.url}/api/v1/health")
        assert status == 200
        assert "snapshot" in health
        assert health["snapshot"]["status"] in {"ok", "unavailable"}

        status, sentiment = _fetch_json(f"{server.url}/api/v1/sentiment?symbol=600519&hours=12")
        assert status == 200
        assert sentiment["scope"] == "600519"
        assert sentiment["hours_lookback"] == 12
        assert sentiment["snapshot"]["status"] == "ok"

        status, cache = _fetch_json(f"{server.url}/api/v1/data/cache")
        assert status == 200
        assert "snapshot" in cache
        assert cache["snapshot"]["status"] in {"ok", "unavailable"}

        status, feeds = _fetch_json(f"{server.url}/api/v1/data/feeds")
        assert status == 200
        assert "snapshot" in feeds
        assert feeds["snapshot"]["status"] in {"ok", "unavailable"}

        status, learning = _fetch_json(f"{server.url}/api/v1/learning/status")
        assert status == 200
        assert "snapshot" in learning
        assert learning["snapshot"]["status"] in {"ok", "unavailable"}

        status, scanner = _fetch_json(f"{server.url}/api/v1/scanner/status")
        assert status == 200
        assert "snapshot" in scanner
        assert scanner["snapshot"]["status"] in {"ok", "unavailable"}

        status, full = _fetch_json(f"{server.url}/api/v1/dashboard/full?limit=5&hours=8")
        assert status == 200
        assert "snapshot" in full
        assert "execution" in full["snapshot"]
        assert "risk" in full["snapshot"]
        assert "health" in full["snapshot"]
        assert "sentiment" in full["snapshot"]
        assert "cache" in full["snapshot"]
        assert "feeds" in full["snapshot"]
        assert "learning" in full["snapshot"]
        assert "scanner" in full["snapshot"]
    finally:
        server.stop()
