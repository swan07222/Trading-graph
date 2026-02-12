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
            assert False, "Expected unauthorized response"
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
