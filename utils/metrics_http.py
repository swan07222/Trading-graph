from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from datetime import UTC, date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from utils.metrics import get_metrics

log = logging.getLogger(__name__)

_SNAPSHOT_LOCK = threading.RLock()
_SNAPSHOT_PROVIDERS: dict[str, Callable[[], Any]] = {}


def register_snapshot_provider(name: str, provider: Callable[[], Any]) -> None:
    """Register a callable that returns JSON-serializable snapshot data."""
    normalized = (name or "").strip()
    if not normalized:
        raise ValueError("Snapshot provider name cannot be empty")
    if not callable(provider):
        raise TypeError("Snapshot provider must be callable")

    with _SNAPSHOT_LOCK:
        _SNAPSHOT_PROVIDERS[normalized] = provider


def unregister_snapshot_provider(name: str) -> bool:
    with _SNAPSHOT_LOCK:
        return _SNAPSHOT_PROVIDERS.pop(name, None) is not None


def list_snapshot_providers() -> list[str]:
    with _SNAPSHOT_LOCK:
        return sorted(_SNAPSHOT_PROVIDERS.keys())


def _normalize_json(value: Any) -> Any:
    """Convert project types into plain JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _normalize_json(value.to_dict())

    if isinstance(value, dict):
        return {str(k): _normalize_json(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_normalize_json(v) for v in value]

    return str(value)


def _build_runtime_snapshot(limit: int = 20) -> dict[str, Any]:
    """Best-effort runtime snapshot from OMS for lightweight dashboards."""
    try:
        from trading.oms import get_oms

        oms = get_oms()
        account = oms.get_account()
        positions = oms.get_positions()
        orders = oms.get_orders()
        fills = oms.get_fills()
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}

    trimmed_limit = max(1, min(int(limit), 200))
    latest_orders = sorted(
        orders,
        key=lambda o: o.updated_at or o.created_at,
        reverse=True,
    )[:trimmed_limit]
    latest_fills = sorted(
        fills,
        key=lambda f: f.timestamp,
        reverse=True,
    )[:trimmed_limit]

    return {
        "status": "ok",
        "account": _normalize_json(account),
        "positions": _normalize_json(list(positions.values())),
        "active_orders": _normalize_json(latest_orders),
        "recent_fills": _normalize_json(latest_fills),
        "limits": {"records": trimmed_limit},
    }


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics, health, and lightweight JSON snapshots."""

    timeout = 10

    def log_message(self, fmt, *args) -> None:
        log.debug(fmt, *args)

    def _api_key(self) -> str:
        return os.environ.get("TRADING_HTTP_API_KEY", "").strip()

    def _is_api_authorized(self, query: dict[str, list[str]]) -> bool:
        expected = self._api_key()
        if not expected:
            return True

        from_header = self.headers.get("X-API-Key", "").strip()
        if from_header and from_header == expected:
            return True

        from_query = (query.get("api_key", [""])[0] or "").strip()
        return bool(from_query and from_query == expected)

    def _send_text(
        self,
        code: int,
        body: str,
        content_type: str = "text/plain",
    ) -> None:
        encoded = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        self._send_text(code, body + "\n", content_type="application/json")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/healthz":
            self._send_text(200, "ok\n")
            return

        if path == "/metrics":
            body = get_metrics().to_prometheus()
            self._send_text(
                200,
                body,
                content_type="text/plain; version=0.0.4",
            )
            return

        if path.startswith("/api/v1/"):
            if not self._is_api_authorized(query):
                self._send_json(
                    401,
                    {"error": "unauthorized", "message": "Invalid or missing API key"},
                )
                return
            self._handle_api(path, query)
            return

        self._send_text(404, "Not Found\n")

    def _handle_api(self, path: str, query: dict[str, list[str]]) -> None:
        if path == "/api/v1/providers":
            self._send_json(
                200,
                {
                    "providers": list_snapshot_providers(),
                    "generated_at": datetime.now(UTC).isoformat(),
                },
            )
            return

        if path.startswith("/api/v1/snapshot/"):
            name = path.rsplit("/", 1)[-1].strip()
            with _SNAPSHOT_LOCK:
                provider = _SNAPSHOT_PROVIDERS.get(name)
            if provider is None:
                self._send_json(
                    404,
                    {"error": "not_found", "message": f"Provider '{name}' not found"},
                )
                return

            try:
                payload = _normalize_json(provider())
                self._send_json(
                    200,
                    {
                        "provider": name,
                        "snapshot": payload,
                        "generated_at": datetime.now(UTC).isoformat(),
                    },
                )
            except Exception as exc:
                self._send_json(
                    500,
                    {"error": "provider_error", "message": str(exc), "provider": name},
                )
            return

        if path == "/api/v1/dashboard":
            try:
                limit = int(query.get("limit", ["20"])[0] or "20")
            except ValueError:
                limit = 20
            self._send_json(
                200,
                {
                    "snapshot": _build_runtime_snapshot(limit=limit),
                    "generated_at": datetime.now(UTC).isoformat(),
                },
            )
            return

        self._send_json(404, {"error": "not_found", "message": "Unknown endpoint"})

    def do_POST(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PUT(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PATCH(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        self._send_text(405, "Method Not Allowed\n")


class MetricsServer:
    """Managed metrics/API HTTP server with start/stop lifecycle."""

    def __init__(self, port: int = 8000, host: str = "127.0.0.1") -> None:
        self._host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            log.warning("MetricsServer already started")
            return

        self._server = ThreadingHTTPServer((self._host, self._port), MetricsHandler)
        self._port = int(self._server.server_port)
        self._server.timeout = 10

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name=f"metrics-http-{self._port}",
        )
        self._thread.start()
        self._started = True
        log.info("Metrics server started on %s:%d", self._host, self._port)

    def stop(self) -> None:
        if not self._started:
            return

        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        self._started = False
        log.info("Metrics server stopped")

    @property
    def is_running(self) -> bool:
        return self._started

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def __enter__(self) -> MetricsServer:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def __repr__(self) -> str:
        state = "running" if self._started else "stopped"
        return f"MetricsServer({self._host}:{self._port}, {state})"


def serve(port: int = 8000, host: str = "127.0.0.1") -> MetricsServer:
    """Start metrics server and return a running server instance."""
    server = MetricsServer(port=port, host=host)
    server.start()
    return server
