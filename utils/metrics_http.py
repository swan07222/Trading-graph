# utils/metrics_http.py
from __future__ import annotations

import threading
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from utils.metrics import get_metrics

log = logging.getLogger(__name__)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for /metrics and /healthz endpoints."""

    # FIX #4: Limit request processing time
    timeout = 10

    def log_message(self, format, *args) -> None:
        """Silence default stdout logging — use our logger instead."""
        log.debug(format, *args)

    def _send_text(self, code: int, body: str, content_type: str = "text/plain") -> None:
        """Helper to send a text response with security headers."""
        encoded = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        # FIX #3: Security headers
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests."""
        if self.path == "/healthz":
            self._send_text(200, "ok\n")
            return

        if self.path == "/metrics":
            body = get_metrics().to_prometheus()
            self._send_text(
                200, body,
                content_type="text/plain; version=0.0.4",
            )
            return

        self._send_text(404, "Not Found\n")

    # FIX #5: Explicitly reject other HTTP methods
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
    """
    FIX #1, #6: Managed metrics HTTP server with start/stop lifecycle.

    Usage:
        server = MetricsServer(port=9090)
        server.start()
        # ... application runs ...
        server.stop()

    Or as context manager:
        with MetricsServer(port=9090) as server:
            # ... application runs ...
    """

    def __init__(
        self,
        port: int = 8000,
        host: str = "127.0.0.1",  # FIX #2: Bind to localhost by default
    ) -> None:
        self._host = host
        self._port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def start(self) -> None:
        """Start the metrics server in a background thread."""
        if self._started:
            log.warning("MetricsServer already started")
            return

        self._server = ThreadingHTTPServer(
            (self._host, self._port), MetricsHandler
        )
        # FIX #4: Set socket timeout to prevent slowloris
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
        """Stop the metrics server gracefully."""
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


# Convenience function — backward compatible but improved
def serve(port: int = 8000, host: str = "127.0.0.1") -> MetricsServer:
    """
    Start metrics server and return the server object.

    FIX #1: No longer blocks the caller. Returns server for lifecycle management.

    Args:
        port: Port to bind to (default: 8000)
        host: Host to bind to (default: 127.0.0.1)

    Returns:
        MetricsServer instance (already started)
    """
    server = MetricsServer(port=port, host=host)
    server.start()
    return server