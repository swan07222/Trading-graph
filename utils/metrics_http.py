from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from datetime import date, datetime, timezone
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


def _parse_int_query(
    query: dict[str, list[str]],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        raw = int(query.get(key, [str(default)])[0] or str(default))
    except (TypeError, ValueError):
        raw = int(default)
    return max(int(minimum), min(int(maximum), int(raw)))


def _build_runtime_snapshot(limit: int = 20) -> dict[str, Any]:
    """Best-effort runtime snapshot for analysis-only dashboards."""
    _ = limit
    execution = _build_execution_engine_snapshot()
    if execution.get("status") != "ok":
        return {
            "status": "unavailable",
            "reason": "runtime_provider_not_available",
        }

    snapshot = execution.get("snapshot")
    if not isinstance(snapshot, dict):
        return {
            "status": "unavailable",
            "reason": "invalid_runtime_snapshot",
        }

    return {
        "status": "ok",
        "runtime": _normalize_json(snapshot),
    }


def _build_alert_snapshot(limit: int = 20) -> dict[str, Any]:
    """Best-effort alert pipeline snapshot."""
    try:
        from trading.alerts import get_alert_manager

        mgr = get_alert_manager()
        pending = mgr.get_pending()
        history = mgr.get_history(limit=max(1, min(int(limit), 200)))
        stats = mgr.get_alert_stats()
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}

    return {
        "status": "ok",
        "pending": _normalize_json(pending),
        "history": _normalize_json(history),
        "stats": _normalize_json(stats),
    }


def _build_policy_snapshot() -> dict[str, Any]:
    """Best-effort governance policy snapshot."""
    try:
        from utils.policy import get_trade_policy_engine

        engine = get_trade_policy_engine()
        raw = getattr(engine, "_policy", {}) or {}
        return {
            "status": "ok",
            "policy_path": str(getattr(engine, "path", "")),
            "policy": _normalize_json(raw),
        }
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _build_strategy_marketplace_snapshot() -> dict[str, Any]:
    """Best-effort strategy marketplace status."""
    try:
        from analysis.strategy_marketplace import StrategyMarketplace

        marketplace = StrategyMarketplace()
        entries = marketplace.list_entries()
        enabled = marketplace.get_enabled_ids()
        summary = marketplace.get_integrity_summary()
        return {
            "status": "ok",
            "entries": _normalize_json(entries),
            "enabled_ids": _normalize_json(enabled),
            "integrity": _normalize_json(summary),
        }
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _build_execution_engine_snapshot() -> dict[str, Any]:
    """Best-effort execution engine snapshot via registered provider."""
    provider = None
    with _SNAPSHOT_LOCK:
        provider = _SNAPSHOT_PROVIDERS.get("execution_engine")
    if provider is None:
        return {
            "status": "unavailable",
            "reason": "provider_not_registered",
            "provider": "execution_engine",
        }
    try:
        return {
            "status": "ok",
            "provider": "execution_engine",
            "snapshot": _normalize_json(provider()),
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": str(exc),
            "provider": "execution_engine",
        }


def _build_execution_quality_snapshot() -> dict[str, Any]:
    """Extract execution quality block from execution snapshot."""
    execution = _build_execution_engine_snapshot()
    if execution.get("status") != "ok":
        return execution
    snap = execution.get("snapshot")
    if not isinstance(snap, dict):
        return {"status": "unavailable", "reason": "invalid_execution_snapshot"}
    quality = snap.get("execution_quality")
    if not isinstance(quality, dict):
        return {"status": "unavailable", "reason": "missing_execution_quality"}
    return {"status": "ok", "snapshot": _normalize_json(quality)}


def _build_risk_snapshot() -> dict[str, Any]:
    """Risk manager endpoint is retained for compatibility only."""
    return {
        "status": "unavailable",
        "reason": "risk_management_removed",
    }


def _build_health_snapshot() -> dict[str, Any]:
    """Best-effort system health snapshot."""
    try:
        from trading.health import get_health_monitor

        monitor = get_health_monitor()
        health = monitor.get_health()
        payload = health.to_dict() if hasattr(health, "to_dict") else health
        return {"status": "ok", "snapshot": _normalize_json(payload)}
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _build_sentiment_snapshot(
    stock_code: str | None = None,
    hours_lookback: int = 24,
) -> dict[str, Any]:
    """Best-effort weighted sentiment/news snapshot."""
    try:
        from data.news import get_news_aggregator

        agg = get_news_aggregator()
        snapshot = agg.get_institutional_snapshot(
            stock_code=stock_code,
            hours_lookback=max(1, min(int(hours_lookback), 168)),
        )
        return {"status": "ok", "snapshot": _normalize_json(snapshot)}
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _build_cache_snapshot() -> dict[str, Any]:
    """Best-effort cache health/efficiency snapshot."""
    try:
        from data.cache import get_cache

        cache = get_cache()
        stats = cache.get_stats()
        l1 = getattr(cache, "_l1", None)
        l1_items = len(l1) if l1 is not None and hasattr(l1, "__len__") else 0
        l1_size_mb = (
            float(getattr(l1, "size_mb", 0.0) or 0.0)
            if l1 is not None
            else 0.0
        )
        payload = {
            "l1_items": int(l1_items),
            "l1_size_mb": float(l1_size_mb),
            "hits": int(getattr(stats, "total_hits", 0)),
            "misses": int(getattr(stats, "total_misses", 0)),
            "hit_rate": float(getattr(stats, "hit_rate", 0.0)),
            "l1_hits": int(getattr(stats, "l1_hits", 0)),
            "l2_hits": int(getattr(stats, "l2_hits", 0)),
            "l3_hits": int(getattr(stats, "l3_hits", 0)),
            "sets": int(getattr(stats, "total_sets", 0)),
            "evictions": int(getattr(stats, "total_evictions", 0)),
        }
        return {"status": "ok", "snapshot": _normalize_json(payload)}
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _build_feed_snapshot() -> dict[str, Any]:
    """Best-effort data-feed runtime snapshot."""
    try:
        from data.feeds import get_feed_manager

        mgr = get_feed_manager(auto_init=False)
        active = getattr(mgr, "_active_feed", None)
        feeds = {}
        for name, feed in dict(getattr(mgr, "_feeds", {}) or {}).items():
            status_obj = getattr(feed, "status", None)
            status_val = getattr(status_obj, "value", status_obj)
            feeds[str(name)] = {
                "status": str(status_val or "unknown"),
                "running": bool(getattr(feed, "_running", False)),
            }
        payload = {
            "active_feed": str(getattr(active, "name", "none") or "none"),
            "subscriptions": int(len(getattr(mgr, "_subscriptions", set()) or set())),
            "feeds": feeds,
        }
        return {"status": "ok", "snapshot": _normalize_json(payload)}
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _build_learning_snapshot() -> dict[str, Any]:
    """Best-effort learner status from provider or persisted learner state."""
    try:
        provider = None
        with _SNAPSHOT_LOCK:
            provider = _SNAPSHOT_PROVIDERS.get("auto_learner")
        if callable(provider):
            return {"status": "ok", "snapshot": _normalize_json(provider())}
    except Exception:
        pass

    try:
        from config.settings import CONFIG

        state_path = CONFIG.data_dir / "learner_state.json"
        if not state_path.exists():
            return {
                "status": "unavailable",
                "reason": "learner_state_not_found",
            }
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        data = raw.get("_data", raw) if isinstance(raw, dict) else {}
        if not isinstance(data, dict):
            return {"status": "unavailable", "reason": "invalid_learner_state"}
        payload = {
            "total_sessions": int(data.get("total_sessions", 0) or 0),
            "total_stocks": int(data.get("total_stocks", 0) or 0),
            "total_hours": float(data.get("total_hours", 0.0) or 0.0),
            "best_accuracy": float(data.get("best_accuracy", 0.0) or 0.0),
            "last_interval": str(data.get("last_interval", "") or ""),
            "last_horizon": int(data.get("last_horizon", 0) or 0),
            "holdout_size": int(len(data.get("holdout_codes", []) or [])),
            "updated_at": str(data.get("saved_at", "") or ""),
        }
        return {"status": "ok", "snapshot": _normalize_json(payload)}
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _build_scanner_snapshot() -> dict[str, Any]:
    """Best-effort scanner/auto-trade status snapshot."""
    try:
        from config.settings import CONFIG

        execution = _build_execution_engine_snapshot()
        auto_state: dict[str, Any] = {}
        if execution.get("status") == "ok":
            snap = execution.get("snapshot")
            if isinstance(snap, dict):
                auto_state = dict(snap.get("auto_trade_state") or {})
        payload = {
            "scan_interval_seconds": int(
                getattr(getattr(CONFIG, "auto_trade", None), "scan_interval_seconds", 0)
                or 0
            ),
            "min_confidence": float(
                getattr(getattr(CONFIG, "auto_trade", None), "min_confidence", 0.0)
                or 0.0
            ),
            "min_signal_strength": float(
                getattr(
                    getattr(CONFIG, "auto_trade", None),
                    "min_signal_strength",
                    0.0,
                )
                or 0.0
            ),
            "runtime": auto_state,
        }
        return {"status": "ok", "snapshot": _normalize_json(payload)}
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _api_index_payload() -> dict[str, Any]:
    return {
        "version": "v1",
        "routes": [
            "/api/v1/providers",
            "/api/v1/snapshot/{name}",
            "/api/v1/dashboard",
            "/api/v1/dashboard/full",
            "/api/v1/alerts/stats",
            "/api/v1/governance/policy",
            "/api/v1/strategy/marketplace",
            "/api/v1/execution",
            "/api/v1/execution/quality",
            "/api/v1/risk/metrics",
            "/api/v1/health",
            "/api/v1/sentiment",
            "/api/v1/data/cache",
            "/api/v1/data/feeds",
            "/api/v1/learning/status",
            "/api/v1/scanner/status",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics, health, and lightweight JSON snapshots."""

    timeout = 10

    def log_message(self, fmt: str, *args: object) -> None:
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

        # FIX: Handle case where query.get returns empty list
        api_key_list = query.get("api_key", [""])
        from_query = (api_key_list[0] if api_key_list else "" or "").strip()
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

        if path == "/api/v1" or path.startswith("/api/v1/"):
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
        if path in ("/api/v1", "/api/v1/"):
            self._send_json(200, _api_index_payload())
            return

        if path == "/api/v1/providers":
            self._send_json(
                200,
                {
                    "providers": list_snapshot_providers(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
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
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except Exception as exc:
                self._send_json(
                    500,
                    {"error": "provider_error", "message": str(exc), "provider": name},
                )
            return

        if path == "/api/v1/dashboard":
            limit = _parse_int_query(query, key="limit", default=20, minimum=1, maximum=200)
            self._send_json(
                200,
                {
                    "snapshot": _build_runtime_snapshot(limit=limit),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        if path == "/api/v1/dashboard/full":
            limit = _parse_int_query(query, key="limit", default=20, minimum=1, maximum=200)
            hours = _parse_int_query(query, key="hours", default=24, minimum=1, maximum=168)
            # FIX: Handle case where query.get returns empty list
            symbol_list = query.get("symbol", [""])
            dashboard_symbol: str | None = (symbol_list[0] if symbol_list else "" or "").strip()
            if not dashboard_symbol:
                dashboard_symbol = None
            self._send_json(
                200,
                {
                    "snapshot": {
                        "runtime": _build_runtime_snapshot(limit=limit),
                        "alerts": _build_alert_snapshot(limit=limit),
                        "policy": _build_policy_snapshot(),
                        "strategy_marketplace": _build_strategy_marketplace_snapshot(),
                        "execution": _build_execution_engine_snapshot(),
                        "execution_quality": _build_execution_quality_snapshot(),
                        "risk": _build_risk_snapshot(),
                        "health": _build_health_snapshot(),
                        "sentiment": _build_sentiment_snapshot(
                            stock_code=dashboard_symbol,
                            hours_lookback=hours,
                        ),
                        "cache": _build_cache_snapshot(),
                        "feeds": _build_feed_snapshot(),
                        "learning": _build_learning_snapshot(),
                        "scanner": _build_scanner_snapshot(),
                    },
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        if path == "/api/v1/alerts/stats":
            limit = _parse_int_query(query, key="limit", default=20, minimum=1, maximum=200)
            self._send_json(
                200,
                {
                    "snapshot": _build_alert_snapshot(limit=limit),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        if path == "/api/v1/execution":
            self._send_json(
                200,
                {
                    "snapshot": _build_execution_engine_snapshot(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        if path == "/api/v1/execution/quality":
            self._send_json(
                200,
                {
                    "snapshot": _build_execution_quality_snapshot(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        if path == "/api/v1/risk/metrics":
            self._send_json(
                200,
                {
                    "snapshot": _build_risk_snapshot(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        if path == "/api/v1/health":
            self._send_json(
                200,
                {
                    "snapshot": _build_health_snapshot(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        if path == "/api/v1/sentiment":
            hours = _parse_int_query(query, key="hours", default=24, minimum=1, maximum=168)
            # FIX: Handle case where query.get returns empty list
            symbol_list = query.get("symbol", [""])
            sentiment_symbol: str | None = (symbol_list[0] if symbol_list else "" or "").strip()
            if not sentiment_symbol:
                sentiment_symbol = None
            self._send_json(
                200,
                {
                    "scope": sentiment_symbol or "market",
                    "hours_lookback": hours,
                    "snapshot": _build_sentiment_snapshot(
                        stock_code=sentiment_symbol,
                        hours_lookback=hours,
                    ),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        if path == "/api/v1/data/cache":
            self._send_json(
                200,
                {
                    "snapshot": _build_cache_snapshot(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        if path == "/api/v1/data/feeds":
            self._send_json(
                200,
                {
                    "snapshot": _build_feed_snapshot(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        if path == "/api/v1/learning/status":
            self._send_json(
                200,
                {
                    "snapshot": _build_learning_snapshot(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        if path == "/api/v1/scanner/status":
            self._send_json(
                200,
                {
                    "snapshot": _build_scanner_snapshot(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        if path == "/api/v1/governance/policy":
            self._send_json(
                200,
                {
                    "snapshot": _build_policy_snapshot(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        if path == "/api/v1/strategy/marketplace":
            self._send_json(
                200,
                {
                    "snapshot": _build_strategy_marketplace_snapshot(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
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

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def __repr__(self) -> str:
        state = "running" if self._started else "stopped"
        return f"MetricsServer({self._host}:{self._port}, {state})"


def serve(port: int = 8000, host: str = "127.0.0.1") -> MetricsServer:
    """Start metrics server and return a running server instance."""
    server = MetricsServer(port=port, host=host)
    server.start()
    return server
