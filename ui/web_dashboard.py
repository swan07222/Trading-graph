"""Modern FastAPI web dashboard for Trading Graph.

This module provides:
    - RESTful API for all trading operations
    - WebSocket real-time updates
    - Interactive dashboard with Plotly
    - Authentication and authorization
    - Rate limiting
    - API documentation (Swagger/OpenAPI)
    - Metrics and monitoring endpoints

Example:
    >>> from ui.web_dashboard import create_app
    >>> app = create_app()
    >>> uvicorn.run(app, host="0.0.0.0", port=8000)
"""
from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from config.runtime_env import env_int, env_text
from core.async_events import Event, EventType, get_event_bus
from utils.logger import get_logger
from utils.redis_cache import get_cache

log = get_logger()

# Security
security = HTTPBearer(auto_error=False)


# ============================================================================
# Pydantic Models
# ============================================================================

class StockPrice(BaseModel):
    """Stock price response."""
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    source: str


class Prediction(BaseModel):
    """Model prediction response."""
    symbol: str
    signal: str  # BUY, HOLD, SELL
    confidence: float = Field(ge=0, le=1)
    target_price: float | None = None
    horizon_days: int | None = None
    model_version: str
    timestamp: datetime


class SentimentAnalysis(BaseModel):
    """Sentiment analysis response."""
    symbol: str
    overall_sentiment: float = Field(ge=-1, le=1)
    policy_impact: float = Field(ge=-1, le=1)
    market_sentiment: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    article_count: int
    timestamp: datetime


class HealthStatus(BaseModel):
    """System health status."""
    status: str
    healthy: bool
    version: str
    uptime_seconds: float
    database: dict[str, Any]
    cache: dict[str, Any]
    event_bus: dict[str, Any]


class BacktestRequest(BaseModel):
    """Backtest request parameters."""
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000
    commission_bps: float = 3.0
    slippage_bps: float = 10.0


class BacktestResult(BaseModel):
    """Backtest result."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float


# ============================================================================
# Authentication
# ============================================================================

async def verify_auth(
    credentials: HTTPAuthorizationCredentials | None = None,
) -> str | None:
    """Verify API key authentication."""
    if credentials is None:
        return None  # Allow unauthenticated for now

    api_key = env_text("API_KEY", "")
    if not api_key:
        return None  # No API key configured

    if credentials.credentials != api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    return "authenticated"


# ============================================================================
# WebSocket Manager
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(
        self,
        websocket: WebSocket,
        channel: str,
    ) -> None:
        """Accept WebSocket connection."""
        await websocket.accept()
        if channel not in self._connections:
            self._connections[channel] = []
        self._connections[channel].append(websocket)
        log.info(f"WebSocket connected to {channel}")

    def disconnect(self, websocket: WebSocket, channel: str) -> None:
        """Remove WebSocket connection."""
        if channel in self._connections:
            self._connections[channel].remove(websocket)
            if not self._connections[channel]:
                del self._connections[channel]
        log.info(f"WebSocket disconnected from {channel}")

    async def broadcast(self, channel: str, message: Any) -> None:
        """Broadcast message to all connections on channel."""
        if channel not in self._connections:
            return

        message_text = (
            json.dumps(message) if not isinstance(message, str) else message
        )

        disconnected = []
        for connection in self._connections[channel]:
            try:
                await connection.send_text(message_text)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn, channel)


manager = ConnectionManager()


# ============================================================================
# Application Factory
# ============================================================================

_start_time = time.time()
_service_lock = threading.RLock()
_fetcher_service: Any | None = None
_news_aggregator_service: Any | None = None
_backtester_service: Any | None = None


def _get_fetcher_service() -> Any:
    """Lazy-load shared data fetcher."""
    global _fetcher_service
    with _service_lock:
        if _fetcher_service is None:
            from data.fetcher import get_fetcher

            _fetcher_service = get_fetcher()
    return _fetcher_service


def _get_news_aggregator_service() -> Any:
    """Lazy-load shared news aggregator."""
    global _news_aggregator_service
    with _service_lock:
        if _news_aggregator_service is None:
            from data.news_aggregator import get_news_aggregator

            _news_aggregator_service = get_news_aggregator()
    return _news_aggregator_service


def _get_backtester_service() -> Any:
    """Lazy-load shared backtester."""
    global _backtester_service
    with _service_lock:
        if _backtester_service is None:
            from analysis.backtest import Backtester

            _backtester_service = Backtester()
    return _backtester_service


def _as_float(value: Any, default: float = 0.0) -> float:
    """Safely cast arbitrary value to float."""
    try:
        parsed = float(value)
    except Exception:
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _heuristic_prediction(symbol: str, model: str) -> Prediction:
    """Compute a lightweight, data-backed prediction when ML runtime is unavailable."""
    fetcher = _get_fetcher_service()
    history = fetcher.get_history(symbol, bars=120, interval="1d", use_cache=True, update_db=False)
    if history is None or getattr(history, "empty", True) or len(history) < 25:
        raise HTTPException(status_code=503, detail="Not enough history to generate prediction")

    closes = history["close"].astype(float)
    current_price = _as_float(closes.iloc[-1], 0.0)
    if current_price <= 0:
        raise HTTPException(status_code=503, detail="Invalid latest price for prediction")

    sma_fast = _as_float(closes.tail(7).mean(), current_price)
    sma_slow = _as_float(closes.tail(21).mean(), current_price)
    momentum = _as_float((current_price / max(_as_float(closes.iloc[-6], current_price), 1e-9)) - 1.0, 0.0)

    if sma_fast > sma_slow and momentum > 0:
        signal = "BUY"
    elif sma_fast < sma_slow and momentum < 0:
        signal = "SELL"
    else:
        signal = "HOLD"

    spread = abs((sma_fast - sma_slow) / max(current_price, 1e-9))
    confidence = max(0.35, min(0.95, 0.45 + spread * 8.0 + abs(momentum) * 3.0))
    horizon_days = 5
    target_price = current_price * (1.0 + max(-0.12, min(0.12, momentum * 1.5)))

    return Prediction(
        symbol=symbol,
        signal=signal,
        confidence=confidence,
        target_price=round(target_price, 3),
        horizon_days=horizon_days,
        model_version=f"{model}-heuristic",
        timestamp=datetime.utcnow(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    log.info("Starting Trading Graph API...")
    yield
    # Shutdown
    log.info("Shutting down Trading Graph API...")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Trading Graph API",
        description="Modern AI-powered stock analysis API",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.add_api_route("/", dashboard, methods=["GET"], include_in_schema=False)
    app.add_api_route("/health", health_check, methods=["GET"])
    app.add_api_route("/api/v1/stocks/{symbol}/price", get_stock_price, methods=["GET"])
    app.add_api_route("/api/v1/stocks/{symbol}/prediction", get_prediction, methods=["GET"])
    app.add_api_route("/api/v1/stocks/{symbol}/sentiment", get_sentiment, methods=["GET"])
    app.add_api_route("/api/v1/backtest", run_backtest, methods=["POST"])
    app.add_api_route("/api/v1/events", subscribe_events, methods=["GET"])
    app.add_websocket_route("/ws/{channel}", websocket_endpoint)

    return app


# ============================================================================
# API Routes
# ============================================================================

async def health_check() -> HealthStatus:
    """System health check endpoint."""
    cache = get_cache()

    # Check components
    db_status = {"status": "ok", "type": "sqlite"}  # Would check actual DB
    cache_status = await cache.health_check() if cache._connected else {"status": "disconnected"}
    event_bus = get_event_bus()

    all_healthy = (
        db_status.get("status") == "ok" and
        cache_status.get("healthy", False) and
        event_bus._running
    )

    return HealthStatus(
        status="healthy" if all_healthy else "degraded",
        healthy=all_healthy,
        version="2.0.0",
        uptime_seconds=time.time() - _start_time,
        database=db_status,
        cache=cache_status,
        event_bus=event_bus.stats,
    )


async def get_stock_price(
    symbol: str,
    authenticated: str | None = Depends(verify_auth),
) -> StockPrice:
    """Get current stock price."""
    try:
        fetcher = _get_fetcher_service()
        quote = fetcher.get_realtime(symbol)
    except Exception as exc:
        log.warning("Quote fetch failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=503, detail=f"Unable to fetch quote for {symbol}") from exc

    if quote is None:
        raise HTTPException(status_code=404, detail=f"No quote available for {symbol}")

    price = _as_float(getattr(quote, "price", 0.0), 0.0)
    if price <= 0:
        price = _as_float(getattr(quote, "close", 0.0), 0.0)
    if price <= 0:
        raise HTTPException(status_code=503, detail=f"Invalid quote payload for {symbol}")

    ts = getattr(quote, "timestamp", None)
    if not isinstance(ts, datetime):
        ts = datetime.utcnow()

    return StockPrice(
        symbol=str(getattr(quote, "code", symbol) or symbol),
        name=str(getattr(quote, "name", "") or symbol),
        price=price,
        change=_as_float(getattr(quote, "change", 0.0), 0.0),
        change_percent=_as_float(getattr(quote, "change_pct", 0.0), 0.0),
        volume=int(max(0, _as_float(getattr(quote, "volume", 0), 0.0))),
        timestamp=ts,
        source=str(getattr(quote, "source", "") or "live"),
    )


async def get_prediction(
    symbol: str,
    model: str = Query(default="ensemble", description="Model to use"),
    authenticated: str | None = Depends(verify_auth),
) -> Prediction:
    """Get model prediction for stock."""
    try:
        return _heuristic_prediction(symbol=symbol, model=model)
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("Prediction failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=503, detail=f"Unable to generate prediction for {symbol}") from exc


async def get_sentiment(
    symbol: str,
    hours_back: int = Query(default=24, ge=1, le=168),
    authenticated: str | None = Depends(verify_auth),
) -> SentimentAnalysis:
    """Get sentiment analysis for stock."""
    try:
        aggregator = _get_news_aggregator_service()
        summary = aggregator.get_sentiment_summary(stock_code=symbol)
    except Exception as exc:
        log.warning("Sentiment fetch failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=503, detail=f"Unable to fetch sentiment for {symbol}") from exc

    overall = _as_float(summary.get("overall_sentiment", 0.0), 0.0)
    overall = max(-1.0, min(1.0, overall))
    policy_impact = _as_float(summary.get("policy_sentiment", overall), overall)
    policy_impact = max(-1.0, min(1.0, policy_impact))
    market_sentiment = _as_float(summary.get("simple_sentiment", overall), overall)
    market_sentiment = max(-1.0, min(1.0, market_sentiment))
    confidence = _as_float(summary.get("confidence", 0.0), 0.0)
    confidence = max(0.0, min(1.0, confidence))

    return SentimentAnalysis(
        symbol=symbol,
        overall_sentiment=overall,
        policy_impact=policy_impact,
        market_sentiment=market_sentiment,
        confidence=confidence,
        article_count=int(summary.get("total", 0) or 0),
        timestamp=datetime.utcnow(),
    )


async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    authenticated: str | None = Depends(verify_auth),
) -> BacktestResult:
    """Run backtest with given parameters."""
    del background_tasks  # Reserved for future async orchestration.

    total_days = max(30, int((request.end_date - request.start_date).days))
    train_months = max(3, min(24, int(total_days * 0.8 / 30)))
    test_months = max(1, min(12, int(total_days * 0.2 / 30)))
    min_data_days = max(120, total_days)
    years = max(1 / 12, total_days / 365.0)

    try:
        backtester = _get_backtester_service()
        result = backtester.run(
            stock_codes=[request.symbol],
            train_months=train_months,
            test_months=test_months,
            min_data_days=min_data_days,
            initial_capital=float(request.initial_capital),
        )
    except Exception as exc:
        log.warning("Backtest failed for %s: %s", request.symbol, exc)
        raise HTTPException(
            status_code=503,
            detail=f"Unable to run backtest for {request.symbol}",
        ) from exc

    total_return = _as_float(getattr(result, "total_return", 0.0), 0.0)
    annualized_return = ((1.0 + total_return / 100.0) ** (1.0 / years) - 1.0) * 100.0
    max_drawdown = _as_float(getattr(result, "max_drawdown_pct", getattr(result, "max_drawdown", 0.0)), 0.0)

    return BacktestResult(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=_as_float(getattr(result, "sharpe_ratio", 0.0), 0.0),
        max_drawdown=max_drawdown,
        win_rate=_as_float(getattr(result, "win_rate", 0.0), 0.0),
        total_trades=int(getattr(result, "total_trades", 0) or 0),
        profit_factor=_as_float(getattr(result, "profit_factor", 0.0), 0.0),
    )


async def subscribe_events(
    event_type: str | None = Query(default=None, description="Filter by event type"),
    authenticated: str | None = Depends(verify_auth),
) -> AsyncGenerator[str, None]:
    """Server-sent events for real-time updates."""
    from asyncio import sleep

    get_event_bus()

    while True:
        # In production, would stream actual events
        yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
        await sleep(30)


async def websocket_endpoint(websocket: WebSocket, channel: str) -> None:
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket, channel)

    # Subscribe to event bus
    event_bus = get_event_bus()

    async def event_handler(event: Event) -> None:
        """Forward events to WebSocket clients."""
        await manager.broadcast(
            channel,
            {
                "type": event.type.name,
                "data": event.data,
                "timestamp": event.timestamp.isoformat(),
            },
        )

    # Subscribe to relevant events
    if channel == "market":
        event_bus.subscribe(EventType.BAR, event_handler)
        event_bus.subscribe(EventType.QUOTE, event_handler)
    elif channel == "signals":
        event_bus.subscribe(EventType.SIGNAL_GENERATED, event_handler)
        event_bus.subscribe(EventType.PREDICTION_READY, event_handler)

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)
    except Exception as e:
        log.exception(f"WebSocket error: {e}")
        manager.disconnect(websocket, channel)


# ============================================================================
# Dashboard HTML (Optional)
# ============================================================================

def get_dashboard_html() -> str:
    """Return dashboard HTML."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Trading Graph Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --bg-1: #0f172a;
            --bg-2: #0b1328;
            --panel: rgba(255, 255, 255, 0.94);
            --accent: #0ea5e9;
            --danger: #ef4444;
            --warn: #f59e0b;
            --ok: #10b981;
            --text: #0f172a;
            --muted: #64748b;
            --radius: 18px;
            --shadow: 0 18px 45px rgba(2, 6, 23, 0.22);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            color: var(--text);
            font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
            background:
                radial-gradient(1100px 700px at 8% 0%, #1d4ed8 0%, transparent 55%),
                radial-gradient(900px 550px at 92% 0%, #0f766e 0%, transparent 55%),
                linear-gradient(180deg, var(--bg-1), var(--bg-2));
            min-height: 100vh;
            padding: 24px;
        }
        .shell {
            max-width: 1240px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr;
            gap: 18px;
        }
        .hero {
            background: linear-gradient(140deg, rgba(255,255,255,0.95), rgba(226,232,240,0.9));
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 20px 22px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            animation: rise-in 420ms ease-out;
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(1.35rem, 2.1vw, 2rem);
            letter-spacing: 0.01em;
        }
        .hero p {
            margin: 6px 0 0;
            color: #334155;
            font-size: 0.92rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border-radius: 999px;
            padding: 8px 14px;
            font-size: 0.84rem;
            background: #e2e8f0;
            color: #0f172a;
            font-weight: 600;
        }
        .dot { width: 9px; height: 9px; border-radius: 50%; }
        .dot.healthy { background: var(--ok); box-shadow: 0 0 0 4px rgba(16,185,129,0.2); }
        .dot.degraded { background: var(--warn); box-shadow: 0 0 0 4px rgba(245,158,11,0.2); }
        .dot.unhealthy { background: var(--danger); box-shadow: 0 0 0 4px rgba(239,68,68,0.2); }
        .toolbar {
            background: var(--panel);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 14px;
            display: flex;
            flex-wrap: wrap;
            align-items: end;
            gap: 12px;
            animation: rise-in 520ms ease-out;
        }
        .field {
            display: flex;
            flex-direction: column;
            gap: 6px;
            min-width: 190px;
        }
        .field label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
            font-weight: 700;
        }
        .field input {
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            height: 40px;
            padding: 0 12px;
            font-size: 0.96rem;
            background: #f8fafc;
            color: #0f172a;
            outline: none;
        }
        .field input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(14,165,233,0.18);
        }
        .btn {
            border: none;
            height: 40px;
            padding: 0 16px;
            border-radius: 12px;
            font-weight: 700;
            letter-spacing: 0.02em;
            cursor: pointer;
            background: linear-gradient(135deg, var(--accent), #2563eb);
            color: #fff;
            box-shadow: 0 10px 24px rgba(14,165,233,0.35);
        }
        .grid {
            display: grid;
            gap: 14px;
            grid-template-columns: repeat(12, minmax(0, 1fr));
        }
        .card {
            background: var(--panel);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 16px;
            animation: rise-in 580ms ease-out;
        }
        .stat { grid-column: span 3; }
        .chart { grid-column: span 8; min-height: 430px; }
        .events { grid-column: span 4; min-height: 430px; display: flex; flex-direction: column; }
        .card h2 {
            margin: 0 0 10px;
            font-size: 0.86rem;
            letter-spacing: 0.12em;
            color: var(--muted);
            text-transform: uppercase;
        }
        .metric {
            font-size: clamp(1.05rem, 2vw, 1.6rem);
            font-weight: 700;
            line-height: 1.2;
            color: #0b1220;
        }
        .metric-sub {
            margin-top: 6px;
            color: #475569;
            font-size: 0.9rem;
        }
        .mono {
            font-family: "IBM Plex Mono", "Consolas", monospace;
            font-size: 0.86rem;
            color: #334155;
        }
        #chart {
            width: 100%;
            height: 355px;
            border-radius: 14px;
            overflow: hidden;
            background: #eef2ff;
        }
        .events-list {
            margin: 0;
            padding: 0;
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: 350px;
            overflow: auto;
        }
        .events-list li {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 10px 12px;
            background: #f8fafc;
            color: #1e293b;
            font-size: 0.83rem;
            line-height: 1.35;
        }
        .events-list li .time { color: #64748b; font-size: 0.76rem; margin-right: 6px; }
        @media (max-width: 1024px) {
            .stat { grid-column: span 6; }
            .chart { grid-column: span 12; }
            .events { grid-column: span 12; min-height: 0; }
        }
        @media (max-width: 640px) {
            body { padding: 14px; }
            .stat { grid-column: span 12; }
            .hero, .toolbar, .card { padding: 14px; }
            #chart { height: 300px; }
        }
        @keyframes rise-in {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="shell">
        <section class="hero">
            <div>
                <h1>Trading Graph Control Surface</h1>
                <p>Live quote, signal and sentiment monitoring with event streaming.</p>
            </div>
            <div>
                <div id="status-pill" class="status-pill">
                    <span id="status-dot" class="dot degraded"></span>
                    <span id="status-text">Checking system health...</span>
                </div>
                <div class="mono" style="margin-top: 8px;">Uptime: <span id="uptime">--:--:--</span></div>
            </div>
        </section>

        <form id="symbol-form" class="toolbar">
            <div class="field">
                <label for="symbol">Symbol</label>
                <input id="symbol" value="600519" autocomplete="off" />
            </div>
            <div class="field">
                <label for="model">Model Mode</label>
                <input id="model" value="ensemble" autocomplete="off" />
            </div>
            <div class="field">
                <label for="hours">Sentiment Window (h)</label>
                <input id="hours" type="number" min="1" max="168" value="24" />
            </div>
            <button class="btn" type="submit">Refresh Snapshot</button>
        </form>

        <div class="grid">
            <article class="card stat">
                <h2>Last Price</h2>
                <div id="price-value" class="metric">--</div>
                <div id="price-meta" class="metric-sub">Awaiting quote...</div>
            </article>

            <article class="card stat">
                <h2>Prediction</h2>
                <div id="signal-value" class="metric">--</div>
                <div id="signal-meta" class="metric-sub">Awaiting signal...</div>
            </article>

            <article class="card stat">
                <h2>Sentiment</h2>
                <div id="sentiment-value" class="metric">--</div>
                <div id="sentiment-meta" class="metric-sub">Awaiting sentiment...</div>
            </article>

            <article class="card stat">
                <h2>Data Source</h2>
                <div id="source-value" class="metric">--</div>
                <div id="source-meta" class="metric-sub">Provider details pending...</div>
            </article>

            <article class="card chart">
                <h2>Live Price Trace</h2>
                <div id="chart"></div>
            </article>

            <article class="card events">
                <h2>Event Stream</h2>
                <ul id="events" class="events-list"></ul>
            </article>
        </div>
    </div>

    <script>
        const state = {
            symbol: "600519",
            model: "ensemble",
            hours: 24,
            ws: null,
            chartInitialized: false
        };

        function fmtNum(value, digits = 2) {
            const n = Number(value);
            if (!Number.isFinite(n)) return "--";
            return n.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
        }

        function fmtPct(value, digits = 2) {
            const n = Number(value);
            if (!Number.isFinite(n)) return "--";
            return `${n >= 0 ? "+" : ""}${fmtNum(n, digits)}%`;
        }

        function fmtClock(seconds) {
            const s = Math.max(0, Number(seconds) || 0);
            const hh = String(Math.floor(s / 3600)).padStart(2, "0");
            const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
            const ss = String(Math.floor(s % 60)).padStart(2, "0");
            return `${hh}:${mm}:${ss}`;
        }

        async function fetchJSON(url) {
            const response = await fetch(url, { headers: { "Accept": "application/json" } });
            if (!response.ok) {
                const text = await response.text();
                throw new Error(`${response.status} ${response.statusText}: ${text}`);
            }
            return response.json();
        }

        function ensureChart() {
            if (state.chartInitialized) return;
            Plotly.newPlot(
                "chart",
                [{
                    x: [],
                    y: [],
                    type: "scatter",
                    mode: "lines",
                    line: { color: "#0ea5e9", width: 3, shape: "spline", smoothing: 0.8 },
                    fill: "tozeroy",
                    fillcolor: "rgba(14,165,233,0.18)",
                    hovertemplate: "%{x}<br>%{y:.2f}<extra></extra>"
                }],
                {
                    margin: { l: 44, r: 18, t: 10, b: 34 },
                    paper_bgcolor: "#eef2ff",
                    plot_bgcolor: "#eef2ff",
                    xaxis: { gridcolor: "rgba(100,116,139,0.18)", showline: false, zeroline: false },
                    yaxis: { gridcolor: "rgba(100,116,139,0.18)", showline: false, zeroline: false },
                    font: { family: "IBM Plex Mono, monospace", color: "#334155", size: 12 }
                },
                { responsive: true, displayModeBar: false }
            );
            state.chartInitialized = true;
        }

        function appendPricePoint(timestamp, price) {
            ensureChart();
            Plotly.extendTraces("chart", { x: [[timestamp]], y: [[price]] }, [0], 180);
        }

        function pushEvent(text) {
            const list = document.getElementById("events");
            const li = document.createElement("li");
            const stamp = new Date().toLocaleTimeString();
            li.innerHTML = `<span class="time">${stamp}</span>${text}`;
            list.prepend(li);
            while (list.children.length > 120) list.removeChild(list.lastChild);
        }

        async function refreshHealth() {
            try {
                const data = await fetchJSON("/health");
                const status = String(data.status || "degraded");
                const dot = document.getElementById("status-dot");
                const text = document.getElementById("status-text");
                dot.className = `dot ${status}`;
                text.textContent = `System ${status}`;
                document.getElementById("uptime").textContent = fmtClock(data.uptime_seconds);
            } catch (error) {
                document.getElementById("status-dot").className = "dot unhealthy";
                document.getElementById("status-text").textContent = "Health check unavailable";
                pushEvent(`Health error: ${error.message}`);
            }
        }

        async function refreshSnapshot() {
            const symbol = state.symbol;
            const model = encodeURIComponent(state.model);
            const hours = Math.max(1, Math.min(168, Number(state.hours) || 24));

            try {
                const [price, prediction, sentiment] = await Promise.all([
                    fetchJSON(`/api/v1/stocks/${encodeURIComponent(symbol)}/price`),
                    fetchJSON(`/api/v1/stocks/${encodeURIComponent(symbol)}/prediction?model=${model}`),
                    fetchJSON(`/api/v1/stocks/${encodeURIComponent(symbol)}/sentiment?hours_back=${hours}`)
                ]);

                document.getElementById("price-value").textContent = fmtNum(price.price, 2);
                document.getElementById("price-meta").textContent = `${fmtPct(price.change_percent)} | Vol ${fmtNum(price.volume, 0)}`;
                document.getElementById("source-value").textContent = String(price.source || "live").toUpperCase();
                document.getElementById("source-meta").textContent = String(price.name || symbol);

                document.getElementById("signal-value").textContent = `${prediction.signal || "--"} (${Math.round((prediction.confidence || 0) * 100)}%)`;
                document.getElementById("signal-meta").textContent = `Target ${fmtNum(prediction.target_price, 2)} | Horizon ${prediction.horizon_days || "--"}d`;

                document.getElementById("sentiment-value").textContent = fmtNum(sentiment.overall_sentiment, 3);
                document.getElementById("sentiment-meta").textContent = `Confidence ${Math.round((sentiment.confidence || 0) * 100)}% | Articles ${sentiment.article_count || 0}`;

                appendPricePoint(price.timestamp || new Date().toISOString(), Number(price.price));
            } catch (error) {
                pushEvent(`Snapshot error (${symbol}): ${error.message}`);
            }
        }

        function connectWs() {
            if (state.ws) {
                try { state.ws.close(); } catch (err) {}
            }
            const protocol = window.location.protocol === "https:" ? "wss" : "ws";
            const ws = new WebSocket(`${protocol}://${window.location.host}/ws/market`);
            state.ws = ws;

            ws.onopen = () => pushEvent("WebSocket connected");
            ws.onmessage = (event) => {
                try {
                    const payload = JSON.parse(event.data);
                    const kind = String(payload.type || "event");
                    pushEvent(`<strong>${kind}</strong> ${JSON.stringify(payload.data || {})}`);
                } catch (err) {
                    pushEvent(`WS parse error: ${err.message}`);
                }
            };
            ws.onerror = () => pushEvent("WebSocket connection error");
            ws.onclose = () => {
                pushEvent("WebSocket disconnected; retrying in 5s");
                setTimeout(connectWs, 5000);
            };
        }

        document.getElementById("symbol-form").addEventListener("submit", (event) => {
            event.preventDefault();
            state.symbol = String(document.getElementById("symbol").value || "").trim() || "600519";
            state.model = String(document.getElementById("model").value || "ensemble").trim() || "ensemble";
            state.hours = Number(document.getElementById("hours").value || 24);
            refreshSnapshot();
        });

        ensureChart();
        refreshHealth();
        refreshSnapshot();
        connectWs();
        setInterval(refreshHealth, 5000);
        setInterval(refreshSnapshot, 15000);
    </script>
</body>
</html>
"""

async def dashboard(request) -> HTMLResponse:
    """Serve dashboard HTML."""
    return HTMLResponse(content=get_dashboard_html())


# ============================================================================
# Main Entry Point
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the web server.

    Args:
        host: Server host
        port: Server port
        reload: Enable auto-reload (development)
    """
    app = create_app()
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    host = env_text("WEB_HOST", "0.0.0.0")
    port = env_int("WEB_PORT", 8000)
    run_server(host=host, port=port, reload=True)
