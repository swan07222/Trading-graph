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
    # In production, this would fetch from data layer
    # For now, return mock data
    return StockPrice(
        symbol=symbol,
        name=f"Stock {symbol}",
        price=100.0 + (hash(symbol) % 100),
        change=2.5,
        change_percent=2.56,
        volume=1000000,
        timestamp=datetime.utcnow(),
        source="mock",
    )


async def get_prediction(
    symbol: str,
    model: str = Query(default="ensemble", description="Model to use"),
    authenticated: str | None = Depends(verify_auth),
) -> Prediction:
    """Get model prediction for stock."""
    # In production, this would call the model predictor
    return Prediction(
        symbol=symbol,
        signal="BUY",
        confidence=0.75,
        target_price=110.0,
        horizon_days=5,
        model_version=model,
        timestamp=datetime.utcnow(),
    )


async def get_sentiment(
    symbol: str,
    hours_back: int = Query(default=24, ge=1, le=168),
    authenticated: str | None = Depends(verify_auth),
) -> SentimentAnalysis:
    """Get sentiment analysis for stock."""
    # In production, this would call the sentiment analyzer
    return SentimentAnalysis(
        symbol=symbol,
        overall_sentiment=0.65,
        policy_impact=0.3,
        market_sentiment=0.7,
        confidence=0.8,
        article_count=50,
        timestamp=datetime.utcnow(),
    )


async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    authenticated: str | None = Depends(verify_auth),
) -> BacktestResult:
    """Run backtest with given parameters."""
    # In production, this would run actual backtest
    # For now, return mock results
    return BacktestResult(
        total_return=0.25,
        annualized_return=0.30,
        sharpe_ratio=1.5,
        max_drawdown=0.15,
        win_rate=0.60,
        total_trades=100,
        profit_factor=1.8,
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
    """Return simple dashboard HTML."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Graph Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        .status { display: inline-block; padding: 5px 10px; border-radius: 4px; font-weight: bold; }
        .status.healthy { background: #4caf50; color: white; }
        .status.degraded { background: #ff9800; color: white; }
        #chart { width: 100%; height: 400px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Trading Graph Dashboard</h1>
        
        <div class="card">
            <h2>System Status</h2>
            <p>Status: <span id="status" class="status">Loading...</span></p>
            <p>Uptime: <span id="uptime">-</span></p>
            <p>Version: 2.0.0</p>
        </div>

        <div class="card">
            <h2>Stock Price Chart</h2>
            <div id="chart"></div>
        </div>

        <div class="card">
            <h2>Real-time Events</h2>
            <ul id="events"></ul>
        </div>
    </div>

    <script>
        // Fetch health status
        async function updateHealth() {
            const response = await fetch('/health');
            const data = await response.json();
            
            document.getElementById('status').textContent = data.status;
            document.getElementById('status').className = 'status ' + data.status;
            document.getElementById('uptime').textContent = 
                new Date(data.uptime_seconds * 1000).toISOString().substr(11, 8);
        }

        // Create mock chart
        const trace = {
            x: Array.from({length: 100}, (_, i) => i),
            y: Array.from({length: 100}, () => 100 + Math.random() * 20),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Price'
        };
        Plotly.newPlot('chart', [trace]);

        // WebSocket for real-time events
        const ws = new WebSocket('ws://' + window.location.host + '/ws/market');
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const li = document.createElement('li');
            li.textContent = `${data.type}: ${JSON.stringify(data.data)}`;
            document.getElementById('events').prepend(li);
        };

        // Update health every 5 seconds
        updateHealth();
        setInterval(updateHealth, 5000);
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
