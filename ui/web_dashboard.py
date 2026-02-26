# ui/web_dashboard.py
"""
Enhanced Web Dashboard with Authentication

FIXES:
- Web-based UI for remote access
- Authentication and authorization
- Real-time updates via WebSocket
- Responsive design for mobile
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from config.settings import CONFIG
from utils.logger import get_logger
from utils.security_hardening import (
    get_auth_manager,
    get_rate_limiter,
    get_audit_logger,
    AuthenticationManager,
    AccessLevel,
    AuditEventType,
)

log = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Trading Graph Dashboard",
    description="Real-time trading analysis dashboard",
    version="2.0.1",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)
auth_manager = get_auth_manager()
rate_limiter = get_rate_limiter()
audit_logger = get_audit_logger()


# ============================================================================
# Authentication Dependencies
# ============================================================================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[str]:
    """Get current authenticated user from token."""
    if not credentials:
        return None
    
    token = credentials.credentials
    payload = auth_manager.verify_token(token)
    
    if not payload:
        return None
    
    return payload.get("user_id")


async def require_authenticated_user(
    user_id: Optional[str] = Depends(get_current_user),
) -> str:
    """Require authenticated user."""
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return user_id


# ============================================================================
# API Routes
# ============================================================================

@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "name": "Trading Graph Dashboard",
        "version": "2.0.1",
        "status": "running",
    }


@app.get("/api/v1/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.1",
    }


@app.post("/api/v1/auth/login")
async def login(
    username: str,
    password: str,
) -> dict[str, Any]:
    """
    Login endpoint.
    
    FIX: Web authentication
    """
    # In production, validate against database
    # This is a simplified example
    
    # For demo: accept any non-empty credentials
    if username and password:
        # Generate token
        token = auth_manager.generate_token(
            user_id=username,
            access_level=AccessLevel.AUTHENTICATED,
        )
        
        # Audit log
        audit_logger.log(
            event_type=AuditEventType.LOGIN,
            user_id=username,
            resource="auth",
            action="login",
            status="success",
        )
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": 3600 * 24,  # 24 hours
        }
    
    # Audit failed login
    audit_logger.log(
        event_type=AuditEventType.AUTH_FAILURE,
        user_id=username,
        resource="auth",
        action="login",
        status="failure",
    )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
    )


@app.get("/api/v1/dashboard")
async def get_dashboard(
    user_id: str = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """
    Get dashboard data.
    
    FIX: Authenticated dashboard access
    """
    # Rate limiting
    allowed, reason = rate_limiter.is_allowed(user_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {reason}",
        )
    
    # Get dashboard data
    return {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "market_status": "open",
        "positions": [],
        "signals": [],
        "portfolio_value": 0.0,
    }


@app.get("/api/v1/stocks/{symbol}")
async def get_stock_data(
    symbol: str,
    user_id: str = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Get stock data."""
    # Rate limiting
    allowed, reason = rate_limiter.is_allowed(user_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {reason}",
        )
    
    # In production, fetch real data
    return {
        "symbol": symbol,
        "price": 0.0,
        "change": 0.0,
        "volume": 0,
    }


# ============================================================================
# WebSocket for Real-time Updates
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}
    
    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
    ) -> None:
        """Accept WebSocket connection."""
        await websocket.accept()
        if user_id not in self._connections:
            self._connections[user_id] = []
        self._connections[user_id].append(websocket)
        log.info(f"WebSocket connected for user: {user_id}")
    
    def disconnect(
        self,
        websocket: WebSocket,
        user_id: str,
    ) -> None:
        """Remove WebSocket connection."""
        if user_id in self._connections:
            self._connections[user_id].remove(websocket)
            if not self._connections[user_id]:
                del self._connections[user_id]
        log.info(f"WebSocket disconnected for user: {user_id}")
    
    async def broadcast_to_user(
        self,
        user_id: str,
        message: dict[str, Any],
    ) -> None:
        """Broadcast message to user's connections."""
        if user_id in self._connections:
            for connection in self._connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    log.error(f"WebSocket send error: {e}")


manager = ConnectionManager()


@app.websocket("/ws/updates")
async def websocket_updates(
    websocket: WebSocket,
    token: Optional[str] = None,
) -> None:
    """
    WebSocket endpoint for real-time updates.
    
    FIX: Real-time data streaming
    """
    # Authenticate
    if not token:
        await websocket.close(code=4001, reason="Token required")
        return
    
    payload = auth_manager.verify_token(token)
    if not payload:
        await websocket.close(code=4002, reason="Invalid token")
        return
    
    user_id = payload.get("user_id")
    
    # Connect
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            try:
                # Receive messages from client
                data = await websocket.receive_json()
                
                # Handle client messages
                action = data.get("action")
                if action == "subscribe":
                    symbols = data.get("symbols", [])
                    log.info(f"User {user_id} subscribed to: {symbols}")
                elif action == "ping":
                    await websocket.send_json({"type": "pong"})
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                continue
    
    finally:
        manager.disconnect(websocket, user_id)


# ============================================================================
# HTML Dashboard (Simple)
# ============================================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_html() -> str:
    """Serve dashboard HTML."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Graph Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid #333;
        }
        .login-form {
            max-width: 400px;
            margin: 100px auto;
            padding: 30px;
            background: #16213e;
            border-radius: 8px;
        }
        .login-form input {
            width: 100%;
            padding: 12px;
            margin: 10px 0;
            border: 1px solid #333;
            border-radius: 4px;
            background: #0f3460;
            color: #eee;
        }
        .login-form button {
            width: 100%;
            padding: 12px;
            background: #e94560;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .login-form button:hover { background: #ff6b6b; }
        .dashboard { display: none; }
        .dashboard.active { display: block; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
        }
        .card h3 { color: #e94560; margin-bottom: 15px; }
        .status { color: #4ade80; }
        .error { color: #f87171; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Trading Graph Dashboard</h1>
            <div id="user-info"></div>
        </div>
        
        <!-- Login Form -->
        <div id="login-section" class="login-form">
            <h2 style="text-align: center; margin-bottom: 20px;">Login</h2>
            <input type="text" id="username" placeholder="Username" />
            <input type="password" id="password" placeholder="Password" />
            <button onclick="login()">Login</button>
            <p id="login-error" class="error" style="display: none; margin-top: 10px;"></p>
        </div>
        
        <!-- Dashboard -->
        <div id="dashboard-section" class="dashboard">
            <div class="grid">
                <div class="card">
                    <h3>System Status</h3>
                    <p class="status">● Online</p>
                    <p style="margin-top: 10px; color: #888;">Version: 2.0.1</p>
                </div>
                <div class="card">
                    <h3>Market Status</h3>
                    <p id="market-status">Loading...</p>
                </div>
                <div class="card">
                    <h3>Portfolio</h3>
                    <p id="portfolio-value">¥0.00</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let authToken = null;
        
        async function login() {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            try {
                const response = await fetch('/api/v1/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`
                });
                
                if (response.ok) {
                    const data = await response.json();
                    authToken = data.access_token;
                    document.getElementById('login-section').style.display = 'none';
                    document.getElementById('dashboard-section').classList.add('active');
                    document.getElementById('user-info').textContent = `User: ${username}`;
                    loadDashboard();
                } else {
                    const error = await response.json();
                    document.getElementById('login-error').textContent = error.detail;
                    document.getElementById('login-error').style.display = 'block';
                }
            } catch (e) {
                document.getElementById('login-error').textContent = 'Connection failed';
                document.getElementById('login-error').style.display = 'block';
            }
        }
        
        async function loadDashboard() {
            try {
                const response = await fetch('/api/v1/dashboard', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                
                if (response.ok) {
                    const data = await response.json();
                    document.getElementById('market-status').textContent = data.market_status;
                    document.getElementById('portfolio-value').textContent = 
                        '¥' + data.portfolio_value.toFixed(2);
                }
            } catch (e) {
                console.error('Failed to load dashboard:', e);
            }
        }
    </script>
</body>
</html>
    """


# ============================================================================
# Server Startup
# ============================================================================

def start_dashboard(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Start web dashboard server."""
    log.info(f"Starting web dashboard at http://{host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    start_dashboard()
