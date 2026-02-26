# ADR 0002: PyQt6 for Desktop UI

## Status
Accepted

## Date
2024-01-15  
Last Updated: 2026-02-25

## Context

The trading system requires a desktop user interface for:
- Real-time charting and visualization
- Prediction display with uncertainty bands
- Order entry and management (analysis-only mode: view-only)
- Portfolio monitoring
- News and sentiment display
- System configuration

We need a UI framework that:
- Provides native desktop experience
- Supports real-time updates (3000ms interval)
- Has good charting capabilities for financial data
- Works on Windows, macOS, and Linux
- Has stable long-term support
- High-performance rendering for charts

## Decision

Use PyQt6 as the desktop UI framework with the following characteristics:

### Technology Stack

- **PyQt6**: Python bindings for Qt 6
- **pyqtgraph**: High-performance real-time charting using OpenGL
- **QSS styling**: Custom dark theme for trading interface
- **Model-View architecture**: Clean separation of data and presentation

### UI Architecture

```
┌─────────────────────────────────────────┐
│           Main Window (QMainWindow)     │
│  ┌───────────┬───────────┬───────────┐  │
│  │  Market   │   Chart   │   News &  │  │
│  │  Watch    │  Widget   │ Sentiment │  │
│  │  Widget   │           │   Panel   │  │
│  └───────────┴───────────┴───────────┘  │
│  ┌─────────────────────────────────────┐│
│  │        Portfolio & Positions        ││
│  └─────────────────────────────────────┘│
│  ┌─────────────────────────────────────┐│
│  │        Status Bar & Health          ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

### Component Breakdown

| Component | File | Purpose |
|-----------|------|---------|
| Main Window | `ui/app.py` | Application bootstrap, state management |
| Chart Widget | `ui/charts.py` | Candlestick charts with predictions |
| Market Watch | `ui/widgets.py` | Stock list, quotes, watchlist |
| News Panel | `ui/news_widget.py` | News feed, sentiment display |
| Monitoring | `ui/app_monitoring_ops.py` | Real-time monitoring, signal detection |

### Chart Features

- Candlestick charts with volume
- AI prediction overlay
- Uncertainty bands (confidence intervals)
- Technical indicators (SMA, EMA, Bollinger, VWAP)
- Multi-interval support (1m, 5m, 15m, 30m, 1h, 1d)
- Real-time updates via WebSocket

## Consequences

### Positive

- Native look and feel across platforms
- Excellent charting performance with pyqtgraph (OpenGL acceleration)
- Rich widget ecosystem
- Strong typing support (PyQt6 has good type stubs)
- Active community and documentation
- Mature framework with long-term support

### Negative

- Larger application bundle size (~50MB for Qt libraries)
- Steeper learning curve than web frameworks
- Less flexible for remote access scenarios
- Requires local installation

### Mitigation

- Provide web-based monitoring as future enhancement (FastAPI dashboard exists)
- Use Qt's theming system for consistent branding (dark theme by default)
- Implement proper threading to avoid UI blocking (worker threads for data fetching)
- Use lazy loading for heavy components

## Implementation

### Application Structure

```python
# ui/app.py
from PyQt6.QtWidgets import QApplication, QMainWindow
from ui.charts import ChartWidget
from ui.widgets import MarketWatchWidget
from ui.news_widget import NewsWidget

class TradingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.chart = ChartWidget()
        self.market_watch = MarketWatchWidget()
        self.news = NewsWidget()
        self._setup_ui()
    
    def _setup_ui(self):
        # Layout setup
        pass
```

### Real-Time Updates

```python
# Use worker threads for data fetching
from ui.realtime_worker import RealtimeWorker

worker = RealtimeWorker(symbol="600519", interval="1m")
worker.quote_received.connect(self.on_quote_update)
worker.start()
```

### Chart Updates

```python
# Efficient chart updates using pyqtgraph
import pyqtgraph as pg

class ChartWidget(pg.PlotWidget):
    def update_candlestick(self, bars):
        # Update existing plot item instead of recreating
        self.candlestick.setData(bars)
```

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| **Tkinter** | Built-in, simple | Limited widgets, poor charting | Rejected |
| **Kivy** | Cross-platform, modern | Smaller community, less mature | Rejected |
| **Web (Electron)** | Web technologies, remote access | Heavy resource usage, complex | Rejected |
| **Dear PyGui** | Fast, immediate mode | Less mature, smaller ecosystem | Rejected |

## References

- PyQt6 Documentation: https://www.riverbankcomputing.com/static/Docs/PyQt6/
- pyqtgraph: https://pyqtgraph.org/
- Qt for Python: https://doc.qt.io/qtforpython-6/
