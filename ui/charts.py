# ui/charts.py
from typing import List, Dict, Optional
import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

from utils.logger import get_logger

log = get_logger(__name__)

# Try to import plotting libraries
try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    pg = None
    log.warning("pyqtgraph not available - using fallback chart")


# =============================================================================
# CANDLESTICK ITEM (module-scope so it can be referenced by name)
# =============================================================================

if HAS_PYQTGRAPH:
    class CandlestickItem(pg.GraphicsObject):
        """
        Custom candlestick chart item for pyqtgraph.
        
        Data format: list of (x, open, close, low, high) tuples.
        """

        def __init__(self):
            super().__init__()
            self.data = []
            self._picture = None

        def setData(self, data):
            """Set candlestick data. Each item: (x, open, close, low, high)"""
            self.data = data or []
            self._picture = None
            self.prepareGeometryChange()
            self.update()

        def _generate_picture(self):
            """Generate QPicture for all candles."""
            pic = pg.QtGui.QPicture()
            p = pg.QtGui.QPainter(pic)
            w = 0.6

            for item in self.data:
                if len(item) < 5:
                    continue
                t, o, c, low, high = item

                if any(v is None for v in (o, c, low, high)):
                    continue

                o = float(o)
                c = float(c)
                low = float(low)
                high = float(high)

                up = (c >= o)
                color = (
                    pg.mkColor("#3fb950") if up
                    else pg.mkColor("#f85149")
                )

                p.setPen(pg.mkPen(color=color, width=1))
                p.setBrush(pg.mkBrush(color=color))

                # Wick
                p.drawLine(
                    pg.QtCore.QPointF(t, low),
                    pg.QtCore.QPointF(t, high)
                )

                # Body
                top = max(o, c)
                bot = min(o, c)
                body_height = max(1e-8, top - bot)
                rect = pg.QtCore.QRectF(
                    t - w / 2.0, bot, w, body_height
                )
                p.drawRect(rect)

            p.end()
            self._picture = pic

        def paint(self, p, *args):
            """Paint the candlestick item."""
            if self._picture is None:
                self._generate_picture()
            if self._picture is not None:
                p.drawPicture(0, 0, self._picture)

        def boundingRect(self):
            """Return bounding rectangle of all candle data."""
            if not self.data:
                return pg.QtCore.QRectF()

            try:
                xs = [d[0] for d in self.data if len(d) >= 5]
                lows = [float(d[3]) for d in self.data if len(d) >= 5]
                highs = [float(d[4]) for d in self.data if len(d) >= 5]

                if not xs or not lows or not highs:
                    return pg.QtCore.QRectF()

                return pg.QtCore.QRectF(
                    min(xs) - 1,
                    min(lows),
                    (max(xs) - min(xs)) + 2,
                    max(highs) - min(lows)
                )
            except (ValueError, TypeError):
                return pg.QtCore.QRectF()

else:
    # Stub when pyqtgraph not available
    CandlestickItem = None


# =============================================================================
# MAIN STOCK CHART
# =============================================================================

class StockChart(QWidget):
    """
    Interactive stock chart with AI prediction overlay.

    Features:
    - Historical price display (line or candlestick)
    - AI-predicted future prices
    - Trading levels (stop loss, targets)
    - Real-time updates
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data
        self._actual_prices: List[float] = []
        self._predicted_prices: List[float] = []
        self._levels: Dict[str, float] = {}

        # Plot references (set in _setup_pyqtgraph)
        self.plot_widget = None
        self.actual_line = None
        self.predicted_line = None
        self.candles = None
        self.level_lines: Dict[str, object] = {}

        self._setup_ui()

    def _setup_ui(self):
        """Setup chart UI with pyqtgraph or fallback."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_PYQTGRAPH:
            self._setup_pyqtgraph()
        else:
            self._setup_fallback()

    def _setup_pyqtgraph(self):
        """Setup pyqtgraph chart — single initialization, no duplicates."""
        pg.setConfigOptions(
            antialias=True,
            background='#0d1117',
            foreground='#c9d1d9'
        )

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Price', units='¥')
        self.plot_widget.setLabel('bottom', 'Time', units='bars')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setBackground('#0d1117')

        # Actual price line
        self.actual_line = self.plot_widget.plot(
            pen=pg.mkPen(color='#58a6ff', width=2),
            name='Actual'
        )

        # Predicted price line (dashed)
        self.predicted_line = self.plot_widget.plot(
            pen=pg.mkPen(
                color='#3fb950', width=2,
                style=Qt.PenStyle.DashLine
            ),
            name='Predicted'
        )

        # Level lines dict
        self.level_lines = {}

        # Legend
        self.plot_widget.addLegend()

        # FIX: Create CandlestickItem ONCE at module-scope class
        self.candles = CandlestickItem()
        self.plot_widget.addItem(self.candles)

        self.layout().addWidget(self.plot_widget)

    def _setup_fallback(self):
        """Setup fallback when pyqtgraph not available"""
        self.fallback_label = QLabel(
            "Chart requires pyqtgraph\n\n"
            "Install with: pip install pyqtgraph"
        )
        self.fallback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fallback_label.setStyleSheet("""
            QLabel {
                background: #0d1117;
                color: #8b949e;
                border: 1px solid #30363d;
                border-radius: 8px;
                font-size: 14px;
            }
        """)
        self.layout().addWidget(self.fallback_label)

    # =========================================================================
    # CANDLESTICK UPDATE
    # =========================================================================

    def update_candles(
        self,
        bars: List[dict],
        predicted_prices: List[float] = None,
        levels: Dict[str, float] = None,
    ):
        """
        Update chart with candlestick bar data.

        Args:
            bars: list of dicts with keys: open/high/low/close/timestamp
            predicted_prices: AI-predicted future prices
            levels: Trading levels dict
        """
        self._levels = levels or {}
        self._predicted_prices = (
            list(predicted_prices) if predicted_prices else []
        )

        if not HAS_PYQTGRAPH or self.candles is None:
            return

        if not bars:
            try:
                self.candles.setData([])
                if self.predicted_line:
                    self.predicted_line.clear()
                self._update_level_lines()
            except Exception:
                pass
            return

        try:
            # Convert to (x, open, close, low, high)
            ohlc = []
            closes = []
            for i, b in enumerate(bars[-180:]):
                try:
                    o = float(b.get("open", 0))
                    h = float(b.get("high", 0))
                    l_val = float(b.get("low", 0))
                    c = float(b.get("close", 0))
                    if o <= 0 or c <= 0 or h <= 0 or l_val <= 0:
                        continue
                    ohlc.append((i, o, c, l_val, h))
                    closes.append(c)
                except (ValueError, TypeError):
                    continue

            self.candles.setData(ohlc)

            # Predicted line anchored from last close
            if closes and self._predicted_prices and self.predicted_line:
                start_x = len(ohlc) - 1
                x_pred = np.arange(
                    start_x,
                    start_x + len(self._predicted_prices) + 1
                )
                y_pred = np.array(
                    [closes[-1]] + list(self._predicted_prices),
                    dtype=float
                )
                self.predicted_line.setData(x_pred, y_pred)
            elif self.predicted_line:
                self.predicted_line.clear()

            # Also store closes for level lines
            self._actual_prices = closes

            self._update_level_lines()

            if self.plot_widget:
                self.plot_widget.autoRange()

        except Exception as e:
            log.warning(f"Candle chart update failed: {e}")

    # =========================================================================
    # LINE CHART UPDATE
    # =========================================================================

    def update_data(
        self,
        actual_prices: List[float],
        predicted_prices: List[float] = None,
        levels: Dict[str, float] = None
    ):
        """
        Update chart with line data.

        Args:
            actual_prices: Historical prices
            predicted_prices: AI-predicted future prices
            levels: Trading levels (stop_loss, target_1, etc.)
        """
        self._actual_prices = list(actual_prices) if actual_prices else []
        self._predicted_prices = (
            list(predicted_prices) if predicted_prices else []
        )
        self._levels = levels or {}

        if not HAS_PYQTGRAPH:
            return

        try:
            self._update_plot()
        except Exception as e:
            log.warning(f"Chart update failed: {e}")

    def _update_plot(self):
        """Update the plot with current data (safe for empty arrays)."""
        if self.actual_line is None:
            return

        if not self._actual_prices:
            self.actual_line.clear()
            if self.predicted_line:
                self.predicted_line.clear()
            self._update_level_lines()
            return

        x_actual = np.arange(len(self._actual_prices))
        y_actual = np.array(self._actual_prices, dtype=float)
        self.actual_line.setData(x_actual, y_actual)

        if (
            self._predicted_prices
            and len(self._actual_prices) >= 1
            and self.predicted_line
        ):
            start_x = len(self._actual_prices) - 1
            x_pred = np.arange(
                start_x,
                start_x + len(self._predicted_prices) + 1
            )
            y_pred = np.array(
                [self._actual_prices[-1]] + list(self._predicted_prices),
                dtype=float
            )
            self.predicted_line.setData(x_pred, y_pred)
        elif self.predicted_line:
            self.predicted_line.clear()

        self._update_level_lines()

        if self.plot_widget:
            self.plot_widget.autoRange()

    # =========================================================================
    # TRADING LEVEL LINES
    # =========================================================================

    def _update_level_lines(self):
        """Update horizontal lines for trading levels."""
        if not HAS_PYQTGRAPH or self.plot_widget is None:
            return

        # Remove old lines
        for line in self.level_lines.values():
            try:
                self.plot_widget.removeItem(line)
            except Exception:
                pass
        self.level_lines.clear()

        if not self._levels or not self._actual_prices:
            return

        level_colors = {
            'stop_loss': '#f85149',
            'target_1': '#3fb950',
            'target_2': '#2ea043',
            'target_3': '#238636',
            'entry': '#58a6ff',
        }

        for name, price in self._levels.items():
            if price and price > 0:
                color = level_colors.get(name, '#888')
                try:
                    line = pg.InfiniteLine(
                        pos=price,
                        angle=0,
                        pen=pg.mkPen(
                            color=color, width=1,
                            style=Qt.PenStyle.DotLine
                        ),
                        label=f'{name}: ¥{price:.2f}',
                        labelOpts={
                            'color': color,
                            'position': 0.95
                        }
                    )
                    self.plot_widget.addItem(line)
                    self.level_lines[name] = line
                except Exception as e:
                    log.debug(f"Failed to add level line {name}: {e}")

    # =========================================================================
    # CLEAR / TITLE
    # =========================================================================

    def clear(self):
        """Clear all data from chart"""
        self._actual_prices = []
        self._predicted_prices = []
        self._levels = {}

        if not HAS_PYQTGRAPH:
            return

        try:
            if self.actual_line:
                self.actual_line.clear()
            if self.predicted_line:
                self.predicted_line.clear()
            if self.candles:
                self.candles.setData([])

            for line in self.level_lines.values():
                try:
                    self.plot_widget.removeItem(line)
                except Exception:
                    pass
            self.level_lines.clear()
        except Exception as e:
            log.debug(f"Chart clear failed: {e}")

    def set_title(self, title: str):
        """Set chart title"""
        if HAS_PYQTGRAPH and self.plot_widget:
            try:
                self.plot_widget.setTitle(
                    title, color='#c9d1d9', size='12pt'
                )
            except Exception:
                pass


# =============================================================================
# MINI CHART (for watchlist)
# =============================================================================

class MiniChart(QWidget):
    """
    Compact mini chart for watchlist items.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 40)
        self._prices: List[float] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_PYQTGRAPH:
            self.plot = pg.PlotWidget()
            self.plot.setBackground('#0d1117')
            self.plot.hideAxis('left')
            self.plot.hideAxis('bottom')
            self.plot.setMouseEnabled(False, False)

            self.line = self.plot.plot(
                pen=pg.mkPen(color='#58a6ff', width=1)
            )
            layout.addWidget(self.plot)
        else:
            self.plot = None
            self.line = None
            self.label = QLabel("--")
            self.label.setStyleSheet("color: #888;")
            layout.addWidget(self.label)

    def update_data(self, prices: List[float]):
        """Update mini chart"""
        self._prices = list(prices) if prices else []

        if not HAS_PYQTGRAPH or not self._prices or self.line is None:
            return

        try:
            x = np.arange(len(self._prices))
            y = np.array(self._prices, dtype=float)

            # Determine color based on trend
            if len(y) > 1:
                if y[-1] > y[0]:
                    color = '#3fb950'
                elif y[-1] < y[0]:
                    color = '#f85149'
                else:
                    color = '#888'
            else:
                color = '#888'

            self.line.setPen(pg.mkPen(color=color, width=1))
            self.line.setData(x, y)
            self.plot.autoRange()

        except Exception:
            pass