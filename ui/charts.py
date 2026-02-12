# ui/charts.py
from typing import List, Dict, Optional
import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

from utils.logger import get_logger

log = get_logger(__name__)

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    pg = None
    log.warning("pyqtgraph not available - using fallback chart")

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

                p.drawLine(
                    pg.QtCore.QPointF(t, low),
                    pg.QtCore.QPointF(t, high)
                )

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
    CandlestickItem = None

# MAIN STOCK CHART - FIXED VERSION

class StockChart(QWidget):
    """
    Interactive stock chart with THREE layers:

    Layer 1 (BOTTOM): Prediction line (dashed green) - AI forecast
    Layer 2 (MIDDLE): Price line (solid blue) - connects close prices  
    Layer 3 (TOP): Candlesticks (red/green) - OHLCV bars

    Plus: Trading level lines (stop loss, targets)
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._bars: List[dict] = []
        self._actual_prices: List[float] = []
        self._predicted_prices: List[float] = []
        self._levels: Dict[str, float] = {}

        self.plot_widget = None
        self.candles = None           # Layer 3: Candlesticks (top)
        self.actual_line = None       # Layer 2: Price line (middle)
        self.predicted_line = None    # Layer 1: Prediction (bottom)
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
        """Setup pyqtgraph chart with all three layers."""
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

        # === Layer 1 (BOTTOM): Prediction line - dashed green ===
        self.predicted_line = self.plot_widget.plot(
            pen=pg.mkPen(
                color='#3fb950',
                width=2,
                style=Qt.PenStyle.DashLine
            ),
            name='AI Prediction'
        )

        # === Layer 2 (MIDDLE): Price line - solid blue ===
        self.actual_line = self.plot_widget.plot(
            pen=pg.mkPen(color='#58a6ff', width=1.5),
            name='Price'
        )

        # === Layer 3 (TOP): Candlesticks ===
        self.candles = CandlestickItem()
        self.plot_widget.addItem(self.candles)

        self.level_lines = {}

        self.plot_widget.addLegend()

        self.layout().addWidget(self.plot_widget)

    def _setup_fallback(self):
        """Setup fallback when pyqtgraph not available."""
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
    # UNIFIED UPDATE METHOD - Draws all three layers together
    # =========================================================================

    def update_chart(
        self,
        bars: List[dict],
        predicted_prices: List[float] = None,
        levels: Dict[str, float] = None,
    ):
        """
        UNIFIED update method - draws all three layers together.

        This is the PRIMARY method that should be called for updates.
        Both update_candles() and update_data() now delegate to this.

        Args:
            bars: List of OHLCV dicts with keys: open, high, low, close
            predicted_prices: AI forecast prices (the "guessed graph")
            levels: Trading levels dict (stop_loss, target_1, etc.)
        """
        self._bars = list(bars) if bars else []
        self._predicted_prices = list(predicted_prices) if predicted_prices else []
        self._levels = levels or {}

        if not HAS_PYQTGRAPH:
            return

        if not self._bars:
            self._clear_all()
            return

        try:
            # Parse bar data - extract closes for line, OHLC for candles
            closes: List[float] = []
            ohlc: List[tuple] = []

            for i, b in enumerate(self._bars[-180:]):
                try:
                    o = float(b.get("open", 0) or 0)
                    h = float(b.get("high", 0) or 0)
                    l_val = float(b.get("low", 0) or 0)
                    c = float(b.get("close", 0) or 0)

                    if c <= 0:
                        continue

                    # Fix missing OHLC values
                    if o <= 0:
                        o = c
                    if h <= 0:
                        h = max(o, c)
                    if l_val <= 0:
                        l_val = min(o, c)

                    # Ensure high >= low
                    if h < l_val:
                        h, l_val = l_val, h

                    ohlc.append((i, o, c, l_val, h))
                    closes.append(c)
                except (ValueError, TypeError):
                    continue

            if not closes:
                self._clear_all()
                return

            # === Layer 3 (TOP): Candlesticks ===
            if self.candles is not None:
                self.candles.setData(ohlc)

            # === Layer 2 (MIDDLE): Price line connecting closes ===
            if self.actual_line is not None:
                x_actual = np.arange(len(closes))
                y_actual = np.array(closes, dtype=float)
                self.actual_line.setData(x_actual, y_actual)

            # === Layer 1 (BOTTOM): Prediction line (guessed graph) ===
            if self.predicted_line is not None:
                if closes and self._predicted_prices:
                    start_x = len(closes) - 1
                    x_pred = np.arange(
                        start_x,
                        start_x + len(self._predicted_prices) + 1
                    )
                    y_pred = np.array(
                        [closes[-1]] + list(self._predicted_prices),
                        dtype=float
                    )
                    self.predicted_line.setData(x_pred, y_pred)
                else:
                    self.predicted_line.clear()

            self._actual_prices = closes

            self._update_level_lines()

            # Auto-range to fit all data
            if self.plot_widget is not None:
                self.plot_widget.autoRange()

        except Exception as e:
            log.warning(f"Chart update failed: {e}")

    # =========================================================================
    # BACKWARD COMPATIBLE METHODS - Now delegate to update_chart()
    # =========================================================================

    def update_candles(
        self,
        bars: List[dict],
        predicted_prices: List[float] = None,
        levels: Dict[str, float] = None,
    ):
        """
        Update chart with candlestick bar data.

        BACKWARD COMPATIBLE: This now delegates to update_chart()
        so all three layers are drawn together.
        """
        self.update_chart(bars, predicted_prices, levels)

    def update_data(
        self,
        actual_prices: List[float],
        predicted_prices: List[float] = None,
        levels: Dict[str, float] = None
    ):
        """
        Update chart with line data.

        BACKWARD COMPATIBLE: Converts price list to bar format,
        then delegates to update_chart() so all three layers work.
        """
        bars = []
        for p in actual_prices:
            try:
                price = float(p)
                if price > 0:
                    bars.append({
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price
                    })
            except (ValueError, TypeError):
                continue

        self.update_chart(bars, predicted_prices, levels)

    # =========================================================================
    # =========================================================================

    def _clear_all(self):
        """Clear all chart elements."""
        try:
            if self.candles is not None:
                self.candles.setData([])
            if self.actual_line is not None:
                self.actual_line.clear()
            if self.predicted_line is not None:
                self.predicted_line.clear()
            self._update_level_lines()
        except Exception as e:
            log.debug(f"Chart clear failed: {e}")

    def _update_level_lines(self):
        """Update horizontal lines for trading levels."""
        if not HAS_PYQTGRAPH or self.plot_widget is None:
            return

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
                            color=color,
                            width=1,
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

    def clear(self):
        """Clear all data from chart."""
        self._bars = []
        self._actual_prices = []
        self._predicted_prices = []
        self._levels = {}
        self._clear_all()

    def set_title(self, title: str):
        """Set chart title."""
        if HAS_PYQTGRAPH and self.plot_widget is not None:
            try:
                self.plot_widget.setTitle(
                    title, color='#c9d1d9', size='12pt'
                )
            except Exception:
                pass

    def get_bar_count(self) -> int:
        """Get current number of bars."""
        return len(self._bars)

# MINI CHART (for watchlist) - unchanged

class MiniChart(QWidget):
    """Compact mini chart for watchlist items."""

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
        """Update mini chart."""
        self._prices = list(prices) if prices else []

        if not HAS_PYQTGRAPH or not self._prices or self.line is None:
            return

        try:
            x = np.arange(len(self._prices))
            y = np.array(self._prices, dtype=float)

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