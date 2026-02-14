# ui/charts.py

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QMenu, QVBoxLayout, QWidget

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
                body_height = top - bot
                if body_height <= 1e-9:
                    # Doji: keep as a thin horizontal mark without inflating body.
                    p.drawLine(
                        pg.QtCore.QPointF(t - (w / 2.0), c),
                        pg.QtCore.QPointF(t + (w / 2.0), c),
                    )
                else:
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
    trade_requested = pyqtSignal(str, float)  # side, price

    def __init__(self, parent=None):
        super().__init__(parent)

        self._bars: list[dict] = []
        self._actual_prices: list[float] = []
        self._predicted_prices: list[float] = []
        self._levels: dict[str, float] = {}

        self.plot_widget = None
        self.candles = None           # Layer 3: Candlesticks (top)
        self.actual_line = None       # Layer 2: Price line (middle)
        self.predicted_line = None    # Layer 1: Prediction (bottom)
        self.level_lines: dict[str, object] = {}
        self.overlay_lines: dict[str, object] = {}
        self.overlay_enabled: dict[str, bool] = {
            "sma20": True,
            "sma50": True,
            "ema21": True,
            "bb_upper": True,
            "bb_lower": True,
            "vwap20": True,
        }
        self._manual_zoom: bool = False

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
        self.plot_widget.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.plot_widget.customContextMenuRequested.connect(
            self._on_context_menu
        )

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
        self.overlay_lines = {
            "sma20": self.plot_widget.plot(
                pen=pg.mkPen(color="#e3b341", width=1),
                name="SMA20",
            ),
            "sma50": self.plot_widget.plot(
                pen=pg.mkPen(color="#d2a8ff", width=1, style=Qt.PenStyle.DashLine),
                name="SMA50",
            ),
            "ema21": self.plot_widget.plot(
                pen=pg.mkPen(color="#ffa657", width=1.2, style=Qt.PenStyle.DotLine),
                name="EMA21",
            ),
            "bb_upper": self.plot_widget.plot(
                pen=pg.mkPen(color="#8b949e", width=1, style=Qt.PenStyle.DashLine),
                name="BB Upper",
            ),
            "bb_lower": self.plot_widget.plot(
                pen=pg.mkPen(color="#8b949e", width=1, style=Qt.PenStyle.DashLine),
                name="BB Lower",
            ),
            "vwap20": self.plot_widget.plot(
                pen=pg.mkPen(color="#79c0ff", width=1, style=Qt.PenStyle.DotLine),
                name="VWAP20",
            ),
        }

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
        bars: list[dict],
        predicted_prices: list[float] = None,
        levels: dict[str, float] = None,
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
            closes: list[float] = []
            ohlc: list[tuple] = []
            prev_close: float | None = None
            default_iv = str(
                self._bars[-1].get("interval", "1m") if self._bars else "1m"
            ).lower()

            # Render the full loaded window (7-day bars are prepared in app layer).
            # Keep a high cap for safety on very large inputs.
            render_bars = self._bars[-3000:]
            recent_range_pcts: list[float] = []
            for b in render_bars:
                try:
                    o = float(b.get("open", 0) or 0)
                    h = float(b.get("high", 0) or 0)
                    l_val = float(b.get("low", 0) or 0)
                    c = float(b.get("close", 0) or 0)

                    if c <= 0:
                        continue

                    # Drop impossible jumps between consecutive candles
                    # to prevent broken spikes from corrupting the chart.
                    if prev_close and prev_close > 0:
                        jump = abs(c / prev_close - 1.0)
                        if jump > 0.20:
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

                    # Clamp obvious outlier candle ranges to avoid visual spikes
                    # from bad ticks or malformed bars.
                    iv = str(b.get("interval", default_iv) or default_iv).lower()
                    if iv == "1m":
                        max_move = 0.012
                        body_cap_pct = 0.004
                        wick_cap_pct = 0.008
                    elif iv == "5m":
                        max_move = 0.025
                        body_cap_pct = 0.009
                        wick_cap_pct = 0.016
                    elif iv in ("15m", "30m", "60m", "1h"):
                        max_move = 0.045
                        body_cap_pct = 0.015
                        wick_cap_pct = 0.03
                    else:
                        max_move = 0.12
                        body_cap_pct = 0.04
                        wick_cap_pct = 0.08
                    ref = prev_close if (prev_close and prev_close > 0) else c
                    if ref > 0:
                        hi_cap = ref * (1.0 + max_move)
                        lo_cap = ref * (1.0 - max_move)
                        h = min(h, hi_cap)
                        l_val = max(l_val, lo_cap)
                        o = min(max(o, lo_cap), hi_cap)
                        c = min(max(c, lo_cap), hi_cap)
                        # Prevent oversized candle bodies from stale/replayed partial bars.
                        body_cap = max(ref * body_cap_pct, 1e-8)
                        body = c - o
                        if abs(body) > body_cap:
                            if body > 0:
                                o = c - body_cap
                            else:
                                o = c + body_cap

                        top = max(o, c)
                        bot = min(o, c)
                        wick_cap = max(ref * wick_cap_pct, body_cap * 1.25)
                        h = min(h, top + wick_cap)
                        l_val = max(l_val, bot - wick_cap)
                        if h < l_val:
                            h, l_val = l_val, h

                    # Adaptive outlier trim based on recent candle ranges.
                    # This suppresses occasional feed spikes without flattening
                    # genuinely volatile symbols.
                    rng_pct = (h - l_val) / max(c, 1e-8)
                    if recent_range_pcts:
                        med = float(np.median(recent_range_pcts[-80:]))
                        if iv == "1m":
                            floor = 0.006
                        elif iv == "5m":
                            floor = 0.010
                        elif iv in ("15m", "30m", "60m", "1h"):
                            floor = 0.018
                        else:
                            floor = 0.035
                        outlier_cap = max(floor, med * 3.6)
                        target_cap = max(floor * 0.85, med * 2.2)
                        if rng_pct > outlier_cap:
                            top = max(o, c)
                            bot = min(o, c)
                            body = max(0.0, top - bot)
                            allow = max(0.0, (target_cap * max(c, 1e-8)) - body)
                            h = min(h, top + (allow * 0.5))
                            l_val = max(l_val, bot - (allow * 0.5))
                            if h < l_val:
                                h, l_val = l_val, h
                            rng_pct = (h - l_val) / max(c, 1e-8)
                    if np.isfinite(rng_pct) and rng_pct > 0:
                        recent_range_pcts.append(float(min(rng_pct, 1.0)))

                    x_pos = len(closes)
                    ohlc.append((x_pos, o, c, l_val, h))
                    closes.append(c)
                    prev_close = c
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

            self._update_overlay_lines(render_bars, closes)
            self._update_level_lines()

            # Auto-range unless user enabled manual zoom.
            if self.plot_widget is not None and not self._manual_zoom:
                self.plot_widget.autoRange()

        except Exception as e:
            log.warning(f"Chart update failed: {e}")

    # =========================================================================
    # BACKWARD COMPATIBLE METHODS - Now delegate to update_chart()
    # =========================================================================

    def update_candles(
        self,
        bars: list[dict],
        predicted_prices: list[float] = None,
        levels: dict[str, float] = None,
    ):
        """
        Update chart with candlestick bar data.

        BACKWARD COMPATIBLE: This now delegates to update_chart()
        so all three layers are drawn together.
        """
        self.update_chart(bars, predicted_prices, levels)

    def update_data(
        self,
        actual_prices: list[float],
        predicted_prices: list[float] = None,
        levels: dict[str, float] = None
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
            for line in self.overlay_lines.values():
                try:
                    line.clear()
                except Exception:
                    pass
            self._update_level_lines()
        except Exception as e:
            log.debug(f"Chart clear failed: {e}")

    def _rolling_mean(self, values: np.ndarray, window: int) -> np.ndarray:
        if len(values) < window or window <= 1:
            return np.full(len(values), np.nan)
        out = np.full(len(values), np.nan)
        c = np.cumsum(np.insert(values, 0, 0.0))
        out[window - 1:] = (c[window:] - c[:-window]) / float(window)
        return out

    def _ema(self, values: np.ndarray, span: int) -> np.ndarray:
        if len(values) == 0:
            return np.array([])
        alpha = 2.0 / (span + 1.0)
        out = np.empty(len(values), dtype=float)
        out[0] = values[0]
        for i in range(1, len(values)):
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
        return out

    def _plot_series(self, key: str, y: np.ndarray):
        line = self.overlay_lines.get(key)
        if line is None:
            return
        if not self.overlay_enabled.get(key, True):
            line.clear()
            return
        if y.size == 0:
            line.clear()
            return
        x = np.arange(len(y))
        mask = ~np.isnan(y)
        if not np.any(mask):
            line.clear()
            return
        line.setData(x[mask], y[mask])

    def _update_overlay_lines(self, bars: list[dict], closes: list[float]):
        if not HAS_PYQTGRAPH or not self.overlay_lines:
            return
        arr = np.array(closes, dtype=float)
        self._plot_series("sma20", self._rolling_mean(arr, 20))
        self._plot_series("sma50", self._rolling_mean(arr, 50))
        self._plot_series("ema21", self._ema(arr, 21))
        bb_mid = self._rolling_mean(arr, 20)
        bb_std = np.full(len(arr), np.nan)
        if len(arr) >= 20:
            for i in range(19, len(arr)):
                bb_std[i] = float(np.std(arr[i - 19:i + 1]))
        self._plot_series("bb_upper", bb_mid + (2.0 * bb_std))
        self._plot_series("bb_lower", bb_mid - (2.0 * bb_std))

        if bars:
            highs = []
            lows = []
            vols = []
            for b in bars:
                c = float(b.get("close", 0) or 0)
                if c <= 0:
                    continue
                h = float(b.get("high", c) or c)
                low = float(b.get("low", c) or c)
                v = float(b.get("volume", 0) or 0)
                highs.append(h if h > 0 else c)
                lows.append(low if low > 0 else c)
                vols.append(max(0.0, v))
            n = min(len(highs), len(arr))
            if n <= 0:
                return
            tp = (np.array(highs[:n]) + np.array(lows[:n]) + arr[:n]) / 3.0
            v = np.array(vols[:n], dtype=float)
            vwap = np.full(n, np.nan)
            if len(tp) >= 20:
                for i in range(19, len(tp)):
                    vv = v[i - 19:i + 1]
                    denom = float(np.sum(vv))
                    if denom > 0:
                        vwap[i] = float(np.sum(tp[i - 19:i + 1] * vv) / denom)
                    else:
                        vwap[i] = float(np.mean(tp[i - 19:i + 1]))
            self._plot_series("vwap20", vwap)

    def _on_context_menu(self, pos):
        if not HAS_PYQTGRAPH or self.plot_widget is None:
            return
        if not self._actual_prices:
            return

        try:
            scene_pos = self.plot_widget.mapToScene(pos)
            view_pos = self.plot_widget.plotItem.vb.mapSceneToView(scene_pos)
            price = float(view_pos.y())
        except Exception:
            price = float(self._actual_prices[-1])

        if not np.isfinite(price) or price <= 0:
            price = float(self._actual_prices[-1])

        menu = QMenu(self)
        buy_action = menu.addAction(f"Buy @ {price:.2f}")
        sell_action = menu.addAction(f"Sell @ {price:.2f}")
        menu.addSeparator()
        overlay_actions: dict[object, str] = {}
        for key, name in (
            ("sma20", "SMA20"),
            ("sma50", "SMA50"),
            ("ema21", "EMA21"),
            ("bb_upper", "Bollinger"),
            ("vwap20", "VWAP20"),
        ):
            act = menu.addAction(f"Overlay: {name}")
            act.setCheckable(True)
            if key == "bb_upper":
                checked = bool(self.overlay_enabled.get("bb_upper", True) and self.overlay_enabled.get("bb_lower", True))
            else:
                checked = bool(self.overlay_enabled.get(key, True))
            act.setChecked(checked)
            overlay_actions[act] = key
        menu.addSeparator()
        reset_view = menu.addAction("Reset View")
        chosen = menu.exec(self.plot_widget.mapToGlobal(pos))
        if chosen == buy_action:
            self.trade_requested.emit("buy", float(price))
        elif chosen == sell_action:
            self.trade_requested.emit("sell", float(price))
        elif chosen == reset_view:
            self.reset_view()
        elif chosen in overlay_actions:
            self._toggle_overlay(overlay_actions[chosen])

    def _toggle_overlay(self, key: str):
        if key == "bb_upper":
            new_state = not (
                self.overlay_enabled.get("bb_upper", True)
                and self.overlay_enabled.get("bb_lower", True)
            )
            self.overlay_enabled["bb_upper"] = new_state
            self.overlay_enabled["bb_lower"] = new_state
        else:
            self.overlay_enabled[key] = not self.overlay_enabled.get(key, True)
        if self._bars:
            closes = []
            for b in self._bars[-3000:]:
                try:
                    c = float(b.get("close", 0) or 0)
                    if c > 0:
                        closes.append(c)
                except Exception:
                    continue
            self._update_overlay_lines(self._bars[-3000:], closes)

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

    def zoom_in(self):
        """Zoom into current view."""
        if not HAS_PYQTGRAPH or self.plot_widget is None:
            return
        try:
            vb = self.plot_widget.plotItem.vb
            vb.scaleBy((0.8, 0.8))
            self._manual_zoom = True
        except Exception:
            pass

    def zoom_out(self):
        """Zoom out from current view."""
        if not HAS_PYQTGRAPH or self.plot_widget is None:
            return
        try:
            vb = self.plot_widget.plotItem.vb
            vb.scaleBy((1.25, 1.25))
            self._manual_zoom = True
        except Exception:
            pass

    def reset_view(self):
        """Reset view and resume auto-follow."""
        if not HAS_PYQTGRAPH or self.plot_widget is None:
            return
        try:
            self._manual_zoom = False
            self.plot_widget.autoRange()
        except Exception:
            pass

    def set_title(self, title: str):
        """Set chart title."""
        if HAS_PYQTGRAPH and self.plot_widget is not None:
            try:
                self.plot_widget.setTitle(
                    title, color='#c9d1d9', size='12pt'
                )
            except Exception:
                pass

    def set_overlay_enabled(self, key: str, enabled: bool):
        """
        Public overlay toggle for UI controls.
        For Bollinger, use key='bbands' to control both bands.
        """
        if key == "bbands":
            self.overlay_enabled["bb_upper"] = bool(enabled)
            self.overlay_enabled["bb_lower"] = bool(enabled)
        else:
            self.overlay_enabled[str(key)] = bool(enabled)
        if self._bars:
            closes = []
            for b in self._bars[-3000:]:
                try:
                    c = float(b.get("close", 0) or 0)
                    if c > 0:
                        closes.append(c)
                except Exception:
                    continue
            self._update_overlay_lines(self._bars[-3000:], closes)

    def get_bar_count(self) -> int:
        """Get current number of bars."""
        return len(self._bars)

# MINI CHART (for watchlist) - unchanged

class MiniChart(QWidget):
    """Compact mini chart for watchlist items."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 40)
        self._prices: list[float] = []
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

    def update_data(self, prices: list[float]):
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
