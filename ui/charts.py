# ui/charts.py
import math
import time
from collections import Counter

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QFont
from PyQt6.QtWidgets import QLabel, QMenu, QToolTip, QVBoxLayout, QWidget

from ui.modern_theme import ModernColors, ModernFonts, get_primary_font_family
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
        """Custom candlestick chart item for pyqtgraph.

        Data format: list of (x, open, close, low, high) tuples.
        """

        def __init__(self) -> None:
            super().__init__()
            self.data: list[tuple] = []
            self._picture: pg.QtGui.QPicture | None = None

        def setData(self, data: list[tuple] | None) -> None:
            """Set candlestick data. Each item: (x, open, close, low, high)."""
            self.data = data or []
            self._picture = None
            self.prepareGeometryChange()
            self.update()

        def _generate_picture(self) -> None:
            """Generate QPicture for all candles."""
            pic = pg.QtGui.QPicture()
            p = pg.QtGui.QPainter(pic)
            w = 0.18
            body_ratio = 0.54  # keep a small visible gap between candles
            try:
                xs = [float(d[0]) for d in self.data if len(d) >= 5]
                if len(xs) >= 2:
                    xs_sorted = sorted(xs)
                    diffs = np.diff(np.array(xs_sorted, dtype=float))
                    diffs = diffs[np.isfinite(diffs)]
                    diffs = diffs[diffs > 0]
                    if diffs.size > 0:
                        step = float(np.median(diffs))
                        # Keep body width proportional to local x spacing.
                        w = max(step * body_ratio, 1e-6)
                elif len(xs) == 1:
                    # Single-bar fallback should stay narrow; avoid giant blocks.
                    w = 0.14
            except Exception:
                w = 0.18

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
                    pg.mkColor(ModernColors.ACCENT_SUCCESS)
                    if up
                    else pg.mkColor(ModernColors.ACCENT_DANGER)
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

        def paint(self, p: pg.QtGui.QPainter, *args: object) -> None:
            """Paint the candlestick item."""
            if self._picture is None:
                self._generate_picture()
            if self._picture is not None:
                p.drawPicture(0, 0, self._picture)

        def boundingRect(self) -> pg.QtCore.QRectF:
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
    CandlestickItem = None  # type: ignore[assignment]

# MAIN STOCK CHART - FIXED VERSION

class StockChart(QWidget):
    """Interactive stock chart with THREE layers:

    Layer 1 (BOTTOM): Prediction line (dashed green) - AI forecast
    Layer 2 (MIDDLE): Price line (solid blue) - connects close prices  
    Layer 3 (TOP): Candlesticks (red/green) - OHLCV bars

    Plus: Trading level lines (stop loss, targets)
    """
    trade_requested = pyqtSignal(str, float)  # side, price

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._bars: list[dict] = []
        self._actual_prices: list[float] = []
        self._predicted_prices: list[float] = []
        self._predicted_prices_low: list[float] = []
        self._predicted_prices_high: list[float] = []
        self._levels: dict[str, float] = {}
        self._candle_meta: list[dict] = []
        self._hover_proxy: object | None = None
        self._last_hover_index: int | None = None
        self._last_hover_tooltip: str = ""

        self.plot_widget: object | None = None
        self.candles: object | None = None           # Layer 3: Candlesticks (top)
        self.actual_line: object | None = None       # Layer 2: Price line (middle)
        self.predicted_line: object | None = None    # Layer 1: Prediction (bottom)
        self.predicted_low_line: object | None = None
        self.predicted_high_line: object | None = None
        self.level_lines: dict[str, object] = {}
        self.overlay_lines: dict[str, object] = {}
        self.overlay_enabled: dict[str, bool] = {
            "sma20": True,
            "sma50": True,
            "sma200": False,
            "ema21": True,
            "ema55": False,
            "bb_upper": True,
            "bb_lower": True,
            "vwap20": True,
        }
        self._manual_zoom: bool = False
        self._dbg_last_emit: dict[str, float] = {}
        self._forecast_x_gap: float = 0.18

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup chart UI with pyqtgraph or fallback."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_PYQTGRAPH:
            self._setup_pyqtgraph()
        else:
            self._setup_fallback()

    @staticmethod
    def _coerce_list(values: object | None) -> list[object]:
        """Convert optional iterable values to a list safely."""
        if values is None:
            return []
        try:
            return list(values)
        except Exception:
            return []

    def _dbg_log(
        self,
        key: str,
        message: str,
        *,
        min_gap_seconds: float = 1.5,
        level: str = "info",
    ) -> None:
        """Throttled chart diagnostics to avoid log flooding."""
        try:
            now = time.monotonic()
            prev = float(self._dbg_last_emit.get(key, 0.0))
            if (now - prev) < float(max(0.0, min_gap_seconds)):
                return
            self._dbg_last_emit[key] = now
            msg = f"[DBG] {message}"
            if str(level).lower() == "warning":
                log.warning(msg)
            else:
                log.info(msg)
        except Exception:
            pass

    @staticmethod
    def _format_bar_timestamp(bar: dict) -> str:
        if not isinstance(bar, dict):
            return ""  # type: ignore[unreachable]
        for key in ("timestamp", "time", "datetime", "date", "_ts_epoch"):
            raw = bar.get(key)
            if raw is None:
                continue
            ts = pd.NaT
            try:
                if isinstance(raw, (int, float, np.integer, np.floating)):
                    v = float(raw)
                    if not np.isfinite(v):
                        continue
                    if abs(v) >= 1e11:
                        v /= 1000.0
                    ts = pd.to_datetime(v, unit="s", errors="coerce", utc=True)
                else:
                    ts = pd.to_datetime(raw, errors="coerce")
            except Exception:
                ts = pd.NaT

            if pd.isna(ts):
                text = str(raw).strip()
                if text:
                    return text
                continue

            try:
                ts_obj = pd.Timestamp(ts)
                if ts_obj.tzinfo is not None:
                    ts_obj = ts_obj.tz_convert("Asia/Shanghai").tz_localize(None)
                else:
                    ts_obj = ts_obj.tz_localize(None)
            except Exception:
                try:
                    ts_obj = pd.Timestamp(ts).tz_localize(None)
                except Exception:
                    ts_obj = pd.Timestamp(ts)

            if (
                int(getattr(ts_obj, "hour", 0)) == 0
                and int(getattr(ts_obj, "minute", 0)) == 0
                and int(getattr(ts_obj, "second", 0)) == 0
            ):
                return ts_obj.strftime("%Y-%m-%d")  # type: ignore[no-any-return]
            return ts_obj.strftime("%Y-%m-%d %H:%M")  # type: ignore[no-any-return]
        return ""

    @staticmethod
    def _format_number(value: object, decimals: int = 2) -> str:
        try:
            v = float(value)
            if not np.isfinite(v):
                return "--"
            return f"{v:.{int(max(0, decimals))}f}"
        except Exception:
            return "--"

    @staticmethod
    def _format_volume(value: object) -> str:
        try:
            v = float(value)
            if not np.isfinite(v):
                return "--"
            if abs(v) >= 1:
                return f"{int(round(v)):,}"
            return f"{v:.2f}"
        except Exception:
            return "--"

    def _build_candle_tooltip(self, meta: dict) -> str:
        ts_text = str(meta.get("ts", "") or "").strip() or "--"
        o = float(meta.get("open", 0.0) or 0.0)
        h = float(meta.get("high", 0.0) or 0.0)
        l_val = float(meta.get("low", 0.0) or 0.0)
        c = float(meta.get("close", 0.0) or 0.0)
        v = float(meta.get("volume", 0.0) or 0.0)
        amount = float(meta.get("amount", 0.0) or 0.0)
        prev_c = float(meta.get("prev_close", 0.0) or 0.0)

        change = 0.0
        change_pct = 0.0
        if prev_c > 0:
            change = c - prev_c
            change_pct = (change / prev_c) * 100.0

        sign = "+" if change > 0 else ""
        pct_sign = "+" if change_pct > 0 else ""

        return (
            f"Time: {ts_text}\n"
            f"Open: {self._format_number(o, 3)}\n"
            f"High: {self._format_number(h, 3)}\n"
            f"Low: {self._format_number(l_val, 3)}\n"
            f"Close: {self._format_number(c, 3)}\n"
            f"Change: {sign}{self._format_number(change, 3)} ({pct_sign}{self._format_number(change_pct, 2)}%)\n"
            f"Volume: {self._format_volume(v)}\n"
            f"Amount: {self._format_number(amount, 2)}"
        )

    def _hide_candle_tooltip(self) -> None:
        self._last_hover_index = None
        self._last_hover_tooltip = ""
        try:
            QToolTip.hideText()
        except Exception:
            pass

    def _on_plot_mouse_moved(self, evt: object) -> None:
        if not HAS_PYQTGRAPH or self.plot_widget is None:
            return  # type: ignore[unreachable]
        if not self._candle_meta:
            self._hide_candle_tooltip()
            return

        try:
            pos = evt[0] if isinstance(evt, (tuple, list)) else evt
            if pos is None:
                self._hide_candle_tooltip()
                return
            if not self.plot_widget.sceneBoundingRect().contains(pos):
                self._hide_candle_tooltip()
                return
            view_pos = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x = float(view_pos.x())
            y = float(view_pos.y())
        except Exception:
            self._hide_candle_tooltip()
            return

        idx = int(round(x))
        if idx < 0 or idx >= len(self._candle_meta):
            self._hide_candle_tooltip()
            return

        # FIX: Tighter hover tolerance to prevent flickering between candles
        if abs(x - float(idx)) > 0.45:
            self._hide_candle_tooltip()
            return

        meta = self._candle_meta[idx]
        try:
            high = float(meta.get("high", 0.0) or 0.0)
            low = float(meta.get("low", 0.0) or 0.0)
            close = float(meta.get("close", 0.0) or 0.0)
        except Exception:
            self._hide_candle_tooltip()
            return

        if high <= 0 or low <= 0 or high < low:
            self._hide_candle_tooltip()
            return

        # FIX: More generous Y padding to prevent tooltip flickering
        y_pad = max((high - low) * 0.40, max(abs(close) * 0.003, 0.015))
        if y < (low - y_pad) or y > (high + y_pad):
            self._hide_candle_tooltip()
            return

        tooltip_text = self._build_candle_tooltip(meta)
        # FIX: Only update if index changed (not full text comparison)
        if idx == self._last_hover_index:
            return

        self._last_hover_index = idx
        self._last_hover_tooltip = tooltip_text
        try:
            QToolTip.showText(QCursor.pos(), tooltip_text, self.plot_widget)
        except Exception:
            pass

    def _setup_pyqtgraph(self) -> None:
        """Setup pyqtgraph chart with all three layers."""
        pg.setConfigOptions(
            antialias=True,
            background=ModernColors.BG_PRIMARY,
            foreground=ModernColors.TEXT_PRIMARY,
        )

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel(
            "left",
            "Price",
            units="CNY",
            color=ModernColors.TEXT_SECONDARY,
        )
        self.plot_widget.setLabel(
            "bottom",
            "Time",
            units="bars",
            color=ModernColors.TEXT_SECONDARY,
        )
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.setBackground(ModernColors.BG_PRIMARY)
        self.plot_widget.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.plot_widget.customContextMenuRequested.connect(
            self._on_context_menu
        )
        try:
            plot_item = self.plot_widget.getPlotItem()
            axis_pen = pg.mkPen(color=ModernColors.BORDER_DEFAULT, width=1)
            text_pen = pg.mkPen(color=ModernColors.TEXT_SECONDARY, width=1)
            tick_font = QFont(get_primary_font_family(), ModernFonts.SIZE_XS)
            for axis_name in ("left", "bottom"):
                axis = plot_item.getAxis(axis_name)
                axis.setPen(axis_pen)
                axis.setTextPen(text_pen)
                axis.setTickFont(tick_font)
            plot_item.showAxis("top", show=False)
            plot_item.showAxis("right", show=False)
        except Exception as exc:
            log.debug("Chart axis styling failed: %s", exc)

        # === Layer 1 (BOTTOM): Prediction line - dashed cyan ===
        self.predicted_line = self.plot_widget.plot(
            pen=pg.mkPen(
                color=ModernColors.ACCENT_SECONDARY,
                width=2,
                style=Qt.PenStyle.DashLine
            ),
            name='AI Prediction'
        )
        self.predicted_low_line = self.plot_widget.plot(
            pen=pg.mkPen(
                color=ModernColors.ACCENT_WARNING,
                width=1,
                style=Qt.PenStyle.DotLine,
            ),
            name='Forecast Low',
        )
        self.predicted_high_line = self.plot_widget.plot(
            pen=pg.mkPen(
                color=ModernColors.ACCENT_WARNING,
                width=1,
                style=Qt.PenStyle.DotLine,
            ),
            name='Forecast High',
        )

        # === Layer 2 (MIDDLE): Price line - solid blue ===
        self.actual_line = self.plot_widget.plot(
            pen=pg.mkPen(color=ModernColors.ACCENT_INFO, width=1.5),
            name="Price",
        )

        # === Layer 3 (TOP): Candlesticks ===
        self.candles = CandlestickItem()
        self.plot_widget.addItem(self.candles)

        self.level_lines = {}
        self.overlay_lines = {
            "sma20": self.plot_widget.plot(
                pen=pg.mkPen(color=ModernColors.ACCENT_WARNING, width=1),
                name="SMA20",
            ),
            "sma50": self.plot_widget.plot(
                pen=pg.mkPen(color="#8ca4d7", width=1, style=Qt.PenStyle.DashLine),
                name="SMA50",
            ),
            "sma200": self.plot_widget.plot(
                pen=pg.mkPen(
                    color=ModernColors.TEXT_SECONDARY,
                    width=1,
                    style=Qt.PenStyle.DashLine,
                ),
                name="SMA200",
            ),
            "ema21": self.plot_widget.plot(
                pen=pg.mkPen(color="#fdba74", width=1.2, style=Qt.PenStyle.DotLine),
                name="EMA21",
            ),
            "ema55": self.plot_widget.plot(
                pen=pg.mkPen(color="#edcf7c", width=1, style=Qt.PenStyle.DotLine),
                name="EMA55",
            ),
            "bb_upper": self.plot_widget.plot(
                pen=pg.mkPen(color="#8ab2ff", width=1, style=Qt.PenStyle.DashLine),
                name="BB Upper",
            ),
            "bb_lower": self.plot_widget.plot(
                pen=pg.mkPen(color="#8ab2ff", width=1, style=Qt.PenStyle.DashLine),
                name="BB Lower",
            ),
            "vwap20": self.plot_widget.plot(
                pen=pg.mkPen(
                    color=ModernColors.ACCENT_INFO,
                    width=1,
                    style=Qt.PenStyle.DotLine,
                ),
                name="VWAP20",
            ),
        }

        self.plot_widget.addLegend()
        try:
            self._hover_proxy = pg.SignalProxy(
                self.plot_widget.scene().sigMouseMoved,
                rateLimit=45,
                slot=self._on_plot_mouse_moved,
            )
        except Exception as exc:
            self._hover_proxy = None
            log.debug("Failed to attach chart hover proxy: %s", exc)

        self.layout().addWidget(self.plot_widget)

    def _setup_fallback(self) -> None:
        """Setup fallback when pyqtgraph not available."""
        self.fallback_label = QLabel(
            "Chart requires pyqtgraph\n\n"
            "Install with: pip install pyqtgraph"
        )
        self.fallback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fallback_label.setStyleSheet(
            f"""
            QLabel {{
                background-color: {ModernColors.BG_PRIMARY};
                color: {ModernColors.ACCENT_INFO};
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 12px;
                font-size: 14px;
                padding: 20px;
            }}
            """
        )
        self.layout().addWidget(self.fallback_label)

    # =========================================================================
    # UNIFIED UPDATE METHOD - Draws all three layers together
    # =========================================================================

    def update_chart(
        self,
        bars: list[dict],
        predicted_prices: list[float] = None,
        predicted_prices_low: list[float] = None,
        predicted_prices_high: list[float] = None,
        levels: dict[str, float] = None,
    ) -> None:
        """UNIFIED update method - draws all three layers together.

        This is the PRIMARY method that should be called for updates.
        Both update_candles() and update_data() now delegate to this.

        Args:
            bars: List of OHLCV dicts with keys: open, high, low, close
            predicted_prices: AI forecast prices (the "guessed graph")
            predicted_prices_low: Lower uncertainty envelope for forecast
            predicted_prices_high: Upper uncertainty envelope for forecast
            levels: Trading levels dict (stop_loss, target_1, etc.)
        """
        self._bars = self._coerce_list(bars)
        self._predicted_prices = self._coerce_list(predicted_prices)
        self._predicted_prices_low = self._coerce_list(predicted_prices_low)
        self._predicted_prices_high = self._coerce_list(predicted_prices_high)
        self._levels = levels or {}
        self._candle_meta = []

        # FIX: Add diagnostic logging for empty bars
        if not self._bars:
            log.warning("Chart update_chart called with EMPTY bars list")
        
        if not HAS_PYQTGRAPH:
            log.warning("Chart update skipped: pyqtgraph not available")
            return

        if not self._bars:
            log.warning("Chart update skipped: no bars after coerce")
            self._clear_all()
            return
        
        # FIX: Log bar count for diagnostics
        log.info(f"Chart update: {len(self._bars)} bars received")

        try:
            # Parse bar data - extract closes for line, OHLC for candles.
            # Data is already sanitized and filtered by app.py pipeline
            # (_prepare_chart_bars_for_interval, _sanitize_ohlc, etc.).
            # This layer only does basic validity checks and builds
            # the rendering data structures.
            closes: list[float] = []
            ohlc: list[tuple] = []
            prev_close: float | None = None
            prev_day_key: str | None = None

            default_iv_raw = str(
                self._bars[-1].get("interval", "1m") if self._bars else "1m"
            ).lower()
            interval_aliases = {
                "1h": "60m",
                "60min": "60m",
                "60mins": "60m",
                "daily": "1d",
                "1day": "1d",
                "day": "1d",
            }
            default_iv = interval_aliases.get(default_iv_raw, default_iv_raw)
            try:
                iv_tokens = []
                for row in self._bars:
                    if not isinstance(row, dict):
                        continue
                    iv_raw = str(row.get("interval", "") or "").strip().lower()
                    if not iv_raw:
                        continue
                    iv_tokens.append(interval_aliases.get(iv_raw, iv_raw))
                if iv_tokens:
                    default_iv = str(Counter(iv_tokens).most_common(1)[0][0])
            except Exception:
                pass

            render_bars = list(self._bars)
            try:
                def _bar_epoch(row: dict) -> float:
                    raw = row.get("_ts_epoch", row.get("timestamp", row.get("time", 0)))
                    try:
                        if isinstance(raw, (int, float, np.integer, np.floating)):
                            v = float(raw)
                            if not np.isfinite(v):
                                return 0.0
                            if abs(v) >= 1e11:
                                v /= 1000.0
                            return float(v)
                    except Exception:
                        pass
                    try:
                        ts = pd.to_datetime(raw, errors="coerce")
                        if pd.isna(ts):
                            return 0.0
                        return float(pd.Timestamp(ts).timestamp())
                    except Exception:
                        return 0.0

                render_bars.sort(key=_bar_epoch)
            except Exception:
                pass
            render_bars = render_bars[-3000:]
            diag = {
                "rows_total": int(len(render_bars)),
                "kept": 0,
                "drop_nonfinite": 0,
                "drop_interval": 0,
                "drop_parse": 0,
                "drop_shape": 0,
                "drop_scale": 0,
            }
            is_intraday = default_iv not in {"1d", "1wk", "1mo"}
            if is_intraday:
                # Render-side guard: keep intraday bars within realistic A-share
                # movement envelopes to prevent striped wick artifacts.
                jump_cap = 0.08
                body_cap = 0.014
                span_cap = 0.022
                wick_cap = 0.013
                scale_lo = 0.35
                scale_hi = 3.00
            else:
                jump_cap = 0.45
                body_cap = 0.55
                span_cap = 0.80
                wick_cap = 0.40
                scale_lo = 0.10
                scale_hi = 10.00
            recent_closes: list[float] = []
            rendered_bars: list[dict] = []

            for b in render_bars:
                try:
                    o = float(b.get("open", 0) or 0)
                    h = float(b.get("high", 0) or 0)
                    l_val = float(b.get("low", 0) or 0)
                    c = float(b.get("close", 0) or 0)

                    if not all(math.isfinite(v) for v in (o, h, l_val, c)):
                        diag["drop_nonfinite"] += 1
                        continue
                    if c <= 0:
                        diag["drop_nonfinite"] += 1
                        continue

                    bar_iv_raw = str(
                        b.get("interval", default_iv) or default_iv
                    ).lower()
                    bar_iv = interval_aliases.get(bar_iv_raw, bar_iv_raw)
                    if bar_iv != default_iv:
                        diag["drop_interval"] += 1
                        continue
                    ts_text = self._format_bar_timestamp(b)
                    day_key = str(ts_text[:10]) if len(ts_text) >= 10 else ""
                    day_boundary = bool(
                        prev_day_key is not None
                        and day_key
                        and day_key != prev_day_key
                    )

                    # Fix missing OHLC values — prefer close (doji) over
                    # prev_close to avoid creating artificial directional candles.
                    if o <= 0:
                        if (
                            is_intraday
                            and (not day_boundary)
                            and prev_close is not None
                            and prev_close > 0
                        ):
                            o = float(prev_close)
                        else:
                            o = c
                    if h <= 0:
                        h = max(o, c)
                    if l_val <= 0:
                        l_val = min(o, c)

                    # Ensure OHLC consistency
                    top = max(o, c)
                    bot = min(o, c)
                    if h < top:
                        h = top
                    if l_val > bot:
                        l_val = bot
                    if h < l_val:
                        h, l_val = l_val, h
                    o = min(max(o, l_val), h)
                    c = min(max(c, l_val), h)

                    # Last-line defense for rendering: drop extreme-scale bars
                    # that can still slip through upstream sanitation and would
                    # otherwise distort chart autoscaling.
                    if recent_closes:
                        ref_scale = float(np.median(np.asarray(recent_closes[-120:], dtype=float)))
                    elif prev_close is not None and prev_close > 0:
                        ref_scale = float(prev_close)
                    else:
                        ref_scale = float(c)
                    ref_scale = max(ref_scale, 1e-8)

                    scale_ratio = float(c) / ref_scale
                    if (scale_ratio < scale_lo) or (scale_ratio > scale_hi):
                        diag["drop_scale"] += 1
                        continue

                    if (
                        (not day_boundary)
                        and prev_close is not None
                        and prev_close > 0
                        and len(recent_closes) >= 2
                    ):
                        jump = abs(float(c) / max(float(prev_close), 1e-8) - 1.0)
                        if jump > jump_cap:
                            diag["drop_shape"] += 1
                            continue

                    body_pct = abs(float(o) - float(c)) / ref_scale
                    span_pct = abs(float(h) - float(l_val)) / ref_scale
                    top = max(float(o), float(c))
                    bot = min(float(o), float(c))
                    upper_wick = max(0.0, float(h) - top) / ref_scale
                    lower_wick = max(0.0, bot - float(l_val)) / ref_scale
                    shape_body_cap = float(body_cap * (2.6 if day_boundary else 1.0))
                    shape_span_cap = float(span_cap * (2.8 if day_boundary else 1.0))
                    shape_wick_cap = float(wick_cap * (2.8 if day_boundary else 1.0))
                    if (
                        body_pct > shape_body_cap
                        or span_pct > shape_span_cap
                        or upper_wick > shape_wick_cap
                        or lower_wick > shape_wick_cap
                    ):
                        diag["drop_shape"] += 1
                        continue

                    try:
                        vol = float(b.get("volume", 0) or 0)
                    except Exception:
                        vol = 0.0
                    if (not np.isfinite(vol)) or vol < 0:
                        vol = 0.0
                    try:
                        amount = float(b.get("amount", 0) or 0)
                    except Exception:
                        amount = 0.0
                    if not np.isfinite(amount):
                        amount = 0.0

                    rendered_bars.append(
                        {
                            "open": float(o),
                            "high": float(h),
                            "low": float(l_val),
                            "close": float(c),
                            "volume": float(vol),
                            "amount": float(amount),
                            "interval": str(bar_iv),
                            "timestamp": b.get("timestamp", b.get("time", "")),
                            "_ts_epoch": b.get("_ts_epoch", None),
                        }
                    )

                    x_pos = len(closes)
                    ohlc.append((x_pos, o, c, l_val, h))
                    self._candle_meta.append(
                        {
                            "x": x_pos,
                            "open": float(o),
                            "high": float(h),
                            "low": float(l_val),
                            "close": float(c),
                            "volume": float(vol),
                            "amount": float(amount),
                            "prev_close": (
                                float(prev_close)
                                if (prev_close is not None and prev_close > 0)
                                else None
                            ),
                            "ts": ts_text,
                            "interval": str(bar_iv),
                        }
                    )
                    closes.append(c)
                    recent_closes.append(float(c))
                    prev_close = c
                    if day_key:
                        prev_day_key = day_key
                    diag["kept"] += 1
                except (ValueError, TypeError):
                    diag["drop_parse"] += 1
                    continue

            if not closes:
                self._dbg_log(
                    f"chart_render:{default_iv}:empty",
                    (
                        f"chart render empty iv={default_iv} "
                        f"rows={diag['rows_total']} kept={diag['kept']} "
                        f"drop_nonfinite={diag['drop_nonfinite']} "
                        f"drop_interval={diag['drop_interval']} "
                        f"drop_shape={diag['drop_shape']} "
                        f"drop_scale={diag['drop_scale']} "
                        f"drop_parse={diag['drop_parse']}"
                    ),
                    min_gap_seconds=0.8,
                    level="warning",
                )
                self._clear_all()
                return

            # FIX: Verify arrays are aligned before rendering
            if len(ohlc) != len(closes) or len(ohlc) != len(self._candle_meta):
                # Arrays misaligned - rebuild from rendered_bars to ensure consistency
                self._dbg_log(
                    f"chart_render_realign:{default_iv}",
                    (
                        f"chart arrays misaligned iv={default_iv}: "
                        f"ohlc={len(ohlc)} closes={len(closes)} meta={len(self._candle_meta)}"
                    ),
                    min_gap_seconds=1.0,
                    level="warning",
                )
                # Rebuild from rendered_bars which is the source of truth
                ohlc = []
                closes = []
                self._candle_meta = []
                prev_close = None
                for i, b in enumerate(rendered_bars):
                    o = float(b.get("open", 0))
                    c = float(b.get("close", 0))
                    h = float(b.get("high", 0))
                    l_val = float(b.get("low", 0))
                    if o > 0 and c > 0 and h > 0 and l_val > 0:
                        ohlc.append((i, o, c, l_val, h))
                        closes.append(c)
                        self._candle_meta.append({
                            "x": i,
                            "open": o,
                            "high": h,
                            "low": l_val,
                            "close": c,
                            "volume": float(b.get("volume", 0)),
                            "amount": float(b.get("amount", 0)),
                            "prev_close": prev_close,
                            "ts": b.get("timestamp", ""),
                            "interval": str(b.get("interval", default_iv)),
                        })
                        prev_close = c

            if not closes or not ohlc:
                log.warning(f"Chart render: NO CANDLES after processing! closes={len(closes)} ohlc={len(ohlc)} rendered_bars={len(rendered_bars)}")
                
                # FIX: Fallback - try to render at least doji candles from rendered_bars
                if rendered_bars:
                    log.info(f"Chart fallback: attempting to render {len(rendered_bars)} bars as-is")
                    ohlc = []
                    closes = []
                    self._candle_meta = []
                    prev_close = None
                    for i, b in enumerate(rendered_bars[:100]):  # Limit to 100 bars
                        try:
                            o = float(b.get("open", 0))
                            c = float(b.get("close", 0))
                            h = float(b.get("high", c))
                            l_val = float(b.get("low", c))
                            
                            # Create doji if OHLC invalid
                            if o <= 0:
                                o = c
                            if h <= 0 or h < c:
                                h = c
                            if l_val <= 0 or l_val > c:
                                l_val = c
                            
                            if c > 0:
                                ohlc.append((i, o, c, l_val, h))
                                closes.append(c)
                                self._candle_meta.append({
                                    "x": i,
                                    "open": o,
                                    "high": h,
                                    "low": l_val,
                                    "close": c,
                                    "volume": float(b.get("volume", 0)),
                                    "amount": float(b.get("amount", 0)),
                                    "prev_close": prev_close,
                                    "ts": b.get("timestamp", ""),
                                    "interval": str(b.get("interval", default_iv)),
                                })
                                prev_close = c
                        except Exception as e:
                            log.debug(f"Fallback bar {i} failed: {e}")
                    
                    if not closes:
                        log.error("Chart fallback failed: still no candles")
                        self._clear_all()
                        return
                    log.info(f"Chart fallback: {len(ohlc)} candles rendered")
                else:
                    log.error("Chart render: no rendered_bars for fallback")
                    self._clear_all()
                    return
            
            # FIX: Log successful candle count
            log.info(f"Chart render: {len(ohlc)} candles prepared, {len(closes)} closes")

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
                    # Start forecast one bar AFTER the last candle so
                    # the prediction line does not overlap the last candle.
                    start_x = float(len(closes)) + float(self._forecast_x_gap)
                    x_pred = start_x + np.arange(
                        len(self._predicted_prices), dtype=float
                    )
                    y_pred = np.array(
                        list(self._predicted_prices),
                        dtype=float
                    )
                    # Avoid drawing a long artificial connector when the first
                    # forecast point is too far from the last real close.
                    last_close = float(closes[-1]) if closes else 0.0
                    first_pred = float(self._predicted_prices[0])
                    attach_cap = 0.035 if default_iv in {"1d", "1wk", "1mo"} else 0.018
                    connect_forecast = bool(
                        last_close > 0
                        and abs(first_pred / max(last_close, 1e-8) - 1.0) <= attach_cap
                    )
                    if connect_forecast:
                        x_conn = np.array(
                            [len(closes) - 1, start_x], dtype=float
                        )
                        y_conn = np.array(
                            [last_close, first_pred], dtype=float
                        )
                        x_full = np.concatenate([x_conn, x_pred])
                        y_full = np.concatenate([y_conn, y_pred])
                        self.predicted_line.setData(x_full, y_full)
                    else:
                        self.predicted_line.setData(x_pred, y_pred)
                    
                    # FIX: Ensure uncertainty bands are rendered
                    if (
                        self.predicted_low_line is not None
                        and self.predicted_high_line is not None
                        and self._predicted_prices_low
                        and self._predicted_prices_high
                        and len(self._predicted_prices_low) == len(self._predicted_prices)
                        and len(self._predicted_prices_high) == len(self._predicted_prices)
                    ):
                        y_low = np.array(
                            list(self._predicted_prices_low),
                            dtype=float,
                        )
                        y_high = np.array(
                            list(self._predicted_prices_high),
                            dtype=float,
                        )
                        # Do not connect uncertainty bands to last real close:
                        # that creates an artificial long vertical band jump at
                        # forecast start when band width is wide.
                        self.predicted_low_line.setData(x_pred, y_low)
                        self.predicted_high_line.setData(x_pred, y_high)
                    else:
                        # FIX: Clear bands only if data is truly missing
                        if self.predicted_low_line is not None:
                            if not self._predicted_prices_low or len(self._predicted_prices_low) != len(self._predicted_prices):
                                self.predicted_low_line.clear()
                        if self.predicted_high_line is not None:
                            if not self._predicted_prices_high or len(self._predicted_prices_high) != len(self._predicted_prices):
                                self.predicted_high_line.clear()
                    try:
                        p = y_pred
                        if p.size > 0:
                            anchor = float(closes[-1]) if closes[-1] > 0 else float(p[0])
                            span = float(np.max(p) - np.min(p)) / max(anchor, 1e-8)
                            steps = np.abs(np.diff(p)) / np.maximum(np.abs(p[:-1]), 1e-8) if p.size >= 2 else np.array([0.0])
                            max_step = float(np.max(steps)) if steps.size > 0 else 0.0
                            quiet_market = False
                            try:
                                real = np.asarray(closes[-96:], dtype=float)
                                real = real[np.isfinite(real)]
                                real = real[real > 0]
                                if real.size >= 8:
                                    real_anchor = float(np.median(real[-24:]))
                                    real_anchor = max(real_anchor, 1e-8)
                                    real_span = float(np.max(real) - np.min(real)) / real_anchor
                                    real_std = float(np.std(real)) / real_anchor
                                    quiet_market = bool(
                                        real_span <= 0.0030 or real_std <= 0.0010
                                    )
                            except Exception:
                                quiet_market = False
                            flips = 0
                            if p.size >= 3:
                                dirs = np.sign(np.diff(p))
                                dirs = dirs[dirs != 0]
                                if dirs.size >= 2:
                                    flips = int(np.sum(dirs[1:] != dirs[:-1]))
                            flip_ratio = (
                                float(flips) / float(max(1, (len(p) - 2)))
                                if len(p) >= 3
                                else 0.0
                            )
                            self._dbg_log(
                                f"forecast_render:{default_iv}",
                                (
                                    f"forecast render iv={default_iv} points={int(p.size)} "
                                    f"span={span:.2%} max_step={max_step:.2%} "
                                    f"flip={flip_ratio:.2f}"
                                ),
                                    min_gap_seconds=2.0,
                                    level="info",
                                )
                            flat_render = span <= 0.0012
                            if (flat_render and not quiet_market) or max_step >= 0.08:
                                self._dbg_log(
                                    f"forecast_render_warn:{default_iv}",
                                    (
                                        f"forecast render anomaly iv={default_iv}: "
                                        f"points={int(p.size)} span={span:.2%} "
                                        f"max_step={max_step:.2%} flip={flip_ratio:.2f}"
                                    ),
                                    min_gap_seconds=1.0,
                                    level="warning",
                                )
                            elif flat_render and quiet_market:
                                self._dbg_log(
                                    f"forecast_render_quiet:{default_iv}",
                                    (
                                        f"forecast render quiet-shape iv={default_iv}: "
                                        f"points={int(p.size)} span={span:.2%}"
                                    ),
                                    min_gap_seconds=2.0,
                                    level="info",
                                )
                    except Exception:
                        pass
                else:
                    self.predicted_line.clear()
                    if self.predicted_low_line is not None:
                        self.predicted_low_line.clear()
                    if self.predicted_high_line is not None:
                        self.predicted_high_line.clear()

            self._actual_prices = closes

            total_drops = int(
                diag["drop_nonfinite"]
                + diag["drop_interval"]
                + diag["drop_shape"]
                + diag["drop_scale"]
                + diag["drop_parse"]
            )
            diag_msg = (
                f"chart render iv={default_iv} rows={diag['rows_total']} "
                f"kept={diag['kept']} drops={total_drops} "
                f"(nf={diag['drop_nonfinite']} iv={diag['drop_interval']} "
                f"shape={diag['drop_shape']} scale={diag['drop_scale']} "
                f"parse={diag['drop_parse']})"
            )
            self._dbg_log(
                f"chart_render:{default_iv}",
                diag_msg,
                min_gap_seconds=2.0,
                level="info",
            )
            if total_drops > max(2, int(diag["rows_total"] * 0.08)):
                self._dbg_log(
                    f"chart_render_warn:{default_iv}",
                    f"chart render anomalies high iv={default_iv}: {diag_msg}",
                    min_gap_seconds=1.0,
                    level="warning",
                )

            self._update_overlay_lines(rendered_bars, closes)
            self._update_level_lines()

            # Auto-range unless user enabled manual zoom.
            # Focus viewport on candle data with only a small portion of
            # the forecast visible, so the prediction area does not
            # compress and distort the real candles.
            # FIX: Smoother auto-ranging with Y-axis padding to prevent jarring jumps
            if self.plot_widget is not None and not self._manual_zoom:
                n_candles = len(closes)
                n_pred = len(self._predicted_prices) if self._predicted_prices else 0
                if n_candles > 0 and n_pred > 0:
                    # FIX: Calculate Y range with padding before applying
                    closes_arr = np.array(closes, dtype=float)
                    y_min = float(np.min(closes_arr))
                    y_max = float(np.max(closes_arr))
                    y_range = y_max - y_min
                    y_padding = max(y_range * 0.08, y_max * 0.005, 0.1)
                    
                    # Apply Y range with padding
                    self.plot_widget.setYRange(
                        max(0, y_min - y_padding),
                        y_max + y_padding,
                        padding=0.02
                    )
                    
                    # X range: show candles + 30% of forecast width
                    pred_visible = max(1, int(n_pred * 0.3))
                    visible_candles = max(120, min(n_candles, 600))
                    x_min = max(0, n_candles - visible_candles)
                    x_max = (
                        float(n_candles)
                        + float(self._forecast_x_gap)
                        + float(pred_visible)
                    )
                    self.plot_widget.setXRange(
                        x_min, x_max, padding=0.02
                    )
                else:
                    # FIX: Use padded autoRange for candle-only mode
                    self.plot_widget.autoRange(padding=0.05)

        except Exception as e:
            log.warning(f"Chart update failed: {e}")

    # =========================================================================
    # BACKWARD COMPATIBLE METHODS - Now delegate to update_chart()
    # =========================================================================

    def update_candles(
        self,
        bars: list[dict],
        predicted_prices: list[float] = None,
        predicted_prices_low: list[float] = None,
        predicted_prices_high: list[float] = None,
        levels: dict[str, float] = None,
    ) -> None:
        """Update chart with candlestick bar data.

        BACKWARD COMPATIBLE: This now delegates to update_chart()
        so all three layers are drawn together.
        """
        self.update_chart(
            bars,
            predicted_prices,
            predicted_prices_low,
            predicted_prices_high,
            levels,
        )

    def update_data(
        self,
        actual_prices: list[float],
        predicted_prices: list[float] = None,
        predicted_prices_low: list[float] = None,
        predicted_prices_high: list[float] = None,
        levels: dict[str, float] = None
    ) -> None:
        """Update chart with line data.

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

        self.update_chart(
            bars,
            predicted_prices,
            predicted_prices_low,
            predicted_prices_high,
            levels,
        )

    # =========================================================================
    # =========================================================================

    def _clear_all(self) -> None:
        """Clear all chart elements."""
        self._candle_meta = []
        self._hide_candle_tooltip()
        try:
            if self.candles is not None:
                self.candles.setData([])
            if self.actual_line is not None:
                self.actual_line.clear()
            if self.predicted_line is not None:
                self.predicted_line.clear()
            if self.predicted_low_line is not None:
                self.predicted_low_line.clear()
            if self.predicted_high_line is not None:
                self.predicted_high_line.clear()
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

    def _plot_series(self, key: str, y: np.ndarray) -> None:
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

    def _update_overlay_lines(self, bars: list[dict], closes: list[float]) -> None:
        if not HAS_PYQTGRAPH or not self.overlay_lines:
            return
        arr = np.array(closes, dtype=float)
        self._plot_series("sma20", self._rolling_mean(arr, 20))
        self._plot_series("sma50", self._rolling_mean(arr, 50))
        self._plot_series("sma200", self._rolling_mean(arr, 200))
        self._plot_series("ema21", self._ema(arr, 21))
        self._plot_series("ema55", self._ema(arr, 55))
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

    def _on_context_menu(self, pos) -> None:
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
            ("sma200", "SMA200"),
            ("ema21", "EMA21"),
            ("ema55", "EMA55"),
            ("bb_upper", "Bollinger"),
            ("vwap20", "VWAP20"),
        ):
            act = menu.addAction(f"Overlay: {name}")
            act.setCheckable(True)
            if key == "bb_upper":
                # FIX: Bollinger Bands are shown as a pair - check both
                bb_upper = bool(self.overlay_enabled.get("bb_upper", True))
                bb_lower = bool(self.overlay_enabled.get("bb_lower", True))
                checked = bb_upper and bb_lower
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

    def _toggle_overlay(self, key: str) -> None:
        """Toggle overlay visibility with proper state management."""
        if key == "bb_upper":
            # FIX: Bollinger Bands toggle - both upper and lower together
            bb_upper = bool(self.overlay_enabled.get("bb_upper", True))
            bb_lower = bool(self.overlay_enabled.get("bb_lower", True))
            # Toggle OFF if both are on, otherwise toggle ON
            new_state = not (bb_upper and bb_lower)
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

    def _update_level_lines(self) -> None:
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
            "stop_loss": ModernColors.ACCENT_DANGER,
            "target_1": ModernColors.ACCENT_SUCCESS,
            "target_2": "#2ec695",
            "target_3": "#24aa83",
            "entry": ModernColors.ACCENT_INFO,
        }

        for name, price in self._levels.items():
            if price and price > 0:
                color = level_colors.get(name, ModernColors.TEXT_MUTED)
                try:
                    line = pg.InfiniteLine(
                        pos=price,
                        angle=0,
                        pen=pg.mkPen(
                            color=color,
                            width=1,
                            style=Qt.PenStyle.DotLine
                        ),
                        label=f'{name}: CNY {price:.2f}',
                        labelOpts={
                            'color': color,
                            'position': 0.95
                        }
                    )
                    self.plot_widget.addItem(line)
                    self.level_lines[name] = line
                except Exception as e:
                    log.debug(f"Failed to add level line {name}: {e}")

    def clear(self) -> None:
        """Clear all data from chart."""
        self._bars = []
        self._actual_prices = []
        self._predicted_prices = []
        self._predicted_prices_low = []
        self._predicted_prices_high = []
        self._levels = {}
        self._clear_all()

    def zoom_in(self) -> None:
        """Zoom into current view."""
        if not HAS_PYQTGRAPH or self.plot_widget is None:
            return
        try:
            vb = self.plot_widget.plotItem.vb
            vb.scaleBy((0.8, 0.8))
            self._manual_zoom = True
        except Exception:
            pass

    def zoom_out(self) -> None:
        """Zoom out from current view."""
        if not HAS_PYQTGRAPH or self.plot_widget is None:
            return
        try:
            vb = self.plot_widget.plotItem.vb
            vb.scaleBy((1.25, 1.25))
            self._manual_zoom = True
        except Exception:
            pass

    def reset_view(self) -> None:
        """Reset view and resume auto-follow."""
        if not HAS_PYQTGRAPH or self.plot_widget is None:
            return
        try:
            self._manual_zoom = False
            
            # FIX: Restore proper view based on current data
            n_candles = len(self._actual_prices)
            n_pred = len(self._predicted_prices) if self._predicted_prices else 0
            
            if n_candles > 0:
                # Calculate Y range with padding
                closes_arr = np.array(self._actual_prices, dtype=float)
                y_min = float(np.min(closes_arr))
                y_max = float(np.max(closes_arr))
                y_range = y_max - y_min
                y_padding = max(y_range * 0.08, y_max * 0.005, 0.1)
                
                # Apply Y range with padding
                self.plot_widget.setYRange(
                    max(0, y_min - y_padding),
                    y_max + y_padding,
                    padding=0.02
                )
                
                # X range: show last 200 candles + 30% of forecast if available
                pred_visible = max(1, int(n_pred * 0.3)) if n_pred > 0 else 0
                visible_candles = min(200, n_candles)
                x_min = max(0, n_candles - visible_candles)
                x_max = (
                    float(n_candles)
                    + float(self._forecast_x_gap)
                    + float(pred_visible)
                )
                
                self.plot_widget.setXRange(
                    x_min, x_max, padding=0.02
                )
            else:
                # No data, just autoRange
                self.plot_widget.autoRange(padding=0.05)
        except Exception:
            pass

    def set_title(self, title: str) -> None:
        """Set chart title."""
        if HAS_PYQTGRAPH and self.plot_widget is not None:
            try:
                self.plot_widget.setTitle(
                    title,
                    color=ModernColors.TEXT_PRIMARY,
                    size="12pt",
                )
            except Exception:
                pass

    def set_overlay_enabled(self, key: str, enabled: bool) -> None:
        """Public overlay toggle for UI controls.
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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(100, 40)
        self._prices: list[float] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_PYQTGRAPH:
            self.plot = pg.PlotWidget()
            self.plot.setBackground(ModernColors.BG_PRIMARY)
            self.plot.hideAxis("left")
            self.plot.hideAxis("bottom")
            self.plot.setMouseEnabled(False, False)

            self.line = self.plot.plot(
                pen=pg.mkPen(color=ModernColors.ACCENT_INFO, width=1)
            )
            layout.addWidget(self.plot)
        else:
            self.plot = None
            self.line = None
            self.label = QLabel("--")
            self.label.setStyleSheet(
                f"color: {ModernColors.TEXT_MUTED};"
            )
            layout.addWidget(self.label)

    def update_data(self, prices: list[float]) -> None:
        """Update mini chart."""
        if prices is None:
            self._prices = []
        else:
            try:
                self._prices = list(prices)
            except Exception:
                self._prices = []

        if not HAS_PYQTGRAPH or not self._prices or self.line is None:
            return

        try:
            x = np.arange(len(self._prices))
            y = np.array(self._prices, dtype=float)

            if len(y) > 1:
                if y[-1] > y[0]:
                    color = ModernColors.ACCENT_SUCCESS
                elif y[-1] < y[0]:
                    color = ModernColors.ACCENT_DANGER
                else:
                    color = ModernColors.TEXT_MUTED
            else:
                color = ModernColors.TEXT_MUTED

            self.line.setPen(pg.mkPen(color=color, width=1))
            self.line.setData(x, y)
            self.plot.autoRange()

        except Exception:
            pass
