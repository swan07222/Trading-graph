# ui/widgets.py
from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.modern_theme import (
    ModernColors,
    ModernFonts,
    ModernSpacing,
    get_signal_bg,
    get_signal_color,
    get_signal_panel_style,
    get_progress_bar_style,
)
from utils.logger import get_logger

log = get_logger(__name__)

def _get_signal_enum():
    """Lazy import Signal to avoid circular imports at module load time."""
    try:
        from models.predictor import Signal
        return Signal
    except ImportError:
        return None

class SignalPanel(QFrame):
    """Large signal display panel with modern professional styling"""

    def __init__(self):
        super().__init__()
        self.setObjectName("signalPanelFrame")
        self.setMinimumHeight(240)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(ModernSpacing.LG)
        layout.setContentsMargins(ModernSpacing.XXL, ModernSpacing.XXL, ModernSpacing.XXL, ModernSpacing.XXL)
        
        # Apply modern card style
        self.setStyleSheet(f"""
            QFrame#signalPanelFrame {{
                background-color: {ModernColors.BG_SECONDARY};
                border: 2px solid {ModernColors.BORDER_DEFAULT};
                border-radius: 16px;
            }}
        """)

        self.signal_label = QLabel("WAITING")
        self.signal_label.setObjectName("signalLabel")
        self.signal_label.setFont(QFont(ModernFonts.FAMILY_PRIMARY, ModernFonts.SIZE_HERO, QFont.Weight.Bold))
        self.signal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.signal_label)

        self.info_label = QLabel("Enter a stock code to analyze")
        self.info_label.setFont(QFont(ModernFonts.FAMILY_PRIMARY, ModernFonts.SIZE_BASE))
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY};")
        layout.addWidget(self.info_label)

        prob_widget = QWidget()
        prob_layout = QHBoxLayout(prob_widget)
        prob_layout.setSpacing(ModernSpacing.LG)
        prob_layout.setContentsMargins(0, ModernSpacing.BASE, 0, ModernSpacing.BASE)

        down_container = QVBoxLayout()
        down_container.setSpacing(ModernSpacing.XS)
        down_label = QLabel("▼ DOWN")
        down_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        down_label.setStyleSheet(
            f"color: {ModernColors.ACCENT_DANGER}; font-weight: {ModernFonts.WEIGHT_SEMIBOLD}; font-size: {ModernFonts.SIZE_SM}px;"
        )
        self.prob_down = QProgressBar()
        self.prob_down.setFormat("%p%")
        self.prob_down.setStyleSheet(get_progress_bar_style("danger"))
        down_container.addWidget(down_label)
        down_container.addWidget(self.prob_down)
        prob_layout.addLayout(down_container)

        neutral_container = QVBoxLayout()
        neutral_container.setSpacing(ModernSpacing.XS)
        neutral_label = QLabel("● NEUTRAL")
        neutral_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        neutral_label.setStyleSheet(
            f"color: {ModernColors.ACCENT_WARNING}; font-weight: {ModernFonts.WEIGHT_SEMIBOLD}; font-size: {ModernFonts.SIZE_SM}px;"
        )
        self.prob_neutral = QProgressBar()
        self.prob_neutral.setFormat("%p%")
        self.prob_neutral.setStyleSheet(get_progress_bar_style("warning"))
        neutral_container.addWidget(neutral_label)
        neutral_container.addWidget(self.prob_neutral)
        prob_layout.addLayout(neutral_container)

        up_container = QVBoxLayout()
        up_container.setSpacing(ModernSpacing.XS)
        up_label = QLabel("▲ UP")
        up_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        up_label.setStyleSheet(
            f"color: {ModernColors.ACCENT_SUCCESS}; font-weight: {ModernFonts.WEIGHT_SEMIBOLD}; font-size: {ModernFonts.SIZE_SM}px;"
        )
        self.prob_up = QProgressBar()
        self.prob_up.setFormat("%p%")
        self.prob_up.setStyleSheet(get_progress_bar_style("success"))
        up_container.addWidget(up_label)
        up_container.addWidget(self.prob_up)
        prob_layout.addLayout(up_container)

        layout.addWidget(prob_widget)

        self.action_label = QLabel("")
        self.action_label.setFont(QFont("Segoe UI", 12))
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_label.setWordWrap(True)
        self.action_label.setStyleSheet("color: #9ca3af; padding: 8px 0;")
        layout.addWidget(self.action_label)

        self.conf_label = QLabel("")
        self.conf_label.setFont(QFont("Segoe UI", 11))
        self.conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.conf_label.setStyleSheet("color: #6b7280;")
        layout.addWidget(self.conf_label)

        self._set_default_style()

    def _set_default_style(self):
        self.setStyleSheet("""
            SignalPanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e293b, stop:1 #0f172a);
                border-radius: 16px;
                border: 1px solid #334155;
            }
            QLabel { color: #e6e9f0; }
        """)

    @staticmethod
    def _safe_float(obj, attr, default=0.0):
        """Safely extract float attribute with fallback."""
        try:
            val = getattr(obj, attr, None)
            if val is None:
                return float(default)
            return float(val)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_int(obj, attr, default=0):
        """Safely extract int attribute with fallback."""
        try:
            val = getattr(obj, attr, None)
            if val is None:
                return int(default)
            return int(val)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _safe_str(obj, attr, default=""):
        """Safely extract string attribute with fallback."""
        try:
            val = getattr(obj, attr, None)
            if val is None:
                return str(default)
            return str(val)
        except Exception:
            return str(default)

    def update_prediction(self, pred):
        """Update display with prediction data (robust to missing fields)."""
        Signal = _get_signal_enum()

        sig = getattr(pred, "signal", None)
        sig_text = (
            sig.value if hasattr(sig, "value")
            else (str(sig) if sig else "HOLD")
        )

        code = self._safe_str(pred, "stock_code")
        name = self._safe_str(pred, "stock_name")
        price = self._safe_float(pred, "current_price")

        self.signal_label.setText(sig_text)
        self.info_label.setText(f"{code} - {name} | CNY {price:.2f}")

        prob_down = self._safe_float(pred, "prob_down", 0.33)
        prob_neutral = self._safe_float(pred, "prob_neutral", 0.34)
        prob_up = self._safe_float(pred, "prob_up", 0.33)

        self.prob_down.setValue(int(prob_down * 100))
        self.prob_neutral.setValue(int(prob_neutral * 100))
        self.prob_up.setValue(int(prob_up * 100))

        pos = getattr(pred, "position", None)
        levels = getattr(pred, "levels", None)
        shares = self._safe_int(pos, "shares") if pos else 0
        entry = self._safe_float(levels, "entry") if levels else 0.0
        stop = self._safe_float(levels, "stop_loss") if levels else 0.0
        tgt2 = self._safe_float(levels, "target_2") if levels else 0.0

        if Signal is not None and shares > 0:
            if sig in (Signal.STRONG_BUY, Signal.BUY):
                self.action_label.setText(
                    f"BUY {shares:,} shares @ CNY {entry:.2f}\n"
                    f"Stop Loss: CNY {stop:.2f} | Target: CNY {tgt2:.2f}"
                )
            elif sig in (Signal.STRONG_SELL, Signal.SELL):
                self.action_label.setText(
                    f"SELL {shares:,} shares @ CNY {entry:.2f}"
                )
            else:
                self.action_label.setText(
                    "HOLD - Wait for clearer signal"
                )
        else:
            self.action_label.setText(
                "HOLD - Wait for clearer signal"
            )

        confidence = self._safe_float(pred, "confidence")
        agreement = self._safe_float(
            pred, "model_agreement",
            self._safe_float(pred, "agreement", 1.0)
        )
        strength = self._safe_float(pred, "signal_strength")
        uncertainty = self._safe_float(pred, "uncertainty_score", 0.5)
        tail_risk = self._safe_float(pred, "tail_risk_score", 0.5)

        self.conf_label.setText(
            f"Confidence: {confidence:.0%} | "
            f"Model Agreement: {agreement:.0%} | "
            f"Signal Strength: {strength:.0%} | "
            f"Uncertainty: {uncertainty:.2f} | "
            f"Tail Risk: {tail_risk:.2f}"
        )

        if Signal is not None:
            colors = {
                Signal.STRONG_BUY: ("#20a56a", "#0c1728"),
                Signal.BUY: ("#35b57c", "#0c1728"),
                Signal.HOLD: ("#d8a03a", "#0c1728"),
                Signal.SELL: ("#e5534b", "#0c1728"),
                Signal.STRONG_SELL: ("#cf3e36", "#0c1728"),
            }
            fg, bg = colors.get(sig, ("#dbe4f3", "#13223a"))
        else:
            fg, bg = "#dbe4f3", "#13223a"

        self.setStyleSheet(f"""
            SignalPanel {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {fg}22, stop:1 {bg});
                border-radius: 12px;
                border: 3px solid {fg};
            }}
            QLabel {{ color: {fg}; }}
        """)

    def reset(self):
        """Reset to default state"""
        self.signal_label.setText("WAITING")
        self.info_label.setText("Enter a stock code to analyze")
        self.action_label.setText("")
        self.conf_label.setText("")
        self.prob_down.setValue(0)
        self.prob_neutral.setValue(0)
        self.prob_up.setValue(0)
        self._set_default_style()

class PositionTable(QTableWidget):
    """Position display table with professional styling"""

    def __init__(self):
        super().__init__()
        self.setColumnCount(8)
        self.setHorizontalHeaderLabels([
            'Code', 'Name', 'Shares', 'Available',
            'Cost', 'Price', 'P&L', 'P&L %'
        ])
        self.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.verticalHeader().setVisible(False)
        self.setStyleSheet("""
            QTableWidget {
                background: #111827;
                color: #e6e9f0;
                border: 1px solid #1f2937;
                border-radius: 10px;
                gridline-color: #1f2937;
                selection-background-color: #1e3a5f;
                selection-color: #ffffff;
                alternate-background-color: #0f1724;
                outline: none;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 10px;
                border: none;
            }
            QTableWidget::item:hover {
                background: #1f2937;
            }
            QHeaderView::section {
                background: #1f2937;
                color: #93c5fd;
                padding: 12px 10px;
                border: none;
                border-right: 1px solid #1f2937;
                border-bottom: 1px solid #1f2937;
                font-weight: 600;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
        """)

    @staticmethod
    def _safe_attr(obj, attr, default=0):
        """Safely get attribute with type coercion."""
        try:
            val = getattr(obj, attr, None)
            if val is None:
                return default
            if isinstance(default, float):
                return float(val)
            if isinstance(default, int):
                return int(val)
            return val
        except (TypeError, ValueError):
            return default

    def update_positions(self, positions: dict):
        """Update table with position data - handles missing attributes."""
        if positions is None:
            positions = {}

        self.setRowCount(len(positions))

        for row, (code, pos) in enumerate(positions.items()):
            self.setItem(row, 0, QTableWidgetItem(str(code)))

            name = self._safe_attr(pos, 'name', '')
            self.setItem(row, 1, QTableWidgetItem(str(name)))

            quantity = self._safe_attr(pos, 'quantity', 0)
            shares_item = QTableWidgetItem(f"{quantity:,}")
            shares_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
            )
            self.setItem(row, 2, shares_item)

            available = self._safe_attr(pos, 'available_qty', 0)
            avail_item = QTableWidgetItem(f"{available:,}")
            avail_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
            )
            self.setItem(row, 3, avail_item)

            avg_cost = self._safe_attr(pos, 'avg_cost', 0.0)
            cost_item = QTableWidgetItem(f"CNY {avg_cost:.2f}")
            cost_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
            )
            self.setItem(row, 4, cost_item)

            current_price = self._safe_attr(pos, 'current_price', 0.0)
            price_item = QTableWidgetItem(f"CNY {current_price:.2f}")
            price_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
            )
            self.setItem(row, 5, price_item)

            # P&L
            unrealized_pnl = self._safe_attr(pos, 'unrealized_pnl', 0.0)
            pnl_item = QTableWidgetItem(f"CNY {unrealized_pnl:+,.2f}")
            pnl_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
            )

            # P&L %
            unrealized_pnl_pct = self._safe_attr(
                pos, 'unrealized_pnl_pct', 0.0
            )
            pnl_pct_item = QTableWidgetItem(
                f"{unrealized_pnl_pct:+.2f}%"
            )
            pnl_pct_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
            )

            # Color based on profit/loss
            if unrealized_pnl >= 0:
                color = QColor("#35b57c")
            else:
                color = QColor("#e5534b")

            pnl_item.setForeground(color)
            pnl_pct_item.setForeground(color)

            self.setItem(row, 6, pnl_item)
            self.setItem(row, 7, pnl_pct_item)

class LogWidget(QTextEdit):
    """System log display with professional styling and bounded history."""

    MAX_LINES = 500
    _TRIM_BATCH = 100  # Lines to remove when trimming

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 11))
        self.setMaximumHeight(200)
        self.setStyleSheet("""
            QTextEdit {
                background: #0f172a;
                color: #a7f3d0;
                border: 1px solid #1e293b;
                border-radius: 10px;
                padding: 10px;
                selection-background-color: #3b82f6;
                selection-color: #ffffff;
                font-family: 'Consolas', 'Cascadia Code', 'JetBrains Mono', monospace;
            }
        """)
        self._line_count = 0

    def log(self, message: str, level: str = "info"):
        """Add log message with color coding"""
        colors = {
            "info": "#95e8bf",
            "warning": "#d8a03a",
            "error": "#e5534b",
            "success": "#35b57c",
            "debug": "#9ab8ea",
        }

        icons = {
            "info": "[I]",
            "warning": "[!]",
            "error": "[X]",
            "success": "[OK]",
            "debug": "[D]",
        }

        color = colors.get(level, "#dbe4f3")
        icon = icons.get(level, "[.]")

        ts = datetime.now().strftime("%H:%M:%S")

        self.append(
            f'<span style="color:#7b8ca7">[{ts}]</span> '
            f'<span style="color:{color}">{icon} {message}</span>'
        )

        self._line_count += 1

        if self._line_count > self.MAX_LINES:
            self._trim_old_lines()

        # Auto-scroll to bottom
        scrollbar = self.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())

    def _trim_old_lines(self):
        """
        Remove oldest lines to keep log bounded.
        Uses document-level block removal instead of fragile cursor
        manipulation.
        """
        try:
            doc = self.document()
            if doc is None:
                return

            block_count = doc.blockCount()
            if block_count <= self.MAX_LINES:
                return

            remove_count = min(self._TRIM_BATCH, block_count - self.MAX_LINES)

            cursor = self.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)

            for _ in range(remove_count):
                cursor.movePosition(
                    cursor.MoveOperation.Down,
                    cursor.MoveMode.KeepAnchor
                )

            cursor.movePosition(
                cursor.MoveOperation.StartOfLine,
                cursor.MoveMode.KeepAnchor
            )

            cursor.removeSelectedText()
            if cursor.atStart():
                cursor.deleteChar()

            self._line_count = doc.blockCount()

        except Exception as e:
            # If trimming fails, just clear and reset
            try:
                log.debug(f"Log trim failed, clearing: {e}")
                self.clear()
                self._line_count = 0
            except Exception:
                pass

    def clear_log(self):
        """Clear all log messages"""
        self.clear()
        self._line_count = 0

class MetricCard(QFrame):
    """Metric display card for dashboard"""

    def __init__(self, title: str, value: str = "--", icon: str = ""):
        super().__init__()
        self.value_label = None
        self._setup_ui(title, value, icon)

    def _setup_ui(self, title: str, value: str, icon: str):
        self.setStyleSheet("""
            MetricCard {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e293b, stop:1 #0f172a);
                border-radius: 12px;
                border: 1px solid #334155;
                padding: 18px;
            }
            MetricCard:hover {
                border-color: #475569;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

        title_text = f"{icon} {title}" if icon else title
        title_label = QLabel(title_text)
        title_label.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;")
        layout.addWidget(title_label)

        self.value_label = QLabel(str(value))
        self.value_label.setStyleSheet("""
            color: #60a5fa;
            font-size: 26px;
            font-weight: 700;
        """)
        layout.addWidget(self.value_label)

    def set_value(self, value: str, color: str = None):
        """Update the displayed value"""
        if self.value_label is None:
            return

        self.value_label.setText(str(value))
        if color:
            self.value_label.setStyleSheet(f"""
                color: {color};
                font-size: 26px;
                font-weight: 700;
            """)

class TradingStatusBar(QFrame):
    """Trading status bar showing connection and market status"""

    def __init__(self):
        super().__init__()
        self.connection_label = None
        self.market_label = None
        self.mode_label = None
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("""
            TradingStatusBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e293b, stop:1 #0f172a);
                border-radius: 12px;
                border: 1px solid #334155;
                padding: 14px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(20)

        self.connection_label = QLabel("● Disconnected")
        self.connection_label.setStyleSheet(
            "color: #f87171; font-weight: 700; font-size: 12px;"
        )
        layout.addWidget(self.connection_label)

        layout.addStretch()

        self.market_label = QLabel("Market: --")
        self.market_label.setStyleSheet("color: #94a3b8; font-size: 12px;")
        layout.addWidget(self.market_label)

        layout.addStretch()

        self.mode_label = QLabel("Mode: Paper Trading")
        self.mode_label.setStyleSheet("color: #fbbf24; font-size: 12px; font-weight: 600;")
        layout.addWidget(self.mode_label)

    def set_connected(self, connected: bool, mode: str = "paper"):
        """Update connection status"""
        if self.connection_label is None:
            return

        if connected:
            self.connection_label.setText("● Connected")
            self.connection_label.setStyleSheet(
                "color: #34d399; font-weight: 700; font-size: 12px;"
            )

            if mode == "live":
                self.mode_label.setText("Mode: ● LIVE TRADING")
                self.mode_label.setStyleSheet(
                    "color: #f87171; font-size: 12px; font-weight: 700;"
                )
            else:
                self.mode_label.setText("Mode: ● Paper Trading")
                self.mode_label.setStyleSheet("color: #fbbf24; font-size: 12px; font-weight: 600;")
        else:
            self.connection_label.setText("● Disconnected")
            self.connection_label.setStyleSheet(
                "color: #f87171; font-weight: 700; font-size: 12px;"
            )

    def set_market_status(self, is_open: bool):
        """Update market status"""
        if self.market_label is None:
            return

        if is_open:
            self.market_label.setText("Market: ● Open")
            self.market_label.setStyleSheet("color: #34d399; font-size: 12px; font-weight: 600;")
        else:
            self.market_label.setText("Market: ○ Closed")
            self.market_label.setStyleSheet("color: #f87171; font-size: 12px; font-weight: 600;")

