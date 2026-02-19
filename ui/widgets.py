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
    """Large signal display panel with professional styling"""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(220)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        self.signal_label = QLabel("WAITING")
        self.signal_label.setFont(QFont("Segoe UI", 36, QFont.Weight.Bold))
        self.signal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.signal_label)

        self.info_label = QLabel("Enter a stock code to analyze")
        self.info_label.setFont(QFont("Segoe UI", 14))
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

        prob_widget = QWidget()
        prob_layout = QHBoxLayout(prob_widget)
        prob_layout.setContentsMargins(0, 10, 0, 10)

        down_container = QVBoxLayout()
        down_label = QLabel("DOWN")
        down_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        down_label.setStyleSheet(
            "color: #e5534b; font-weight: bold; font-size: 11px;"
        )
        self.prob_down = QProgressBar()
        self.prob_down.setFormat("%p%")
        self.prob_down.setStyleSheet("""
            QProgressBar {
                background: #101f34;
                border: 1px solid #304968;
                border-radius: 6px;
                text-align: center;
                color: #eaf1ff;
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e5534b, stop:1 #cf3e36);
                border-radius: 5px;
            }
        """)
        down_container.addWidget(down_label)
        down_container.addWidget(self.prob_down)
        prob_layout.addLayout(down_container)

        neutral_container = QVBoxLayout()
        neutral_label = QLabel("NEUTRAL")
        neutral_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        neutral_label.setStyleSheet(
            "color: #d8a03a; font-weight: bold; font-size: 11px;"
        )
        self.prob_neutral = QProgressBar()
        self.prob_neutral.setFormat("%p%")
        self.prob_neutral.setStyleSheet("""
            QProgressBar {
                background: #101f34;
                border: 1px solid #304968;
                border-radius: 6px;
                text-align: center;
                color: #eaf1ff;
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #d8a03a, stop:1 #bd8224);
                border-radius: 5px;
            }
        """)
        neutral_container.addWidget(neutral_label)
        neutral_container.addWidget(self.prob_neutral)
        prob_layout.addLayout(neutral_container)

        up_container = QVBoxLayout()
        up_label = QLabel("UP")
        up_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        up_label.setStyleSheet(
            "color: #35b57c; font-weight: bold; font-size: 11px;"
        )
        self.prob_up = QProgressBar()
        self.prob_up.setFormat("%p%")
        self.prob_up.setStyleSheet("""
            QProgressBar {
                background: #101f34;
                border: 1px solid #304968;
                border-radius: 6px;
                text-align: center;
                color: #eaf1ff;
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #35b57c, stop:1 #239760);
                border-radius: 5px;
            }
        """)
        up_container.addWidget(up_label)
        up_container.addWidget(self.prob_up)
        prob_layout.addLayout(up_container)

        layout.addWidget(prob_widget)

        self.action_label = QLabel("")
        self.action_label.setFont(QFont("Segoe UI", 12))
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_label.setWordWrap(True)
        layout.addWidget(self.action_label)

        self.conf_label = QLabel("")
        self.conf_label.setFont(QFont("Segoe UI", 11))
        self.conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.conf_label)

        self._set_default_style()

    def _set_default_style(self):
        self.setStyleSheet("""
            SignalPanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #14243d, stop:1 #0f1b2e);
                border-radius: 12px;
                border: 2px solid #2b4266;
            }
            QLabel { color: #b8c8e3; }
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
        self.setFont(QFont("Consolas", 10))
        self.setMaximumHeight(200)
        self.setStyleSheet("""
            QTextEdit {
                background: #0c1728;
                color: #cde8d7;
                border: 1px solid #253754;
                border-radius: 8px;
                padding: 5px;
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
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #14243d, stop:1 #0f1b2e);
                border-radius: 10px;
                border: 1px solid #2f4466;
                padding: 15px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(5)

        title_text = f"{icon} {title}" if icon else title
        title_label = QLabel(title_text)
        title_label.setStyleSheet("color: #aac3ec; font-size: 12px;")
        layout.addWidget(title_label)

        self.value_label = QLabel(str(value))
        self.value_label.setStyleSheet("""
            color: #79a6ff;
            font-size: 24px;
            font-weight: bold;
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
                font-size: 24px;
                font-weight: bold;
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
                background: #0f1b2e;
                border-radius: 8px;
                border: 1px solid #2f4466;
                padding: 10px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)

        self.connection_label = QLabel("Disconnected")
        self.connection_label.setStyleSheet(
            "color: #e5534b; font-weight: bold;"
        )
        layout.addWidget(self.connection_label)

        layout.addStretch()

        self.market_label = QLabel("Market: --")
        self.market_label.setStyleSheet("color: #aac3ec;")
        layout.addWidget(self.market_label)

        layout.addStretch()

        self.mode_label = QLabel("Mode: Paper Trading")
        self.mode_label.setStyleSheet("color: #d8a03a;")
        layout.addWidget(self.mode_label)

    def set_connected(self, connected: bool, mode: str = "paper"):
        """Update connection status"""
        if self.connection_label is None:
            return

        if connected:
            self.connection_label.setText("Connected")
            self.connection_label.setStyleSheet(
                "color: #35b57c; font-weight: bold;"
            )

            if mode == "live":
                self.mode_label.setText("Mode: LIVE TRADING")
                self.mode_label.setStyleSheet(
                    "color: #e5534b; font-weight: bold;"
                )
            else:
                self.mode_label.setText("Mode: Paper Trading")
                self.mode_label.setStyleSheet("color: #d8a03a;")
        else:
            self.connection_label.setText("Disconnected")
            self.connection_label.setStyleSheet(
                "color: #e5534b; font-weight: bold;"
            )

    def set_market_status(self, is_open: bool):
        """Update market status"""
        if self.market_label is None:
            return

        if is_open:
            self.market_label.setText("Market Open")
            self.market_label.setStyleSheet("color: #35b57c;")
        else:
            self.market_label.setText("Market Closed")
            self.market_label.setStyleSheet("color: #e5534b;")

