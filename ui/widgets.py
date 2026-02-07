"""
Custom UI Widgets - Professional English Interface
"""
from datetime import datetime
from typing import Dict, List

from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from models.predictor import Signal


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
        
        # Main signal display
        self.signal_label = QLabel("WAITING")
        self.signal_label.setFont(QFont("Segoe UI", 36, QFont.Weight.Bold))
        self.signal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.signal_label)
        
        # Stock info
        self.info_label = QLabel("Enter a stock code to analyze")
        self.info_label.setFont(QFont("Segoe UI", 14))
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)
        
        # Probability bars
        prob_widget = QWidget()
        prob_layout = QHBoxLayout(prob_widget)
        prob_layout.setContentsMargins(0, 10, 0, 10)
        
        # DOWN probability
        down_container = QVBoxLayout()
        down_label = QLabel("DOWN")
        down_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        down_label.setStyleSheet("color: #f85149; font-weight: bold; font-size: 11px;")
        self.prob_down = QProgressBar()
        self.prob_down.setFormat("%p%")
        self.prob_down.setStyleSheet("""
            QProgressBar { 
                background: #21262d; 
                border-radius: 5px; 
                text-align: center; 
                color: white;
                height: 20px;
            }
            QProgressBar::chunk { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f85149, stop:1 #da3633);
                border-radius: 5px; 
            }
        """)
        down_container.addWidget(down_label)
        down_container.addWidget(self.prob_down)
        prob_layout.addLayout(down_container)
        
        # NEUTRAL probability
        neutral_container = QVBoxLayout()
        neutral_label = QLabel("NEUTRAL")
        neutral_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        neutral_label.setStyleSheet("color: #d29922; font-weight: bold; font-size: 11px;")
        self.prob_neutral = QProgressBar()
        self.prob_neutral.setFormat("%p%")
        self.prob_neutral.setStyleSheet("""
            QProgressBar { 
                background: #21262d; 
                border-radius: 5px; 
                text-align: center; 
                color: white;
                height: 20px;
            }
            QProgressBar::chunk { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #d29922, stop:1 #bb8009);
                border-radius: 5px; 
            }
        """)
        neutral_container.addWidget(neutral_label)
        neutral_container.addWidget(self.prob_neutral)
        prob_layout.addLayout(neutral_container)
        
        # UP probability
        up_container = QVBoxLayout()
        up_label = QLabel("UP")
        up_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        up_label.setStyleSheet("color: #3fb950; font-weight: bold; font-size: 11px;")
        self.prob_up = QProgressBar()
        self.prob_up.setFormat("%p%")
        self.prob_up.setStyleSheet("""
            QProgressBar { 
                background: #21262d; 
                border-radius: 5px; 
                text-align: center; 
                color: white;
                height: 20px;
            }
            QProgressBar::chunk { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3fb950, stop:1 #238636);
                border-radius: 5px; 
            }
        """)
        up_container.addWidget(up_label)
        up_container.addWidget(self.prob_up)
        prob_layout.addLayout(up_container)
        
        layout.addWidget(prob_widget)
        
        # Action recommendation
        self.action_label = QLabel("")
        self.action_label.setFont(QFont("Segoe UI", 12))
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_label.setWordWrap(True)
        layout.addWidget(self.action_label)
        
        # Confidence meters
        self.conf_label = QLabel("")
        self.conf_label.setFont(QFont("Segoe UI", 11))
        self.conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.conf_label)
        
        self._set_default_style()
    
    def _set_default_style(self):
        self.setStyleSheet("""
            SignalPanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #21262d, stop:1 #161b22);
                border-radius: 12px;
                border: 2px solid #30363d;
            }
            QLabel { color: #8b949e; }
        """)
    
    def update_prediction(self, pred):
        """Update display with prediction data (robust to missing fields)."""
        sig = getattr(pred, "signal", None)
        sig_text = sig.value if hasattr(sig, "value") else (str(sig) if sig else "HOLD")

        code = str(getattr(pred, "stock_code", "") or "")
        name = str(getattr(pred, "stock_name", "") or "")
        price = float(getattr(pred, "current_price", 0.0) or 0.0)

        self.signal_label.setText(sig_text)
        self.info_label.setText(f"{code} - {name} | ¥{price:.2f}")

        prob_down = float(getattr(pred, "prob_down", 0.33) or 0.33)
        prob_neutral = float(getattr(pred, "prob_neutral", 0.34) or 0.34)
        prob_up = float(getattr(pred, "prob_up", 0.33) or 0.33)
        self.prob_down.setValue(int(prob_down * 100))
        self.prob_neutral.setValue(int(prob_neutral * 100))
        self.prob_up.setValue(int(prob_up * 100))

        # Action text
        pos = getattr(pred, "position", None)
        levels = getattr(pred, "levels", None)
        shares = int(getattr(pos, "shares", 0) or 0)
        entry = float(getattr(levels, "entry", 0.0) or 0.0)
        stop = float(getattr(levels, "stop_loss", 0.0) or 0.0)
        tgt2 = float(getattr(levels, "target_2", 0.0) or 0.0)

        from models.predictor import Signal
        if shares > 0 and sig in (Signal.STRONG_BUY, Signal.BUY):
            self.action_label.setText(
                f"BUY {shares:,} shares @ ¥{entry:.2f}\nStop Loss: ¥{stop:.2f} | Target: ¥{tgt2:.2f}"
            )
        elif shares > 0 and sig in (Signal.STRONG_SELL, Signal.SELL):
            self.action_label.setText(f"SELL {shares:,} shares @ ¥{entry:.2f}")
        else:
            self.action_label.setText("HOLD - Wait for clearer signal")

        confidence = float(getattr(pred, "confidence", 0.0) or 0.0)
        agreement = getattr(pred, "model_agreement", getattr(pred, "agreement", 1.0))
        agreement = float(agreement or 1.0)
        strength = float(getattr(pred, "signal_strength", 0.0) or 0.0)

        self.conf_label.setText(
            f"Confidence: {confidence:.0%} | Model Agreement: {agreement:.0%} | Signal Strength: {strength:.0%}"
        )

        # Styling
        colors = {
            Signal.STRONG_BUY: ("#2ea043", "#0d1117"),
            Signal.BUY: ("#3fb950", "#0d1117"),
            Signal.HOLD: ("#d29922", "#0d1117"),
            Signal.SELL: ("#f85149", "#0d1117"),
            Signal.STRONG_SELL: ("#da3633", "#0d1117"),
        }
        fg, bg = colors.get(sig, ("#c9d1d9", "#21262d"))

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
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.verticalHeader().setVisible(False)
    
    def update_positions(self, positions: Dict):
        """Update table with position data"""
        self.setRowCount(len(positions))
        
        for row, (code, pos) in enumerate(positions.items()):
            # Code
            self.setItem(row, 0, QTableWidgetItem(code))
            
            # Name
            self.setItem(row, 1, QTableWidgetItem(pos.name))
            
            # Shares
            shares_item = QTableWidgetItem(f"{pos.quantity:,}")
            shares_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.setItem(row, 2, shares_item)
            
            # Available
            avail_item = QTableWidgetItem(f"{pos.available_qty:,}")
            avail_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.setItem(row, 3, avail_item)
            
            # Cost
            cost_item = QTableWidgetItem(f"¥{pos.avg_cost:.2f}")
            cost_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.setItem(row, 4, cost_item)
            
            # Current Price
            price_item = QTableWidgetItem(f"¥{pos.current_price:.2f}")
            price_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.setItem(row, 5, price_item)
            
            # P&L
            pnl_item = QTableWidgetItem(f"¥{pos.unrealized_pnl:+,.2f}")
            pnl_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            # P&L %
            pnl_pct_item = QTableWidgetItem(f"{pos.unrealized_pnl_pct:+.2f}%")
            pnl_pct_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            # Color based on profit/loss
            if pos.unrealized_pnl >= 0:
                color = QColor("#3fb950")
            else:
                color = QColor("#f85149")
            
            pnl_item.setForeground(color)
            pnl_pct_item.setForeground(color)
            
            self.setItem(row, 6, pnl_item)
            self.setItem(row, 7, pnl_pct_item)


class LogWidget(QTextEdit):
    """System log display with professional styling"""
    
    MAX_LINES = 500
    
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self.setMaximumHeight(200)
        self.setStyleSheet("""
            QTextEdit {
                background: #0d1117;
                color: #7ee787;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 5px;
            }
        """)
        self._line_count = 0
    
    def log(self, message: str, level: str = "info"):
        """Add log message with color coding"""
        colors = {
            "info": "#7ee787",      # Green
            "warning": "#d29922",    # Yellow/Orange
            "error": "#f85149",      # Red
            "success": "#3fb950",    # Bright Green
            "debug": "#8b949e",      # Gray
        }
        
        icons = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅",
            "debug": "🔧",
        }
        
        color = colors.get(level, "#c9d1d9")
        icon = icons.get(level, "•")
        
        ts = datetime.now().strftime("%H:%M:%S")
        
        self.append(
            f'<span style="color:#484f58">[{ts}]</span> '
            f'<span style="color:{color}">{icon} {message}</span>'
        )
        
        self._line_count += 1
        
        # Trim old lines
        if self._line_count > self.MAX_LINES:
            cursor = self.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(cursor.MoveOperation.Down, cursor.MoveMode.KeepAnchor, 100)
            cursor.removeSelectedText()
            self._line_count -= 100
        
        # Auto-scroll to bottom
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_log(self):
        """Clear all log messages"""
        self.clear()
        self._line_count = 0


class MetricCard(QFrame):
    """Metric display card for dashboard"""
    
    def __init__(self, title: str, value: str = "--", icon: str = ""):
        super().__init__()
        self._setup_ui(title, value, icon)
    
    def _setup_ui(self, title: str, value: str, icon: str):
        self.setStyleSheet("""
            MetricCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #21262d, stop:1 #161b22);
                border-radius: 10px;
                border: 1px solid #30363d;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(5)
        
        # Title with icon
        title_label = QLabel(f"{icon} {title}" if icon else title)
        title_label.setStyleSheet("color: #8b949e; font-size: 12px;")
        layout.addWidget(title_label)
        
        # Value
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("""
            color: #58a6ff; 
            font-size: 24px; 
            font-weight: bold;
        """)
        layout.addWidget(self.value_label)
    
    def set_value(self, value: str, color: str = None):
        """Update the displayed value"""
        self.value_label.setText(value)
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
        self._setup_ui()
    
    def _setup_ui(self):
        self.setStyleSheet("""
            TradingStatusBar {
                background: #161b22;
                border-radius: 8px;
                border: 1px solid #30363d;
                padding: 10px;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Connection status
        self.connection_label = QLabel("● Disconnected")
        self.connection_label.setStyleSheet("color: #f85149; font-weight: bold;")
        layout.addWidget(self.connection_label)
        
        layout.addStretch()
        
        # Market status
        self.market_label = QLabel("Market: --")
        self.market_label.setStyleSheet("color: #8b949e;")
        layout.addWidget(self.market_label)
        
        layout.addStretch()
        
        # Mode indicator
        self.mode_label = QLabel("Mode: Paper Trading")
        self.mode_label.setStyleSheet("color: #d29922;")
        layout.addWidget(self.mode_label)
    
    def set_connected(self, connected: bool, mode: str = "paper"):
        """Update connection status"""
        if connected:
            self.connection_label.setText("● Connected")
            self.connection_label.setStyleSheet("color: #3fb950; font-weight: bold;")
            
            if mode == "live":
                self.mode_label.setText("⚠️ Mode: LIVE TRADING")
                self.mode_label.setStyleSheet("color: #f85149; font-weight: bold;")
            else:
                self.mode_label.setText("Mode: Paper Trading")
                self.mode_label.setStyleSheet("color: #d29922;")
        else:
            self.connection_label.setText("● Disconnected")
            self.connection_label.setStyleSheet("color: #f85149; font-weight: bold;")
    
    def set_market_status(self, is_open: bool):
        """Update market status"""
        if is_open:
            self.market_label.setText("🟢 Market Open")
            self.market_label.setStyleSheet("color: #3fb950;")
        else:
            self.market_label.setText("🔴 Market Closed")
            self.market_label.setStyleSheet("color: #f85149;")