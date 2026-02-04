"""
Custom Widgets
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
    """Large signal display panel"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(220)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Signal
        self.signal_label = QLabel("Waiting")
        self.signal_label.setFont(QFont("Arial", 36, QFont.Weight.Bold))
        self.signal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.signal_label)
        
        # Stock info
        self.info_label = QLabel("Enter stock code to analyze")
        self.info_label.setFont(QFont("Arial", 14))
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)
        
        # Probabilities
        prob_widget = QWidget()
        prob_layout = QHBoxLayout(prob_widget)
        prob_layout.setContentsMargins(0, 0, 0, 0)
        
        self.prob_down = QProgressBar()
        self.prob_down.setFormat("DOWN %p%")
        self.prob_down.setStyleSheet("""
            QProgressBar { background: #1a1a3e; border-radius: 5px; text-align: center; color: white; }
            QProgressBar::chunk { background: #FF5252; border-radius: 5px; }
        """)
        
        self.prob_neutral = QProgressBar()
        self.prob_neutral.setFormat("NEUTRAL %p%")
        self.prob_neutral.setStyleSheet("""
            QProgressBar { background: #1a1a3e; border-radius: 5px; text-align: center; color: white; }
            QProgressBar::chunk { background: #FFD54F; border-radius: 5px; }
        """)
        
        self.prob_up = QProgressBar()
        self.prob_up.setFormat("UP %p%")
        self.prob_up.setStyleSheet("""
            QProgressBar { background: #1a1a3e; border-radius: 5px; text-align: center; color: white; }
            QProgressBar::chunk { background: #4CAF50; border-radius: 5px; }
        """)
        
        prob_layout.addWidget(self.prob_down)
        prob_layout.addWidget(self.prob_neutral)
        prob_layout.addWidget(self.prob_up)
        layout.addWidget(prob_widget)
        
        # Action
        self.action_label = QLabel("")
        self.action_label.setFont(QFont("Arial", 12))
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_label.setWordWrap(True)
        layout.addWidget(self.action_label)
        
        # Confidence
        self.conf_label = QLabel("")
        self.conf_label.setFont(QFont("Arial", 11))
        self.conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.conf_label)
        
        self._set_default_style()
    
    def _set_default_style(self):
        self.setStyleSheet("""
            SignalPanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a5a, stop:1 #1a1a3e);
                border-radius: 15px;
                border: 3px solid #3a3a7a;
            }
            QLabel { color: #888; }
        """)
    
    def update_prediction(self, pred):
        """Update with prediction data"""
        self.signal_label.setText(pred.signal.value)
        self.info_label.setText(
            f"{pred.stock_code} - {pred.stock_name} | ¥{pred.current_price:.2f}"
        )
        
        self.prob_down.setValue(int(pred.prob_down * 100))
        self.prob_neutral.setValue(int(pred.prob_neutral * 100))
        self.prob_up.setValue(int(pred.prob_up * 100))
        
        if pred.position.shares > 0:
            if pred.signal in [Signal.STRONG_BUY, Signal.BUY]:
                self.action_label.setText(
                    f"BUY {pred.position.shares:,} shares @ ¥{pred.levels.entry:.2f}\n"
                    f"Stop: ¥{pred.levels.stop_loss:.2f} | Target: ¥{pred.levels.target_2:.2f}"
                )
            else:
                self.action_label.setText(f"SELL {pred.position.shares:,} shares")
        else:
            self.action_label.setText("HOLD - Wait for clearer signal")
        
        self.conf_label.setText(
            f"Confidence: {pred.confidence:.0%} | "
            f"Agreement: {pred.model_agreement:.0%} | "
            f"Strength: {pred.signal_strength:.0%}"
        )
        
        # Style by signal
        colors = {
            Signal.STRONG_BUY: ("#00E676", "#003d1a"),
            Signal.BUY: ("#69F0AE", "#1b4e20"),
            Signal.HOLD: ("#FFD54F", "#4d3800"),
            Signal.SELL: ("#FF8A80", "#4d1a1a"),
            Signal.STRONG_SELL: ("#FF1744", "#4d0000"),
        }
        
        fg, bg = colors.get(pred.signal, ("#fff", "#2a2a5a"))
        
        self.setStyleSheet(f"""
            SignalPanel {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {fg}33, stop:1 {bg});
                border-radius: 15px;
                border: 4px solid {fg};
            }}
            QLabel {{ color: {fg}; }}
        """)
    
    def reset(self):
        """Reset to default state"""
        self.signal_label.setText("Waiting")
        self.info_label.setText("Enter stock code to analyze")
        self.action_label.setText("")
        self.conf_label.setText("")
        self.prob_down.setValue(0)
        self.prob_neutral.setValue(0)
        self.prob_up.setValue(0)
        self._set_default_style()


class PositionTable(QTableWidget):
    """Position display table"""
    
    def __init__(self):
        super().__init__()
        self.setColumnCount(8)
        self.setHorizontalHeaderLabels([
            'Code', 'Name', 'Qty', 'Available', 'Cost', 'Price', 'P&L', 'P&L%'
        ])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setAlternatingRowColors(True)
    
    def update_positions(self, positions: Dict):
        """Update table with positions"""
        self.setRowCount(len(positions))
        
        for row, (code, pos) in enumerate(positions.items()):
            self.setItem(row, 0, QTableWidgetItem(code))
            self.setItem(row, 1, QTableWidgetItem(pos.stock_name))
            self.setItem(row, 2, QTableWidgetItem(str(pos.quantity)))
            self.setItem(row, 3, QTableWidgetItem(str(pos.available_qty)))
            self.setItem(row, 4, QTableWidgetItem(f"¥{pos.avg_cost:.2f}"))
            self.setItem(row, 5, QTableWidgetItem(f"¥{pos.current_price:.2f}"))
            
            pnl = QTableWidgetItem(f"¥{pos.unrealized_pnl:.2f}")
            pnl_pct = QTableWidgetItem(f"{pos.unrealized_pnl_pct:.2f}%")
            
            color = QColor("#4CAF50") if pos.unrealized_pnl >= 0 else QColor("#FF5252")
            pnl.setForeground(color)
            pnl_pct.setForeground(color)
            
            self.setItem(row, 6, pnl)
            self.setItem(row, 7, pnl_pct)


class LogWidget(QTextEdit):
    """Log display widget"""
    
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self.setMaximumHeight(200)
        self.setStyleSheet("""
            QTextEdit {
                background: #0a0a1a;
                color: #0f0;
                border: 1px solid #2a2a5a;
                border-radius: 5px;
            }
        """)
    
    def log(self, message: str, level: str = "info"):
        """Add log message"""
        colors = {
            "info": "#0f0",
            "warning": "#FFD54F",
            "error": "#FF5252",
            "success": "#4CAF50",
        }
        color = colors.get(level, "#0f0")
        
        ts = datetime.now().strftime("%H:%M:%S")
        self.append(
            f'<span style="color:#666">[{ts}]</span> '
            f'<span style="color:{color}">{message}</span>'
        )
        
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
    
    def clear_log(self):
        self.clear()