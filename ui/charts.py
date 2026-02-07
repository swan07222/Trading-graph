# ui/charts.py
"""
Stock Chart Widget with AI Predictions
"""
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
    log.warning("pyqtgraph not available - using fallback chart")


class StockChart(QWidget):
    """
    Interactive stock chart with AI prediction overlay.
    
    Features:
    - Historical price display
    - AI-predicted future prices
    - Trading levels (stop loss, targets)
    - Real-time updates
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
        # Data
        self._actual_prices: List[float] = []
        self._predicted_prices: List[float] = []
        self._levels: Dict[str, float] = {}
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        if HAS_PYQTGRAPH:
            self._setup_pyqtgraph()
        else:
            self._setup_fallback()
    
    def _setup_pyqtgraph(self):
        """Setup pyqtgraph chart"""
        # Configure pyqtgraph
        pg.setConfigOptions(antialias=True, background='#0d1117', foreground='#c9d1d9')
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Price', units='¥')
        self.plot_widget.setLabel('bottom', 'Time', units='bars')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Style
        self.plot_widget.setBackground('#0d1117')
        
        # Create plot items
        self.actual_line = self.plot_widget.plot(
            pen=pg.mkPen(color='#58a6ff', width=2),
            name='Actual'
        )
        
        self.predicted_line = self.plot_widget.plot(
            pen=pg.mkPen(color='#3fb950', width=2, style=Qt.PenStyle.DashLine),
            name='Predicted'
        )
        
        # Level lines (will be updated)
        self.level_lines = {}
        
        # Add legend
        self.plot_widget.addLegend()
        
        self.layout().addWidget(self.plot_widget)
    
    def _setup_fallback(self):
        """Setup fallback when pyqtgraph not available"""
        self.fallback_label = QLabel("Chart requires pyqtgraph\n\nInstall with: pip install pyqtgraph")
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
    
    def update_data(
        self,
        actual_prices: List[float],
        predicted_prices: List[float] = None,
        levels: Dict[str, float] = None
    ):
        """
        Update chart with new data.
        
        Args:
            actual_prices: Historical prices
            predicted_prices: AI-predicted future prices
            levels: Trading levels (stop_loss, target_1, etc.)
        """
        self._actual_prices = list(actual_prices) if actual_prices else []
        self._predicted_prices = list(predicted_prices) if predicted_prices else []
        self._levels = levels or {}
        
        if not HAS_PYQTGRAPH:
            return
        
        try:
            self._update_plot()
        except Exception as e:
            log.warning(f"Chart update failed: {e}")
    
    def _update_plot(self):
        """Update the plot with current data"""
        if not self._actual_prices:
            return
        
        # X axis for actual data
        x_actual = np.arange(len(self._actual_prices))
        y_actual = np.array(self._actual_prices)
        
        # Update actual line
        self.actual_line.setData(x_actual, y_actual)
        
        # Update predicted line
        if self._predicted_prices:
            # Predicted starts from last actual point
            start_x = len(self._actual_prices) - 1
            x_pred = np.arange(start_x, start_x + len(self._predicted_prices) + 1)
            
            # Include last actual price as first point for continuity
            y_pred = np.array([self._actual_prices[-1]] + self._predicted_prices)
            
            self.predicted_line.setData(x_pred, y_pred)
        else:
            self.predicted_line.clear()
        
        # Update level lines
        self._update_level_lines()
        
        # Auto-range
        self.plot_widget.autoRange()
    
    def _update_level_lines(self):
        """Update horizontal lines for trading levels"""
        # Remove old lines
        for line in self.level_lines.values():
            self.plot_widget.removeItem(line)
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
                line = pg.InfiniteLine(
                    pos=price,
                    angle=0,
                    pen=pg.mkPen(color=color, width=1, style=Qt.PenStyle.DotLine),
                    label=f'{name}: ¥{price:.2f}',
                    labelOpts={'color': color, 'position': 0.95}
                )
                self.plot_widget.addItem(line)
                self.level_lines[name] = line
    
    def clear(self):
        """Clear all data from chart"""
        self._actual_prices = []
        self._predicted_prices = []
        self._levels = {}
        
        if HAS_PYQTGRAPH:
            self.actual_line.clear()
            self.predicted_line.clear()
            for line in self.level_lines.values():
                self.plot_widget.removeItem(line)
            self.level_lines.clear()
    
    def set_title(self, title: str):
        """Set chart title"""
        if HAS_PYQTGRAPH:
            self.plot_widget.setTitle(title, color='#c9d1d9', size='12pt')


class MiniChart(QWidget):
    """
    Compact mini chart for watchlist items.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 40)
        self._setup_ui()
        self._prices: List[float] = []
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        if HAS_PYQTGRAPH:
            self.plot = pg.PlotWidget()
            self.plot.setBackground('#0d1117')
            self.plot.hideAxis('left')
            self.plot.hideAxis('bottom')
            self.plot.setMouseEnabled(False, False)
            
            self.line = self.plot.plot(pen=pg.mkPen(color='#58a6ff', width=1))
            layout.addWidget(self.plot)
        else:
            self.label = QLabel("--")
            self.label.setStyleSheet("color: #888;")
            layout.addWidget(self.label)
    
    def update_data(self, prices: List[float]):
        """Update mini chart"""
        self._prices = list(prices) if prices else []
        
        if not HAS_PYQTGRAPH or not self._prices:
            return
        
        try:
            x = np.arange(len(self._prices))
            y = np.array(self._prices)
            
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