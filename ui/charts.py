# ui/charts.py
"""
Chart Widgets - Professional stock charts with prediction overlay
"""
import numpy as np
from typing import List, Dict, Optional

try:
    import pyqtgraph as pg
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    pg = None

from utils.logger import get_logger

log = get_logger(__name__)


if HAS_PYQTGRAPH:
    class StockChart(pg.PlotWidget):
        """
        Professional stock chart with:
        - Price line
        - Moving averages
        - Prediction overlay
        - Trading levels
        """
        
        def __init__(self):
            super().__init__()

            self.setBackground('#0a0a1a')
            self.showGrid(x=True, y=True, alpha=0.3)
            self.setLabel('left', 'Price', color='#888')
            self.setLabel('bottom', 'Time', color='#888')

            self.addLegend(offset=(10, 10))

            # Persistent plot items (no clear/replot flicker)
            self._actual_item = self.plot([], [], pen=pg.mkPen("#00E5FF", width=2), name="Actual")
            self._forecast_item = self.plot([], [], pen=pg.mkPen("#4CAF50", width=2, style=Qt.PenStyle.DashLine), name="Forecast")

            self._ma5_item = self.plot([], [], pen=pg.mkPen("#FF9800", width=1), name="MA5")
            self._ma20_item = self.plot([], [], pen=pg.mkPen("#9C27B0", width=1), name="MA20")

            self._level_lines: List = []
        
        def update_data(self, prices: List[float], predictions: List[float] = None, levels: Dict[str, float] = None):
            """Fast update: update existing plot items instead of clear()."""
            # Remove old level lines
            for ln in self._level_lines:
                try:
                    self.removeItem(ln)
                except Exception:
                    pass
            self._level_lines = []

            if not prices:
                self._actual_item.setData([], [])
                self._forecast_item.setData([], [])
                self._ma5_item.setData([], [])
                self._ma20_item.setData([], [])
                return

            x = np.arange(len(prices), dtype=float)
            self._actual_item.setData(x, np.array(prices, dtype=float))

            # Moving averages
            if len(prices) >= 20:
                p = np.array(prices, dtype=float)
                ma5 = np.convolve(p, np.ones(5)/5, mode="valid")
                ma20 = np.convolve(p, np.ones(20)/20, mode="valid")
                self._ma5_item.setData(np.arange(4, len(prices), dtype=float), ma5)
                self._ma20_item.setData(np.arange(19, len(prices), dtype=float), ma20)
            else:
                self._ma5_item.setData([], [])
                self._ma20_item.setData([], [])

            # Forecast overlay: predictions include current point at index 0
            if predictions and len(predictions) > 1:
                pred_x = np.arange(len(prices) - 1, len(prices) - 1 + len(predictions), dtype=float)
                self._forecast_item.setData(pred_x, np.array(predictions, dtype=float))
            else:
                self._forecast_item.setData([], [])

            # Levels
            if levels:
                if levels.get("stop_loss"):
                    line = self.addLine(
                        y=float(levels["stop_loss"]),
                        pen=pg.mkPen("#FF5252", width=1, style=Qt.PenStyle.DashLine)
                    )
                    self._level_lines.append(line)
                
                for key in ("target_1", "target_2", "target_3"):
                    if levels.get(key):
                        line = self.addLine(
                            y=float(levels[key]),
                            pen=pg.mkPen("#4CAF50", width=1, style=Qt.PenStyle.DashLine)
                        )
                        self._level_lines.append(line)
        
        def clear_chart(self):
            """Clear all chart data"""
            self._actual_item.setData([], [])
            self._forecast_item.setData([], [])
            self._ma5_item.setData([], [])
            self._ma20_item.setData([], [])
            
            for ln in self._level_lines:
                try:
                    self.removeItem(ln)
                except Exception:
                    pass
            self._level_lines = []

else:
    # Fallback if pyqtgraph not available
    from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
    
    class StockChart(QWidget):
        """Fallback chart when pyqtgraph is not available"""
        
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout(self)
            self.label = QLabel("Chart requires pyqtgraph. Install with: pip install pyqtgraph")
            self.label.setStyleSheet("color: #888; padding: 20px;")
            layout.addWidget(self.label)
        
        def update_data(self, prices, predictions=None, levels=None):
            pass
        
        def clear_chart(self):
            pass