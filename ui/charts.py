"""
Chart Widgets
"""
import numpy as np
from typing import List, Dict, Optional

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor


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
        
        # Plot items
        self.price_line = None
        self.ma_lines = {}
        self.prediction_line = None
        self.level_lines = []
        
        # Legend
        self.addLegend(offset=(10, 10))
    
    def update_data(self, prices: List[float], predictions: List[float] = None, levels: Dict[str, float] = None):
        """Update chart with actual + forecast overlay (forecast anchored to last actual)."""
        self.clear()
        self.level_lines = []

        if not prices:
            return

        x = np.arange(len(prices), dtype=float)

        self.plot(x, prices, pen=pg.mkPen("#00E5FF", width=2), name="Actual")

        if len(prices) >= 20:
            ma5 = np.convolve(prices, np.ones(5)/5, mode="valid")
            ma20 = np.convolve(prices, np.ones(20)/20, mode="valid")

            self.plot(np.arange(4, len(prices)), ma5, pen=pg.mkPen("#FF9800", width=1), name="MA5")
            self.plot(np.arange(19, len(prices)), ma20, pen=pg.mkPen("#9C27B0", width=1), name="MA20")

        # predictions include current point at index 0
        if predictions and len(predictions) > 1:
            pred_x = np.arange(len(prices) - 1, len(prices) - 1 + len(predictions), dtype=float)
            self.plot(pred_x, predictions, pen=pg.mkPen("#4CAF50", width=2, style=Qt.PenStyle.DashLine), name="Forecast")

        if levels:
            if levels.get("stop_loss"):
                self.level_lines.append(self.addLine(
                    y=float(levels["stop_loss"]),
                    pen=pg.mkPen("#FF5252", width=1, style=Qt.PenStyle.DashLine)
                ))

            for key in ("target_1", "target_2", "target_3"):
                if levels.get(key):
                    self.level_lines.append(self.addLine(
                        y=float(levels[key]),
                        pen=pg.mkPen("#4CAF50", width=1, style=Qt.PenStyle.DashLine)
                    ))
    
    def clear_chart(self):
        """Clear all chart data"""
        self.clear()
        self.level_lines = []