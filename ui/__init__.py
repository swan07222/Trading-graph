"""
UI Module
Professional trading application interface
"""

from .app import MainApp, run_app
from .widgets import SignalPanel, PositionTable, LogWidget
from .charts import StockChart
from .dialogs import TrainingDialog, BacktestDialog
from .auto_learn_dialog import AutoLearnDialog, show_auto_learn_dialog

__all__ = [
    'MainApp',
    'run_app',
    'SignalPanel',
    'PositionTable', 
    'LogWidget',
    'StockChart',
    'TrainingDialog',
    'BacktestDialog',
    'AutoLearnDialog',
    'show_auto_learn_dialog',
]