# ui/__init__.py — Replace entire file

"""
UI Module
Professional trading application interface
"""

from .app import MainApp, run_app
from .charts import StockChart
from .auto_learn_dialog import AutoLearnDialog, show_auto_learn_dialog

# Lazy imports to avoid cascade failures when torch/models not installed
try:
    from .widgets import SignalPanel, PositionTable, LogWidget
except ImportError:
    SignalPanel = None
    PositionTable = None
    LogWidget = None

try:
    from .dialogs import TrainingDialog, BacktestDialog
except ImportError:
    TrainingDialog = None
    BacktestDialog = None

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