# ui/__init__.py

"""
UI Module
Professional trading application interface
"""

try:
    from .app import MainApp, run_app
except ImportError:
    MainApp = None
    run_app = None

from .charts import StockChart
from .auto_learn_dialog import AutoLearnDialog, show_auto_learn_dialog

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
    "MainApp",
    "run_app",
    "SignalPanel",
    "PositionTable",
    "LogWidget",
    "StockChart",
    "TrainingDialog",
    "BacktestDialog",
    "AutoLearnDialog",
    "show_auto_learn_dialog",
]