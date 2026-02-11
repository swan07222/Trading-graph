# ui/__init__.py

try:
    from .app import MainApp, run_app
except ImportError:
    MainApp = None
    run_app = None

try:
    from .charts import StockChart
except ImportError:
    StockChart = None

try:
    from .auto_learn_dialog import AutoLearnDialog, show_auto_learn_dialog
except ImportError:
    AutoLearnDialog = None
    show_auto_learn_dialog = None

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

try:
    from .news_widget import NewsPanel
except ImportError:
    NewsPanel = None

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
    "NewsPanel",
]