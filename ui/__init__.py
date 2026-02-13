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
    from .widgets import LogWidget, PositionTable, SignalPanel
except ImportError:
    SignalPanel = None
    PositionTable = None
    LogWidget = None

try:
    from .dialogs import BacktestDialog, TrainingDialog
except ImportError:
    TrainingDialog = None
    BacktestDialog = None

try:
    from .news_widget import NewsPanel
except ImportError:
    NewsPanel = None

try:
    from .strategy_marketplace_dialog import StrategyMarketplaceDialog
except ImportError:
    StrategyMarketplaceDialog = None

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
    "StrategyMarketplaceDialog",
]
