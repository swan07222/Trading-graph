# ui/__init__.py

try:
    from .app import MainApp, run_app
except Exception:
    MainApp = None
    run_app = None

try:
    from .charts import StockChart
except Exception:
    StockChart = None

try:
    from .auto_learn_dialog import AutoLearnDialog, show_auto_learn_dialog
except Exception:
    AutoLearnDialog = None
    show_auto_learn_dialog = None

try:
    from .widgets import LogWidget, PositionTable, SignalPanel
except Exception:
    SignalPanel = None
    PositionTable = None
    LogWidget = None

try:
    from .dialogs import BacktestDialog, TrainingDialog
except Exception:
    TrainingDialog = None
    BacktestDialog = None

try:
    from .news_widget import NewsPanel
except Exception:
    NewsPanel = None

try:
    from .strategy_marketplace_dialog import StrategyMarketplaceDialog
except Exception:
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
