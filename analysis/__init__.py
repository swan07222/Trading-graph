"""Analysis Module - Technical, Sentiment, and Backtesting
"""

try:
    from .technical import TechnicalAnalyzer
except Exception:
    TechnicalAnalyzer = None

try:
    from .sentiment import NewsScraper, SentimentAnalyzer
except Exception:
    SentimentAnalyzer = None
    NewsScraper = None

try:
    from .backtest import Backtester, BacktestResult
except Exception:
    Backtester = None
    BacktestResult = None

try:
    from .replay import MarketReplay, ReplayBar
except Exception:
    MarketReplay = None
    ReplayBar = None

try:
    from .strategy_engine import StrategyScriptEngine, StrategySignal
except Exception:
    StrategyScriptEngine = None
    StrategySignal = None

try:
    from .strategy_marketplace import StrategyMarketplace
except Exception:
    StrategyMarketplace = None

__all__ = [
    'TechnicalAnalyzer',
    'SentimentAnalyzer',
    'NewsScraper',
    'Backtester',
    'BacktestResult',
    'MarketReplay',
    'ReplayBar',
    'StrategyScriptEngine',
    'StrategySignal',
    'StrategyMarketplace',
]
