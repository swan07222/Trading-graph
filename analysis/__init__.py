"""Analysis Module - Technical, Sentiment, and Backtesting."""

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

__all__ = [
    'TechnicalAnalyzer',
    'SentimentAnalyzer',
    'NewsScraper',
    'Backtester',
    'BacktestResult',
    'MarketReplay',
    'ReplayBar',
]
