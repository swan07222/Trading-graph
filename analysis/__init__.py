"""Analysis Module - Technical, Sentiment, and Backtesting.

Enhanced with:
- Advanced chart types (Heikin-Ashi, Renko, Kagi, etc.)
- 400+ technical indicators
- Professional drawing tools with auto-detection
- Advanced backtesting with China-specific costs
- Walk-forward optimization and Monte Carlo simulation
"""

# Original imports
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

# Enhanced features
try:
    from .advanced_charts import (
        AdvancedChartEngine,
        ChartType,
        HeikinAshiCalculator,
        RenkoCalculator,
        KagiCalculator,
        PointFigureCalculator,
        LineBreakCalculator,
        VolumeProfileCalculator,
        get_chart_engine,
    )
except Exception:
    AdvancedChartEngine = None
    ChartType = None
    get_chart_engine = None

try:
    from .technical_indicators_extended import (
        ExtendedTechnicalEngine,
        TrendIndicators,
        MomentumIndicators,
        VolatilityIndicators,
        VolumeIndicators,
        CycleIndicators,
        StatisticalIndicators,
        PatternRecognition,
        get_extended_technical_engine,
    )
except Exception:
    ExtendedTechnicalEngine = None
    get_extended_technical_engine = None

try:
    from .drawing_tools import (
        DrawingManager,
        AutoDrawingEngine,
        DrawingType,
        Trendline,
        Channel,
        FibonacciRetracement,
        AndrewsPitchfork,
        GannTools,
        get_drawing_manager,
    )
except Exception:
    DrawingManager = None
    get_drawing_manager = None

try:
    from .advanced_backtest import (
        AdvancedBacktestEngine,
        ChinaTransactionCosts,
        SlippageModel,
        BacktestMetrics,
        BacktestTrade,
        WalkForwardOptimizer,
        MonteCarloSimulator,
        get_advanced_backtest_engine,
    )
except Exception:
    AdvancedBacktestEngine = None
    get_advanced_backtest_engine = None

__all__ = [
    # Original
    'TechnicalAnalyzer',
    'SentimentAnalyzer',
    'NewsScraper',
    'Backtester',
    'BacktestResult',
    'MarketReplay',
    'ReplayBar',
    
    # Enhanced - Charts
    'AdvancedChartEngine',
    'ChartType',
    'HeikinAshiCalculator',
    'RenkoCalculator',
    'KagiCalculator',
    'PointFigureCalculator',
    'LineBreakCalculator',
    'VolumeProfileCalculator',
    'get_chart_engine',
    
    # Enhanced - Indicators
    'ExtendedTechnicalEngine',
    'TrendIndicators',
    'MomentumIndicators',
    'VolatilityIndicators',
    'VolumeIndicators',
    'CycleIndicators',
    'StatisticalIndicators',
    'PatternRecognition',
    'get_extended_technical_engine',
    
    # Enhanced - Drawing
    'DrawingManager',
    'AutoDrawingEngine',
    'DrawingType',
    'Trendline',
    'Channel',
    'FibonacciRetracement',
    'AndrewsPitchfork',
    'GannTools',
    'get_drawing_manager',
    
    # Enhanced - Backtest
    'AdvancedBacktestEngine',
    'ChinaTransactionCosts',
    'SlippageModel',
    'BacktestMetrics',
    'BacktestTrade',
    'WalkForwardOptimizer',
    'MonteCarloSimulator',
    'get_advanced_backtest_engine',
]
