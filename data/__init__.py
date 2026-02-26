# data/__init__.py

def __getattr__(name: str):
    """Lazy import dispatcher for the data package."""
    _CACHE = {'TieredCache', 'get_cache', 'cached', 'CacheStats'}
    _DATABASE = {'MarketDatabase', 'get_database'}
    _FETCHER = {'DataFetcher', 'Quote', 'get_fetcher'}
    _FETCHER_UNIFIED = {
        'UnifiedDataFetcher', 'FetchOptions', 'FetchResult', 'LoadStatus',
        'get_unified_fetcher', 'fetch_unified',
    }
    _FETCHER_CONFIG = {'FetcherConfig', 'get_config'}
    _SOURCE_HEALTH = {
        'DataSourceHealthMonitor', 'SourceHealthStatus', 'SourceHealthState',
        'get_health_monitor', 'record_source_success', 'record_source_failure',
        'get_healthy_source',
    }
    _TIMEZONE_UTILS = {
        'TradingSessionChecker', 'TimezoneConverter',
        'get_session_checker', 'get_timezone_converter',
        'is_trading_time', 'filter_trading_hours', 'ensure_shanghai_datetime',
    }
    _PROGRESSIVE_LOADER = {
        'ProgressiveDataLoader', 'LoadResult', 'LoadStatus',
        'get_progressive_loader', 'load_with_progressive',
    }
    _FEATURES = {'FeatureEngine'}
    _PROCESSOR = {'DataProcessor'}
    _DISCOVERY = {'UniversalStockDiscovery', 'DiscoveredStock'}
    _FEEDS = {
        'DataFeed', 'PollingFeed', 'AggregatedFeed',
        'FeedManager', 'get_feed_manager',
    }
    _SESSION = {'SessionBarCache', 'get_session_bar_cache'}
    _VALIDATORS = {
        'ValidationResult', 'StockCodeValidator', 'OHLCVValidator',
        'FeatureValidator', 'OrderValidator',
        'validate_stock_code', 'validate_ohlcv', 'validate_features',
    }
    _SEARCH = {
        'WebSearchEngine', 'SearchResult', 'SearchEngine',
        'SearchQuery', 'get_search_engine',
    }
    _NEWS = {'NewsCollector', 'NewsArticle', 'get_collector'}
    _SENTIMENT = {'LLM_sentimentAnalyzer', 'get_analyzer'}

    if name in _CACHE:
        from . import cache as _cache
        return getattr(_cache, name)

    if name in _DATABASE:
        from . import database as _database
        return getattr(_database, name)

    if name in _FETCHER:
        from . import fetcher as _fetcher
        return getattr(_fetcher, name)

    if name in _FETCHER_UNIFIED:
        from . import fetcher_unified as _fetcher_unified
        return getattr(_fetcher_unified, name)

    if name in _FETCHER_CONFIG:
        from . import fetcher_config as _fetcher_config
        return getattr(_fetcher_config, name)

    if name in _SOURCE_HEALTH:
        from . import source_health as _source_health
        return getattr(_source_health, name)

    if name in _TIMEZONE_UTILS:
        from . import timezone_utils as _timezone_utils
        return getattr(_timezone_utils, name)

    if name in _PROGRESSIVE_LOADER:
        from . import progressive_loader as _progressive_loader
        return getattr(_progressive_loader, name)

    if name in _FEATURES:
        from .features import FeatureEngine
        return FeatureEngine

    if name in _PROCESSOR:
        from .processor import DataProcessor
        return DataProcessor

    if name in _DISCOVERY:
        from . import discovery as _discovery
        return getattr(_discovery, name)

    if name in _FEEDS:
        from . import feeds as _feeds
        return getattr(_feeds, name)

    if name in _SESSION:
        from . import session_cache as _session_cache
        return getattr(_session_cache, name)

    if name in _VALIDATORS:
        from . import validators as _validators
        return getattr(_validators, name)

    if name in _SEARCH:
        from . import web_search as _web_search
        return getattr(_web_search, name)

    if name in _NEWS:
        from . import news_collector as _news_collector
        return getattr(_news_collector, name)

    if name in _SENTIMENT:
        from . import llm_sentiment as _llm_sentiment
        return getattr(_llm_sentiment, name)

    # Social sentiment (enhanced features)
    if name == 'SocialSentimentAnalyzer' or name == 'get_social_sentiment_analyzer':
        from . import social_sentiment as _social_sentiment
        return getattr(_social_sentiment, name)

    raise AttributeError(f"module 'data' has no attribute {name!r}")

__all__ = [
    'TieredCache',
    'get_cache',
    'cached',
    'CacheStats',
    'MarketDatabase',
    'get_database',
    'DataFetcher',
    'Quote',
    'get_fetcher',
    # Unified fetcher (FIX 2026-02-26)
    'UnifiedDataFetcher',
    'FetchOptions',
    'FetchResult',
    'get_unified_fetcher',
    'fetch_unified',
    'FetcherConfig',
    'get_config',
    # Source health
    'DataSourceHealthMonitor',
    'SourceHealthStatus',
    'SourceHealthState',
    'get_health_monitor',
    'record_source_success',
    'record_source_failure',
    'get_healthy_source',
    # Timezone utilities
    'TradingSessionChecker',
    'TimezoneConverter',
    'get_session_checker',
    'get_timezone_converter',
    'is_trading_time',
    'filter_trading_hours',
    'ensure_shanghai_datetime',
    # Progressive loading
    'ProgressiveDataLoader',
    'LoadResult',
    'LoadStatus',
    'get_progressive_loader',
    'load_with_progressive',
    'FeatureEngine',
    'DataProcessor',
    'UniversalStockDiscovery',
    'DiscoveredStock',
    'DataFeed',
    'PollingFeed',
    'AggregatedFeed',
    'FeedManager',
    'get_feed_manager',
    'SessionBarCache',
    'get_session_bar_cache',
    'ValidationResult',
    'StockCodeValidator',
    'OHLCVValidator',
    'FeatureValidator',
    'OrderValidator',
    'validate_stock_code',
    'validate_ohlcv',
    'validate_features',
    # Web search
    'WebSearchEngine',
    'SearchResult',
    'SearchEngine',
    'SearchQuery',
    'get_search_engine',
    # News and sentiment
    'NewsCollector',
    'NewsArticle',
    'get_collector',
    'LLM_sentimentAnalyzer',
    'get_analyzer',
    # Enhanced social sentiment
    'SocialSentimentAnalyzer',
    'get_social_sentiment_analyzer',
]
