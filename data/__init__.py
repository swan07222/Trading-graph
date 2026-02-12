# data/__init__.py

def __getattr__(name: str):
    """Lazy import dispatcher for the data package."""

    _CACHE = {'TieredCache', 'get_cache', 'cached', 'CacheStats'}
    _DATABASE = {'MarketDatabase', 'get_database'}
    _FETCHER = {'DataFetcher', 'Quote', 'get_fetcher'}
    _FEATURES = {'FeatureEngine'}
    _PROCESSOR = {'DataProcessor'}
    _DISCOVERY = {'UniversalStockDiscovery', 'DiscoveredStock'}
    _FEEDS = {
        'DataFeed', 'PollingFeed', 'AggregatedFeed',
        'FeedManager', 'get_feed_manager',
    }
    _VALIDATORS = {
        'ValidationResult', 'StockCodeValidator', 'OHLCVValidator',
        'FeatureValidator', 'OrderValidator',
        'validate_stock_code', 'validate_ohlcv', 'validate_features',
    }

    if name in _CACHE:
        from .cache import TieredCache, get_cache, cached, CacheStats
        return locals()[name]

    if name in _DATABASE:
        from .database import MarketDatabase, get_database
        return locals()[name]

    if name in _FETCHER:
        from .fetcher import DataFetcher, Quote, get_fetcher
        return locals()[name]

    if name in _FEATURES:
        from .features import FeatureEngine
        return FeatureEngine

    if name in _PROCESSOR:
        from .processor import DataProcessor
        return DataProcessor

    if name in _DISCOVERY:
        from .discovery import UniversalStockDiscovery, DiscoveredStock
        return locals()[name]

    if name in _FEEDS:
        from .feeds import (
            DataFeed, PollingFeed, AggregatedFeed,
            FeedManager, get_feed_manager,
        )
        return locals()[name]

    if name in _VALIDATORS:
        from .validators import (
            ValidationResult, StockCodeValidator, OHLCVValidator,
            FeatureValidator, OrderValidator,
            validate_stock_code, validate_ohlcv, validate_features,
        )
        return locals()[name]

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
    'FeatureEngine',
    'DataProcessor',
    'UniversalStockDiscovery',
    'DiscoveredStock',
    'DataFeed',
    'PollingFeed',
    'AggregatedFeed',
    'FeedManager',
    'get_feed_manager',
    'ValidationResult',
    'StockCodeValidator',
    'OHLCVValidator',
    'FeatureValidator',
    'OrderValidator',
    'validate_stock_code',
    'validate_ohlcv',
    'validate_features',
]