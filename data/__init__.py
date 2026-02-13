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
    _SESSION = {'SessionBarCache', 'get_session_bar_cache'}
    _VALIDATORS = {
        'ValidationResult', 'StockCodeValidator', 'OHLCVValidator',
        'FeatureValidator', 'OrderValidator',
        'validate_stock_code', 'validate_ohlcv', 'validate_features',
    }

    if name in _CACHE:
        from . import cache as _cache
        return getattr(_cache, name)

    if name in _DATABASE:
        from . import database as _database
        return getattr(_database, name)

    if name in _FETCHER:
        from . import fetcher as _fetcher
        return getattr(_fetcher, name)

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
]
