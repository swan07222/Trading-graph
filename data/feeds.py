# data/feeds.py
"""
Real-Time Data Feeds
"""
import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json

from config.settings import CONFIG  # FIXED
from core.events import EVENT_BUS, EventType, TickEvent, BarEvent
from core.exceptions import DataSourceUnavailableError
from utils.logger import get_logger

log = get_logger(__name__)


class FeedStatus(Enum):
    """Feed connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class Subscription:
    """Data subscription"""
    symbol: str
    data_type: str  # 'tick', 'bar', 'quote'
    interval: int = 0  # For bars, in seconds
    callback: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.now)


# Import Quote here to avoid circular import at module level
def _get_quote_class():
    from data.fetcher import Quote
    return Quote


class DataFeed(ABC):
    """Abstract base class for data feeds"""
    
    name: str = "base"
    
    def __init__(self):
        self.status = FeedStatus.DISCONNECTED
        self._subscriptions: Dict[str, Subscription] = {}
        self._callbacks: List[Callable] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to feed"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from feed"""
        pass
    
    @abstractmethod
    def subscribe(self, symbol: str, data_type: str = 'quote') -> bool:
        """Subscribe to symbol"""
        pass
    
    @abstractmethod
    def unsubscribe(self, symbol: str):
        """Unsubscribe from symbol"""
        pass
    
    def add_callback(self, callback: Callable):
        """Add data callback"""
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove data callback"""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def _notify(self, data):
        """Notify all callbacks"""
        with self._lock:
            callbacks = self._callbacks.copy()
        
        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                log.warning(f"Feed callback error: {e}")


class PollingFeed(DataFeed):
    """
    Polling-based data feed
    """
    
    name = "polling"
    
    def __init__(self, interval: float = 3.0):
        super().__init__()
        self._interval = interval
        self._fetcher = None  # Lazy init
        self._symbols: Set[str] = set()
        self._last_quotes: Dict[str, 'Quote'] = {}
    
    def connect(self) -> bool:
        """Start polling"""
        if self._running:
            return True
        
        self._running = True
        self.status = FeedStatus.CONNECTED
        
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        
        log.info(f"Polling feed started (interval={self._interval}s)")
        return True
    
    def _get_fetcher(self):
        """Lazy init fetcher to avoid circular imports"""
        if self._fetcher is None:
            from data.fetcher import get_fetcher
            self._fetcher = get_fetcher()
        return self._fetcher

    def disconnect(self):
        """Stop polling"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self.status = FeedStatus.DISCONNECTED
        log.info("Polling feed stopped")
    
    def subscribe(self, symbol: str, data_type: str = 'quote') -> bool:
        """Subscribe to symbol"""
        with self._lock:
            self._symbols.add(symbol)
            self._subscriptions[symbol] = Subscription(
                symbol=symbol,
                data_type=data_type
            )
        log.debug(f"Subscribed to {symbol}")
        return True
    
    def unsubscribe(self, symbol: str):
        """Unsubscribe from symbol"""
        with self._lock:
            self._symbols.discard(symbol)
            self._subscriptions.pop(symbol, None)
        log.debug(f"Unsubscribed from {symbol}")
    
    def _poll_loop(self):
        """Main polling loop"""
        while self._running:
            try:
                with self._lock:
                    symbols = list(self._symbols)
                
                if not symbols:
                    time.sleep(self._interval)
                    continue
                
                quotes = self._fetch_batch_quotes(symbols)
                
                for symbol, quote in quotes.items():
                    if quote and quote.price > 0:
                        self._last_quotes[symbol] = quote
                        self._notify(quote)

                        EVENT_BUS.publish(TickEvent(
                            symbol=symbol,
                            price=quote.price,
                            volume=quote.volume,
                            bid=quote.bid,
                            ask=quote.ask,
                            source=self.name
                        ))
                
                time.sleep(self._interval)
                
            except Exception as e:
                log.error(f"Polling loop error: {e}")
                time.sleep(1)

    def _fetch_batch_quotes(self, symbols: List[str]) -> Dict[str, 'Quote']:
        """Fetch quotes using shared spot cache"""
        from data.fetcher import get_spot_cache, Quote
        
        result = {}
        cache = get_spot_cache()
        
        df = cache.get()
        
        if df is None or df.empty:
            fetcher = self._get_fetcher()
            for symbol in symbols:
                quote = fetcher.get_realtime(symbol)
                if quote and quote.price > 0:
                    result[symbol] = quote
            return result
        
        for symbol in symbols:
            data = cache.get_quote(symbol)
            if data and data['price'] > 0:
                quote = Quote(
                    code=symbol,
                    name=data['name'],
                    price=data['price'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    volume=data['volume'],
                    amount=data['amount'],
                    change=data['change'],
                    change_pct=data['change_pct'],
                    source='spot_cache'
                )
                result[symbol] = quote
        
        return result

    def get_quote(self, symbol: str) -> Optional['Quote']:
        """Get last quote for symbol"""
        return self._last_quotes.get(symbol)
    
    def get_all_quotes(self) -> Dict[str, 'Quote']:
        """Get all last quotes"""
        with self._lock:
            return self._last_quotes.copy()


class WebSocketFeed(DataFeed):
    """WebSocket-based real-time data feed"""
    
    name = "websocket"
    
    def __init__(self):
        super().__init__()
        self._ws = None
        self._reconnect_delay = 1
        self._max_reconnect_delay = 60
        self._last_message_time: Dict[str, datetime] = {}
        self._staleness_threshold = timedelta(seconds=30)
        self._heartbeat_interval = 10
        self._symbols: Set[str] = set()
    
    def connect(self) -> bool:
        """Connect to WebSocket feed"""
        try:
            import websocket
            
            ws_url = "wss://push.sina.cn/ws"
            
            self._ws = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            self._running = True
            self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
            self._thread.start()
            
            self.status = FeedStatus.CONNECTING
            return True
            
        except ImportError:
            log.warning("websocket-client not installed, using polling")
            return False
        except Exception as e:
            log.error(f"WebSocket connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect WebSocket"""
        self._running = False
        if self._ws:
            self._ws.close()
        self.status = FeedStatus.DISCONNECTED
    
    def subscribe(self, symbol: str, data_type: str = 'quote') -> bool:
        """Subscribe to symbol"""
        with self._lock:
            self._symbols.add(symbol)
            
            if self._ws and self.status == FeedStatus.CONNECTED:
                msg = json.dumps({
                    "action": "subscribe",
                    "symbols": [symbol]
                })
                self._ws.send(msg)
            
            return True
    
    def unsubscribe(self, symbol: str):
        """Unsubscribe from symbol"""
        with self._lock:
            self._symbols.discard(symbol)
            
            if self._ws and self.status == FeedStatus.CONNECTED:
                msg = json.dumps({
                    "action": "unsubscribe",
                    "symbols": [symbol]
                })
                self._ws.send(msg)
    
    def _on_open(self, ws):
        """Handle connection open"""
        self.status = FeedStatus.CONNECTED
        self._reconnect_delay = 1
        log.info("WebSocket connected")
        
        with self._lock:
            for symbol in self._symbols:
                msg = json.dumps({
                    "action": "subscribe",
                    "symbols": [symbol]
                })
                ws.send(msg)
        
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
    
    def _on_message(self, ws, message):
        """Handle incoming message"""
        from data.fetcher import Quote
        
        try:
            data = json.loads(message)
            
            symbol = data.get('symbol') or data.get('code')
            if not symbol:
                return
            
            self._last_message_time[symbol] = datetime.now()
            
            quote = Quote(
                code=symbol,
                name=data.get('name', ''),
                price=float(data.get('price') or data.get('current') or 0),
                open=float(data.get('open', 0)),
                high=float(data.get('high', 0)),
                low=float(data.get('low', 0)),
                close=float(data.get('close') or data.get('preclose', 0)),
                volume=int(data.get('volume', 0)),
                amount=float(data.get('amount', 0)),
                change=float(data.get('change', 0)),
                change_pct=float(data.get('change_pct') or data.get('pct', 0)),
                bid=float(data.get('bid1', 0)),
                ask=float(data.get('ask1', 0)),
                source='websocket'
            )
            
            if quote.price > 0:
                self._notify(quote)
                
                EVENT_BUS.publish(TickEvent(
                    symbol=symbol,
                    price=quote.price,
                    volume=quote.volume,
                    bid=quote.bid,
                    ask=quote.ask,
                    source=self.name
                ))
                
        except Exception as e:
            log.debug(f"Message parse error: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket error"""
        log.error(f"WebSocket error: {error}")
        self.status = FeedStatus.ERROR
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle connection close"""
        if self._running:
            self.status = FeedStatus.RECONNECTING
            log.warning(f"WebSocket closed, reconnecting in {self._reconnect_delay}s...")
            time.sleep(self._reconnect_delay)
            
            self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
            
            self.connect()
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self._running and self.status == FeedStatus.CONNECTED:
            try:
                if self._ws:
                    self._ws.send('{"action":"heartbeat"}')
            except Exception:
                pass
            time.sleep(self._heartbeat_interval)
    
    def is_stale(self, symbol: str) -> bool:
        """Check if data for symbol is stale"""
        last_time = self._last_message_time.get(symbol)
        if not last_time:
            return True
        return datetime.now() - last_time > self._staleness_threshold
    
    def get_staleness(self, symbol: str) -> float:
        """Get staleness in seconds"""
        last_time = self._last_message_time.get(symbol)
        if not last_time:
            return float('inf')
        return (datetime.now() - last_time).total_seconds()


class AggregatedFeed(DataFeed):
    """Aggregated data feed combining multiple sources"""
    
    name = "aggregated"
    
    def __init__(self):
        super().__init__()
        self._feeds: List[DataFeed] = []
        self._primary_feed: Optional[DataFeed] = None
        self._quote_queue: queue.Queue = queue.Queue()
    
    def add_feed(self, feed: DataFeed, primary: bool = False):
        """Add a data feed source"""
        self._feeds.append(feed)
        if primary:
            self._primary_feed = feed
        
        feed.add_callback(self._on_feed_data)
    
    def connect(self) -> bool:
        """Connect all feeds"""
        connected = False
        
        for feed in self._feeds:
            try:
                if feed.connect():
                    connected = True
                    log.info(f"Connected to {feed.name}")
            except Exception as e:
                log.warning(f"Failed to connect {feed.name}: {e}")
        
        if connected:
            self.status = FeedStatus.CONNECTED
            self._running = True
        
        return connected
    
    def disconnect(self):
        """Disconnect all feeds"""
        self._running = False
        
        for feed in self._feeds:
            try:
                feed.disconnect()
            except Exception:
                pass
        
        self.status = FeedStatus.DISCONNECTED
    
    def subscribe(self, symbol: str, data_type: str = 'quote') -> bool:
        """Subscribe on all feeds"""
        success = False
        
        for feed in self._feeds:
            try:
                if feed.subscribe(symbol, data_type):
                    success = True
            except Exception:
                pass
        
        if success:
            with self._lock:
                self._subscriptions[symbol] = Subscription(
                    symbol=symbol,
                    data_type=data_type
                )
        
        return success
    
    def unsubscribe(self, symbol: str):
        """Unsubscribe from all feeds"""
        for feed in self._feeds:
            try:
                feed.unsubscribe(symbol)
            except Exception:
                pass
        
        with self._lock:
            self._subscriptions.pop(symbol, None)
    
    def _on_feed_data(self, data):
        """Handle data from any feed"""
        self._notify(data)


class BarAggregator:
    """Aggregates ticks into bars"""
    
    def __init__(self, interval_seconds: int = 60):
        self._interval = interval_seconds
        self._current_bars: Dict[str, Dict] = {}
        self._callbacks: List[Callable] = []
        self._lock = threading.RLock()
    
    def add_callback(self, callback: Callable):
        """Add bar callback"""
        self._callbacks.append(callback)
    
    def on_tick(self, quote):
        """Process tick into bar"""
        symbol = quote.code
        
        with self._lock:
            if symbol not in self._current_bars:
                self._current_bars[symbol] = self._new_bar(quote)
            
            bar = self._current_bars[symbol]
            
            last_cum_vol = bar.get('last_cum_vol', 0)
            
            bar['high'] = max(bar['high'], quote.price)
            bar['low'] = min(bar['low'], quote.price)
            bar['close'] = quote.price
            
            # Handle volume (could be None, NaN, or cumulative)
            current_cum_vol = 0
            if quote.volume is not None:
                try:
                    current_cum_vol = int(quote.volume)
                except (ValueError, TypeError):
                    current_cum_vol = 0
            
            if current_cum_vol < last_cum_vol:
                delta_vol = current_cum_vol  # reset/new session
            else:
                delta_vol = current_cum_vol - last_cum_vol

            bar["volume"] += max(delta_vol, 0)
            bar['last_cum_vol'] = current_cum_vol
            
            now = datetime.now()
            bar_end = bar['timestamp'] + timedelta(seconds=self._interval)
            
            if now >= bar_end:
                self._emit_bar(symbol, bar)
                self._current_bars[symbol] = self._new_bar(quote)
    
    def _new_bar(self, quote) -> Dict:
        """Create new bar"""
        now = datetime.now()
        seconds = (now.minute * 60 + now.second) % self._interval
        bar_start = now - timedelta(seconds=seconds, microseconds=now.microsecond)
        
        initial_vol = 0
        if quote.volume is not None:
            try:
                initial_vol = int(quote.volume)
            except (ValueError, TypeError):
                initial_vol = 0
        
        return {
            'timestamp': bar_start,
            'open': quote.price,
            'high': quote.price,
            'low': quote.price,
            'close': quote.price,
            'volume': 0,
            'last_cum_vol': initial_vol,
        }
    
    def _emit_bar(self, symbol: str, bar: Dict):
        """Emit completed bar"""
        EVENT_BUS.publish(BarEvent(
            symbol=symbol,
            open=bar['open'],
            high=bar['high'],
            low=bar['low'],
            close=bar['close'],
            volume=bar['volume'],
            timestamp=bar['timestamp']
        ))
        
        for callback in self._callbacks:
            try:
                callback(symbol, bar)
            except Exception as e:
                log.warning(f"Bar callback error: {e}")


class FeedManager:
    """Central manager for all data feeds"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._feeds: Dict[str, DataFeed] = {}
        self._active_feed: Optional[DataFeed] = None
        self._subscriptions: Set[str] = set()
        self._bar_aggregator = BarAggregator()
        self._lock = threading.RLock()
    
    def initialize(self):
        """Initialize feeds"""
        interval = CONFIG.data.poll_interval_seconds
        polling = PollingFeed(interval=interval)
        self._feeds['polling'] = polling
        self._active_feed = polling
        
        polling.connect()
        polling.add_callback(self._bar_aggregator.on_tick)
        
        log.info(f"Feed manager initialized (poll interval: {interval}s)")
    
    def subscribe(self, symbol: str) -> bool:
        """Subscribe to symbol"""
        with self._lock:
            if symbol in self._subscriptions:
                return True
            
            if self._active_feed:
                if self._active_feed.subscribe(symbol):
                    self._subscriptions.add(symbol)
                    return True
            
            return False
    
    def unsubscribe(self, symbol: str):
        """Unsubscribe from symbol"""
        with self._lock:
            if symbol in self._subscriptions:
                if self._active_feed:
                    self._active_feed.unsubscribe(symbol)
                self._subscriptions.discard(symbol)
    
    def subscribe_many(self, symbols: List[str]):
        """Subscribe to multiple symbols"""
        for symbol in symbols:
            self.subscribe(symbol)
    
    def get_quote(self, symbol: str) -> Optional['Quote']:
        """Get latest quote for a symbol"""
        if isinstance(self._active_feed, PollingFeed):
            return self._active_feed.get_quote(symbol)
        return None

    def get_last_quote_time(self, symbol: str) -> Optional[datetime]:
        """Get timestamp of last quote for a symbol"""
        quote = self.get_quote(symbol)
        if quote and hasattr(quote, 'timestamp'):
            return quote.timestamp
        return None

    def add_tick_callback(self, callback: Callable):
        """Add tick callback"""
        if self._active_feed:
            self._active_feed.add_callback(callback)
    
    def add_bar_callback(self, callback: Callable):
        """Add bar callback"""
        self._bar_aggregator.add_callback(callback)
    
    def shutdown(self):
        """Shutdown all feeds"""
        for feed in self._feeds.values():
            try:
                feed.disconnect()
            except Exception:
                pass
        
        log.info("Feed manager shutdown")


_feed_manager: Optional[FeedManager] = None


def get_feed_manager() -> FeedManager:
    """Get global feed manager"""
    global _feed_manager
    if _feed_manager is None:
        _feed_manager = FeedManager()
    return _feed_manager