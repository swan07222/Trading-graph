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
        """Drift-resistant polling loop."""
        next_tick = time.monotonic()
        while self._running:
            next_tick += float(self._interval)

            try:
                with self._lock:
                    symbols = list(self._symbols)

                if symbols:
                    quotes = self._fetch_batch_quotes(symbols)
                    now_ts = datetime.now()

                    for symbol, quote in quotes.items():
                        if quote and quote.price > 0:
                            if getattr(quote, "timestamp", None) is None:
                                quote.timestamp = now_ts

                            self._last_quotes[symbol] = quote
                            self._notify(quote)

                            EVENT_BUS.publish(TickEvent(
                                symbol=symbol,
                                price=float(quote.price),
                                volume=int(quote.volume or 0),
                                bid=float(getattr(quote, "bid", 0.0) or 0.0),
                                ask=float(getattr(quote, "ask", 0.0) or 0.0),
                                source=self.name
                            ))
            except Exception as e:
                log.error(f"Polling loop error: {e}")

            time.sleep(max(0.0, next_tick - time.monotonic()))

    def _fetch_batch_quotes(self, symbols: List[str]) -> Dict[str, 'Quote']:
        """
        Fast batch quotes:
        1) DataFetcher.get_realtime_batch
        2) SpotCache fill missing
        3) Per-symbol fallback only if few missing
        """
        result: Dict[str, 'Quote'] = {}
        fetcher = self._get_fetcher()

        # 1) Batch
        try:
            batch = fetcher.get_realtime_batch(symbols)
            if isinstance(batch, dict) and batch:
                result.update(batch)
        except Exception:
            pass

        missing = [s for s in symbols if s not in result]

        # 2) SpotCache fill
        if missing:
            try:
                from data.fetcher import get_spot_cache, Quote
                cache = get_spot_cache()
                df = cache.get()
                if df is not None and not df.empty:
                    for symbol in missing:
                        data = cache.get_quote(symbol)
                        if data and data.get("price", 0) > 0:
                            result[symbol] = Quote(
                                code=symbol,
                                name=data.get("name", ""),
                                price=float(data["price"]),
                                open=float(data.get("open", 0) or 0),
                                high=float(data.get("high", 0) or 0),
                                low=float(data.get("low", 0) or 0),
                                close=float(data.get("close", 0) or 0),
                                volume=int(data.get("volume", 0) or 0),
                                amount=float(data.get("amount", 0) or 0),
                                change=float(data.get("change", 0) or 0),
                                change_pct=float(data.get("change_pct", 0) or 0),
                                source="spot_cache",
                                is_delayed=False,
                            )
            except Exception:
                pass

        # 3) Per-symbol only if few missing
        missing = [s for s in symbols if s not in result]
        if missing and len(missing) <= 8:
            for symbol in missing:
                try:
                    q = fetcher.get_realtime(symbol)
                    if q and q.price > 0:
                        result[symbol] = q
                except Exception:
                    continue

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
        symbol = quote.code
        ts = getattr(quote, "timestamp", None) or datetime.now()
        px = float(getattr(quote, "price", 0) or 0)
        if px <= 0:
            return

        with self._lock:
            if symbol not in self._current_bars:
                self._current_bars[symbol] = self._new_bar(quote)

            bar = self._current_bars[symbol]

            # Day/session cut
            if bar.get("session_date") and bar["session_date"] != ts.date():
                self._emit_bar(symbol, bar)
                self._current_bars[symbol] = self._new_bar(quote)
                bar = self._current_bars[symbol]

            # OHLC
            bar["high"] = max(float(bar["high"]), px)
            bar["low"] = min(float(bar["low"]), px)
            bar["close"] = px

            # volume delta from cumulative - with safety checks
            last_cum = int(bar.get("last_cum_vol", 0) or 0)
            cur_cum = 0
            
            vol = getattr(quote, 'volume', None)
            if vol is not None:
                try:
                    cur_cum = int(vol)
                except (ValueError, TypeError):
                    cur_cum = 0

            delta = cur_cum if cur_cum < last_cum else (cur_cum - last_cum)
            bar["volume"] += max(int(delta), 0)
            bar["last_cum_vol"] = cur_cum

            # Bar end boundary
            if ts >= bar["timestamp"] + timedelta(seconds=self._interval):
                self._emit_bar(symbol, bar)
                self._current_bars[symbol] = self._new_bar(quote)
    
    def _new_bar(self, quote) -> Dict:
        ts = getattr(quote, "timestamp", None) or datetime.now()
        seconds = (ts.minute * 60 + ts.second) % self._interval
        bar_start = ts - timedelta(seconds=seconds, microseconds=ts.microsecond)

        initial_vol = 0
        if quote.volume is not None:
            try:
                initial_vol = int(quote.volume)
            except (ValueError, TypeError):
                initial_vol = 0

        return {
            "timestamp": bar_start,
            "open": float(quote.price),
            "high": float(quote.price),
            "low": float(quote.price),
            "close": float(quote.price),
            "volume": 0,
            "last_cum_vol": initial_vol,
            "session_date": bar_start.date(),
        }
    
    def _emit_bar(self, symbol: str, bar: Dict):
        """Emit completed bar + persist to local DB (intraday history recorder)."""
        EVENT_BUS.publish(BarEvent(
            symbol=symbol,
            open=bar['open'],
            high=bar['high'],
            low=bar['low'],
            close=bar['close'],
            volume=bar['volume'],
            timestamp=bar['timestamp']
        ))

        # -------- NEW: persist to DB as 1m bars --------
        try:
            import pandas as pd
            from data.database import get_database
            db = get_database()

            df = pd.DataFrame([{
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": int(bar["volume"]),
                "amount": float(0.0),
            }], index=pd.DatetimeIndex([bar["timestamp"]]))

            # interval_seconds -> interval string
            interval = f"{int(self._interval // 60)}m" if self._interval >= 60 else f"{int(self._interval)}s"
            # for 60s it becomes "1m" as desired
            if interval == "1m":
                db.upsert_intraday_bars(symbol, "1m", df)
            else:
                db.upsert_intraday_bars(symbol, interval, df)
        except Exception:
            pass

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
        """Initialize feeds: prefer WebSocket if available; fallback to polling."""
        interval = float(CONFIG.data.poll_interval_seconds)

        # Always create polling fallback
        polling = PollingFeed(interval=interval)
        self._feeds["polling"] = polling

        # Try websocket first (if websocket-client installed and connect works)
        active = None
        try:
            ws = WebSocketFeed()
            ok = ws.connect()
            if ok:
                self._feeds["websocket"] = ws
                active = ws
                log.info("Using WebSocket feed as primary")
            else:
                ws.disconnect()
        except Exception:
            active = None

        if active is None:
            polling.connect()
            active = polling
            log.info("Using polling feed as primary")

        self._active_feed = active

        # Bar interval: 1m by default
        bar_seconds = 60
        self._bar_aggregator = BarAggregator(interval_seconds=bar_seconds)

        # Attach bar aggregator to primary feed
        try:
            self._active_feed.add_callback(self._bar_aggregator.on_tick)
        except Exception:
            pass

        log.info(f"Feed manager initialized (primary={self._active_feed.name}, poll={interval}s, bar={bar_seconds}s)")
    
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
    global _feed_manager
    if _feed_manager is None:
        _feed_manager = FeedManager()
        _feed_manager.initialize()  # ADD THIS
    return _feed_manager