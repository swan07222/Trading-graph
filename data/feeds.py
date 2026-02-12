# data/feeds.py
import json
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

from config.settings import CONFIG
from core.events import EVENT_BUS, BarEvent, TickEvent
from utils.logger import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------
# Enums / dataclasses
# ------------------------------------------------------------------


class FeedStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class Subscription:
    symbol: str
    data_type: str  # 'tick', 'bar', 'quote'
    interval: int = 0
    callback: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.now)


# ------------------------------------------------------------------
# Abstract base
# ------------------------------------------------------------------


class DataFeed(ABC):
    """Abstract base class for data feeds."""

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
        ...

    @abstractmethod
    def disconnect(self):
        ...

    @abstractmethod
    def subscribe(self, symbol: str, data_type: str = "quote") -> bool:
        ...

    @abstractmethod
    def unsubscribe(self, symbol: str):
        ...

    def add_callback(self, callback: Callable):
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def _notify(self, data):
        with self._lock:
            callbacks = self._callbacks.copy()
        for cb in callbacks:
            try:
                cb(data)
            except Exception as e:
                log.warning(f"Feed callback error: {e}")


# ------------------------------------------------------------------
# Polling feed
# ------------------------------------------------------------------


class PollingFeed(DataFeed):
    """Polling-based data feed with drift-resistant loop."""

    name = "polling"

    def __init__(self, interval: float = 3.0):
        super().__init__()
        self._interval = max(0.5, float(interval))
        self._fetcher = None
        self._symbols: Set[str] = set()
        self._last_quotes: Dict[str, object] = {}
        self._quotes_lock = threading.RLock()

    def connect(self) -> bool:
        if self._running:
            return True
        self._running = True
        self.status = FeedStatus.CONNECTED
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="polling_feed"
        )
        self._thread.start()
        log.info(f"Polling feed started (interval={self._interval}s)")
        return True

    def _get_fetcher(self):
        if self._fetcher is None:
            from data.fetcher import get_fetcher
            self._fetcher = get_fetcher()
        return self._fetcher

    def disconnect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self.status = FeedStatus.DISCONNECTED
        log.info("Polling feed stopped")

    def subscribe(self, symbol: str, data_type: str = "quote") -> bool:
        with self._lock:
            self._symbols.add(symbol)
            self._subscriptions[symbol] = Subscription(
                symbol=symbol, data_type=data_type
            )
        log.debug(f"Subscribed to {symbol}")
        return True

    def unsubscribe(self, symbol: str):
        with self._lock:
            self._symbols.discard(symbol)
            self._subscriptions.pop(symbol, None)
        log.debug(f"Unsubscribed from {symbol}")

    def _poll_loop(self):
        """Drift-resistant polling loop."""
        next_tick = time.monotonic()

        while self._running:
            next_tick += self._interval

            try:
                with self._lock:
                    symbols = list(self._symbols)

                if symbols:
                    quotes = self._fetch_batch_quotes(symbols)
                    now_ts = datetime.now()

                    for symbol, quote in quotes.items():
                        if quote and getattr(quote, 'price', 0) > 0:
                            if getattr(quote, "timestamp", None) is None:
                                quote.timestamp = now_ts

                            with self._quotes_lock:
                                self._last_quotes[symbol] = quote

                            self._notify(quote)

                            EVENT_BUS.publish(
                                TickEvent(
                                    symbol=symbol,
                                    price=float(quote.price),
                                    volume=int(
                                        getattr(quote, "volume", 0) or 0
                                    ),
                                    bid=float(
                                        getattr(quote, "bid", 0.0) or 0.0
                                    ),
                                    ask=float(
                                        getattr(quote, "ask", 0.0) or 0.0
                                    ),
                                    source=self.name,
                                )
                            )
            except Exception as e:
                log.error(f"Polling loop error: {e}")

            # Smooth back-pressure
            now = time.monotonic()
            if next_tick <= now:
                missed = int((now - next_tick) / self._interval) + 1
                next_tick += missed * self._interval
                time.sleep(min(0.1, self._interval * 0.1))
            else:
                time.sleep(max(0.0, next_tick - now))

    def _fetch_batch_quotes(self, symbols: List[str]) -> Dict[str, object]:
        """Fetch quotes for all symbols with fallback chain."""
        result: Dict[str, object] = {}
        fetcher = self._get_fetcher()

        try:
            batch = fetcher.get_realtime_batch(symbols)
            if isinstance(batch, dict) and batch:
                result.update(batch)
        except Exception:
            pass

        missing = [s for s in symbols if s not in result]

        if missing:
            try:
                from data.fetcher import Quote, get_spot_cache

                cache = get_spot_cache()
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
                            change_pct=float(
                                data.get("change_pct", 0) or 0
                            ),
                            source="spot_cache",
                            is_delayed=False,
                        )
            except Exception:
                pass

        missing = [s for s in symbols if s not in result]
        if missing and len(missing) <= 8:
            for symbol in missing:
                try:
                    q = fetcher.get_realtime(symbol)
                    if q and getattr(q, 'price', 0) > 0:
                        result[symbol] = q
                except Exception:
                    continue

        return result

    def get_quote(self, symbol: str) -> Optional[object]:
        with self._quotes_lock:
            return self._last_quotes.get(symbol)

    def get_all_quotes(self) -> Dict[str, object]:
        with self._quotes_lock:
            return self._last_quotes.copy()


# ------------------------------------------------------------------
# WebSocket feed
# ------------------------------------------------------------------


class WebSocketFeed(DataFeed):
    """WebSocket-based real-time data feed with bounded reconnection."""

    name = "websocket"

    _MAX_RECONNECT_THREADS = 1
    _MAX_RECONNECT_ATTEMPTS = 50

    def __init__(self):
        super().__init__()
        self._ws = None
        self._reconnect_delay = 1
        self._max_reconnect_delay = 60
        self._reconnect_count = 0
        self._last_message_time: Dict[str, datetime] = {}
        self._staleness_threshold = timedelta(seconds=30)
        self._heartbeat_interval = 10
        self._symbols: Set[str] = set()
        self._reconnect_semaphore = threading.Semaphore(
            self._MAX_RECONNECT_THREADS
        )

    def connect(self) -> bool:
        try:
            import websocket

            ws_url = "wss://push.sina.cn/ws"
            self._ws = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open,
            )
            self._running = True
            self._thread = threading.Thread(
                target=self._ws.run_forever,
                daemon=True,
                name="ws_feed"
            )
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
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        self.status = FeedStatus.DISCONNECTED

    def subscribe(self, symbol: str, data_type: str = "quote") -> bool:
        with self._lock:
            self._symbols.add(symbol)
            self._subscriptions[symbol] = Subscription(
                symbol=symbol, data_type=data_type
            )
            if self._ws and self.status == FeedStatus.CONNECTED:
                self._send_subscribe(symbol)
        return True

    def unsubscribe(self, symbol: str):
        with self._lock:
            self._symbols.discard(symbol)
            self._subscriptions.pop(symbol, None)
            if self._ws and self.status == FeedStatus.CONNECTED:
                try:
                    self._ws.send(
                        json.dumps({
                            "action": "unsubscribe",
                            "symbols": [symbol]
                        })
                    )
                except Exception:
                    pass

    def _send_subscribe(self, symbol: str):
        try:
            self._ws.send(
                json.dumps({
                    "action": "subscribe",
                    "symbols": [symbol]
                })
            )
        except Exception:
            pass

    def _on_open(self, ws):
        self.status = FeedStatus.CONNECTED
        self._reconnect_delay = 1
        self._reconnect_count = 0
        log.info("WebSocket connected")

        with self._lock:
            for symbol in self._symbols:
                self._send_subscribe(symbol)

        threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="ws_heartbeat"
        ).start()

    def _on_message(self, ws, message):
        from data.fetcher import Quote

        try:
            data = json.loads(message)
            symbol = data.get("symbol") or data.get("code")
            if not symbol:
                return

            self._last_message_time[symbol] = datetime.now()

            raw_vol = data.get("volume", 0)
            try:
                vol = int(float(raw_vol))
            except (ValueError, TypeError):
                vol = 0

            quote = Quote(
                code=symbol,
                name=data.get("name", ""),
                price=float(
                    data.get("price") or data.get("current") or 0
                ),
                open=float(data.get("open", 0) or 0),
                high=float(data.get("high", 0) or 0),
                low=float(data.get("low", 0) or 0),
                close=float(
                    data.get("close") or data.get("preclose", 0) or 0
                ),
                volume=vol,
                amount=float(data.get("amount", 0) or 0),
                change=float(data.get("change", 0) or 0),
                change_pct=float(
                    data.get("change_pct") or data.get("pct", 0) or 0
                ),
                bid=float(data.get("bid1", 0) or 0),
                ask=float(data.get("ask1", 0) or 0),
                source="websocket",
            )

            if quote.price > 0:
                self._notify(quote)
                EVENT_BUS.publish(
                    TickEvent(
                        symbol=symbol,
                        price=quote.price,
                        volume=quote.volume,
                        bid=quote.bid,
                        ask=quote.ask,
                        source=self.name,
                    )
                )
        except Exception as e:
            log.debug(f"Message parse error: {e}")

    def _on_error(self, ws, error):
        log.error(f"WebSocket error: {error}")
        self.status = FeedStatus.ERROR

    def _on_close(self, ws, close_status_code, close_msg):
        if not self._running:
            self.status = FeedStatus.DISCONNECTED
            return

        self._reconnect_count += 1

        if self._reconnect_count > self._MAX_RECONNECT_ATTEMPTS:
            log.error(
                f"Max reconnect attempts ({self._MAX_RECONNECT_ATTEMPTS}) "
                f"exceeded. Giving up."
            )
            self.status = FeedStatus.ERROR
            self._running = False
            return

        self.status = FeedStatus.RECONNECTING
        delay = int(self._reconnect_delay)
        log.warning(
            f"WebSocket closed, reconnecting in {delay}s "
            f"(attempt {self._reconnect_count}/"
            f"{self._MAX_RECONNECT_ATTEMPTS})..."
        )

        if not self._reconnect_semaphore.acquire(blocking=False):
            log.debug("Reconnect already in progress, skipping")
            return

        def _reconnect():
            try:
                time.sleep(delay)
                if not self._running:
                    return
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay
                )
                try:
                    self.connect()
                except Exception as e:
                    log.debug(f"Reconnect failed: {e}")
            finally:
                self._reconnect_semaphore.release()

        threading.Thread(
            target=_reconnect, daemon=True, name="ws_reconnect"
        ).start()

    def _heartbeat_loop(self):
        while self._running and self.status == FeedStatus.CONNECTED:
            try:
                ws = self._ws
                if ws:
                    ws.send('{"action":"heartbeat"}')
            except Exception:
                break
            time.sleep(self._heartbeat_interval)

    def is_stale(self, symbol: str) -> bool:
        last = self._last_message_time.get(symbol)
        if not last:
            return True
        return datetime.now() - last > self._staleness_threshold

    def get_staleness(self, symbol: str) -> float:
        last = self._last_message_time.get(symbol)
        if not last:
            return float("inf")
        return (datetime.now() - last).total_seconds()


# ------------------------------------------------------------------
# Aggregated feed
# ------------------------------------------------------------------


class AggregatedFeed(DataFeed):
    """Aggregated data feed combining multiple sources."""

    name = "aggregated"

    def __init__(self):
        super().__init__()
        self._feeds: List[DataFeed] = []
        self._primary_feed: Optional[DataFeed] = None
        self._quote_queue: queue.Queue = queue.Queue()

    def add_feed(self, feed: DataFeed, primary: bool = False):
        self._feeds.append(feed)
        if primary:
            self._primary_feed = feed
        feed.add_callback(self._on_feed_data)

    def connect(self) -> bool:
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
        self._running = False
        for feed in self._feeds:
            try:
                feed.disconnect()
            except Exception:
                pass
        self.status = FeedStatus.DISCONNECTED

    def subscribe(self, symbol: str, data_type: str = "quote") -> bool:
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
                    symbol=symbol, data_type=data_type
                )
        return success

    def unsubscribe(self, symbol: str):
        for feed in self._feeds:
            try:
                feed.unsubscribe(symbol)
            except Exception:
                pass
        with self._lock:
            self._subscriptions.pop(symbol, None)

    def _on_feed_data(self, data):
        self._notify(data)


# ------------------------------------------------------------------
# Volume mode enum
# ------------------------------------------------------------------


class VolumeMode(Enum):
    """Volume interpretation mode."""
    CUMULATIVE = "cumulative"
    DELTA = "delta"


# ------------------------------------------------------------------
# Bar aggregator - FIXED to emit partial bars
# ------------------------------------------------------------------


class BarAggregator:
    """
    Aggregates ticks into OHLCV bars with configurable interval.
    
    FIXED: Now emits PARTIAL bars on every tick so the chart
    updates in real-time, not just on bar boundaries.
    """

    def __init__(
        self,
        interval_seconds: int = 60,
        volume_mode: VolumeMode = VolumeMode.CUMULATIVE
    ):
        self._interval = max(1, int(interval_seconds))
        self._volume_mode = volume_mode
        self._current_bars: Dict[str, Dict] = {}
        self._callbacks: List[Callable] = []
        self._lock = threading.RLock()

    def add_callback(self, callback: Callable):
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def set_volume_mode(self, mode: VolumeMode):
        """Change volume interpretation mode."""
        with self._lock:
            self._volume_mode = mode

    def on_tick(self, quote):
        """
        Process incoming tick/quote.
        
        FIXED: Now emits partial bar on EVERY tick for real-time updates.
        """
        symbol = getattr(quote, "code", None)
        if not symbol:
            return

        ts = getattr(quote, "timestamp", None) or datetime.now()
        px = float(getattr(quote, "price", 0) or 0)
        if px <= 0:
            return

        with self._lock:
            if symbol not in self._current_bars:
                self._current_bars[symbol] = self._new_bar(quote)

            bar = self._current_bars[symbol]

            # Session/day boundary
            if (
                bar.get("session_date")
                and bar["session_date"] != ts.date()
            ):
                self._emit_bar(symbol, bar, final=True)
                self._current_bars[symbol] = self._new_bar(quote)
                bar = self._current_bars[symbol]

            # Update OHLC
            bar["high"] = max(float(bar["high"]), px)
            bar["low"] = min(float(bar["low"]), px)
            bar["close"] = px

            # Volume handling based on mode
            self._update_volume(bar, quote)

            # Bar boundary check
            bar_end = bar["timestamp"] + timedelta(seconds=self._interval)
            
            if ts >= bar_end:
                # Bar complete - emit as final and start new bar
                self._emit_bar(symbol, bar, final=True)
                self._current_bars[symbol] = self._new_bar(quote)
            else:
                # FIXED: Emit partial bar for real-time chart updates
                self._emit_bar(symbol, bar, final=False)

    def _update_volume(self, bar: Dict, quote):
        """Update bar volume based on configured mode."""
        raw_vol = getattr(quote, "volume", None)
        if raw_vol is None:
            return

        try:
            vol_value = int(float(raw_vol))
        except (ValueError, TypeError):
            return

        if self._volume_mode == VolumeMode.CUMULATIVE:
            last_cum = int(bar.get("last_cum_vol", 0) or 0)

            if vol_value < last_cum:
                delta = 0
            else:
                delta = vol_value - last_cum

            bar["volume"] += max(delta, 0)
            bar["last_cum_vol"] = vol_value

        elif self._volume_mode == VolumeMode.DELTA:
            bar["volume"] += max(vol_value, 0)

    def _new_bar(self, quote) -> Dict:
        """Create a new bar with proper time alignment."""
        ts = getattr(quote, "timestamp", None) or datetime.now()

        total_seconds = ts.hour * 3600 + ts.minute * 60 + ts.second
        remainder = total_seconds % max(self._interval, 1)
        bar_start = ts - timedelta(
            seconds=remainder, microseconds=ts.microsecond
        )

        initial_vol = 0
        if self._volume_mode == VolumeMode.CUMULATIVE:
            raw_vol = getattr(quote, "volume", None)
            if raw_vol is not None:
                try:
                    initial_vol = int(float(raw_vol))
                except (ValueError, TypeError):
                    initial_vol = 0

        return {
            "timestamp": bar_start,
            "open": float(getattr(quote, "price", 0)),
            "high": float(getattr(quote, "price", 0)),
            "low": float(getattr(quote, "price", 0)),
            "close": float(getattr(quote, "price", 0)),
            "volume": 0,
            "last_cum_vol": initial_vol,
            "session_date": bar_start.date(),
        }

    def set_interval(self, interval_seconds: int):
        """Change bar interval; clears partial bars."""
        with self._lock:
            self._interval = max(1, int(interval_seconds))
            self._current_bars.clear()

    def _emit_bar(self, symbol: str, bar: Dict, final: bool = True):
        """
        Emit bar to callbacks.
        
        Args:
            symbol: Stock symbol
            bar: Bar data dict
            final: If True, this is a completed bar. If False, partial/live bar.
        """
        # Add final flag to bar data
        bar_copy = dict(bar)
        bar_copy["final"] = final

        # Publish event only for final bars (to avoid spamming event bus)
        if final:
            EVENT_BUS.publish(
                BarEvent(
                    symbol=symbol,
                    open=bar["open"],
                    high=bar["high"],
                    low=bar["low"],
                    close=bar["close"],
                    volume=bar["volume"],
                    timestamp=bar["timestamp"],
                )
            )

            # Persist final bars to DB
            try:
                import pandas as pd
                from data.database import get_database

                db = get_database()
                df = pd.DataFrame(
                    [{
                        "open": float(bar["open"]),
                        "high": float(bar["high"]),
                        "low": float(bar["low"]),
                        "close": float(bar["close"]),
                        "volume": int(bar["volume"]),
                        "amount": 0.0,
                    }],
                    index=pd.DatetimeIndex([bar["timestamp"]]),
                )

                if self._interval >= 60:
                    mins = self._interval / 60
                    if mins == int(mins):
                        label = f"{int(mins)}m"
                    else:
                        label = f"{self._interval}s"
                else:
                    label = f"{self._interval}s"

                db.upsert_intraday_bars(symbol, label, df)
            except ImportError:
                pass
            except Exception as e:
                log.debug(f"Bar DB persist failed for {symbol}: {e}")

        # Notify callbacks for BOTH partial and final bars
        with self._lock:
            callbacks = self._callbacks.copy()

        for cb in callbacks:
            try:
                cb(symbol, bar_copy)
            except Exception as e:
                log.warning(f"Bar callback error: {e}")


# ------------------------------------------------------------------
# Feed manager (singleton)
# ------------------------------------------------------------------


class FeedManager:
    """Central manager for all data feeds."""

    _instance = None
    _cls_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._cls_lock:
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
        self._last_quotes: Dict[str, object] = {}
        self._quotes_lock = threading.RLock()
        self._lock = threading.RLock()
        self._initialized_runtime = False

    def initialize(self, force: bool = False):
        """Initialize feeds. Idempotent unless force=True."""
        with self._lock:
            if self._initialized_runtime and not force:
                return
            self._initialized_runtime = True

        interval = float(CONFIG.data.poll_interval_seconds)

        polling = PollingFeed(interval=interval)
        self._feeds["polling"] = polling

        active = None
        try:
            ws = WebSocketFeed()
            if ws.connect():
                self._feeds["websocket"] = ws
                active = ws
                log.info("Using WebSocket feed as primary")
            else:
                try:
                    ws.disconnect()
                except Exception:
                    pass
        except Exception:
            active = None

        if active is None:
            polling.connect()
            active = polling
            log.info("Using polling feed as primary")

        self._active_feed = active

        # Preserve existing bar callbacks across re-init
        old_callbacks: List[Callable] = []
        with self._lock:
            if (
                hasattr(self, "_bar_aggregator")
                and self._bar_aggregator
            ):
                with self._bar_aggregator._lock:
                    old_callbacks = self._bar_aggregator._callbacks.copy()

        bar_seconds = 60
        self._bar_aggregator = BarAggregator(
            interval_seconds=bar_seconds
        )
        for cb in old_callbacks:
            self._bar_aggregator.add_callback(cb)

        # Attach feed callbacks
        try:
            self._active_feed.add_callback(self._cache_quote)
        except Exception:
            pass
        try:
            self._active_feed.add_callback(self._bar_aggregator.on_tick)
        except Exception:
            pass

        log.info(
            f"Feed manager initialized "
            f"(primary={self._active_feed.name}, "
            f"poll={interval}s, bar={bar_seconds}s)"
        )

    def set_bar_interval_seconds(self, seconds: int):
        """Change bar aggregation interval."""
        try:
            self._bar_aggregator.set_interval(int(seconds))
        except Exception:
            pass

    def set_bar_volume_mode(self, mode: VolumeMode):
        """Change bar volume interpretation mode."""
        try:
            self._bar_aggregator.set_volume_mode(mode)
        except Exception:
            pass

    def ensure_initialized(self, async_init: bool = True):
        if self._initialized_runtime:
            return
        if async_init:
            threading.Thread(
                target=self.initialize,
                daemon=True,
                name="feed_init"
            ).start()
        else:
            self.initialize()

    def subscribe(self, symbol: str) -> bool:
        with self._lock:
            if symbol in self._subscriptions:
                return True
            if (
                self._active_feed
                and self._active_feed.subscribe(symbol)
            ):
                self._subscriptions.add(symbol)
                return True
            return False

    def unsubscribe(self, symbol: str):
        with self._lock:
            if symbol in self._subscriptions:
                if self._active_feed:
                    self._active_feed.unsubscribe(symbol)
                self._subscriptions.discard(symbol)

    def _cache_quote(self, data):
        """Thread-safe quote caching."""
        try:
            code = (
                getattr(data, "code", None)
                or getattr(data, "symbol", None)
            )
            if not code:
                return
            price = float(getattr(data, "price", 0.0) or 0.0)
            if price <= 0:
                return
            with self._quotes_lock:
                self._last_quotes[str(code)] = data
        except Exception:
            pass

    def subscribe_many(self, symbols: List[str]):
        for symbol in symbols:
            self.subscribe(symbol)

    def get_quote(self, symbol: str) -> Optional[object]:
        with self._quotes_lock:
            q = self._last_quotes.get(str(symbol))
            if q:
                return q

        if isinstance(self._active_feed, PollingFeed):
            return self._active_feed.get_quote(symbol)
        return None

    def get_last_quote_time(self, symbol: str) -> Optional[datetime]:
        quote = self.get_quote(symbol)
        if quote and hasattr(quote, "timestamp"):
            return quote.timestamp
        return None

    def add_tick_callback(self, callback: Callable):
        if self._active_feed:
            self._active_feed.add_callback(callback)

    def add_bar_callback(self, callback: Callable):
        self._bar_aggregator.add_callback(callback)

    def shutdown(self):
        """Shutdown all feeds and reset state."""
        for feed in self._feeds.values():
            try:
                feed.disconnect()
            except Exception:
                pass

        self._feeds.clear()
        self._active_feed = None
        self._subscriptions.clear()

        with self._quotes_lock:
            self._last_quotes.clear()

        self._initialized_runtime = False
        log.info("Feed manager shutdown")


# ------------------------------------------------------------------
# Module-level singleton accessor
# ------------------------------------------------------------------

_feed_manager: Optional[FeedManager] = None
_feed_lock = threading.Lock()


def get_feed_manager(
    auto_init: bool = True, async_init: bool = True
) -> FeedManager:
    global _feed_manager
    if _feed_manager is None:
        with _feed_lock:
            if _feed_manager is None:
                _feed_manager = FeedManager()
    if auto_init:
        _feed_manager.ensure_initialized(async_init=async_init)
    return _feed_manager