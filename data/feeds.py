# data/feeds.py
import importlib.util
import json
import os
import queue
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from config.settings import CONFIG
from core.events import EVENT_BUS, BarEvent, TickEvent
from utils.logger import get_logger

log = get_logger(__name__)

# Enums / dataclasses

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
    callback: Callable | None = None
    created_at: datetime = field(default_factory=datetime.now)

class DataFeed(ABC):
    """Abstract base class for data feeds."""

    name: str = "base"

    def __init__(self):
        self.status = FeedStatus.DISCONNECTED
        self._subscriptions: dict[str, Subscription] = {}
        self._callbacks: list[Callable] = []
        self._running = False
        self._thread: threading.Thread | None = None
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

class PollingFeed(DataFeed):
    """Polling-based data feed with drift-resistant loop."""

    name = "polling"

    def __init__(self, interval: float = 3.0):
        super().__init__()
        self._interval = max(0.5, float(interval))
        self._fetcher = None
        self._symbols: set[str] = set()
        self._last_quotes: dict[str, object] = {}
        self._quotes_lock = threading.RLock()
        self._consecutive_errors: int = 0
        self._last_error_log_ts: float = 0.0

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
                    try:
                        from zoneinfo import ZoneInfo

                        now_ts = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
                    except Exception:
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
                self._consecutive_errors = 0
            except Exception as e:
                self._consecutive_errors = int(self._consecutive_errors) + 1
                now_log = time.monotonic()
                if (
                    self._consecutive_errors <= 2
                    or (now_log - self._last_error_log_ts) >= 12.0
                ):
                    log.warning(f"Polling loop error: {e}")
                    self._last_error_log_ts = now_log
                backoff = min(
                    5.0,
                    self._interval * (1.5 ** min(5, self._consecutive_errors)),
                )
                next_tick = max(next_tick, time.monotonic() + backoff)

            # Smooth back-pressure
            now = time.monotonic()
            if next_tick <= now:
                missed = int((now - next_tick) / self._interval) + 1
                next_tick += missed * self._interval
                time.sleep(min(0.1, self._interval * 0.1))
            else:
                time.sleep(max(0.0, next_tick - now))

    def _fetch_batch_quotes(self, symbols: list[str]) -> dict[str, object]:
        """Fetch quotes for all symbols with fallback chain."""
        result: dict[str, object] = {}
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

    def get_quote(self, symbol: str) -> object | None:
        with self._quotes_lock:
            return self._last_quotes.get(symbol)

    def get_all_quotes(self) -> dict[str, object]:
        with self._quotes_lock:
            return self._last_quotes.copy()

class WebSocketFeed(DataFeed):
    """WebSocket-based real-time data feed with bounded reconnection."""

    name = "websocket"

    _MAX_RECONNECT_THREADS = 1
    _MAX_RECONNECT_ATTEMPTS = 50
    _DNS_FAILURE_DISABLE_THRESHOLD = 6
    _DNS_DISABLE_COOLDOWN_SECONDS = 15 * 60
    _NETWORK_DISABLE_COOLDOWN_SECONDS = 5 * 60
    _ERROR_LOG_COOLDOWN_SECONDS = 15.0
    _DNS_PRECHECK_TIMEOUT_SECONDS = 1.0
    _DNS_PRECHECK_CACHE_SECONDS = 30.0

    def __init__(self):
        super().__init__()
        self._ws = None
        self._ws_client_installed = importlib.util.find_spec("websocket") is not None
        self._missing_dependency_logged = False
        self._ws_force_disabled = str(
            os.environ.get("TRADING_DISABLE_WEBSOCKET", "0")
        ).strip().lower() in ("1", "true", "yes", "on")
        self._ws_force_disable_logged = False
        self._allow_ws_on_vpn = str(
            os.environ.get("TRADING_ALLOW_WEBSOCKET_ON_VPN", "0")
        ).strip().lower() in ("1", "true", "yes", "on")
        self._network_block_logged = False
        self._ws_host = "push.sina.cn"
        self._reconnect_delay = 1
        self._max_reconnect_delay = 60
        self._reconnect_count = 0
        self._last_message_time: dict[str, datetime] = {}
        self._staleness_threshold = timedelta(seconds=30)
        self._heartbeat_interval = 10
        self._symbols: set[str] = set()
        self._reconnect_semaphore = threading.Semaphore(
            self._MAX_RECONNECT_THREADS
        )
        self._consecutive_dns_failures = 0
        self._ws_disabled_until_ts = 0.0
        self._ws_disabled_reason = ""
        self._last_ws_error_log_ts = 0.0
        self._last_dns_probe_host = ""
        self._last_dns_probe_ok = True
        self._last_dns_probe_ts = 0.0

    def supports_websocket(self) -> bool:
        if self._ws_force_disabled:
            return False
        if not self._ws_client_installed:
            return False
        return time.monotonic() >= float(self._ws_disabled_until_ts)

    def _is_dns_error(self, error: object) -> bool:
        txt = str(error or "").strip().lower()
        if not txt:
            return False
        patterns = (
            "getaddrinfo",
            "name or service not known",
            "temporary failure in name resolution",
            "nodename nor servname",
            "11001",
            "11004",
        )
        return any(p in txt for p in patterns)

    def _temporarily_disable(self, reason: str, cooldown_s: float | None = None):
        duration = max(60.0, float(cooldown_s or self._DNS_DISABLE_COOLDOWN_SECONDS))
        now = time.monotonic()
        until = now + duration
        prev_until = float(self._ws_disabled_until_ts)
        self._ws_disabled_until_ts = max(prev_until, until)
        self._ws_disabled_reason = str(reason or "unavailable")
        self._running = False
        self.status = FeedStatus.ERROR
        self._reconnect_count = 0
        self._reconnect_delay = 1
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if (prev_until - now) <= 1.0:
            mins = max(1, int(round(duration / 60.0)))
            log.warning(
                f"WebSocket temporarily disabled for {mins}m "
                f"({self._ws_disabled_reason}); using polling"
            )

    def _network_allows_websocket(self) -> tuple[bool, str]:
        """Check whether current network mode is compatible with WS endpoint."""
        if self._allow_ws_on_vpn:
            return True, ""
        try:
            from core.network import peek_network_env

            env = peek_network_env()
            if env is None:
                return True, ""
            if bool(getattr(env, "is_vpn_active", False)):
                return False, "network mode VPN_FOREIGN"
        except Exception:
            # If detector fails, do not block WS preemptively.
            return True, ""
        return True, ""

    @staticmethod
    def _resolve_host_with_timeout(host: str, timeout_s: float) -> bool | None:
        """
        Resolve host with a hard timeout.

        Returns:
            True/False when resolution completes within timeout,
            None when probe timed out (unknown).
        """
        result: dict[str, bool] = {"done": False, "ok": False}

        def _probe():
            try:
                socket.getaddrinfo(host, 443)
                result["ok"] = True
            except Exception:
                result["ok"] = False
            finally:
                result["done"] = True

        t = threading.Thread(target=_probe, daemon=True, name="ws_dns_probe")
        t.start()
        t.join(timeout=max(0.05, float(timeout_s)))
        if not bool(result.get("done", False)):
            return None
        return bool(result.get("ok", False))

    def _host_resolves(self, host: str) -> bool:
        """
        Best-effort DNS pre-check to avoid WS spin on invalid host.

        Timeout path returns True (unknown) to avoid startup/UI stalls; actual
        connect path will still fail over to polling if endpoint is unreachable.
        """
        now = time.monotonic()
        if (
            str(host) == self._last_dns_probe_host
            and (now - float(self._last_dns_probe_ts)) <= float(self._DNS_PRECHECK_CACHE_SECONDS)
        ):
            return bool(self._last_dns_probe_ok)

        probe = self._resolve_host_with_timeout(
            str(host),
            timeout_s=float(self._DNS_PRECHECK_TIMEOUT_SECONDS),
        )
        if probe is None:
            log.debug("WebSocket DNS pre-check timed out for host=%s", host)
            return True

        self._last_dns_probe_host = str(host)
        self._last_dns_probe_ok = bool(probe)
        self._last_dns_probe_ts = now
        return bool(probe)

    def connect(self) -> bool:
        if self._ws_force_disabled:
            if not self._ws_force_disable_logged:
                log.info("WebSocket disabled by TRADING_DISABLE_WEBSOCKET; using polling")
                self._ws_force_disable_logged = True
            return False

        allowed, reason = self._network_allows_websocket()
        if not allowed:
            if not self._network_block_logged:
                log.info(
                    f"WebSocket disabled for current network ({reason}); using polling"
                )
                self._network_block_logged = True
            self._temporarily_disable(
                reason=reason,
                cooldown_s=self._NETWORK_DISABLE_COOLDOWN_SECONDS,
            )
            return False
        self._network_block_logged = False

        if not self._host_resolves(self._ws_host):
            self._temporarily_disable(
                reason=f"dns failed for {self._ws_host}",
                cooldown_s=self._NETWORK_DISABLE_COOLDOWN_SECONDS,
            )
            return False

        if not self._ws_client_installed:
            if not self._missing_dependency_logged:
                log.warning(
                    "websocket-client not installed, using polling "
                    "(install with: pip install websocket-client)"
                )
                self._missing_dependency_logged = True
            return False

        if not self.supports_websocket():
            return False

        try:
            import websocket

            ws_url = f"wss://{self._ws_host}/ws"
            self._ws = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open,
            )
            self._ws_disabled_reason = ""
            if not self._running:
                self._reconnect_delay = 1
                self._reconnect_count = 0
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
            self._ws_client_installed = False
            if not self._missing_dependency_logged:
                log.warning(
                    "websocket-client not installed, using polling "
                    "(install with: pip install websocket-client)"
                )
                self._missing_dependency_logged = True
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
        self._consecutive_dns_failures = 0
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
        now = time.monotonic()
        if not self._running:
            # When feed is intentionally stopped/disabled, callbacks may still
            # arrive briefly from the previous socket/thread. Ignore to avoid
            # log spam and accidental reconnect state churn.
            if now < float(self._ws_disabled_until_ts):
                return
            if (now - self._last_ws_error_log_ts) >= self._ERROR_LOG_COOLDOWN_SECONDS:
                log.debug(f"WebSocket error while inactive: {error}")
                self._last_ws_error_log_ts = now
            return

        is_dns = self._is_dns_error(error)
        if is_dns:
            self._consecutive_dns_failures += 1
        else:
            self._consecutive_dns_failures = 0

        should_log = (
            self._consecutive_dns_failures <= 2
            or (now - self._last_ws_error_log_ts) >= self._ERROR_LOG_COOLDOWN_SECONDS
        )
        if should_log:
            log.warning(f"WebSocket error: {error}")
            self._last_ws_error_log_ts = now

        if (
            is_dns
            and self._consecutive_dns_failures >= self._DNS_FAILURE_DISABLE_THRESHOLD
        ):
            self._temporarily_disable(reason="DNS resolution failures")
            return

        self.status = FeedStatus.ERROR
        if not self._running:
            return
        try:
            if ws is not None:
                ws.close()
        except Exception:
            pass

    def _on_close(self, ws, close_status_code, close_msg):
        if (
            not self._running
            and time.monotonic() < float(self._ws_disabled_until_ts)
        ):
            self.status = FeedStatus.ERROR
            return

        if not self._running:
            self.status = FeedStatus.DISCONNECTED
            return

        self._reconnect_count += 1

        if self._reconnect_count > self._MAX_RECONNECT_ATTEMPTS:
            self._temporarily_disable(
                reason=(
                    f"reconnect attempts exceeded "
                    f"({self._MAX_RECONNECT_ATTEMPTS})"
                ),
                cooldown_s=self._DNS_DISABLE_COOLDOWN_SECONDS,
            )
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

class AggregatedFeed(DataFeed):
    """Aggregated data feed combining multiple sources."""

    name = "aggregated"

    def __init__(self):
        super().__init__()
        self._feeds: list[DataFeed] = []
        self._primary_feed: DataFeed | None = None
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

class VolumeMode(Enum):
    """Volume interpretation mode."""
    CUMULATIVE = "cumulative"
    DELTA = "delta"

# Bar aggregator - FIXED to emit partial bars

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
        self._current_bars: dict[str, dict] = {}
        self._callbacks: list[Callable] = []
        self._lock = threading.RLock()
        self._last_partial_emit_ts: dict[str, float] = {}
        self._min_partial_emit_interval_s: float = 0.20

    @staticmethod
    def _to_shanghai_naive(ts_raw) -> datetime:
        """
        Normalize quote timestamps to naive Asia/Shanghai time.
        """
        try:
            from zoneinfo import ZoneInfo

            sh_tz = ZoneInfo("Asia/Shanghai")
        except Exception:
            sh_tz = timezone.utc

        if ts_raw is None:
            return datetime.now(tz=sh_tz).replace(tzinfo=None)

        if isinstance(ts_raw, datetime):
            dt = ts_raw
        else:
            text = str(ts_raw).strip()
            if not text:
                return datetime.now(tz=sh_tz).replace(tzinfo=None)
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except Exception:
                return datetime.now(tz=sh_tz).replace(tzinfo=None)

        try:
            if dt.tzinfo is None:
                # Treat naive provider timestamps as market-local time.
                dt = dt.replace(tzinfo=sh_tz)
            dt = dt.astimezone(sh_tz)
            return dt.replace(tzinfo=None)
        except Exception:
            try:
                return dt.replace(tzinfo=None)
            except Exception:
                return datetime.now(tz=sh_tz).replace(tzinfo=None)

    @staticmethod
    def _is_cn_session_time(ts_val: datetime) -> bool:
        """
        Check whether a timestamp is within CN A-share regular trading session.
        """
        if not isinstance(ts_val, datetime):
            return False
        # Monday..Friday
        if ts_val.weekday() >= 5:
            return False
        hhmm = (ts_val.hour * 100) + ts_val.minute
        morning = 930 <= hhmm <= 1130
        afternoon = 1300 <= hhmm <= 1500
        return bool(morning or afternoon)

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

        ts = self._to_shanghai_naive(getattr(quote, "timestamp", None))
        try:
            quote.timestamp = ts
        except Exception:
            pass
        px = float(getattr(quote, "price", 0) or 0)
        if px <= 0:
            return

        with self._lock:
            # Ignore off-session ticks, but flush any pending live bar first.
            if not self._is_cn_session_time(ts):
                stale = self._current_bars.pop(symbol, None)
                if stale is not None:
                    try:
                        self._emit_bar(symbol, stale, final=True)
                    except Exception:
                        pass
                self._last_partial_emit_ts.pop(symbol, None)
                return

            if symbol not in self._current_bars:
                self._current_bars[symbol] = self._new_bar(quote)

            bar = self._current_bars[symbol]

            # Day boundary: finalize old day bar before touching current tick.
            if (
                bar.get("session_date")
                and bar["session_date"] != ts.date()
            ):
                self._emit_bar(symbol, bar, final=True)
                self._current_bars[symbol] = self._new_bar(quote)
                bar = self._current_bars[symbol]

            bar_end = bar["timestamp"] + timedelta(seconds=self._interval)
            if ts >= bar_end:
                # CRITICAL: roll first, then apply tick to new bucket.
                self._emit_bar(symbol, bar, final=True)
                self._current_bars[symbol] = self._new_bar(quote)
                bar = self._current_bars[symbol]
                self._last_partial_emit_ts.pop(symbol, None)

            bar["high"] = max(float(bar["high"]), px)
            bar["low"] = min(float(bar["low"]), px)
            bar["close"] = px

            self._update_volume(bar, quote)

            # Emit partial bar for real-time chart updates.
            now_ts = time.monotonic()
            last_emit = float(self._last_partial_emit_ts.get(symbol, 0.0))
            if (now_ts - last_emit) >= self._min_partial_emit_interval_s:
                self._emit_bar(symbol, bar, final=False)
                self._last_partial_emit_ts[symbol] = now_ts

    def _update_volume(self, bar: dict, quote):
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

    def _new_bar(self, quote) -> dict:
        """Create a new bar with proper time alignment."""
        ts = self._to_shanghai_naive(getattr(quote, "timestamp", None))

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

    @staticmethod
    def _interval_label(interval_seconds: int) -> str:
        """Canonical interval token used by UI/cache layers."""
        sec = max(1, int(interval_seconds))
        known = {
            60: "1m",
            300: "5m",
            900: "15m",
            1800: "30m",
            3600: "60m",
            86400: "1d",
        }
        if sec in known:
            return known[sec]
        if sec % 60 == 0:
            mins = int(sec // 60)
            return f"{mins}m"
        return f"{sec}s"

    def set_interval(self, interval_seconds: int):
        """Change bar interval; clears partial bars."""
        with self._lock:
            self._interval = max(1, int(interval_seconds))
            self._current_bars.clear()
            self._last_partial_emit_ts.clear()

    def _emit_bar(self, symbol: str, bar: dict, final: bool = True):
        """
        Emit bar to callbacks.

        Args:
            symbol: Stock symbol
            bar: Bar data dict
            final: If True, this is a completed bar. If False, partial/live bar.
        """
        bar_copy = dict(bar)
        interval_label = self._interval_label(self._interval)
        bar_copy["interval"] = interval_label
        bar_copy["interval_seconds"] = int(self._interval)
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

            try:
                from data.session_cache import get_session_bar_cache

                cache_bar = dict(bar_copy)
                get_session_bar_cache().append_bar(
                    symbol,
                    interval_label,
                    cache_bar,
                )
            except Exception as e:
                log.debug(f"Bar session persist failed for {symbol}: {e}")

            should_persist_db = not bool(CONFIG.is_market_open())
            if should_persist_db:
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

                    db.upsert_intraday_bars(symbol, interval_label, df)
                except ImportError:
                    pass
                except Exception as e:
                    log.debug(f"Bar DB persist failed for {symbol}: {e}")

        with self._lock:
            callbacks = self._callbacks.copy()

        for cb in callbacks:
            try:
                cb(symbol, bar_copy)
            except Exception as e:
                log.warning(f"Bar callback error: {e}")

# Feed manager (singleton)

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
        self._feeds: dict[str, DataFeed] = {}
        self._active_feed: DataFeed | None = None
        self._subscriptions: set[str] = set()
        self._tick_callbacks: list[Callable] = []
        self._bar_aggregator = BarAggregator()
        self._last_quotes: dict[str, object] = {}
        self._quotes_lock = threading.RLock()
        self._lock = threading.RLock()
        self._initialized_runtime = False
        self._health_thread: threading.Thread | None = None
        self._health_running: bool = False
        self._health_poll_interval_s: float = 3.0
        self._last_feed_switch_ts: float = 0.0
        self._feed_switch_cooldown_s: float = 8.0
        self._health_generation: int = 0

    def initialize(self, force: bool = False):
        """Initialize feeds. Idempotent unless force=True."""
        with self._lock:
            if self._initialized_runtime and not force:
                return

            old_feeds = list(self._feeds.values()) if force else []
            old_callbacks: list[Callable] = []
            if (
                hasattr(self, "_bar_aggregator")
                and self._bar_aggregator
            ):
                with self._bar_aggregator._lock:
                    old_callbacks = self._bar_aggregator._callbacks.copy()
            self._initialized_runtime = True
            self._feeds = {}
            self._active_feed = None

        self._stop_health_watchdog()
        for feed in old_feeds:
            try:
                feed.disconnect()
            except Exception:
                pass

        interval = float(CONFIG.data.poll_interval_seconds)
        bar_seconds = 60

        polling = PollingFeed(interval=interval)
        polling_ok = False
        try:
            polling_ok = bool(polling.connect())
        except Exception as e:
            log.warning(f"Polling feed connect failed: {e}")
        self._feeds["polling"] = polling

        ws_ok = False
        ws = WebSocketFeed()
        self._feeds["websocket"] = ws
        try:
            ws_ok = bool(ws.connect())
        except Exception as e:
            log.warning(f"WebSocket feed connect failed: {e}")

        with self._lock:
            self._bar_aggregator = BarAggregator(
                interval_seconds=bar_seconds
            )
            for cb in old_callbacks:
                self._bar_aggregator.add_callback(cb)

            if ws_ok:
                self._active_feed = ws
            else:
                self._active_feed = polling if polling_ok else None
            self._last_feed_switch_ts = time.monotonic()

        self._attach_active_callbacks()
        self._resubscribe_active()
        self._start_health_watchdog()

        active_name = self._active_feed.name if self._active_feed else "none"
        log.info(
            f"Feed manager initialized "
            f"(primary={active_name}, poll={interval}s, bar={bar_seconds}s)"
        )

    def _attach_active_callbacks(self) -> None:
        """Bind cache/tick/bar callbacks to the currently active feed only."""
        for feed in list(self._feeds.values()):
            try:
                feed.remove_callback(self._cache_quote)
            except Exception:
                pass
            try:
                feed.remove_callback(self._bar_aggregator.on_tick)
            except Exception:
                pass
            for cb in list(self._tick_callbacks):
                try:
                    feed.remove_callback(cb)
                except Exception:
                    pass

        feed = self._active_feed
        if feed is None:
            return
        try:
            feed.add_callback(self._cache_quote)
        except Exception:
            pass
        try:
            feed.add_callback(self._bar_aggregator.on_tick)
        except Exception:
            pass
        for cb in list(self._tick_callbacks):
            try:
                feed.add_callback(cb)
            except Exception:
                pass

    def _resubscribe_active(self) -> None:
        feed = self._active_feed
        if feed is None:
            return
        with self._lock:
            symbols = list(self._subscriptions)
        for sym in symbols:
            try:
                feed.subscribe(sym)
            except Exception:
                continue

    def _is_feed_healthy(self, feed: DataFeed | None) -> bool:
        if feed is None:
            return False
        if isinstance(feed, PollingFeed):
            return bool(feed.status == FeedStatus.CONNECTED and feed._running)
        if isinstance(feed, WebSocketFeed):
            if feed.status != FeedStatus.CONNECTED or not feed._running:
                return False
            with self._lock:
                symbols = list(self._subscriptions)
            if not symbols:
                return True
            risk = getattr(CONFIG, "risk", None)
            stale_max = float(
                getattr(risk, "quote_staleness_seconds", 8.0) or 8.0
            )
            stale_limit = max(6.0, min(30.0, stale_max * 1.6))
            sample = symbols[: min(12, len(symbols))]
            stale = 0
            for sym in sample:
                try:
                    age = float(feed.get_staleness(sym))
                except Exception:
                    age = float("inf")
                if age > stale_limit:
                    stale += 1
            return (stale / max(1, len(sample))) < 0.60
        return feed.status == FeedStatus.CONNECTED

    def _activate_feed(
        self, name: str, reason: str = "", ignore_cooldown: bool = False
    ) -> bool:
        with self._lock:
            target = self._feeds.get(str(name))
            current = self._active_feed
            if target is None:
                return False
            if current is target:
                return True
            now = time.monotonic()
            if (
                (not ignore_cooldown)
                and (now - self._last_feed_switch_ts) < self._feed_switch_cooldown_s
            ):
                return False

        if target.status != FeedStatus.CONNECTED:
            try:
                if not target.connect():
                    return False
            except Exception:
                return False

        with self._lock:
            self._active_feed = target
            self._last_feed_switch_ts = time.monotonic()
        self._attach_active_callbacks()
        self._resubscribe_active()
        msg = f"Feed failover: active={target.name}"
        if reason:
            msg += f" ({reason})"
        log.info(msg)
        return True

    def _start_health_watchdog(self) -> None:
        with self._lock:
            if (
                self._health_running
                and self._health_thread is not None
                and self._health_thread.is_alive()
            ):
                return
            self._health_generation += 1
            generation = int(self._health_generation)
            self._health_running = True
            t = threading.Thread(
                target=self._health_loop,
                args=(generation,),
                daemon=True,
                name="feed_health_watchdog",
            )
            self._health_thread = t
        t.start()

    def _stop_health_watchdog(self) -> None:
        with self._lock:
            self._health_running = False
            self._health_generation += 1
            t = self._health_thread
            self._health_thread = None
        if t is not None:
            try:
                t.join(timeout=max(2.0, float(self._health_poll_interval_s) + 1.0))
            except Exception:
                pass

    def _health_loop(self, generation: int) -> None:
        """Monitor feed health and switch between websocket/polling as needed."""
        while True:
            with self._lock:
                if (
                    not self._health_running
                    or generation != int(self._health_generation)
                ):
                    return
                active = self._active_feed
                ws = self._feeds.get("websocket")
                polling = self._feeds.get("polling")

            try:
                if isinstance(active, WebSocketFeed):
                    if (not self._is_feed_healthy(active)) and polling is not None:
                        self._activate_feed(
                            "polling",
                            reason="websocket stale/unhealthy",
                            ignore_cooldown=True,
                        )
                elif isinstance(active, PollingFeed):
                    if isinstance(ws, WebSocketFeed):
                        if (
                            ws.status != FeedStatus.CONNECTED
                            and not ws._running
                            and ws.supports_websocket()
                        ):
                            try:
                                ws.connect()
                            except Exception:
                                pass
                        cooldown_ok = (
                            time.monotonic() - self._last_feed_switch_ts
                        ) >= (self._feed_switch_cooldown_s * 2.0)
                        if cooldown_ok and self._is_feed_healthy(ws):
                            self._activate_feed("websocket", reason="websocket healthy")
                else:
                    if polling is not None and self._is_feed_healthy(polling):
                        self._activate_feed(
                            "polling",
                            reason="no active feed",
                            ignore_cooldown=True,
                        )
            except Exception as e:
                log.debug(f"Feed health watchdog error: {e}")

            sleep_for = max(1.0, float(self._health_poll_interval_s))
            deadline = time.monotonic() + sleep_for
            while True:
                with self._lock:
                    if (
                        not self._health_running
                        or generation != int(self._health_generation)
                    ):
                        return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.25, remaining))

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
        sym = str(symbol)
        with self._lock:
            if sym in self._subscriptions:
                return True
            self._subscriptions.add(sym)
            active = self._active_feed

        ok = False
        if active is not None:
            try:
                ok = bool(active.subscribe(sym))
            except Exception:
                ok = False

        if not ok:
            with self._lock:
                backups = [
                    f for f in self._feeds.values()
                    if f is not active
                ]
            for feed in backups:
                try:
                    if feed.subscribe(sym):
                        ok = True
                        break
                except Exception:
                    continue

        if not ok:
            with self._lock:
                self._subscriptions.discard(sym)
        return ok

    def unsubscribe(self, symbol: str):
        sym = str(symbol)
        with self._lock:
            if sym not in self._subscriptions:
                return
            self._subscriptions.discard(sym)
            feeds = list(self._feeds.values())

        for feed in feeds:
            try:
                feed.unsubscribe(sym)
            except Exception:
                continue

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

    def subscribe_many(self, symbols: list[str]):
        for symbol in symbols:
            self.subscribe(symbol)

    def get_quote(self, symbol: str) -> object | None:
        sym = str(symbol)
        with self._quotes_lock:
            q = self._last_quotes.get(sym)
            if q:
                return q

        active = self._active_feed
        if isinstance(active, PollingFeed):
            q = active.get_quote(sym)
            if q is not None:
                return q
        poll = self._feeds.get("polling")
        if isinstance(poll, PollingFeed):
            return poll.get_quote(sym)
        return None

    def get_last_quote_time(self, symbol: str) -> datetime | None:
        quote = self.get_quote(symbol)
        if quote and hasattr(quote, "timestamp"):
            return quote.timestamp
        return None

    def add_tick_callback(self, callback: Callable):
        with self._lock:
            if callback not in self._tick_callbacks:
                self._tick_callbacks.append(callback)
            active = self._active_feed
        if active:
            try:
                active.add_callback(callback)
            except Exception:
                pass

    def add_bar_callback(self, callback: Callable):
        self._bar_aggregator.add_callback(callback)

    def shutdown(self):
        """Shutdown all feeds and reset state."""
        self._stop_health_watchdog()

        feeds = list(self._feeds.values())
        for feed in feeds:
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

# Module-level singleton accessor

_feed_manager: FeedManager | None = None
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
