# trading/broker_live.py
from __future__ import annotations

import time
from abc import abstractmethod
from collections import deque
from datetime import date, datetime
from pathlib import Path

from config.settings import CONFIG
from core.types import Account, Fill, Order, OrderSide, OrderStatus, OrderType, Position
from utils.logger import get_logger

from .broker import BrokerInterface, make_fill_uid, parse_broker_status

log = get_logger(__name__)

class EasytraderBroker(BrokerInterface):
    """
    Base class for all easytrader-based brokers (THS, ZSZQ, HT, etc).

    Subclasses only need to override:
    - name (property)
    - _get_easytrader_type() -> str
    - _get_balance_fields() -> dict (optional, for field name differences)
    """

    def __init__(self):
        super().__init__()
        self._client = None
        self._connected = False
        self._orders: dict[str, Order] = {}
        self._seen_fill_ids: set = set()
        self._fetcher = None

        try:
            import easytrader
            self._easytrader = easytrader
            self._available = True
        except ImportError:
            self._easytrader = None
            self._available = False
            log.warning(
                "easytrader not installed - live trading unavailable"
            )

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _get_easytrader_type(self) -> str:
        """Return the easytrader.use() type string."""
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    def connect(self, exe_path: str = None, **kwargs) -> bool:
        if not self._available:
            log.error("easytrader not installed")
            return False

        exe_path = (
            exe_path
            or kwargs.get('broker_path')
            or CONFIG.broker_path
        )
        if not exe_path or not Path(exe_path).exists():
            log.error(f"Broker executable not found: {exe_path}")
            return False

        try:
            self._client = self._easytrader.use(
                self._get_easytrader_type(),
            )
            self._client.connect(exe_path)

            balance = self._client.balance
            if balance:
                self._connected = True
                log.info(f"Connected to {self.name}")
                return True

        except Exception as e:
            log.error(f"Connection failed: {e}")

        return False

    def disconnect(self):
        self._client = None
        self._connected = False
        log.info(f"Disconnected from {self.name}")

    def get_quote(self, symbol: str) -> float | None:
        try:
            from data.feeds import get_feed_manager
            fm = get_feed_manager(auto_init=False)
            q = fm.get_quote(symbol)
            if q and getattr(q, "price", 0) and float(q.price) > 0:
                return float(q.price)
        except Exception as e:
            log.debug("Easytrader feed quote unavailable for %s: %s", symbol, e)

        try:
            fetcher = self._get_fetcher()
            quote = fetcher.get_realtime(symbol)
            return float(quote.price) if quote and quote.price > 0 else None
        except Exception as e:
            log.debug("Easytrader realtime quote unavailable for %s: %s", symbol, e)
            return None

    def get_account(self) -> Account:
        if not self.is_connected:
            return Account()

        try:
            balance = self._client.balance
            positions = self.get_positions()

            cash = float(
                balance.get("\u8d44\u91d1\u4f59\u989d")  # 璧勯噾浣欓
                or balance.get("\u603b\u8d44\u4ea7")  # 鎬昏祫浜?
                or balance.get("\u53ef\u7528\u8d44\u91d1")  # 鍙敤璧勯噾
                or balance.get("cash")
                or 0
            )
            available = float(
                balance.get("\u53ef\u7528\u91d1\u989d")  # 鍙敤閲戦
                or balance.get("\u53ef\u7528\u8d44\u91d1")  # 鍙敤璧勯噾
                or balance.get("\u53ef\u53d6\u8d44\u91d1")  # 鍙彇璧勯噾
                or balance.get("available")
                or cash
            )
            frozen = float(
                balance.get("\u51bb\u7ed3\u91d1\u989d")  # 鍐荤粨閲戦
                or balance.get("frozen")
                or 0
            )

            return Account(
                broker_name=self.name,
                cash=cash,
                available=available,
                frozen=frozen,
                positions=positions,
                last_updated=datetime.now(),
            )
        except Exception as e:
            log.error(f"Failed to get account: {e}")
            return Account()

    def get_positions(self) -> dict[str, Position]:
        if not self.is_connected:
            return {}

        try:
            raw = self._client.position
            positions = {}

            for p in raw:
                code = str(
                    p.get("\u8bc1\u5238\u4ee3\u7801")  # 璇佸埜浠ｇ爜
                    or p.get("\u80a1\u7968\u4ee3\u7801")  # 鑲＄エ浠ｇ爜
                    or ""
                ).zfill(6)

                if not code or code == "000000":
                    continue

                positions[code] = Position(
                    symbol=code,
                    name=(
                        p.get("\u8bc1\u5238\u540d\u79f0")  # 璇佸埜鍚嶇О
                        or p.get("\u80a1\u7968\u540d\u79f0")  # 鑲＄エ鍚嶇О
                        or ""
                    ),
                    quantity=int(
                        p.get("\u80a1\u7968\u4f59\u989d")  # 鑲＄エ浣欓
                        or p.get("\u6301\u4ed3\u6570\u91cf")  # 鎸佷粨鏁伴噺
                        or p.get("\u5f53\u524d\u6301\u4ed3")  # 褰撳墠鎸佷粨
                        or 0
                    ),
                    available_qty=int(
                        p.get("\u53ef\u5356\u4f59\u989d")  # 鍙崠浣欓
                        or p.get("\u53ef\u7528\u4f59\u989d")  # 鍙敤浣欓
                        or p.get("\u53ef\u5356\u6570\u91cf")  # 鍙崠鏁伴噺
                        or 0
                    ),
                    avg_cost=float(
                        p.get("\u6210\u672c\u4ef7")  # 鎴愭湰浠?
                        or p.get("\u4e70\u5165\u6210\u672c")  # 涔板叆鎴愭湰
                        or p.get("\u53c2\u8003\u6210\u672c\u4ef7")  # 鍙傝€冩垚鏈环
                        or 0
                    ),
                    current_price=float(
                        p.get("\u5f53\u524d\u4ef7")  # 褰撳墠浠?
                        or p.get("\u6700\u65b0\u4ef7")  # 鏈€鏂颁环
                        or p.get("\u5e02\u4ef7")  # 甯備环
                        or 0
                    ),
                )

            return positions

        except Exception as e:
            log.error(f"Failed to get positions: {e}")
            return {}

    def get_position(self, symbol: str) -> Position | None:
        return self.get_positions().get(symbol)

    def submit_order(self, order: Order) -> Order:
        if not self.is_connected:
            order.status = OrderStatus.REJECTED
            order.message = "Not connected to broker"
            return order

        try:
            order.created_at = order.created_at or datetime.now()
            order.submitted_at = datetime.now()

            if order.side == OrderSide.BUY:
                if order.order_type == OrderType.MARKET:
                    result = self._client.market_buy(
                        order.symbol, order.quantity,
                    )
                else:
                    result = self._client.buy(
                        order.symbol, order.quantity, order.price,
                    )
            else:
                if order.order_type == OrderType.MARKET:
                    result = self._client.market_sell(
                        order.symbol, order.quantity,
                    )
                else:
                    result = self._client.sell(
                        order.symbol, order.quantity, order.price,
                    )

            if result and isinstance(result, dict):
                entrust_no = (
                    result.get("\u59d4\u6258\u7f16\u53f7")
                    or result.get('entrust_no')
                    or result.get('order_id')
                )

                if entrust_no:
                    order.status = OrderStatus.SUBMITTED
                    order.broker_id = str(entrust_no)
                    order.message = f"Entrust: {entrust_no}"
                    self.register_order_mapping(
                        order.id, order.broker_id,
                    )
                    log.info(
                        f"Order submitted: {order.id} "
                        f"-> broker {order.broker_id}"
                    )
                else:
                    order.status = OrderStatus.REJECTED
                    order.message = str(
                        result.get('msg')
                        or result.get('message')
                        or result
                    )
            else:
                order.status = OrderStatus.REJECTED
                order.message = "Unknown response from broker"

            self._orders[order.id] = order
            self._emit('order_update', order)
            return order

        except Exception as e:
            log.error(f"Order submission error: {e}")
            order.status = OrderStatus.REJECTED
            order.message = str(e)
            return order

    def get_fills(self, since: datetime = None) -> list[Fill]:
        """Get fills from broker - deduplicates by broker_fill_id."""
        if not self.is_connected:
            return []

        fills: list[Fill] = []
        try:
            trades = self._client.today_trades

            for trade in trades:
                broker_fill_id = str(
                    trade.get("\u6210\u4ea4\u7f16\u53f7", "") or ""  # 鎴愪氦缂栧彿
                ).strip()
                if not broker_fill_id:
                    continue

                if broker_fill_id in self._seen_fill_ids:
                    continue

                ts = (
                    trade.get("\u6210\u4ea4\u65f6\u95f4")  # 鎴愪氦鏃堕棿
                    or trade.get("time")
                    or None
                )
                fill_time = datetime.now()
                if ts:
                    try:
                        t = datetime.strptime(
                            str(ts), "%H:%M:%S"
                        ).time()
                        fill_time = datetime.combine(
                            date.today(), t,
                        )
                    except Exception as e:
                        log.debug("Fill timestamp parse failed for %r: %s", ts, e)

                # since-filter
                if (
                    since
                    and isinstance(fill_time, datetime)
                    and fill_time < since
                ):
                    continue

                broker_entrust = str(
                    trade.get("\u59d4\u6258\u7f16\u53f7", "") or ""  # 濮旀墭缂栧彿
                ).strip()
                our_order_id = self.get_order_id(broker_entrust)
                if not our_order_id:
                    log.warning(
                        f"Unknown entrust number: {broker_entrust}"
                    )
                    continue

                self._seen_fill_ids.add(broker_fill_id)

                trade_side = trade.get(
                    "\u4e70\u5356\u6807\u5fd7", trade.get("\u64cd\u4f5c", ""),  # 涔板崠鏍囧織 / 鎿嶄綔
                )
                side = (
                    OrderSide.BUY
                    if "\u4e70" in str(trade_side)  # 涔?
                    else OrderSide.SELL
                )

                symbol = str(
                    trade.get("\u8bc1\u5238\u4ee3\u7801", "") or ""  # 璇佸埜浠ｇ爜
                ).zfill(6)
                qty = int(trade.get("\u6210\u4ea4\u6570\u91cf", 0) or 0)  # 鎴愪氦鏁伴噺
                price = float(trade.get("\u6210\u4ea4\u4ef7\u683c", 0) or 0.0)  # 鎴愪氦浠锋牸
                comm = float(trade.get("\u624b\u7eed\u8d39", 0) or 0.0)  # 鎵嬬画璐?
                tax = float(trade.get("\u5370\u82b1\u7a0e", 0) or 0.0)  # 鍗拌姳绋?

                fid = make_fill_uid(
                    self.name, broker_fill_id, symbol,
                    fill_time, price, qty,
                )

                fills.append(Fill(
                    id=fid,
                    broker_fill_id=broker_fill_id,
                    order_id=our_order_id,
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    price=price,
                    commission=comm,
                    stamp_tax=tax,
                    timestamp=fill_time,
                ))

        except Exception as e:
            log.error(f"Failed to get fills: {e}")

        return fills

    def get_order_status(
        self, order_id: str,
    ) -> OrderStatus | None:
        if not self.is_connected:
            return None

        broker_id = self.get_broker_id(order_id)
        if not broker_id:
            return None

        try:
            entrusts = self._client.today_entrusts

            for entrust in entrusts:
                if str(entrust.get("\u59d4\u6258\u7f16\u53f7", "")) == broker_id:
                    status_str = entrust.get(
                        "\u59d4\u6258\u72b6\u6001",  # 濮旀墭鐘舵€?
                        entrust.get("\u72b6\u6001", ""),  # 鐘舵€?
                    )
                    # FIX(3): Use shared parser
                    return parse_broker_status(status_str)

            return None

        except Exception as e:
            log.error(f"Failed to get order status: {e}")
            return None

    def sync_order(self, order: Order) -> Order:
        if not self.is_connected:
            return order

        broker_id = self.get_broker_id(order.id)
        if not broker_id:
            return order

        try:
            entrusts = self._client.today_entrusts

            for entrust in entrusts:
                if str(entrust.get("\u59d4\u6258\u7f16\u53f7", "")) == broker_id:
                    # FIX(3): Use shared parser
                    order.status = parse_broker_status(
                        entrust.get("\u59d4\u6258\u72b6\u6001", ""),  # 濮旀墭鐘舵€?
                    )
                    order.filled_qty = int(
                        entrust.get("\u6210\u4ea4\u6570\u91cf", 0) or 0  # 鎴愪氦鏁伴噺
                    )

                    avg_price = entrust.get(
                        "\u6210\u4ea4\u5747\u4ef7",  # 鎴愪氦鍧囦环
                        entrust.get("\u6210\u4ea4\u4ef7\u683c", 0),  # 鎴愪氦浠锋牸
                    )
                    if avg_price:
                        order.avg_price = float(avg_price)

                    order.updated_at = datetime.now()
                    break

        except Exception as e:
            log.error(f"Failed to sync order: {e}")

        return order

    def cancel_order(self, order_id: str) -> bool:
        if not self.is_connected:
            return False

        order = self._orders.get(order_id)
        broker_id = self.get_broker_id(order_id)

        if not broker_id:
            return False

        try:
            self._client.cancel_entrust(broker_id)
            if order:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                self._emit('order_update', order)
            log.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            log.error(f"Cancel failed: {e}")
            return False

    def get_orders(self, active_only: bool = True) -> list[Order]:
        if active_only:
            return [
                o for o in self._orders.values() if o.is_active
            ]
        return list(self._orders.values())

    def _get_fetcher(self):
        if self._fetcher is None:
            from data.fetcher import get_fetcher
            self._fetcher = get_fetcher()
        return self._fetcher

# FIX(11): Thin subclasses - all shared logic is in base

class THSBroker(EasytraderBroker):
    """THS/HT/GJ/YH broker via easytrader."""

    BROKER_TYPES = {
        "ths": "THS",
        "ht": "HT",
        "gj": "GJ",
        "yh": "YH",
    }

    def __init__(self, broker_type: str = "ths"):
        self._broker_type = broker_type
        super().__init__()

    @property
    def name(self) -> str:
        return self.BROKER_TYPES.get(
            self._broker_type, "Unknown Broker",
        )

    def _get_easytrader_type(self) -> str:
        return self._broker_type

class ZSZQBroker(EasytraderBroker):
    """ZSZQ broker via easytrader (universal mode)."""

    @property
    def name(self) -> str:
        return "ZSZQ"

    def _get_easytrader_type(self) -> str:
        return 'universal'


class MultiVenueBroker(BrokerInterface):
    """
    Multi-venue router with active failover.

    - Uses a priority list of underlying brokers.
    - Routes writes to the active venue.
    - On write failure, rotates to next venue with cooldown.
    """

    def __init__(self, venues: list[BrokerInterface], failover_cooldown_seconds: int = 30):
        super().__init__()
        self._venues = [v for v in venues if v is not None]
        self._active_idx = 0
        self._cooldown_seconds = max(1, int(failover_cooldown_seconds or 30))
        self._last_fail_ts: dict[int, float] = {}
        self._fail_counts: dict[str, int] = {}
        self._submit_counts: dict[str, int] = {}
        self._read_counts: dict[str, int] = {}
        self._failure_events: dict[int, deque[float]] = {}
        self._read_latency_ms: dict[str, deque[float]] = {}
        self._last_errors: dict[str, str] = {}
        self._latency_samples_max: int = 200
        self._recent_failure_window_seconds: float = 300.0
        self._order_venue_idx: dict[str, int] = {}
        if not self._venues:
            raise ValueError("MultiVenueBroker requires at least one venue")

    @property
    def name(self) -> str:
        names = ",".join(v.name for v in self._venues)
        return f"MultiVenueRouter[{names}]"

    @property
    def is_connected(self) -> bool:
        return any(v.is_connected for v in self._venues)

    def connect(self, **kwargs) -> bool:
        ok_any = False
        for i, venue in enumerate(self._venues):
            try:
                ok = bool(venue.connect(**kwargs))
            except Exception as e:
                ok = False
                log.warning("Venue connect failed (%s): %s", venue.name, e)
            if ok:
                ok_any = True
                if self._active_idx == 0:
                    self._active_idx = i
        return ok_any

    def disconnect(self):
        for venue in self._venues:
            try:
                venue.disconnect()
            except Exception as e:
                log.debug("Venue disconnect failed (%s): %s", venue.name, e)

    def _eligible_indices(self) -> list[int]:
        now = time.time()
        out: list[int] = []
        for i, venue in enumerate(self._venues):
            if not venue.is_connected:
                continue
            last_fail = float(self._last_fail_ts.get(i, 0.0))
            if last_fail > 0 and (now - last_fail) < self._cooldown_seconds:
                continue
            out.append(i)
        return out

    def _ordered_indices(self) -> list[int]:
        eligible = self._eligible_indices()
        if not eligible:
            return []
        return sorted(
            eligible,
            key=lambda i: self._venue_score(i),
            reverse=True,
        )

    def _connected_indices(self) -> list[int]:
        out: list[int] = []
        for i, venue in enumerate(self._venues):
            try:
                if venue.is_connected:
                    out.append(i)
            except Exception as e:
                log.debug("Venue connectivity probe failed at index %s: %s", i, e)
                continue
        return out

    def _preferred_indices_for_order(self, order_id: str) -> list[int]:
        preferred = self._order_venue_idx.get(str(order_id or ""))
        if preferred is None:
            return self._ordered_indices()

        ordered = self._ordered_indices()
        if preferred in ordered:
            return [preferred] + [i for i in ordered if i != preferred]
        if 0 <= preferred < len(self._venues):
            return [preferred] + ordered
        return ordered

    def _mark_failure(self, idx: int, exc: Exception) -> None:
        self._last_fail_ts[idx] = time.time()
        venue = self._venues[idx]
        name = str(venue.name)
        self._fail_counts[name] = self._fail_counts.get(name, 0) + 1
        bucket = self._failure_events.get(idx)
        if bucket is None:
            bucket = deque()
            self._failure_events[idx] = bucket
        now = time.time()
        bucket.append(now)
        cutoff = now - float(self._recent_failure_window_seconds)
        while bucket and float(bucket[0]) < cutoff:
            bucket.popleft()
        self._last_errors[name] = str(exc)[:300]
        log.warning("Venue failure (%s): %s", venue.name, exc)

    def _mark_submit(self, idx: int) -> None:
        venue = self._venues[idx]
        self._submit_counts[venue.name] = self._submit_counts.get(venue.name, 0) + 1

    def _mark_read(self, idx: int, latency_ms: float | None = None) -> None:
        venue = self._venues[idx]
        name = str(venue.name)
        self._read_counts[name] = self._read_counts.get(name, 0) + 1
        if latency_ms is not None and latency_ms >= 0:
            hist = self._read_latency_ms.get(name)
            if hist is None:
                hist = deque(maxlen=self._latency_samples_max)
                self._read_latency_ms[name] = hist
            hist.append(float(latency_ms))

    def _recent_failures(self, idx: int, window_seconds: float | None = None) -> int:
        bucket = self._failure_events.get(int(idx))
        if not bucket:
            return 0
        window = float(window_seconds or self._recent_failure_window_seconds)
        cutoff = time.time() - max(1.0, window)
        while bucket and float(bucket[0]) < cutoff:
            bucket.popleft()
        return int(len(bucket))

    def _avg_read_latency_ms(self, idx: int) -> float:
        if not (0 <= idx < len(self._venues)):
            return 0.0
        name = str(self._venues[idx].name)
        vals = self._read_latency_ms.get(name)
        if not vals:
            return 0.0
        return float(sum(vals) / max(1, len(vals)))

    def _venue_score(self, idx: int) -> float:
        """
        Adaptive routing score.

        Higher is better:
        - rewards venues with successful submits/reads
        - penalizes recent failures and active cooldown
        - slight preference for current active venue to reduce thrash
        """
        if not (0 <= idx < len(self._venues)):
            return -1.0

        venue = self._venues[idx]
        name = str(venue.name)
        fails = float(self._fail_counts.get(name, 0))
        submits = float(self._submit_counts.get(name, 0))
        reads = float(self._read_counts.get(name, 0))
        total_ops = submits + reads
        reliability = (total_ops + 1.0) / (total_ops + fails + 1.0)

        now = time.time()
        last_fail = float(self._last_fail_ts.get(idx, 0.0))
        cooldown_penalty = 0.0
        if last_fail > 0:
            elapsed = now - last_fail
            remain = max(0.0, float(self._cooldown_seconds) - elapsed)
            cooldown_penalty = min(0.5, remain / max(1.0, float(self._cooldown_seconds)))

        recent_failures = float(self._recent_failures(idx, window_seconds=300.0))
        recent_fail_penalty = min(0.40, recent_failures * 0.06)

        latency_ms = float(self._avg_read_latency_ms(idx))
        latency_penalty = 0.0
        if latency_ms > 120.0:
            latency_penalty = min(0.25, (latency_ms - 120.0) / 1200.0)

        read_bonus = min(0.04, reads / 600.0)
        active_bonus = 0.03 if idx == self._active_idx else 0.0
        return float(
            reliability
            + read_bonus
            + active_bonus
            - cooldown_penalty
            - recent_fail_penalty
            - latency_penalty
        )

    @staticmethod
    def _is_transient_reject(order: Order) -> bool:
        """
        Detect infrastructure-style rejects that should trigger failover.

        Business rejects (insufficient funds, rule violations, etc.) should
        not fan out to other venues.
        """
        if getattr(order, "status", None) != OrderStatus.REJECTED:
            return False

        msg = str(getattr(order, "message", "") or "").lower()
        if not msg:
            return False

        transient_markers = (
            "not connected",
            "timeout",
            "timed out",
            "network",
            "connection",
            "temporar",
            "unavailable",
            "service down",
            "gateway",
            "try again",
        )
        return any(marker in msg for marker in transient_markers)

    def _first_read(self, fn_name: str, *args, **kwargs):
        for idx in self._ordered_indices():
            venue = self._venues[idx]
            try:
                t0 = time.time()
                fn = getattr(venue, fn_name)
                out = fn(*args, **kwargs)
                self._active_idx = idx
                latency_ms = (time.time() - t0) * 1000.0
                self._mark_read(idx, latency_ms=latency_ms)
                return out
            except Exception as e:
                self._mark_failure(idx, e)
        raise RuntimeError(f"All venues failed for {fn_name}")

    def get_account(self) -> Account:
        return self._first_read("get_account")

    def get_positions(self) -> dict[str, Position]:
        return self._first_read("get_positions")

    def get_position(self, symbol: str) -> Position | None:
        return self._first_read("get_position", symbol)

    def submit_order(self, order: Order) -> Order:
        last_exc: Exception | None = None
        for idx in self._ordered_indices():
            venue = self._venues[idx]
            try:
                result = venue.submit_order(order)
                if self._is_transient_reject(result):
                    last_exc = RuntimeError(
                        f"{venue.name} transient rejection: {result.message}"
                    )
                    self._mark_failure(idx, last_exc)
                    continue
                self._active_idx = idx
                self._mark_submit(idx)
                if getattr(result, "id", ""):
                    self._order_venue_idx[str(result.id)] = idx
                if getattr(result, "broker_id", ""):
                    self.register_order_mapping(str(result.id), str(result.broker_id))
                return result
            except Exception as e:
                last_exc = e
                self._mark_failure(idx, e)
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("No connected venue available")

    def cancel_order(self, order_id: str) -> bool:
        for idx in self._preferred_indices_for_order(order_id):
            venue = self._venues[idx]
            try:
                ok = bool(venue.cancel_order(order_id))
                if ok:
                    self._active_idx = idx
                    self._order_venue_idx[str(order_id)] = idx
                    return True
            except Exception as e:
                self._mark_failure(idx, e)
        return False

    def get_orders(self, active_only: bool = True) -> list[Order]:
        out: list[Order] = []
        seen: set[str] = set()
        for idx in self._connected_indices():
            venue = self._venues[idx]
            try:
                t0 = time.time()
                rows = venue.get_orders(active_only)
                self._mark_read(idx, latency_ms=(time.time() - t0) * 1000.0)
            except Exception as e:
                self._mark_failure(idx, e)
                continue
            for order in (rows or []):
                oid = str(getattr(order, "id", "") or "")
                if not oid or oid in seen:
                    continue
                seen.add(oid)
                out.append(order)
                self._order_venue_idx[oid] = idx
                bid = str(getattr(order, "broker_id", "") or "").strip()
                if bid:
                    self.register_order_mapping(oid, bid)
        return out

    def get_quote(self, symbol: str) -> float | None:
        return self._first_read("get_quote", symbol)

    def get_fills(self, since: datetime = None) -> list[Fill]:
        out: list[Fill] = []
        seen: set[str] = set()
        for idx in self._connected_indices():
            venue = self._venues[idx]
            try:
                t0 = time.time()
                rows = venue.get_fills(since)
                self._mark_read(idx, latency_ms=(time.time() - t0) * 1000.0)
            except Exception as e:
                self._mark_failure(idx, e)
                continue
            for fill in (rows or []):
                fid = str(getattr(fill, "id", "") or "").strip()
                if not fid:
                    bfid = str(getattr(fill, "broker_fill_id", "") or "").strip()
                    fid = "|".join(
                        [
                            str(getattr(fill, "order_id", "") or ""),
                            bfid,
                            str(getattr(fill, "symbol", "") or ""),
                            str(getattr(fill, "quantity", 0) or 0),
                            f"{float(getattr(fill, 'price', 0.0) or 0.0):.6f}",
                            str(getattr(fill, "timestamp", "") or ""),
                        ]
                    )
                if fid in seen:
                    continue
                seen.add(fid)
                out.append(fill)
        return out

    def get_order_status(self, order_id: str) -> OrderStatus | None:
        for idx in self._preferred_indices_for_order(order_id):
            venue = self._venues[idx]
            try:
                t0 = time.time()
                status = venue.get_order_status(order_id)
                self._mark_read(idx, latency_ms=(time.time() - t0) * 1000.0)
                self._active_idx = idx
                if status is not None:
                    self._order_venue_idx[str(order_id)] = idx
                    return status
            except Exception as e:
                self._mark_failure(idx, e)
        return None

    def sync_order(self, order: Order) -> Order:
        order_id = str(getattr(order, "id", "") or "")
        for idx in self._preferred_indices_for_order(order_id):
            venue = self._venues[idx]
            try:
                t0 = time.time()
                synced = venue.sync_order(order)
                self._mark_read(idx, latency_ms=(time.time() - t0) * 1000.0)
                self._active_idx = idx
                if order_id:
                    self._order_venue_idx[order_id] = idx
                bid = str(getattr(synced, "broker_id", "") or "").strip()
                if order_id and bid:
                    self.register_order_mapping(order_id, bid)
                return synced
            except Exception as e:
                self._mark_failure(idx, e)
        return order

    def get_health_snapshot(self) -> dict[str, object]:
        active_name = None
        if 0 <= self._active_idx < len(self._venues):
            active_name = self._venues[self._active_idx].name
        venues = []
        for idx, venue in enumerate(self._venues):
            last_fail = float(self._last_fail_ts.get(idx, 0.0))
            cooldown_until = (
                last_fail + float(self._cooldown_seconds)
                if last_fail > 0
                else 0.0
            )
            venues.append(
                {
                    "name": venue.name,
                    "connected": bool(venue.is_connected),
                    "fail_count": int(self._fail_counts.get(venue.name, 0)),
                    "submit_count": int(self._submit_counts.get(venue.name, 0)),
                    "read_count": int(self._read_counts.get(venue.name, 0)),
                    "avg_read_latency_ms": round(
                        float(self._avg_read_latency_ms(idx)), 3
                    ),
                    "recent_failures_5m": int(self._recent_failures(idx, window_seconds=300.0)),
                    "last_error": str(self._last_errors.get(str(venue.name), "")),
                    "score": round(float(self._venue_score(idx)), 4),
                    "cooldown_until": cooldown_until,
                }
            )
        return {
            "active_venue": active_name,
            "cooldown_seconds": self._cooldown_seconds,
            "order_affinity_count": int(len(self._order_venue_idx)),
            "venues": venues,
        }


def _create_live_broker_by_type(broker_type: str) -> BrokerInterface:
    broker_type = str(broker_type or "ths").lower()
    if broker_type in ('zszq', 'zhaoshang'):
        return ZSZQBroker()
    if broker_type in ('ths', 'ht', 'gj', 'yh'):
        return THSBroker(broker_type=broker_type)
    log.warning("Unknown live broker_type '%s', fallback to ths", broker_type)
    return THSBroker(broker_type='ths')

