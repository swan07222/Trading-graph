from __future__ import annotations

import hashlib
from collections import OrderedDict
from datetime import date, datetime
from typing import Any

from core.types import OrderStatus


class BoundedOrderedDict(OrderedDict[Any, Any]):
    """OrderedDict with max size - evicts oldest on overflow."""

    def __init__(
        self,
        maxsize: int = 10000,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self._maxsize:
            self.popitem(last=False)


def make_fill_uid(
    broker_name: str,
    broker_fill_id: str,
    symbol: str,
    ts: datetime,
    price: float,
    qty: int,
) -> str:
    """
    Create a globally-unique, stable Fill.id for OMS primary key.

    Format with broker_fill_id:
      FILL|<YYYY-MM-DD>|<broker>|<broker_fill_id>|<symbol>
    Fallback (no broker_fill_id):
      FILL|<YYYY-MM-DD>|<broker>|<symbol>|<hash>
    """
    broker = (broker_name or "broker").replace("|", "_")
    broker_fill_id = (broker_fill_id or "").strip()
    sym = str(symbol or "").strip()

    day = (
        ts.date().isoformat()
        if isinstance(ts, datetime)
        else date.today().isoformat()
    )

    if broker_fill_id:
        return f"FILL|{day}|{broker}|{broker_fill_id}|{sym}"

    iso = ts.isoformat() if isinstance(ts, datetime) else day
    raw = f"{iso}|{broker}|{sym}|{int(qty)}|{float(price):.4f}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"FILL|{day}|{broker}|{sym}|{h}"


def parse_broker_status(status_str: str) -> OrderStatus:
    """
    Parse broker/native status strings into internal OrderStatus.

    Handles both English keywords and common Chinese statuses.
    Unicode escapes are used for Chinese terms to avoid source-encoding drift.
    """
    s = str(status_str or "").strip().lower()
    if not s:
        return OrderStatus.SUBMITTED

    def _has_any(*tokens: str) -> bool:
        return any(t and (t in s) for t in tokens)

    if _has_any(
        "partial",
        "partially filled",
        "\u90e8\u5206\u6210\u4ea4",  # 閮ㄥ垎鎴愪氦
    ):
        return OrderStatus.PARTIAL

    if (
        s == "filled"
        or _has_any(
            "fully filled",
            "all traded",
            "\u5168\u90e8\u6210\u4ea4",  # 鍏ㄩ儴鎴愪氦
            "\u5df2\u6210",  # 宸叉垚
        )
    ):
        return OrderStatus.FILLED

    if _has_any(
        "accepted",
        "submitted",
        "pending",
        "new",
        "\u5df2\u62a5",  # 宸叉姤
        "\u5df2\u59d4\u6258",  # 宸插鎵?
    ):
        return OrderStatus.ACCEPTED

    if _has_any(
        "cancelled",
        "canceled",
        "cancelled by user",
        "\u5df2\u64a4",  # 宸叉挙
        "\u64a4\u5355",  # 鎾ゅ崟
    ):
        return OrderStatus.CANCELLED

    if _has_any(
        "rejected",
        "reject",
        "invalid",
        "\u5e9f\u5355",  # 搴熷崟
        "\u62d2\u7edd",  # 鎷掔粷
    ):
        return OrderStatus.REJECTED

    return OrderStatus.SUBMITTED

