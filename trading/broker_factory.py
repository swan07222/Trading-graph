from __future__ import annotations

from typing import TYPE_CHECKING, Any

from config import CONFIG
from utils.logger import get_logger

if TYPE_CHECKING:
    from .broker import BrokerInterface

log = get_logger(__name__)


def resolve_live_export(name: str) -> Any:
    if name in {
        "EasytraderBroker",
        "THSBroker",
        "ZSZQBroker",
        "MultiVenueBroker",
    }:
        from . import broker_live as _broker_live

        return getattr(_broker_live, name)
    raise AttributeError(name)


def create_broker(mode: str | None = None, **kwargs: Any) -> BrokerInterface:
    from .broker import SimulatorBroker
    from .broker_live import (
        MultiVenueBroker,
        THSBroker,
        ZSZQBroker,
        _create_live_broker_by_type,
    )

    if mode is None:
        mode = (
            CONFIG.trading_mode.value
            if hasattr(CONFIG.trading_mode, "value")
            else str(CONFIG.trading_mode)
        )

    mode = mode.lower()
    if mode in ("simulation", "paper"):
        return SimulatorBroker(kwargs.get("capital", CONFIG.capital))

    if mode == "live":
        priority = kwargs.get("venue_priority")
        if priority is None:
            priority = getattr(CONFIG.trading, "venue_priority", []) or []

        enable_multi = bool(kwargs.get("enable_multi_venue", False))
        if not enable_multi:
            enable_multi = bool(getattr(CONFIG.trading, "enable_multi_venue", False))
        if not enable_multi and isinstance(priority, list) and len(priority) > 1:
            enable_multi = True

        if enable_multi:
            venues: list[BrokerInterface] = []
            for item in priority:
                broker_type = str(item or "").strip().lower()
                if not broker_type:
                    continue
                venues.append(_create_live_broker_by_type(broker_type))
            if not venues:
                venues = [_create_live_broker_by_type(kwargs.get("broker_type", "ths"))]
            cooldown = kwargs.get(
                "venue_failover_cooldown_seconds",
                getattr(CONFIG.trading, "venue_failover_cooldown_seconds", 30),
            )
            return MultiVenueBroker(venues, failover_cooldown_seconds=int(cooldown))

        return _create_live_broker_by_type(kwargs.get("broker_type", "ths"))

    if mode in ("ths", "ht", "gj", "yh"):
        return THSBroker(broker_type=mode)
    if mode in ("zszq", "zhaoshang"):
        return ZSZQBroker()

    log.warning("Unknown broker mode: %s, using simulator", mode)
    return SimulatorBroker()
