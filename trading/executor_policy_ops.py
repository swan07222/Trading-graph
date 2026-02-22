from __future__ import annotations

from config import CONFIG, TradingMode
from core.types import OrderType
from trading.executor_error_policy import SOFT_FAIL_EXCEPTIONS, should_escalate_exception
from utils.logger import get_logger

log = get_logger(__name__)
_SOFT_FAIL_EXCEPTIONS = SOFT_FAIL_EXCEPTIONS
_ADVANCED_ORDER_TYPES = frozenset(
    {
        OrderType.STOP,
        OrderType.STOP_LIMIT,
        OrderType.IOC,
        OrderType.FOK,
        OrderType.TRAIL_MARKET,
        OrderType.TRAIL_LIMIT,
    }
)


def _should_escalate_runtime_exception(self, exc: BaseException) -> bool:
    return should_escalate_exception(getattr(self, "mode", None), exc)


def _is_advanced_order_type(self, order_type: OrderType) -> bool:
    return order_type in _ADVANCED_ORDER_TYPES


def _allow_order_type_emulation(self, requested: OrderType) -> tuple[bool, str]:
    if not _is_advanced_order_type(self, requested):
        return True, ""
    sec_cfg = getattr(CONFIG, "security", None)
    mode_is_live = self.mode == TradingMode.LIVE
    allowed = (
        bool(getattr(sec_cfg, "allow_live_order_type_emulation", False))
        if mode_is_live
        else bool(getattr(sec_cfg, "allow_non_live_order_type_emulation", True))
    )
    if allowed:
        return True, ""
    scope = "LIVE mode" if mode_is_live else "simulation/paper mode"
    return (
        False,
        f"{requested.value} requires broker-native support; "
        f"emulation is disabled in {scope}",
    )


def _broker_supports_order_type(self, order_type: OrderType) -> bool:
    if order_type in {OrderType.MARKET, OrderType.LIMIT}:
        return True

    broker = getattr(self, "broker", None)
    if broker is None:
        return False

    try:
        checker = getattr(broker, "supports_order_type", None)
        if callable(checker):
            return bool(checker(order_type))
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.debug(
            "Broker supports_order_type probe failed (%s): %s",
            order_type.value,
            e,
        )

    try:
        supported = getattr(broker, "supported_order_types", None)
        if callable(supported):
            supported = supported()
        if isinstance(supported, (list, tuple, set, frozenset)):
            normalized = {
                str(getattr(item, "value", item) or "").strip().lower()
                for item in supported
            }
            return order_type.value in normalized
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.debug(
            "Broker supported_order_types probe failed (%s): %s",
            order_type.value,
            e,
        )

    return False
