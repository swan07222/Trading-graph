from core.types import OrderStatus
from trading.broker import parse_broker_status


def test_parse_broker_status_handles_chinese_terms() -> None:
    assert parse_broker_status("\u5168\u90e8\u6210\u4ea4") == OrderStatus.FILLED  # 全部成交
    assert parse_broker_status("\u90e8\u5206\u6210\u4ea4") == OrderStatus.PARTIAL  # 部分成交
    assert parse_broker_status("\u5df2\u62a5") == OrderStatus.ACCEPTED  # 已报
    assert parse_broker_status("\u5df2\u64a4\u5355") == OrderStatus.CANCELLED  # 已撤单
    assert parse_broker_status("\u5e9f\u5355") == OrderStatus.REJECTED  # 废单


def test_parse_broker_status_handles_english_terms() -> None:
    assert parse_broker_status("filled") == OrderStatus.FILLED
    assert parse_broker_status("partially filled") == OrderStatus.PARTIAL
    assert parse_broker_status("accepted") == OrderStatus.ACCEPTED
    assert parse_broker_status("cancelled") == OrderStatus.CANCELLED
    assert parse_broker_status("rejected") == OrderStatus.REJECTED


def test_parse_broker_status_defaults_to_submitted() -> None:
    assert parse_broker_status("") == OrderStatus.SUBMITTED
    assert parse_broker_status("unknown-state") == OrderStatus.SUBMITTED
