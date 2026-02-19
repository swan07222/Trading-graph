import json
from types import SimpleNamespace

from utils.policy import TradePolicyEngine


def _signal(**kwargs):
    base = {
        "symbol": "600519",
        "side": SimpleNamespace(value="buy"),
        "quantity": 100,
        "price": 100.0,
        "approvals_count": 2,
        "approver_ids": ["a", "b"],
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_policy_engine_blocks_symbol(tmp_path):
    policy_path = tmp_path / "security_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "version": "9.1",
                "enabled": True,
                "live_trade": {
                    "min_approvals": 1,
                    "blocked_symbols": ["600519"],
                    "allowed_sides": ["buy", "sell"],
                    "max_order_notional": 1000000,
                },
            }
        ),
        encoding="utf-8",
    )
    eng = TradePolicyEngine(policy_path=policy_path)
    d = eng.evaluate_live_trade(_signal())
    assert d.allowed is False
    assert "blocked symbol" in d.reason


def test_policy_engine_enforces_notional_and_approvals(tmp_path):
    policy_path = tmp_path / "security_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "version": "9.2",
                "enabled": True,
                "live_trade": {
                    "min_approvals": 2,
                    "require_distinct_approvers": True,
                    "max_order_notional": 5000,
                    "allowed_sides": ["buy", "sell"],
                },
            }
        ),
        encoding="utf-8",
    )
    eng = TradePolicyEngine(policy_path=policy_path)
    d1 = eng.evaluate_live_trade(_signal(quantity=100, price=100.0))
    assert d1.allowed is False
    assert "notional" in d1.reason

    d2 = eng.evaluate_live_trade(_signal(quantity=10, price=100.0, approver_ids=["a"], approvals_count=1))
    assert d2.allowed is False
    assert "distinct approvals" in d2.reason


def test_policy_engine_enforces_manual_and_change_ticket(tmp_path):
    policy_path = tmp_path / "security_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "version": "9.3",
                "enabled": True,
                "live_trade": {
                    "min_approvals": 1,
                    "allowed_sides": ["buy", "sell"],
                    "allowed_order_types": ["limit"],
                    "require_manual_for_live": True,
                    "require_change_ticket": True,
                },
            }
        ),
        encoding="utf-8",
    )
    eng = TradePolicyEngine(policy_path=policy_path)

    d1 = eng.evaluate_live_trade(
        _signal(auto_generated=True, order_type="limit", approvals_count=1)
    )
    assert d1.allowed is False
    assert "manual" in d1.reason

    d2 = eng.evaluate_live_trade(
        _signal(auto_generated=False, order_type="limit", approvals_count=1)
    )
    assert d2.allowed is False
    assert "change ticket" in d2.reason.lower()

    d3 = eng.evaluate_live_trade(
        _signal(
            auto_generated=False,
            order_type="limit",
            approvals_count=1,
            change_ticket="CHG-1024",
        )
    )
    assert d3.allowed is True


def test_policy_engine_enforces_strategy_and_justification(tmp_path):
    policy_path = tmp_path / "security_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "version": "9.4",
                "enabled": True,
                "live_trade": {
                    "min_approvals": 1,
                    "allowed_sides": ["buy", "sell"],
                    "blocked_strategies": ["high_risk_scalper"],
                    "require_business_justification": True,
                },
            }
        ),
        encoding="utf-8",
    )
    eng = TradePolicyEngine(policy_path=policy_path)

    d1 = eng.evaluate_live_trade(
        _signal(strategy="high_risk_scalper", approvals_count=1)
    )
    assert d1.allowed is False
    assert "blocked strategy" in d1.reason.lower()

    d2 = eng.evaluate_live_trade(
        _signal(strategy="swing_alpha", approvals_count=1, reasons=[])
    )
    assert d2.allowed is False
    assert "justification" in d2.reason.lower()

    d3 = eng.evaluate_live_trade(
        _signal(
            strategy="swing_alpha",
            approvals_count=1,
            reasons=["earnings drift setup"],
        )
    )
    assert d3.allowed is True


def test_policy_engine_normalizes_symbol_and_order_sanity(tmp_path):
    policy_path = tmp_path / "security_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "version": "9.5",
                "enabled": True,
                "live_trade": {
                    "min_approvals": 1,
                    "blocked_symbols": ["600519"],
                    "allowed_sides": ["buy", "sell"],
                    "allowed_order_types": ["limit", "market"],
                },
            }
        ),
        encoding="utf-8",
    )
    eng = TradePolicyEngine(policy_path=policy_path)

    d1 = eng.evaluate_live_trade(_signal(symbol="sh600519", approvals_count=1))
    assert d1.allowed is False
    assert "blocked symbol" in d1.reason.lower()

    d2 = eng.evaluate_live_trade(
        _signal(symbol="000001", quantity=0, price=10.0, approvals_count=1)
    )
    assert d2.allowed is False
    assert "quantity" in d2.reason.lower()

    d3 = eng.evaluate_live_trade(
        _signal(
            symbol="000001",
            quantity=10,
            price=0.0,
            order_type="limit",
            approvals_count=1,
        )
    )
    assert d3.allowed is False
    assert "limit price" in d3.reason.lower()


def test_policy_engine_rejects_missing_symbol_or_side(tmp_path):
    policy_path = tmp_path / "security_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "version": "9.6",
                "enabled": True,
                "live_trade": {
                    "min_approvals": 1,
                    "allowed_sides": ["buy", "sell"],
                    "allowed_order_types": ["limit", "market"],
                    "require_change_ticket": False,
                    "require_business_justification": False,
                },
            }
        ),
        encoding="utf-8",
    )
    eng = TradePolicyEngine(policy_path=policy_path)

    missing_symbol = eng.evaluate_live_trade(
        _signal(symbol="", approvals_count=1, approver_ids=["a"])
    )
    assert missing_symbol.allowed is False
    assert "missing symbol" in missing_symbol.reason.lower()

    missing_side = eng.evaluate_live_trade(
        _signal(side="", approvals_count=1, approver_ids=["a"])
    )
    assert missing_side.allowed is False
    assert "missing side" in missing_side.reason.lower()


def test_policy_engine_prevents_market_notional_bypass(tmp_path):
    policy_path = tmp_path / "security_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "version": "9.7",
                "enabled": True,
                "live_trade": {
                    "min_approvals": 1,
                    "allowed_sides": ["buy", "sell"],
                    "allowed_order_types": ["market"],
                    "max_order_notional": 5000,
                    "require_change_ticket": False,
                    "require_business_justification": False,
                },
            }
        ),
        encoding="utf-8",
    )
    eng = TradePolicyEngine(policy_path=policy_path)

    no_price = eng.evaluate_live_trade(
        _signal(
            order_type="market",
            quantity=100000,
            price=0.0,
            approvals_count=1,
            approver_ids=["a"],
        )
    )
    assert no_price.allowed is False
    assert "notional price" in no_price.reason.lower()

    with_reference = eng.evaluate_live_trade(
        _signal(
            order_type="market",
            quantity=100,
            price=0.0,
            reference_price=100.0,
            approvals_count=1,
            approver_ids=["a"],
        )
    )
    assert with_reference.allowed is False
    assert "notional exceeds" in with_reference.reason.lower()
