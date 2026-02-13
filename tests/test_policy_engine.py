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
