from types import SimpleNamespace

from utils.institutional import collect_institutional_readiness


def test_institutional_readiness_passes_with_strict_controls():
    sec = SimpleNamespace(
        enable_runtime_lease=True,
        runtime_lease_backend="sqlite",
        runtime_lease_ttl_seconds=10.0,
        audit_logging=True,
        audit_hash_chain=True,
        audit_retention_days=365,
        require_live_trade_permission=True,
        require_2fa_for_live=True,
        strict_live_governance=True,
        min_live_approvals=2,
    )
    policy = {
        "version": "9.9",
        "enabled": True,
        "live_trade": {
            "min_approvals": 2,
            "require_distinct_approvers": True,
            "require_change_ticket": True,
            "require_business_justification": True,
            "max_order_notional": 100000.0,
        },
    }
    report = collect_institutional_readiness(
        security_config=sec,
        policy_payload=policy,
    )
    assert report["pass"] is True
    assert report["failed_required_controls"] == []


def test_institutional_readiness_flags_missing_controls():
    sec = SimpleNamespace(
        enable_runtime_lease=False,
        runtime_lease_backend="file",
        runtime_lease_ttl_seconds=1.0,
        audit_logging=False,
        audit_hash_chain=False,
        audit_retention_days=30,
        require_live_trade_permission=False,
        require_2fa_for_live=False,
        strict_live_governance=False,
        min_live_approvals=1,
    )
    policy = {
        "version": "0.1",
        "enabled": False,
        "live_trade": {
            "min_approvals": 1,
            "require_distinct_approvers": False,
            "require_change_ticket": False,
            "require_business_justification": False,
            "max_order_notional": 0.0,
        },
    }
    report = collect_institutional_readiness(
        security_config=sec,
        policy_payload=policy,
    )
    assert report["pass"] is False
    failed = set(report["failed_required_controls"])
    assert "runtime_lease_enabled" in failed
    assert "strict_live_governance" in failed
    assert "policy_change_ticket" in failed
