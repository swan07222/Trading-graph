from datetime import datetime, timedelta

import pytest

from config.settings import CONFIG
from utils.security import AccessControl


def test_access_control_requires_2fa_for_live() -> None:
    ac = AccessControl()
    ac.set_role("live_trader")
    ac.mark_2fa_verified(False)
    assert ac.check("trade_live") is False
    ac.mark_2fa_verified(True)
    assert ac.check("trade_live") is True


def test_access_control_session_expiry_blocks_live() -> None:
    ac = AccessControl()
    ac.set_role("live_trader")
    ac.mark_2fa_verified(True)
    ac._session_started_at = datetime.now() - timedelta(hours=CONFIG.security.max_session_hours + 1)
    assert ac.check("trade_live") is False


def test_access_control_custom_role_grants() -> None:
    ac = AccessControl()
    ac.create_role("qa", ["view"])
    ac.grant_permission("qa", "analyze")
    ac.set_role("qa")
    assert ac.check("view") is True
    assert ac.check("analyze") is True


def test_access_control_2fa_ttl_expiry_blocks_live() -> None:
    old_ttl = int(getattr(CONFIG.security, "two_factor_ttl_minutes", 30))
    try:
        CONFIG.security.two_factor_ttl_minutes = 1
        ac = AccessControl()
        ac.set_role("live_trader")
        ac.mark_2fa_verified(True)
        ac._two_factor_verified_at = datetime.now() - timedelta(minutes=2)
        assert ac.check("trade_live") is False
    finally:
        CONFIG.security.two_factor_ttl_minutes = old_ttl


def test_access_control_identity_lock_blocks_mutation() -> None:
    ac = AccessControl()
    ac.lock_identity("unit_test")

    with pytest.raises(RuntimeError, match="identity is locked"):
        ac.set_role("viewer")
    with pytest.raises(RuntimeError, match="identity is locked"):
        ac.set_user("alice")
