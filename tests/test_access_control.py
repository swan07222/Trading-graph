from datetime import datetime, timedelta

from config.settings import CONFIG
from utils.security import AccessControl


def test_access_control_requires_2fa_for_live():
    ac = AccessControl()
    ac.set_role("live_trader")
    ac.mark_2fa_verified(False)
    assert ac.check("trade_live") is False
    ac.mark_2fa_verified(True)
    assert ac.check("trade_live") is True


def test_access_control_session_expiry_blocks_live():
    ac = AccessControl()
    ac.set_role("live_trader")
    ac.mark_2fa_verified(True)
    ac._session_started_at = datetime.now() - timedelta(hours=CONFIG.security.max_session_hours + 1)
    assert ac.check("trade_live") is False


def test_access_control_custom_role_grants():
    ac = AccessControl()
    ac.create_role("qa", ["view"])
    ac.grant_permission("qa", "analyze")
    ac.set_role("qa")
    assert ac.check("view") is True
    assert ac.check("analyze") is True
