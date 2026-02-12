from trading.alerts import Alert, AlertCategory, AlertManager, AlertPriority


def test_alert_repeat_escalation():
    mgr = AlertManager()
    for _ in range(3):
        mgr.send_immediate(
            Alert(
                category=AlertCategory.SYSTEM,
                priority=AlertPriority.MEDIUM,
                title="repeated-system-warning",
                message="degraded",
                throttle_key="repeat-warning",
            )
        )
    hist = mgr.get_history(limit=5)
    assert hist
    assert hist[-1].priority in {AlertPriority.HIGH, AlertPriority.CRITICAL}


def test_alert_stats_snapshot():
    mgr = AlertManager()
    mgr.send_immediate(
        Alert(
            category=AlertCategory.RISK,
            priority=AlertPriority.HIGH,
            title="risk-alert",
            message="risk threshold hit",
            throttle_key="risk-alert",
        )
    )
    stats = mgr.get_alert_stats()
    assert stats["total"] >= 1
    assert "by_priority" in stats
    assert "by_category" in stats
