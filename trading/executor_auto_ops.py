from __future__ import annotations

from typing import Any

from config import CONFIG, TradingMode
from core.types import AutoTradeMode, AutoTradeState
from trading.auto_trader import AutoTrader
from trading.executor_error_policy import SOFT_FAIL_EXCEPTIONS
from trading.health import ComponentType, HealthStatus, get_health_monitor
from utils.logger import get_logger
from utils.security import get_audit_log

log = get_logger(__name__)
_SOFT_FAIL_EXCEPTIONS = SOFT_FAIL_EXCEPTIONS


def trigger_model_drift_alarm(
    cls,
    reason: str,
    *,
    severity: str = "critical",
    metadata: dict[str, Any] | None = None,
) -> int:
    """Raise a runtime model-drift alarm and disable live auto-trading."""
    sev = str(severity or "critical").strip().lower()
    status = (
        HealthStatus.UNHEALTHY
        if sev in {"critical", "unhealthy", "block"}
        else HealthStatus.DEGRADED
    )
    msg = str(reason or "model_drift_alarm").strip() or "model_drift_alarm"

    try:
        get_health_monitor().report_component_health(
            ComponentType.MODEL,
            status,
            error=msg,
        )
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.debug("Model drift health-report update failed: %s", e)

    handled = 0
    with cls._ACTIVE_ENGINES_LOCK:
        engines = list(cls._ACTIVE_ENGINES)
    for eng in engines:
        try:
            if eng._apply_model_drift_alarm(  # noqa: SLF001
                msg,
                status=status,
                metadata=metadata,
            ):
                handled += 1
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Model drift alarm delivery failed: %s", e)
    return int(handled)


def _apply_model_drift_alarm(
    self,
    reason: str,
    *,
    status: HealthStatus = HealthStatus.UNHEALTHY,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Apply drift alarm policy to this engine instance."""
    del metadata
    if self.mode != TradingMode.LIVE:
        return False
    cfg = getattr(CONFIG, "auto_trade", None)
    if not bool(getattr(cfg, "auto_disable_on_model_drift", True)):
        return False

    pause_seconds = int(
        max(
            60,
            float(getattr(cfg, "model_drift_pause_seconds", 3600) or 3600),
        )
    )
    action_taken = False
    if self.auto_trader is not None:
        try:
            if self.auto_trader.get_mode() != AutoTradeMode.MANUAL:
                self.auto_trader.set_mode(AutoTradeMode.MANUAL)
                action_taken = True
            self.auto_trader.pause(
                f"Model drift alarm: {reason}",
                duration_seconds=pause_seconds,
            )
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Model drift auto-trader pause failed: %s", e)

    try:
        CONFIG.auto_trade.enabled = False
    except _SOFT_FAIL_EXCEPTIONS as exc:
        log.warning("Failed to disable auto-trade after model drift alarm: %s", exc)

    try:
        self._alert_manager.risk_alert(
            "Model drift alarm",
            f"Auto-trade forced to MANUAL ({status.value}): {reason}",
        )
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.debug("Model drift alert dispatch failed: %s", e)

    try:
        get_audit_log().log_risk_event(
            "model_drift_auto_disable",
            {
                "mode": str(getattr(self.mode, "value", self.mode)),
                "status": str(status.value),
                "reason": str(reason),
                "pause_seconds": int(pause_seconds),
                "auto_trader_present": bool(self.auto_trader is not None),
            },
        )
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.debug("Model drift audit log failed: %s", e)
    return bool(action_taken)


def init_auto_trader(self, predictor: Any, watch_list: list[str]) -> None:
    """Initialize the auto-trader with a predictor and watchlist."""
    self.auto_trader = AutoTrader(
        engine=self,
        predictor=predictor,
        watch_list=watch_list,
    )
    self._restore_auto_trader_state()
    log.info("Auto-trader initialized")


def start_auto_trade(self, mode: AutoTradeMode = AutoTradeMode.AUTO) -> None:
    """Start auto-trading in the specified mode."""
    if self.auto_trader is None:
        log.error("Auto-trader not initialized. Call init_auto_trader() first.")
        return

    if not self._running:
        log.error("Execution engine not running. Call start() first.")
        return

    if (
        self.mode == TradingMode.LIVE
        and CONFIG.auto_trade.confirm_live_auto_trade
        and mode != AutoTradeMode.MANUAL
    ):
        log.warning(
            "Live auto-trading requested but confirm_live_auto_trade is True. "
            "UI must confirm before proceeding."
        )

    self.auto_trader.set_mode(mode)
    log.info("Auto-trading started: mode=%s", mode.value)


def stop_auto_trade(self) -> None:
    """Stop auto-trading (switch to MANUAL)."""
    if self.auto_trader:
        self.auto_trader.set_mode(AutoTradeMode.MANUAL)
        log.info("Auto-trading stopped (switched to MANUAL)")


def set_auto_mode(self, mode: AutoTradeMode) -> None:
    """Change auto-trade mode."""
    if self.auto_trader:
        self.auto_trader.set_mode(mode)


def get_auto_trade_state(self) -> AutoTradeState | None:
    """Get auto-trade state snapshot."""
    if self.auto_trader:
        return self.auto_trader.get_state()
    return None
