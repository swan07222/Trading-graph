from __future__ import annotations

import os
import queue
import socket
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from config import CONFIG, TradingMode
from core.types import (
    AutoTradeMode,
    Fill,
    Order,
    TradeSignal,
)
from trading.alerts import AlertPriority, get_alert_manager
from trading.auto_trader import AutoTrader
from trading.broker import BrokerInterface, create_broker
from trading.executor_error_policy import SOFT_FAIL_EXCEPTIONS
from trading.health import ComponentType, HealthStatus, get_health_monitor
from trading.kill_switch import get_kill_switch
from trading.risk import RiskManager, get_risk_manager
from trading.runtime_lease import RuntimeLeaseClient
from utils.logger import get_logger
from utils.metrics import set_gauge
from utils.security import get_access_control, get_audit_log

log = get_logger(__name__)
_SOFT_FAIL_EXCEPTIONS = SOFT_FAIL_EXCEPTIONS
_CRITICAL_ENGINE_THREADS = frozenset(
    {
        "exec",
        "fill_sync",
        "status_sync",
        "recon",
        "checkpoint",
    }
)

try:
    from utils.metrics_http import register_snapshot_provider, unregister_snapshot_provider
except (ImportError, OSError):  # pragma: no cover - optional runtime integration
    register_snapshot_provider = None
    unregister_snapshot_provider = None

def _resolve_access_control():
    try:
        from trading import executor as _executor_mod

        factory = getattr(_executor_mod, "get_access_control", None)
        if callable(factory):
            return factory()
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.debug("Executor-scoped access control resolver unavailable: %s", e)
    return get_access_control()


def _resolve_audit_log():
    try:
        from trading import executor as _executor_mod

        factory = getattr(_executor_mod, "get_audit_log", None)
        if callable(factory):
            return factory()
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.debug("Executor-scoped audit log resolver unavailable: %s", e)
    return get_audit_log()

def __init__(self, mode: TradingMode = None) -> None:
    self.mode = mode or CONFIG.trading_mode
    self.broker: BrokerInterface = create_broker(self.mode.value)
    self.risk_manager: RiskManager = get_risk_manager()

    self._kill_switch = get_kill_switch()
    self._health_monitor = get_health_monitor()
    self._alert_manager = get_alert_manager()
    self._fills_lock = threading.RLock()

    self._queue: queue.Queue[TradeSignal | None] = queue.Queue()
    self._running = False
    self._stop_event = threading.Event()

    self._exec_thread: threading.Thread | None = None
    self._fill_sync_thread: threading.Thread | None = None
    self._status_sync_thread: threading.Thread | None = None
    self._recon_thread: threading.Thread | None = None
    self._reconnect_thread: threading.Thread | None = None
    self._watchdog_thread: threading.Thread | None = None
    self._checkpoint_thread: threading.Thread | None = None

    self._processed_fill_ids: set[str] = set()

    self._last_fill_sync: datetime | None = None
    self._last_checkpoint_ts: float = 0.0
    self._thread_heartbeats: dict[str, float] = {}
    self._thread_hb_lock = threading.RLock()
    self._recent_submit_keys: dict[str, float] = {}
    self._recent_submissions: dict[str, deque] = {}
    self._recent_rejections: deque = deque()
    self._synthetic_exits: dict[str, dict[str, Any]] = {}
    self._synthetic_exit_lock = threading.RLock()
    self._exec_quality_lock = threading.RLock()
    self._exec_quality: dict[str, Any] = {
        "fills": 0,
        "slippage_bps_sum": 0.0,
        "slippage_bps_abs_sum": 0.0,
        "by_reason": {},
        "last_update": "",
    }
    self._last_watchdog_warning_ts: float = 0.0
    self._runtime_state_path = Path(CONFIG.data_dir) / self._RUNTIME_STATE_FILE
    self._synthetic_exit_state_path = (
        Path(CONFIG.data_dir) / self._SYNTHETIC_EXITS_FILE
    )
    self._last_synthetic_persist_ts: float = 0.0
    self._synthetic_persist_min_interval_s: float = 1.0
    sec_cfg = getattr(CONFIG, "security", None)
    self._runtime_lease_backend = str(
        getattr(sec_cfg, "runtime_lease_backend", "sqlite") or "sqlite"
    ).strip().lower()
    self._runtime_lease_cluster = str(
        getattr(sec_cfg, "runtime_lease_cluster", "execution_engine")
        or "execution_engine"
    ).strip()
    node_cfg = str(getattr(sec_cfg, "runtime_lease_node_id", "") or "").strip()
    node_id = node_cfg or f"{socket.gethostname()}:{os.getpid()}"
    self._runtime_lease_id = f"{node_id}:{uuid.uuid4().hex[:10]}"
    lease_path_cfg = str(getattr(sec_cfg, "runtime_lease_path", "") or "").strip()
    default_lease_name = (
        self._RUNTIME_LEASE_DB_FILE
        if self._runtime_lease_backend == "sqlite"
        else self._RUNTIME_LEASE_FILE
    )
    self._runtime_lease_path = (
        Path(lease_path_cfg)
        if lease_path_cfg
        else (Path(CONFIG.data_dir) / default_lease_name)
    )
    self._runtime_lease_enabled = bool(
        getattr(sec_cfg, "enable_runtime_lease", True)
    )
    self._runtime_lease_ttl_seconds = float(
        getattr(sec_cfg, "runtime_lease_ttl_seconds", 20.0) or 20.0
    )
    self._runtime_lease_client: RuntimeLeaseClient | None = None
    self._runtime_lease_owner_hint: dict[str, Any] | None = None
    self._runtime_lease_fencing_token: int = 0
    self._runtime_recovered = False
    self._recovered_auto_state: dict | None = None

    self.on_fill: Callable[[Order, Fill], None] | None = None
    self.on_reject: Callable[[Order, str], None] | None = None

    self._kill_switch.on_activate(self._on_kill_switch)
    self._health_monitor.on_degraded(self._on_health_degraded)
    self._processed_fill_ids = self._load_processed_fills()

    # Auto-trader (created but not started until explicitly requested)
    self.auto_trader: AutoTrader | None = None
    self._snapshot_provider_name = "execution_engine"
    self._restore_runtime_state()
    self._restore_synthetic_exits()
    with self.__class__._ACTIVE_ENGINES_LOCK:
        self.__class__._ACTIVE_ENGINES.add(self)


def start(self) -> bool:
    if self._running:
        return True

    ok_live, live_msg = self._evaluate_live_start_readiness()
    if not ok_live:
        log.error(live_msg)
        self._health_monitor.report_component_health(
            ComponentType.RISK_MANAGER,
            HealthStatus.UNHEALTHY,
            error=live_msg,
        )
        return False

    if not self._acquire_runtime_lease():
        owner = self._runtime_lease_owner_hint or {}
        owner_id = str(owner.get("owner_id", "unknown") or "unknown")
        hb_ts = float(owner.get("heartbeat_ts", 0.0) or 0.0)
        age = max(0.0, time.time() - hb_ts) if hb_ts > 0 else -1.0
        log.error(
            "Execution runtime lease is held by another process: owner=%s age=%.1fs",
            owner_id,
            age,
        )
        return False

    if not self.broker.connect():
        log.error("Broker connection failed")
        self._health_monitor.report_component_health(
            ComponentType.BROKER, HealthStatus.UNHEALTHY,
            error="Connection failed",
        )
        return False

    from trading.oms import get_oms

    oms = get_oms()

    # Rebuild broker ID mappings from persisted orders (crash recovery)
    self._rebuild_broker_mappings(oms)

    account = oms.get_account()

    if self.risk_manager:
        self.risk_manager.initialize(account)
        self.risk_manager.update(account)

    self._health_monitor.start()
    self._alert_manager.start()

    self._running = True
    self._stop_event.clear()
    self._heartbeat("main")
    self._persist_runtime_state(clean_shutdown=False)

    thread_specs: list[tuple[str, Callable[[], None], str]] = [
        ("_exec_thread", self._execution_loop, "exec"),
        ("_fill_sync_thread", self._fill_sync_loop, "fill_sync"),
        ("_status_sync_thread", self._status_sync_loop, "status_sync"),
        ("_recon_thread", self._reconciliation_loop, "recon"),
        ("_watchdog_thread", self._watchdog_loop, "watchdog"),
        ("_checkpoint_thread", self._checkpoint_loop, "checkpoint"),
    ]
    for attr_name, target, thread_name in thread_specs:
        _start_engine_thread(self, attr_name, target, thread_name)

    self._health_monitor.attach_broker(self.broker)
    self._health_monitor.report_component_health(
        ComponentType.BROKER, HealthStatus.HEALTHY
    )
    if register_snapshot_provider is not None:
        try:
            register_snapshot_provider(
                self._snapshot_provider_name,
                self._build_execution_snapshot,
            )
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug(f"Execution snapshot provider registration failed: {e}")

    log.info(f"Execution engine started ({self.mode.value})")
    if self._runtime_recovered:
        self._alert_manager.risk_alert(
            "Runtime recovery mode",
            "Previous session ended uncleanly. Auto-trading starts in safe MANUAL mode.",
        )

    self._alert_manager.system_alert(
        "Trading System Started",
        f"Mode: {self.mode.value}, Equity: {account.equity:,.2f}",
        priority=AlertPriority.MEDIUM,
    )

    self._startup_sync()

    _start_engine_thread(
        self,
        "_reconnect_thread",
        self._broker_reconnect_loop,
        "broker_reconnect",
    )

    # Start auto-trader if it was initialized and config says enabled
    if self.auto_trader and CONFIG.auto_trade.enabled:
        if self._runtime_recovered:
            self.auto_trader.set_mode(AutoTradeMode.MANUAL)
            self.auto_trader.pause(
                "Recovered from unclean shutdown; manual review required.",
                duration_seconds=0,
            )
            log.warning("Auto-trader held in MANUAL due to runtime recovery mode")
        else:
            self.auto_trader.start()
            log.info("Auto-trader auto-started (config.auto_trade.enabled=True)")

    return True


def _start_engine_thread(
    self,
    attr_name: str,
    target: Callable[[], None],
    thread_name: str,
) -> None:
    existing = getattr(self, attr_name, None)
    if isinstance(existing, threading.Thread) and existing.is_alive():
        log.warning(
            "Execution worker thread already running: %s (attr=%s)",
            str(thread_name),
            str(attr_name),
        )
        return

    def _runner() -> None:
        self._heartbeat(thread_name)
        try:
            target()
        except _SOFT_FAIL_EXCEPTIONS as e:
            _handle_worker_thread_crash(self, thread_name=thread_name, exc=e)
        finally:
            with self._thread_hb_lock:
                self._thread_heartbeats[str(thread_name)] = 0.0

    thread = threading.Thread(
        target=_runner,
        name=str(thread_name),
        daemon=False,
    )
    setattr(self, attr_name, thread)
    thread.start()


def _handle_worker_thread_crash(
    self,
    *,
    thread_name: str,
    exc: BaseException,
) -> None:
    if not bool(getattr(self, "_running", False)):
        log.debug("Worker %s exited during shutdown: %s", thread_name, exc)
        return

    msg = f"Worker thread crashed: {thread_name}: {exc}"
    log.exception(msg)
    try:
        self._health_monitor.report_component_health(
            ComponentType.RISK_MANAGER,
            HealthStatus.DEGRADED,
            error=msg,
        )
    except _SOFT_FAIL_EXCEPTIONS as hm_err:
        log.debug("Worker-crash health-report update failed: %s", hm_err)

    try:
        self._alert_manager.risk_alert("Runtime worker crash", msg)
    except _SOFT_FAIL_EXCEPTIONS as alert_err:
        log.debug("Worker-crash alert dispatch failed: %s", alert_err)

    if str(thread_name) not in _CRITICAL_ENGINE_THREADS:
        return
    try:
        if bool(getattr(self._kill_switch, "can_trade", False)):
            self._kill_switch.activate(
                msg,
                activated_by=f"thread_crash:{thread_name}",
            )
    except _SOFT_FAIL_EXCEPTIONS as ks_err:
        log.critical("Kill-switch activation failed after worker crash: %s", ks_err)


def stop(self) -> None:
    if not self._running:
        self._stop_event.set()
        self._release_runtime_lease()
        with self.__class__._ACTIVE_ENGINES_LOCK:
            self.__class__._ACTIVE_ENGINES.discard(self)
        return

    # Stop auto-trader first
    if self.auto_trader:
        try:
            self.auto_trader.stop()
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.warning(f"Auto-trader stop error: {e}")

    self._running = False
    self._stop_event.set()

    try:
        self._queue.put_nowait(None)
    except queue.Full as e:
        log.debug("Execution queue sentinel enqueue skipped: %s", e)

    self._join_worker_threads(timeout_seconds=5.0)

    try:
        self.broker.disconnect()
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.warning(f"Broker disconnect error: {e}")

    try:
        self._health_monitor.stop()
        self._alert_manager.stop()
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.warning("Shutdown monitor stop error: %s", e)
    if unregister_snapshot_provider is not None:
        try:
            unregister_snapshot_provider(self._snapshot_provider_name)
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Execution snapshot provider unregister failed: %s", e)
    self._persist_synthetic_exits(force=True)
    self._persist_runtime_state(clean_shutdown=True)
    self._release_runtime_lease()
    with self.__class__._ACTIVE_ENGINES_LOCK:
        self.__class__._ACTIVE_ENGINES.discard(self)

    log.info("Execution engine stopped")


def _build_execution_snapshot(self) -> dict[str, object]:
    broker = self.broker
    auto_state = None
    if self.auto_trader is not None:
        try:
            auto_state = self.auto_trader.get_state()
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Auto-trader state snapshot unavailable: %s", e)
            auto_state = None

    snapshot: dict[str, object] = {
        "running": bool(self._running),
        "mode": str(getattr(self.mode, "value", self.mode)),
        "broker": {
            "name": getattr(broker, "name", "unknown"),
            "connected": bool(getattr(broker, "is_connected", False)),
        },
        "auto_trade": {
            "enabled": auto_state is not None,
            "state": (
                auto_state.to_dict()
                if hasattr(auto_state, "to_dict")
                else auto_state
            ),
        },
    }

    if hasattr(broker, "get_health_snapshot"):
        try:
            snapshot["broker"]["routing"] = broker.get_health_snapshot()
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Broker health snapshot unavailable: %s", e)
    try:
        with self._thread_hb_lock:
            now = time.time()
            hb_age = {
                k: max(0.0, now - float(v))
                for k, v in self._thread_heartbeats.items()
            }
        snapshot["runtime"] = {
            "recovered_from_checkpoint": bool(self._runtime_recovered),
            "last_checkpoint_age_seconds": (
                max(0.0, now - float(self._last_checkpoint_ts))
                if self._last_checkpoint_ts > 0
                else None
            ),
            "lease_enabled": bool(self._runtime_lease_enabled),
            "lease_backend": str(getattr(self, "_runtime_lease_backend", "file")),
            "lease_cluster": str(getattr(self, "_runtime_lease_cluster", "execution_engine")),
            "lease_owner_id": str(self._runtime_lease_id),
            "lease_fencing_token": int(getattr(self, "_runtime_lease_fencing_token", 0) or 0),
            "lease_path": str(getattr(self, "_runtime_lease_path", "")),
            "queue_depth": int(self._queue.qsize()),
            "thread_heartbeat_age_seconds": hb_age,
            "recent_rejections_window": int(len(self._recent_rejections)),
        }
        try:
            lease_client = self._get_runtime_lease_client()
            if lease_client is not None:
                record = lease_client.read()
                if isinstance(record, dict) and record:
                    snapshot["runtime"]["lease_record"] = record
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Runtime lease snapshot read failed: %s", e)
        snapshot["execution_quality"] = self._get_execution_quality_snapshot()
        with self._synthetic_exit_lock:
            snapshot["synthetic_exits"] = {
                "active_plans": int(len(self._synthetic_exits)),
                "plans": list(self._synthetic_exits.values())[:50],
                "state_file": str(
                    getattr(self, "_synthetic_exit_state_path", "")
                ),
            }
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.warning("Execution snapshot build degraded: %s", e)
    return snapshot


def _get_execution_quality_snapshot(self) -> dict[str, object]:
    with self._exec_quality_lock:
        fills = int(self._exec_quality.get("fills", 0) or 0)
        sum_signed = float(self._exec_quality.get("slippage_bps_sum", 0.0) or 0.0)
        sum_abs = float(self._exec_quality.get("slippage_bps_abs_sum", 0.0) or 0.0)
        avg_signed = (sum_signed / fills) if fills > 0 else 0.0
        avg_abs = (sum_abs / fills) if fills > 0 else 0.0
        return {
            "fills": fills,
            "avg_signed_slippage_bps": avg_signed,
            "avg_abs_slippage_bps": avg_abs,
            "by_reason": dict(self._exec_quality.get("by_reason", {})),
            "last_update": str(self._exec_quality.get("last_update", "")),
        }


def _watchdog_loop(self) -> None:
    """Watchdog for core execution threads.
    On heartbeat stall, pause auto-trader and report degraded health.
    """
    stall_seconds = float(getattr(CONFIG, "runtime_watchdog_stall_seconds", 25.0) or 25.0)
    stall_seconds = max(8.0, stall_seconds)
    while self._running:
        if self._wait_or_stop(2.0):
            break
        now = time.time()
        self._heartbeat("watchdog")
        stalled: list[str] = []
        with self._thread_hb_lock:
            for name in ("exec", "fill_sync", "status_sync", "recon"):
                ts = float(self._thread_heartbeats.get(name, 0.0) or 0.0)
                if ts <= 0.0:
                    continue
                age = now - ts
                set_gauge("runtime_thread_heartbeat_age_seconds", age, labels={"thread": name})
                if age > stall_seconds:
                    stalled.append(f"{name}:{age:.1f}s")
        set_gauge("runtime_queue_depth", float(self._queue.qsize()))

        if not stalled:
            continue

        # Throttle repeated alerts.
        if (now - self._last_watchdog_warning_ts) < 15.0:
            continue
        self._last_watchdog_warning_ts = now
        msg = f"Watchdog stall detected ({', '.join(stalled)})"
        log.warning(msg)
        try:
            self._health_monitor.report_component_health(
                ComponentType.RISK_MANAGER,
                HealthStatus.DEGRADED,
                error=msg,
            )
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Watchdog health-report update failed: %s", e)
        try:
            if self.auto_trader is not None:
                self.auto_trader.pause(msg, duration_seconds=300)
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Watchdog auto-trader pause failed: %s", e)
        try:
            self._alert_manager.risk_alert("Runtime watchdog", msg)
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Watchdog alert dispatch failed: %s", e)


def _order_ops():
    from trading import executor_order_ops

    return executor_order_ops


def submit(self, signal: TradeSignal) -> bool:
    return _order_ops().submit(self, signal)


def _execute(self, signal: TradeSignal):
    return _order_ops()._execute(self, signal)


def _startup_sync(self):
    return _order_ops()._startup_sync(self)


def _process_pending_fills(self):
    return _order_ops()._process_pending_fills(self)


def _cancel_oco_siblings(self, oms, filled_order: Order, fill: Fill) -> None:
    return _order_ops()._cancel_oco_siblings(self, oms, filled_order, fill)


def _record_execution_quality(self, order: Order, fill: Fill) -> None:
    return _order_ops()._record_execution_quality(self, order, fill)


def _maybe_register_synthetic_exit(self, order: Order, fill: Fill) -> None:
    return _order_ops()._maybe_register_synthetic_exit(self, order, fill)


def _evaluate_synthetic_exits(self) -> None:
    return _order_ops()._evaluate_synthetic_exits(self)


def _submit_synthetic_exit(
    self,
    plan: dict[str, Any],
    trigger_price: float,
    reason: str,
) -> bool:
    return _order_ops()._submit_synthetic_exit(
        self,
        plan=plan,
        trigger_price=trigger_price,
        reason=reason,
    )


def _status_sync_loop(self):
    return _order_ops()._status_sync_loop(self)


