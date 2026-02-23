# trading/executor.py
from __future__ import annotations

import os
import queue
import socket
import threading
import time
import weakref
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from config import CONFIG, TradingMode
from core.types import (
    Account,
    AutoTradeMode,
    AutoTradeState,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TradeSignal,
)
from trading import executor_auto_ops as _executor_auto_ops
from trading.alerts import AlertPriority
from trading.auto_trader import AutoTrader
from trading.executor_core_ops import __init__ as _exec_init_impl
from trading.executor_core_ops import _build_execution_snapshot as _build_execution_snapshot_impl
from trading.executor_core_ops import _cancel_oco_siblings as _cancel_oco_siblings_impl
from trading.executor_core_ops import _evaluate_synthetic_exits as _evaluate_synthetic_exits_impl
from trading.executor_core_ops import _execute as _execute_impl
from trading.executor_core_ops import (
    _get_execution_quality_snapshot as _get_execution_quality_snapshot_impl,
)
from trading.executor_core_ops import (
    _maybe_register_synthetic_exit as _maybe_register_synthetic_exit_impl,
)
from trading.executor_core_ops import _process_pending_fills as _process_pending_fills_impl
from trading.executor_core_ops import _record_execution_quality as _record_execution_quality_impl
from trading.executor_core_ops import _startup_sync as _startup_sync_impl
from trading.executor_core_ops import _status_sync_loop as _status_sync_loop_impl
from trading.executor_core_ops import _submit_synthetic_exit as _submit_synthetic_exit_impl
from trading.executor_core_ops import _watchdog_loop as _watchdog_loop_impl
from trading.executor_core_ops import start as _start_impl
from trading.executor_core_ops import stop as _stop_impl
from trading.executor_core_ops import submit as _submit_impl
from trading.executor_error_policy import SOFT_FAIL_EXCEPTIONS
from trading.executor_reconcile_ops import _reconciliation_loop as _reconciliation_loop_impl
from trading.executor_reconcile_ops import _submit_with_retry as _submit_with_retry_impl
from trading.health import HealthStatus, SystemHealth
from trading.runtime_lease import RuntimeLeaseClient, create_runtime_lease_client
from utils.atomic_io import atomic_write_json, read_json
from utils.logger import get_logger
from utils.metrics import inc_counter, set_gauge
from utils.security import get_access_control, get_audit_log  # noqa: F401

try:
    from utils.metrics_http import register_snapshot_provider, unregister_snapshot_provider
except (ImportError, OSError):  # pragma: no cover - optional runtime integration
    register_snapshot_provider = None
    unregister_snapshot_provider = None
log = get_logger(__name__)
_SOFT_FAIL_EXCEPTIONS = SOFT_FAIL_EXCEPTIONS

__all__ = [
    "ExecutionEngine",
    "AutoTrader",
]
# AUTO-TRADER
class ExecutionEngine:
    """Production execution engine with correct broker synchronization.

    DESIGN PRINCIPLES:
    1. Fills are ONLY processed from broker.get_fills() - never fabricated
    2. OMS is the single source of truth for order state
    3. Broker ID mapping is persisted through OMS for crash recovery
    4. Status sync captures previous state before mutation

    AUTO-TRADE INTEGRATION:
    - Owns an AutoTrader instance
    - Provides start_auto_trade() / stop_auto_trade() / set_auto_mode()
    - AutoTrader submits through self.submit() so all risk checks apply
    """

    # Configurable: whether to auto-cancel stuck orders
    AUTO_CANCEL_STUCK_ORDERS: bool = False
    # Watermark overlap to avoid missing same-timestamp fills
    _FILL_WATERMARK_OVERLAP_SECONDS: float = 2.0
    _RUNTIME_STATE_FILE: str = "execution_runtime_state.json"
    _RUNTIME_LEASE_FILE: str = "execution_runtime_lease.json"
    _RUNTIME_LEASE_DB_FILE: str = "execution_runtime_lease.db"
    _SYNTHETIC_EXITS_FILE: str = "synthetic_exits_state.json"
    _ACTIVE_ENGINES_LOCK = threading.RLock()
    _ACTIVE_ENGINES: weakref.WeakSet[ExecutionEngine] = weakref.WeakSet()

    def __init__(self, mode: TradingMode = None) -> None:
        _exec_init_impl(self, mode=mode)

    @classmethod
    def trigger_model_drift_alarm(
        cls,
        reason: str,
        *,
        severity: str = "critical",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        return _executor_auto_ops.trigger_model_drift_alarm(
            cls,
            reason,
            severity=severity,
            metadata=metadata,
        )

    def _apply_model_drift_alarm(
        self,
        reason: str,
        *,
        status: HealthStatus = HealthStatus.UNHEALTHY,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        return _executor_auto_ops._apply_model_drift_alarm(
            self,
            reason,
            status=status,
            metadata=metadata,
        )

    # -----------------------------------------------------------------
    # Auto-trade public API
    # -----------------------------------------------------------------

    def init_auto_trader(self, predictor: Any, watch_list: list[str]) -> None:
        _executor_auto_ops.init_auto_trader(
            self,
            predictor=predictor,
            watch_list=watch_list,
        )

    def start_auto_trade(self, mode: AutoTradeMode = AutoTradeMode.AUTO) -> None:
        _executor_auto_ops.start_auto_trade(self, mode=mode)

    def stop_auto_trade(self) -> None:
        _executor_auto_ops.stop_auto_trade(self)

    def set_auto_mode(self, mode: AutoTradeMode) -> None:
        _executor_auto_ops.set_auto_mode(self, mode=mode)

    def get_auto_trade_state(self) -> AutoTradeState | None:
        return _executor_auto_ops.get_auto_trade_state(self)

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------

    def _evaluate_live_start_readiness(self) -> tuple[bool, str]:
        """Check institutional controls before enabling LIVE execution.

        Returns:
            (ok, message). message is empty when ok=True.
        """
        if self.mode != TradingMode.LIVE:
            return True, ""

        strict = bool(
            getattr(getattr(CONFIG, "security", None), "strict_live_governance", False)
        )
        try:
            from utils.institutional import collect_institutional_readiness

            report = collect_institutional_readiness()
            if bool(report.get("pass", False)):
                return True, ""

            failed = report.get("failed_required_controls", [])
            if isinstance(failed, list):
                failed_controls = [str(x).strip() for x in failed if str(x).strip()]
            else:
                failed_controls = []

            if failed_controls:
                preview = ", ".join(failed_controls[:6])
                if len(failed_controls) > 6:
                    preview += f", +{len(failed_controls) - 6} more"
                msg = f"Institutional readiness failed: {preview}"
            else:
                msg = "Institutional readiness failed"

            if strict:
                return False, msg

            log.warning("%s (strict_live_governance=False; continuing)", msg)
            return True, msg
        except _SOFT_FAIL_EXCEPTIONS as e:
            msg = f"Institutional readiness check error: {e}"
            if strict:
                return False, msg
            log.warning("%s (strict_live_governance=False; continuing)", msg)
            return True, msg
    def _load_processed_fills(self) -> set[str]:
        """Load already-processed fill IDs from OMS DB (source of truth)."""
        try:
            from trading.oms import get_oms

            oms = get_oms()
            fills = oms.get_fills()
            out = set()
            for f in fills:
                fid = getattr(f, "id", None)
                if fid:
                    out.add(str(fid))
            return out
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.warning(f"Could not load processed fills: {e}")
            return set()

    @staticmethod
    def _normalize_synthetic_exit_plan(plan: dict[str, Any]) -> dict[str, Any] | None:
        """Validate/normalize one synthetic exit plan row from disk/runtime."""
        if not isinstance(plan, dict):
            return None
        try:
            plan_id = str(plan.get("plan_id", "") or "").strip()
            source_order_id = str(plan.get("source_order_id", "") or plan_id).strip()
            symbol = str(plan.get("symbol", "") or "").strip()
            if not plan_id or not symbol:
                return None

            open_qty = int(plan.get("open_qty", 0) or 0)
            if open_qty <= 0:
                return None

            stop_loss = max(0.0, float(plan.get("stop_loss", 0.0) or 0.0))
            take_profit = max(0.0, float(plan.get("take_profit", 0.0) or 0.0))
            trailing_stop_pct = max(
                0.0, float(plan.get("trailing_stop_pct", 0.0) or 0.0)
            )
            if stop_loss <= 0 and take_profit <= 0 and trailing_stop_pct <= 0:
                return None

            highest_price = max(
                0.0,
                float(plan.get("highest_price", 0.0) or 0.0),
            )
            armed_at = str(plan.get("armed_at", "") or "").strip()
            if not armed_at:
                armed_at = datetime.now().isoformat()

            return {
                "plan_id": plan_id,
                "source_order_id": source_order_id or plan_id,
                "symbol": symbol,
                "side": str(plan.get("side", "long") or "long"),
                "open_qty": open_qty,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop_pct": trailing_stop_pct,
                "highest_price": highest_price,
                "armed_at": armed_at,
            }
        except (TypeError, ValueError) as e:
            log.debug("Synthetic exit plan normalize failed for %r: %s", plan, e)
            return None

    def _persist_synthetic_exits(self, force: bool = False) -> None:
        """Persist synthetic exit plans atomically.

        This makes synthetic brackets/OCO plans recoverable after restarts.
        """
        path_attr = getattr(self, "_synthetic_exit_state_path", None)
        if path_attr is None:
            return
        path = Path(path_attr)

        now = time.time()
        min_interval = float(
            getattr(self, "_synthetic_persist_min_interval_s", 1.0) or 1.0
        )
        last_ts = float(getattr(self, "_last_synthetic_persist_ts", 0.0) or 0.0)
        if not force and (now - last_ts) < max(0.2, min_interval):
            return

        lock = getattr(self, "_synthetic_exit_lock", None)
        plans: list[dict[str, Any]]
        if lock is None:
            raw = list(getattr(self, "_synthetic_exits", {}).values())
        else:
            with lock:
                raw = list(getattr(self, "_synthetic_exits", {}).values())

        plans = []
        for p in raw:
            norm = self._normalize_synthetic_exit_plan(p)
            if norm is not None:
                plans.append(norm)

        payload = {
            "ts": datetime.now().isoformat(),
            "plans": plans,
        }
        try:
            atomic_write_json(path, payload, indent=2)
            self._last_synthetic_persist_ts = now
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug(f"Synthetic exit state persist failed: {e}")

    def _restore_synthetic_exits(self) -> None:
        """Best-effort restore synthetic exit plans from prior session."""
        path_attr = getattr(self, "_synthetic_exit_state_path", None)
        if path_attr is None:
            return
        path = Path(path_attr)
        if not path.exists():
            return
        try:
            payload = read_json(path)
            raw_plans = []
            if isinstance(payload, dict):
                raw_plans = list(payload.get("plans", []) or [])

            restored: dict[str, dict[str, Any]] = {}
            for row in raw_plans:
                norm = self._normalize_synthetic_exit_plan(row)
                if norm is None:
                    continue
                restored[str(norm["plan_id"])] = norm

            lock = getattr(self, "_synthetic_exit_lock", None)
            if lock is None:
                self._synthetic_exits = restored
            else:
                with lock:
                    self._synthetic_exits = restored

            if restored:
                log.warning(
                    "Recovered %d synthetic exit plan(s) from %s",
                    len(restored),
                    path.name,
                )
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug(f"Synthetic exit state restore failed: {e}")

    def _resolve_price(self, symbol: str, hinted_price: float = 0.0) -> float:
        """Resolve one authoritative price for this submission."""
        try:
            px = float(hinted_price or 0.0)
            if px > 0:
                return px
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Hinted price parse failed for %s: %s", symbol, e)

        try:
            from data.feeds import get_feed_manager

            fm = get_feed_manager(auto_init=False)
            q = fm.get_quote(symbol)
            if q and getattr(q, "price", 0) and float(q.price) > 0:
                return float(q.price)
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Feed price resolve failed for %s: %s", symbol, e)

        try:
            px = self.broker.get_quote(symbol)
            if px and float(px) > 0:
                return float(px)
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Broker price resolve failed for %s: %s", symbol, e)

        try:
            from data.fetcher import get_fetcher

            q = get_fetcher().get_realtime(symbol)
            if q and getattr(q, "price", 0) and float(q.price) > 0:
                return float(q.price)
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Fetcher price resolve failed for %s: %s", symbol, e)

        return 0.0

    def _rebuild_broker_mappings(self, oms: Any) -> None:
        """Rebuild broker ID mappings from persisted orders after restart."""
        try:
            active_orders = oms.get_active_orders()
            for order in active_orders:
                if order.broker_id:
                    self.broker.register_order_mapping(order.id, order.broker_id)
                    log.debug(f"Recovered mapping: {order.id} -> {order.broker_id}")
            log.info(f"Recovered {len(active_orders)} order mappings from DB")
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.warning(f"Failed to rebuild broker mappings: {e}")

    def _join_worker_threads(self, timeout_seconds: float = 5.0) -> None:
        """Join worker threads and log any that fail to stop cleanly."""
        threads = [
            self._exec_thread,
            self._fill_sync_thread,
            self._status_sync_thread,
            self._recon_thread,
            self._reconnect_thread,
            self._watchdog_thread,
            self._checkpoint_thread,
        ]
        deadline = time.monotonic() + max(0.2, float(timeout_seconds))

        for thread in threads:
            if thread is None or not thread.is_alive():
                continue
            remaining = max(0.05, deadline - time.monotonic())
            thread.join(timeout=remaining)
            if thread.is_alive():
                log.warning(
                    "Worker thread did not stop within %.1fs: %s",
                    float(timeout_seconds),
                    str(thread.name),
                )

        self._exec_thread = None
        self._fill_sync_thread = None
        self._status_sync_thread = None
        self._recon_thread = None
        self._reconnect_thread = None
        self._watchdog_thread = None
        self._checkpoint_thread = None

    def _heartbeat(self, name: str) -> None:
        """Record thread heartbeat for watchdog + observability."""
        with self._thread_hb_lock:
            self._thread_heartbeats[str(name)] = time.time()

    def _wait_or_stop(self, timeout_seconds: float) -> bool:
        """Wait for timeout or stop request. Returns True when stopping."""
        return bool(self._stop_event.wait(timeout=max(0.0, float(timeout_seconds))))

    def _runtime_state_payload(self, clean_shutdown: bool) -> dict[str, object]:
        """Persistable runtime checkpoint for crash recovery."""
        auto_state = None
        if self.auto_trader is not None:
            try:
                auto_state = self.auto_trader.get_state()
            except _SOFT_FAIL_EXCEPTIONS as e:
                log.debug("Auto-trader runtime-state snapshot unavailable: %s", e)
                auto_state = None
        with self._thread_hb_lock:
            hb = dict(self._thread_heartbeats)
        return {
            "ts": datetime.now().isoformat(),
            "running": bool(self._running),
            "clean_shutdown": bool(clean_shutdown),
            "mode": str(getattr(self.mode, "value", self.mode)),
            "runtime_lease_id": str(self._runtime_lease_id),
            "runtime_lease_backend": str(getattr(self, "_runtime_lease_backend", "file")),
            "runtime_lease_cluster": str(getattr(self, "_runtime_lease_cluster", "execution_engine")),
            "runtime_lease_fencing_token": int(getattr(self, "_runtime_lease_fencing_token", 0) or 0),
            "queue_depth": int(self._queue.qsize()),
            "auto_trade_state": (
                auto_state.to_dict() if hasattr(auto_state, "to_dict") else None
            ),
            "thread_heartbeats": hb,
        }

    def _runtime_lease_payload(self) -> dict[str, object]:
        return {
            "owner_id": str(self._runtime_lease_id),
            "pid": int(os.getpid()),
            "host": str(socket.gethostname()),
            "mode": str(getattr(self.mode, "value", self.mode)),
            "heartbeat_ts": float(time.time()),
            "cluster": str(getattr(self, "_runtime_lease_cluster", "execution_engine")),
            "backend": str(getattr(self, "_runtime_lease_backend", "file")),
        }

    def _get_runtime_lease_client(self) -> RuntimeLeaseClient | None:
        client = getattr(self, "_runtime_lease_client", None)
        if client is not None:
            return client
        try:
            backend = str(getattr(self, "_runtime_lease_backend", "file") or "file").strip().lower()
            cluster = str(getattr(self, "_runtime_lease_cluster", "execution_engine") or "execution_engine").strip()
            path_attr = getattr(self, "_runtime_lease_path", None)
            if path_attr is None:
                name = (
                    self._RUNTIME_LEASE_DB_FILE
                    if backend == "sqlite"
                    else self._RUNTIME_LEASE_FILE
                )
                path = Path(CONFIG.data_dir) / name
                self._runtime_lease_path = path
            else:
                path = Path(path_attr)
            client = create_runtime_lease_client(
                backend=backend,
                cluster=cluster,
                path=path,
            )
            self._runtime_lease_client = client
            return client
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.warning("Runtime lease client init failed: %s", e)
            return None

    def _acquire_runtime_lease(self) -> bool:
        """Acquire a local single-writer runtime lease.

        This prevents split-brain auto execution when two engine processes
        run against the same working directory.
        """
        if not self._runtime_lease_enabled:
            return True

        try:
            client = self._get_runtime_lease_client()
            if client is None:
                return False
            result = client.acquire(
                owner_id=self._runtime_lease_id,
                ttl_seconds=self._runtime_lease_ttl_seconds,
                metadata=self._runtime_lease_payload(),
            )
            self._runtime_lease_owner_hint = (
                dict(result.record or {})
                if isinstance(result.record, dict)
                else None
            )
            if result.ok:
                row = dict(result.record or {})
                self._runtime_lease_fencing_token = int(row.get("generation", 0) or 0)
                self._runtime_lease_owner_hint = None
                return True
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.warning(f"Runtime lease acquire failed: {e}")
            return False

        return False

    def _refresh_runtime_lease(self) -> bool:
        if not self._runtime_lease_enabled:
            return True
        try:
            client = self._get_runtime_lease_client()
            if client is None:
                return False
            result = client.refresh(
                owner_id=self._runtime_lease_id,
                ttl_seconds=self._runtime_lease_ttl_seconds,
                metadata=self._runtime_lease_payload(),
            )
            self._runtime_lease_owner_hint = (
                dict(result.record or {})
                if isinstance(result.record, dict)
                else None
            )
            if result.ok:
                row = dict(result.record or {})
                self._runtime_lease_fencing_token = int(row.get("generation", 0) or 0)
                self._runtime_lease_owner_hint = None
                return True
            return False
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.warning(f"Runtime lease refresh failed: {e}")
            return False

    def _release_runtime_lease(self) -> None:
        if not self._runtime_lease_enabled:
            return
        try:
            client = self._get_runtime_lease_client()
            if client is None:
                return
            client.release(
                owner_id=self._runtime_lease_id,
                metadata={"released_by": self._runtime_lease_id},
            )
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.warning("Runtime lease release failed: %s", e)

    def _persist_runtime_state(self, clean_shutdown: bool = False) -> None:
        """Write runtime checkpoint atomically."""
        try:
            payload = self._runtime_state_payload(clean_shutdown=clean_shutdown)
            atomic_write_json(self._runtime_state_path, payload, indent=2)
            self._last_checkpoint_ts = time.time()
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug(f"Runtime checkpoint write failed: {e}")

    def _restore_runtime_state(self) -> None:
        """Best-effort restore markers from prior run for HA/DR awareness."""
        try:
            if not self._runtime_state_path.exists():
                return
            state = read_json(self._runtime_state_path)
            if not isinstance(state, dict):
                return
            was_running = bool(state.get("running", False))
            clean_shutdown = bool(state.get("clean_shutdown", False))
            if was_running and not clean_shutdown:
                self._runtime_recovered = True
                self._recovered_auto_state = state.get("auto_trade_state")
                log.warning(
                    "Recovered unclean runtime state from previous session; "
                    "autonomous trading will start in safe mode until reviewed."
                )
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug(f"Runtime checkpoint restore failed: {e}")

    def _restore_auto_trader_state(self) -> None:
        """Apply recovered auto-trader pause markers after crash recovery."""
        if self.auto_trader is None or not isinstance(self._recovered_auto_state, dict):
            return
        try:
            recovered_mode = str(self._recovered_auto_state.get("mode", "manual")).lower()
            if recovered_mode in ("auto", "semi_auto"):
                self.auto_trader.pause(
                    "Recovered from unclean shutdown; manual review required.",
                    duration_seconds=0,
                )
                self.auto_trader.set_mode(AutoTradeMode.MANUAL)
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug(f"Auto-trader recovery state apply failed: {e}")

    def _checkpoint_loop(self) -> None:
        """Periodic runtime checkpoint loop for crash recovery."""
        interval = float(getattr(CONFIG, "runtime_checkpoint_seconds", 5.0) or 5.0)
        lease_interval = float(getattr(CONFIG, "runtime_lease_heartbeat_seconds", 5.0) or 5.0)
        interval = max(2.0, min(interval, 60.0))
        lease_interval = max(1.0, min(lease_interval, 30.0))
        last_lease = 0.0
        while self._running:
            try:
                self._heartbeat("checkpoint")
                self._persist_runtime_state(clean_shutdown=False)
                now = time.time()
                if (now - last_lease) >= lease_interval:
                    if not self._refresh_runtime_lease():
                        msg = "Runtime lease lost to another process; kill switch engaged"
                        log.critical(msg)
                        try:
                            self._kill_switch.activate(msg, activated_by="runtime_lease")
                        except _SOFT_FAIL_EXCEPTIONS as e:
                            log.critical(
                                "Kill switch activation failed after lease loss: %s", e
                            )
                    last_lease = now
            except _SOFT_FAIL_EXCEPTIONS as e:
                log.debug(f"Checkpoint loop error: {e}")
            if self._wait_or_stop(interval):
                break
    def _get_quote_snapshot(
        self, symbol: str
    ) -> tuple[float, datetime | None, str, bool]:
        """Returns (price, timestamp, source, is_delayed)."""
        # 1) feed cache
        try:
            from data.feeds import get_feed_manager

            fm = get_feed_manager(auto_init=False)
            q = fm.get_quote(symbol)
            if q and float(getattr(q, "price", 0.0) or 0.0) > 0:
                ts = getattr(q, "timestamp", None)
                delayed = bool(getattr(q, "is_delayed", False))
                return float(q.price), ts, "feed", delayed
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Feed quote snapshot failed for %s: %s", symbol, e)

        # 2) broker quote
        try:
            px = self.broker.get_quote(symbol)
            if px and float(px) > 0:
                return float(px), None, "broker", False
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Broker quote snapshot failed for %s: %s", symbol, e)

        # 3) fetcher realtime
        try:
            from data.fetcher import get_fetcher

            q = get_fetcher().get_realtime(symbol)
            if q and float(getattr(q, "price", 0.0) or 0.0) > 0:
                delayed = bool(getattr(q, "is_delayed", False))
                return (
                    float(q.price),
                    getattr(q, "timestamp", None),
                    f"fetcher:{getattr(q, 'source', '')}",
                    delayed,
                )
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Fetcher quote snapshot failed for %s: %s", symbol, e)

        return 0.0, None, "none", True

    @staticmethod
    def _coerce_quote_snapshot(
        snapshot: object,
    ) -> tuple[float, datetime | None, str, bool]:
        """Backward-compatible snapshot parser.

        Accepts both legacy 3-tuple ``(px, ts, src)`` and current 4-tuple.
        """
        if isinstance(snapshot, tuple):
            if len(snapshot) >= 4:
                px, ts, src, delayed = snapshot[:4]
                try:
                    return (
                        float(px or 0.0),
                        ts if isinstance(ts, datetime) else ts,
                        str(src or ""),
                        bool(delayed),
                    )
                except _SOFT_FAIL_EXCEPTIONS:
                    return 0.0, None, "none", True
            if len(snapshot) == 3:
                px, ts, src = snapshot
                try:
                    return (
                        float(px or 0.0),
                        ts if isinstance(ts, datetime) else ts,
                        str(src or ""),
                        False,
                    )
                except _SOFT_FAIL_EXCEPTIONS:
                    return 0.0, None, "none", True
        return 0.0, None, "none", True

    def _require_fresh_quote(
        self,
        symbol: str,
        max_age_seconds: float = 15.0,
        block_delayed: bool = False,
    ) -> tuple[bool, str, float]:
        """Strict quote freshness gate for order submission.
        Returns (ok, message, price).
        """
        px, ts, src, delayed = self._coerce_quote_snapshot(
            self._get_quote_snapshot(symbol)
        )
        if px <= 0:
            return False, "No valid quote", 0.0

        if block_delayed and delayed:
            return False, f"Quote delayed/stale (source={src})", 0.0

        # If no timestamp, be conservative in LIVE mode
        if ts is None:
            if block_delayed or str(self.mode.value).lower() == "live":
                return False, f"No timestamped quote (source={src})", 0.0
            return True, "OK", px

        try:
            now_ts = (
                datetime.now(tz=ts.tzinfo)
                if getattr(ts, "tzinfo", None) is not None
                else datetime.now()
            )
            age = (now_ts - ts).total_seconds()
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Quote timestamp age calculation failed for %s: %s", symbol, e)
            age = 0.0

        if age > float(max_age_seconds):
            return False, f"Quote stale: {age:.0f}s (source={src})", 0.0

        return True, "OK", px

    def check_quote_freshness(
        self, symbol: str
    ) -> tuple[bool, str, float]:
        """Public wrapper for quote freshness gating."""
        max_age = 15.0
        if hasattr(CONFIG, "risk") and hasattr(CONFIG.risk, "quote_staleness_seconds"):
            max_age = float(CONFIG.risk.quote_staleness_seconds)
        return self._require_fresh_quote(symbol, max_age_seconds=max_age)

    @staticmethod
    def _normalize_requested_order_type(raw: object) -> str:
        requested = str(raw or "limit").strip().lower().replace("-", "_")
        alias = {
            "trailing": "trail_market",
            "trailing_market": "trail_market",
            "trailing_stop": "trail_market",
            "trailing_limit": "trail_limit",
            "market_ioc": "ioc",
            "market_fok": "fok",
            "stop_loss": "stop",
        }
        requested = alias.get(requested, requested)
        valid = {t.value for t in OrderType}
        return requested if requested in valid else OrderType.LIMIT.value

    @staticmethod
    def _make_submit_fingerprint(signal: TradeSignal) -> str:
        """Deterministic key for duplicate submission suppression."""
        sym = str(getattr(signal, "symbol", "") or "").strip()
        side = str(getattr(getattr(signal, "side", None), "value", "") or "").strip()
        requested = ExecutionEngine._normalize_requested_order_type(
            getattr(signal, "order_type", "limit")
        )
        qty = int(getattr(signal, "quantity", 0) or 0)
        px = float(getattr(signal, "price", 0.0) or 0.0)
        trigger = float(getattr(signal, "trigger_price", 0.0) or 0.0)
        market_style = {"market", "ioc", "fok", "stop", "trail_market"}
        price_key = "mkt" if requested in market_style else f"{round(px, 4):.4f}"
        trigger_key = f"{round(trigger, 4):.4f}" if trigger > 0 else "0"
        return f"{sym}:{side}:{requested}:{qty}:{price_key}:{trigger_key}"

    def _check_submission_guardrails(
        self, signal: TradeSignal
    ) -> tuple[bool, str]:
        """Extra exchange-style guardrails:
        - duplicate suppression
        - per-symbol burst cap
        - max single-order notional cap.
        """
        now = time.time()
        risk_cfg = getattr(CONFIG, "risk", None)

        dedupe_seconds = float(getattr(risk_cfg, "duplicate_signal_cooldown_seconds", 5.0) or 5.0)
        per_symbol_per_min = int(getattr(risk_cfg, "max_orders_per_symbol_per_minute", 4) or 4)
        max_notional = float(getattr(risk_cfg, "max_single_order_value", 500000.0) or 500000.0)

        # Duplicate suppression
        key = self._make_submit_fingerprint(signal)
        last_seen = float(self._recent_submit_keys.get(key, 0.0) or 0.0)
        if last_seen > 0 and (now - last_seen) < dedupe_seconds:
            return False, f"Duplicate signal suppressed ({now - last_seen:.1f}s)"
        self._recent_submit_keys[key] = now

        # Trim old dedupe keys
        expire_before = now - max(30.0, dedupe_seconds * 4.0)
        stale_keys = [k for k, ts in self._recent_submit_keys.items() if float(ts) < expire_before]
        for k in stale_keys[:500]:
            self._recent_submit_keys.pop(k, None)

        # Per-symbol burst cap
        sym = str(getattr(signal, "symbol", "") or "").strip()
        bucket = self._recent_submissions.get(sym)
        if bucket is None:
            bucket = deque()
            self._recent_submissions[sym] = bucket
        while bucket and (now - float(bucket[0])) > 60.0:
            bucket.popleft()
        if len(bucket) >= max(1, per_symbol_per_min):
            return False, f"Per-symbol order burst limit ({per_symbol_per_min}/min)"
        bucket.append(now)

        # Max single order notional
        qty = int(getattr(signal, "quantity", 0) or 0)
        px = float(getattr(signal, "price", 0.0) or 0.0)
        notional = max(0.0, qty * px)
        if max_notional > 0 and notional > max_notional:
            return False, f"Single-order notional {notional:,.2f} exceeds {max_notional:,.2f}"

        return True, ""

    def _record_rejection_guardrail(self, reason: str) -> None:
        """Track reject bursts and trip kill-switch on persistent failures."""
        now = time.time()
        self._recent_rejections.append(now)
        while self._recent_rejections and (now - float(self._recent_rejections[0])) > 120.0:
            self._recent_rejections.popleft()
        set_gauge("order_rejections_window", float(len(self._recent_rejections)))

        threshold = int(getattr(getattr(CONFIG, "risk", None), "reject_kill_switch_threshold", 12) or 12)
        if len(self._recent_rejections) < max(3, threshold):
            return
        try:
            if self._kill_switch.can_trade:
                msg = f"Excessive rejections detected ({len(self._recent_rejections)} in 120s)"
                self._kill_switch.activate(msg, activated_by="execution_guardrail")
                get_audit_log().log_risk_event(
                    "execution_reject_kill_switch",
                    {"count": int(len(self._recent_rejections)), "last_reason": str(reason)},
                )
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug(f"Reject kill-switch guard failed: {e}")

    def _on_health_degraded(self, health: SystemHealth) -> None:
        """Runbook automation: pause autonomous trading on degraded health."""
        try:
            sec_cfg = getattr(CONFIG, "security", None)
            if not bool(getattr(sec_cfg, "auto_pause_auto_trader_on_degraded", True)):
                return
            if self.auto_trader and self.auto_trader.get_mode() != AutoTradeMode.MANUAL:
                self.auto_trader.set_mode(AutoTradeMode.MANUAL)
                log.warning("Auto-trader paused due to degraded system health")
                self._alert_manager.risk_alert(
                    "Auto-trader paused",
                    f"System degraded (status={health.status.value})",
                )
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug(f"Degraded health handler failed: {e}")

    def submit_from_prediction(self, pred: Any) -> bool:
        """Submit order from AI prediction."""
        from models.predictor import Signal as UiSignal
        if pred.signal == UiSignal.HOLD or pred.position.shares == 0:
            return False
        side = (
            OrderSide.BUY
            if pred.signal in (UiSignal.STRONG_BUY, UiSignal.BUY)
            else OrderSide.SELL
        )
        signal = TradeSignal(
            symbol=pred.stock_code,
            name=pred.stock_name,
            side=side,
            quantity=int(pred.position.shares),
            price=float(pred.levels.entry),
            stop_loss=float(pred.levels.stop_loss) if pred.levels.stop_loss else 0.0,
            take_profit=(
                float(pred.levels.target_2) if pred.levels.target_2 else 0.0
            ),
            confidence=float(pred.confidence),
            reasons=list(pred.reasons),
        )
        return self.submit(signal)

    def _execution_loop(self) -> None:
        """Main execution loop."""
        last_risk_update = 0.0
        while self._running:
            self._heartbeat("exec")
            try:
                signal = self._queue.get(timeout=0.2)
                if signal is None:
                    break
                self._execute(signal)
            except queue.Empty:
                pass
            except _SOFT_FAIL_EXCEPTIONS as e:
                log.error(f"Execution loop error: {e}")
                self._alert_manager.system_alert(
                    "Execution Loop Error", str(e), AlertPriority.HIGH
                )

            now = time.time()
            if (
                self.risk_manager
                and self.broker.is_connected
                and (now - last_risk_update) >= 1.0
            ):
                try:
                    account = self.broker.get_account()
                    self.risk_manager.update(account)
                    set_gauge("account_equity", account.equity)
                    set_gauge("account_cash", account.cash)
                    set_gauge("positions_count", len(account.positions))
                    last_risk_update = now
                except _SOFT_FAIL_EXCEPTIONS as e:
                    log.warning(f"Risk update error: {e}")

            if self._wait_or_stop(0.05):
                break

    def get_risk_metrics(self) -> dict[str, Any] | None:
        if self.risk_manager:
            return self.risk_manager.get_metrics()
        return None
    def _update_auto_trade_fill(self, order: Order, fill: Fill) -> None:
        """Update auto-trader state with fill information."""
        if not self.auto_trader:
            return
        action_id = order.tags.get("auto_trade_action_id", "")
        if not action_id:
            return
        with self.auto_trader._lock:
            for action in self.auto_trader.state.recent_actions:
                if action.id == action_id:
                    action.fill_price = fill.price
                    action.fill_quantity += fill.quantity
                    action.order_id = order.id
                    break

    def _fill_sync_loop(self) -> None:
        """Poll broker for fills."""
        while self._running:
            self._heartbeat("fill_sync")
            try:
                if self._wait_or_stop(1.0):
                    break
                if not self.broker.is_connected:
                    continue
                self._process_pending_fills()
                self._evaluate_synthetic_exits()
            except _SOFT_FAIL_EXCEPTIONS as e:
                log.error(f"Fill sync loop error: {e}")

    def _prune_processed_fills_unlocked(self, max_size: int = 50000) -> None:
        """Prevent unbounded growth of processed fill IDs.
        MUST be called with self._fills_lock already held.

        FIX: Single lock acquisition instead of double-lock pattern.
        """
        if len(self._processed_fill_ids) <= int(max_size):
            return

        try:
            from trading.oms import get_oms

            oms = get_oms()
            fills = oms.get_fills()
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Processed-fill pruning fallback (OMS unavailable): %s", e)
            fills = []

        keep = set()
        for f in fills[:int(max_size)]:
            fid = str(getattr(f, "id", "") or "").strip()
            if fid:
                keep.add(fid)

        self._processed_fill_ids = keep

    def _broker_reconnect_loop(self) -> None:
        """Reconnect with exponential backoff."""
        backoff = 1.0
        while self._running:
            self._heartbeat("reconnect")
            try:
                if self._wait_or_stop(2.0):
                    break
                if self.broker.is_connected:
                    backoff = 1.0
                    continue

                log.warning(
                    f"Broker disconnected. Reconnecting in {backoff:.0f}s..."
                )
                if self._wait_or_stop(backoff):
                    break

                try:
                    ok = self.broker.connect(
                        exe_path=(
                            getattr(CONFIG, "broker_path", "")
                            or getattr(CONFIG, "BROKER_PATH", "")
                        )
                    )
                except TypeError:
                    ok = self.broker.connect()

                if ok:
                    log.info("Broker reconnected successfully")
                    backoff = 1.0
                    self._startup_sync()
                else:
                    backoff = min(backoff * 2.0, 60.0)

            except _SOFT_FAIL_EXCEPTIONS as e:
                log.warning(f"Reconnect loop error: {e}")
                backoff = min(backoff * 2.0, 60.0)

    def _on_kill_switch(self, reason: str) -> None:
        """Handle kill switch activation."""
        log.critical(f"Kill switch activated: {reason}")

        # Stop auto-trading immediately
        if self.auto_trader:
            try:
                self.auto_trader.pause(f"Kill switch: {reason}")
            except _SOFT_FAIL_EXCEPTIONS as e:
                log.warning("Failed to pause auto-trader during kill switch: %s", e)

        try:
            for order in self.broker.get_orders(active_only=True):
                try:
                    self.broker.cancel_order(order.id)
                except _SOFT_FAIL_EXCEPTIONS as cancel_err:
                    log.warning(
                        "Kill-switch cancel failed for order_id=%s: %s",
                        order.id,
                        cancel_err,
                    )
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.error(f"Failed to cancel orders: {e}")

        try:
            while not self._queue.empty():
                self._queue.get_nowait()
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Kill-switch queue drain interrupted: %s", e)

        self._alert_manager.critical_alert(
            "KILL SWITCH ACTIVATED", f"All trading halted: {reason}"
        )

    def _reject_signal(self, signal: TradeSignal, reason: str) -> None:
        """Handle signal rejection."""
        log.warning(f"Signal rejected: {signal.symbol} - {reason}")
        inc_counter(
            "signals_rejected_total",
            labels={"symbol": str(getattr(signal, "symbol", ""))[:12]},
        )
        self._record_rejection_guardrail(reason)
        if self.on_reject:
            order = Order(
                symbol=signal.symbol,
                side=signal.side,
                quantity=signal.quantity,
                price=signal.price,
            )
            order.status = OrderStatus.REJECTED
            order.message = reason
            self.on_reject(order, reason)

    def get_account(self) -> Account:
        from trading.oms import get_oms

        return get_oms().get_account()

    def get_positions(self) -> Any:
        from trading.oms import get_oms

        return get_oms().get_positions()

    def get_orders(self) -> Any:
        return self.broker.get_orders()
ExecutionEngine.start = _start_impl
ExecutionEngine.stop = _stop_impl
ExecutionEngine._build_execution_snapshot = _build_execution_snapshot_impl
ExecutionEngine._get_execution_quality_snapshot = _get_execution_quality_snapshot_impl
ExecutionEngine._watchdog_loop = _watchdog_loop_impl
ExecutionEngine.submit = _submit_impl
ExecutionEngine._execute = _execute_impl
ExecutionEngine._startup_sync = _startup_sync_impl
ExecutionEngine._process_pending_fills = _process_pending_fills_impl
ExecutionEngine._cancel_oco_siblings = _cancel_oco_siblings_impl
ExecutionEngine._record_execution_quality = _record_execution_quality_impl
ExecutionEngine._maybe_register_synthetic_exit = _maybe_register_synthetic_exit_impl
ExecutionEngine._evaluate_synthetic_exits = _evaluate_synthetic_exits_impl
ExecutionEngine._submit_synthetic_exit = _submit_synthetic_exit_impl
ExecutionEngine._status_sync_loop = _status_sync_loop_impl
ExecutionEngine._reconciliation_loop = _reconciliation_loop_impl
ExecutionEngine._submit_with_retry = _submit_with_retry_impl
