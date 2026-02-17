# trading/executor.py
from __future__ import annotations

import copy
import os
import queue
import socket
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from config import CONFIG, TradingMode
from core.types import (
    Account,
    AutoTradeAction,
    AutoTradeMode,
    AutoTradeState,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TradeSignal,
)
from trading.alerts import AlertPriority, get_alert_manager
from trading.broker import BrokerInterface, create_broker
from trading.health import ComponentType, HealthStatus, get_health_monitor
from trading.kill_switch import get_kill_switch
from trading.risk import RiskManager, get_risk_manager
from trading.runtime_lease import RuntimeLeaseClient, create_runtime_lease_client
from utils.atomic_io import atomic_write_json, read_json
from utils.logger import get_logger
from utils.metrics import inc_counter, observe, set_gauge
from utils.policy import get_trade_policy_engine
from utils.security import get_access_control, get_audit_log

try:
    from utils.metrics_http import register_snapshot_provider, unregister_snapshot_provider
except Exception:  # pragma: no cover - optional runtime integration
    register_snapshot_provider = None
    unregister_snapshot_provider = None

log = get_logger(__name__)

# AUTO-TRADER

class AutoTrader:
    """
    Autonomous trading engine that scans the watchlist using the AI
    predictor and submits orders via ExecutionEngine when signals meet
    the configured thresholds.

    Lifecycle:
        auto_trader = AutoTrader(engine, predictor, watchlist)
        auto_trader.set_mode(AutoTradeMode.AUTO)
        auto_trader.start()
        ...
        auto_trader.stop()

    Thread Safety:
        - All public methods acquire ``_lock``
        - The scan loop runs on a dedicated daemon thread
        - State mutations are atomic under the lock
        - UI reads ``state`` via ``get_state()`` which returns a snapshot

    Modes:
        MANUAL:    Scan loop does NOT run. All trading is user-initiated.
        AUTO:      Scan loop runs. Qualifying signals auto-execute.
        SEMI_AUTO: Scan loop runs. Qualifying signals queued as pending
                   approvals for one-click accept/reject in the UI.
    """

    # Safety: max consecutive errors before auto-pause
    MAX_CONSECUTIVE_ERRORS: int = 5
    ERROR_PAUSE_SECONDS: int = 300  # 5 min pause after too many errors

    def __init__(
        self,
        engine: ExecutionEngine,
        predictor,
        watch_list: list[str],
    ):
        self._engine = engine
        self._predictor = predictor
        self._watch_list = list(watch_list)

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.state = AutoTradeState()

        self.on_action: Callable[[AutoTradeAction], None] | None = None
        self.on_state_changed: Callable[[AutoTradeState], None] | None = None
        self.on_pending_approval: Callable[[AutoTradeAction], None] | None = None

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------

    def set_mode(self, mode: AutoTradeMode):
        """Change trading mode. Stops/starts scan loop as needed."""
        with self._lock:
            old_mode = self.state.mode
            self.state.mode = mode

            if mode == AutoTradeMode.MANUAL:
                if self._is_loop_running():
                    self._stop_loop()
                self.state.is_running = False
            else:
                # AUTO or SEMI_AUTO — start loop if not running
                if not self._is_loop_running():
                    self._start_loop()
                self.state.is_running = True

            log.info(f"Auto-trade mode changed: {old_mode.value} -> {mode.value}")
            self._notify_state_changed()

    def get_mode(self) -> AutoTradeMode:
        """Get current mode."""
        with self._lock:
            return self.state.mode

    def start(self):
        """Start the auto-trader (uses current mode)."""
        with self._lock:
            if self.state.mode == AutoTradeMode.MANUAL:
                log.info("Auto-trader in MANUAL mode — scan loop not started")
                return

            if self._is_loop_running():
                return

            self._start_loop()
            self.state.is_running = True
            self._notify_state_changed()
            log.info(f"Auto-trader started in {self.state.mode.value} mode")

    def stop(self):
        """Stop the auto-trader scan loop."""
        with self._lock:
            self._stop_loop()
            self.state.is_running = False
            self._notify_state_changed()
            log.info("Auto-trader stopped")

    def update_watchlist(self, watch_list: list[str]):
        """Update the watchlist (thread-safe)."""
        with self._lock:
            self._watch_list = list(watch_list)[:50]

    def update_predictor(self, predictor):
        """Update the predictor instance (after model reload)."""
        with self._lock:
            self._predictor = predictor

    def get_state(self) -> AutoTradeState:
        """Get a snapshot of the current state."""
        with self._lock:
            return copy.deepcopy(self.state)

    def get_recent_actions(self, n: int = 50) -> list[AutoTradeAction]:
        """Get recent auto-trade actions."""
        with self._lock:
            return copy.deepcopy(self.state.recent_actions[:n])

    def get_pending_approvals(self) -> list[AutoTradeAction]:
        """Get pending approvals (SEMI_AUTO mode)."""
        with self._lock:
            return copy.deepcopy(self.state.pending_approvals)

    def _passes_precision_quality_gate(self, pred) -> tuple[bool, str]:
        """
        Additional precision guardrails for autonomous trading.
        Uses entropy and directional edge thresholds from PrecisionConfig.
        """
        try:
            p_cfg = getattr(CONFIG, "precision", None)
            if not p_cfg or not bool(getattr(p_cfg, "enabled", True)):
                return True, ""

            entropy = float(getattr(pred, "entropy", 0.0) or 0.0)
            max_entropy = float(getattr(p_cfg, "max_entropy", 1.0) or 1.0)
            if entropy > max_entropy:
                return False, f"High uncertainty (entropy {entropy:.2f} > {max_entropy:.2f})"

            prob_up = float(getattr(pred, "prob_up", 0.0) or 0.0)
            prob_down = float(getattr(pred, "prob_down", 0.0) or 0.0)
            edge = abs(prob_up - prob_down)
            min_edge = float(getattr(p_cfg, "min_edge", 0.0) or 0.0)
            if edge < min_edge:
                return False, f"Weak directional edge ({edge:.2f} < {min_edge:.2f})"
        except Exception:
            # Fail-open to avoid accidental trading halt from malformed config.
            return True, ""

        return True, ""

    def approve_pending(self, action_id: str) -> bool:
        """Approve a pending auto-trade action (SEMI_AUTO mode)."""
        with self._lock:
            action = self.state.remove_pending(action_id)
            if action is None:
                log.warning(f"Pending approval not found: {action_id}")
                return False

            success = self._execute_action(action)
            if success:
                action.decision = "EXECUTED"
                log.info(
                    f"Pending approval APPROVED and executed: "
                    f"{action.stock_code} {action.side}"
                )
            else:
                action.decision = "REJECTED"
                action.skip_reason = "Execution failed after approval"
                log.warning(f"Pending approval execution failed: {action_id}")

            self.state.record_action(action)
            self._notify_action(action)
            self._notify_state_changed()
            return success

    def reject_pending(self, action_id: str) -> bool:
        """Reject a pending auto-trade action (SEMI_AUTO mode)."""
        with self._lock:
            action = self.state.remove_pending(action_id)
            if action is None:
                return False

            action.decision = "REJECTED"
            action.skip_reason = "Manually rejected by user"
            self.state.record_action(action)
            self.state.record_skip()
            self._notify_action(action)
            self._notify_state_changed()
            log.info(f"Pending approval REJECTED: {action.stock_code}")
            return True

    def pause(self, reason: str, duration_seconds: int = 0):
        """Pause auto-trading temporarily."""
        with self._lock:
            self.state.is_paused = True
            self.state.pause_reason = reason
            if duration_seconds > 0:
                self.state.pause_until = (
                    datetime.now() + timedelta(seconds=duration_seconds)
                )
            else:
                self.state.pause_until = None
            log.info(f"Auto-trader paused: {reason}")
            self._notify_state_changed()

    def resume(self):
        """Resume auto-trading after pause."""
        with self._lock:
            self.state.is_paused = False
            self.state.pause_reason = ""
            self.state.pause_until = None
            log.info("Auto-trader resumed")
            self._notify_state_changed()

    # -----------------------------------------------------------------
    # Internal: loop management
    # -----------------------------------------------------------------

    def _is_loop_running(self) -> bool:
        """Check if scan thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def _start_loop(self):
        """Start the scan loop thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._scan_loop,
            name="auto_trader",
            daemon=True,
        )
        self._thread.start()

    def _stop_loop(self):
        """Stop the scan loop thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._thread = None

    # -----------------------------------------------------------------
    # Internal: main scan loop
    # -----------------------------------------------------------------

    def _scan_loop(self):
        """
        Main auto-trading scan loop.

        Runs on a dedicated daemon thread.  Each cycle:
        1. Check pre-conditions (market open, broker connected, etc.)
        2. Day rollover if needed
        3. Run predictor batch scan on watchlist
        4. Filter signals against auto-trade thresholds
        5. Execute or queue for approval
        6. Sleep until next cycle
        """
        log.info("Auto-trade scan loop started")

        while not self._stop_event.is_set():
            cycle_start = time.time()

            try:
                with self._lock:
                    today = date.today()
                    if self.state.session_date != today:
                        self.state.reset_daily()
                        log.info("Auto-trader: new trading day, counters reset")

                    if not self._should_scan():
                        self._sleep_interruptible(5)
                        continue

                    self.state.last_scan_time = datetime.now()

                self._run_scan_cycle()

            except Exception as e:
                with self._lock:
                    self.state.record_error(str(e))
                    log.error(f"Auto-trade scan error: {e}", exc_info=True)

                    # Auto-pause on too many consecutive errors
                    if self.state.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                        self.state.is_paused = True
                        self.state.pause_reason = (
                            f"Too many errors ({self.state.consecutive_errors})"
                        )
                        self.state.pause_until = (
                            datetime.now()
                            + timedelta(seconds=self.ERROR_PAUSE_SECONDS)
                        )
                        log.warning(
                            f"Auto-trader paused for {self.ERROR_PAUSE_SECONDS}s "
                            f"due to {self.state.consecutive_errors} consecutive errors"
                        )
                        self._notify_state_changed()

            elapsed = time.time() - cycle_start
            interval = CONFIG.auto_trade.scan_interval_seconds
            remaining = max(1.0, interval - elapsed)
            self._sleep_interruptible(remaining)

        log.info("Auto-trade scan loop exited")

    def _should_scan(self) -> bool:
        """
        Check all pre-conditions for scanning.
        MUST be called with ``_lock`` held.
        """
        cfg = CONFIG.auto_trade

        if self.state.mode == AutoTradeMode.MANUAL:
            return False

        if not cfg.enabled:
            return False

        if self.state.is_safety_paused:
            return False

        if cfg.require_market_open and not CONFIG.is_market_open():
            return False

        if cfg.require_broker_connected:
            if not self._engine or not self._engine._running:
                return False
            if not self._engine.broker.is_connected:
                return False

        if self.state.trades_today >= cfg.max_trades_per_day:
            return False

        if self._predictor is None or self._predictor.ensemble is None:
            return False

        if not self._watch_list:
            return False

        return True

    def _run_scan_cycle(self):
        """
        Execute one scan cycle: predict → filter → execute/queue.
        Called WITHOUT lock held (acquires as needed).
        """
        with self._lock:
            watch_list = list(self._watch_list)
            predictor = self._predictor

        if not predictor or not watch_list:
            return

        cfg = CONFIG.auto_trade

        # --- Step 1: Quick batch prediction ---
        try:
            preds = predictor.predict_quick_batch(
                watch_list,
                use_realtime_price=True,
            )
        except Exception as e:
            with self._lock:
                self.state.record_error(f"Batch predict failed: {e}")
            log.warning(f"Auto-trade batch prediction error: {e}")
            return

        if not preds:
            return

        # --- Step 2: Import Signal enum ---
        try:
            from models.predictor import Signal
        except ImportError:
            log.error("Cannot import Signal enum")
            return

        # --- Step 3: Filter by auto-trade thresholds ---
        signal_allow_map = {
            Signal.STRONG_BUY: cfg.allow_strong_buy,
            Signal.BUY: cfg.allow_buy,
            Signal.SELL: cfg.allow_sell,
            Signal.STRONG_SELL: cfg.allow_strong_sell,
            Signal.HOLD: cfg.allow_hold,
        }
        # Optional precision profile: disable weak BUY/SELL auto-trades.
        try:
            p_cfg = getattr(CONFIG, "precision", None)
            if p_cfg and bool(getattr(p_cfg, "force_strong_signals_auto_trade", False)):
                signal_allow_map[Signal.BUY] = False
                signal_allow_map[Signal.SELL] = False
        except Exception:
            pass

        candidates = []
        for p in preds:
            sig = getattr(p, 'signal', Signal.HOLD)
            conf = getattr(p, 'confidence', 0.0)
            strength = getattr(p, 'signal_strength', 0.0)
            agreement = getattr(p, 'model_agreement', 1.0)

            # Signal type allowed?
            if not signal_allow_map.get(sig, False):
                continue

            if conf < cfg.min_confidence:
                continue

            if strength < cfg.min_signal_strength:
                continue

            if agreement < cfg.min_model_agreement:
                continue

            candidates.append(p)

        if not candidates:
            return

        candidates.sort(
            key=lambda x: getattr(x, 'confidence', 0),
            reverse=True,
        )

        # --- Step 4: Process each candidate ---
        for pred in candidates:
            if self._stop_event.is_set():
                break

            with self._lock:
                # Re-check daily limit
                if self.state.trades_today >= cfg.max_trades_per_day:
                    break

                # Check per-stock daily limit
                code = getattr(pred, 'stock_code', '')
                if not self.state.can_trade_stock(
                    code, cfg.max_trades_per_stock_per_day
                ):
                    self._record_skip(
                        pred, Signal,
                        f"Per-stock daily limit ({cfg.max_trades_per_stock_per_day})"
                    )
                    continue

                if self.state.is_on_cooldown(code):
                    self._record_skip(pred, Signal, "On cooldown")
                    continue

                # Safety pause re-check
                if self.state.is_safety_paused:
                    break

            # Run full prediction for this candidate (outside lock)
            try:
                full_pred = predictor.predict(
                    code,
                    use_realtime_price=True,
                    skip_cache=True,
                )
            except Exception as e:
                log.warning(f"Auto-trade full predict failed for {code}: {e}")
                with self._lock:
                    self.state.record_error(f"Predict {code}: {e}")
                continue

            # Re-validate after full prediction
            full_sig = getattr(full_pred, 'signal', Signal.HOLD)
            full_conf = getattr(full_pred, 'confidence', 0.0)

            if not signal_allow_map.get(full_sig, False):
                continue
            if full_conf < cfg.min_confidence:
                continue
            ok, reason = self._passes_precision_quality_gate(full_pred)
            if not ok:
                with self._lock:
                    self._record_skip(full_pred, Signal, reason)
                continue

            price = getattr(full_pred, 'current_price', 0.0)
            position = getattr(full_pred, 'position', None)
            shares = getattr(position, 'shares', 0) if position else 0
            order_value = shares * price if (shares > 0 and price > 0) else 0

            if order_value > cfg.max_auto_order_value:
                if price > 0:
                    lot_size = max(1, CONFIG.LOT_SIZE)
                    shares = int(cfg.max_auto_order_value / price)
                    shares = (shares // lot_size) * lot_size
                    if shares <= 0:
                        with self._lock:
                            self._record_skip(
                                full_pred, Signal,
                                f"Order value exceeds limit (¥{cfg.max_auto_order_value:,.0f})"
                            )
                        continue
                    order_value = shares * price

            if shares <= 0 or price <= 0:
                with self._lock:
                    self._record_skip(full_pred, Signal, "No valid position size")
                continue

            if cfg.pause_on_high_volatility:
                atr_pct = getattr(full_pred, 'atr_pct_value', 0.02)
                if atr_pct * 100 > cfg.volatility_pause_threshold:
                    with self._lock:
                        self._record_skip(
                            full_pred, Signal,
                            f"High volatility (ATR {atr_pct*100:.1f}%)"
                        )
                    continue

            if full_sig in (Signal.STRONG_BUY, Signal.BUY):
                side = OrderSide.BUY
            elif full_sig in (Signal.STRONG_SELL, Signal.SELL):
                side = OrderSide.SELL
            else:
                continue

            # Auto-trade position limit (separate from risk manager)
            if side == OrderSide.BUY:
                try:
                    account = self._engine.get_account()
                    current_positions = len([
                        p for p in account.positions.values()
                        if p.quantity > 0
                    ])
                    if current_positions >= cfg.max_auto_positions:
                        with self._lock:
                            self._record_skip(
                                full_pred, Signal,
                                f"Auto-trade position limit ({cfg.max_auto_positions})"
                            )
                        continue

                    # Position concentration for auto-trade
                    equity = account.equity
                    if equity > 0:
                        existing = account.positions.get(code)
                        existing_value = (
                            existing.market_value if existing else 0.0
                        )
                        new_pct = (
                            (existing_value + order_value) / equity * 100
                        )
                        if new_pct > cfg.max_auto_position_pct:
                            with self._lock:
                                self._record_skip(
                                    full_pred, Signal,
                                    f"Auto position limit ({new_pct:.1f}% > {cfg.max_auto_position_pct}%)"
                                )
                            continue
                except Exception as e:
                    log.debug(f"Account check failed: {e}")

            action = AutoTradeAction(
                stock_code=code,
                stock_name=getattr(full_pred, 'stock_name', ''),
                signal_type=full_sig.value if hasattr(full_sig, 'value') else str(full_sig),
                confidence=full_conf,
                signal_strength=getattr(full_pred, 'signal_strength', 0.0),
                model_agreement=getattr(full_pred, 'model_agreement', 1.0),
                price=price,
                predicted_direction=(
                    "UP" if full_sig in (Signal.STRONG_BUY, Signal.BUY)
                    else "DOWN"
                ),
                side=side.value,
                quantity=shares,
            )

            # --- Step 5: Execute or queue ---
            with self._lock:
                if self.state.mode == AutoTradeMode.AUTO:
                    success = self._execute_action(action)
                    if success:
                        action.decision = "EXECUTED"
                        self.state.record_trade(code, side.value)
                        self.state.set_cooldown(
                            code, cfg.cooldown_after_trade_seconds
                        )
                    else:
                        action.decision = "REJECTED"
                        action.skip_reason = "Execution/risk check failed"

                    self.state.record_action(action)
                    self._notify_action(action)
                    self._notify_state_changed()

                elif self.state.mode == AutoTradeMode.SEMI_AUTO:
                    action.decision = "PENDING"
                    self.state.add_pending_approval(action)
                    self._notify_pending(action)
                    self._notify_state_changed()
                    log.info(
                        f"Auto-trade PENDING approval: "
                        f"{action.signal_type} {code} @ ¥{price:.2f}"
                    )

    def _execute_action(self, action: AutoTradeAction) -> bool:
        """
        Execute a single auto-trade action via ExecutionEngine.submit().

        Returns True if the signal was accepted by the engine.
        MUST be called with ``_lock`` held (for state updates).
        """
        try:
            import importlib.util
            signal_spec = importlib.util.find_spec("models.predictor")
        except ImportError:
            signal_spec = None
        if signal_spec is None:
            action.skip_reason = "Cannot import Signal"
            return False

        cfg = CONFIG.auto_trade

        side = OrderSide(action.side)
        levels_entry = action.price
        levels_stop = 0.0
        levels_target = 0.0

        try:
            pred = self._predictor.predict(
                action.stock_code,
                use_realtime_price=True,
                skip_cache=True,
            )
            if pred and pred.levels:
                levels_entry = pred.levels.entry or action.price
                levels_stop = pred.levels.stop_loss or 0.0
                levels_target = pred.levels.target_2 or 0.0
                if pred.current_price > 0:
                    action.price = pred.current_price
                    levels_entry = pred.current_price
        except Exception:
            pass

        signal = TradeSignal(
            symbol=action.stock_code,
            name=action.stock_name,
            side=side,
            quantity=action.quantity,
            price=levels_entry,
            stop_loss=levels_stop if cfg.auto_stop_loss else 0.0,
            take_profit=levels_target,
            confidence=action.confidence,
            strategy="auto_trade",
            reasons=[
                f"Auto-trade: {action.signal_type}",
                f"Confidence: {action.confidence:.0%}",
                f"Agreement: {action.model_agreement:.0%}",
            ],
            auto_generated=True,
            auto_trade_action_id=action.id,
        )

        # Submit through execution engine (which runs risk checks)
        success = self._engine.submit(signal)

        if success:
            action.signal_id = signal.id
            log.info(
                f"Auto-trade EXECUTED: {side.value.upper()} "
                f"{action.quantity} {action.stock_code} "
                f"@ ¥{levels_entry:.2f} (conf={action.confidence:.0%})"
            )
        else:
            log.warning(
                f"Auto-trade REJECTED by engine: {action.stock_code}"
            )

        return success

    def _record_skip(self, pred, Signal, reason: str):
        """Record a skipped signal. MUST be called with _lock held."""
        code = getattr(pred, 'stock_code', '?')
        sig = getattr(pred, 'signal', Signal.HOLD)

        action = AutoTradeAction(
            stock_code=code,
            stock_name=getattr(pred, 'stock_name', ''),
            signal_type=sig.value if hasattr(sig, 'value') else str(sig),
            confidence=getattr(pred, 'confidence', 0.0),
            signal_strength=getattr(pred, 'signal_strength', 0.0),
            price=getattr(pred, 'current_price', 0.0),
            decision="SKIPPED",
            skip_reason=reason,
        )

        self.state.record_action(action)
        self.state.record_skip()

        if CONFIG.auto_trade.notify_on_skip:
            self._notify_action(action)

    # -----------------------------------------------------------------
    # Internal: notification helpers
    # -----------------------------------------------------------------

    def _notify_action(self, action: AutoTradeAction):
        """Notify UI of an action."""
        if self.on_action:
            try:
                self.on_action(action)
            except Exception as e:
                log.debug(f"Action callback error: {e}")

    def _notify_state_changed(self):
        """Notify UI of state change."""
        if self.on_state_changed:
            try:
                self.on_state_changed(copy.deepcopy(self.state))
            except Exception as e:
                log.debug(f"State callback error: {e}")

    def _notify_pending(self, action: AutoTradeAction):
        """Notify UI of pending approval."""
        if self.on_pending_approval:
            try:
                self.on_pending_approval(action)
            except Exception as e:
                log.debug(f"Pending callback error: {e}")

    def _sleep_interruptible(self, seconds: float):
        """Sleep that can be interrupted by stop event."""
        end = time.time() + seconds
        while time.time() < end and not self._stop_event.is_set():
            time.sleep(min(1.0, end - time.time()))

class ExecutionEngine:
    """
    Production execution engine with correct broker synchronization.

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

    def __init__(self, mode: TradingMode = None):
        self.mode = mode or CONFIG.trading_mode
        self.broker: BrokerInterface = create_broker(self.mode.value)
        self.risk_manager: RiskManager = get_risk_manager()

        self._kill_switch = get_kill_switch()
        self._health_monitor = get_health_monitor()
        self._alert_manager = get_alert_manager()
        self._fills_lock = threading.RLock()

        self._queue: queue.Queue[TradeSignal | None] = queue.Queue()
        self._running = False

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

    # -----------------------------------------------------------------
    # Auto-trade public API
    # -----------------------------------------------------------------

    def init_auto_trader(self, predictor, watch_list: list[str]):
        """
        Initialize the auto-trader with a predictor and watchlist.
        Must be called before start_auto_trade().
        """
        self.auto_trader = AutoTrader(
            engine=self,
            predictor=predictor,
            watch_list=watch_list,
        )
        self._restore_auto_trader_state()
        log.info("Auto-trader initialized")

    def start_auto_trade(self, mode: AutoTradeMode = AutoTradeMode.AUTO):
        """Start auto-trading in the specified mode."""
        if self.auto_trader is None:
            log.error("Auto-trader not initialized. Call init_auto_trader() first.")
            return

        if not self._running:
            log.error("Execution engine not running. Call start() first.")
            return

        # Safety confirmation for live auto-trading
        if (
            self.mode == TradingMode.LIVE
            and CONFIG.auto_trade.confirm_live_auto_trade
            and mode != AutoTradeMode.MANUAL
        ):
            log.warning(
                "Live auto-trading requested — "
                "CONFIG.auto_trade.confirm_live_auto_trade is True. "
                "UI must confirm before proceeding."
            )
            # and calls set_auto_mode() directly after user confirms.

        self.auto_trader.set_mode(mode)
        log.info(f"Auto-trading started: mode={mode.value}")

    def stop_auto_trade(self):
        """Stop auto-trading (switch to MANUAL)."""
        if self.auto_trader:
            self.auto_trader.set_mode(AutoTradeMode.MANUAL)
            log.info("Auto-trading stopped (switched to MANUAL)")

    def set_auto_mode(self, mode: AutoTradeMode):
        """Change auto-trade mode."""
        if self.auto_trader:
            self.auto_trader.set_mode(mode)

    def get_auto_trade_state(self) -> AutoTradeState | None:
        """Get auto-trade state snapshot."""
        if self.auto_trader:
            return self.auto_trader.get_state()
        return None

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------

    def start(self) -> bool:
        if self._running:
            return True

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
        self._heartbeat("main")
        self._persist_runtime_state(clean_shutdown=False)

        self._exec_thread = threading.Thread(
            target=self._execution_loop, name="exec", daemon=True
        )
        self._exec_thread.start()

        self._fill_sync_thread = threading.Thread(
            target=self._fill_sync_loop, name="fill_sync", daemon=True
        )
        self._fill_sync_thread.start()

        self._status_sync_thread = threading.Thread(
            target=self._status_sync_loop, name="status_sync", daemon=True
        )
        self._status_sync_thread.start()

        self._recon_thread = threading.Thread(
            target=self._reconciliation_loop, name="recon", daemon=True
        )
        self._recon_thread.start()

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="watchdog", daemon=True
        )
        self._watchdog_thread.start()

        self._checkpoint_thread = threading.Thread(
            target=self._checkpoint_loop, name="checkpoint", daemon=True
        )
        self._checkpoint_thread.start()

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
            except Exception as e:
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

        self._reconnect_thread = threading.Thread(
            target=self._broker_reconnect_loop,
            name="broker_reconnect",
            daemon=True,
        )
        self._reconnect_thread.start()

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
        except Exception as e:
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
        except Exception:
            return None

    def _persist_synthetic_exits(self, force: bool = False) -> None:
        """
        Persist synthetic exit plans atomically.

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
        except Exception as e:
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
        except Exception as e:
            log.debug(f"Synthetic exit state restore failed: {e}")

    def _resolve_price(self, symbol: str, hinted_price: float = 0.0) -> float:
        """
        Resolve one authoritative price for this submission.
        """
        try:
            px = float(hinted_price or 0.0)
            if px > 0:
                return px
        except Exception:
            pass

        try:
            from data.feeds import get_feed_manager

            fm = get_feed_manager(auto_init=False)
            q = fm.get_quote(symbol)
            if q and getattr(q, "price", 0) and float(q.price) > 0:
                return float(q.price)
        except Exception:
            pass

        try:
            px = self.broker.get_quote(symbol)
            if px and float(px) > 0:
                return float(px)
        except Exception:
            pass

        try:
            from data.fetcher import get_fetcher

            q = get_fetcher().get_realtime(symbol)
            if q and getattr(q, "price", 0) and float(q.price) > 0:
                return float(q.price)
        except Exception:
            pass

        return 0.0

    def _rebuild_broker_mappings(self, oms):
        """Rebuild broker ID mappings from persisted orders after restart."""
        try:
            active_orders = oms.get_active_orders()
            for order in active_orders:
                if order.broker_id:
                    self.broker.register_order_mapping(order.id, order.broker_id)
                    log.debug(f"Recovered mapping: {order.id} -> {order.broker_id}")
            log.info(f"Recovered {len(active_orders)} order mappings from DB")
        except Exception as e:
            log.warning(f"Failed to rebuild broker mappings: {e}")

    def stop(self):
        if not self._running:
            self._release_runtime_lease()
            return

        # Stop auto-trader first
        if self.auto_trader:
            try:
                self.auto_trader.stop()
            except Exception as e:
                log.warning(f"Auto-trader stop error: {e}")

        self._running = False

        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

        for t in [
            self._exec_thread,
            self._fill_sync_thread,
            self._status_sync_thread,
            self._recon_thread,
            self._reconnect_thread,
            self._watchdog_thread,
            self._checkpoint_thread,
        ]:
            if t and t.is_alive():
                t.join(timeout=5)

        try:
            self.broker.disconnect()
        except Exception as e:
            log.warning(f"Broker disconnect error: {e}")

        try:
            self._health_monitor.stop()
            self._alert_manager.stop()
        except Exception:
            pass
        if unregister_snapshot_provider is not None:
            try:
                unregister_snapshot_provider(self._snapshot_provider_name)
            except Exception:
                pass
        self._persist_synthetic_exits(force=True)
        self._persist_runtime_state(clean_shutdown=True)
        self._release_runtime_lease()

        log.info("Execution engine stopped")

    def _build_execution_snapshot(self) -> dict[str, object]:
        broker = self.broker
        auto_state = None
        if self.auto_trader is not None:
            try:
                auto_state = self.auto_trader.get_state()
            except Exception:
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
            except Exception:
                pass
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
            except Exception:
                pass
            snapshot["execution_quality"] = self._get_execution_quality_snapshot()
            with self._synthetic_exit_lock:
                snapshot["synthetic_exits"] = {
                    "active_plans": int(len(self._synthetic_exits)),
                    "plans": list(self._synthetic_exits.values())[:50],
                    "state_file": str(
                        getattr(self, "_synthetic_exit_state_path", "")
                    ),
                }
        except Exception:
            pass
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

    def _heartbeat(self, name: str):
        """Record thread heartbeat for watchdog + observability."""
        with self._thread_hb_lock:
            self._thread_heartbeats[str(name)] = time.time()

    def _runtime_state_payload(self, clean_shutdown: bool) -> dict[str, object]:
        """Persistable runtime checkpoint for crash recovery."""
        auto_state = None
        if self.auto_trader is not None:
            try:
                auto_state = self.auto_trader.get_state()
            except Exception:
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
        except Exception as e:
            log.warning("Runtime lease client init failed: %s", e)
            return None

    def _acquire_runtime_lease(self) -> bool:
        """
        Acquire a local single-writer runtime lease.

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
        except Exception as e:
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
        except Exception as e:
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
        except Exception:
            pass

    def _persist_runtime_state(self, clean_shutdown: bool = False):
        """Write runtime checkpoint atomically."""
        try:
            payload = self._runtime_state_payload(clean_shutdown=clean_shutdown)
            atomic_write_json(self._runtime_state_path, payload, indent=2)
            self._last_checkpoint_ts = time.time()
        except Exception as e:
            log.debug(f"Runtime checkpoint write failed: {e}")

    def _restore_runtime_state(self):
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
        except Exception as e:
            log.debug(f"Runtime checkpoint restore failed: {e}")

    def _restore_auto_trader_state(self):
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
        except Exception as e:
            log.debug(f"Auto-trader recovery state apply failed: {e}")

    def _checkpoint_loop(self):
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
                        except Exception:
                            pass
                    last_lease = now
            except Exception as e:
                log.debug(f"Checkpoint loop error: {e}")
            time.sleep(interval)

    def _watchdog_loop(self):
        """
        Watchdog for core execution threads.
        On heartbeat stall, pause auto-trader and report degraded health.
        """
        stall_seconds = float(getattr(CONFIG, "runtime_watchdog_stall_seconds", 25.0) or 25.0)
        stall_seconds = max(8.0, stall_seconds)
        while self._running:
            time.sleep(2.0)
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
            except Exception:
                pass
            try:
                if self.auto_trader is not None:
                    self.auto_trader.pause(msg, duration_seconds=300)
            except Exception:
                pass
            try:
                self._alert_manager.risk_alert("Runtime watchdog", msg)
            except Exception:
                pass

    def _get_quote_snapshot(
        self, symbol: str
    ) -> tuple[float, datetime | None, str]:
        """
        Returns (price, timestamp, source).
        """
        # 1) feed cache
        try:
            from data.feeds import get_feed_manager

            fm = get_feed_manager(auto_init=False)
            q = fm.get_quote(symbol)
            if q and float(getattr(q, "price", 0.0) or 0.0) > 0:
                ts = getattr(q, "timestamp", None)
                return float(q.price), ts, "feed"
        except Exception:
            pass

        # 2) broker quote
        try:
            px = self.broker.get_quote(symbol)
            if px and float(px) > 0:
                return float(px), None, "broker"
        except Exception:
            pass

        # 3) fetcher realtime
        try:
            from data.fetcher import get_fetcher

            q = get_fetcher().get_realtime(symbol)
            if q and float(getattr(q, "price", 0.0) or 0.0) > 0:
                return (
                    float(q.price),
                    getattr(q, "timestamp", None),
                    f"fetcher:{getattr(q, 'source', '')}",
                )
        except Exception:
            pass

        return 0.0, None, "none"

    def _require_fresh_quote(
        self, symbol: str, max_age_seconds: float = 15.0
    ) -> tuple[bool, str, float]:
        """
        Strict quote freshness gate for order submission.
        Returns (ok, message, price).
        """
        px, ts, src = self._get_quote_snapshot(symbol)
        if px <= 0:
            return False, "No valid quote", 0.0

        # If no timestamp, be conservative in LIVE mode
        if ts is None:
            if str(self.mode.value).lower() == "live":
                return False, f"No timestamped quote (source={src})", 0.0
            return True, "OK", px

        try:
            age = (datetime.now() - ts).total_seconds()
        except Exception:
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
        """
        Extra exchange-style guardrails:
        - duplicate suppression
        - per-symbol burst cap
        - max single-order notional cap
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

    def _record_rejection_guardrail(self, reason: str):
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
        except Exception as e:
            log.debug(f"Reject kill-switch guard failed: {e}")

    def submit(self, signal: TradeSignal) -> bool:
        """Submit a trading signal for execution with strict quote freshness."""
        if not self._running:
            log.warning("Execution engine not running")
            return False
        hinted_price = float(getattr(signal, "price", 0.0) or 0.0)
        requested_order_type = self._normalize_requested_order_type(
            getattr(signal, "order_type", "limit")
        )
        market_style_types = {"market", "ioc", "fok", "stop", "trail_market"}
        limit_style_types = {"limit", "stop_limit", "trail_limit"}

        # Operational guardrails: block/allow trading based on health status.
        try:
            sec_cfg = getattr(CONFIG, "security", None)
            block_unhealthy = bool(
                getattr(sec_cfg, "block_trading_when_unhealthy", True)
            )
            block_degraded = bool(
                getattr(sec_cfg, "block_trading_when_degraded", False)
            )
            h = self._health_monitor.get_health()
            if block_unhealthy and h.status in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL):
                self._reject_signal(signal, f"System unhealthy: {h.status.value}")
                return False
            if block_degraded and h.status == HealthStatus.DEGRADED:
                self._reject_signal(signal, "System degraded: trading paused by policy")
                return False
        except Exception:
            pass

        # Institutional controls: permission gating + optional dual-control.
        try:
            sec_cfg = getattr(CONFIG, "security", None)
            access = get_access_control()
            audit = get_audit_log()
            mode_is_live = str(getattr(self.mode, "value", "")).lower() == "live"

            require_perm = bool(
                getattr(sec_cfg, "require_live_trade_permission", True)
            )
            strict_live = bool(
                getattr(sec_cfg, "strict_live_governance", False)
            )
            min_approvals = int(getattr(sec_cfg, "min_live_approvals", 2))

            if mode_is_live and require_perm:
                if not access.check("trade_live"):
                    self._reject_signal(
                        signal, "Access denied: trade_live permission required"
                    )
                    audit.log_risk_event(
                        "access_denied_live_trade",
                        {"symbol": signal.symbol, "signal_id": signal.id},
                    )
                    return False
            elif not mode_is_live:
                if not access.check("trade_paper"):
                    self._reject_signal(
                        signal, "Access denied: trade_paper permission required"
                    )
                    audit.log_risk_event(
                        "access_denied_paper_trade",
                        {"symbol": signal.symbol, "signal_id": signal.id},
                    )
                    return False

            if mode_is_live and strict_live and not signal.auto_generated:
                approvals = int(getattr(signal, "approvals_count", 0) or 0)
                approver_ids = getattr(signal, "approver_ids", None)
                if not isinstance(approver_ids, list):
                    approver_ids = getattr(signal, "approved_by", None)
                if isinstance(approver_ids, list):
                    approvals = max(
                        approvals,
                        len({str(x).strip().lower() for x in approver_ids if str(x).strip()}),
                    )
                required = max(1, min_approvals)
                if approvals < required:
                    self._reject_signal(
                        signal, f"Insufficient live approvals ({approvals}/{required})"
                    )
                    audit.log_risk_event(
                        "live_trade_blocked_dual_control",
                        {
                            "symbol": signal.symbol,
                            "signal_id": signal.id,
                            "approvals": approvals,
                            "required": required,
                        },
                    )
                    return False

            if mode_is_live:
                decision = get_trade_policy_engine().evaluate_live_trade(signal)
                if not decision.allowed:
                    self._reject_signal(signal, f"Policy blocked: {decision.reason}")
                    audit.log_risk_event(
                        "live_trade_blocked_policy",
                        {
                            "symbol": signal.symbol,
                            "signal_id": signal.id,
                            "reason": decision.reason,
                            "policy_version": decision.policy_version,
                            "metadata": decision.metadata or {},
                        },
                    )
                    return False
        except Exception as e:
            log.debug(f"Security governance check skipped due to error: {e}")

        try:
            if not CONFIG.is_market_open():
                self._reject_signal(signal, "Market closed")
                return False
        except Exception:
            pass

        if not self._kill_switch.can_trade:
            self._reject_signal(signal, "Trading halted - kill switch active")
            return False

        if not self.risk_manager:
            self._reject_signal(signal, "Risk manager not initialized")
            return False

        try:
            from data.fetcher import DataFetcher

            signal.symbol = DataFetcher.clean_code(signal.symbol)
        except Exception:
            pass

        max_age = 15.0
        if hasattr(CONFIG, "risk") and hasattr(CONFIG.risk, "quote_staleness_seconds"):
            max_age = float(CONFIG.risk.quote_staleness_seconds)

        ok, msg, fresh_px = self._require_fresh_quote(
            signal.symbol, max_age_seconds=max_age
        )
        if not ok:
            self._reject_signal(signal, msg)
            return False

        trigger_price = float(getattr(signal, "trigger_price", 0.0) or 0.0)
        if requested_order_type in {"stop", "stop_limit"} and trigger_price <= 0:
            self._reject_signal(signal, "Missing trigger price for stop order")
            return False
        if requested_order_type in limit_style_types and hinted_price <= 0:
            self._reject_signal(
                signal,
                f"{requested_order_type} order requires a positive limit price",
            )
            return False

        execution_ref_price = hinted_price
        if requested_order_type in market_style_types or execution_ref_price <= 0:
            execution_ref_price = float(fresh_px)
        if execution_ref_price <= 0:
            self._reject_signal(signal, "No valid execution reference price")
            return False

        # Best-execution sanity check: avoid large drift from hinted signal price.
        try:
            max_bps = float(getattr(CONFIG.risk, "max_quote_deviation_bps", 80.0))
            if (
                requested_order_type in market_style_types
                and hinted_price > 0
                and max_bps > 0
            ):
                dev_bps = abs((float(fresh_px) / hinted_price - 1.0) * 10000.0)
                if dev_bps > max_bps:
                    self._reject_signal(
                        signal,
                        f"Best-exec guard: quote deviation {dev_bps:.1f}bps > {max_bps:.1f}bps",
                    )
                    try:
                        get_audit_log().log_risk_event(
                            "best_execution_guard_reject",
                            {
                                "symbol": signal.symbol,
                                "hinted_price": hinted_price,
                                "fresh_price": float(fresh_px),
                                "deviation_bps": float(dev_bps),
                                "limit_bps": float(max_bps),
                            },
                        )
                    except Exception:
                        pass
                    return False
        except Exception:
            pass

        signal._arrival_price = float(fresh_px)
        signal.price = (
            float(execution_ref_price)
            if requested_order_type in market_style_types
            else float(hinted_price)
        )

        guard_ok, guard_msg = self._check_submission_guardrails(signal)
        if not guard_ok:
            self._reject_signal(signal, guard_msg)
            return False

        # CN limit up/down sanity
        try:
            from core.constants import get_price_limit
            from data.fetcher import get_fetcher

            q = get_fetcher().get_realtime(signal.symbol)
            prev_close = float(getattr(q, "close", 0.0) or 0.0)
            if prev_close > 0:
                lim = float(
                    get_price_limit(signal.symbol, getattr(q, "name", None))
                )
                up = prev_close * (1.0 + lim)
                dn = prev_close * (1.0 - lim)

                if signal.side == OrderSide.BUY and signal.price >= up * 0.999:
                    self._reject_signal(
                        signal, f"At/near limit-up ({lim * 100:.0f}%)"
                    )
                    return False
                if signal.side == OrderSide.SELL and signal.price <= dn * 1.001:
                    self._reject_signal(
                        signal, f"At/near limit-down ({lim * 100:.0f}%)"
                    )
                    return False
        except Exception:
            pass

        passed, rmsg = self.risk_manager.check_order(
            signal.symbol, signal.side, int(signal.quantity), float(signal.price)
        )
        if not passed:
            log.warning(f"Risk check failed: {rmsg}")
            self._alert_manager.risk_alert(
                "Order Rejected (Risk)", f"{signal.symbol}: {rmsg}"
            )
            self._reject_signal(signal, rmsg)
            return False

        self._queue.put(signal)
        set_gauge("runtime_queue_depth", float(self._queue.qsize()))
        log.info(
            f"Signal queued: {signal.side.value} {signal.quantity} "
            f"{signal.symbol} @ {signal.price:.2f}"
            f"{' [AUTO]' if signal.auto_generated else ''}"
        )
        return True

    def _on_health_degraded(self, health):
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
        except Exception as e:
            log.debug(f"Degraded health handler failed: {e}")

    def submit_from_prediction(self, pred) -> bool:
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

    def _execution_loop(self):
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
            except Exception as e:
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
                except Exception as e:
                    log.warning(f"Risk update error: {e}")

            time.sleep(0.05)

    def get_risk_metrics(self):
        if self.risk_manager:
            return self.risk_manager.get_metrics()
        return None

    def _execute(self, signal: TradeSignal):
        """Execute a single signal - NEVER fabricate fills."""
        from trading.oms import get_oms

        oms = get_oms()
        order: Order | None = None

        try:
            if not self._kill_switch.can_trade:
                log.warning("Trading halted during execution")
                return

            order = Order(
                symbol=signal.symbol,
                name=signal.name,
                side=signal.side,
                quantity=int(signal.quantity),
                price=float(signal.price),
                order_type=OrderType.LIMIT,
                stop_loss=(
                    float(signal.stop_loss) if signal.stop_loss else 0.0
                ),
                take_profit=(
                    float(signal.take_profit) if signal.take_profit else 0.0
                ),
                signal_id=signal.id,
                strategy=signal.strategy or "",
            )

            requested_order_type = str(
                self._normalize_requested_order_type(
                    getattr(signal, "order_type", "limit")
                )
            )
            requested_enum = OrderType(requested_order_type)

            broker_order_type = requested_enum
            emulation_reason = ""
            if requested_enum in {
                OrderType.STOP,
                OrderType.STOP_LIMIT,
                OrderType.IOC,
                OrderType.FOK,
                OrderType.TRAIL_MARKET,
                OrderType.TRAIL_LIMIT,
            }:
                # Broker abstraction is market/limit centric; advanced types are
                # represented in tags and handled by runtime guardrails/simulator.
                if requested_enum in {
                    OrderType.STOP,
                    OrderType.IOC,
                    OrderType.FOK,
                    OrderType.TRAIL_MARKET,
                }:
                    broker_order_type = OrderType.MARKET
                else:
                    broker_order_type = OrderType.LIMIT
                emulation_reason = (
                    f"{requested_enum.value}_as_{broker_order_type.value}"
                )

            trigger_px = float(getattr(signal, "trigger_price", 0.0) or 0.0)
            if trigger_px <= 0 and requested_enum in {OrderType.STOP, OrderType.STOP_LIMIT}:
                trigger_px = float(signal.price or 0.0)

            if broker_order_type == OrderType.LIMIT and order.price <= 0:
                fallback_px = float(signal.price or 0.0)
                if fallback_px <= 0 and trigger_px > 0:
                    fallback_px = trigger_px
                order.price = fallback_px

            if trigger_px > 0:
                order.stop_price = float(trigger_px)

            order.order_type = broker_order_type

            # Tag auto-trade orders for audit
            if signal.auto_generated:
                order.tags["auto_trade"] = True
                order.tags["auto_trade_action_id"] = signal.auto_trade_action_id
                order.strategy = order.strategy or "auto_trade"
            arrival_price = float(getattr(signal, "_arrival_price", 0.0) or 0.0)
            if arrival_price <= 0:
                arrival_price = float(signal.price or 0.0)
            order.tags["arrival_price"] = arrival_price
            order.tags["requested_order_type"] = requested_enum.value
            order.tags["order_type_emulated"] = bool(
                broker_order_type != requested_enum
            )
            if emulation_reason:
                order.tags["order_type_emulation_reason"] = emulation_reason
            if trigger_px > 0:
                order.tags["trigger_price"] = float(trigger_px)
            oco_group = str(getattr(signal, "oco_group", "") or "").strip()
            if oco_group:
                order.tags["oco_group"] = oco_group

            tif = str(getattr(signal, "time_in_force", "day") or "day").strip().lower()
            tif = tif.replace("-", "_")
            if requested_enum in {OrderType.IOC, OrderType.FOK}:
                tif = requested_enum.value
            if tif not in {"day", "gtc", "ioc", "fok"}:
                tif = "day"
            order.tags["time_in_force"] = tif
            if tif in {"ioc", "fok"}:
                order.tags["strict_time_in_force"] = True

            order.tags["bracket_enabled"] = bool(
                getattr(signal, "bracket", False)
                or order.stop_loss > 0
                or order.take_profit > 0
            )
            trailing_pct = float(getattr(signal, "trailing_stop_pct", 0.0) or 0.0)
            if requested_enum in {OrderType.TRAIL_MARKET, OrderType.TRAIL_LIMIT} and trailing_pct <= 0:
                trailing_pct = float(
                    getattr(getattr(CONFIG, "auto_trade", None), "trailing_stop_pct", 0.0) or 0.0
                )
            if trailing_pct > 0:
                order.tags["trailing_stop_pct"] = trailing_pct
            trail_limit_offset_pct = float(
                getattr(signal, "trail_limit_offset_pct", 0.0) or 0.0
            )
            if trail_limit_offset_pct > 0:
                order.tags["trail_limit_offset_pct"] = trail_limit_offset_pct

            order = oms.submit_order(order)

            result = self._submit_with_retry(order, attempts=3)

            inc_counter(
                "orders_submitted_total",
                labels={"side": order.side.value, "symbol": order.symbol},
            )

            # Persist broker_id/status in OMS
            oms.update_order_status(
                order.id,
                result.status,
                message=result.message or "",
                broker_id=result.broker_id or "",
            )

            if result.status == OrderStatus.REJECTED:
                self._alert_manager.risk_alert(
                    "Order Rejected (Broker)",
                    f"{order.symbol}: {result.message}",
                )
                if self.on_reject:
                    self.on_reject(order, result.message or "Rejected")
                return

            # Pull fills immediately (sim) / early sync
            self._process_pending_fills()

            log.info(
                f"Order sent: {order.id} -> broker_id={result.broker_id}, "
                f"status={result.status.value}"
                f"{' [AUTO]' if signal.auto_generated else ''}"
            )

        except Exception as e:
            log.error(f"Execution error: {e}")
            if order:
                try:
                    oms.update_order_status(
                        order.id, OrderStatus.REJECTED, message=str(e)
                    )
                except Exception:
                    pass
            self._alert_manager.system_alert(
                "Execution Failed",
                f"{signal.symbol}: {e}",
                AlertPriority.HIGH,
            )

    def _startup_sync(self):
        """Run once after broker.connect()."""
        from trading.oms import get_oms

        oms = get_oms()

        self._rebuild_broker_mappings(oms)

        self._process_pending_fills()

        for order in oms.get_active_orders():
            try:
                synced = self.broker.sync_order(order)
                if synced and synced.status and synced.status != order.status:
                    oms.update_order_status(
                        order.id,
                        synced.status,
                        message="Startup sync",
                        broker_id=synced.broker_id or order.broker_id or "",
                    )
            except Exception:
                continue

    def _process_pending_fills(self):
        """Process pending fills with safe watermark overlap."""
        from trading.oms import get_oms

        oms = get_oms()

        with self._fills_lock:
            try:
                query_start = datetime.now()

                fills = self.broker.get_fills(since=self._last_fill_sync)

                newest_ts = None
                for f in fills:
                    ts = getattr(f, "timestamp", None)
                    if ts:
                        newest_ts = (
                            ts if newest_ts is None else max(newest_ts, ts)
                        )

                if newest_ts:
                    self._last_fill_sync = newest_ts - timedelta(
                        seconds=self._FILL_WATERMARK_OVERLAP_SECONDS
                    )
                else:
                    self._last_fill_sync = query_start

                for fill in fills:
                    fill_id = str(getattr(fill, "id", "") or "").strip()
                    if not fill_id:
                        fill_id = (
                            f"{fill.order_id}|{fill.symbol}|"
                            f"{fill.side.value}|{fill.quantity}|"
                            f"{fill.price}|{getattr(fill, 'timestamp', None)}"
                        )

                    if fill_id in self._processed_fill_ids:
                        continue
                    self._processed_fill_ids.add(fill_id)

                    order = oms.get_order(fill.order_id)

                    # Fallback: broker may have put broker_id into fill.order_id
                    if order is None:
                        try:
                            order = oms.get_order_by_broker_id(fill.order_id)
                            if order:
                                fill.order_id = order.id
                        except Exception:
                            order = None

                    if not order:
                        log.warning(
                            f"Fill for unknown order: {fill.order_id}"
                        )
                        continue

                    oms.process_fill(order, fill)
                    self._record_execution_quality(order, fill)
                    self._maybe_register_synthetic_exit(order, fill)
                    self._cancel_oco_siblings(oms, order, fill)
                    log.info(
                        f"Fill processed: {fill_id} for order {order.id}"
                    )

                    if self.on_fill:
                        try:
                            self.on_fill(order, fill)
                        except Exception as e:
                            log.warning(f"Fill callback error: {e}")

                    # Update auto-trader state if this was an auto-trade
                    if (
                        self.auto_trader
                        and order.tags.get("auto_trade")
                    ):
                        self._update_auto_trade_fill(order, fill)

                    inc_counter(
                        "fills_processed_total",
                        labels={"side": fill.side.value},
                    )
                    observe(
                        "fill_latency_seconds",
                        (
                            (datetime.now() - fill.timestamp).total_seconds()
                            if fill.timestamp
                            else 0.0
                        ),
                    )

            except Exception as e:
                log.error(f"Fill processing error: {e}")

            self._prune_processed_fills_unlocked(max_size=50000)

    def _update_auto_trade_fill(self, order: Order, fill: Fill):
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

    def _cancel_oco_siblings(self, oms, filled_order: Order, fill: Fill) -> None:
        tags = dict(getattr(filled_order, "tags", {}) or {})
        oco_group = str(tags.get("oco_group", "") or "").strip()
        if not oco_group or int(getattr(fill, "quantity", 0) or 0) <= 0:
            return

        try:
            siblings = oms.get_orders(filled_order.symbol)
        except Exception as e:
            log.debug(f"OCO sibling scan failed for {filled_order.symbol}: {e}")
            return

        active_statuses = {
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIAL,
        }
        cancelled = 0
        for sibling in siblings:
            if sibling.id == filled_order.id:
                continue
            if sibling.status not in active_statuses:
                continue
            s_tags = dict(getattr(sibling, "tags", {}) or {})
            sibling_group = str(s_tags.get("oco_group", "") or "").strip()
            if sibling_group != oco_group:
                continue

            cancel_ok = False
            broker_ids = [
                str(getattr(sibling, "broker_id", "") or "").strip(),
                str(getattr(sibling, "id", "") or "").strip(),
            ]
            for oid in broker_ids:
                if not oid:
                    continue
                try:
                    if bool(self.broker.cancel_order(oid)):
                        cancel_ok = True
                        break
                except Exception:
                    continue

            if not cancel_ok and sibling.status == OrderStatus.PENDING:
                cancel_ok = True

            if not cancel_ok:
                log.warning(
                    "OCO sibling cancel failed: %s group=%s source_fill=%s",
                    sibling.id,
                    oco_group,
                    filled_order.id,
                )
                continue

            try:
                oms.update_order_status(
                    sibling.id,
                    OrderStatus.CANCELLED,
                    message=f"OCO cancelled after fill {filled_order.id}",
                )
                cancelled += 1
            except Exception as e:
                log.warning(f"OCO sibling OMS cancel update failed: {sibling.id}: {e}")

        if cancelled > 0:
            log.info(
                "OCO siblings cancelled: group=%s symbol=%s count=%s",
                oco_group,
                filled_order.symbol,
                cancelled,
            )

    def _fill_sync_loop(self):
        """Poll broker for fills."""
        while self._running:
            self._heartbeat("fill_sync")
            try:
                time.sleep(1.0)
                if not self.broker.is_connected:
                    continue
                self._process_pending_fills()
                self._evaluate_synthetic_exits()
            except Exception as e:
                log.error(f"Fill sync loop error: {e}")

    def _record_execution_quality(self, order: Order, fill: Fill) -> None:
        try:
            tags = dict(getattr(order, "tags", {}) or {})
            arrival = float(tags.get("arrival_price", 0.0) or 0.0)
            if arrival <= 0:
                arrival = float(getattr(order, "price", 0.0) or 0.0)
            if arrival <= 0 or float(fill.price) <= 0:
                return

            if fill.side == OrderSide.BUY:
                slip_bps = (float(fill.price) / arrival - 1.0) * 10000.0
            else:
                slip_bps = (arrival / float(fill.price) - 1.0) * 10000.0

            reason = str(tags.get("exit_reason", "entry") or "entry")
            with self._exec_quality_lock:
                self._exec_quality["fills"] = int(
                    self._exec_quality.get("fills", 0)
                ) + 1
                self._exec_quality["slippage_bps_sum"] = float(
                    self._exec_quality.get("slippage_bps_sum", 0.0)
                ) + float(slip_bps)
                self._exec_quality["slippage_bps_abs_sum"] = float(
                    self._exec_quality.get("slippage_bps_abs_sum", 0.0)
                ) + abs(float(slip_bps))
                by_reason = self._exec_quality.setdefault("by_reason", {})
                by_reason[str(reason)] = int(by_reason.get(str(reason), 0)) + 1
                self._exec_quality["last_update"] = datetime.now().isoformat()

            observe(
                "execution_slippage_bps",
                float(slip_bps),
                labels={"side": fill.side.value},
            )
        except Exception as e:
            log.debug(f"Execution quality accounting failed: {e}")

    def _maybe_register_synthetic_exit(self, order: Order, fill: Fill) -> None:
        # Synthetic bracket/OCO exits for filled entry orders.
        if order.side != OrderSide.BUY:
            return
        tags = dict(getattr(order, "tags", {}) or {})
        if bool(tags.get("synthetic_exit", False)):
            return

        stop_loss = float(getattr(order, "stop_loss", 0.0) or 0.0)
        take_profit = float(getattr(order, "take_profit", 0.0) or 0.0)
        trailing_pct = float(tags.get("trailing_stop_pct", 0.0) or 0.0)
        if stop_loss <= 0 and take_profit <= 0 and trailing_pct <= 0:
            return

        changed = False
        with self._synthetic_exit_lock:
            plan = self._synthetic_exits.get(order.id)
            if plan is None:
                plan = {
                    "plan_id": order.id,
                    "source_order_id": order.id,
                    "symbol": order.symbol,
                    "side": "long",
                    "open_qty": 0,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "trailing_stop_pct": trailing_pct,
                    "highest_price": float(fill.price),
                    "armed_at": datetime.now().isoformat(),
                }
                self._synthetic_exits[order.id] = plan
                changed = True
            prev_qty = int(plan.get("open_qty", 0) or 0)
            prev_high = float(plan.get("highest_price", 0.0) or 0.0)
            plan["open_qty"] = prev_qty + int(fill.quantity)
            plan["highest_price"] = max(
                float(plan.get("highest_price", 0.0) or 0.0),
                float(fill.price),
            )
            if plan["open_qty"] != prev_qty or plan["highest_price"] != prev_high:
                changed = True
            if trailing_pct > 0:
                dyn_stop = float(plan["highest_price"]) * (1.0 - trailing_pct / 100.0)
                cur_stop = float(plan.get("stop_loss", 0.0) or 0.0)
                new_stop = max(cur_stop, dyn_stop) if cur_stop > 0 else dyn_stop
                if float(new_stop) != float(cur_stop):
                    changed = True
                plan["stop_loss"] = new_stop

        if changed:
            self._persist_synthetic_exits(force=False)

    def _evaluate_synthetic_exits(self) -> None:
        with self._synthetic_exit_lock:
            plans = list(self._synthetic_exits.values())

        for plan in plans:
            persist_needed = False
            symbol = str(plan.get("symbol", "") or "").strip()
            qty = int(plan.get("open_qty", 0) or 0)
            if not symbol or qty <= 0:
                continue

            px, _, _ = self._get_quote_snapshot(symbol)
            if px <= 0:
                continue

            trailing_pct = float(plan.get("trailing_stop_pct", 0.0) or 0.0)
            if trailing_pct > 0:
                with self._synthetic_exit_lock:
                    cur = self._synthetic_exits.get(str(plan.get("plan_id", "")))
                    if cur is not None:
                        old_high = float(cur.get("highest_price", 0.0) or 0.0)
                        old_stop = float(cur.get("stop_loss", 0.0) or 0.0)
                        cur["highest_price"] = max(
                            float(cur.get("highest_price", 0.0) or 0.0),
                            float(px),
                        )
                        dyn_stop = float(cur["highest_price"]) * (
                            1.0 - trailing_pct / 100.0
                        )
                        cur_stop = float(cur.get("stop_loss", 0.0) or 0.0)
                        cur["stop_loss"] = max(cur_stop, dyn_stop) if cur_stop > 0 else dyn_stop
                        if (
                            float(cur.get("highest_price", 0.0) or 0.0) != old_high
                            or float(cur.get("stop_loss", 0.0) or 0.0) != old_stop
                        ):
                            persist_needed = True
                        plan = dict(cur)

            take_profit = float(plan.get("take_profit", 0.0) or 0.0)
            stop_loss = float(plan.get("stop_loss", 0.0) or 0.0)
            reason = ""
            if take_profit > 0 and px >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0 and px <= stop_loss:
                reason = "stop_loss"
            if not reason:
                if persist_needed:
                    self._persist_synthetic_exits(force=False)
                continue

            if self._submit_synthetic_exit(plan, trigger_price=float(px), reason=reason):
                with self._synthetic_exit_lock:
                    removed = self._synthetic_exits.pop(str(plan.get("plan_id", "")), None)
                    if removed is not None:
                        persist_needed = True
            if persist_needed:
                self._persist_synthetic_exits(force=True)

    def _submit_synthetic_exit(
        self,
        plan: dict[str, Any],
        trigger_price: float,
        reason: str,
    ) -> bool:
        from trading.oms import get_oms

        symbol = str(plan.get("symbol", "") or "").strip()
        qty = int(plan.get("open_qty", 0) or 0)
        if not symbol or qty <= 0:
            return False

        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=qty,
            order_type=OrderType.MARKET,
            price=0.0,
            strategy="synthetic_bracket_exit",
            parent_id=str(plan.get("source_order_id", "") or ""),
        )
        order.tags["synthetic_exit"] = True
        order.tags["exit_reason"] = str(reason)
        order.tags["exit_parent_order_id"] = str(plan.get("source_order_id", "") or "")
        order.tags["arrival_price"] = float(trigger_price or 0.0)

        try:
            oms = get_oms()
            order = oms.submit_order(order)
            result = self._submit_with_retry(order, attempts=2)
            oms.update_order_status(
                order.id,
                result.status,
                message=result.message or "",
                broker_id=result.broker_id or "",
            )
            if result.status == OrderStatus.REJECTED:
                self._alert_manager.risk_alert(
                    "Synthetic exit rejected",
                    f"{symbol}: {result.message or 'rejected'}",
                    {"reason": reason, "qty": qty},
                )
                return False
            log.warning(
                "Synthetic %s exit submitted: %s qty=%s trigger=%.2f",
                reason,
                symbol,
                qty,
                float(trigger_price or 0.0),
            )
            return True
        except Exception as e:
            log.error(f"Synthetic exit submit failed: {symbol} {reason}: {e}")
            return False

    def _prune_processed_fills_unlocked(self, max_size: int = 50000):
        """
        Prevent unbounded growth of processed fill IDs.
        MUST be called with self._fills_lock already held.

        FIX: Single lock acquisition instead of double-lock pattern.
        """
        if len(self._processed_fill_ids) <= int(max_size):
            return

        try:
            from trading.oms import get_oms

            oms = get_oms()
            fills = oms.get_fills()
        except Exception:
            fills = []

        keep = set()
        for f in fills[:int(max_size)]:
            fid = str(getattr(f, "id", "") or "").strip()
            if fid:
                keep.add(fid)

        self._processed_fill_ids = keep

    def _status_sync_loop(self):
        """
        Poll broker for order status updates.
        Includes stuck order watchdog.
        """
        from trading.oms import get_oms

        oms = get_oms()
        first_seen: dict[str, datetime] = {}

        stuck_seconds = 60
        if hasattr(CONFIG, "risk") and hasattr(CONFIG.risk, "order_stuck_seconds"):
            stuck_seconds = int(CONFIG.risk.order_stuck_seconds)

        while self._running:
            self._heartbeat("status_sync")
            try:
                time.sleep(3.0)
                if not self.broker.is_connected:
                    continue

                active_orders = oms.get_active_orders()
                now = datetime.now()

                for order in active_orders:
                    first_seen.setdefault(order.id, now)
                    age = (now - first_seen[order.id]).total_seconds()

                    broker_status = None
                    try:
                        broker_status = self.broker.get_order_status(order.id)
                    except Exception:
                        broker_status = None

                    # If broker_status missing, try broker.sync_order
                    if broker_status is None:
                        try:
                            synced = self.broker.sync_order(order)
                            broker_status = getattr(synced, "status", None)
                            if (
                                synced
                                and getattr(synced, "broker_id", None)
                                and not order.broker_id
                            ):
                                oms.update_order_status(
                                    order.id,
                                    order.status,
                                    broker_id=synced.broker_id,
                                    message="Recovered broker_id",
                                )
                        except Exception:
                            broker_status = None

                    # If broker says FILLED: process fills immediately
                    if broker_status == OrderStatus.FILLED:
                        self._process_pending_fills()
                        refreshed = oms.get_order(order.id)
                        if (
                            refreshed
                            and refreshed.status != OrderStatus.FILLED
                            and age > 30
                        ):
                            self._alert_manager.risk_alert(
                                "Missing Fills After Broker FILLED",
                                (
                                    f"{order.symbol}: broker FILLED but "
                                    f"OMS {refreshed.status.value}"
                                ),
                                details={
                                    "order_id": order.id,
                                    "broker_id": order.broker_id,
                                },
                            )
                        first_seen.pop(order.id, None)
                        continue

                    if broker_status and broker_status != order.status:
                        oms.update_order_status(
                            order.id,
                            broker_status,
                            message=f"Status sync: {broker_status.value}",
                        )
                        if broker_status in (
                            OrderStatus.CANCELLED,
                            OrderStatus.REJECTED,
                            OrderStatus.FILLED,
                            OrderStatus.EXPIRED,
                        ):
                            first_seen.pop(order.id, None)
                        continue

                    if age >= stuck_seconds and order.status in (
                        OrderStatus.SUBMITTED,
                        OrderStatus.ACCEPTED,
                        OrderStatus.PARTIAL,
                    ):
                        self._alert_manager.risk_alert(
                            "Order Stuck Watchdog",
                            (
                                f"{order.symbol}: {order.status.value} "
                                f"for {int(age)}s"
                            ),
                            details={
                                "order_id": order.id,
                                "broker_id": order.broker_id,
                                "status": order.status.value,
                            },
                        )
                        # FIX: Only auto-cancel if explicitly enabled
                        if self.AUTO_CANCEL_STUCK_ORDERS:
                            try:
                                self.broker.cancel_order(order.id)
                            except Exception:
                                pass

            except Exception as e:
                log.error(f"Status sync error: {e}")

    def _reconciliation_loop(self):
        """Periodic reconciliation."""
        from trading.oms import get_oms

        oms = get_oms()

        while self._running:
            self._heartbeat("recon")
            try:
                time.sleep(300)
                if not self.broker.is_connected:
                    continue

                broker_account = self.broker.get_account()
                broker_positions = self.broker.get_positions()
                discrepancies = oms.reconcile(
                    broker_positions, broker_account.cash
                )

                if (
                    abs(discrepancies.get('cash_diff', 0.0)) > 1.0
                    or discrepancies.get('position_diffs')
                    or discrepancies.get('missing_positions')
                    or discrepancies.get('extra_positions')
                ):
                    self._alert_manager.risk_alert(
                        "Reconciliation Discrepancy",
                        f"Cash diff: {discrepancies.get('cash_diff', 0):.2f}",
                        discrepancies,
                    )
            except Exception as e:
                log.error(f"Reconciliation error: {e}")

    def _submit_with_retry(self, order: Order, attempts: int = 3) -> Order:
        """
        Retry broker.submit_order for transient failures.
        Does NOT retry validation failures (broker REJECTED).
        """
        delay = 0.5
        last_exc = None
        for _i in range(int(attempts)):
            try:
                result = self.broker.submit_order(order)
                if getattr(result, "status", None) == OrderStatus.REJECTED:
                    return result
                return result
            except Exception as e:
                last_exc = e
                time.sleep(delay)
                delay = min(delay * 2.0, 5.0)

        raise last_exc if last_exc else RuntimeError("submit_order failed")

    def _broker_reconnect_loop(self):
        """Reconnect with exponential backoff."""
        backoff = 1.0
        while self._running:
            self._heartbeat("reconnect")
            try:
                time.sleep(2.0)
                if self.broker.is_connected:
                    backoff = 1.0
                    continue

                log.warning(
                    f"Broker disconnected. Reconnecting in {backoff:.0f}s..."
                )
                time.sleep(backoff)

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

            except Exception as e:
                log.warning(f"Reconnect loop error: {e}")
                backoff = min(backoff * 2.0, 60.0)

    def _on_kill_switch(self, reason: str):
        """Handle kill switch activation."""
        log.critical(f"Kill switch activated: {reason}")

        # Stop auto-trading immediately
        if self.auto_trader:
            try:
                self.auto_trader.pause(f"Kill switch: {reason}")
            except Exception:
                pass

        try:
            for order in self.broker.get_orders(active_only=True):
                try:
                    self.broker.cancel_order(order.id)
                except Exception:
                    pass
        except Exception as e:
            log.error(f"Failed to cancel orders: {e}")

        try:
            while not self._queue.empty():
                self._queue.get_nowait()
        except Exception:
            pass

        self._alert_manager.critical_alert(
            "KILL SWITCH ACTIVATED", f"All trading halted: {reason}"
        )

    def _reject_signal(self, signal: TradeSignal, reason: str):
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

    def get_positions(self):
        from trading.oms import get_oms

        return get_oms().get_positions()

    def get_orders(self):
        return self.broker.get_orders()
