# trading/executor.py
from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timedelta, date
from typing import Dict, Optional, Callable, Set, List, Tuple

from config import CONFIG, TradingMode
from core.types import (
    Order, OrderSide, OrderStatus, TradeSignal, Account, Fill,
    AutoTradeMode, AutoTradeState, AutoTradeAction,
)
from trading.broker import BrokerInterface, create_broker
from trading.risk import RiskManager, get_risk_manager
from trading.kill_switch import get_kill_switch
from trading.health import get_health_monitor, ComponentType, HealthStatus
from trading.alerts import get_alert_manager, AlertPriority
from utils.security import get_access_control, get_audit_log
from utils.logger import get_logger
from utils.metrics import inc_counter, set_gauge, observe

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
        engine: "ExecutionEngine",
        predictor,
        watch_list: List[str],
    ):
        self._engine = engine
        self._predictor = predictor
        self._watch_list = list(watch_list)

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.state = AutoTradeState()

        self.on_action: Optional[Callable[[AutoTradeAction], None]] = None
        self.on_state_changed: Optional[Callable[[AutoTradeState], None]] = None
        self.on_pending_approval: Optional[Callable[[AutoTradeAction], None]] = None

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

    def update_watchlist(self, watch_list: List[str]):
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
            return self.state

    def get_recent_actions(self, n: int = 50) -> List[AutoTradeAction]:
        """Get recent auto-trade actions."""
        with self._lock:
            return list(self.state.recent_actions[:n])

    def get_pending_approvals(self) -> List[AutoTradeAction]:
        """Get pending approvals (SEMI_AUTO mode)."""
        with self._lock:
            return list(self.state.pending_approvals)

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
            from models.predictor import Signal as PredSignal
        except ImportError:
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
                self.on_state_changed(self.state)
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

    def __init__(self, mode: TradingMode = None):
        self.mode = mode or CONFIG.trading_mode
        self.broker: BrokerInterface = create_broker(self.mode.value)
        self.risk_manager: RiskManager = get_risk_manager()

        self._kill_switch = get_kill_switch()
        self._health_monitor = get_health_monitor()
        self._alert_manager = get_alert_manager()
        self._fills_lock = threading.RLock()

        self._queue: queue.Queue[Optional[TradeSignal]] = queue.Queue()
        self._running = False

        self._exec_thread: Optional[threading.Thread] = None
        self._fill_sync_thread: Optional[threading.Thread] = None
        self._status_sync_thread: Optional[threading.Thread] = None
        self._recon_thread: Optional[threading.Thread] = None
        self._reconnect_thread: Optional[threading.Thread] = None

        self._processed_fill_ids: Set[str] = set()

        self._last_fill_sync: Optional[datetime] = None

        self.on_fill: Optional[Callable[[Order, Fill], None]] = None
        self.on_reject: Optional[Callable[[Order, str], None]] = None

        self._kill_switch.on_activate(self._on_kill_switch)
        self._health_monitor.on_degraded(self._on_health_degraded)
        self._processed_fill_ids = self._load_processed_fills()

        # Auto-trader (created but not started until explicitly requested)
        self.auto_trader: Optional[AutoTrader] = None

    # -----------------------------------------------------------------
    # Auto-trade public API
    # -----------------------------------------------------------------

    def init_auto_trader(self, predictor, watch_list: List[str]):
        """
        Initialize the auto-trader with a predictor and watchlist.
        Must be called before start_auto_trade().
        """
        self.auto_trader = AutoTrader(
            engine=self,
            predictor=predictor,
            watch_list=watch_list,
        )
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

    def get_auto_trade_state(self) -> Optional[AutoTradeState]:
        """Get auto-trade state snapshot."""
        if self.auto_trader:
            return self.auto_trader.get_state()
        return None

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------

    def start(self) -> bool:
        if self._running:
            return True

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

        self._health_monitor.attach_broker(self.broker)
        self._health_monitor.report_component_health(
            ComponentType.BROKER, HealthStatus.HEALTHY
        )

        log.info(f"Execution engine started ({self.mode.value})")

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
            auto_mode = AutoTradeMode.AUTO
            self.auto_trader.start()
            log.info("Auto-trader auto-started (config.auto_trade.enabled=True)")

        return True

    def _load_processed_fills(self) -> Set[str]:
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

        log.info("Execution engine stopped")

    def _get_quote_snapshot(
        self, symbol: str
    ) -> Tuple[float, Optional[datetime], str]:
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
    ) -> Tuple[bool, str, float]:
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
    ) -> Tuple[bool, str, float]:
        """Public wrapper for quote freshness gating."""
        max_age = 15.0
        if hasattr(CONFIG, "risk") and hasattr(CONFIG.risk, "quote_staleness_seconds"):
            max_age = float(CONFIG.risk.quote_staleness_seconds)
        return self._require_fresh_quote(symbol, max_age_seconds=max_age)

    def submit(self, signal: TradeSignal) -> bool:
        """Submit a trading signal for execution with strict quote freshness."""
        if not self._running:
            log.warning("Execution engine not running")
            return False

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

        signal.price = float(fresh_px)

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
        order: Optional[Order] = None

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
                stop_loss=(
                    float(signal.stop_loss) if signal.stop_loss else 0.0
                ),
                take_profit=(
                    float(signal.take_profit) if signal.take_profit else 0.0
                ),
                signal_id=signal.id,
                strategy=signal.strategy or "",
            )

            # Tag auto-trade orders for audit
            if signal.auto_generated:
                order.tags["auto_trade"] = True
                order.tags["auto_trade_action_id"] = signal.auto_trade_action_id
                order.strategy = order.strategy or "auto_trade"

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

    def _fill_sync_loop(self):
        """Poll broker for fills."""
        while self._running:
            try:
                time.sleep(1.0)
                if not self.broker.is_connected:
                    continue
                self._process_pending_fills()
            except Exception as e:
                log.error(f"Fill sync loop error: {e}")

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
        first_seen: Dict[str, datetime] = {}

        stuck_seconds = 60
        if hasattr(CONFIG, "risk") and hasattr(CONFIG.risk, "order_stuck_seconds"):
            stuck_seconds = int(CONFIG.risk.order_stuck_seconds)

        while self._running:
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
        for i in range(int(attempts)):
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
