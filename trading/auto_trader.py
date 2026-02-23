# trading/auto_trader.py
from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from config import CONFIG
from core.types import AutoTradeAction, AutoTradeMode, AutoTradeState, OrderSide, TradeSignal
from utils.logger import get_logger

if TYPE_CHECKING:
    from trading.executor import ExecutionEngine

log = get_logger(__name__)

class AutoTrader:
    """Autonomous trading engine that scans the watchlist using the AI
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
        - The scan loop runs on a dedicated worker thread
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
        self._loop_lock = threading.RLock()

        self.state = AutoTradeState()

        self.on_action: Callable[[AutoTradeAction], None] | None = None
        self.on_state_changed: Callable[[AutoTradeState], None] | None = None
        self.on_pending_approval: Callable[[AutoTradeAction], None] | None = None

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------

    def set_mode(self, mode: AutoTradeMode):
        """Change trading mode. Stops/starts scan loop as needed."""
        should_stop = False
        should_start = False
        with self._lock:
            old_mode = self.state.mode
            self.state.mode = mode

            if mode == AutoTradeMode.MANUAL:
                should_stop = self._is_loop_running()
            else:
                # AUTO or SEMI_AUTO 鈥?start loop if not running
                should_start = not self._is_loop_running()

        if should_stop:
            self._stop_loop()
        elif should_start:
            self._start_loop()

        with self._lock:
            self.state.is_running = (
                self.state.mode != AutoTradeMode.MANUAL
                and self._is_loop_running()
            )
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
                log.info("Auto-trader in MANUAL mode 鈥?scan loop not started")
                return

            if self._is_loop_running():
                return

            self._start_loop()
            self.state.is_running = True
            self._notify_state_changed()
            log.info(f"Auto-trader started in {self.state.mode.value} mode")

    def stop(self):
        """Stop the auto-trader scan loop."""
        self._stop_loop()
        with self._lock:
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
        """Additional precision guardrails for autonomous trading.
        Uses entropy and directional edge thresholds from PrecisionConfig.
        """
        try:
            p_cfg = getattr(CONFIG, "precision", None)
            if not p_cfg or not bool(getattr(p_cfg, "enabled", True)):
                return True, ""

            block_short_history = bool(
                getattr(
                    p_cfg,
                    "block_auto_trade_on_short_history_fallback",
                    True,
                )
            )
            warnings = list(getattr(pred, "warnings", []) or [])
            short_history_fallback = bool(getattr(pred, "short_history_fallback", False))
            if not short_history_fallback:
                short_history_fallback = any(
                    "short-history fallback used" in str(item).lower()
                    for item in warnings
                )
            if block_short_history and short_history_fallback:
                return False, "Short-history fallback prediction blocked for auto-trade"

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
        except Exception as e:
            p_cfg = getattr(CONFIG, "precision", None)
            fail_closed = bool(
                getattr(p_cfg, "fail_closed_on_quality_gate_error", True)
            )
            if fail_closed:
                log.warning(
                    "Precision quality gate failed-closed due to config/runtime error: %s",
                    e,
                )
                return False, "Quality gate error (fail-closed policy)"
            log.debug("Precision quality gate failed-open due to config error: %s", e)
            return True, ""

        return True, ""

    def _cap_buy_shares_by_risk(
        self,
        pred,
        shares: int,
        price: float,
    ) -> tuple[int, str]:
        """Cap model-proposed BUY size using RiskManager position sizing.

        Keeps the model quantity as an upper bound and never increases size.
        """
        shares_i = int(shares)
        price_f = float(price)
        if shares_i <= 0:
            return shares_i, ""
        if price_f <= 0:
            return 0, "No valid position size"

        risk_mgr = getattr(self._engine, "risk_manager", None)
        if risk_mgr is None:
            return shares_i, ""

        stop_loss = 0.0
        levels = getattr(pred, "levels", None)
        if levels is not None:
            try:
                stop_loss = float(getattr(levels, "stop_loss", 0.0) or 0.0)
            except (TypeError, ValueError):
                stop_loss = 0.0

        if not (0.0 < stop_loss < price_f):
            atr_pct = float(getattr(pred, "atr_pct_value", 0.0) or 0.0)
            fallback_pct = (atr_pct * 1.5) if atr_pct > 0 else 0.02
            fallback_pct = max(0.01, min(0.08, float(fallback_pct)))
            stop_loss = price_f * (1.0 - fallback_pct)

        try:
            capped = int(
                risk_mgr.calculate_position_size(
                    symbol=str(getattr(pred, "stock_code", "") or ""),
                    entry_price=price_f,
                    stop_loss=float(stop_loss),
                    confidence=float(getattr(pred, "confidence", 0.0) or 0.0),
                    signal_strength=float(
                        getattr(pred, "signal_strength", 1.0) or 1.0
                    ),
                )
            )
        except Exception as e:
            log.debug(
                "Risk size cap failed for %s: %s",
                getattr(pred, "stock_code", "?"),
                e,
            )
            return shares_i, ""

        if capped <= 0:
            return 0, "Risk sizing blocked position"

        return min(shares_i, int(capped)), ""

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
        with self._loop_lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._scan_loop,
                name="auto_trader",
                daemon=False,
            )
            self._thread.start()

    def _stop_loop(self):
        """Stop the scan loop thread."""
        with self._loop_lock:
            self._stop_event.set()
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=10)
        with self._loop_lock:
            if self._thread is thread:
                self._thread = None

    # -----------------------------------------------------------------
    # Internal: main scan loop
    # -----------------------------------------------------------------

    def _scan_loop(self):
        """Main auto-trading scan loop.

        Runs on a dedicated worker thread. Each cycle:
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
                should_scan = False
                with self._lock:
                    today = date.today()
                    if self.state.session_date != today:
                        self.state.reset_daily()
                        log.info("Auto-trader: new trading day, counters reset")

                    should_scan = self._should_scan()
                    if should_scan:
                        self.state.last_scan_time = datetime.now()

                if not should_scan:
                    self._sleep_interruptible(5)
                    continue

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
        """Check all pre-conditions for scanning.
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
        """Execute one scan cycle: predict 鈫?filter 鈫?execute/queue.
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
        except Exception as e:
            log.debug("Precision auto-trade filter unavailable: %s", e)

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

            if full_sig in (Signal.STRONG_BUY, Signal.BUY):
                side = OrderSide.BUY
            elif full_sig in (Signal.STRONG_SELL, Signal.SELL):
                side = OrderSide.SELL
            else:
                continue

            price = getattr(full_pred, 'current_price', 0.0)
            position = getattr(full_pred, 'position', None)
            shares = getattr(position, 'shares', 0) if position else 0
            if side == OrderSide.BUY and int(shares) > 0:
                shares, risk_msg = self._cap_buy_shares_by_risk(
                    full_pred, shares, price
                )
                if shares <= 0:
                    with self._lock:
                        self._record_skip(
                            full_pred,
                            Signal,
                            risk_msg or "Risk sizing blocked position",
                        )
                    continue
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
                                f"Order value exceeds limit (楼{cfg.max_auto_order_value:,.0f})"
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
                        f"{action.signal_type} {code} @ 楼{price:.2f}"
                    )

    def _execute_action(self, action: AutoTradeAction) -> bool:
        """Execute a single auto-trade action via ExecutionEngine.submit().

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
        except Exception as e:
            log.debug("Auto-trade level enrichment failed for %s: %s", action.stock_code, e)

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
                f"@ 楼{levels_entry:.2f} (conf={action.confidence:.0%})"
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


