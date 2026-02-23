from __future__ import annotations

from datetime import datetime
from typing import Any

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config.settings import CONFIG, TradingMode
from core.types import AutoTradeAction, AutoTradeMode
from ui.modern_theme import (
    ModernColors,
    ModernFonts,
    get_status_badge_style,
)
from utils.logger import get_logger

log = get_logger(__name__)


def _init_auto_trader(self: Any) -> None:
    """Initialize auto-trader on the execution engine."""
    if self.executor and self.predictor:
        try:
            self.executor.init_auto_trader(self.predictor, self.watch_list)

            if self.executor.auto_trader:
                self.executor.auto_trader.on_action = self._on_auto_trade_action_safe
                self.executor.auto_trader.on_pending_approval = self._on_pending_approval_safe

            self.log("Auto-trader initialized", "info")
        except Exception as exc:
            log.warning("Auto-trader init failed: %s", exc)
    elif self.predictor and not self.executor:
        # Executor not connected yet; will init when connected.
        pass


def _on_trade_mode_changed(self: Any, index: int) -> None:
    """Handle trade mode combo box change."""
    mode_map = {
        0: AutoTradeMode.MANUAL,
        1: AutoTradeMode.AUTO,
        2: AutoTradeMode.SEMI_AUTO,
    }
    new_mode = mode_map.get(index, AutoTradeMode.MANUAL)

    if new_mode == AutoTradeMode.AUTO:
        if self.predictor is None or (self.predictor and self.predictor.ensemble is None):
            QMessageBox.warning(
                self,
                "Cannot Enable Auto-Trade",
                "No AI model loaded. Train a model first.",
            )
            self.trade_mode_combo.setCurrentIndex(0)
            return

        if self.executor is None:
            QMessageBox.warning(
                self,
                "Cannot Enable Auto-Trade",
                "Not connected to broker. Connect first.",
            )
            self.trade_mode_combo.setCurrentIndex(0)
            return

        if (
            self.executor
            and self.executor.mode == TradingMode.LIVE
            and CONFIG.auto_trade.confirm_live_auto_trade
        ):
            reply = QMessageBox.warning(
                self,
                "LIVE Auto-Trading",
                "You are enabling AUTOMATIC trading with REAL MONEY!\n\n"
                "The AI will execute trades WITHOUT your confirmation.\n\n"
                "Risk limits still apply, but trades happen automatically.\n\n"
                "Are you absolutely sure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self.trade_mode_combo.setCurrentIndex(0)
                return

        reply = QMessageBox.question(
            self,
            "Enable Auto-Trading",
            "Enable fully automatic trading?\n\n"
            f"- Min confidence: {CONFIG.auto_trade.min_confidence:.0%}\n"
            f"- Max trades/day: {CONFIG.auto_trade.max_trades_per_day}\n"
            f"- Max order value: CNY {CONFIG.auto_trade.max_auto_order_value:,.0f}\n"
            f"- Max auto positions: {CONFIG.auto_trade.max_auto_positions}\n\n"
            "You can pause or switch to Manual at any time.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self.trade_mode_combo.setCurrentIndex(0)
            return

    self._auto_trade_mode = new_mode
    self._apply_auto_trade_mode(new_mode)


def _apply_auto_trade_mode(self: Any, mode: AutoTradeMode) -> None:
    """Apply the auto-trade mode to the system."""
    CONFIG.auto_trade.enabled = mode != AutoTradeMode.MANUAL

    if self.executor and self.executor.auto_trader:
        self.executor.set_auto_mode(mode)
        self.executor.auto_trader.update_watchlist(self.watch_list)
        if self.predictor:
            self.executor.auto_trader.update_predictor(self.predictor)
    elif mode != AutoTradeMode.MANUAL:
        self._init_auto_trader()
        if self.executor and self.executor.auto_trader:
            self.executor.set_auto_mode(mode)

    self._update_auto_trade_status_label(mode)

    if mode == AutoTradeMode.AUTO:
        self.buy_btn.setEnabled(False)
        self.sell_btn.setEnabled(False)
        self.auto_pause_btn.setEnabled(True)
        self.log("AUTO mode enabled: AI executes trades automatically", "success")
    elif mode == AutoTradeMode.SEMI_AUTO:
        self.auto_pause_btn.setEnabled(True)
        self.auto_approve_all_btn.setEnabled(True)
        self.auto_reject_all_btn.setEnabled(True)
        self.log(
            "SEMI-AUTO mode enabled: AI suggests and you approve",
            "success",
        )
    else:
        self.auto_pause_btn.setEnabled(False)
        self.auto_approve_all_btn.setEnabled(False)
        self.auto_reject_all_btn.setEnabled(False)
        self.log("MANUAL mode enabled: you control all trades", "info")


def _update_auto_trade_status_label(self: Any, mode: AutoTradeMode) -> None:
    """Update the toolbar status label."""
    if mode == AutoTradeMode.AUTO:
        self.auto_trade_status_label.setText("  AUTO  ")
        self.auto_trade_status_label.setStyleSheet(
            get_status_badge_style("auto")
        )
    elif mode == AutoTradeMode.SEMI_AUTO:
        self.auto_trade_status_label.setText("  SEMI-AUTO  ")
        self.auto_trade_status_label.setStyleSheet(
            get_status_badge_style("semi-auto")
        )
    else:
        self.auto_trade_status_label.setText("  MANUAL  ")
        self.auto_trade_status_label.setStyleSheet(
            get_status_badge_style("manual")
        )


def _toggle_auto_pause(self: Any) -> None:
    """Pause/resume auto-trading."""
    if not self.executor or not self.executor.auto_trader:
        return

    state = self.executor.auto_trader.get_state()
    if state.is_safety_paused or state.is_paused:
        self.executor.auto_trader.resume()
        self.auto_pause_btn.setText("Pause Auto")
        self.log("Auto-trading resumed", "info")
    else:
        self.executor.auto_trader.pause("Manually paused by user")
        self.auto_pause_btn.setText("Resume Auto")
        self.log("Auto-trading paused", "warning")


def _approve_all_pending(self: Any) -> None:
    """Approve all pending auto-trade actions."""
    if not self.executor or not self.executor.auto_trader:
        return

    pending = self.executor.auto_trader.get_pending_approvals()
    if not pending:
        self.log("No pending approvals", "info")
        return

    reply = QMessageBox.question(
        self,
        "Approve All",
        f"Approve all {len(pending)} pending trades?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if reply != QMessageBox.StandardButton.Yes:
        return

    approved = 0
    for action in pending:
        if self.executor.auto_trader.approve_pending(action.id):
            approved += 1

    self.log(f"Approved {approved}/{len(pending)} pending trades", "success")


def _reject_all_pending(self: Any) -> None:
    """Reject all pending auto-trade actions."""
    if not self.executor or not self.executor.auto_trader:
        return

    pending = self.executor.auto_trader.get_pending_approvals()
    for action in pending:
        self.executor.auto_trader.reject_pending(action.id)

    if pending:
        self.log(f"Rejected {len(pending)} pending trades", "warning")


def _show_auto_trade_settings(self: Any) -> None:
    """Show auto-trade settings dialog."""
    dialog = QDialog(self)
    dialog.setWindowTitle("Auto-Trade Settings")
    dialog.setMinimumWidth(500)

    layout = QVBoxLayout(dialog)
    group = QGroupBox("Auto-Trade Parameters")
    form = QFormLayout(group)

    cfg = CONFIG.auto_trade

    min_conf_spin = QDoubleSpinBox()
    min_conf_spin.setRange(0.50, 0.99)
    min_conf_spin.setValue(cfg.min_confidence)
    min_conf_spin.setSingleStep(0.05)
    min_conf_spin.setSuffix(" ")
    form.addRow("Min Confidence:", min_conf_spin)

    min_strength_spin = QDoubleSpinBox()
    min_strength_spin.setRange(0.30, 0.99)
    min_strength_spin.setValue(cfg.min_signal_strength)
    min_strength_spin.setSingleStep(0.05)
    form.addRow("Min Signal Strength:", min_strength_spin)

    min_agreement_spin = QDoubleSpinBox()
    min_agreement_spin.setRange(0.30, 0.99)
    min_agreement_spin.setValue(cfg.min_model_agreement)
    min_agreement_spin.setSingleStep(0.05)
    form.addRow("Min Model Agreement:", min_agreement_spin)

    max_positions_spin = QSpinBox()
    max_positions_spin.setRange(1, 20)
    max_positions_spin.setValue(cfg.max_auto_positions)
    form.addRow("Max Auto Positions:", max_positions_spin)

    max_order_spin = QDoubleSpinBox()
    max_order_spin.setRange(1000, 1000000)
    max_order_spin.setValue(cfg.max_auto_order_value)
    max_order_spin.setPrefix("CNY ")
    max_order_spin.setSingleStep(5000)
    form.addRow("Max Order Value:", max_order_spin)

    max_trades_spin = QSpinBox()
    max_trades_spin.setRange(1, 50)
    max_trades_spin.setValue(cfg.max_trades_per_day)
    form.addRow("Max Trades/Day:", max_trades_spin)

    max_per_stock_spin = QSpinBox()
    max_per_stock_spin.setRange(1, 10)
    max_per_stock_spin.setValue(cfg.max_trades_per_stock_per_day)
    form.addRow("Max Trades/Stock/Day:", max_per_stock_spin)

    cooldown_spin = QSpinBox()
    cooldown_spin.setRange(30, 3600)
    cooldown_spin.setValue(cfg.cooldown_after_trade_seconds)
    cooldown_spin.setSuffix(" sec")
    form.addRow("Cooldown After Trade:", cooldown_spin)

    scan_interval_spin = QSpinBox()
    scan_interval_spin.setRange(10, 600)
    scan_interval_spin.setValue(cfg.scan_interval_seconds)
    scan_interval_spin.setSuffix(" sec")
    form.addRow("Scan Interval:", scan_interval_spin)

    max_pos_pct_spin = QDoubleSpinBox()
    max_pos_pct_spin.setRange(1.0, 30.0)
    max_pos_pct_spin.setValue(cfg.max_auto_position_pct)
    max_pos_pct_spin.setSuffix(" %")
    form.addRow("Max Auto Position %:", max_pos_pct_spin)

    vol_pause_check = QCheckBox("Pause on high volatility")
    vol_pause_check.setChecked(cfg.pause_on_high_volatility)
    form.addRow("", vol_pause_check)

    auto_stop_check = QCheckBox("Auto stop-loss")
    auto_stop_check.setChecked(cfg.auto_stop_loss)
    form.addRow("", auto_stop_check)

    layout.addWidget(group)

    signals_group = QGroupBox("Allowed Signals")
    signals_layout = QGridLayout(signals_group)

    strong_buy_check = QCheckBox("STRONG_BUY")
    strong_buy_check.setChecked(cfg.allow_strong_buy)
    signals_layout.addWidget(strong_buy_check, 0, 0)

    buy_check = QCheckBox("BUY")
    buy_check.setChecked(cfg.allow_buy)
    signals_layout.addWidget(buy_check, 0, 1)

    sell_check = QCheckBox("SELL")
    sell_check.setChecked(cfg.allow_sell)
    signals_layout.addWidget(sell_check, 1, 0)

    strong_sell_check = QCheckBox("STRONG_SELL")
    strong_sell_check.setChecked(cfg.allow_strong_sell)
    signals_layout.addWidget(strong_sell_check, 1, 1)

    layout.addWidget(signals_group)

    btns = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Save
        | QDialogButtonBox.StandardButton.Cancel
    )

    def save_settings() -> None:
        cfg.min_confidence = min_conf_spin.value()
        cfg.min_signal_strength = min_strength_spin.value()
        cfg.min_model_agreement = min_agreement_spin.value()
        cfg.max_auto_positions = max_positions_spin.value()
        cfg.max_auto_order_value = max_order_spin.value()
        cfg.max_trades_per_day = max_trades_spin.value()
        cfg.max_trades_per_stock_per_day = max_per_stock_spin.value()
        cfg.cooldown_after_trade_seconds = cooldown_spin.value()
        cfg.scan_interval_seconds = scan_interval_spin.value()
        cfg.max_auto_position_pct = max_pos_pct_spin.value()
        cfg.pause_on_high_volatility = vol_pause_check.isChecked()
        cfg.auto_stop_loss = auto_stop_check.isChecked()
        cfg.allow_strong_buy = strong_buy_check.isChecked()
        cfg.allow_buy = buy_check.isChecked()
        cfg.allow_sell = sell_check.isChecked()
        cfg.allow_strong_sell = strong_sell_check.isChecked()

        try:
            CONFIG.save()
        except Exception as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)

        self.log("Auto-trade settings saved", "success")
        dialog.accept()

    btns.accepted.connect(save_settings)
    btns.rejected.connect(dialog.reject)
    layout.addWidget(btns)

    dialog.exec()


def _on_auto_trade_action_safe(self: Any, action: AutoTradeAction) -> None:
    """Thread-safe callback from auto-trader action."""
    QTimer.singleShot(0, lambda: self._on_auto_trade_action(action))


def _on_auto_trade_action(self: Any, action: AutoTradeAction) -> None:
    """Handle auto-trade action on UI thread."""
    row = 0
    self.auto_actions_table.insertRow(row)

    self.auto_actions_table.setItem(
        row,
        0,
        QTableWidgetItem(
            action.timestamp.strftime("%H:%M:%S") if action.timestamp else "--"
        ),
    )

    code_text = action.stock_code
    if action.stock_name:
        code_text += f" {action.stock_name}"
    self.auto_actions_table.setItem(row, 1, QTableWidgetItem(code_text))

    signal_item = QTableWidgetItem(action.signal_type)
    if action.signal_type in ("STRONG_BUY", "BUY"):
        signal_item.setForeground(QColor(ModernColors.ACCENT_SUCCESS))
    elif action.signal_type in ("STRONG_SELL", "SELL"):
        signal_item.setForeground(QColor(ModernColors.ACCENT_DANGER))
    self.auto_actions_table.setItem(row, 2, signal_item)

    self.auto_actions_table.setItem(row, 3, QTableWidgetItem(f"{action.confidence:.0%}"))

    decision_item = QTableWidgetItem(action.decision)
    if action.decision == "EXECUTED":
        decision_item.setForeground(QColor(ModernColors.ACCENT_SUCCESS))
    elif action.decision == "SKIPPED":
        decision_item.setForeground(QColor(ModernColors.ACCENT_WARNING))
    elif action.decision == "REJECTED":
        decision_item.setForeground(QColor(ModernColors.ACCENT_DANGER))
    self.auto_actions_table.setItem(row, 4, decision_item)

    self.auto_actions_table.setItem(
        row,
        5,
        QTableWidgetItem(f"{action.quantity:,}" if action.quantity else "--"),
    )
    self.auto_actions_table.setItem(
        row,
        6,
        QTableWidgetItem(action.skip_reason if action.skip_reason else "--"),
    )

    while self.auto_actions_table.rowCount() > 100:
        self.auto_actions_table.removeRow(self.auto_actions_table.rowCount() - 1)

    if action.decision == "EXECUTED":
        self.log(
            f"AUTO-TRADE: {action.side.upper()} "
            f"{action.quantity} {action.stock_code} "
            f"@ CNY {action.price:.2f} ({action.confidence:.0%})",
            "success",
        )
    elif action.decision == "SKIPPED":
        self.log(
            f"Auto-trade skipped {action.stock_code}: {action.skip_reason}",
            "info",
        )

    if action.decision == "EXECUTED":
        QApplication.alert(self)


def _on_pending_approval_safe(self: Any, action: AutoTradeAction) -> None:
    """Thread-safe callback for pending approval."""
    QTimer.singleShot(0, lambda: self._on_pending_approval(action))


def _on_pending_approval(self: Any, action: AutoTradeAction) -> None:
    """Handle pending approval on UI thread."""
    row = self.pending_table.rowCount()
    self.pending_table.insertRow(row)

    self.pending_table.setItem(
        row,
        0,
        QTableWidgetItem(
            action.timestamp.strftime("%H:%M:%S") if action.timestamp else "--"
        ),
    )
    self.pending_table.setItem(row, 1, QTableWidgetItem(action.stock_code))

    signal_item = QTableWidgetItem(action.signal_type)
    if action.signal_type in ("STRONG_BUY", "BUY"):
        signal_item.setForeground(QColor(ModernColors.ACCENT_SUCCESS))
    else:
        signal_item.setForeground(QColor(ModernColors.ACCENT_DANGER))
    self.pending_table.setItem(row, 2, signal_item)

    self.pending_table.setItem(row, 3, QTableWidgetItem(f"{action.confidence:.0%}"))
    self.pending_table.setItem(row, 4, QTableWidgetItem(f"CNY {action.price:.2f}"))

    btn_widget = QWidget()
    btn_layout = QHBoxLayout(btn_widget)
    btn_layout.setContentsMargins(2, 2, 2, 2)

    approve_btn = QPushButton("Approve")
    approve_btn.setFixedWidth(84)
    approve_btn.setToolTip("Approve this trade")
    action_id = action.id

    def do_approve() -> None:
        if self.executor and self.executor.auto_trader:
            self.executor.auto_trader.approve_pending(action_id)
            self._refresh_pending_table()

    approve_btn.clicked.connect(do_approve)

    reject_btn = QPushButton("Reject")
    reject_btn.setFixedWidth(84)
    reject_btn.setToolTip("Reject this trade")

    def do_reject() -> None:
        if self.executor and self.executor.auto_trader:
            self.executor.auto_trader.reject_pending(action_id)
            self._refresh_pending_table()

    reject_btn.clicked.connect(do_reject)

    btn_layout.addWidget(approve_btn)
    btn_layout.addWidget(reject_btn)
    self.pending_table.setCellWidget(row, 5, btn_widget)

    self.log(
        f"PENDING: {action.signal_type} {action.stock_code} "
        f"@ CNY {action.price:.2f} - approve or reject",
        "warning",
    )
    QApplication.alert(self)


def _refresh_pending_table(self: Any) -> None:
    """Rebuild pending table from auto-trader state."""
    self.pending_table.setRowCount(0)

    if not self.executor or not self.executor.auto_trader:
        return

    pending = self.executor.auto_trader.get_pending_approvals()
    for action in pending:
        self._on_pending_approval(action)


def _refresh_auto_trade_ui(self: Any) -> None:
    """Periodic refresh of auto-trade status display."""
    self._update_correct_guess_profit_ui()

    if not self.executor or not self.executor.auto_trader:
        self.auto_trade_labels.get("mode", QLabel()).setText(self._auto_trade_mode.value.upper())
        self.auto_trade_labels.get("trades", QLabel()).setText("0")
        self.auto_trade_labels.get("pnl", QLabel()).setText("--")
        self.auto_trade_labels.get("status", QLabel()).setText("--")
        self.auto_pause_btn.setText("Pause Auto")
        self.auto_pause_btn.setEnabled(False)
        self.auto_approve_all_btn.setText("Approve All")
        self.auto_approve_all_btn.setEnabled(False)
        self.auto_reject_all_btn.setEnabled(False)
        return

    state = self.executor.auto_trader.get_state()

    mode_label = self.auto_trade_labels.get("mode")
    if mode_label:
        mode_text = state.mode.value.upper()
        if state.is_safety_paused:
            mode_text += " (PAUSED)"
        mode_label.setText(mode_text)

        if state.mode == AutoTradeMode.AUTO:
            color = (
                ModernColors.ACCENT_DANGER
                if state.is_safety_paused
                else ModernColors.ACCENT_SUCCESS
            )
        elif state.mode == AutoTradeMode.SEMI_AUTO:
            color = ModernColors.ACCENT_WARNING
        else:
            color = ModernColors.ACCENT_INFO
        mode_label.setStyleSheet(
            
                f"color: {color}; "
                f"font-size: {ModernFonts.SIZE_XL}px; "
                f"font-weight: {ModernFonts.WEIGHT_BOLD};"
            
        )

    trades_label = self.auto_trade_labels.get("trades")
    if trades_label:
        trades_label.setText(f"{state.trades_today} (B:{state.buys_today} S:{state.sells_today})")

    pnl_label = self.auto_trade_labels.get("pnl")
    if pnl_label:
        pnl = state.auto_trade_pnl
        pnl_color = (
            ModernColors.ACCENT_SUCCESS
            if pnl >= 0
            else ModernColors.ACCENT_DANGER
        )
        pnl_label.setText(f"CNY {pnl:+,.2f}")
        pnl_label.setStyleSheet(
            
                f"color: {pnl_color}; "
                f"font-size: {ModernFonts.SIZE_XL}px; "
                f"font-weight: {ModernFonts.WEIGHT_BOLD};"
            
        )

    status_label = self.auto_trade_labels.get("status")
    if status_label:
        if state.is_safety_paused:
            status_label.setText(f"Paused: {state.pause_reason}")
            status_label.setStyleSheet(
                
                    f"color: {ModernColors.ACCENT_DANGER}; "
                    f"font-size: {ModernFonts.SIZE_LG}px; "
                    f"font-weight: {ModernFonts.WEIGHT_BOLD};"
                
            )
        elif state.is_running:
            last_scan = ""
            if state.last_scan_time:
                elapsed = (datetime.now() - state.last_scan_time).total_seconds()
                last_scan = f" ({elapsed:.0f}s ago)"
            status_label.setText(f"Running{last_scan}")
            status_label.setStyleSheet(
                
                    f"color: {ModernColors.ACCENT_SUCCESS}; "
                    f"font-size: {ModernFonts.SIZE_LG}px; "
                    f"font-weight: {ModernFonts.WEIGHT_BOLD};"
                
            )
        else:
            status_label.setText("Idle")
            status_label.setStyleSheet(
                
                    f"color: {ModernColors.ACCENT_INFO}; "
                    f"font-size: {ModernFonts.SIZE_LG}px;"
                
            )

    if state.is_safety_paused or state.is_paused:
        self.auto_pause_btn.setText("Resume Auto")
    else:
        self.auto_pause_btn.setText("Pause Auto")
    self.auto_pause_btn.setEnabled(state.mode != AutoTradeMode.MANUAL)

    pending_count = len(state.pending_approvals)
    can_bulk_decide = state.mode == AutoTradeMode.SEMI_AUTO and pending_count > 0
    if can_bulk_decide:
        self.auto_approve_all_btn.setText(f"Approve All ({pending_count})")
        self.auto_approve_all_btn.setEnabled(True)
        self.auto_reject_all_btn.setEnabled(True)
    else:
        self.auto_approve_all_btn.setText("Approve All")
        self.auto_approve_all_btn.setEnabled(False)
        self.auto_reject_all_btn.setEnabled(False)
