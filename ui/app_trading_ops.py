from __future__ import annotations

from importlib import import_module
from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
)

from config.settings import CONFIG, TradingMode
from core.constants import get_lot_size
from core.types import AutoTradeMode, OrderSide, OrderType, TradeSignal
from ui.background_tasks import (
    collect_live_readiness_failures as _collect_live_readiness_failures,
)
from ui.background_tasks import normalize_stock_code as _normalize_stock_code
from ui.modern_theme import (
    ModernColors,
    get_connection_button_style,
    get_connection_status_style,
)
from utils.logger import get_logger

log = get_logger(__name__)


def _lazy_get(module: str, name: str) -> Any:
    return getattr(import_module(module), name)


def _refresh_all(self: Any) -> None:
    """Refresh all data."""
    self._update_watchlist()
    self._refresh_portfolio()
    self.log("Refreshed all data", "info")


def _toggle_trading(self: Any) -> None:
    """Toggle trading connection."""
    if self.executor is None:
        self._connect_trading()
    else:
        self._disconnect_trading()


def _on_mode_combo_changed(self: Any, index: int) -> None:
    if self._syncing_mode_ui:
        return
    mode = TradingMode.SIMULATION if int(index) == 0 else TradingMode.LIVE
    self._set_trading_mode(mode, prompt_reconnect=True)


def _set_trading_mode(
    self: Any,
    mode: TradingMode,
    prompt_reconnect: bool = False,
) -> None:
    mode = TradingMode.LIVE if mode == TradingMode.LIVE else TradingMode.SIMULATION
    try:
        CONFIG.trading_mode = mode
    except Exception as e:
        log.warning(f"Failed to set trading mode config: {e}")

    self._syncing_mode_ui = True
    try:
        self.mode_combo.setCurrentIndex(0 if mode != TradingMode.LIVE else 1)
        if hasattr(self, "paper_action"):
            self.paper_action.setChecked(mode != TradingMode.LIVE)
        if hasattr(self, "live_action"):
            self.live_action.setChecked(mode == TradingMode.LIVE)
    finally:
        self._syncing_mode_ui = False

    self.log(f"Trading mode set: {mode.value}", "info")

    if not prompt_reconnect or self.executor is None:
        return

    current = getattr(self.executor, "mode", TradingMode.SIMULATION)
    if current == mode:
        return

    reply = QMessageBox.question(
        self,
        "Reconnect Required",
        "Trading mode changed. Reconnect now to apply new mode?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if reply == QMessageBox.StandardButton.Yes:
        self._disconnect_trading()
        self._connect_trading()


def _connect_trading(self: Any) -> None:
    """Connect to trading system."""
    mode = (
        TradingMode.SIMULATION
        if self.mode_combo.currentIndex() == 0
        else TradingMode.LIVE
    )

    if mode == TradingMode.LIVE:
        try:
            from core.network import get_network_env

            env = get_network_env()
            if not env.is_vpn_active:
                reply = QMessageBox.warning(
                    self,
                    "VPN Not Detected",
                    "LIVE trading in China typically requires VPN routing.\n\n"
                    "No VPN was detected by the network probe.\n"
                    "If you are on VPN, set TRADING_VPN=1 and retry.\n\n"
                    "Continue anyway?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.mode_combo.setCurrentIndex(0)
                    return
        except Exception as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)

        failed_controls = _collect_live_readiness_failures()
        if failed_controls:
            strict_live = bool(
                getattr(
                    getattr(CONFIG, "security", None),
                    "strict_live_governance",
                    False,
                )
            )
            preview = "\n".join(f"- {x}" for x in failed_controls[:10])
            more = ""
            if len(failed_controls) > 10:
                more = f"\n... and {len(failed_controls) - 10} more"
            msg = (
                "Institutional live-readiness checks failed.\n\n"
                f"{preview}{more}\n\n"
                "Run `python scripts/regulatory_readiness.py` for details."
            )
            if strict_live:
                QMessageBox.critical(
                    self,
                    "Live Readiness Failed",
                    msg,
                )
                self.mode_combo.setCurrentIndex(0)
                return
            reply = QMessageBox.warning(
                self,
                "Live Readiness Warning",
                msg + "\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self.mode_combo.setCurrentIndex(0)
                return

        reply = QMessageBox.warning(
            self,
            "Live Trading Warning",
            "You are switching to LIVE TRADING mode!\n\n"
            "This will use REAL MONEY.\n\n"
            "Are you absolutely sure?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self.mode_combo.setCurrentIndex(0)
            return

    try:
        execution_engine = _lazy_get("trading.executor", "ExecutionEngine")
        self.executor = execution_engine(mode)
        self.executor.on_fill = self._on_order_filled
        self.executor.on_reject = self._on_order_rejected

        if self.executor.start():
            self.connection_status.setText("Connected")
            self.connection_status.setStyleSheet(
                get_connection_status_style(True)
            )
            self.connect_btn.setText("Disconnect")
            self.connect_btn.setStyleSheet(
                get_connection_button_style(True)
            )

            self.log(
                f"Connected to {mode.value} trading",
                "success",
            )
            self._refresh_portfolio()

            # Initialize auto-trader after broker connection.
            self._init_auto_trader()
            if self._auto_trade_mode != AutoTradeMode.MANUAL:
                self._apply_auto_trade_mode(self._auto_trade_mode)
        else:
            self.executor = None
            self.log("Failed to connect to broker", "error")
    except Exception as e:
        self.log(f"Connection error: {e}", "error")
        self.executor = None


def _disconnect_trading(self: Any) -> None:
    """Disconnect from trading."""
    if self.executor:
        try:
            self.executor.stop()
        except Exception as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)
        self.executor = None

    self.connection_status.setText("Disconnected")
    self.connection_status.setStyleSheet(
        get_connection_status_style(False)
    )
    self.connect_btn.setText("Connect to Broker")
    self.connect_btn.setStyleSheet(
        get_connection_button_style(False)
    )

    self.log("Disconnected from broker", "info")


def _on_chart_trade_requested(self: Any, side: str, price: float) -> None:
    """Handle right-click chart quick trade request."""
    if self.executor is None:
        self.log("Connect broker before trading from chart", "warning")
        return
    symbol = _normalize_stock_code(self.stock_input.text())
    if not symbol and self.current_prediction is not None:
        symbol = _normalize_stock_code(
            getattr(self.current_prediction, "stock_code", "")
        )
    if not symbol:
        self.log("No active symbol for chart trade", "warning")
        return
    if price <= 0:
        self.log("Invalid chart price", "warning")
        return

    try:
        lot = max(1, int(get_lot_size(symbol)))
    except Exception:
        lot = 1

    order_params = self._show_chart_trade_dialog(
        symbol=symbol,
        side=side,
        clicked_price=float(price),
        lot=lot,
    )
    if not order_params:
        return

    order_side = OrderSide.BUY if str(side).lower() == "buy" else OrderSide.SELL
    self._submit_chart_order(
        symbol=symbol,
        side=order_side,
        qty=int(order_params["qty"]),
        price=float(order_params["price"]),
        order_type=str(order_params["order_type"]),
        time_in_force=str(order_params["time_in_force"]),
        trigger_price=float(order_params["trigger_price"]),
        trailing_stop_pct=float(order_params["trailing_stop_pct"]),
        trail_limit_offset_pct=float(order_params["trail_limit_offset_pct"]),
        strict_time_in_force=bool(order_params["strict_time_in_force"]),
        stop_loss=float(order_params["stop_loss"]),
        take_profit=float(order_params["take_profit"]),
        bracket=bool(order_params["bracket"]),
    )


def _show_chart_trade_dialog(
    self: Any,
    symbol: str,
    side: str,
    clicked_price: float,
    lot: int,
) -> dict[str, float | int | str | bool] | None:
    """Collect advanced chart trade parameters from user."""
    from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout

    dialog = QDialog(self)
    dialog.setWindowTitle("Chart Quick Trade")
    dialog.setMinimumWidth(420)

    layout = QVBoxLayout(dialog)
    heading = QLabel(f"{str(side).upper()} {symbol} | Chart Price: {clicked_price:.2f}")
    heading.setStyleSheet("font-weight: bold;")
    layout.addWidget(heading)

    form = QFormLayout()
    layout.addLayout(form)

    qty_spin = QSpinBox()
    qty_spin.setRange(max(1, lot), 5_000_000)
    qty_spin.setSingleStep(max(1, lot))
    qty_spin.setValue(max(1, lot))
    qty_spin.setSuffix(f" (lot {lot})")
    form.addRow("Quantity:", qty_spin)

    order_type_combo = QComboBox()
    order_types = [
        ("Limit", OrderType.LIMIT.value),
        ("Market", OrderType.MARKET.value),
        ("Stop", OrderType.STOP.value),
        ("Stop Limit", OrderType.STOP_LIMIT.value),
        ("IOC", OrderType.IOC.value),
        ("FOK", OrderType.FOK.value),
        ("Trailing Market", OrderType.TRAIL_MARKET.value),
        ("Trailing Limit", OrderType.TRAIL_LIMIT.value),
    ]
    for label, value in order_types:
        order_type_combo.addItem(label, value)
    form.addRow("Order Type:", order_type_combo)

    tif_combo = QComboBox()
    for label, value in (("DAY", "day"), ("GTC", "gtc"), ("IOC", "ioc"), ("FOK", "fok")):
        tif_combo.addItem(label, value)
    form.addRow("Time In Force:", tif_combo)

    strict_tif = QCheckBox("Strict TIF (cancel if unsupported)")
    form.addRow("", strict_tif)

    price_spin = QDoubleSpinBox()
    price_spin.setRange(0.01, 1_000_000.0)
    price_spin.setDecimals(3)
    price_spin.setValue(max(0.01, float(clicked_price)))
    price_spin.setSingleStep(max(0.01, float(clicked_price) * 0.002))
    form.addRow("Order Price:", price_spin)

    trigger_spin = QDoubleSpinBox()
    trigger_spin.setRange(0.0, 1_000_000.0)
    trigger_spin.setDecimals(3)
    trigger_spin.setValue(max(0.0, float(clicked_price)))
    trigger_spin.setSingleStep(max(0.01, float(clicked_price) * 0.002))
    form.addRow("Trigger Price:", trigger_spin)

    trailing_spin = QDoubleSpinBox()
    trailing_spin.setRange(0.0, 20.0)
    trailing_spin.setDecimals(2)
    trailing_spin.setSingleStep(0.1)
    trailing_spin.setSuffix(" %")
    trailing_spin.setValue(0.8)
    form.addRow("Trailing Stop:", trailing_spin)

    trail_limit_offset_spin = QDoubleSpinBox()
    trail_limit_offset_spin.setRange(0.0, 10.0)
    trail_limit_offset_spin.setDecimals(2)
    trail_limit_offset_spin.setSingleStep(0.05)
    trail_limit_offset_spin.setSuffix(" %")
    trail_limit_offset_spin.setValue(0.15)
    form.addRow("Trail Limit Offset:", trail_limit_offset_spin)

    bracket_check = QCheckBox("Attach stop-loss / take-profit")
    form.addRow("", bracket_check)

    stop_loss_spin = QDoubleSpinBox()
    stop_loss_spin.setRange(0.0, 1_000_000.0)
    stop_loss_spin.setDecimals(3)
    stop_loss_spin.setValue(0.0)
    form.addRow("Stop-Loss:", stop_loss_spin)

    take_profit_spin = QDoubleSpinBox()
    take_profit_spin.setRange(0.0, 1_000_000.0)
    take_profit_spin.setDecimals(3)
    take_profit_spin.setValue(0.0)
    form.addRow("Take-Profit:", take_profit_spin)

    def _sync_widgets() -> None:
        ot = str(order_type_combo.currentData() or "limit")
        is_market_like = ot in {
            OrderType.MARKET.value,
            OrderType.IOC.value,
            OrderType.FOK.value,
            OrderType.TRAIL_MARKET.value,
        }
        needs_trigger = ot in {
            OrderType.STOP.value,
            OrderType.STOP_LIMIT.value,
            OrderType.TRAIL_MARKET.value,
            OrderType.TRAIL_LIMIT.value,
        }
        needs_trailing = ot in {
            OrderType.TRAIL_MARKET.value,
            OrderType.TRAIL_LIMIT.value,
        }
        needs_trail_limit_offset = ot == OrderType.TRAIL_LIMIT.value

        price_spin.setEnabled(not is_market_like or ot == OrderType.TRAIL_LIMIT.value)
        trigger_spin.setEnabled(needs_trigger)
        trailing_spin.setEnabled(needs_trailing)
        trail_limit_offset_spin.setEnabled(needs_trail_limit_offset)

        if ot in (OrderType.IOC.value, OrderType.FOK.value):
            forced = "ioc" if ot == OrderType.IOC.value else "fok"
            idx = tif_combo.findData(forced)
            if idx >= 0:
                tif_combo.setCurrentIndex(idx)

    order_type_combo.currentIndexChanged.connect(_sync_widgets)
    _sync_widgets()

    btns = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    btns.accepted.connect(dialog.accept)
    btns.rejected.connect(dialog.reject)
    layout.addWidget(btns)

    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None

    return {
        "qty": int(qty_spin.value()),
        "price": float(price_spin.value()),
        "order_type": str(order_type_combo.currentData() or "limit"),
        "time_in_force": str(tif_combo.currentData() or "day"),
        "trigger_price": float(trigger_spin.value()),
        # Percent units (e.g., 0.8 means 0.8%).
        "trailing_stop_pct": float(trailing_spin.value()),
        "trail_limit_offset_pct": float(trail_limit_offset_spin.value()),
        "strict_time_in_force": bool(strict_tif.isChecked()),
        "bracket": bool(bracket_check.isChecked()),
        "stop_loss": float(stop_loss_spin.value()),
        "take_profit": float(take_profit_spin.value()),
    }


def _submit_chart_order(
    self: Any,
    symbol: str,
    side: OrderSide,
    qty: int,
    price: float,
    order_type: str = "limit",
    time_in_force: str = "day",
    trigger_price: float = 0.0,
    trailing_stop_pct: float = 0.0,
    trail_limit_offset_pct: float = 0.0,
    strict_time_in_force: bool = False,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    bracket: bool = False,
) -> None:
    if self.executor is None:
        return
    try:
        lot = max(1, int(get_lot_size(symbol)))
    except Exception:
        lot = 1

    requested_qty = max(1, int(qty))
    normalized_qty = max(lot, requested_qty)
    if normalized_qty % lot != 0:
        normalized_qty = (normalized_qty // lot) * lot
        if normalized_qty <= 0:
            normalized_qty = lot
    if normalized_qty != requested_qty:
        self.log(
            f"Adjusted quantity {requested_qty} -> {normalized_qty} (lot {lot})",
            "info",
        )

    normalized_order_type = str(order_type or "limit").strip().lower().replace("-", "_")
    valid_order_types = {
        OrderType.LIMIT.value,
        OrderType.MARKET.value,
        OrderType.STOP.value,
        OrderType.STOP_LIMIT.value,
        OrderType.IOC.value,
        OrderType.FOK.value,
        OrderType.TRAIL_MARKET.value,
        OrderType.TRAIL_LIMIT.value,
    }
    if normalized_order_type not in valid_order_types:
        normalized_order_type = OrderType.LIMIT.value

    normalized_tif = str(time_in_force or "day").strip().lower()
    if normalized_tif not in {"day", "gtc", "ioc", "fok"}:
        normalized_tif = "day"
    if normalized_order_type in {OrderType.IOC.value, OrderType.FOK.value}:
        normalized_tif = normalized_order_type

    normalized_price = max(0.0, float(price))
    if normalized_order_type in {
        OrderType.MARKET.value,
        OrderType.IOC.value,
        OrderType.FOK.value,
        OrderType.TRAIL_MARKET.value,
    } and normalized_price <= 0:
        normalized_price = 0.01

    normalized_trigger = max(0.0, float(trigger_price))
    if normalized_order_type in {
        OrderType.STOP.value,
        OrderType.STOP_LIMIT.value,
        OrderType.TRAIL_MARKET.value,
        OrderType.TRAIL_LIMIT.value,
    } and normalized_trigger <= 0:
        normalized_trigger = normalized_price
    if normalized_order_type not in {
        OrderType.STOP.value,
        OrderType.STOP_LIMIT.value,
        OrderType.TRAIL_MARKET.value,
        OrderType.TRAIL_LIMIT.value,
    }:
        normalized_trigger = 0.0

    normalized_trailing_stop = max(0.0, float(trailing_stop_pct))
    # Backward compatibility: older UI path sent fractional units (0.008).
    if 0.0 < normalized_trailing_stop < 0.05:
        normalized_trailing_stop *= 100.0
    normalized_trailing_stop = min(20.0, normalized_trailing_stop)
    if normalized_order_type not in {
        OrderType.TRAIL_MARKET.value,
        OrderType.TRAIL_LIMIT.value,
    }:
        normalized_trailing_stop = 0.0

    normalized_trail_limit_offset = max(0.0, float(trail_limit_offset_pct))
    if 0.0 < normalized_trail_limit_offset < 0.05:
        normalized_trail_limit_offset *= 100.0
    normalized_trail_limit_offset = min(10.0, normalized_trail_limit_offset)
    if normalized_order_type != OrderType.TRAIL_LIMIT.value:
        normalized_trail_limit_offset = 0.0

    normalized_stop_loss = max(0.0, float(stop_loss))
    normalized_take_profit = max(0.0, float(take_profit))
    use_bracket = bool(bracket) and (normalized_stop_loss > 0 or normalized_take_profit > 0)

    signal = TradeSignal(
        symbol=symbol,
        side=side,
        quantity=normalized_qty,
        price=normalized_price,
        strategy="chart_manual",
        reasons=[
            "Manual chart quick-trade",
            f"order_type={normalized_order_type}",
            f"tif={normalized_tif}",
        ],
        confidence=1.0,
        order_type=normalized_order_type,
        time_in_force=normalized_tif,
        trigger_price=normalized_trigger,
        trailing_stop_pct=normalized_trailing_stop,
        trail_limit_offset_pct=normalized_trail_limit_offset,
        stop_loss=normalized_stop_loss if use_bracket else 0.0,
        take_profit=normalized_take_profit if use_bracket else 0.0,
        bracket=use_bracket,
    )
    signal.strict_time_in_force = bool(strict_time_in_force)
    try:
        ok = self.executor.submit(signal)
        if ok:
            price_text = f"{normalized_price:.2f}" if normalized_price > 0 else "MKT"
            self.log(
                "Chart trade submitted: "
                f"{side.value.upper()} {normalized_qty} {symbol} "
                f"@ {price_text} ({normalized_order_type}, {normalized_tif})",
                "success",
            )
        else:
            self.log("Chart trade rejected by risk/permissions", "warning")
    except Exception as e:
        self.log(f"Chart trade failed: {e}", "error")


def _execute_buy(self: Any) -> None:
    """Execute buy order."""
    if not self.current_prediction or not self.executor:
        return

    pred = self.current_prediction
    levels = getattr(pred, "levels", None)
    position = getattr(pred, "position", None)

    if not levels or not position:
        self.log("Missing trading levels or position info", "error")
        return

    shares = getattr(position, "shares", 0)
    entry = getattr(levels, "entry", 0)
    value = getattr(position, "value", 0)
    stop_loss = getattr(levels, "stop_loss", 0)
    target_2 = getattr(levels, "target_2", 0)
    stock_name = getattr(pred, "stock_name", "")

    try:
        if not CONFIG.is_market_open():
            QMessageBox.warning(
                self, "Market Closed", "Market is currently closed. Live orders are blocked."
            )
            return
    except Exception as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    try:
        ok, msg, fresh_px = self.executor.check_quote_freshness(pred.stock_code)
        if not ok:
            QMessageBox.warning(self, "Stale Quote", f"Order blocked: {msg}")
            return
        if fresh_px > 0:
            entry = float(fresh_px)
    except Exception as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    reply = QMessageBox.question(
        self,
        "Confirm Buy Order",
        f"<b>Buy {pred.stock_code} - {stock_name}</b><br><br>"
        f"Quantity: {shares:,} shares<br>"
        f"Price: CNY {entry:.2f}<br>"
        f"Value: CNY {value:,.2f}<br>"
        f"Stop Loss: CNY {stop_loss:.2f}<br>"
        f"Target: CNY {target_2:.2f}",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )

    if reply == QMessageBox.StandardButton.Yes:
        try:
            if hasattr(self.executor, "submit_from_prediction"):
                success = self.executor.submit_from_prediction(pred)
            else:
                success = False

            if success:
                self.log(f"Buy order submitted: {pred.stock_code}", "info")
            else:
                self.log("Buy order failed risk checks", "error")
        except Exception as e:
            self.log(f"Buy order error: {e}", "error")


def _execute_sell(self: Any) -> None:
    """Execute sell order."""
    if not self.current_prediction or not self.executor:
        return

    pred = self.current_prediction

    try:
        if not CONFIG.is_market_open():
            QMessageBox.warning(
                self, "Market Closed", "Market is currently closed. Live orders are blocked."
            )
            return
    except Exception as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    try:
        ok, msg, fresh_px = self.executor.check_quote_freshness(pred.stock_code)
        if not ok:
            QMessageBox.warning(self, "Stale Quote", f"Order blocked: {msg}")
            return
    except Exception:
        fresh_px = 0.0

    try:
        positions = self.executor.get_positions()
        position = positions.get(pred.stock_code)

        if not position:
            self.log("No position to sell", "warning")
            return

        available_qty = getattr(position, "available_qty", 0)
        current_price = getattr(position, "current_price", 0) or fresh_px
        stock_name = getattr(pred, "stock_name", "")

        reply = QMessageBox.question(
            self,
            "Confirm Sell Order",
            f"<b>Sell {pred.stock_code} - {stock_name}</b><br><br>"
            f"Available: {available_qty:,} shares<br>"
            f"Current Price: CNY {current_price:.2f}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            signal = TradeSignal(
                symbol=pred.stock_code,
                name=stock_name,
                side=OrderSide.SELL,
                quantity=available_qty,
                price=current_price,
            )

            success = self.executor.submit(signal)
            if success:
                self.log(f"Sell order submitted: {pred.stock_code}", "info")
            else:
                self.log("Sell order failed", "error")
    except Exception as e:
        self.log(f"Sell order error: {e}", "error")


def _on_order_filled(self: Any, order: Any, fill: Any) -> None:
    """Handle order fill."""
    side = order.side.value.upper() if hasattr(order.side, "value") else str(order.side)
    qty = getattr(fill, "quantity", 0)
    price = getattr(fill, "price", 0)

    self.log(
        f"Order filled: {side} {qty} {order.symbol} @ CNY {price:.2f}",
        "success",
    )
    self._refresh_portfolio()


def _on_order_rejected(self: Any, order: Any, reason: Any) -> None:
    """Handle order rejection."""
    self.log(f"Order rejected: {order.symbol} - {reason}", "error")


def _refresh_portfolio(self: Any) -> None:
    """Refresh portfolio display with visible error handling."""
    if not self.executor:
        return

    try:
        account = self.executor.get_account()

        equity = getattr(account, "equity", 0)
        available = getattr(account, "available", 0)
        market_value = getattr(account, "market_value", 0)
        total_pnl = getattr(account, "total_pnl", 0)
        positions = getattr(account, "positions", {})

        self.account_labels["equity"].setText(f"CNY {equity:,.2f}")
        self.account_labels["cash"].setText(f"CNY {available:,.2f}")
        self.account_labels["positions"].setText(f"CNY {market_value:,.2f}")

        pnl_color = (
            ModernColors.ACCENT_SUCCESS
            if total_pnl >= 0
            else ModernColors.ACCENT_DANGER
        )
        self.account_labels["pnl"].setText(f"CNY {total_pnl:,.2f}")
        self.account_labels["pnl"].setStyleSheet(
            f"color: {pnl_color}; font-size: 18px; font-weight: bold;"
        )

        if hasattr(self.positions_table, "update_positions"):
            self.positions_table.update_positions(positions)

    except Exception as e:
        # Keep this visible instead of silent to aid live troubleshooting.
        log.warning(f"Portfolio refresh error: {e}")
        self.log(f"Portfolio refresh failed: {e}", "warning")
