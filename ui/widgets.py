# ui/widgets.py
from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.modern_theme import (
    ModernColors,
    ModernFonts,
    ModernSpacing,
    get_display_font_family,
    get_monospace_font_family,
    get_primary_font_family,
    get_progress_bar_style,
)
from utils.logger import get_logger
from utils.type_utils import (
    safe_float_attr,
    safe_int_attr,
    safe_str_attr,
)

log = get_logger(__name__)


def _get_signal_enum():
    """Lazy import Signal to avoid circular imports at module load time."""
    try:
        from models.predictor import Signal

        return Signal
    except ImportError:
        return None


class SignalPanel(QFrame):
    """Large signal display panel with modern professional styling."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("signalPanelFrame")
        self.setMinimumHeight(160)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(ModernSpacing.BASE)
        layout.setContentsMargins(
            ModernSpacing.LG,
            ModernSpacing.LG,
            ModernSpacing.LG,
            ModernSpacing.LG,
        )

        self.signal_label = QLabel("WAITING")
        self.signal_label.setObjectName("signalLabel")
        self.signal_label.setFont(
            QFont(
                get_display_font_family(),
                ModernFonts.SIZE_XXL,
                QFont.Weight.Bold,
            )
        )
        self.signal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.signal_label)

        self.info_label = QLabel("Enter a stock code to analyze")
        self.info_label.setFont(
            QFont(get_primary_font_family(), ModernFonts.SIZE_SM)
        )
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet(
            f"color: {ModernColors.TEXT_SECONDARY};"
        )
        layout.addWidget(self.info_label)

        prob_widget = QWidget()
        prob_layout = QHBoxLayout(prob_widget)
        prob_layout.setSpacing(ModernSpacing.LG)
        prob_layout.setContentsMargins(
            0,
            ModernSpacing.BASE,
            0,
            ModernSpacing.BASE,
        )

        down_container = QVBoxLayout()
        down_container.setSpacing(ModernSpacing.XS)
        down_label = QLabel("DOWN")
        down_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        down_label.setStyleSheet(
            
                f"color: {ModernColors.ACCENT_DANGER}; "
                f"font-weight: {ModernFonts.WEIGHT_SEMIBOLD}; "
                f"font-size: {ModernFonts.SIZE_SM}px;"
            
        )
        self.prob_down = QProgressBar()
        self.prob_down.setFormat("%p%")
        self.prob_down.setFixedHeight(14)
        self.prob_down.setStyleSheet(get_progress_bar_style("danger"))
        down_container.addWidget(down_label)
        down_container.addWidget(self.prob_down)
        prob_layout.addLayout(down_container)

        neutral_container = QVBoxLayout()
        neutral_container.setSpacing(ModernSpacing.XS)
        neutral_label = QLabel("NEUTRAL")
        neutral_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        neutral_label.setStyleSheet(
            
                f"color: {ModernColors.ACCENT_WARNING}; "
                f"font-weight: {ModernFonts.WEIGHT_SEMIBOLD}; "
                f"font-size: {ModernFonts.SIZE_SM}px;"
            
        )
        self.prob_neutral = QProgressBar()
        self.prob_neutral.setFormat("%p%")
        self.prob_neutral.setFixedHeight(14)
        self.prob_neutral.setStyleSheet(get_progress_bar_style("warning"))
        neutral_container.addWidget(neutral_label)
        neutral_container.addWidget(self.prob_neutral)
        prob_layout.addLayout(neutral_container)

        up_container = QVBoxLayout()
        up_container.setSpacing(ModernSpacing.XS)
        up_label = QLabel("UP")
        up_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        up_label.setStyleSheet(
            
                f"color: {ModernColors.ACCENT_SUCCESS}; "
                f"font-weight: {ModernFonts.WEIGHT_SEMIBOLD}; "
                f"font-size: {ModernFonts.SIZE_SM}px;"
            
        )
        self.prob_up = QProgressBar()
        self.prob_up.setFormat("%p%")
        self.prob_up.setFixedHeight(14)
        self.prob_up.setStyleSheet(get_progress_bar_style("success"))
        up_container.addWidget(up_label)
        up_container.addWidget(self.prob_up)
        prob_layout.addLayout(up_container)

        layout.addWidget(prob_widget)

        self.action_label = QLabel("")
        self.action_label.setFont(
            QFont(get_primary_font_family(), ModernFonts.SIZE_SM)
        )
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_label.setWordWrap(True)
        self.action_label.setStyleSheet(
            f"color: {ModernColors.TEXT_SECONDARY}; padding: 4px 0;"
        )
        layout.addWidget(self.action_label)

        self.conf_label = QLabel("")
        self.conf_label.setFont(
            QFont(get_primary_font_family(), ModernFonts.SIZE_XS)
        )
        self.conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.conf_label.setStyleSheet(
            f"color: {ModernColors.TEXT_MUTED};"
        )
        layout.addWidget(self.conf_label)

        self._set_default_style()

    def _set_default_style(self) -> None:
        self.setStyleSheet(
            f"""
            QFrame#signalPanelFrame {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {ModernColors.BG_SECONDARY},
                    stop:1 {ModernColors.BG_PRIMARY}
                );
                border-radius: 12px;
                border: 1px solid {ModernColors.BORDER_DEFAULT};
            }}
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
            }}
            """
        )

    def update_prediction(self, pred) -> None:
        """Update display with prediction data (robust to missing fields)."""
        Signal = _get_signal_enum()

        sig = getattr(pred, "signal", None)
        sig_text = (
            sig.value if hasattr(sig, "value") else (str(sig) if sig else "HOLD")
        )

        code = safe_str_attr(pred, "stock_code")
        name = safe_str_attr(pred, "stock_name")
        price = safe_float_attr(pred, "current_price")
        warnings = [
            str(x).strip()
            for x in list(getattr(pred, "warnings", []) or [])
            if str(x).strip()
        ]
        forecast_vals = list(getattr(pred, "predicted_prices", []) or [])
        model_missing = any(
            "no trained model artifacts loaded" in w.lower()
            for w in warnings
        )
        data_not_ready = any(
            ("insufficient data" in w.lower()) or ("prediction error" in w.lower())
            for w in warnings
        )
        prediction_unavailable = bool(
            (model_missing and not forecast_vals)
            or (data_not_ready and not forecast_vals)
        )

        if model_missing and prediction_unavailable:
            self.signal_label.setText("NO_MODEL")
        elif data_not_ready and prediction_unavailable:
            self.signal_label.setText("WARMING_UP")
        else:
            self.signal_label.setText(sig_text)
        self.info_label.setText(f"{code} - {name} | CNY {price:.2f}")

        # Compute probabilities from model output; fallback to signal-derived
        # distribution only when explicit probabilities are missing.
        confidence = safe_float_attr(pred, "confidence", 0.5)

        def _opt_prob(attr: str) -> float | None:
            raw = getattr(pred, attr, None)
            if raw is None:
                return None
            try:
                val = float(raw)
            except (TypeError, ValueError, OverflowError):
                return None
            if not (val >= 0.0):
                return None
            return val

        # Try explicit model probabilities first.
        prob_down = _opt_prob("prob_down")
        prob_neutral = _opt_prob("prob_neutral")
        prob_up = _opt_prob("prob_up")

        # If probabilities not provided, derive from signal and confidence
        if prob_down is None or prob_neutral is None or prob_up is None:
            if Signal is not None and sig is not None:
                if sig in (Signal.STRONG_BUY, Signal.BUY):
                    # Bullish: high up probability
                    base_up = 0.70 if sig == Signal.STRONG_BUY else 0.55
                    prob_up = min(0.95, base_up + confidence * 0.25)
                    prob_down = max(0.05, (1.0 - prob_up) * 0.3)
                    prob_neutral = 1.0 - prob_up - prob_down
                elif sig in (Signal.STRONG_SELL, Signal.SELL):
                    # Bearish: high down probability
                    base_down = 0.70 if sig == Signal.STRONG_SELL else 0.55
                    prob_down = min(0.95, base_down + confidence * 0.25)
                    prob_up = max(0.05, (1.0 - prob_down) * 0.3)
                    prob_neutral = 1.0 - prob_down - prob_up
                else:
                    # HOLD: neutral dominant
                    prob_neutral = 0.60 + confidence * 0.30
                    prob_up = (1.0 - prob_neutral) * 0.5
                    prob_down = 1.0 - prob_neutral - prob_up
            else:
                # Default uniform distribution
                prob_down, prob_neutral, prob_up = 0.33, 0.34, 0.33

        if prediction_unavailable:
            prob_down, prob_neutral, prob_up = 0.0, 0.0, 0.0

        # Normalize to ensure they sum to 1.0 (100%)
        total = prob_down + prob_neutral + prob_up
        if total > 0 and abs(total - 1.0) > 1e-6:
            prob_down /= total
            prob_neutral /= total
            prob_up /= total

        # Ensure all probabilities are in valid range [0, 1]
        prob_down = max(0.0, min(1.0, prob_down))
        prob_neutral = max(0.0, min(1.0, prob_neutral))
        prob_up = max(0.0, min(1.0, prob_up))

        # Final normalization after clamping
        total = prob_down + prob_neutral + prob_up
        if total > 0:
            prob_down /= total
            prob_neutral /= total
            prob_up /= total

        # UI stability guard: avoid displaying hard 0/100/0 for HOLD when
        # probabilities are numerically saturated near neutral.
        sig_is_hold = bool(
            (Signal is not None and sig == Signal.HOLD)
            or str(sig_text).strip().upper() == "HOLD"
        )
        if (
            not prediction_unavailable
            and sig_is_hold
            and prob_neutral >= 0.985
            and (prob_up + prob_down) <= 0.02
        ):
            neutral_mass = 0.90
            side_mass = 1.0 - neutral_mass
            side_total = prob_up + prob_down
            if side_total > 1e-8:
                up_ratio = prob_up / side_total
            else:
                up_ratio = 0.5
            prob_up = side_mass * up_ratio
            prob_down = side_mass - prob_up
            prob_neutral = neutral_mass

        self.prob_down.setValue(int(round(prob_down * 100)))
        self.prob_neutral.setValue(int(round(prob_neutral * 100)))
        self.prob_up.setValue(int(round(prob_up * 100)))
        self.prob_down.setFormat(f"{prob_down * 100.0:.1f}%")
        self.prob_neutral.setFormat(f"{prob_neutral * 100.0:.1f}%")
        self.prob_up.setFormat(f"{prob_up * 100.0:.1f}%")

        pos = getattr(pred, "position", None)
        levels = getattr(pred, "levels", None)
        shares = safe_int_attr(pos, "shares") if pos else 0
        entry = safe_float_attr(levels, "entry") if levels else 0.0
        stop = safe_float_attr(levels, "stop_loss") if levels else 0.0
        tgt2 = safe_float_attr(levels, "target_2") if levels else 0.0

        if model_missing and prediction_unavailable:
            self.action_label.setText("Model unavailable - train model to enable guessing")
        elif data_not_ready and prediction_unavailable:
            self.action_label.setText("Data warming up - waiting for enough valid candles")
        elif Signal is not None and shares > 0:
            if sig in (Signal.STRONG_BUY, Signal.BUY):
                self.action_label.setText(
                    f"BUY {shares:,} shares @ CNY {entry:.2f}\n"
                    f"Stop Loss: CNY {stop:.2f} | Target: CNY {tgt2:.2f}"
                )
            elif sig in (Signal.STRONG_SELL, Signal.SELL):
                self.action_label.setText(f"SELL {shares:,} shares @ CNY {entry:.2f}")
            else:
                self.action_label.setText("HOLD - Wait for clearer signal")
        else:
            self.action_label.setText("HOLD - Wait for clearer signal")

        confidence = safe_float_attr(pred, "confidence")
        agreement = safe_float_attr(
            pred,
            "model_agreement",
            safe_float_attr(pred, "agreement", 1.0),
        )
        strength = safe_float_attr(pred, "signal_strength")
        display_strength = strength
        if sig_is_hold:
            # HOLD should emphasize directional edge, not confidence.
            display_strength = max(0.0, min(1.0, abs(prob_up - prob_down)))
        uncertainty = safe_float_attr(pred, "uncertainty_score", 0.5)
        tail_risk = safe_float_attr(pred, "tail_risk_score", 0.5)

        if model_missing and prediction_unavailable:
            self.conf_label.setText("Confidence: N/A | Guessing disabled until model is trained")
        elif data_not_ready and prediction_unavailable:
            self.conf_label.setText("Confidence: N/A | Waiting for enough valid bars")
        else:
            self.conf_label.setText(
                f"Confidence: {confidence:.0%} | "
                f"Model Agreement: {agreement:.0%} | "
                f"Signal Strength: {display_strength:.0%} | "
                f"Uncertainty: {uncertainty:.2f} | "
                f"Tail Risk: {tail_risk:.2f}"
            )

        if Signal is not None:
            colors = {
                Signal.STRONG_BUY: (ModernColors.ACCENT_SUCCESS, "#0d1e32"),
                Signal.BUY: (ModernColors.ACCENT_SUCCESS, "#0d1e32"),
                Signal.HOLD: (ModernColors.ACCENT_WARNING, "#0d1e32"),
                Signal.SELL: (ModernColors.ACCENT_DANGER, "#0d1e32"),
                Signal.STRONG_SELL: (ModernColors.ACCENT_DANGER, "#0d1e32"),
            }
            fg, bg = colors.get(sig, (ModernColors.TEXT_PRIMARY, "#13223a"))
        else:
            fg, bg = ModernColors.TEXT_PRIMARY, "#13223a"

        self.setStyleSheet(
            f"""
            QFrame#signalPanelFrame {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {fg}22, stop:1 {bg}
                );
                border-radius: 12px;
                border: 1px solid {fg};
            }}
            QLabel {{
                color: {fg};
            }}
            """
        )

    def reset(self) -> None:
        """Reset to default state."""
        self.signal_label.setText("WAITING")
        self.info_label.setText("Enter a stock code to analyze")
        self.action_label.setText("")
        self.conf_label.setText("")
        self.prob_down.setValue(0)
        self.prob_neutral.setValue(0)
        self.prob_up.setValue(0)
        self._set_default_style()


class PositionTable(QTableWidget):
    """Position display table with professional styling."""

    def __init__(self) -> None:
        super().__init__()
        self.setColumnCount(8)
        self.setHorizontalHeaderLabels(
            [
                "Code",
                "Name",
                "Shares",
                "Available",
                "Cost",
                "Price",
                "P&L",
                "P&L %",
            ]
        )
        self.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.verticalHeader().setVisible(False)

    @staticmethod
    def _safe_attr(obj, attr, default=0):
        """Safely get attribute with type coercion."""
        try:
            val = getattr(obj, attr, None)
            if val is None:
                return default
            if isinstance(default, float):
                return float(val)
            if isinstance(default, int):
                return int(val)
            return val
        except (TypeError, ValueError):
            return default

    def update_positions(self, positions: dict) -> None:
        """Update table with position data - handles missing attributes."""
        if positions is None:
            positions = {}

        self.setRowCount(len(positions))

        for row, (code, pos) in enumerate(positions.items()):
            self.setItem(row, 0, QTableWidgetItem(str(code)))

            name = self._safe_attr(pos, "name", "")
            self.setItem(row, 1, QTableWidgetItem(str(name)))

            quantity = self._safe_attr(pos, "quantity", 0)
            shares_item = QTableWidgetItem(f"{quantity:,}")
            shares_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.setItem(row, 2, shares_item)

            available = self._safe_attr(pos, "available_qty", 0)
            avail_item = QTableWidgetItem(f"{available:,}")
            avail_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.setItem(row, 3, avail_item)

            avg_cost = self._safe_attr(pos, "avg_cost", 0.0)
            cost_item = QTableWidgetItem(f"CNY {avg_cost:.2f}")
            cost_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.setItem(row, 4, cost_item)

            current_price = self._safe_attr(pos, "current_price", 0.0)
            price_item = QTableWidgetItem(f"CNY {current_price:.2f}")
            price_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.setItem(row, 5, price_item)

            unrealized_pnl = self._safe_attr(pos, "unrealized_pnl", 0.0)
            pnl_item = QTableWidgetItem(f"CNY {unrealized_pnl:+,.2f}")
            pnl_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )

            unrealized_pnl_pct = self._safe_attr(pos, "unrealized_pnl_pct", 0.0)
            pnl_pct_item = QTableWidgetItem(f"{unrealized_pnl_pct:+.2f}%")
            pnl_pct_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )

            if unrealized_pnl >= 0:
                color = QColor(ModernColors.ACCENT_SUCCESS)
            else:
                color = QColor(ModernColors.ACCENT_DANGER)

            pnl_item.setForeground(color)
            pnl_pct_item.setForeground(color)

            self.setItem(row, 6, pnl_item)
            self.setItem(row, 7, pnl_pct_item)


class LogWidget(QTextEdit):
    """System log display with professional styling and bounded history."""

    MAX_LINES = 500
    _TRIM_BATCH = 100

    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        mono_font = get_monospace_font_family()
        self.setFont(QFont(mono_font, ModernFonts.SIZE_SM))
        self.setMinimumHeight(220)
        self.setMaximumHeight(380)
        self.setStyleSheet(
            f"""
            QTextEdit {{
                background: {ModernColors.BG_PRIMARY};
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 10px;
                padding: 8px;
                selection-background-color: #2f689d;
                selection-color: {ModernColors.TEXT_STRONG};
                font-family: '{mono_font}', monospace;
            }}
            """
        )
        self._line_count = 0

    def log(self, message: str, level: str = "info") -> None:
        """Add log message with color coding."""
        colors = {
            "info": ModernColors.ACCENT_SECONDARY,
            "warning": ModernColors.ACCENT_WARNING,
            "error": ModernColors.ACCENT_DANGER,
            "success": ModernColors.ACCENT_SUCCESS,
            "debug": ModernColors.ACCENT_INFO,
        }
        icons = {
            "info": "[I]",
            "warning": "[!]",
            "error": "[X]",
            "success": "[OK]",
            "debug": "[D]",
        }

        color = colors.get(level, ModernColors.TEXT_PRIMARY)
        icon = icons.get(level, "[.]")
        ts = datetime.now().strftime("%H:%M:%S")

        self.append(
            f'<span style="color:{ModernColors.TEXT_MUTED}">[{ts}]</span> '
            f'<span style="color:{color}">{icon} {message}</span>'
        )

        self._line_count += 1
        if self._line_count > self.MAX_LINES:
            self._trim_old_lines()

        scrollbar = self.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())

    def _trim_old_lines(self) -> None:
        """Remove oldest lines to keep log bounded.
        Uses document-level block removal instead of fragile cursor manipulation.
        FIX: Improved trim logic to prevent text corruption.
        """
        try:
            doc = self.document()
            if doc is None:
                return

            block_count = doc.blockCount()
            if block_count <= self.MAX_LINES:
                return

            remove_count = min(self._TRIM_BATCH, block_count - self.MAX_LINES)

            # FIX: Use simpler and more reliable trim approach
            # Get all text, split into lines, remove old ones, restore
            cursor = self.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(
                cursor.MoveOperation.End,
                cursor.MoveMode.KeepAnchor,
            )
            all_text = cursor.selectedText()
            
            # Split by line separator (Qt uses \u2029 paragraph separator)
            lines = all_text.split('\u2029')
            
            # Keep only the last MAX_LINES
            lines_to_keep = lines[remove_count:]
            
            # Restore text
            self.setPlainText('\n'.join(lines_to_keep))
            
            # Scroll to bottom
            scrollbar = self.verticalScrollBar()
            if scrollbar:
                scrollbar.setValue(scrollbar.maximum())
            
            self._line_count = len(lines_to_keep)
        except Exception as e:
            try:
                log.debug(f"Log trim failed, clearing: {e}")
                # FIX: Instead of clearing, just truncate to last MAX_LINES
                doc = self.document()
                if doc:
                    cursor = self.textCursor()
                    cursor.movePosition(cursor.MoveOperation.Start)
                    cursor.movePosition(
                        cursor.MoveOperation.End,
                        cursor.MoveMode.KeepAnchor,
                    )
                    all_text = cursor.selectedText()
                    lines = all_text.split('\u2029')
                    lines_to_keep = lines[-self.MAX_LINES:]
                    self.setPlainText('\n'.join(lines_to_keep))
                    self._line_count = len(lines_to_keep)
            except Exception:
                pass

    def clear_log(self) -> None:
        """Clear all log messages."""
        self.clear()
        self._line_count = 0


class MetricCard(QFrame):
    """Metric display card for dashboard."""

    def __init__(self, title: str, value: str = "--", icon: str = "") -> None:
        super().__init__()
        self.value_label = None
        self._setup_ui(title, value, icon)

    def _setup_ui(self, title: str, value: str, icon: str) -> None:
        self.setObjectName("metricCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

        title_text = f"{icon} {title}" if icon else title
        title_label = QLabel(title_text)
        title_label.setStyleSheet(
            
                f"color: {ModernColors.TEXT_SECONDARY}; "
                f"font-size: {ModernFonts.SIZE_SM}px; "
                f"font-weight: {ModernFonts.WEIGHT_SEMIBOLD};"
            
        )
        layout.addWidget(title_label)

        self.value_label = QLabel(str(value))
        self.value_label.setStyleSheet(
            f"""
            color: {ModernColors.ACCENT_PRIMARY};
            font-size: 26px;
            font-weight: 700;
            """
        )
        layout.addWidget(self.value_label)

    def set_value(self, value: str, color: str = None) -> None:
        """Update the displayed value."""
        if self.value_label is None:
            return

        self.value_label.setText(str(value))
        if color:
            self.value_label.setStyleSheet(
                f"""
                color: {color};
                font-size: 26px;
                font-weight: 700;
                """
            )


class TradingStatusBar(QFrame):
    """Trading status bar showing connection and market status."""

    def __init__(self) -> None:
        super().__init__()
        self.connection_label = None
        self.market_label = None
        self.mode_label = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setObjectName("statusCard")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(20)

        self.connection_label = QLabel("Disconnected")
        self.connection_label.setStyleSheet(
            f"color: {ModernColors.ACCENT_DANGER}; font-weight: 700; font-size: 12px;"
        )
        layout.addWidget(self.connection_label)

        layout.addStretch()

        self.market_label = QLabel("Market: --")
        self.market_label.setStyleSheet(
            f"color: {ModernColors.TEXT_SECONDARY}; font-size: 12px;"
        )
        layout.addWidget(self.market_label)

        layout.addStretch()

        self.mode_label = QLabel("Mode: Paper Trading")
        self.mode_label.setStyleSheet(
            f"color: {ModernColors.ACCENT_WARNING}; font-size: 12px; font-weight: 600;"
        )
        layout.addWidget(self.mode_label)

    def set_connected(self, connected: bool, mode: str = "paper") -> None:
        """Update connection status."""
        if self.connection_label is None:
            return

        if connected:
            self.connection_label.setText("Connected")
            self.connection_label.setStyleSheet(
                f"color: {ModernColors.ACCENT_SUCCESS}; font-weight: 700; font-size: 12px;"
            )

            if mode == "live":
                self.mode_label.setText("Mode: LIVE TRADING")
                self.mode_label.setStyleSheet(
                    f"color: {ModernColors.ACCENT_DANGER}; font-size: 12px; font-weight: 700;"
                )
            else:
                self.mode_label.setText("Mode: Paper Trading")
                self.mode_label.setStyleSheet(
                    f"color: {ModernColors.ACCENT_WARNING}; font-size: 12px; font-weight: 600;"
                )
        else:
            self.connection_label.setText("Disconnected")
            self.connection_label.setStyleSheet(
                f"color: {ModernColors.ACCENT_DANGER}; font-weight: 700; font-size: 12px;"
            )

    def set_market_status(self, is_open: bool) -> None:
        """Update market status."""
        if self.market_label is None:
            return

        if is_open:
            self.market_label.setText("Market: Open")
            self.market_label.setStyleSheet(
                f"color: {ModernColors.ACCENT_SUCCESS}; font-size: 12px; font-weight: 600;"
            )
        else:
            self.market_label.setText("Market: Closed")
            self.market_label.setStyleSheet(
                f"color: {ModernColors.ACCENT_DANGER}; font-size: 12px; font-weight: 600;"
            )
