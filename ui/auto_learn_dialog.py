# ui/auto_learn_dialog.py

import time
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.auto_learn_workers import (
    AutoLearnWorker,
    StockValidatorWorker,
    TargetedLearnWorker,
    _get_auto_learner,
)
from ui.modern_theme import (
    ModernColors,
    ModernFonts,
    get_dialog_style,
    get_monospace_font_family,
)
from utils.logger import get_logger

log = get_logger(__name__)

class AutoLearnDialog(QDialog):
    """Dialog for automatic learning."""
    session_finished = pyqtSignal(dict)

    def __init__(self, parent=None, seed_stock_codes: list[str] | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto Train GM")
        self.setMinimumSize(700, 540)
        self.resize(920, 660)
        self.setSizeGripEnabled(True)
        self.setModal(False)

        self.worker: AutoLearnWorker | None = None
        self.targeted_worker: TargetedLearnWorker | None = None
        self._validator: StockValidatorWorker | None = None

        self._is_running = False
        self._active_mode = ""  # "auto" or "targeted"
        self._targeted_stock_codes: list[str] = []
        self._seed_stock_codes: list[str] = [
            str(c).strip() for c in (seed_stock_codes or []) if str(c).strip()
        ]
        self._validation_request_id = 0
        self._last_progress_percent = 0
        self._error_dialog_shown = False
        self._elapsed_seconds = 0
        self._run_started_monotonic = 0.0
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._on_elapsed_tick)

        # Last validated stock (for Add button)
        self._last_validated_code = ""
        self._last_validated_name = ""
        self._last_validated_bars = 0

        self._setup_ui()
        self._load_seed_stocks()
        self._apply_style()

    # =========================================================================
    # =========================================================================

    def _setup_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setSpacing(8)
        root_layout.setContentsMargins(10, 10, 10, 10)

        header = QLabel("Auto Train GM (Continuous Discovery + Learning)")
        header.setObjectName("dialogTitle")
        root_layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setObjectName("dialogScroll")
        scroll.setWidgetResizable(True)
        root_layout.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(content)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("dialogTabs")

        # Tab 1: Auto Learn
        auto_tab = self._create_auto_tab()
        self.tabs.addTab(auto_tab, "Auto Train GM")

        # Keep targeted-search controls instantiated for compatibility but hide
        # the tab so users cannot start specific-stock training from the UI.
        self._hidden_targeted_tab = self._create_search_tab()

        layout.addWidget(self.tabs)

        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.status_label = QLabel("Ready to start")
        self.status_label.setObjectName("dialogStatus")
        progress_layout.addWidget(self.status_label)

        self.elapsed_label = QLabel("Elapsed: 00:00:00")
        self.elapsed_label.setObjectName("dialogHint")
        progress_layout.addWidget(self.elapsed_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setFormat("0%")
        progress_layout.addWidget(self.progress_bar)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setObjectName("dialogLog")
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        self.log_text.setFont(
            QFont(get_monospace_font_family(), ModernFonts.SIZE_SM)
        )
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.setMinimumHeight(34)
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

    # =========================================================================
    # TAB 1: Auto Learn
    # =========================================================================

    def _create_auto_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        desc = QLabel(
            "Automatically discover stocks from market internet sources in "
            "China-direct mode, fetch data, and train the GM model."
        )
        desc.setWordWrap(True)
        desc.setObjectName("dialogHint")
        layout.addWidget(desc)

        settings_group = QGroupBox("Settings")
        settings_layout = QGridLayout()
        settings_layout.setSpacing(10)

        settings_layout.addWidget(QLabel("Learning Mode:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Full (Discover + Fetch + Train)",
            "Discovery + Fetch Only",
        ])
        self.mode_combo.setMinimumWidth(250)
        settings_layout.addWidget(self.mode_combo, 0, 1)

        settings_layout.addWidget(QLabel("Max Stocks:"), 1, 0)
        self.max_stocks_spin = QSpinBox()
        self.max_stocks_spin.setRange(10, 10000)
        self.max_stocks_spin.setValue(200)
        self.max_stocks_spin.setSuffix(" stocks")
        settings_layout.addWidget(self.max_stocks_spin, 1, 1)

        settings_layout.addWidget(QLabel("Epochs:"), 2, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 200)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setSuffix(" epochs")
        settings_layout.addWidget(self.epochs_spin, 2, 1)

        settings_layout.addWidget(QLabel("Interval:"), 3, 0)
        self.auto_interval_combo = QComboBox()
        self.auto_interval_combo.addItems(["1m"])
        self.auto_interval_combo.setCurrentText("1m")
        self.auto_interval_combo.setEnabled(False)
        self.auto_interval_combo.setToolTip(
            "Learning interval is locked to 1m for real-time training."
        )
        settings_layout.addWidget(self.auto_interval_combo, 3, 1)

        settings_layout.addWidget(QLabel("Horizon:"), 4, 0)
        self.auto_horizon_spin = QSpinBox()
        self.auto_horizon_spin.setRange(1, 500)
        self.auto_horizon_spin.setValue(30)
        self.auto_horizon_spin.setSuffix(" bars")
        settings_layout.addWidget(self.auto_horizon_spin, 4, 1)

        self.discover_check = QCheckBox("Discover new stocks from internet")
        self.discover_check.setChecked(True)
        settings_layout.addWidget(self.discover_check, 5, 0, 1, 2)

        self.incremental_check = QCheckBox(
            "Incremental training (keep existing weights)"
        )
        self.incremental_check.setChecked(True)
        self.incremental_check.setEnabled(False)
        settings_layout.addWidget(self.incremental_check, 6, 0, 1, 2)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        btn_layout = QHBoxLayout()

        self.auto_start_btn = QPushButton("Start")
        self.auto_start_btn.setMinimumHeight(45)
        self.auto_start_btn.setStyleSheet(self._green_button_style())
        self.auto_start_btn.clicked.connect(
            lambda _checked=False: self._start_auto_learning(resume=False)
        )
        btn_layout.addWidget(self.auto_start_btn)

        self.auto_resume_btn = QPushButton("Resume")
        self.auto_resume_btn.setMinimumHeight(45)
        self.auto_resume_btn.setEnabled(False)
        self.auto_resume_btn.clicked.connect(
            lambda _checked=False: self._start_auto_learning(resume=True)
        )
        btn_layout.addWidget(self.auto_resume_btn)

        self.auto_stop_btn = QPushButton("Stop")
        self.auto_stop_btn.setMinimumHeight(45)
        self.auto_stop_btn.setEnabled(False)
        self.auto_stop_btn.clicked.connect(self._stop_auto_learning)
        btn_layout.addWidget(self.auto_stop_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return tab

    # =========================================================================
    # TAB 2: Train by Search
    # =========================================================================

    def _create_search_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        desc = QLabel(
            "Search for specific stocks and train the model on them. "
            "Enter stock codes (e.g., sh600519, sz000001, 000858) "
            "and add them to the training list."
        )
        desc.setWordWrap(True)
        desc.setObjectName("dialogHint")
        layout.addWidget(desc)

        # --- Search section ---
        search_group = QGroupBox("Search and Add Stocks")
        search_layout = QVBoxLayout()

        search_row = QHBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Enter stock code (e.g., sh600519, sz000001, 000858)..."
        )
        self.search_input.setMinimumHeight(38)
        self.search_input.returnPressed.connect(self._search_stock)
        search_row.addWidget(self.search_input)

        self.search_btn = QPushButton("Search")
        self.search_btn.setMinimumHeight(38)
        self.search_btn.setMinimumWidth(100)
        self.search_btn.clicked.connect(self._search_stock)
        search_row.addWidget(self.search_btn)

        self.add_btn = QPushButton("Add")
        self.add_btn.setMinimumHeight(38)
        self.add_btn.setMinimumWidth(80)
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(self._add_searched_stock)
        search_row.addWidget(self.add_btn)

        search_layout.addLayout(search_row)

        self.search_result_label = QLabel("")
        self.search_result_label.setObjectName("searchResultLabel")
        search_layout.addWidget(self.search_result_label)

        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        # --- Stock list section ---
        list_group = QGroupBox("Training Stock List")
        list_layout = QVBoxLayout()

        quick_row = QHBoxLayout()
        quick_label = QLabel("Quick add:")
        quick_label.setObjectName("dialogHint")
        quick_row.addWidget(quick_label)

        # Popular stocks - codes match CONFIG.STOCK_POOL format (bare 6-digit)
        # but we add sh/sz prefix so they display nicely and the fetcher
        # strips the prefix internally via clean_code() -> parse_instrument()
        popular_stocks = [
            ("Kweichow Moutai", "sh600519"),
            ("Ping An Bank", "sz000001"),
            ("Wuliangye", "sz000858"),
            ("China Merchants Bank", "sh600036"),
            ("Ping An Insurance", "sh601318"),
        ]
        for name, code in popular_stocks:
            btn = QPushButton(name)
            btn.setToolTip(code)
            btn.setMaximumHeight(28)
            btn.setObjectName("chipButton")
            btn.clicked.connect(
                lambda checked, c=code: self._quick_add_stock(c)
            )
            quick_row.addWidget(btn)

        quick_row.addStretch()
        list_layout.addLayout(quick_row)

        self.stock_list = QListWidget()
        self.stock_list.setMinimumHeight(120)
        self.stock_list.setObjectName("dialogStockList")
        self.stock_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        list_layout.addWidget(self.stock_list)

        list_btn_row = QHBoxLayout()

        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self._remove_selected_stocks)
        list_btn_row.addWidget(self.remove_btn)

        self.clear_list_btn = QPushButton("Clear All")
        self.clear_list_btn.clicked.connect(self._clear_stock_list)
        list_btn_row.addWidget(self.clear_list_btn)

        list_btn_row.addStretch()

        self.stock_count_label = QLabel("0 stocks in list")
        self.stock_count_label.setObjectName("stockCountLabel")
        list_btn_row.addWidget(self.stock_count_label)

        list_layout.addLayout(list_btn_row)
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)

        # --- Training settings for targeted ---
        target_settings = QGroupBox("Training Settings")
        ts_layout = QGridLayout()
        ts_layout.setSpacing(8)

        ts_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.target_epochs_spin = QSpinBox()
        self.target_epochs_spin.setRange(10, 200)
        self.target_epochs_spin.setValue(50)
        self.target_epochs_spin.setSuffix(" epochs")
        ts_layout.addWidget(self.target_epochs_spin, 0, 1)

        ts_layout.addWidget(QLabel("Interval:"), 0, 2)
        self.target_interval_combo = QComboBox()
        self.target_interval_combo.addItems(["1m"])
        self.target_interval_combo.setCurrentText("1m")
        self.target_interval_combo.setEnabled(False)
        ts_layout.addWidget(self.target_interval_combo, 0, 3)

        ts_layout.addWidget(QLabel("Horizon:"), 1, 0)
        self.target_horizon_spin = QSpinBox()
        self.target_horizon_spin.setRange(1, 500)
        self.target_horizon_spin.setValue(30)
        self.target_horizon_spin.setSuffix(" bars")
        ts_layout.addWidget(self.target_horizon_spin, 1, 1)

        self.target_incremental_check = QCheckBox("Incremental (keep weights)")
        self.target_incremental_check.setChecked(True)
        ts_layout.addWidget(self.target_incremental_check, 1, 2, 1, 2)

        target_settings.setLayout(ts_layout)
        layout.addWidget(target_settings)

        # --- Train button ---
        train_btn_layout = QHBoxLayout()

        self.target_start_btn = QPushButton("Train on Selected Stocks")
        self.target_start_btn.setMinimumHeight(45)
        self.target_start_btn.setStyleSheet(self._blue_button_style())
        self.target_start_btn.clicked.connect(self._start_targeted_learning)
        train_btn_layout.addWidget(self.target_start_btn)

        self.target_stop_btn = QPushButton("Stop")
        self.target_stop_btn.setMinimumHeight(45)
        self.target_stop_btn.setEnabled(False)
        self.target_stop_btn.clicked.connect(self._stop_targeted_learning)
        train_btn_layout.addWidget(self.target_stop_btn)

        train_btn_layout.addStretch()
        layout.addLayout(train_btn_layout)

        return tab

    # =========================================================================
    # =========================================================================

    def _normalize_code(self, code: str) -> str:
        """Normalize stock code input.

        The fetcher's clean_code() strips prefixes anyway, but we add
        them here so the UI shows a recognizable format and the code
        survives the round-trip through ParallelFetcher -> Trainer.

        Handles: 600519, sh600519, SH600519, 000001, sz000001, SZ.000001
        """
        code = code.strip()

        for ch in (" ", ".", "-", "_"):
            code = code.replace(ch, "")

        code_lower = code.lower()

        # Already has valid prefix
        if code_lower.startswith(("sh", "sz", "bj")):
            return code_lower

        # Pure digits - guess prefix from first digit
        digits = "".join(ch for ch in code if ch.isdigit())
        if len(digits) == 6:
            first = digits[0]
            if first in ("6", "5"):
                return f"sh{digits}"
            elif first in ("0", "3", "1", "2"):
                return f"sz{digits}"

        # Return as-is if we can't normalize
        return code_lower

    def _search_stock(self) -> None:
        """Search/validate a stock code."""
        raw = self.search_input.text().strip()
        if not raw:
            self.search_result_label.setText(
                f'<span style="color: {ModernColors.ACCENT_WARNING};">'
                "Please enter a stock code"
                "</span>"
            )
            return

        code = self._normalize_code(raw)

        if code in self._targeted_stock_codes:
            self.search_result_label.setText(
                f'<span style="color: {ModernColors.ACCENT_WARNING};">'
                f"Warning: {code} is already in the list"
                f"</span>"
            )
            return

        self.search_btn.setEnabled(False)
        self.search_btn.setText("Searching...")
        self.add_btn.setEnabled(False)
        self.search_result_label.setText(
            f'<span style="color: {ModernColors.TEXT_SECONDARY};">'
            f"Validating {code}..."
            f"</span>"
        )

        interval = "1m"
        self._validation_request_id += 1
        self._validator = StockValidatorWorker(
            code, interval, request_id=self._validation_request_id
        )
        self._validator.validation_result.connect(self._on_validation_result)
        self._validator.start()

    def _on_validation_result(self, result: dict) -> None:
        """Handle stock validation result from background thread."""
        request_id = int(result.get("request_id", 0) or 0)
        if request_id != self._validation_request_id:
            # Ignore stale async result from an older search action.
            return

        # Re-enable search button
        self.search_btn.setEnabled(True)
        self.search_btn.setText("Search")

        code = result.get("code", "")
        valid = result.get("valid", False)
        name = result.get("name", "")
        bars = result.get("bars", 0)
        message = result.get("message", "")

        if valid:
            display = code
            if name:
                display += f" ({name})"
            display += f" - {bars} bars"

            self.search_result_label.setText(
                f'<span style="color: {ModernColors.ACCENT_SUCCESS};">Valid: {display}</span>'
            )
            self.add_btn.setEnabled(True)

            self._last_validated_code = code
            self._last_validated_name = name
            self._last_validated_bars = bars
        else:
            self.search_result_label.setText(
                f'<span style="color: {ModernColors.ACCENT_DANGER};">Invalid: {code}: {message}</span>'
            )
            self.add_btn.setEnabled(False)
            self._last_validated_code = ""

        self._validator = None

    def _add_searched_stock(self) -> None:
        """Add the last validated stock to the training list."""
        code = self._last_validated_code
        if not code:
            return

        if code in self._targeted_stock_codes:
            self.search_result_label.setText(
                f'<span style="color: {ModernColors.ACCENT_WARNING};">'
                f"Warning: {code} already in list"
                f"</span>"
            )
            return

        self._targeted_stock_codes.append(code)
        self._add_stock_to_list_widget(
            code, self._last_validated_name, self._last_validated_bars
        )

        self.search_input.clear()
        self.search_result_label.setText(
            f'<span style="color: {ModernColors.ACCENT_SUCCESS};">'
            f"Added {code} to training list"
            f"</span>"
        )
        self.add_btn.setEnabled(False)
        self._last_validated_code = ""
        self._update_stock_count()

    def _quick_add_stock(self, code: str) -> None:
        """Add a stock from the quick-add buttons without validation."""
        if code in self._targeted_stock_codes:
            self._log(f"{code} already in list", "warning")
            return

        self._targeted_stock_codes.append(code)
        self._add_stock_to_list_widget(code, "", 0)
        self._update_stock_count()
        self._log(f"Added {code} to training list", "info")

    def _add_stock_to_list_widget(
        self, code: str, name: str, bars: int
    ) -> None:
        """Add a stock item to the QListWidget."""
        display = f"  {code}"
        if name:
            display += f"  -  {name}"
        if bars > 0:
            display += f"  ({bars} bars)"

        item = QListWidgetItem(display)
        item.setData(Qt.ItemDataRole.UserRole, code)
        self.stock_list.addItem(item)

    def _remove_selected_stocks(self) -> None:
        """Remove selected stocks from the list."""
        selected = self.stock_list.selectedItems()
        if not selected:
            return

        for item in selected:
            code = item.data(Qt.ItemDataRole.UserRole)
            if code in self._targeted_stock_codes:
                self._targeted_stock_codes.remove(code)
            row = self.stock_list.row(item)
            self.stock_list.takeItem(row)

        self._update_stock_count()

    def _clear_stock_list(self) -> None:
        """Clear all stocks from the list."""
        if not self._targeted_stock_codes:
            return

        reply = QMessageBox.question(
            self,
            "Clear List",
            f"Remove all {len(self._targeted_stock_codes)} stocks "
            f"from the list?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._targeted_stock_codes.clear()
        self.stock_list.clear()
        self._update_stock_count()

    def _update_stock_count(self) -> None:
        """Update the stock count label."""
        count = len(self._targeted_stock_codes)
        if count == 0:
            self.stock_count_label.setText("0 stocks in list")
            self.stock_count_label.setStyleSheet(
                f"color: {ModernColors.TEXT_SECONDARY}; "
                f"font-size: {ModernFonts.SIZE_SM}px;"
            )
        else:
            suffix = "s" if count != 1 else ""
            self.stock_count_label.setText(
                f"{count} stock{suffix} in list"
            )
            self.stock_count_label.setStyleSheet(
                f"color: {ModernColors.ACCENT_SUCCESS}; "
                f"font-size: {ModernFonts.SIZE_SM}px;"
            )

    def _load_seed_stocks(self) -> None:
        """Session-seeded training is disabled."""
        return

    def _collect_priority_codes(self, mode: str = "auto") -> list[str]:
        _ = mode
        return []

    # =========================================================================
    # AUTO LEARN START/STOP
    # =========================================================================

    def _start_auto_learning(self, resume: bool = False) -> None:
        """Start or resume auto-learning (random rotation)."""
        if self._is_running:
            return

        LearnerClass = _get_auto_learner()
        if LearnerClass is None:
            QMessageBox.critical(
                self,
                "Module Not Found",
                "AutoLearner module not found.\n\n"
                "Ensure models/auto_learner.py exists.",
            )
            return

        mode_map = {0: "full", 1: "discovery"}

        config = {
            "mode": mode_map.get(self.mode_combo.currentIndex(), "full"),
            "discover_new": self.discover_check.isChecked(),
            "max_stocks": self.max_stocks_spin.value(),
            "epochs": self.epochs_spin.value(),
            "interval": "1m",
            "horizon": int(self.auto_horizon_spin.value()),
            "incremental": True,
        }

        # China-only mode: network check for diagnostics
        try:
            from core.network import get_network_env
            env = get_network_env()
            if not env.is_china_direct:
                self._log(
                    "China direct mode: ensure stable network connection for large stock batches",
                    "info",
                )
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            log.debug("Network advisory check skipped", exc_info=True)

        is_resume = bool(resume and self._elapsed_seconds > 0)
        self._log(
            "Resuming Auto Train GM..." if is_resume else "Starting Auto Train GM...",
            "info",
        )
        self._log(
            f"Mode: {config['mode']}, "
            f"Max stocks: {config['max_stocks']}, "
            f"Epochs: {config['epochs']}, "
            f"Interval: {config['interval']}, "
            f"Horizon: {config['horizon']}",
            "info",
        )

        self._set_running(True, mode="auto", keep_progress=is_resume)
        self._start_elapsed_clock(reset=not is_resume)
        if is_resume:
            self.status_label.setText("Resuming...")
        else:
            self.status_label.setText("Starting...")

        self.worker = AutoLearnWorker(config)
        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._log)
        self.worker.finished_result.connect(self._on_auto_finished)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

    def start_or_resume_auto_learning(self) -> None:
        """Public entrypoint used by parent to auto-start this panel."""
        if self._is_running:
            return
        self._start_auto_learning(resume=bool(self._elapsed_seconds > 0))

    def _stop_auto_learning(self) -> None:
        """Stop auto-learning gracefully."""
        if not self._is_running:
            return

        self._log("Stopping Auto Train GM...", "warning")
        self.status_label.setText("Stopping...")
        self.auto_stop_btn.setEnabled(False)

        if self.worker:
            self.worker.stop()
            self._log(
                "Stop requested. Finalizing current training step...",
                "info",
            )
            # Non-blocking stop: keep UI responsive and let worker emit
            # final stopped status through _on_auto_finished.
            return

        self._stop_elapsed_clock()
        self._set_running(False)

    # =========================================================================
    # TARGETED LEARN START/STOP
    # =========================================================================

    def _start_targeted_learning(self) -> None:
        """Start training on user-selected stocks."""
        QMessageBox.information(
            self,
            "Disabled",
            "Specific-stock training is disabled.\n\n"
            "Use Auto Train GM to continue training with broad market data.",
        )
        return

    def _stop_targeted_learning(self) -> None:
        """Stop targeted training gracefully."""
        if not self._is_running:
            return

        self._log("Stopping targeted training...", "warning")
        self.status_label.setText("Stopping...")
        self.target_stop_btn.setEnabled(False)

        if self.targeted_worker:
            self.targeted_worker.stop()
            self._log(
                "Stop requested. Finalizing current training step...",
                "info"
            )
            # Non-blocking stop: completion is handled by _on_targeted_finished.
            return

        self._stop_elapsed_clock()
        self._set_running(False)

    # =========================================================================
    # =========================================================================

    def _set_running(
        self,
        running: bool,
        mode: str = "",
        keep_progress: bool = False,
    ) -> None:
        """Set UI running state.
        Disables all interactive controls while training is active.
        """
        self._is_running = running
        self._active_mode = mode if running else ""

        if running:
            if not keep_progress:
                self.progress_bar.setValue(0)
                self.progress_bar.setFormat("0%")
                self.status_label.setText("Initializing...")
                self._last_progress_percent = 0
            self._error_dialog_shown = False
            self.close_btn.setEnabled(False)

            # Disable BOTH tabs' start buttons (only one can run)
            self.auto_start_btn.setEnabled(False)
            self.auto_resume_btn.setEnabled(False)
            self.target_start_btn.setEnabled(False)

            if mode == "auto":
                self.auto_stop_btn.setEnabled(True)
                self.target_stop_btn.setEnabled(False)
                self.mode_combo.setEnabled(False)
                self.max_stocks_spin.setEnabled(False)
                self.epochs_spin.setEnabled(False)
                self.auto_interval_combo.setEnabled(False)
                self.auto_horizon_spin.setEnabled(False)
                self.discover_check.setEnabled(False)
                self.incremental_check.setEnabled(False)
                if hasattr(self, "use_session_cache_check"):
                    self.use_session_cache_check.setEnabled(False)

            elif mode == "targeted":
                self.auto_stop_btn.setEnabled(False)
                self.target_stop_btn.setEnabled(True)
                self.search_input.setEnabled(False)
                self.search_btn.setEnabled(False)
                self.add_btn.setEnabled(False)
                self.remove_btn.setEnabled(False)
                self.clear_list_btn.setEnabled(False)
                self.target_epochs_spin.setEnabled(False)
                self.target_interval_combo.setEnabled(False)
                self.target_horizon_spin.setEnabled(False)
                self.target_incremental_check.setEnabled(False)

        else:
            # Re-enable everything
            self.close_btn.setEnabled(True)
            self.auto_start_btn.setEnabled(True)
            self.auto_resume_btn.setEnabled(
                bool(self._elapsed_seconds > 0 and self.progress_bar.value() < 100)
            )
            self.target_start_btn.setEnabled(True)
            self.auto_stop_btn.setEnabled(False)
            self.target_stop_btn.setEnabled(False)

            self.mode_combo.setEnabled(True)
            self.max_stocks_spin.setEnabled(True)
            self.epochs_spin.setEnabled(True)
            self.auto_interval_combo.setEnabled(False)
            self.auto_horizon_spin.setEnabled(True)
            self.discover_check.setEnabled(True)
            self.incremental_check.setEnabled(True)
            if hasattr(self, "use_session_cache_check"):
                self.use_session_cache_check.setEnabled(True)

            self.search_input.setEnabled(True)
            self.search_btn.setEnabled(True)
            # add_btn stays disabled until next search
            self.remove_btn.setEnabled(True)
            self.clear_list_btn.setEnabled(True)
            self.target_epochs_spin.setEnabled(True)
            self.target_interval_combo.setEnabled(False)
            self.target_horizon_spin.setEnabled(True)
            self.target_incremental_check.setEnabled(True)

            self.worker = None
            self.targeted_worker = None
            self._last_progress_percent = int(max(0, self.progress_bar.value()))

    # =========================================================================
    # =========================================================================

    def _on_progress(self, percent: int, message: str) -> None:
        """Handle progress update from either worker."""
        try:
            p_raw = int(float(percent))
        except (TypeError, ValueError, OverflowError):
            p_raw = self._last_progress_percent
        p = int(max(0, min(100, p_raw)))
        msg = str(message or "").strip()
        # Keep progress stable on noisy callback order.
        if p < self._last_progress_percent and "cycle" not in msg.lower():
            p = self._last_progress_percent
        self._last_progress_percent = p
        self.progress_bar.setValue(p)
        self.progress_bar.setFormat(f"{p}%")
        self.status_label.setText(msg or f"Running ({p}%)")

    def _on_auto_finished(self, results: dict) -> None:
        """Handle auto-learning completion."""
        self._stop_elapsed_clock()
        self._log("=" * 50, "info")
        status = results.get("status", "ok")
        if status == "stopped":
            self._log("Auto Train GM stopped by user", "warning")
            self._set_running(False)
            self.status_label.setText("Stopped")
            self.progress_bar.setFormat("Stopped")
            self.session_finished.emit(dict(results or {"status": "stopped"}))
            return
        if status == "error":
            err = str(results.get("error") or "Auto Train GM failed")
            self._log(f"Auto Train GM failed: {err}", "error")
            self._log_results(results)
            self._set_running(False)
            current = max(0, min(99, self.progress_bar.value()))
            self.progress_bar.setValue(current)
            self.progress_bar.setFormat("Failed")
            self.status_label.setText("Auto Train GM failed")
            if not self._error_dialog_shown:
                self._error_dialog_shown = True
                QMessageBox.critical(
                    self,
                    "Learning Failed",
                    f"Auto Train GM failed:\n\n{err}",
                )
            payload = dict(results or {})
            payload["status"] = "error"
            payload["error"] = err
            self.session_finished.emit(payload)
            return

        self._log("Auto Train GM completed", "success")
        self._log_results(results)

        self._set_running(False)
        self.status_label.setText("Auto Train GM completed")
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Complete")
        self.auto_resume_btn.setEnabled(False)

        QMessageBox.information(
            self,
            "Learning Complete",
            f"Auto Train GM completed!\n\n"
            f"Stocks discovered: {results.get('discovered', 0)}\n"
            f"Stocks processed: {results.get('processed', 0)}\n"
            f"Accuracy: {results.get('accuracy', 0):.1%}",
        )
        done_payload = dict(results or {})
        done_payload["status"] = str(done_payload.get("status", "ok") or "ok")
        self.session_finished.emit(done_payload)

    def _on_targeted_finished(self, results: dict) -> None:
        """Handle targeted training completion."""
        self._stop_elapsed_clock()
        self._log("=" * 50, "info")
        status = results.get("status", "ok")
        if status == "stopped":
            self._log("Targeted training stopped by user", "warning")
            self._set_running(False)
            self.status_label.setText("Stopped")
            self.progress_bar.setFormat("Stopped")
            self.session_finished.emit(dict(results or {"status": "stopped"}))
            return
        if status == "error":
            err = str(results.get("error") or "Targeted training failed")
            self._log(f"Targeted training failed: {err}", "error")
            self._log_results(results)
            self._set_running(False)
            current = max(0, min(99, self.progress_bar.value()))
            self.progress_bar.setValue(current)
            self.progress_bar.setFormat("Failed")
            self.status_label.setText("Targeted training failed")
            if not self._error_dialog_shown:
                self._error_dialog_shown = True
                QMessageBox.critical(
                    self,
                    "Training Failed",
                    f"Targeted training failed:\n\n{err}",
                )
            payload = dict(results or {})
            payload["status"] = "error"
            payload["error"] = err
            self.session_finished.emit(payload)
            return

        self._log("Targeted training completed", "success")
        self._log_results(results)

        stocks = results.get("stocks_trained", [])
        if stocks:
            display = ", ".join(stocks[:10])
            if len(stocks) > 10:
                display += f"... (+{len(stocks) - 10})"
            self._log(f"Trained on: {display}", "info")

        self._set_running(False)
        self.status_label.setText("Targeted training completed")
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Complete")

        QMessageBox.information(
            self,
            "Training Complete",
            f"Targeted training completed!\n\n"
            f"Stocks trained: {len(stocks)}\n"
            f"Stocks processed: {results.get('processed', 0)}\n"
            f"Accuracy: {results.get('accuracy', 0):.1%}",
        )
        done_payload = dict(results or {})
        done_payload["status"] = str(done_payload.get("status", "ok") or "ok")
        self.session_finished.emit(done_payload)

    def _on_error(self, error: str) -> None:
        """Handle error from either worker."""
        self._stop_elapsed_clock()
        error = str(error or "Unknown error")
        display_error = error[:300] if len(error) > 300 else error
        self._log(f"Error: {display_error}", "error")

        self._set_running(False)
        current = max(0, min(99, self.progress_bar.value()))
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat("Failed")
        self.status_label.setText("Error occurred")

        if not self._error_dialog_shown:
            self._error_dialog_shown = True
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred during learning:\n\n{error[:500]}",
            )
        self.session_finished.emit({"status": "error", "error": error})

    def _log_results(self, results: dict) -> None:
        """Log training results to the activity log."""
        discovered = int(results.get("discovered", 0))
        processed = int(results.get("processed", 0))
        accuracy = float(results.get("accuracy", 0))

        if discovered > 0:
            self._log(f"Stocks discovered: {discovered}", "info")
        if processed > 0:
            self._log(f"Stocks processed: {processed}", "info")
        if accuracy > 0:
            self._log(f"Model accuracy: {accuracy:.1%}", "success")

    def _start_elapsed_clock(self, *, reset: bool) -> None:
        if reset:
            self._elapsed_seconds = 0
        self._run_started_monotonic = float(time.monotonic()) - float(self._elapsed_seconds)
        self._on_elapsed_tick()
        self._elapsed_timer.start()

    def _stop_elapsed_clock(self) -> None:
        self._elapsed_timer.stop()
        self._on_elapsed_tick()

    def _on_elapsed_tick(self) -> None:
        if self._is_running and self._run_started_monotonic > 0.0:
            self._elapsed_seconds = int(
                max(0.0, float(time.monotonic()) - self._run_started_monotonic)
            )
        self.elapsed_label.setText(f"Elapsed: {self._format_elapsed(self._elapsed_seconds)}")

    @staticmethod
    def _format_elapsed(seconds: int) -> str:
        safe = max(0, int(seconds))
        h = safe // 3600
        m = (safe % 3600) // 60
        s = safe % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # =========================================================================
    # =========================================================================

    def _log(self, message: str, level: str = "info") -> None:
        """Add message to activity log with timestamp and color."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        colors = {
            "info": ModernColors.TEXT_SECONDARY,
            "success": ModernColors.ACCENT_SUCCESS,
            "warning": ModernColors.ACCENT_WARNING,
            "error": ModernColors.ACCENT_DANGER,
        }
        tags = {
            "info": "INFO",
            "success": "SUCCESS",
            "warning": "WARN",
            "error": "ERROR",
        }
        color = colors.get(level, ModernColors.TEXT_SECONDARY)
        tag = tags.get(level, "INFO")

        self.log_text.append(
            f'<span style="color: {ModernColors.TEXT_MUTED};">[{timestamp}]</span> '
            f'<span style="color: {ModernColors.TEXT_MUTED};">[{tag}]</span> '
            f'<span style="color: {color};">{message}</span>'
        )

        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())

    # =========================================================================
    # =========================================================================

    @staticmethod
    def _green_button_style() -> str:
        return """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1f8f5f, stop:1 #30b475);
                color: #f5f9ff;
                border: 1px solid #49cd95;
                padding: 11px 24px;
                border-radius: 10px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #24a96e, stop:1 #3bc487);
            }
            QPushButton:pressed {
                background: #1d7f53;
            }
            QPushButton:disabled {
                background: #13243d;
                border-color: #2a3f5f;
                color: #6b7d9c;
            }
        """

    @staticmethod
    def _blue_button_style() -> str:
        return """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2f67de, stop:1 #3b82f6);
                color: #f5f9ff;
                border: 1px solid #5d95ff;
                padding: 11px 24px;
                border-radius: 10px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b75ec, stop:1 #4a95ff);
            }
            QPushButton:pressed {
                background: #2a5fcc;
            }
            QPushButton:disabled {
                background: #13243d;
                border-color: #2a3f5f;
                color: #6b7d9c;
            }
        """

    # =========================================================================
    # =========================================================================

    def _apply_style(self) -> None:
        self.setStyleSheet(
            get_dialog_style()
            + f"""
            QLabel#dialogTitle {{
                font-size: {ModernFonts.SIZE_XXL}px;
                font-weight: {ModernFonts.WEIGHT_BOLD};
                color: {ModernColors.ACCENT_INFO};
                padding: 2px 0 4px 0;
            }}
            QLabel#dialogStatus {{
                font-weight: {ModernFonts.WEIGHT_BOLD};
                font-size: {ModernFonts.SIZE_BASE}px;
                color: {ModernColors.TEXT_PRIMARY};
            }}
            QLabel#searchResultLabel {{
                font-size: {ModernFonts.SIZE_SM}px;
                padding: 4px 2px;
                color: {ModernColors.TEXT_SECONDARY};
            }}
            QPushButton#chipButton {{
                background: #14243d;
                color: {ModernColors.TEXT_SECONDARY};
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 6px;
                padding: 2px 8px;
                font-size: {ModernFonts.SIZE_XS}px;
                min-height: 26px;
            }}
            QPushButton#chipButton:hover {{
                background: #20406b;
                color: {ModernColors.TEXT_STRONG};
                border-color: {ModernColors.BORDER_FOCUS};
            }}
            QListWidget#dialogStockList {{
                background: #0c1728;
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 8px;
                color: {ModernColors.TEXT_PRIMARY};
                font-size: {ModernFonts.SIZE_SM}px;
            }}
            QListWidget#dialogStockList::item {{
                padding: 6px 10px;
                border-bottom: 1px solid #14243d;
            }}
            QListWidget#dialogStockList::item:selected {{
                background: #2f5fda;
                color: {ModernColors.TEXT_STRONG};
            }}
            QLabel#stockCountLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: {ModernFonts.SIZE_SM}px;
            }}
            """
        )

    # =========================================================================
    # =========================================================================

    def closeEvent(self, event) -> None:
        """Handle close - stop any running training first."""
        if self._is_running:
            reply = QMessageBox.question(
                self,
                "Stop Learning?",
                "Learning is still in progress.\n\nStop and close?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                if self.worker:
                    self.worker.stop()
                    self.worker.wait(5000)
                if self.targeted_worker:
                    self.targeted_worker.stop()
                    self.targeted_worker.wait(5000)
                if self._validator:
                    self._validator.wait(2000)
                    self._validator = None
                self._stop_elapsed_clock()
                event.accept()
            else:
                event.ignore()
                return
        else:
            if self._validator:
                self._validator.wait(2000)
                self._validator = None
            self._stop_elapsed_clock()
            event.accept()

        super().closeEvent(event)

def show_auto_learn_dialog(parent=None, seed_stock_codes: list[str] | None = None):
    """Show the auto-learn dialog in non-modal mode."""
    dialog = AutoLearnDialog(parent, seed_stock_codes=seed_stock_codes)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
