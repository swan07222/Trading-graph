# ui/auto_learn_dialog.py

import threading
import time
import traceback
from datetime import datetime

from PyQt6.QtCore import Qt, QThread, pyqtSignal
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

from utils.logger import get_logger

log = get_logger(__name__)

def _get_cancellation_token():
    """Lazy import CancellationToken."""
    from utils.cancellation import CancellationToken
    return CancellationToken()

def _get_auto_learner():
    """
    Lazy import AutoLearner/ContinuousLearner.
    Returns the class, not an instance.
    """
    try:
        from models.auto_learner import AutoLearner
        return AutoLearner
    except ImportError:
        pass

    try:
        from models.auto_learner import ContinuousLearner
        return ContinuousLearner
    except ImportError:
        pass

    return None

# WORKER: Auto Learn (random rotation)

class AutoLearnWorker(QThread):
    """
    Worker thread for auto-learning (random stock rotation).
    Runs the learner in a separate daemon thread so that
    learner.start() (which may block) doesn't prevent the
    QThread from responding to stop requests.
    """
    progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str, str)
    finished_result = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config) if config else {}
        self.running = False
        self.token = _get_cancellation_token()
        self._learner = None
        self._learner_thread: threading.Thread | None = None
        self._error_flag = False
        self._error_message = ""

    def run(self):
        self.running = True
        results = {
            "discovered": 0,
            "processed": 0,
            "samples": 0,
            "accuracy": 0.0,
        }

        try:
            LearnerClass = _get_auto_learner()
            if LearnerClass is None:
                self.error_occurred.emit(
                    "AutoLearner/ContinuousLearner not found. "
                    "Ensure models/auto_learner.py exists."
                )
                return

            self._learner = LearnerClass()

            max_stocks = int(self.config.get("max_stocks", 50))
            epochs = int(self.config.get("epochs", 10))
            incremental = bool(self.config.get("incremental", True))
            mode = str(self.config.get("mode", "full"))
            interval = "1m"
            horizon = 30
            lookback_bars = 3000
            cycle_interval_seconds = 900

            def on_progress(p):
                if not self.running:
                    return
                try:
                    percent = int(max(0, min(100, getattr(p, 'progress', 0))))
                    message = str(
                        getattr(p, 'message', '') or getattr(p, 'stage', '')
                    )
                    stage = str(getattr(p, 'stage', ''))
                    if stage == "error":
                        self._error_flag = True
                        self._error_message = message or "Auto-learning failed"
                    self.progress.emit(percent, message)
                    self.log_message.emit(
                        f"{stage}: {message}" if stage else message,
                        "info",
                    )
                    results["discovered"] = int(
                        getattr(p, "stocks_found", 0) or 0
                    )
                    results["processed"] = int(
                        getattr(p, "stocks_processed", 0) or 0
                    )
                    results["accuracy"] = float(
                        getattr(p, "validation_accuracy", 0.0) or 0.0
                    )
                except Exception as e:
                    log.debug(f"Progress callback error: {e}")

            if hasattr(self._learner, 'add_callback'):
                self._learner.add_callback(on_progress)

            learner_kwargs = {
                "mode": mode,
                "max_stocks": max_stocks,
                "epochs_per_cycle": epochs,
                "min_market_cap": 10,
                "include_all_markets": True,
                "continuous": True,
                "learning_while_trading": True,
                "interval": interval,
                "prediction_horizon": horizon,
                "lookback_bars": lookback_bars,
                "cycle_interval_seconds": cycle_interval_seconds,
                "incremental": incremental,
                "priority_stock_codes": list(
                    self.config.get("priority_stock_codes", []) or []
                ),
            }

            self._learner_thread = threading.Thread(
                target=self._run_learner,
                args=(learner_kwargs,),
                daemon=True,
                name="auto_learner_inner",
            )
            self._learner_thread.start()
            self.log_message.emit("Auto-learning started", "success")

            while self.running and not self.token.is_cancelled:
                if (
                    self._learner_thread is not None
                    and not self._learner_thread.is_alive()
                ):
                    break
                self.msleep(200)

            self._stop_learner()
            if self._error_flag:
                results["status"] = "error"
                results["error"] = self._error_message or "Auto-learning failed"
                if self.running:
                    self.error_occurred.emit(results["error"])
            else:
                results["status"] = "ok"
            self.finished_result.emit(results)

        except Exception as e:
            error_msg = str(e)
            log.error(f"AutoLearnWorker error: {error_msg}")
            log.debug(traceback.format_exc())
            if self.running:
                self.error_occurred.emit(error_msg)

    def _run_learner(self, kwargs: dict):
        try:
            if self._learner is not None:
                self._learner.start(**kwargs)
                # AutoLearner.start() spawns its own thread and returns immediately.
                # Join that internal thread so this worker doesn't finish early.
                t = getattr(self._learner, "_thread", None)
                if t is not None:
                    try:
                        t.join()
                        return
                    except Exception:
                        pass

                # Fallback: wait on progress flag if internal thread not exposed.
                while self.running and not self.token.is_cancelled:
                    if not getattr(self._learner.progress, "is_running", False):
                        break
                    time.sleep(0.2)
        except Exception as e:
            log.error(f"Learner thread error: {e}")
            log.debug(traceback.format_exc())
            if self.running:
                try:
                    self.error_occurred.emit(str(e))
                except RuntimeError:
                    pass

    def _stop_learner(self):
        if self._learner is not None:
            try:
                if hasattr(self._learner, 'stop'):
                    self._learner.stop()
            except Exception as e:
                log.debug(f"Learner stop error: {e}")

        if self._learner_thread is not None:
            try:
                self._learner_thread.join(timeout=10)
                if self._learner_thread.is_alive():
                    log.warning("Learner thread did not stop cleanly")
            except Exception:
                pass

        self._learner = None
        self._learner_thread = None

    def stop(self):
        self.running = False
        self.token.cancel()

# WORKER: Targeted Training (NEW)

class TargetedLearnWorker(QThread):
    """
    Worker thread for targeted training on user-selected stocks.
    Calls learner.start_targeted() which uses the fixed stock list
    instead of rotation/discovery.
    """
    progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str, str)
    finished_result = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config) if config else {}
        self.running = False
        self.token = _get_cancellation_token()
        self._learner = None
        self._learner_thread: threading.Thread | None = None

    def run(self):
        self.running = True
        results = {
            "discovered": 0,
            "processed": 0,
            "samples": 0,
            "accuracy": 0.0,
            "stocks_trained": [],
        }

        try:
            LearnerClass = _get_auto_learner()
            if LearnerClass is None:
                self.error_occurred.emit(
                    "AutoLearner/ContinuousLearner not found."
                )
                return

            stock_codes = list(self.config.get("stock_codes", []))
            if not stock_codes:
                self.error_occurred.emit("No stock codes provided.")
                return

            self._learner = LearnerClass()

            epochs = int(self.config.get("epochs", 10))
            incremental = bool(self.config.get("incremental", True))
            interval = str(self.config.get("interval", "1m"))
            horizon = int(self.config.get("horizon", 30))
            lookback = self.config.get("lookback_bars", None)
            continuous = bool(self.config.get("continuous", False))

            def on_progress(p):
                if not self.running:
                    return
                try:
                    percent = int(max(0, min(100, getattr(p, 'progress', 0))))
                    message = str(
                        getattr(p, 'message', '') or getattr(p, 'stage', '')
                    )
                    stage = str(getattr(p, 'stage', ''))
                    self.progress.emit(percent, message)
                    self.log_message.emit(
                        f"{stage}: {message}" if stage else message,
                        "info",
                    )
                    results["processed"] = int(
                        getattr(p, "stocks_processed", 0) or 0
                    )
                    results["accuracy"] = float(
                        getattr(p, "validation_accuracy", 0.0) or 0.0
                    )
                    results["discovered"] = int(
                        getattr(p, "stocks_found", 0) or 0
                    )
                except Exception as e:
                    log.debug(f"Targeted progress callback error: {e}")

            if hasattr(self._learner, 'add_callback'):
                self._learner.add_callback(on_progress)

            learner_kwargs = {
                "stock_codes": stock_codes,
                "epochs_per_cycle": epochs,
                "interval": interval,
                "prediction_horizon": horizon,
                "incremental": incremental,
                "continuous": continuous,
                "cycle_interval_seconds": 900,
            }
            if lookback is not None:
                learner_kwargs["lookback_bars"] = int(lookback)

            self._learner_thread = threading.Thread(
                target=self._run_targeted,
                args=(learner_kwargs,),
                daemon=True,
                name="targeted_learner_inner",
            )
            self._learner_thread.start()

            self.log_message.emit(
                f"Targeted training started on {len(stock_codes)} stocks",
                "success",
            )

            while self.running and not self.token.is_cancelled:
                if (
                    self._learner_thread is not None
                    and not self._learner_thread.is_alive()
                ):
                    break
                self.msleep(200)

            self._stop_learner()
            results["stocks_trained"] = stock_codes
            self.finished_result.emit(results)

        except Exception as e:
            error_msg = str(e)
            log.error(f"TargetedLearnWorker error: {error_msg}")
            log.debug(traceback.format_exc())
            if self.running:
                self.error_occurred.emit(error_msg)

    def _run_targeted(self, kwargs: dict):
        try:
            if self._learner is not None:
                self._learner.start_targeted(**kwargs)
        except Exception as e:
            log.error(f"Targeted learner thread error: {e}")
            log.debug(traceback.format_exc())
            if self.running:
                try:
                    self.error_occurred.emit(str(e))
                except RuntimeError:
                    pass

    def _stop_learner(self):
        if self._learner is not None:
            try:
                if hasattr(self._learner, 'stop'):
                    self._learner.stop()
            except Exception as e:
                log.debug(f"Targeted learner stop error: {e}")

        if self._learner_thread is not None:
            try:
                self._learner_thread.join(timeout=10)
            except Exception:
                pass

        self._learner = None
        self._learner_thread = None

    def stop(self):
        self.running = False
        self.token.cancel()

# STOCK VALIDATOR WORKER (for search)

class StockValidatorWorker(QThread):
    """
    Validates a stock code in background thread.
    Calls learner.validate_stock_code() which checks:
    - Code exists in the fetcher's data sources
    - Has enough bars for training (SEQUENCE_LENGTH + 20)
    - Returns stock name from spot cache if available
    """
    validation_result = pyqtSignal(dict)

    def __init__(self, code: str, interval: str = "1d", request_id: int = 0):
        super().__init__()
        self.code = code
        self.interval = interval
        self.request_id = int(request_id)

    def run(self):
        try:
            LearnerClass = _get_auto_learner()
            if LearnerClass is None:
                self.validation_result.emit({
                    'valid': False,
                    'code': self.code,
                    'name': '',
                    'bars': 0,
                    'request_id': self.request_id,
                    'message': 'Learner module not available',
                })
                return

            learner = LearnerClass()
            result = learner.validate_stock_code(self.code, self.interval)
            result["request_id"] = self.request_id
            self.validation_result.emit(result)

        except Exception as e:
            self.validation_result.emit({
                'valid': False,
                'code': self.code,
                'name': '',
                'bars': 0,
                'request_id': self.request_id,
                'message': f'Validation error: {str(e)[:200]}',
            })

class AutoLearnDialog(QDialog):
    """
    Dialog for automatic learning with two tabs:
    - Auto Learn: random stock rotation (existing)
    - Train by Search: user-selected stocks (new)
    """

    def __init__(self, parent=None, seed_stock_codes: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("🤖 Auto Learning")
        self.setMinimumSize(700, 540)
        self.resize(920, 660)
        self.setSizeGripEnabled(True)

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

        # Last validated stock (for Add button)
        self._last_validated_code = ""
        self._last_validated_name = ""
        self._last_validated_bars = 0

        self._setup_ui()
        self._load_seed_stocks()
        self._apply_style()

    # =========================================================================
    # =========================================================================

    def _setup_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setSpacing(8)
        root_layout.setContentsMargins(10, 10, 10, 10)

        header = QLabel("🧠 Automatic Stock Discovery & Learning")
        header.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #58a6ff;"
        )
        root_layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        root_layout.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(content)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #30363d;
                border-radius: 8px;
                background: #161b22;
            }
            QTabBar::tab {
                background: #21262d;
                color: #8b949e;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background: #161b22;
                color: #58a6ff;
                border-bottom: 2px solid #58a6ff;
            }
            QTabBar::tab:hover:!selected {
                background: #30363d;
                color: #c9d1d9;
            }
        """)

        # Tab 1: Auto Learn
        auto_tab = self._create_auto_tab()
        self.tabs.addTab(auto_tab, "🔄 Auto Learn")

        # Tab 2: Train by Search
        search_tab = self._create_search_tab()
        self.tabs.addTab(search_tab, "🔍 Train by Search")

        layout.addWidget(self.tabs)

        progress_group = QGroupBox("📊 Progress")
        progress_layout = QVBoxLayout()

        self.status_label = QLabel("Ready to start")
        self.status_label.setStyleSheet(
            "font-weight: bold; font-size: 13px;"
        )
        progress_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(25)
        progress_layout.addWidget(self.progress_bar)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        log_group = QGroupBox("📋 Activity Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
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
            "Automatically discover stocks from the market, fetch data, "
            "and train the AI model on new patterns."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #8b949e; font-size: 12px;")
        layout.addWidget(desc)

        settings_group = QGroupBox("⚙️ Settings")
        settings_layout = QGridLayout()
        settings_layout.setSpacing(10)

        settings_layout.addWidget(QLabel("Learning Mode:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Full (Discover + Fetch + Train)",
            "Discovery + Fetch Only",
            "Training Only (use cached data)",
        ])
        self.mode_combo.setMinimumWidth(250)
        settings_layout.addWidget(self.mode_combo, 0, 1)

        settings_layout.addWidget(QLabel("Max Stocks:"), 1, 0)
        self.max_stocks_spin = QSpinBox()
        self.max_stocks_spin.setRange(10, 500)
        self.max_stocks_spin.setValue(50)
        self.max_stocks_spin.setSuffix(" stocks")
        settings_layout.addWidget(self.max_stocks_spin, 1, 1)

        settings_layout.addWidget(QLabel("Epochs:"), 2, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 200)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setSuffix(" epochs")
        settings_layout.addWidget(self.epochs_spin, 2, 1)

        self.discover_check = QCheckBox("Discover new stocks from internet")
        self.discover_check.setChecked(True)
        settings_layout.addWidget(self.discover_check, 3, 0, 1, 2)

        self.incremental_check = QCheckBox(
            "Incremental training (keep existing weights)"
        )
        self.incremental_check.setChecked(False)
        settings_layout.addWidget(self.incremental_check, 4, 0, 1, 2)

        self.use_session_cache_check = QCheckBox(
            "Include stocks captured from real-time UI session"
        )
        self.use_session_cache_check.setChecked(True)
        settings_layout.addWidget(self.use_session_cache_check, 5, 0, 1, 2)

        self.session_seed_label = QLabel("")
        self.session_seed_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        settings_layout.addWidget(self.session_seed_label, 6, 0, 1, 2)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        btn_layout = QHBoxLayout()

        self.auto_start_btn = QPushButton("🚀 Start Auto Learning")
        self.auto_start_btn.setMinimumHeight(45)
        self.auto_start_btn.setStyleSheet(self._green_button_style())
        self.auto_start_btn.clicked.connect(self._start_auto_learning)
        btn_layout.addWidget(self.auto_start_btn)

        self.auto_stop_btn = QPushButton("⏹️ Stop")
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
        desc.setStyleSheet("color: #8b949e; font-size: 12px;")
        layout.addWidget(desc)

        # --- Search section ---
        search_group = QGroupBox("🔍 Search & Add Stocks")
        search_layout = QVBoxLayout()

        search_row = QHBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Enter stock code (e.g., sh600519, sz000001, 000858)..."
        )
        self.search_input.setMinimumHeight(38)
        self.search_input.returnPressed.connect(self._search_stock)
        search_row.addWidget(self.search_input)

        self.search_btn = QPushButton("🔍 Search")
        self.search_btn.setMinimumHeight(38)
        self.search_btn.setMinimumWidth(100)
        self.search_btn.clicked.connect(self._search_stock)
        search_row.addWidget(self.search_btn)

        self.add_btn = QPushButton("➕ Add")
        self.add_btn.setMinimumHeight(38)
        self.add_btn.setMinimumWidth(80)
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(self._add_searched_stock)
        search_row.addWidget(self.add_btn)

        search_layout.addLayout(search_row)

        self.search_result_label = QLabel("")
        self.search_result_label.setStyleSheet(
            "font-size: 12px; padding: 4px;"
        )
        search_layout.addWidget(self.search_result_label)

        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        # --- Stock list section ---
        list_group = QGroupBox("📋 Training Stock List")
        list_layout = QVBoxLayout()

        quick_row = QHBoxLayout()
        quick_label = QLabel("Quick add:")
        quick_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        quick_row.addWidget(quick_label)

        # Popular stocks — codes match CONFIG.STOCK_POOL format (bare 6-digit)
        # but we add sh/sz prefix so they display nicely and the fetcher
        # strips the prefix internally via clean_code() → parse_instrument()
        popular_stocks = [
            ("贵州茅台", "sh600519"),
            ("平安银行", "sz000001"),
            ("五粮液", "sz000858"),
            ("招商银行", "sh600036"),
            ("中国平安", "sh601318"),
        ]
        for name, code in popular_stocks:
            btn = QPushButton(name)
            btn.setToolTip(code)
            btn.setMaximumHeight(28)
            btn.setStyleSheet("""
                QPushButton {
                    background: #21262d;
                    color: #8b949e;
                    border: 1px solid #30363d;
                    border-radius: 4px;
                    padding: 2px 8px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background: #30363d;
                    color: #58a6ff;
                    border-color: #58a6ff;
                }
            """)
            btn.clicked.connect(
                lambda checked, c=code: self._quick_add_stock(c)
            )
            quick_row.addWidget(btn)

        quick_row.addStretch()
        list_layout.addLayout(quick_row)

        self.stock_list = QListWidget()
        self.stock_list.setMinimumHeight(120)
        self.stock_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.stock_list.setStyleSheet("""
            QListWidget {
                background: #0d1117;
                border: 1px solid #30363d;
                border-radius: 6px;
                color: #c9d1d9;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 6px 10px;
                border-bottom: 1px solid #21262d;
            }
            QListWidget::item:selected {
                background: #1f6feb;
                color: white;
            }
            QListWidget::item:hover:!selected {
                background: #161b22;
            }
        """)
        list_layout.addWidget(self.stock_list)

        list_btn_row = QHBoxLayout()

        self.remove_btn = QPushButton("🗑️ Remove Selected")
        self.remove_btn.clicked.connect(self._remove_selected_stocks)
        list_btn_row.addWidget(self.remove_btn)

        self.clear_list_btn = QPushButton("🧹 Clear All")
        self.clear_list_btn.clicked.connect(self._clear_stock_list)
        list_btn_row.addWidget(self.clear_list_btn)

        list_btn_row.addStretch()

        self.stock_count_label = QLabel("0 stocks in list")
        self.stock_count_label.setStyleSheet(
            "color: #8b949e; font-size: 12px;"
        )
        list_btn_row.addWidget(self.stock_count_label)

        list_layout.addLayout(list_btn_row)
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)

        # --- Training settings for targeted ---
        target_settings = QGroupBox("⚙️ Training Settings")
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
        self.target_interval_combo.addItems([
            "1m", "5m", "15m", "30m", "1h", "1d",
        ])
        self.target_interval_combo.setCurrentText("1m")
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

        self.target_start_btn = QPushButton("🎯 Train on Selected Stocks")
        self.target_start_btn.setMinimumHeight(45)
        self.target_start_btn.setStyleSheet(self._blue_button_style())
        self.target_start_btn.clicked.connect(self._start_targeted_learning)
        train_btn_layout.addWidget(self.target_start_btn)

        self.target_stop_btn = QPushButton("⏹️ Stop")
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
        """
        Normalize stock code input.

        The fetcher's clean_code() strips prefixes anyway, but we add
        them here so the UI shows a recognizable format and the code
        survives the round-trip through ParallelFetcher → Trainer.

        Handles: 600519, sh600519, SH600519, 000001, sz000001, SZ.000001
        """
        code = code.strip()

        for ch in (" ", ".", "-", "_"):
            code = code.replace(ch, "")

        code_lower = code.lower()

        # Already has valid prefix
        if code_lower.startswith(("sh", "sz", "bj")):
            return code_lower

        # Pure digits — guess prefix from first digit
        digits = "".join(ch for ch in code if ch.isdigit())
        if len(digits) == 6:
            first = digits[0]
            if first in ("6", "5"):
                return f"sh{digits}"
            elif first in ("0", "3", "1", "2"):
                return f"sz{digits}"

        # Return as-is if we can't normalize
        return code_lower

    def _search_stock(self):
        """Search/validate a stock code."""
        raw = self.search_input.text().strip()
        if not raw:
            self.search_result_label.setText(
                '<span style="color: #d29922;">'
                "Please enter a stock code"
                "</span>"
            )
            return

        code = self._normalize_code(raw)

        if code in self._targeted_stock_codes:
            self.search_result_label.setText(
                f'<span style="color: #d29922;">'
                f"⚠️ {code} is already in the list"
                f"</span>"
            )
            return

        self.search_btn.setEnabled(False)
        self.search_btn.setText("🔄 Searching...")
        self.add_btn.setEnabled(False)
        self.search_result_label.setText(
            f'<span style="color: #8b949e;">'
            f"Validating {code}..."
            f"</span>"
        )

        interval = self.target_interval_combo.currentText()
        self._validation_request_id += 1
        self._validator = StockValidatorWorker(
            code, interval, request_id=self._validation_request_id
        )
        self._validator.validation_result.connect(self._on_validation_result)
        self._validator.start()

    def _on_validation_result(self, result: dict):
        """Handle stock validation result from background thread."""
        request_id = int(result.get("request_id", 0) or 0)
        if request_id != self._validation_request_id:
            # Ignore stale async result from an older search action.
            return

        # Re-enable search button
        self.search_btn.setEnabled(True)
        self.search_btn.setText("🔍 Search")

        code = result.get("code", "")
        valid = result.get("valid", False)
        name = result.get("name", "")
        bars = result.get("bars", 0)
        message = result.get("message", "")

        if valid:
            display = code
            if name:
                display += f" ({name})"
            display += f" — {bars} bars"

            self.search_result_label.setText(
                f'<span style="color: #3fb950;">✅ {display}</span>'
            )
            self.add_btn.setEnabled(True)

            self._last_validated_code = code
            self._last_validated_name = name
            self._last_validated_bars = bars
        else:
            self.search_result_label.setText(
                f'<span style="color: #f85149;">❌ {code}: {message}</span>'
            )
            self.add_btn.setEnabled(False)
            self._last_validated_code = ""

        self._validator = None

    def _add_searched_stock(self):
        """Add the last validated stock to the training list."""
        code = self._last_validated_code
        if not code:
            return

        if code in self._targeted_stock_codes:
            self.search_result_label.setText(
                f'<span style="color: #d29922;">'
                f"⚠️ {code} already in list"
                f"</span>"
            )
            return

        self._targeted_stock_codes.append(code)
        self._add_stock_to_list_widget(
            code, self._last_validated_name, self._last_validated_bars
        )

        self.search_input.clear()
        self.search_result_label.setText(
            f'<span style="color: #3fb950;">'
            f"✅ Added {code} to training list"
            f"</span>"
        )
        self.add_btn.setEnabled(False)
        self._last_validated_code = ""
        self._update_stock_count()

    def _quick_add_stock(self, code: str):
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
    ):
        """Add a stock item to the QListWidget."""
        display = f"  {code}"
        if name:
            display += f"  —  {name}"
        if bars > 0:
            display += f"  ({bars} bars)"

        item = QListWidgetItem(display)
        item.setData(Qt.ItemDataRole.UserRole, code)
        self.stock_list.addItem(item)

    def _remove_selected_stocks(self):
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

    def _clear_stock_list(self):
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

    def _update_stock_count(self):
        """Update the stock count label."""
        count = len(self._targeted_stock_codes)
        if count == 0:
            self.stock_count_label.setText("0 stocks in list")
            self.stock_count_label.setStyleSheet(
                "color: #8b949e; font-size: 12px;"
            )
        else:
            suffix = "s" if count != 1 else ""
            self.stock_count_label.setText(
                f"{count} stock{suffix} in list"
            )
            self.stock_count_label.setStyleSheet(
                "color: #3fb950; font-size: 12px;"
            )

    def _load_seed_stocks(self):
        """Preload targeted list from session-captured symbols."""
        if not self._seed_stock_codes:
            self.session_seed_label.setText("Session seed stocks: 0")
            return

        added = 0
        for code in self._seed_stock_codes:
            if code in self._targeted_stock_codes:
                continue
            self._targeted_stock_codes.append(code)
            self._add_stock_to_list_widget(code, "Session cache", 0)
            added += 1

        self._update_stock_count()
        self.session_seed_label.setText(
            f"Session seed stocks: {len(self._seed_stock_codes)} (added {added})"
        )

    def _collect_priority_codes(self, mode: str = "auto") -> list[str]:
        codes = list(self._seed_stock_codes)
        try:
            from data.session_cache import get_session_bar_cache
            cache = get_session_bar_cache()
            interval = "1m"
            try:
                interval = self.target_interval_combo.currentText().strip().lower()
            except Exception:
                interval = "1m"
            live_codes = cache.get_recent_symbols(interval=interval, min_rows=10)
            codes.extend(live_codes)
        except Exception:
            pass

        dedup = []
        seen = set()
        for code in codes:
            c = str(code).strip()
            if not c or c in seen:
                continue
            seen.add(c)
            dedup.append(c)
        return dedup

    # =========================================================================
    # AUTO LEARN START/STOP
    # =========================================================================

    def _start_auto_learning(self):
        """Start auto-learning (random rotation)."""
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

        mode_map = {0: "full", 1: "discovery", 2: "training"}

        config = {
            "mode": mode_map.get(self.mode_combo.currentIndex(), "full"),
            "discover_new": self.discover_check.isChecked(),
            "max_stocks": self.max_stocks_spin.value(),
            "epochs": self.epochs_spin.value(),
            "incremental": self.incremental_check.isChecked(),
        }

        if self.use_session_cache_check.isChecked():
            priority_codes = self._collect_priority_codes(mode="auto")
            if priority_codes:
                config["priority_stock_codes"] = priority_codes
                self._log(
                    f"Session cache boost: {len(priority_codes)} priority stocks",
                    "info",
                )

        # VPN mode is more fragile for large Yahoo-heavy batches.
        try:
            from core.network import get_network_env
            env = get_network_env()
            if env.is_vpn_active and config["max_stocks"] > 30:
                config["max_stocks"] = 30
                self.max_stocks_spin.setValue(30)
                self._log(
                    "VPN detected: Max stocks auto-capped to 30 for stability",
                    "warning",
                )
        except Exception:
            pass

        self._log("Starting auto-learning...", "info")
        self._log(
            f"Mode: {config['mode']}, "
            f"Max stocks: {config['max_stocks']}, "
            f"Epochs: {config['epochs']}",
            "info",
        )

        self._set_running(True, mode="auto")

        self.worker = AutoLearnWorker(config)
        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._log)
        self.worker.finished_result.connect(self._on_auto_finished)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

    def _stop_auto_learning(self):
        """Stop auto-learning gracefully."""
        if not self._is_running:
            return

        self._log("Stopping auto-learning...", "warning")
        self.status_label.setText("Stopping...")
        self.auto_stop_btn.setEnabled(False)

        if self.worker:
            self.worker.stop()
            if not self.worker.wait(20000):
                self._log("Worker did not stop cleanly", "warning")
            self.worker = None

        self._log("Auto-learning stopped by user", "warning")
        self._set_running(False)

    # =========================================================================
    # TARGETED LEARN START/STOP
    # =========================================================================

    def _start_targeted_learning(self):
        """Start training on user-selected stocks."""
        if self._is_running:
            return

        if not self._targeted_stock_codes:
            QMessageBox.warning(
                self,
                "No Stocks Selected",
                "Please add at least one stock to the training list.\n\n"
                "Use the search bar or quick-add buttons above.",
            )
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

        # Verify start_targeted exists
        if not hasattr(LearnerClass, "start_targeted"):
            QMessageBox.critical(
                self,
                "Feature Not Available",
                "The current AutoLearner does not support targeted training.\n\n"
                "Please update models/auto_learner.py with the latest version.",
            )
            return

        config = {
            "stock_codes": list(self._targeted_stock_codes),
            "epochs": self.target_epochs_spin.value(),
            "interval": self.target_interval_combo.currentText(),
            "horizon": self.target_horizon_spin.value(),
            "incremental": self.target_incremental_check.isChecked(),
            "continuous": False,
        }

        stock_display = ", ".join(self._targeted_stock_codes[:5])
        if len(self._targeted_stock_codes) > 5:
            stock_display += (
                f"... (+{len(self._targeted_stock_codes) - 5} more)"
            )

        self._log(
            f"Starting targeted training on: {stock_display}", "info"
        )
        self._log(
            f"Epochs: {config['epochs']}, "
            f"Interval: {config['interval']}, "
            f"Horizon: {config['horizon']}, "
            f"Incremental: {config['incremental']}",
            "info",
        )

        self._set_running(True, mode="targeted")

        self.targeted_worker = TargetedLearnWorker(config)
        self.targeted_worker.progress.connect(self._on_progress)
        self.targeted_worker.log_message.connect(self._log)
        self.targeted_worker.finished_result.connect(
            self._on_targeted_finished
        )
        self.targeted_worker.error_occurred.connect(self._on_error)
        self.targeted_worker.start()

    def _stop_targeted_learning(self):
        """Stop targeted training gracefully."""
        if not self._is_running:
            return

        self._log("Stopping targeted training...", "warning")
        self.status_label.setText("Stopping...")
        self.target_stop_btn.setEnabled(False)

        if self.targeted_worker:
            self.targeted_worker.stop()
            if not self.targeted_worker.wait(20000):
                self._log(
                    "Targeted worker did not stop cleanly", "warning"
                )
            self.targeted_worker = None

        self._log("Targeted training stopped by user", "warning")
        self._set_running(False)

    # =========================================================================
    # =========================================================================

    def _set_running(self, running: bool, mode: str = ""):
        """
        Set UI running state.
        Disables all interactive controls while training is active.
        """
        self._is_running = running
        self._active_mode = mode if running else ""

        if running:
            self.progress_bar.setValue(0)
            self.status_label.setText("Initializing...")
            self.close_btn.setEnabled(False)

            # Disable BOTH tabs' start buttons (only one can run)
            self.auto_start_btn.setEnabled(False)
            self.target_start_btn.setEnabled(False)

            if mode == "auto":
                self.auto_stop_btn.setEnabled(True)
                self.target_stop_btn.setEnabled(False)
                self.mode_combo.setEnabled(False)
                self.max_stocks_spin.setEnabled(False)
                self.epochs_spin.setEnabled(False)
                self.discover_check.setEnabled(False)
                self.incremental_check.setEnabled(False)
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
            self.target_start_btn.setEnabled(True)
            self.auto_stop_btn.setEnabled(False)
            self.target_stop_btn.setEnabled(False)

            self.mode_combo.setEnabled(True)
            self.max_stocks_spin.setEnabled(True)
            self.epochs_spin.setEnabled(True)
            self.discover_check.setEnabled(True)
            self.incremental_check.setEnabled(True)
            self.use_session_cache_check.setEnabled(True)

            self.search_input.setEnabled(True)
            self.search_btn.setEnabled(True)
            # add_btn stays disabled until next search
            self.remove_btn.setEnabled(True)
            self.clear_list_btn.setEnabled(True)
            self.target_epochs_spin.setEnabled(True)
            self.target_interval_combo.setEnabled(True)
            self.target_horizon_spin.setEnabled(True)
            self.target_incremental_check.setEnabled(True)

            self.worker = None
            self.targeted_worker = None

    # =========================================================================
    # =========================================================================

    def _on_progress(self, percent: int, message: str):
        """Handle progress update from either worker."""
        self.progress_bar.setValue(percent)
        self.status_label.setText(str(message))

    def _on_auto_finished(self, results: dict):
        """Handle auto-learning completion."""
        self._log("=" * 50, "info")
        status = results.get("status", "ok")
        if status == "error":
            err = str(results.get("error") or "Auto-learning failed")
            self._log(f"Auto-learning failed: {err}", "error")
            self._log_results(results)
            self._set_running(False)
            self.status_label.setText("Auto-learning failed")
            self.progress_bar.setValue(100)
            QMessageBox.critical(
                self,
                "Learning Failed",
                f"Auto-learning failed:\n\n{err}",
            )
            return

        self._log("🎉 Auto-learning completed!", "success")
        self._log_results(results)

        self._set_running(False)
        self.status_label.setText("✅ Auto-learning completed!")
        self.progress_bar.setValue(100)

        QMessageBox.information(
            self,
            "Learning Complete",
            f"Auto-learning completed!\n\n"
            f"Stocks discovered: {results.get('discovered', 0)}\n"
            f"Stocks processed: {results.get('processed', 0)}\n"
            f"Accuracy: {results.get('accuracy', 0):.1%}",
        )

    def _on_targeted_finished(self, results: dict):
        """Handle targeted training completion."""
        self._log("=" * 50, "info")
        self._log("🎉 Targeted training completed!", "success")
        self._log_results(results)

        stocks = results.get("stocks_trained", [])
        if stocks:
            display = ", ".join(stocks[:10])
            if len(stocks) > 10:
                display += f"... (+{len(stocks) - 10})"
            self._log(f"Trained on: {display}", "info")

        self._set_running(False)
        self.status_label.setText("✅ Targeted training completed!")
        self.progress_bar.setValue(100)

        QMessageBox.information(
            self,
            "Training Complete",
            f"Targeted training completed!\n\n"
            f"Stocks trained: {len(stocks)}\n"
            f"Stocks processed: {results.get('processed', 0)}\n"
            f"Accuracy: {results.get('accuracy', 0):.1%}",
        )

    def _on_error(self, error: str):
        """Handle error from either worker."""
        display_error = error[:300] if len(error) > 300 else error
        self._log(f"Error: {display_error}", "error")

        self._set_running(False)
        self.status_label.setText("❌ Error occurred")

        QMessageBox.critical(
            self,
            "Error",
            f"An error occurred during learning:\n\n{error[:500]}",
        )

    def _log_results(self, results: dict):
        """Log training results to the activity log."""
        discovered = int(results.get("discovered", 0))
        processed = int(results.get("processed", 0))
        accuracy = float(results.get("accuracy", 0))

        if discovered > 0:
            self._log(f"📊 Stocks discovered: {discovered}", "info")
        if processed > 0:
            self._log(f"📈 Stocks processed: {processed}", "info")
        if accuracy > 0:
            self._log(f"🎯 Model accuracy: {accuracy:.1%}", "success")

    # =========================================================================
    # =========================================================================

    def _log(self, message: str, level: str = "info"):
        """Add message to activity log with timestamp and color."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        colors = {
            "info": "#c9d1d9",
            "success": "#3fb950",
            "warning": "#d29922",
            "error": "#f85149",
        }
        color = colors.get(level, "#c9d1d9")

        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
        }
        icon = icons.get(level, "")

        self.log_text.append(
            f'<span style="color: #6e7681;">[{timestamp}]</span> '
            f'{icon} <span style="color: {color};">{message}</span>'
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
                    stop:0 #238636, stop:1 #2ea043);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2ea043, stop:1 #3fb950);
            }
            QPushButton:disabled {
                background: #21262d;
                color: #484f58;
            }
        """

    @staticmethod
    def _blue_button_style() -> str:
        return """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1f6feb, stop:1 #388bfd);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #388bfd, stop:1 #58a6ff);
            }
            QPushButton:disabled {
                background: #21262d;
                color: #484f58;
            }
        """

    # =========================================================================
    # =========================================================================

    def _apply_style(self):
        self.setStyleSheet("""
            QDialog {
                background: #0d1117;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #30363d;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 15px;
                color: #58a6ff;
                background: #161b22;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
            QLabel {
                color: #c9d1d9;
            }
            QComboBox, QSpinBox {
                padding: 8px;
                border: 1px solid #30363d;
                border-radius: 6px;
                background: #21262d;
                color: #c9d1d9;
                min-width: 120px;
            }
            QComboBox:focus, QSpinBox:focus {
                border-color: #58a6ff;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #30363d;
                border-radius: 6px;
                background: #21262d;
                color: #c9d1d9;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: #58a6ff;
            }
            QCheckBox {
                color: #c9d1d9;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #30363d;
                background: #21262d;
            }
            QCheckBox::indicator:checked {
                background: #238636;
                border-color: #238636;
            }
            QProgressBar {
                border: none;
                background: #21262d;
                border-radius: 6px;
                text-align: center;
                color: #c9d1d9;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #238636, stop:1 #2ea043);
                border-radius: 6px;
            }
            QTextEdit {
                background: #0d1117;
                color: #7ee787;
                border: 1px solid #30363d;
                border-radius: 6px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
            QPushButton {
                background: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #30363d;
                border-color: #58a6ff;
            }
            QPushButton:disabled {
                background: #161b22;
                color: #484f58;
            }
        """)

    # =========================================================================
    # =========================================================================

    def closeEvent(self, event):
        """Handle close — stop any running training first."""
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
                event.accept()
            else:
                event.ignore()
                return
        else:
            if self._validator:
                self._validator.wait(2000)
                self._validator = None
            event.accept()

        super().closeEvent(event)

def show_auto_learn_dialog(parent=None, seed_stock_codes: list[str] | None = None):
    """Show the auto-learn dialog — convenience function."""
    dialog = AutoLearnDialog(parent, seed_stock_codes=seed_stock_codes)
    dialog.exec()
    return dialog
