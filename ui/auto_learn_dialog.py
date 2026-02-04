"""
Auto-Learning Dialog
GUI for automatic AI learning with real-time progress
"""
from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QProgressBar, QTextEdit,
    QSpinBox, QCheckBox, QGroupBox, QFrame,
    QTabWidget, QWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor

from config import CONFIG
from utils.logger import log


class LearningWorker(QThread):
    """Background worker for auto-learning"""
    progress_update = pyqtSignal(str, float, str)  # stage, progress%, message
    log_message = pyqtSignal(str, str)  # message, level
    finished = pyqtSignal(bool, str)  # success, message
    stats_update = pyqtSignal(int, int, float)  # stocks_found, stocks_processed, accuracy
    
    def __init__(self, auto_search: bool, max_stocks: int, epochs: int, incremental: bool):
        super().__init__()
        self.auto_search = auto_search
        self.max_stocks = max_stocks
        self.epochs = epochs
        self.incremental = incremental
        self._stop_requested = False
    
    def stop(self):
        self._stop_requested = True
    
    def run(self):
        try:
            from models.auto_learner import AutoLearner, LearningProgress
            
            learner = AutoLearner()
            
            # Custom callback to emit signals
            def on_progress(progress: LearningProgress):
                if self._stop_requested:
                    return
                
                self.progress_update.emit(
                    progress.stage,
                    progress.progress,
                    progress.message
                )
                
                self.stats_update.emit(
                    progress.stocks_found,
                    progress.stocks_processed,
                    progress.training_accuracy
                )
                
                for error in progress.errors:
                    self.log_message.emit(f"⚠️ {error}", "warning")
            
            learner.add_callback(on_progress)
            
            self.log_message.emit("🚀 Starting auto-learning process...", "info")
            
            # Start learning (blocking in this thread)
            learner.start_learning(
                auto_search=self.auto_search,
                max_stocks=self.max_stocks,
                epochs=self.epochs,
                incremental=self.incremental
            )
            
            # Wait for completion
            import time
            while learner.progress.is_running and not self._stop_requested:
                time.sleep(0.5)
            
            if self._stop_requested:
                learner.stop_learning()
                self.finished.emit(False, "Learning stopped by user")
            elif learner.progress.stage == 'complete':
                self.finished.emit(True, f"Training complete! Accuracy: {learner.progress.training_accuracy:.1%}")
            else:
                self.finished.emit(False, f"Learning failed: {learner.progress.message}")
                
        except Exception as e:
            self.log_message.emit(f"❌ Error: {str(e)}", "error")
            self.finished.emit(False, str(e))


class AutoLearnDialog(QDialog):
    """
    Dialog for auto-learning AI model
    
    Features:
    - One-click learning from internet data
    - Real-time progress visualization
    - Learning history tracking
    - Customizable settings
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤖 AI Auto-Learning System")
        self.setMinimumSize(900, 700)
        
        self.worker: Optional[LearningWorker] = None
        
        self._setup_ui()
        self._apply_style()
        self._load_history()
    
    def _setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QLabel("🤖 AI Auto-Learning System")
        header.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #00E5FF; padding: 10px;")
        layout.addWidget(header)
        
        # Description
        desc = QLabel(
            "Automatically search the internet for stocks, download data, "
            "and train AI models with one click."
        )
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("color: #888; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._create_learn_tab(), "🚀 Quick Learn")
        tabs.addTab(self._create_settings_tab(), "⚙️ Settings")
        tabs.addTab(self._create_history_tab(), "📊 History")
        layout.addWidget(tabs)
        
        # Progress Section
        progress_group = QGroupBox("Learning Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Stage indicator
        stage_row = QHBoxLayout()
        self.stage_label = QLabel("Status: Ready")
        self.stage_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        stage_row.addWidget(self.stage_label)
        stage_row.addStretch()
        self.time_label = QLabel("")
        self.time_label.setStyleSheet("color: #888;")
        stage_row.addWidget(self.time_label)
        progress_layout.addLayout(stage_row)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        progress_layout.addWidget(self.progress_bar)
        
        # Current action
        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("color: #aaa;")
        progress_layout.addWidget(self.message_label)
        
        # Stats row
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background: #1a1a3e;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        stats_layout = QHBoxLayout(stats_frame)
        
        self.stocks_found_label = self._create_stat_widget("Stocks Found", "0")
        self.stocks_processed_label = self._create_stat_widget("Processed", "0")
        self.accuracy_label = self._create_stat_widget("Accuracy", "--")
        
        stats_layout.addWidget(self.stocks_found_label)
        stats_layout.addWidget(self.stocks_processed_label)
        stats_layout.addWidget(self.accuracy_label)
        
        progress_layout.addWidget(stats_frame)
        layout.addWidget(progress_group)
        
        # Log section
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background: #0a0a1a;
                color: #0f0;
                border: 1px solid #2a2a5a;
                border-radius: 5px;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("🚀 Start Learning")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00E5FF, stop:1 #00BCD4);
                color: white;
                border: none;
                padding: 15px 50px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00BCD4, stop:1 #0097A7);
            }
            QPushButton:disabled {
                background: #333;
                color: #666;
            }
        """)
        self.start_btn.clicked.connect(self._start_learning)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #F44336;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #D32F2F;
            }
            QPushButton:disabled {
                background: #333;
                color: #666;
            }
        """)
        self.stop_btn.clicked.connect(self._stop_learning)
        btn_layout.addWidget(self.stop_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)
        
        layout.addLayout(btn_layout)
        
        # Timer for elapsed time
        self.elapsed_timer = QTimer()
        self.elapsed_timer.timeout.connect(self._update_elapsed_time)
        self.start_time = None
    
    def _create_stat_widget(self, title: str, value: str) -> QWidget:
        """Create a statistics display widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 5, 10, 5)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #888; font-size: 11px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        value_label = QLabel(value)
        value_label.setStyleSheet("color: #00E5FF; font-size: 18px; font-weight: bold;")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setObjectName("value")
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        
        return widget
    
    def _create_learn_tab(self) -> QWidget:
        """Create the quick learn tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Steps explanation
        steps_group = QGroupBox("Learning Process")
        steps_layout = QVBoxLayout(steps_group)
        
        steps = [
            ("1. 🔍 Search Internet", "Find trending stocks from financial websites"),
            ("2. 📥 Download Data", "Get historical price data for selected stocks"),
            ("3. 🧮 Create Features", "Calculate 80+ technical indicators"),
            ("4. 🧠 Train Models", "Train 6 neural network models"),
            ("5. ✅ Save & Validate", "Save best model and validate performance"),
        ]
        
        for title, desc in steps:
            step_row = QHBoxLayout()
            title_label = QLabel(title)
            title_label.setStyleSheet("color: #00E5FF; font-weight: bold; min-width: 150px;")
            desc_label = QLabel(desc)
            desc_label.setStyleSheet("color: #888;")
            step_row.addWidget(title_label)
            step_row.addWidget(desc_label)
            step_row.addStretch()
            steps_layout.addLayout(step_row)
        
        layout.addWidget(steps_group)
        
        # Quick settings
        settings_group = QGroupBox("Quick Settings")
        settings_layout = QGridLayout(settings_group)
        
        settings_layout.addWidget(QLabel("Training Epochs:"), 0, 0)
        self.quick_epochs = QSpinBox()
        self.quick_epochs.setRange(30, 300)
        self.quick_epochs.setValue(100)
        self.quick_epochs.setToolTip("Number of training iterations. More = better but slower.")
        settings_layout.addWidget(self.quick_epochs, 0, 1)
        
        settings_layout.addWidget(QLabel("Max Stocks:"), 0, 2)
        self.quick_stocks = QSpinBox()
        self.quick_stocks.setRange(10, 200)
        self.quick_stocks.setValue(50)
        self.quick_stocks.setToolTip("Maximum number of stocks to include in training.")
        settings_layout.addWidget(self.quick_stocks, 0, 3)
        
        self.quick_search = QCheckBox("Search internet for stocks")
        self.quick_search.setChecked(True)
        self.quick_search.setToolTip("If unchecked, uses default stock pool from config.")
        settings_layout.addWidget(self.quick_search, 1, 0, 1, 2)
        
        self.quick_incremental = QCheckBox("Incremental learning (keep old knowledge)")
        self.quick_incremental.setChecked(False)
        self.quick_incremental.setToolTip("Continue training from existing model instead of starting fresh.")
        settings_layout.addWidget(self.quick_incremental, 1, 2, 1, 2)
        
        layout.addWidget(settings_group)
        
        # Estimated time
        time_label = QLabel("⏱️ Estimated time: 30-60 minutes depending on settings and network speed")
        time_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(time_label)
        
        layout.addStretch()
        
        return widget
    
    def _create_settings_tab(self) -> QWidget:
        """Create the settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Training parameters
        train_group = QGroupBox("Training Parameters")
        train_layout = QGridLayout(train_group)
        
        train_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 500)
        self.epochs_spin.setValue(CONFIG.EPOCHS)
        train_layout.addWidget(self.epochs_spin, 0, 1)
        
        train_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(16, 256)
        self.batch_spin.setValue(CONFIG.BATCH_SIZE)
        train_layout.addWidget(self.batch_spin, 0, 3)
        
        train_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lr_label = QLabel(f"{CONFIG.LEARNING_RATE}")
        train_layout.addWidget(self.lr_label, 1, 1)
        
        train_layout.addWidget(QLabel("Sequence Length:"), 1, 2)
        self.seq_label = QLabel(f"{CONFIG.SEQUENCE_LENGTH} days")
        train_layout.addWidget(self.seq_label, 1, 3)
        
        layout.addWidget(train_group)
        
        # Data sources
        data_group = QGroupBox("Data Sources to Search")
        data_layout = QVBoxLayout(data_group)
        
        self.source_gainers = QCheckBox("Top Gainers (涨幅榜)")
        self.source_gainers.setChecked(True)
        data_layout.addWidget(self.source_gainers)
        
        self.source_losers = QCheckBox("Top Losers (跌幅榜)")
        self.source_losers.setChecked(True)
        data_layout.addWidget(self.source_losers)
        
        self.source_volume = QCheckBox("High Volume (成交额榜)")
        self.source_volume.setChecked(True)
        data_layout.addWidget(self.source_volume)
        
        self.source_hot = QCheckBox("Hot Stocks (热门股票)")
        self.source_hot.setChecked(True)
        data_layout.addWidget(self.source_hot)
        
        self.source_analyst = QCheckBox("Analyst Picks (机构推荐)")
        self.source_analyst.setChecked(True)
        data_layout.addWidget(self.source_analyst)
        
        layout.addWidget(data_group)
        
        # Model selection
        model_group = QGroupBox("Models to Train")
        model_layout = QVBoxLayout(model_group)
        
        self.model_lstm = QCheckBox("LSTM with Attention")
        self.model_lstm.setChecked(True)
        model_layout.addWidget(self.model_lstm)
        
        self.model_transformer = QCheckBox("Transformer")
        self.model_transformer.setChecked(True)
        model_layout.addWidget(self.model_transformer)
        
        self.model_gru = QCheckBox("GRU")
        self.model_gru.setChecked(True)
        model_layout.addWidget(self.model_gru)
        
        self.model_tcn = QCheckBox("TCN (Temporal Convolutional)")
        self.model_tcn.setChecked(True)
        model_layout.addWidget(self.model_tcn)
        
        layout.addWidget(model_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_history_tab(self) -> QWidget:
        """Create the history tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Summary stats
        summary_frame = QFrame()
        summary_frame.setStyleSheet("""
            QFrame {
                background: #1a1a3e;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        summary_layout = QHBoxLayout(summary_frame)
        
        self.total_sessions_label = self._create_stat_widget("Total Sessions", "0")
        self.best_accuracy_label = self._create_stat_widget("Best Accuracy", "0%")
        self.total_stocks_label = self._create_stat_widget("Stocks Trained", "0")
        
        summary_layout.addWidget(self.total_sessions_label)
        summary_layout.addWidget(self.best_accuracy_label)
        summary_layout.addWidget(self.total_stocks_label)
        
        layout.addWidget(summary_frame)
        
        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            "Date/Time", "Stocks", "Samples", "Epochs", "Accuracy", "Duration"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setAlternatingRowColors(True)
        layout.addWidget(self.history_table)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh History")
        refresh_btn.clicked.connect(self._load_history)
        layout.addWidget(refresh_btn)
        
        return widget
    
    def _load_history(self):
        """Load learning history from file"""
        try:
            import json
            history_path = CONFIG.DATA_DIR / "learning_history.json"
            
            if not history_path.exists():
                return
            
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # Update summary
            sessions = history.get('sessions', [])
            best_acc = history.get('best_accuracy', 0)
            total_stocks = history.get('total_stocks', 0)
            
            self._update_stat_label(self.total_sessions_label, str(len(sessions)))
            self._update_stat_label(self.best_accuracy_label, f"{best_acc*100:.1f}%")
            self._update_stat_label(self.total_stocks_label, str(total_stocks))
            
            # Populate table
            self.history_table.setRowCount(len(sessions))
            
            for i, session in enumerate(reversed(sessions)):
                self.history_table.setItem(i, 0, QTableWidgetItem(
                    session.get('timestamp', '')[:16]
                ))
                self.history_table.setItem(i, 1, QTableWidgetItem(
                    str(session.get('stocks_used', 0))
                ))
                self.history_table.setItem(i, 2, QTableWidgetItem(
                    str(session.get('samples', 0))
                ))
                self.history_table.setItem(i, 3, QTableWidgetItem(
                    str(session.get('epochs', 0))
                ))
                
                acc = session.get('test_accuracy', 0) * 100
                acc_item = QTableWidgetItem(f"{acc:.1f}%")
                if acc > 55:
                    acc_item.setForeground(QColor("#4CAF50"))
                elif acc < 45:
                    acc_item.setForeground(QColor("#FF5252"))
                self.history_table.setItem(i, 4, acc_item)
                
                self.history_table.setItem(i, 5, QTableWidgetItem(
                    f"{session.get('duration_minutes', 0):.1f} min"
                ))
                
        except Exception as e:
            self._log(f"Failed to load history: {e}", "warning")
    
    def _update_stat_label(self, widget: QWidget, value: str):
        """Update the value in a stat widget"""
        value_label = widget.findChild(QLabel, "value")
        if value_label:
            value_label.setText(value)
    
    def _apply_style(self):
        """Apply dark theme styling"""
        self.setStyleSheet("""
            QDialog {
                background: #0a0a1a;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2a2a5a;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 12px;
                color: #00E5FF;
                background: #0f0f2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
            QLabel {
                color: #ddd;
            }
            QSpinBox, QCheckBox {
                color: #fff;
                background: #1a1a3e;
                border: 1px solid #2a2a5a;
                border-radius: 4px;
                padding: 5px;
            }
            QProgressBar {
                border: none;
                background: #1a1a3e;
                border-radius: 8px;
                text-align: center;
                color: #fff;
                height: 30px;
                font-size: 14px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00E5FF, stop:1 #00BCD4);
                border-radius: 8px;
            }
            QTableWidget {
                background: #1a1a3e;
                color: #fff;
                border: none;
                gridline-color: #2a2a5a;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background: #2a2a5a;
                color: #00E5FF;
                padding: 10px;
                border: none;
                font-weight: bold;
            }
            QPushButton {
                background: #3a3a7a;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #4a4a9a;
            }
            QPushButton:disabled {
                background: #222;
                color: #555;
            }
            QTabWidget::pane {
                border: 2px solid #2a2a5a;
                background: #0a0a1a;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #1a1a3e;
                color: #888;
                padding: 12px 25px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #2a2a5a;
                color: #00E5FF;
            }
        """)
    
    def _start_learning(self):
        """Start the auto-learning process"""
        reply = QMessageBox.question(
            self,
            "Start Learning",
            "The AI will automatically:\n\n"
            "1. Search the internet for stocks\n"
            "2. Download historical data\n"
            "3. Train neural network models\n\n"
            "This may take 30-60 minutes.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Update UI state
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_text.clear()
        self.progress_bar.setValue(0)
        
        # Start timer
        self.start_time = datetime.now()
        self.elapsed_timer.start(1000)
        
        # Create and start worker
        self.worker = LearningWorker(
            auto_search=self.quick_search.isChecked(),
            max_stocks=self.quick_stocks.value(),
            epochs=self.quick_epochs.value(),
            incremental=self.quick_incremental.isChecked()
        )
        
        self.worker.progress_update.connect(self._on_progress)
        self.worker.log_message.connect(self._log)
        self.worker.stats_update.connect(self._on_stats)
        self.worker.finished.connect(self._on_finished)
        
        self.worker.start()
    
    def _stop_learning(self):
        """Stop the learning process"""
        if self.worker:
            self._log("⏹️ Stopping learning...", "warning")
            self.worker.stop()
    
    def _on_progress(self, stage: str, progress: float, message: str):
        """Handle progress update from worker"""
        stage_text = {
            'idle': '⏸️ Idle',
            'searching': '🔍 Searching',
            'downloading': '📥 Downloading',
            'preparing': '🧮 Preparing',
            'training': '🧠 Training',
            'evaluating': '📊 Evaluating',
            'complete': '✅ Complete',
            'error': '❌ Error'
        }
        
        self.stage_label.setText(f"Status: {stage_text.get(stage, stage)}")
        self.progress_bar.setValue(int(progress))
        self.message_label.setText(message)
        
        # Color based on stage
        if stage == 'complete':
            self.stage_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif stage == 'error':
            self.stage_label.setStyleSheet("color: #FF5252; font-weight: bold;")
        elif stage == 'training':
            self.stage_label.setStyleSheet("color: #FFD54F; font-weight: bold;")
        else:
            self.stage_label.setStyleSheet("color: #00E5FF; font-weight: bold;")
    
    def _on_stats(self, stocks_found: int, stocks_processed: int, accuracy: float):
        """Handle stats update"""
        self._update_stat_label(self.stocks_found_label, str(stocks_found))
        self._update_stat_label(self.stocks_processed_label, str(stocks_processed))
        
        if accuracy > 0:
            self._update_stat_label(self.accuracy_label, f"{accuracy:.1%}")
    
    def _on_finished(self, success: bool, message: str):
        """Handle learning completion"""
        self.elapsed_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if success:
            self._log(f"✅ {message}", "success")
            self._load_history()
            QMessageBox.information(self, "Success", message)
        else:
            self._log(f"❌ {message}", "error")
            if "stopped" not in message.lower():
                QMessageBox.warning(self, "Learning Failed", message)
    
    def _update_elapsed_time(self):
        """Update elapsed time display"""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            minutes = int(elapsed.total_seconds() // 60)
            seconds = int(elapsed.total_seconds() % 60)
            self.time_label.setText(f"Elapsed: {minutes:02d}:{seconds:02d}")
    
    def _log(self, message: str, level: str = "info"):
        """Add log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        colors = {
            "info": "#7ee787",
            "warning": "#d29922",
            "error": "#f85149",
            "success": "#3fb950"
        }
        color = colors.get(level, "#c9d1d9")
        
        self.log_text.append(
            f'<span style="color:#484f58">[{timestamp}]</span> '
            f'<span style="color:{color}">{message}</span>'
        )
        
        # Auto-scroll
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """Handle dialog close"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Learning in Progress",
                "Learning is still in progress. Stop and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                self.worker.wait(5000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def show_auto_learn_dialog(parent=None):
    """Show the auto-learn dialog"""
    dialog = AutoLearnDialog(parent)
    dialog.exec()