"""
Auto-Learning Dialog
GUI for automatic AI learning with real-time progress tracking
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


class AutoLearnWorker(QThread):
    """Background worker for auto-learning"""
    progress_updated = pyqtSignal(object)
    log_message = pyqtSignal(str, str)  # message, level
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings
        self._stop_requested = False
    
    def run(self):
        """Main learning loop"""
        try:
            from models.auto_learner import AutoLearner, LearningProgress
            
            learner = AutoLearner()
            
            def on_progress(progress: LearningProgress):
                if self._stop_requested:
                    return
                self.progress_updated.emit(progress)
                
                if progress.message:
                    level = "error" if "error" in progress.stage.lower() else "info"
                    self.log_message.emit(progress.message, level)
            
            learner.add_callback(on_progress)
            
            # Start learning in this thread (blocking)
            learner.start_learning(
                auto_search=self.settings.get('auto_search', True),
                max_stocks=self.settings.get('max_stocks', 80),
                epochs=self.settings.get('epochs', 100),
                incremental=self.settings.get('incremental', True)
            )
            
            # Wait for completion
            import time
            while learner.progress.is_running and not self._stop_requested:
                time.sleep(0.5)
            
            if self._stop_requested:
                self.finished.emit(False, "Learning stopped by user")
            elif learner.progress.stage == 'complete':
                self.finished.emit(True, f"Training complete! Accuracy: {learner.progress.training_accuracy:.1%}")
            else:
                self.finished.emit(False, f"Training failed: {learner.progress.message}")
                
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")
    
    def stop(self):
        """Request stop"""
        self._stop_requested = True


class AutoLearnDialog(QDialog):
    """
    Dialog for auto-learning AI model
    
    Features:
    - One-click learning
    - Progress visualization
    - Learning history
    - Settings customization
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤖 AI Auto-Learning System")
        self.setMinimumSize(900, 700)
        
        self.worker: Optional[AutoLearnWorker] = None
        self._progress_stage = "idle"
        self._progress_pct = 0
        self._progress_message = ""
        
        self._setup_ui()
        self._apply_style()
    
    def _setup_ui(self):
        """Setup UI components"""
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
            "Automatically search for stocks, download data, and train AI models. "
            "The system uses multiple data sources with fallback for reliability."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("color: #888; font-size: 12px; padding: 5px;")
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
        
        # Status row
        status_row = QHBoxLayout()
        self.status_icon = QLabel("⏸️")
        self.status_icon.setFont(QFont("Segoe UI", 16))
        status_row.addWidget(self.status_icon)
        
        self.status_label = QLabel("Ready to start")
        self.status_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        progress_layout.addLayout(status_row)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v/100")
        progress_layout.addWidget(self.progress_bar)
        
        # Stats row
        stats_frame = QFrame()
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(0, 10, 0, 0)
        
        self.stat_labels = {}
        for key, label in [
            ('stocks', '📈 Stocks Found'),
            ('processed', '✅ Processed'),
            ('accuracy', '🎯 Accuracy')
        ]:
            container = QWidget()
            cont_layout = QVBoxLayout(container)
            cont_layout.setContentsMargins(10, 5, 10, 5)
            
            title = QLabel(label)
            title.setStyleSheet("color: #888; font-size: 11px;")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            value = QLabel("--")
            value.setStyleSheet("color: #00E5FF; font-size: 18px; font-weight: bold;")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            cont_layout.addWidget(title)
            cont_layout.addWidget(value)
            stats_layout.addWidget(container)
            self.stat_labels[key] = value
        
        progress_layout.addWidget(stats_frame)
        layout.addWidget(progress_group)
        
        # Log
        log_group = QGroupBox("📋 Activity Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        
        self.start_btn = QPushButton("🚀 Start Learning")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.start_btn.clicked.connect(self._start_learning)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_learning)
        btn_layout.addWidget(self.stop_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.setMinimumHeight(50)
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)
        
        layout.addLayout(btn_layout)
    
    def _create_learn_tab(self) -> QWidget:
        """Create quick learn tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Info card
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a237e, stop:1 #0d47a1);
                border-radius: 12px;
                padding: 20px;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        
        info_title = QLabel("🎯 What Auto-Learning Does")
        info_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        info_title.setStyleSheet("color: white;")
        info_layout.addWidget(info_title)
        
        steps = [
            ("1️⃣", "Search Internet", "Find trending stocks from multiple sources"),
            ("2️⃣", "Download Data", "Get historical price data with fallback sources"),
            ("3️⃣", "Create Features", "Generate 80+ technical indicators"),
            ("4️⃣", "Train Models", "Train ensemble of 6 neural networks"),
            ("5️⃣", "Save & Evaluate", "Save best model and show performance"),
        ]
        
        for icon, title, desc in steps:
            step_layout = QHBoxLayout()
            
            icon_label = QLabel(icon)
            icon_label.setFont(QFont("Segoe UI", 14))
            icon_label.setFixedWidth(30)
            step_layout.addWidget(icon_label)
            
            text_layout = QVBoxLayout()
            title_label = QLabel(title)
            title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            title_label.setStyleSheet("color: #64b5f6;")
            text_layout.addWidget(title_label)
            
            desc_label = QLabel(desc)
            desc_label.setStyleSheet("color: #90caf9; font-size: 10px;")
            text_layout.addWidget(desc_label)
            
            step_layout.addLayout(text_layout)
            step_layout.addStretch()
            info_layout.addLayout(step_layout)
        
        layout.addWidget(info_frame)
        
        # Quick settings
        settings_group = QGroupBox("Quick Settings")
        settings_layout = QGridLayout(settings_group)
        
        settings_layout.addWidget(QLabel("Training Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(20, 500)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setToolTip("More epochs = longer training but potentially better results")
        settings_layout.addWidget(self.epochs_spin, 0, 1)
        
        settings_layout.addWidget(QLabel("Max Stocks:"), 0, 2)
        self.stocks_spin = QSpinBox()
        self.stocks_spin.setRange(10, 200)
        self.stocks_spin.setValue(50)
        self.stocks_spin.setToolTip("Number of stocks to use for training")
        settings_layout.addWidget(self.stocks_spin, 0, 3)
        
        self.search_check = QCheckBox("Search internet for stocks")
        self.search_check.setChecked(True)
        self.search_check.setToolTip("If unchecked, uses default stock pool")
        settings_layout.addWidget(self.search_check, 1, 0, 1, 2)
        
        self.incremental_check = QCheckBox("Incremental learning (keep old knowledge)")
        self.incremental_check.setChecked(True)
        self.incremental_check.setToolTip("Build on existing model instead of training from scratch")
        settings_layout.addWidget(self.incremental_check, 1, 2, 1, 2)
        
        layout.addWidget(settings_group)
        
        # Time estimate
        estimate_label = QLabel("⏱️ Estimated time: 30-60 minutes depending on settings and network speed")
        estimate_label.setStyleSheet("color: #888; font-style: italic;")
        estimate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(estimate_label)
        
        layout.addStretch()
        return widget
    
    def _create_settings_tab(self) -> QWidget:
        """Create settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Data sources
        data_group = QGroupBox("📊 Data Sources")
        data_layout = QVBoxLayout(data_group)
        
        self.akshare_check = QCheckBox("AkShare (Primary - Chinese A-shares)")
        self.akshare_check.setChecked(True)
        self.akshare_check.setEnabled(False)  # Always enabled
        data_layout.addWidget(self.akshare_check)
        
        self.yahoo_check = QCheckBox("Yahoo Finance (Fallback)")
        self.yahoo_check.setChecked(True)
        data_layout.addWidget(self.yahoo_check)
        
        data_note = QLabel("Note: System automatically falls back to working sources")
        data_note.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        data_layout.addWidget(data_note)
        
        layout.addWidget(data_group)
        
        # Model settings
        model_group = QGroupBox("🧠 Model Settings")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("Hidden Size:"), 0, 0)
        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(64, 512)
        self.hidden_spin.setValue(CONFIG.HIDDEN_SIZE)
        self.hidden_spin.setSingleStep(64)
        model_layout.addWidget(self.hidden_spin, 0, 1)
        
        model_layout.addWidget(QLabel("Sequence Length:"), 0, 2)
        self.seq_spin = QSpinBox()
        self.seq_spin.setRange(20, 120)
        self.seq_spin.setValue(CONFIG.SEQUENCE_LENGTH)
        model_layout.addWidget(self.seq_spin, 0, 3)
        
        model_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(16, 256)
        self.batch_spin.setValue(CONFIG.BATCH_SIZE)
        model_layout.addWidget(self.batch_spin, 1, 1)
        
        model_layout.addWidget(QLabel("Learning Rate:"), 1, 2)
        self.lr_label = QLabel(f"{CONFIG.LEARNING_RATE}")
        model_layout.addWidget(self.lr_label, 1, 3)
        
        layout.addWidget(model_group)
        
        # Stock discovery
        discovery_group = QGroupBox("🔍 Stock Discovery")
        discovery_layout = QVBoxLayout(discovery_group)
        
        self.gainers_check = QCheckBox("Top Gainers (涨幅榜)")
        self.gainers_check.setChecked(True)
        discovery_layout.addWidget(self.gainers_check)
        
        self.losers_check = QCheckBox("Top Losers (跌幅榜)")
        self.losers_check.setChecked(True)
        discovery_layout.addWidget(self.losers_check)
        
        self.volume_check = QCheckBox("High Volume (成交额榜)")
        self.volume_check.setChecked(True)
        discovery_layout.addWidget(self.volume_check)
        
        self.analyst_check = QCheckBox("Analyst Picks (机构推荐)")
        self.analyst_check.setChecked(True)
        discovery_layout.addWidget(self.analyst_check)
        
        layout.addWidget(discovery_group)
        
        layout.addStretch()
        return widget
    
    def _create_history_tab(self) -> QWidget:
        """Create history tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Stats cards
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background: #1a1a3e;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        stats_layout = QHBoxLayout(stats_frame)
        
        # Load history stats
        stats = self._load_history_stats()
        
        for title, value, color in [
            ("Total Sessions", str(stats.get('sessions', 0)), "#00E5FF"),
            ("Best Accuracy", f"{stats.get('best_accuracy', 0)*100:.1f}%", "#4CAF50"),
            ("Stocks Learned", str(stats.get('total_stocks', 0)), "#FF9800"),
        ]:
            card = QWidget()
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(20, 10, 20, 10)
            
            title_label = QLabel(title)
            title_label.setStyleSheet("color: #888; font-size: 11px;")
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            value_label = QLabel(value)
            value_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            card_layout.addWidget(title_label)
            card_layout.addWidget(value_label)
            stats_layout.addWidget(card)
        
        layout.addWidget(stats_frame)
        
        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            "Date/Time", "Stocks", "Samples", "Epochs", "Accuracy", "Duration"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setAlternatingRowColors(True)
        
        self._load_history_table()
        
        layout.addWidget(self.history_table)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh History")
        refresh_btn.clicked.connect(self._load_history_table)
        layout.addWidget(refresh_btn)
        
        return widget
    
    def _load_history_stats(self) -> dict:
        """Load history statistics"""
        try:
            import json
            path = CONFIG.DATA_DIR / "learning_history.json"
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {
                    'sessions': len(data.get('sessions', [])),
                    'best_accuracy': data.get('best_accuracy', 0),
                    'total_stocks': data.get('total_stocks', 0)
                }
        except Exception as e:
            log.warning(f"Failed to load history: {e}")
        return {'sessions': 0, 'best_accuracy': 0, 'total_stocks': 0}
    
    def _load_history_table(self):
        """Load history into table"""
        try:
            import json
            path = CONFIG.DATA_DIR / "learning_history.json"
            
            if not path.exists():
                self.history_table.setRowCount(0)
                return
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            sessions = data.get('sessions', [])
            self.history_table.setRowCount(len(sessions))
            
            for i, session in enumerate(reversed(sessions)):
                # Date/Time
                timestamp = session.get('timestamp', '')[:16]
                self.history_table.setItem(i, 0, QTableWidgetItem(timestamp))
                
                # Stocks
                self.history_table.setItem(i, 1, QTableWidgetItem(
                    str(session.get('stocks_used', 0))
                ))
                
                # Samples
                self.history_table.setItem(i, 2, QTableWidgetItem(
                    f"{session.get('samples', 0):,}"
                ))
                
                # Epochs
                self.history_table.setItem(i, 3, QTableWidgetItem(
                    str(session.get('epochs', 0))
                ))
                
                # Accuracy
                acc = session.get('test_accuracy', 0) * 100
                acc_item = QTableWidgetItem(f"{acc:.1f}%")
                acc_item.setForeground(QColor("#4CAF50" if acc > 50 else "#FF5252"))
                self.history_table.setItem(i, 4, acc_item)
                
                # Duration
                self.history_table.setItem(i, 5, QTableWidgetItem(
                    f"{session.get('duration_minutes', 0):.1f} min"
                ))
                
        except Exception as e:
            log.warning(f"Failed to load history table: {e}")
    
    def _apply_style(self):
        """Apply dialog styling"""
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
                padding: 0 8px;
            }
            QLabel {
                color: #ddd;
            }
            QSpinBox, QCheckBox {
                color: #fff;
                background: #1a1a3e;
                border: 1px solid #3a3a6a;
                border-radius: 5px;
                padding: 5px;
            }
            QSpinBox:focus {
                border-color: #00E5FF;
            }
            QProgressBar {
                border: none;
                background: #1a1a3e;
                border-radius: 8px;
                text-align: center;
                color: #fff;
                height: 30px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00E5FF, stop:1 #00BCD4);
                border-radius: 8px;
            }
            QTextEdit {
                background: #0a0a1a;
                color: #0f0;
                border: 1px solid #2a2a5a;
                border-radius: 8px;
                font-family: Consolas;
            }
            QTableWidget {
                background: #0f0f2a;
                color: #fff;
                border: none;
                gridline-color: #2a2a5a;
                selection-background-color: #2a2a5a;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background: #1a1a3e;
                color: #00E5FF;
                padding: 10px;
                border: none;
                font-weight: bold;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3a3a7a, stop:1 #2a2a5a);
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4a4a9a, stop:1 #3a3a7a);
            }
            QPushButton:disabled {
                background: #222;
                color: #555;
            }
            QPushButton#start_btn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00E5FF, stop:1 #00BCD4);
            }
            QPushButton#start_btn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00BCD4, stop:1 #0097A7);
            }
            QTabWidget::pane {
                border: 2px solid #2a2a5a;
                background: #0a0a1a;
                border-radius: 10px;
            }
            QTabBar::tab {
                background: #1a1a3e;
                color: #888;
                padding: 12px 25px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 3px;
            }
            QTabBar::tab:selected {
                background: #2a2a5a;
                color: #00E5FF;
            }
        """)
        
        # Apply specific button styles
        self.start_btn.setObjectName("start_btn")
    
    def _start_learning(self):
        """Start the learning process"""
        reply = QMessageBox.question(
            self,
            "Start Auto-Learning",
            "The AI will automatically:\n\n"
            "• Search for stocks online\n"
            "• Download historical data\n"
            "• Train neural network models\n\n"
            f"Settings:\n"
            f"• Epochs: {self.epochs_spin.value()}\n"
            f"• Max stocks: {self.stocks_spin.value()}\n\n"
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
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        self._log("🚀 Starting auto-learning process...", "info")
        self._update_status("searching", "Initializing...")
        
        # Prepare settings
        settings = {
            'auto_search': self.search_check.isChecked(),
            'max_stocks': self.stocks_spin.value(),
            'epochs': self.epochs_spin.value(),
            'incremental': self.incremental_check.isChecked()
        }
        
        # Start worker
        self.worker = AutoLearnWorker(settings)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.log_message.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
    
    def _stop_learning(self):
        """Stop the learning process"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Stop Learning",
                "Are you sure you want to stop the learning process?\n\n"
                "Progress will be lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                self._log("⏹️ Stopping...", "warning")
    
    def _on_progress(self, progress):
        """Handle progress update"""
        self._progress_stage = progress.stage
        self._progress_pct = progress.progress
        self._progress_message = progress.message
        
        # Update progress bar
        self.progress_bar.setValue(int(progress.progress))
        
        # Update stats
        self.stat_labels['stocks'].setText(str(progress.stocks_found))
        self.stat_labels['processed'].setText(str(progress.stocks_processed))
        
        if progress.training_accuracy > 0:
            self.stat_labels['accuracy'].setText(f"{progress.training_accuracy:.1%}")
        
        # Update status
        self._update_status(progress.stage, progress.message)
    
    def _on_finished(self, success: bool, message: str):
        """Handle completion"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if success:
            self._log(f"✅ {message}", "success")
            self._update_status("complete", message)
            self.progress_bar.setValue(100)
            
            QMessageBox.information(
                self,
                "Learning Complete",
                f"Auto-learning completed successfully!\n\n{message}"
            )
        else:
            self._log(f"❌ {message}", "error")
            self._update_status("error", message)
        
        # Refresh history
        self._load_history_table()
        
        self.worker = None
    
    def _update_status(self, stage: str, message: str):
        """Update status display"""
        icons = {
            'idle': '⏸️',
            'searching': '🔍',
            'downloading': '📥',
            'preparing': '🔧',
            'training': '🧠',
            'evaluating': '📊',
            'complete': '✅',
            'error': '❌'
        }
        
        colors = {
            'idle': '#888',
            'searching': '#2196F3',
            'downloading': '#FF9800',
            'preparing': '#9C27B0',
            'training': '#00E5FF',
            'evaluating': '#4CAF50',
            'complete': '#4CAF50',
            'error': '#F44336'
        }
        
        icon = icons.get(stage, '⏳')
        color = colors.get(stage, '#888')
        
        self.status_icon.setText(icon)
        self.status_label.setText(message[:50] + "..." if len(message) > 50 else message)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    
    def _log(self, message: str, level: str = "info"):
        """Add log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        colors = {
            "info": "#888",
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#F44336"
        }
        color = colors.get(level, "#888")
        
        self.log_text.append(
            f'<span style="color:#555">[{timestamp}]</span> '
            f'<span style="color:{color}">{message}</span>'
        )
        
        # Auto-scroll
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """Handle close"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Close",
                "Learning is in progress. Stop and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                self.worker.wait(3000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def show_auto_learn_dialog(parent=None):
    """Show the auto-learn dialog"""
    dialog = AutoLearnDialog(parent)
    dialog.exec()