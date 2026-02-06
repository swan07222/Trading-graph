"""
Auto-Learning Dialog
Provides UI for continuous learning functionality
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QSpinBox, QComboBox,
    QCheckBox, QMessageBox, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from datetime import datetime
from typing import Optional, Dict, Any
from utils.cancellation import CancellationToken


class AutoLearnWorker(QThread):
    """Worker thread for auto-learning"""
    progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.running = True
        self.token = CancellationToken()
    
    def run(self):
        """
        Continuous mode:
        - starts ContinuousLearner in continuous loop
        - streams progress into UI
        - stops only when user presses Stop
        """
        try:
            from models.auto_learner import AutoLearner  # alias of ContinuousLearner

            learner = AutoLearner()

            # map UI config -> learner params
            max_stocks = int(self.config.get("max_stocks", 50))
            epochs = int(self.config.get("epochs", 10))
            incremental = bool(self.config.get("incremental", True))

            # continuous dealing-learning parameters (tune as needed)
            interval = "1m"
            horizon = 30
            lookback_bars = 3000
            cycle_interval_seconds = 900  # 15 minutes

            results = {"discovered": 0, "processed": 0, "samples": 0, "accuracy": 0.0}

            def on_progress(p):
                # p is LearningProgress
                if not self.running:
                    return
                # progress bar expects 0..100
                percent = int(max(0, min(100, p.progress)))
                self.progress.emit(percent, p.message or p.stage)
                self.log_message.emit(f"{p.stage}: {p.message}", "info")

                results["discovered"] = int(getattr(p, "stocks_found", 0) or 0)
                results["processed"] = int(getattr(p, "stocks_processed", 0) or 0)
                results["accuracy"] = float(getattr(p, "validation_accuracy", 0.0) or 0.0)

            learner.add_callback(on_progress)

            # Start continuous learner
            learner.start(
                mode="full",
                max_stocks=max_stocks,
                epochs_per_cycle=epochs,
                min_market_cap=10,
                include_all_markets=True,
                continuous=True,
                learning_while_trading=True,
                interval=interval,
                prediction_horizon=horizon,
                lookback_bars=lookback_bars,
                cycle_interval_seconds=cycle_interval_seconds,
                incremental=incremental,
            )

            # Keep this QThread alive until stop() called
            while self.running and not self.token.is_cancelled:
                self.token.raise_if_cancelled()
                # sleep lightly to keep UI responsive
                self.msleep(200)

            # Stop learner when UI requests stop
            try:
                learner.stop()
            except Exception:
                pass

            self.finished.emit(results)

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            if self.running:
                self.error.emit(error_msg)
    
    def stop(self):
        self.running = False
        self.token.cancel()


class AutoLearnDialog(QDialog):
    """Dialog for automatic learning"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤖 Auto Learning")
        self.setMinimumSize(750, 650)
        self.worker: Optional[AutoLearnWorker] = None
        self._setup_ui()
        self._apply_style()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QLabel("🧠 Automatic Stock Discovery & Learning")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #58a6ff;")
        layout.addWidget(header)
        
        desc = QLabel(
            "This will automatically discover stocks from the internet, "
            "fetch their historical data, and train the AI model on new patterns.\n\n"
            "⚠️ This process may take 10-30 minutes depending on settings."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #8b949e; font-size: 12px;")
        layout.addWidget(desc)
        
        # Settings Group
        settings_group = QGroupBox("⚙️ Learning Settings")
        settings_layout = QGridLayout()
        settings_layout.setSpacing(10)
        
        # Row 0: Mode selection
        settings_layout.addWidget(QLabel("Learning Mode:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Full (Discover + Fetch + Train)",
            "Discovery + Fetch Only",
            "Training Only (use cached data)"
        ])
        self.mode_combo.setMinimumWidth(250)
        settings_layout.addWidget(self.mode_combo, 0, 1)
        
        # Row 1: Max stocks
        settings_layout.addWidget(QLabel("Max Stocks to Process:"), 1, 0)
        self.max_stocks_spin = QSpinBox()
        self.max_stocks_spin.setRange(10, 500)
        self.max_stocks_spin.setValue(50)
        self.max_stocks_spin.setSuffix(" stocks")
        settings_layout.addWidget(self.max_stocks_spin, 1, 1)
        
        # Row 2: Training epochs
        settings_layout.addWidget(QLabel("Training Epochs:"), 2, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 200)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setSuffix(" epochs")
        settings_layout.addWidget(self.epochs_spin, 2, 1)
        
        # Row 3: Options
        self.discover_check = QCheckBox("Discover new stocks from internet")
        self.discover_check.setChecked(True)
        settings_layout.addWidget(self.discover_check, 3, 0, 1, 2)
        
        self.incremental_check = QCheckBox("Incremental training (keep existing model weights)")
        self.incremental_check.setChecked(False)
        settings_layout.addWidget(self.incremental_check, 4, 0, 1, 2)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Progress Group
        progress_group = QGroupBox("📊 Progress")
        progress_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready to start")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(25)
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Log Group
        log_group = QGroupBox("📋 Activity Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("🚀 Start Learning")
        self.start_btn.setMinimumHeight(45)
        self.start_btn.setStyleSheet("""
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
        """)
        self.start_btn.clicked.connect(self._start_learning)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.setMinimumHeight(45)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_learning)
        btn_layout.addWidget(self.stop_btn)
        
        btn_layout.addStretch()
        
        self.close_btn = QPushButton("Close")
        self.close_btn.setMinimumHeight(45)
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)
        
        layout.addLayout(btn_layout)
    
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
                min-width: 150px;
            }
            QComboBox:focus, QSpinBox:focus {
                border-color: #58a6ff;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
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
    
    def _log(self, message: str, level: str = "info"):
        """Add message to log with timestamp and color"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        colors = {
            "info": "#c9d1d9",
            "success": "#3fb950",
            "warning": "#d29922",
            "error": "#f85149"
        }
        color = colors.get(level, "#c9d1d9")
        
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }
        icon = icons.get(level, "")
        
        self.log_text.append(
            f'<span style="color: #6e7681;">[{timestamp}]</span> '
            f'{icon} <span style="color: {color};">{message}</span>'
        )
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _start_learning(self):
        """Start auto-learning process"""
        mode_map = {
            0: "full",
            1: "discovery",
            2: "training"
        }
        
        config = {
            'mode': mode_map[self.mode_combo.currentIndex()],
            'discover_new': self.discover_check.isChecked(),
            'max_stocks': self.max_stocks_spin.value(),
            'epochs': self.epochs_spin.value(),
            'incremental': self.incremental_check.isChecked()
        }
        
        self._log(f"Starting auto-learning...", "info")
        self._log(f"Mode: {config['mode']}, Max stocks: {config['max_stocks']}", "info")
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.close_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing...")
        
        # Disable settings
        self.mode_combo.setEnabled(False)
        self.max_stocks_spin.setEnabled(False)
        self.epochs_spin.setEnabled(False)
        self.discover_check.setEnabled(False)
        self.incremental_check.setEnabled(False)
        
        # Start worker
        self.worker = AutoLearnWorker(config)
        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()
    
    def _stop_learning(self):
        """Stop auto-learning"""
        if self.worker:
            self._log("Stopping learning process...", "warning")
            self.status_label.setText("Stopping...")
            self.worker.stop()
            self.worker.wait(5000)
            self.worker = None
            self._log("Learning stopped by user", "warning")
        
        self._reset_ui()
    
    def _on_progress(self, percent: int, message: str):
        """Handle progress update"""
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)
    
    def _on_finished(self, results: dict):
        """Handle completion"""
        self._log("=" * 50, "info")
        self._log("🎉 Learning completed successfully!", "success")
        
        if results:
            self._log(f"📊 Stocks discovered: {results.get('discovered', 0)}", "info")
            self._log(f"📈 Stocks processed: {results.get('processed', 0)}", "info")
            self._log(f"🔢 Training samples: {results.get('samples', 0)}", "info")
            
            if results.get('accuracy', 0) > 0:
                self._log(f"🎯 Model accuracy: {results['accuracy']:.1%}", "success")
        
        self._reset_ui()
        self.status_label.setText("✅ Completed successfully!")
        self.progress_bar.setValue(100)
        
        QMessageBox.information(
            self, "Learning Complete",
            f"Auto-learning has completed successfully!\n\n"
            f"Stocks discovered: {results.get('discovered', 0)}\n"
            f"Stocks processed: {results.get('processed', 0)}\n"
            f"Training samples: {results.get('samples', 0)}\n\n"
            f"The AI model has been updated with new data."
        )
    
    def _on_error(self, error: str):
        """Handle error"""
        self._log(f"Error occurred: {error}", "error")
        self._reset_ui()
        self.status_label.setText("❌ Error occurred")
        
        QMessageBox.critical(
            self, "Learning Error",
            f"An error occurred during learning:\n\n{error[:500]}"
        )
    
    def _reset_ui(self):
        """Reset UI state"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        
        # Re-enable settings
        self.mode_combo.setEnabled(True)
        self.max_stocks_spin.setEnabled(True)
        self.epochs_spin.setEnabled(True)
        self.discover_check.setEnabled(True)
        self.incremental_check.setEnabled(True)
        
        self.worker = None
    
    def closeEvent(self, event):
        """Handle close event"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Stop Learning?",
                "Learning is still in progress.\n\nStop and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._stop_learning()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def show_auto_learn_dialog(parent=None):
    """Show the auto-learn dialog - convenience function"""
    dialog = AutoLearnDialog(parent)
    dialog.exec()
    return dialog