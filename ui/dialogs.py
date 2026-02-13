# ui/dialogs.py
from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from config.settings import CONFIG, TradingMode
from utils.logger import get_logger

log = get_logger(__name__)

def _apply_dialog_theme(dialog: QDialog) -> None:
    """Apply consistent professional styling for modal dialogs."""
    dialog.setStyleSheet("""
        QDialog { background: #0f1728; color: #d7e0f2; }
        QGroupBox {
            border: 1px solid #243454;
            border-radius: 10px;
            margin-top: 12px;
            padding-top: 12px;
            font-weight: 700;
            color: #9eb9ff;
            background: #101b30;
        }
        QGroupBox::title { left: 12px; padding: 0 6px; }
        QLabel { color: #d7e0f2; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QTextEdit {
            background: #131f36;
            color: #d7e0f2;
            border: 1px solid #2b3a5b;
            border-radius: 7px;
            padding: 6px 8px;
            selection-background-color: #3558c8;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus {
            border-color: #4c78ff;
            background: #18243d;
        }
        QPushButton {
            background: #1a2a49;
            color: #e6eeff;
            border: 1px solid #335084;
            border-radius: 7px;
            padding: 7px 12px;
            font-weight: 700;
        }
        QPushButton:hover { background: #22365f; border-color: #4c78ff; }
        QPushButton:disabled { background: #121d33; color: #5f6d89; border-color: #243454; }
        QProgressBar {
            border: 1px solid #2c3f63;
            border-radius: 6px;
            background: #101b2f;
            color: #dbe5ff;
            text-align: center;
            min-height: 18px;
        }
        QProgressBar::chunk {
            border-radius: 5px;
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #2c7be5, stop:1 #32c48d);
        }
    """)

def _get_cancellation_token():
    """Get CancellationToken class."""
    from utils.cancellation import CancellationToken
    return CancellationToken

def _get_cancelled_exception():
    """Get CancelledException class."""
    from utils.cancellation import CancelledException
    return CancelledException

class TrainWorker(QThread):
    """Worker thread for model training with proper cancellation."""
    progress = pyqtSignal(str)
    epoch = pyqtSignal(str, int, float)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, stocks: list[str], epochs: int):
        super().__init__()
        self.stocks = list(stocks)
        self.epochs = int(epochs)

        CancellationToken = _get_cancellation_token()
        self.cancel_token = CancellationToken()

    def run(self):
        try:
            from models.trainer import Trainer

            CancelledException = _get_cancelled_exception()

            trainer = Trainer()

            def cb(model_name: str, epoch_idx: int, val_acc: float):
                self.cancel_token.raise_if_cancelled()
                self.epoch.emit(
                    str(model_name),
                    int(epoch_idx) + 1,
                    float(val_acc)
                )

            self.progress.emit(
                f"Loading data for {len(self.stocks)} stocks..."
            )

            results = trainer.train(
                stock_codes=self.stocks,
                epochs=self.epochs,
                callback=cb,
                stop_flag=self.cancel_token,
                save_model=True
            )

            self.finished.emit(results if results else {})

        except Exception as e:
            # Check if it's a cancellation
            try:
                CancelledException = _get_cancelled_exception()
                if isinstance(e, CancelledException):
                    self.finished.emit({"cancelled": True})
                    return
            except ImportError:
                pass

            if self.cancel_token.is_cancelled:
                self.finished.emit({"cancelled": True})
                return

            self.failed.emit(str(e))

    def cancel(self):
        """Cancel training gracefully via token."""
        self.cancel_token.cancel()

class BacktestWorker(QThread):
    """Worker thread for backtesting with cancellation support."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)  # BacktestResult
    failed = pyqtSignal(str)

    def __init__(
        self,
        stocks: list[str],
        train_months: int,
        test_months: int
    ):
        super().__init__()
        self.stocks = list(stocks)
        self.train_months = int(train_months)
        self.test_months = int(test_months)
        self._cancelled = False

    def run(self):
        try:
            if self._cancelled:
                return

            from analysis.backtest import Backtester
            bt = Backtester()
            self.progress.emit("Running walk-forward backtest...")
            result = bt.run(
                self.stocks,
                train_months=self.train_months,
                test_months=self.test_months
            )

            if not self._cancelled:
                self.finished.emit(result)

        except Exception as e:
            if not self._cancelled:
                self.failed.emit(str(e))

    def cancel(self):
        """Cancel backtest."""
        self._cancelled = True

class TrainingDialog(QDialog):
    """Dialog for training the AI model from scratch or incrementally."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train AI Model")
        self.setMinimumSize(720, 520)
        _apply_dialog_theme(self)

        self.worker: TrainWorker | None = None
        self._is_training = False

        layout = QVBoxLayout(self)

        settings_group = QGroupBox("Training Settings")
        settings_layout = QGridLayout(settings_group)

        settings_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(5, 500)
        self.epochs_spin.setValue(CONFIG.EPOCHS)
        settings_layout.addWidget(self.epochs_spin, 0, 1)

        settings_layout.addWidget(QLabel("Sequence Length:"), 0, 2)
        self.seq_label = QLabel(str(CONFIG.SEQUENCE_LENGTH))
        settings_layout.addWidget(self.seq_label, 0, 3)

        settings_layout.addWidget(QLabel("Hidden Size:"), 1, 0)
        self.hidden_label = QLabel(str(CONFIG.HIDDEN_SIZE))
        settings_layout.addWidget(self.hidden_label, 1, 1)

        settings_layout.addWidget(QLabel("Models in Ensemble:"), 1, 2)
        self.models_label = QLabel("lstm, transformer, gru, tcn, hybrid")
        settings_layout.addWidget(self.models_label, 1, 3)

        layout.addWidget(settings_group)

        stocks_group = QGroupBox("Training Stocks")
        stocks_layout = QHBoxLayout(stocks_group)

        self.stocks_list = QListWidget()
        self.stocks_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )
        for code in CONFIG.STOCK_POOL:
            item = QListWidgetItem(str(code))
            self.stocks_list.addItem(item)
        # pre-select first 10
        for i in range(min(10, self.stocks_list.count())):
            self.stocks_list.item(i).setSelected(True)

        stocks_layout.addWidget(self.stocks_list)

        right = QVBoxLayout()
        self.add_stock_edit = QLineEdit()
        self.add_stock_edit.setPlaceholderText(
            "Add stock code (e.g. 600519)"
        )
        right.addWidget(self.add_stock_edit)

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_stock)
        right.addWidget(add_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        right.addWidget(remove_btn)

        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(
            lambda: self.stocks_list.selectAll()
        )
        right.addWidget(select_all_btn)

        clear_sel_btn = QPushButton("Clear Selection")
        clear_sel_btn.clicked.connect(
            lambda: self.stocks_list.clearSelection()
        )
        right.addWidget(clear_sel_btn)

        right.addStretch()
        stocks_layout.addLayout(right)

        layout.addWidget(stocks_group)

        # Progress + logs
        prog_group = QGroupBox("Progress")
        prog_layout = QVBoxLayout(prog_group)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        prog_layout.addWidget(self.progress)

        self.status = QLabel("Ready")
        prog_layout.addWidget(self.status)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setFont(QFont("Consolas", 10))
        prog_layout.addWidget(self.logs)

        layout.addWidget(prog_group)

        btns = QDialogButtonBox()
        self.start_btn = btns.addButton(
            "Start Training",
            QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.stop_btn = btns.addButton(
            "Stop",
            QDialogButtonBox.ButtonRole.DestructiveRole
        )
        self.close_btn = btns.addButton(
            "Close",
            QDialogButtonBox.ButtonRole.RejectRole
        )

        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        self.close_btn.clicked.connect(self.close)

        self.stop_btn.setEnabled(False)

        layout.addWidget(btns)

    def _selected_stocks(self) -> list[str]:
        """Get list of selected stock codes."""
        items = self.stocks_list.selectedItems()
        return [
            it.text().strip()
            for it in items
            if it.text().strip()
        ]

    def _add_stock(self):
        """Add a stock code to the list."""
        code = self.add_stock_edit.text().strip()
        if not code:
            return

        # Validate: must be digits
        digits = "".join(c for c in code if c.isdigit())
        if len(digits) != 6:
            QMessageBox.warning(
                self, "Invalid Code",
                "Stock code must be 6 digits."
            )
            return

        code = digits

        for i in range(self.stocks_list.count()):
            if self.stocks_list.item(i).text().strip() == code:
                return
        self.stocks_list.addItem(QListWidgetItem(code))
        self.add_stock_edit.clear()

    def _remove_selected(self):
        """Remove selected stocks from the list."""
        for it in self.stocks_list.selectedItems():
            row = self.stocks_list.row(it)
            self.stocks_list.takeItem(row)

    def start_training(self):
        """Start the training process."""
        if self._is_training:
            return

        stocks = self._selected_stocks()
        if not stocks:
            QMessageBox.warning(
                self, "No stocks selected",
                "Please select at least one stock."
            )
            return

        epochs = int(self.epochs_spin.value())

        self.logs.clear()
        self.logs.append(
            f"Starting training for {epochs} epochs "
            f"on {len(stocks)} stocks..."
        )
        self.status.setText("Training...")
        self.progress.setValue(0)

        self._is_training = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.close_btn.setEnabled(False)

        self.epochs_spin.setEnabled(False)
        self.stocks_list.setEnabled(False)

        self.worker = TrainWorker(stocks=stocks, epochs=epochs)
        self.worker.progress.connect(self._on_log)
        self.worker.epoch.connect(self._on_epoch)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def stop_training(self):
        """Stop training gracefully via cancellation token."""
        if self.worker:
            self.logs.append("Requesting cancellation...")
            self.status.setText("Stopping...")
            self.stop_btn.setEnabled(False)

            self.worker.cancel()

            # Wait up to 10 seconds for clean exit
            if not self.worker.wait(10000):
                self.logs.append(
                    "Warning: Training thread did not stop cleanly"
                )

        self._set_idle("Stopped by user")

    def _on_log(self, msg: str):
        """Handle log message from worker."""
        self.logs.append(str(msg))

    def _on_epoch(self, model_name: str, epoch: int, val_acc: float):
        """Handle epoch completion from worker."""
        self.logs.append(
            f"[{model_name}] epoch={epoch} val_acc={val_acc:.2%}"
        )

        epochs = int(self.epochs_spin.value())
        pct = int(min(100, (epoch / max(1, epochs)) * 100))
        self.progress.setValue(pct)

    def _on_finished(self, results: dict):
        """Handle training completion."""
        if results.get("cancelled"):
            self.logs.append("")
            self.logs.append("Training was cancelled by user.")
            self._set_idle("Cancelled")
            return

        if results.get("status") == "cancelled":
            self.logs.append("")
            self.logs.append("Training was cancelled by user.")
            self._set_idle("Cancelled")
            return

        best_acc = float(results.get("best_accuracy", 0.0))
        self.logs.append("")
        self.logs.append(
            f"Training finished. "
            f"Best validation accuracy: {best_acc:.2%}"
        )

        tm = (results.get("test_metrics") or {}).get("trading")
        if tm:
            total_ret = float(tm.get('total_return', 0))
            win_rate = float(tm.get('win_rate', 0))
            sharpe = float(tm.get('sharpe_ratio', 0))
            self.logs.append(
                f"Test trading sim: return={total_ret:+.2f}% "
                f"win_rate={win_rate:.1%} sharpe={sharpe:.2f}"
            )

        self.progress.setValue(100)
        self._set_idle("Done")

        QMessageBox.information(
            self, "Training Complete",
            f"Model training completed successfully!\n\n"
            f"Best accuracy: {best_acc:.2%}\n"
            f"Models: {results.get('num_models', 0)}\n"
            f"Samples: {results.get('train_samples', 0)}"
        )

    def _on_failed(self, err: str):
        """Handle training failure."""
        self.logs.append(f"ERROR: {err}")
        self._set_idle("Failed")

        QMessageBox.critical(
            self, "Training Failed",
            f"Training failed with error:\n\n{err[:500]}"
        )

    def _set_idle(self, status: str):
        """Reset dialog to idle state."""
        self._is_training = False
        self.status.setText(status)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)

        # Re-enable settings
        self.epochs_spin.setEnabled(True)
        self.stocks_list.setEnabled(True)

        self.worker = None

    def closeEvent(self, event):
        """Handle close — stop training if running."""
        if self._is_training and self.worker:
            reply = QMessageBox.question(
                self, "Stop Training?",
                "Training is still in progress.\n\nStop and close?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.stop_training()
                event.accept()
            else:
                event.ignore()
                return
        else:
            event.accept()

        super().closeEvent(event)

class BacktestDialog(QDialog):
    """Dialog for walk-forward backtesting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Walk-Forward Backtest")
        self.setMinimumSize(720, 520)
        _apply_dialog_theme(self)

        self.worker: BacktestWorker | None = None

        layout = QVBoxLayout(self)

        settings_group = QGroupBox("Backtest Settings")
        form = QFormLayout(settings_group)

        self.train_months = QSpinBox()
        self.train_months.setRange(3, 36)
        self.train_months.setValue(12)
        form.addRow("Train months:", self.train_months)

        self.test_months = QSpinBox()
        self.test_months.setRange(1, 6)
        self.test_months.setValue(1)
        form.addRow("Test months:", self.test_months)

        form.addRow(
            QLabel("Stocks (using default pool):"),
            QLabel("")
        )

        layout.addWidget(settings_group)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setFont(QFont("Consolas", 10))
        layout.addWidget(self.logs)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Backtest")
        self.stop_btn = QPushButton("Stop")
        self.close_btn = QPushButton("Close")

        self.stop_btn.setEnabled(False)

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

        self.run_btn.clicked.connect(self.run_backtest)
        self.stop_btn.clicked.connect(self.stop_backtest)
        self.close_btn.clicked.connect(self.close)

    def run_backtest(self):
        """Start backtest."""
        self.logs.clear()
        self.progress.setVisible(True)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        stocks = CONFIG.STOCK_POOL[:5]
        self.worker = BacktestWorker(
            stocks=stocks,
            train_months=int(self.train_months.value()),
            test_months=int(self.test_months.value()),
        )
        self.worker.progress.connect(lambda m: self.logs.append(str(m)))
        self.worker.finished.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def stop_backtest(self):
        """Stop backtest."""
        if self.worker:
            self.worker.cancel()
            self.worker.wait(5000)
            self.worker = None
        self._reset_ui("Stopped")

    def _on_done(self, result):
        """Handle backtest completion."""
        self._reset_ui("Done")
        try:
            if hasattr(result, 'summary'):
                self.logs.append(result.summary())
            else:
                self.logs.append(str(result))
        except Exception:
            self.logs.append(str(result))

    def _on_failed(self, err: str):
        """Handle backtest failure."""
        self._reset_ui("Failed")
        self.logs.append(f"ERROR: {err}")

    def _reset_ui(self, status: str = "Ready"):
        """Reset UI to idle state."""
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None

    def closeEvent(self, event):
        """Handle close — stop backtest if running."""
        if self.worker and self.worker.isRunning():
            self.stop_backtest()
        event.accept()
        super().closeEvent(event)

class BrokerSettingsDialog(QDialog):
    """
    Configure broker path (THS) and mode.
    For real trading with THS: easytrader requires THS client.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Broker Settings")
        self.setMinimumWidth(520)
        _apply_dialog_theme(self)

        layout = QVBoxLayout(self)

        form_group = QGroupBox("Broker Configuration")
        form = QFormLayout(form_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["simulation", "live"])
        try:
            current_mode = CONFIG.trading_mode.value
        except Exception:
            current_mode = "simulation"
        self.mode_combo.setCurrentText(current_mode)
        form.addRow("Trading mode:", self.mode_combo)

        self.path_edit = QLineEdit()
        try:
            current_path = CONFIG.broker_path or ""
        except Exception:
            current_path = ""
        self.path_edit.setText(current_path)
        form.addRow("THS broker executable path:", self.path_edit)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        form.addRow("", browse)

        layout.addWidget(form_group)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _browse(self):
        """Browse for broker executable."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select broker executable"
        )
        if path:
            self.path_edit.setText(path)

    def _save(self):
        """Save broker settings."""
        mode = self.mode_combo.currentText().strip().lower()
        try:
            CONFIG.trading_mode = (
                TradingMode.LIVE if mode == "live"
                else TradingMode.SIMULATION
            )
        except Exception as e:
            log.warning(f"Failed to set trading mode: {e}")

        try:
            CONFIG.broker_path = self.path_edit.text().strip()
        except Exception as e:
            log.warning(f"Failed to set broker path: {e}")

        QMessageBox.information(
            self, "Saved",
            "Broker settings updated (in-memory)."
        )
        self.accept()

class RiskSettingsDialog(QDialog):
    """
    Adjust risk parameters at runtime (in-memory).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Risk Settings")
        self.setMinimumWidth(520)
        _apply_dialog_theme(self)

        layout = QVBoxLayout(self)

        group = QGroupBox("Risk Parameters")
        form = QFormLayout(group)

        max_pos = self._safe_config_float('MAX_POSITION_PCT', 15.0)
        max_daily = self._safe_config_float('MAX_DAILY_LOSS_PCT', 3.0)
        risk_trade = self._safe_config_float('RISK_PER_TRADE', 2.0)
        max_positions = self._safe_config_int('MAX_POSITIONS', 10)

        self.max_pos_pct = QDoubleSpinBox()
        self.max_pos_pct.setRange(1, 50)
        self.max_pos_pct.setValue(max_pos)
        self.max_pos_pct.setSuffix(" %")
        form.addRow("Max position per stock:", self.max_pos_pct)

        self.max_daily_loss = QDoubleSpinBox()
        self.max_daily_loss.setRange(0.5, 20)
        self.max_daily_loss.setValue(max_daily)
        self.max_daily_loss.setSuffix(" %")
        form.addRow("Max daily loss:", self.max_daily_loss)

        self.risk_per_trade = QDoubleSpinBox()
        self.risk_per_trade.setRange(0.1, 10)
        self.risk_per_trade.setValue(risk_trade)
        self.risk_per_trade.setSuffix(" %")
        form.addRow("Risk per trade:", self.risk_per_trade)

        self.max_positions = QSpinBox()
        self.max_positions.setRange(1, 50)
        self.max_positions.setValue(max_positions)
        form.addRow("Max open positions:", self.max_positions)

        layout.addWidget(group)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    @staticmethod
    def _safe_config_float(attr: str, default: float) -> float:
        """Safely read a float from CONFIG."""
        try:
            val = getattr(CONFIG, attr, None)
            if val is not None:
                return float(val)
        except (TypeError, ValueError, AttributeError):
            pass
        return float(default)

    @staticmethod
    def _safe_config_int(attr: str, default: int) -> int:
        """Safely read an int from CONFIG."""
        try:
            val = getattr(CONFIG, attr, None)
            if val is not None:
                return int(val)
        except (TypeError, ValueError, AttributeError):
            pass
        return int(default)

    def _save(self):
        """Save risk settings to CONFIG (in-memory)."""
        try:
            CONFIG.MAX_POSITION_PCT = float(self.max_pos_pct.value())
        except Exception as e:
            log.warning(f"Failed to set MAX_POSITION_PCT: {e}")

        try:
            CONFIG.MAX_DAILY_LOSS_PCT = float(self.max_daily_loss.value())
        except Exception as e:
            log.warning(f"Failed to set MAX_DAILY_LOSS_PCT: {e}")

        try:
            CONFIG.RISK_PER_TRADE = float(self.risk_per_trade.value())
        except Exception as e:
            log.warning(f"Failed to set RISK_PER_TRADE: {e}")

        try:
            CONFIG.MAX_POSITIONS = int(self.max_positions.value())
        except Exception as e:
            log.warning(f"Failed to set MAX_POSITIONS: {e}")

        QMessageBox.information(
            self, "Saved",
            "Risk settings updated (in-memory)."
        )
        self.accept()
