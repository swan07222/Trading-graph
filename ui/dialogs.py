"""
Dialogs: Training, Backtest, Broker Settings, Risk Settings
PyQt6
"""
from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QProgressBar, QTextEdit, QSpinBox,
    QDoubleSpinBox, QCheckBox, QLineEdit, QMessageBox, QFileDialog,
    QGroupBox, QDialogButtonBox, QListWidget, QListWidgetItem, QComboBox
)

from config import CONFIG, TradingMode
from utils.logger import log


# ----------------------------
# Worker Threads
# ----------------------------

class TrainWorker(QThread):
    progress = pyqtSignal(str)                 # text log line
    epoch = pyqtSignal(str, int, float)        # model_name, epoch, val_acc
    finished = pyqtSignal(dict)                # results
    failed = pyqtSignal(str)                   # error message

    def __init__(self, stocks: List[str], epochs: int):
        super().__init__()
        self.stocks = stocks
        self.epochs = epochs

    def run(self):
        try:
            from models.trainer import Trainer

            trainer = Trainer()

            def cb(model_name: str, epoch_idx: int, val_acc: float):
                # epoch_idx is 0-based inside ensemble training
                self.epoch.emit(model_name, epoch_idx + 1, float(val_acc))

            self.progress.emit(f"Loading data for {len(self.stocks)} stocks...")
            results = trainer.train(stock_codes=self.stocks, epochs=self.epochs, callback=cb, save_model=True)

            self.finished.emit(results)
        except Exception as e:
            self.failed.emit(str(e))


class BacktestWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)  # BacktestResult
    failed = pyqtSignal(str)

    def __init__(self, stocks: List[str], train_months: int, test_months: int):
        super().__init__()
        self.stocks = stocks
        self.train_months = train_months
        self.test_months = test_months

    def run(self):
        try:
            from analysis.backtest import Backtester
            bt = Backtester()
            self.progress.emit("Running walk-forward backtest...")
            result = bt.run(self.stocks, train_months=self.train_months, test_months=self.test_months)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


# ----------------------------
# Training Dialog
# ----------------------------

class TrainingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train AI Model (from zero)")
        self.setMinimumSize(720, 520)

        self.worker: Optional[TrainWorker] = None

        layout = QVBoxLayout(self)

        # Settings
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

        # Stock list
        stocks_group = QGroupBox("Training Stocks")
        stocks_layout = QHBoxLayout(stocks_group)

        self.stocks_list = QListWidget()
        self.stocks_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for code in CONFIG.STOCK_POOL:
            item = QListWidgetItem(code)
            self.stocks_list.addItem(item)
        # pre-select first 10
        for i in range(min(10, self.stocks_list.count())):
            self.stocks_list.item(i).setSelected(True)

        stocks_layout.addWidget(self.stocks_list)

        right = QVBoxLayout()
        self.add_stock_edit = QLineEdit()
        self.add_stock_edit.setPlaceholderText("Add stock code (e.g. 600519)")
        right.addWidget(self.add_stock_edit)

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_stock)
        right.addWidget(add_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        right.addWidget(remove_btn)

        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: self.stocks_list.selectAll())
        right.addWidget(select_all_btn)

        clear_sel_btn = QPushButton("Clear Selection")
        clear_sel_btn.clicked.connect(lambda: self.stocks_list.clearSelection())
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

        # Buttons
        btns = QDialogButtonBox()
        self.start_btn = btns.addButton("Start Training", QDialogButtonBox.ButtonRole.AcceptRole)
        self.stop_btn = btns.addButton("Stop", QDialogButtonBox.ButtonRole.DestructiveRole)
        self.close_btn = btns.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)

        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        self.close_btn.clicked.connect(self.close)

        self.stop_btn.setEnabled(False)

        layout.addWidget(btns)

    def _selected_stocks(self) -> List[str]:
        items = self.stocks_list.selectedItems()
        return [it.text().strip() for it in items if it.text().strip()]

    def _add_stock(self):
        code = self.add_stock_edit.text().strip()
        if not code:
            return
        # avoid duplicates
        for i in range(self.stocks_list.count()):
            if self.stocks_list.item(i).text().strip() == code:
                return
        self.stocks_list.addItem(QListWidgetItem(code))
        self.add_stock_edit.clear()

    def _remove_selected(self):
        for it in self.stocks_list.selectedItems():
            row = self.stocks_list.row(it)
            self.stocks_list.takeItem(row)

    def start_training(self):
        stocks = self._selected_stocks()
        if not stocks:
            QMessageBox.warning(self, "No stocks selected", "Please select at least one stock.")
            return

        epochs = int(self.epochs_spin.value())

        self.logs.clear()
        self.logs.append(f"Starting training for {epochs} epochs on {len(stocks)} stocks...")
        self.status.setText("Training...")
        self.progress.setValue(0)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = TrainWorker(stocks=stocks, epochs=epochs)
        self.worker.progress.connect(self._on_log)
        self.worker.epoch.connect(self._on_epoch)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def stop_training(self):
        # Safe stop in this simplified version: just terminate the thread (not ideal).
        # For production: implement cooperative cancellation in Trainer/Ensemble training loops.
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(1000)
        self._set_idle("Stopped by user")

    def _on_log(self, msg: str):
        self.logs.append(msg)

    def _on_epoch(self, model_name: str, epoch: int, val_acc: float):
        self.logs.append(f"[{model_name}] epoch={epoch} val_acc={val_acc:.2%}")
        # Progress heuristic: 0..100 based on epoch only (not per-model)
        # If you want exact: also send total epochs and model count.
        epochs = int(self.epochs_spin.value())
        pct = int(min(100, (epoch / max(1, epochs)) * 100))
        self.progress.setValue(pct)

    def _on_finished(self, results: dict):
        best_acc = results.get("best_accuracy", 0.0)
        self.logs.append("")
        self.logs.append(f"Training finished. Best validation accuracy: {best_acc:.2%}")

        tm = (results.get("test_metrics") or {}).get("trading")
        if tm:
            self.logs.append(f"Test trading sim: return={tm.get('total_return',0):+.2f}% "
                             f"win_rate={tm.get('win_rate',0):.1%} sharpe={tm.get('sharpe_ratio',0):.2f}")

        self._set_idle("Done")

    def _on_failed(self, err: str):
        self.logs.append(f"ERROR: {err}")
        self._set_idle("Failed")

    def _set_idle(self, status: str):
        self.status.setText(status)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


# ----------------------------
# Backtest Dialog
# ----------------------------

class BacktestDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Walk-Forward Backtest")
        self.setMinimumSize(720, 520)

        self.worker: Optional[BacktestWorker] = None

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

        form.addRow(QLabel("Stocks (use default pool selection):"), QLabel(""))

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
        self.close_btn = QPushButton("Close")
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

        self.run_btn.clicked.connect(self.run_backtest)
        self.close_btn.clicked.connect(self.close)

    def run_backtest(self):
        self.logs.clear()
        self.progress.setVisible(True)

        stocks = CONFIG.STOCK_POOL[:5]  # default subset; you can add selection UI
        self.worker = BacktestWorker(
            stocks=stocks,
            train_months=int(self.train_months.value()),
            test_months=int(self.test_months.value()),
        )
        self.worker.progress.connect(lambda m: self.logs.append(m))
        self.worker.finished.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_done(self, result):
        self.progress.setVisible(False)
        try:
            self.logs.append(result.summary())
        except Exception:
            self.logs.append(str(result))

    def _on_failed(self, err: str):
        self.progress.setVisible(False)
        self.logs.append(f"ERROR: {err}")


# ----------------------------
# Broker Settings Dialog
# ----------------------------

class BrokerSettingsDialog(QDialog):
    """
    Configure broker path (THS) and mode.
    For real trading with THS: easytrader requires THS client installed and configured.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Broker Settings")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)

        form_group = QGroupBox("Broker Configuration")
        form = QFormLayout(form_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["simulation", "live"])
        self.mode_combo.setCurrentText(CONFIG.TRADING_MODE.value)
        form.addRow("Trading mode:", self.mode_combo)

        self.path_edit = QLineEdit()
        self.path_edit.setText(CONFIG.BROKER_PATH or "")
        form.addRow("THS broker executable path:", self.path_edit)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        form.addRow("", browse)

        layout.addWidget(form_group)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select broker executable")
        if path:
            self.path_edit.setText(path)

    def _save(self):
        mode = self.mode_combo.currentText().strip().lower()
        CONFIG.TRADING_MODE = TradingMode.LIVE if mode == "live" else TradingMode.SIMULATION
        CONFIG.BROKER_PATH = self.path_edit.text().strip()

        QMessageBox.information(self, "Saved", "Broker settings updated (in-memory).")
        self.accept()


# ----------------------------
# Risk Settings Dialog
# ----------------------------

class RiskSettingsDialog(QDialog):
    """
    Adjust risk parameters at runtime (in-memory).
    For persistence you can write these values into a JSON file and load on startup.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Risk Settings")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)

        group = QGroupBox("Risk Parameters")
        form = QFormLayout(group)

        self.max_pos_pct = QDoubleSpinBox()
        self.max_pos_pct.setRange(1, 50)
        self.max_pos_pct.setValue(float(CONFIG.MAX_POSITION_PCT))
        self.max_pos_pct.setSuffix(" %")
        form.addRow("Max position per stock:", self.max_pos_pct)

        self.max_daily_loss = QDoubleSpinBox()
        self.max_daily_loss.setRange(0.5, 20)
        self.max_daily_loss.setValue(float(CONFIG.MAX_DAILY_LOSS_PCT))
        self.max_daily_loss.setSuffix(" %")
        form.addRow("Max daily loss:", self.max_daily_loss)

        self.risk_per_trade = QDoubleSpinBox()
        self.risk_per_trade.setRange(0.1, 10)
        self.risk_per_trade.setValue(float(CONFIG.RISK_PER_TRADE))
        self.risk_per_trade.setSuffix(" %")
        form.addRow("Risk per trade:", self.risk_per_trade)

        self.max_positions = QSpinBox()
        self.max_positions.setRange(1, 50)
        self.max_positions.setValue(int(CONFIG.MAX_POSITIONS))
        form.addRow("Max open positions:", self.max_positions)

        layout.addWidget(group)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save(self):
        CONFIG.MAX_POSITION_PCT = float(self.max_pos_pct.value())
        CONFIG.MAX_DAILY_LOSS_PCT = float(self.max_daily_loss.value())
        CONFIG.RISK_PER_TRADE = float(self.risk_per_trade.value())
        CONFIG.MAX_POSITIONS = int(self.max_positions.value())

        QMessageBox.information(self, "Saved", "Risk settings updated (in-memory).")
        self.accept()