from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Any

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from ui.modern_theme import (
    ModernColors,
    ModernFonts,
    get_dialog_style,
    get_monospace_font_family,
)
from utils.logger import get_logger

log = get_logger(__name__)


def _force_china_direct_network_mode() -> None:
    os.environ["TRADING_CHINA_DIRECT"] = "1"
    os.environ["TRADING_VPN"] = "0"
    try:
        from core.network import invalidate_network_cache

        invalidate_network_cache()
    except Exception:
        pass


class LLMAutoTrainWorker(QThread):
    progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str, str)
    finished_result = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config or {})
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            _force_china_direct_network_mode()
            self.log_message.emit(
                "China-direct network mode enabled for LLM auto internet training.",
                "info",
            )

            from data.llm_sentiment import get_llm_analyzer

            analyzer = get_llm_analyzer()
            idle_seconds = max(0, int(self.config.get("idle_seconds", 3) or 3))
            cycle_index = 0
            total_collected = 0
            total_trained = 0
            last_report: dict[str, Any] = {}

            while not self._stop_event.is_set():
                cycle_index += 1
                self.progress.emit(0, f"Cycle {cycle_index}: starting")
                self.log_message.emit(
                    f"Starting auto-train cycle #{cycle_index}...",
                    "info",
                )

                def _progress(payload: dict[str, Any]) -> None:
                    if self._stop_event.is_set():
                        return
                    pct = int(max(0, min(100, int(payload.get("percent", 0) or 0))))
                    msg = str(payload.get("message", "") or "").strip()
                    stage = str(payload.get("stage", "") or "").strip()
                    prefixed = (
                        f"Cycle {cycle_index}: {msg}"
                        if msg
                        else f"Cycle {cycle_index}: {stage}"
                    ).strip()
                    if msg:
                        self.log_message.emit(f"[Cycle {cycle_index}] {msg}", "info")
                    self.progress.emit(pct, prefixed)

                try:
                    report = analyzer.auto_train_from_internet(
                        hours_back=int(self.config.get("hours_back", 120) or 120),
                        limit_per_query=int(self.config.get("limit_per_query", 220) or 220),
                        max_samples=int(self.config.get("max_samples", 1200) or 1200),
                        stop_flag=self._stop_event.is_set,
                        progress_callback=_progress,
                        force_china_direct=True,
                    )
                except Exception as exc:
                    # Keep auto mode alive on transient failures; user can stop anytime.
                    self.log_message.emit(
                        f"Cycle {cycle_index} failed: {exc}. Retrying...",
                        "warning",
                    )
                    if self._stop_event.wait(timeout=max(1, idle_seconds)):
                        break
                    continue

                out = dict(report or {})
                last_report = dict(out)
                status = str(out.get("status", "") or "").strip().lower()
                collected = int(out.get("collected_articles", 0) or 0)
                trained = int(out.get("trained_samples", 0) or 0)
                total_collected += max(0, collected)
                total_trained += max(0, trained)

                if self._stop_event.is_set() or status == "stopped":
                    break
                if status in {"error", "failed"}:
                    self.log_message.emit(
                        f"Cycle {cycle_index} reported status={status}. Continuing...",
                        "warning",
                    )
                else:
                    self.log_message.emit(
                        (
                            f"Cycle {cycle_index} complete: collected={collected}, "
                            f"trained={trained}. Waiting {idle_seconds}s..."
                        ),
                        "success",
                    )
                    self.progress.emit(100, f"Cycle {cycle_index}: complete")

                if self._stop_event.wait(timeout=idle_seconds):
                    break

            stopped_payload = dict(last_report)
            stopped_payload["status"] = "stopped"
            stopped_payload["cycles_completed"] = int(cycle_index)
            stopped_payload["total_collected_articles"] = int(total_collected)
            stopped_payload["total_trained_samples"] = int(total_trained)
            self.finished_result.emit(stopped_payload)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class LLMTrainDialog(QDialog):
    session_finished = pyqtSignal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto Train LLM")
        self.setMinimumSize(700, 500)
        self.resize(860, 620)
        self.setModal(False)

        self.worker: LLMAutoTrainWorker | None = None
        self._is_running = False
        self._last_progress_percent = 0
        self._elapsed_seconds = 0
        self._run_started_monotonic = 0.0

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._on_elapsed_tick)

        self._setup_ui()
        self._apply_style()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Auto Train LLM (Hybrid Neural Network)")
        title.setObjectName("dialogTitle")
        root.addWidget(title)

        hint = QLabel(
            "Collects internet data automatically in China-direct mode and trains the LLM "
            "hybrid model continuously until you press Stop."
        )
        hint.setObjectName("dialogHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        settings_group = QGroupBox("Settings")
        settings = QGridLayout(settings_group)
        settings.setSpacing(10)

        settings.addWidget(QLabel("Hours Back:"), 0, 0)
        self.hours_back_spin = QSpinBox()
        self.hours_back_spin.setRange(12, 720)
        self.hours_back_spin.setValue(120)
        self.hours_back_spin.setSuffix(" hours")
        settings.addWidget(self.hours_back_spin, 0, 1)

        settings.addWidget(QLabel("Limit / Query:"), 1, 0)
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(20, 2000)
        self.limit_spin.setValue(220)
        self.limit_spin.setSuffix(" items")
        settings.addWidget(self.limit_spin, 1, 1)

        settings.addWidget(QLabel("Max Samples:"), 2, 0)
        self.max_samples_spin = QSpinBox()
        self.max_samples_spin.setRange(80, 6000)
        self.max_samples_spin.setValue(1200)
        self.max_samples_spin.setSuffix(" samples")
        settings.addWidget(self.max_samples_spin, 2, 1)

        self.parallel_gm_check = QCheckBox("Train GM in parallel")
        self.parallel_gm_check.setChecked(False)
        settings.addWidget(self.parallel_gm_check, 3, 0, 1, 2)

        root.addWidget(settings_group)

        control_row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setMinimumHeight(42)
        self.start_btn.clicked.connect(
            lambda _checked=False: self._start_training(resume=False)
        )
        control_row.addWidget(self.start_btn)

        self.resume_btn = QPushButton("Resume")
        self.resume_btn.setMinimumHeight(42)
        self.resume_btn.setEnabled(False)
        self.resume_btn.clicked.connect(
            lambda _checked=False: self._start_training(resume=True)
        )
        control_row.addWidget(self.resume_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(42)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        control_row.addWidget(self.stop_btn)

        control_row.addStretch()
        root.addLayout(control_row)

        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("dialogStatus")
        progress_layout.addWidget(self.status_label)

        self.elapsed_label = QLabel("Elapsed: 00:00:00")
        self.elapsed_label.setObjectName("dialogHint")
        progress_layout.addWidget(self.elapsed_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(24)
        self.progress_bar.setFormat("0%")
        progress_layout.addWidget(self.progress_bar)

        root.addWidget(progress_group)

        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(180)
        self.log_text.setFont(QFont(get_monospace_font_family(), ModernFonts.SIZE_SM))
        log_layout.addWidget(self.log_text)

        root.addWidget(log_group)

        close_row = QHBoxLayout()
        close_row.addStretch()
        self.close_btn = QPushButton("Close")
        self.close_btn.setMinimumHeight(34)
        self.close_btn.clicked.connect(self.close)
        close_row.addWidget(self.close_btn)
        root.addLayout(close_row)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            get_dialog_style()
            + f"""
            QLabel#dialogTitle {{
                font-size: {ModernFonts.SIZE_XXL}px;
                font-weight: {ModernFonts.WEIGHT_BOLD};
                color: {ModernColors.ACCENT_INFO};
            }}
            QLabel#dialogStatus {{
                font-weight: {ModernFonts.WEIGHT_BOLD};
                color: {ModernColors.TEXT_PRIMARY};
                font-size: {ModernFonts.SIZE_BASE}px;
            }}
            """
        )

    def _set_running(self, running: bool, *, keep_progress: bool = False) -> None:
        self._is_running = running
        if running:
            if not keep_progress:
                self.progress_bar.setValue(0)
                self.progress_bar.setFormat("0%")
                self._last_progress_percent = 0
            self.start_btn.setEnabled(False)
            self.resume_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.close_btn.setEnabled(False)
            self.hours_back_spin.setEnabled(False)
            self.limit_spin.setEnabled(False)
            self.max_samples_spin.setEnabled(False)
            self.parallel_gm_check.setEnabled(False)
            return

        self.start_btn.setEnabled(True)
        self.resume_btn.setEnabled(bool(self._elapsed_seconds > 0 and self.progress_bar.value() < 100))
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.hours_back_spin.setEnabled(True)
        self.limit_spin.setEnabled(True)
        self.max_samples_spin.setEnabled(True)
        self.parallel_gm_check.setEnabled(True)
        self.worker = None

    def _start_training(self, resume: bool = False) -> None:
        if self._is_running:
            return

        is_resume = bool(resume and self._elapsed_seconds > 0)
        self._set_running(True, keep_progress=is_resume)
        self._start_elapsed_clock(reset=not is_resume)

        config = {
            "hours_back": int(self.hours_back_spin.value()),
            "limit_per_query": int(self.limit_spin.value()),
            "max_samples": int(self.max_samples_spin.value()),
        }

        self.status_label.setText("Resuming..." if is_resume else "Starting...")
        self._log(
            (
                f"Resuming LLM auto-training: hours_back={config['hours_back']}, "
                f"limit_per_query={config['limit_per_query']}, "
                f"max_samples={config['max_samples']}"
            )
            if is_resume
            else (
                f"Starting LLM auto-training: hours_back={config['hours_back']}, "
                f"limit_per_query={config['limit_per_query']}, "
                f"max_samples={config['max_samples']}"
            ),
            "info",
        )

        if bool(self.parallel_gm_check.isChecked()):
            parent = self.parent()
            if parent is not None and hasattr(parent, "_show_auto_learn"):
                try:
                    parent._show_auto_learn(auto_start=True)
                    self._log("Started Auto Train GM in parallel.", "info")
                except Exception as exc:
                    self._log(f"Failed to start Auto Train GM in parallel: {exc}", "warning")

        self.worker = LLMAutoTrainWorker(config)
        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._log)
        self.worker.finished_result.connect(self._on_finished)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

    def start_or_resume_auto_train(self) -> None:
        if self._is_running:
            return
        self._start_training(resume=bool(self._elapsed_seconds > 0))

    def _stop_training(self) -> None:
        if not self._is_running:
            return
        self.status_label.setText("Stopping...")
        self.stop_btn.setEnabled(False)
        self._log("Stop requested. Waiting for current step to finish...", "warning")
        if self.worker is not None:
            self.worker.stop()

    def _on_progress(self, percent: int, message: str) -> None:
        p = int(max(0, min(100, int(percent))))
        self._last_progress_percent = p
        self.progress_bar.setValue(p)
        self.progress_bar.setFormat(f"{p}%")
        msg = str(message or "").strip()
        if msg:
            self.status_label.setText(msg)

    def _on_finished(self, results: dict[str, Any]) -> None:
        self._stop_elapsed_clock()
        payload = dict(results or {})
        status = str(payload.get("status", "ok") or "ok").strip().lower()

        if status == "stopped":
            self._set_running(False)
            self.status_label.setText("Stopped")
            self.progress_bar.setFormat("Stopped")
            self._log("LLM auto-training stopped.", "warning")
            self.session_finished.emit(payload)
            return

        if status in {"error", "failed"}:
            err = str(payload.get("error", "LLM auto-training failed"))
            self._set_running(False)
            self.status_label.setText("Failed")
            self.progress_bar.setFormat("Failed")
            self._log(f"LLM auto-training failed: {err}", "error")
            self.session_finished.emit(payload)
            QMessageBox.critical(self, "Auto Train LLM Failed", err[:500])
            return

        self._set_running(False)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Complete")
        self.status_label.setText("Complete")
        self.resume_btn.setEnabled(False)
        self._log(
            (
                "LLM auto-training completed: "
                f"trained={payload.get('trained_samples', 0)}, "
                f"zh={payload.get('zh_samples', 0)}, "
                f"en={payload.get('en_samples', 0)}"
            ),
            "success",
        )
        self.session_finished.emit(payload)
        QMessageBox.information(
            self,
            "Auto Train LLM Complete",
            (
                "LLM auto-training completed.\n\n"
                f"Status: {payload.get('status', 'unknown')}\n"
                f"Collected: {payload.get('collected_articles', 0)}\n"
                f"Trained: {payload.get('trained_samples', 0)}\n"
                f"ZH: {payload.get('zh_samples', 0)}\n"
                f"EN: {payload.get('en_samples', 0)}"
            ),
        )

    def _on_error(self, error: str) -> None:
        self._stop_elapsed_clock()
        err = str(error or "Unknown error")
        self._set_running(False)
        self.status_label.setText("Failed")
        self.progress_bar.setFormat("Failed")
        self._log(f"Error: {err}", "error")
        self.session_finished.emit({"status": "error", "error": err})
        QMessageBox.critical(self, "Auto Train LLM Error", err[:500])

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

    def _log(self, message: str, level: str = "info") -> None:
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
        sb = self.log_text.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def closeEvent(self, event) -> None:
        if self._is_running:
            reply = QMessageBox.question(
                self,
                "Stop Auto Train LLM?",
                "LLM training is still running. Stop and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            if self.worker is not None:
                self.worker.stop()
                self.worker.wait(5000)
            self._stop_elapsed_clock()
            event.accept()
        else:
            self._stop_elapsed_clock()
            event.accept()
        super().closeEvent(event)
