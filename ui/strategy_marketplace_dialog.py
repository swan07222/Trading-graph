from __future__ import annotations

from typing import Dict, List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from analysis.strategy_marketplace import StrategyMarketplace
from utils.logger import get_logger

log = get_logger(__name__)


class StrategyMarketplaceDialog(QDialog):
    """Manage installed strategy scripts with integrity/status visibility."""

    COL_ENABLE = 0
    COL_ID = 1
    COL_NAME = 2
    COL_VERSION = 3
    COL_AUTHOR = 4
    COL_CATEGORY = 5
    COL_RISK = 6
    COL_INTEGRITY = 7
    COL_DESCRIPTION = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Strategy Marketplace")
        self.setMinimumSize(980, 560)
        self.resize(1100, 650)
        self._marketplace = StrategyMarketplace()
        self._rows: List[Dict] = []
        self._setup_ui()
        self._reload()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        info = QLabel(
            "Enable/disable strategy scripts. "
            "Only enabled and integrity-valid scripts are executed."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #8b949e; font-size: 12px;")
        layout.addWidget(info)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            [
                "Enable",
                "ID",
                "Name",
                "Version",
                "Author",
                "Category",
                "Risk",
                "Integrity",
                "Description",
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        btns = QHBoxLayout()
        self.reload_btn = QPushButton("Reload")
        self.reload_btn.clicked.connect(self._reload)
        btns.addWidget(self.reload_btn)

        self.enable_all_btn = QPushButton("Enable All")
        self.enable_all_btn.clicked.connect(self._enable_all)
        btns.addWidget(self.enable_all_btn)

        self.disable_all_btn = QPushButton("Disable All")
        self.disable_all_btn.clicked.connect(self._disable_all)
        btns.addWidget(self.disable_all_btn)
        btns.addStretch()

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._save)
        btns.addWidget(self.save_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btns.addWidget(self.close_btn)

        layout.addLayout(btns)

    def _add_item(self, row: int, col: int, text: str) -> None:
        item = QTableWidgetItem(str(text))
        if col == self.COL_INTEGRITY:
            t = str(text).lower()
            if t == "ok":
                item.setForeground(Qt.GlobalColor.darkGreen)
            elif t in ("mismatch", "error", "missing"):
                item.setForeground(Qt.GlobalColor.red)
        self.table.setItem(row, col, item)

    def _reload(self) -> None:
        self._rows = self._marketplace.list_entries()
        self.table.setRowCount(len(self._rows))
        for row, entry in enumerate(self._rows):
            enable_item = QTableWidgetItem("")
            enable_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
            )
            check = Qt.CheckState.Checked if entry.get("enabled", False) else Qt.CheckState.Unchecked
            enable_item.setCheckState(check)
            self.table.setItem(row, self.COL_ENABLE, enable_item)

            self._add_item(row, self.COL_ID, entry.get("id", ""))
            self._add_item(row, self.COL_NAME, entry.get("name", ""))
            self._add_item(row, self.COL_VERSION, entry.get("version", ""))
            self._add_item(row, self.COL_AUTHOR, entry.get("author", ""))
            self._add_item(row, self.COL_CATEGORY, entry.get("category", ""))
            self._add_item(row, self.COL_RISK, entry.get("risk_level", ""))
            self._add_item(row, self.COL_INTEGRITY, entry.get("integrity", "unknown"))
            self._add_item(row, self.COL_DESCRIPTION, entry.get("description", ""))
        self.table.resizeColumnsToContents()

    def _enable_all(self) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.COL_ENABLE)
            if item is not None:
                item.setCheckState(Qt.CheckState.Checked)

    def _disable_all(self) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.COL_ENABLE)
            if item is not None:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _save(self) -> None:
        enabled_ids: List[str] = []
        for row, entry in enumerate(self._rows):
            integrity = str(entry.get("integrity", "")).lower()
            if integrity in ("mismatch", "missing", "error"):
                continue
            item = self.table.item(row, self.COL_ENABLE)
            if item is None:
                continue
            if item.checkState() == Qt.CheckState.Checked:
                enabled_ids.append(str(entry.get("id", "")))
        self._marketplace.save_enabled_ids(enabled_ids)
        log.info("Saved strategy marketplace enabled list (%d strategies)", len(enabled_ids))
        self.accept()
