from __future__ import annotations

import time
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QListWidgetItem

from config.settings import CONFIG
from ui.background_tasks import WorkerThread
from ui.modern_theme import ModernColors
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)
_UI_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS
# Show full market list by default; only cap large filtered searches.
_UNIVERSE_RENDER_LIMIT = 8000
_UNIVERSE_MIN_REFRESH_GAP_S = 8.0


def _normalize_code(raw: str) -> str:
    digits = "".join(ch for ch in str(raw or "").strip() if ch.isdigit())
    if len(digits) == 6:
        return digits
    if 0 < len(digits) < 6:
        return digits.zfill(6)
    return ""


def _extract_name_map_from_spot_df(frame: Any) -> dict[str, str]:
    if frame is None or getattr(frame, "empty", True):
        return {}

    columns = list(getattr(frame, "columns", []) or [])
    code_col: str | None = None
    for candidate in (
        "\u4ee3\u7801",
        "\u80a1\u7968\u4ee3\u7801",
        "code",
        "symbol",
    ):
        if candidate in columns:
            code_col = candidate
            break
    if not code_col:
        return {}

    name_col: str | None = None
    for candidate in (
        "\u540d\u79f0",
        "\u80a1\u7968\u540d\u79f0",
        "name",
    ):
        if candidate in columns:
            name_col = candidate
            break
    if not name_col:
        return {}

    out: dict[str, str] = {}
    try:
        codes = (
            frame[code_col]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .fillna("")
            .str.zfill(6)
        )
        names = frame[name_col].astype(str).fillna("").str.strip()
        for code_raw, name_raw in zip(codes.tolist(), names.tolist(), strict=False):
            code = _normalize_code(code_raw)
            if not code or code in out:
                continue
            name = str(name_raw or "").strip()
            if name:
                out[code] = name
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Spot name map build failed: %s", exc)
    return out


def _refresh_universe_catalog(self: Any, *, force: bool = False) -> None:
    """Reload searchable stock universe from online/cache sources."""
    if not hasattr(self, "_universe_catalog"):
        self._universe_catalog = []
    if not hasattr(self, "_universe_last_refresh_ts"):
        self._universe_last_refresh_ts = 0.0

    now = float(time.time())
    last_refresh = float(getattr(self, "_universe_last_refresh_ts", 0.0) or 0.0)
    if not force and (now - last_refresh) < _UNIVERSE_MIN_REFRESH_GAP_S:
        return

    existing = self.workers.get("universe_catalog")
    if existing and existing.isRunning():
        if not force:
            return
        existing.cancel()

    if hasattr(self, "universe_status_label"):
        self.universe_status_label.setText("Universe: refreshing...")

    def load_universe() -> dict[str, Any]:
        from data.fetcher import get_spot_cache
        from data.universe import (
            get_new_listings,
            get_universe_codes,
            persist_runtime_universe_codes,
        )

        max_age_h = 0.15 if not force else 0.0
        codes_raw = list(
            get_universe_codes(
                force_refresh=bool(force),
                max_age_hours=max_age_h,
            )
            or []
        )
        new_codes_raw = list(
            get_new_listings(
                days=90,
                force_refresh=bool(force),
                max_age_seconds=2.0 if force else 5.0,
            )
            or []
        )

        ordered: list[str] = []
        seen: set[str] = set()
        new_codes: set[str] = set()

        for raw in new_codes_raw:
            code = _normalize_code(str(raw))
            if not code:
                continue
            new_codes.add(code)
            if code in seen:
                continue
            seen.add(code)
            ordered.append(code)

        for raw in codes_raw:
            code = _normalize_code(str(raw))
            if not code or code in seen:
                continue
            seen.add(code)
            ordered.append(code)

        if not ordered:
            for raw in list(getattr(CONFIG, "STOCK_POOL", []) or []):
                code = _normalize_code(str(raw))
                if not code or code in seen:
                    continue
                seen.add(code)
                ordered.append(code)

        if ordered:
            try:
                persist_runtime_universe_codes(
                    ordered,
                    source="ui_universe_refresh",
                )
            except _UI_RECOVERABLE_EXCEPTIONS as exc:
                log.debug("Universe runtime persistence skipped: %s", exc)

        try:
            spot_df = get_spot_cache().get(force_refresh=bool(force))
            name_map = _extract_name_map_from_spot_df(spot_df)
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Universe spot-name refresh skipped: %s", exc)
            name_map = {}

        items = [
            {
                "code": code,
                "name": str(name_map.get(code, "")),
                "is_new": bool(code in new_codes),
            }
            for code in ordered
        ]
        return {
            "items": items,
            "total": int(len(items)),
            "new_count": int(len(new_codes)),
            "fetched_at": float(time.time()),
        }

    worker = WorkerThread(load_universe, timeout_seconds=120)
    self._track_worker(worker)
    worker.result.connect(self._on_universe_catalog_loaded)
    worker.error.connect(self._on_universe_catalog_error)
    self.workers["universe_catalog"] = worker
    worker.start()


def _on_universe_catalog_loaded(self: Any, payload: Any) -> None:
    self.workers.pop("universe_catalog", None)
    data = payload if isinstance(payload, dict) else {}
    rows = list(data.get("items", []) or [])
    self._universe_catalog = rows
    self._universe_last_refresh_ts = float(
        data.get("fetched_at", time.time()) or time.time()
    )

    query = ""
    if hasattr(self, "universe_search_input"):
        query = str(self.universe_search_input.text() or "")
    self._filter_universe_list(query)

    query = str(query or "").strip()
    if query:
        return
    total = int(data.get("total", len(rows)) or len(rows))
    new_count = int(data.get("new_count", 0) or 0)
    if hasattr(self, "universe_status_label"):
        if new_count > 0:
            self.universe_status_label.setText(
                f"Universe: {total:,} stocks ({new_count} new)"
            )
        else:
            self.universe_status_label.setText(f"Universe: {total:,} stocks")
    if bool(getattr(self, "_startup_loading_active", False)) and hasattr(
        self, "_complete_startup_loading"
    ):
        try:
            self._complete_startup_loading("Ready")
        except _UI_RECOVERABLE_EXCEPTIONS:
            pass


def _on_universe_catalog_error(self: Any, error: str) -> None:
    self.workers.pop("universe_catalog", None)
    msg = str(error or "").strip()
    if hasattr(self, "universe_status_label"):
        self.universe_status_label.setText("Universe refresh failed")
    if msg:
        self.log(f"Universe refresh failed: {msg}", "warning")
    if bool(getattr(self, "_startup_loading_active", False)) and hasattr(
        self, "_complete_startup_loading"
    ):
        try:
            self._complete_startup_loading("Ready")
        except _UI_RECOVERABLE_EXCEPTIONS:
            pass


def _filter_universe_list(self: Any, text: str) -> None:
    """Filter and render universe list by code/name query."""
    if not hasattr(self, "universe_list"):
        return
    catalog = list(getattr(self, "_universe_catalog", []) or [])
    query = str(text or "").strip().lower()

    if query:
        matched = [
            row
            for row in catalog
            if query in str(row.get("code", "")).lower()
            or query in str(row.get("name", "")).lower()
        ]
    else:
        matched = catalog

    if query:
        display = matched[:_UNIVERSE_RENDER_LIMIT]
    else:
        display = matched

    self.universe_list.blockSignals(True)
    self.universe_list.clear()
    for row in display:
        code = str(row.get("code", "") or "").strip()
        if not code:
            continue
        name = str(row.get("name", "") or "").strip()
        label = f"{code}  {name}" if name else code
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, code)
        if bool(row.get("is_new", False)):
            item.setForeground(QColor(ModernColors.ACCENT_SUCCESS))
            item.setToolTip("Newly listed")
        self.universe_list.addItem(item)
    self.universe_list.blockSignals(False)

    if hasattr(self, "universe_status_label"):
        total = int(len(catalog))
        if query:
            self.universe_status_label.setText(
                f"Universe: {len(matched):,} matches "
                f"(showing {len(display):,}/{total:,})"
            )
        else:
            self.universe_status_label.setText(
                f"Universe: showing {len(display):,}/{total:,}"
            )


def _on_universe_item_activated(self: Any, item: Any) -> None:
    """Activate a stock from universe search list and run analysis."""
    if item is None:
        return
    try:
        raw = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
    except _UI_RECOVERABLE_EXCEPTIONS:
        raw = ""
    if not raw:
        raw = str(getattr(item, "text", lambda: "")() or "").split()[0]
    code = self._ui_norm(raw)
    if not code:
        return

    if hasattr(self, "stock_input"):
        self.stock_input.setText(code)
    try:
        self._pin_watchlist_symbol(code)
    except _UI_RECOVERABLE_EXCEPTIONS:
        pass

    try:
        self._on_watchlist_click(-1, 0, code_override=code)
        return
    except _UI_RECOVERABLE_EXCEPTIONS:
        pass

    self._analyze_stock()
