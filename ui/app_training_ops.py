from __future__ import annotations

import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime
from importlib import import_module
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem

from core.types import AutoTradeMode
from ui.background_tasks import WorkerThread
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)
_UI_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS

def _lazy_get(module: str, name: str) -> Any:
    return getattr(import_module(module), name)

def _get_trained_stock_codes(self) -> list[str]:
    """Read trained stock list from loaded predictor metadata."""
    if self.predictor is None:
        return []
    try:
        fn = getattr(self.predictor, "get_trained_stock_codes", None)
        if callable(fn):
            out = fn()
            if isinstance(out, list):
                return [
                    str(x).strip()
                    for x in out
                    if str(x).strip()
                ]
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)
    return []

def _invalidate_trained_stock_cache(self) -> None:
    """Compatibility no-op: trained stock cache has been removed."""
    return

def _sync_trained_stock_last_train_from_model(self) -> None:
    """Use loaded model artifacts as source-of-truth for last-train metadata."""
    if self.predictor is None:
        return
    fn = getattr(self.predictor, "get_trained_stock_last_train", None)
    if not callable(fn):
        return
    try:
        raw = fn()
    except _UI_RECOVERABLE_EXCEPTIONS:
        return
    if not isinstance(raw, dict):
        return
    out: dict[str, str] = {}
    for k, v in raw.items():
        code = self._ui_norm(str(k or ""))
        if not code:
            continue
        ts = str(v or "").strip()
        if not ts:
            continue
        out[code] = ts
    if out == dict(self._trained_stock_last_train or {}):
        return
    self._trained_stock_last_train = out
    self._save_trained_stock_last_train_meta()

def _get_trained_stock_set(self) -> set[str]:
    """Normalized trained stock set from predictor metadata."""
    raw = self._get_trained_stock_codes()
    out: set[str] = set()
    for item in raw:
        code = self._ui_norm(item)
        if code:
            out.add(code)
    return out

def _is_trained_stock(self, symbol: str) -> bool:
    """Treat any valid symbol as eligible for model inference."""
    code = self._ui_norm(symbol)
    return bool(code)

def _persist_session_bar(
    self,
    symbol: str,
    interval: str,
    bar: dict[str, Any] | None,
    *,
    channel: str = "tick",
    min_gap_seconds: float = 0.9,
) -> None:
    """Persist latest live 1m bar snapshot when a session cache is present."""
    _ = (channel, min_gap_seconds)
    if self._session_bar_cache is None or not isinstance(bar, dict):
        return
    iv = self._normalize_interval_token(interval)
    if iv != "1m":
        return
    payload = dict(bar)
    payload["interval"] = iv
    payload["source"] = str(payload.get("source", "") or "tencent_rt")
    try:
        self._session_bar_cache.append_bar(symbol, iv, payload)
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Session cache persist failed for %s: %s", symbol, exc)

def _submit_session_cache_write(
    self,
    symbol: str,
    interval: str,
    payload: dict[str, Any],
) -> None:
    _ = (self, symbol, interval, payload)
    return

def _on_session_cache_write_done(self, future: Future[object]) -> None:
    _ = (self, future)
    return

def _shutdown_session_cache_writer(self) -> None:
    self._session_cache_io_futures = set()
    self._session_cache_io_pool = None

def _filter_trained_stocks_ui(self, text: str) -> None:
    """Filter right-panel trained stock list by search query."""
    self._refresh_trained_stock_list(
        list(getattr(self, "_trained_stock_codes_cache", [])),
        str(text or ""),
    )

def _pin_watchlist_symbol(self, code: str) -> None:
    """Ensure symbol is present and visible in watchlist (move to top)."""
    normalized = self._ui_norm(code)
    if not normalized:
        return

    changed = False
    if normalized in self.watch_list:
        if self.watch_list and self.watch_list[0] != normalized:
            self.watch_list.remove(normalized)
            self.watch_list.insert(0, normalized)
            changed = True
    else:
        if len(self.watch_list) >= self.MAX_WATCHLIST_SIZE:
            self.watch_list = self.watch_list[: self.MAX_WATCHLIST_SIZE - 1]
        self.watch_list.insert(0, normalized)
        changed = True

    if changed:
        self._update_watchlist()
        if self.executor and self.executor.auto_trader:
            try:
                self.executor.auto_trader.update_watchlist(self.watch_list)
            except _UI_RECOVERABLE_EXCEPTIONS as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    # Keep selection aligned with the active symbol.
    try:
        for row in range(self.watchlist.rowCount()):
            item = self.watchlist.item(row, 0)
            if item and self._ui_norm(item.text()) == normalized:
                self.watchlist.setCurrentCell(row, 0)
                break
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)

def _on_trained_stock_activated(self, item: QListWidgetItem) -> None:
    """Load selected trained stock from right-panel list."""
    if item is None:
        return
    code_hint = ""
    try:
        code_hint = str(
            item.data(Qt.ItemDataRole.UserRole) or ""
        ).strip()
    except _UI_RECOVERABLE_EXCEPTIONS:
        code_hint = ""
    code = self._ui_norm(code_hint or item.text())
    if not code:
        return
    interval = self._normalize_interval_token(
        self.interval_combo.currentText()
    )
    try:
        self._queue_history_refresh(code, interval)
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)
    try:
        target_lookback = int(
            max(
                self._recommended_lookback(interval),
                self._trained_stock_window_bars(interval),
            )
        )
        if int(self.lookback_spin.value()) < target_lookback:
            self.lookback_spin.setValue(target_lookback)
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)
    self._pin_watchlist_symbol(code)
    self.stock_input.setText(code)
    try:
        self._ensure_feed_subscription(code)
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)
    self._on_watchlist_click(-1, -1, code_override=code)

def _refresh_trained_stock_list(
    self, stocks: list[str], query: str = ""
) -> None:
    """Render searchable trained-stock list in the right panel."""
    if not hasattr(self, "trained_stock_list"):
        return
    all_codes = [
        str(x).strip()
        for x in list(stocks or [])
        if str(x).strip()
    ]
    q = str(query or "").strip().lower()
    if q:
        view_codes = [c for c in all_codes if q in c.lower()]
    else:
        view_codes = all_codes

    self.trained_stock_list.clear()
    if view_codes:
        for code in view_codes:
            last_train = str(
                self._trained_stock_last_train.get(code, "")
            ).strip()
            last_text = self._format_last_train_text(last_train)
            item = QListWidgetItem(
                f"{code}  | last train: {last_text}"
            )
            item.setData(Qt.ItemDataRole.UserRole, code)
            item.setToolTip(
                f"Stock: {code}\n"
                f"Last Train: {last_text}"
            )
            self.trained_stock_list.addItem(item)
    elif all_codes and q:
        self.trained_stock_list.addItem(
            "No matching trained stocks for current search."
        )
    else:
        self.trained_stock_list.addItem(
            "No trained stock metadata found in the loaded model."
        )

    if hasattr(self, "trained_stock_count_label"):
        self.trained_stock_count_label.setText(
            f"Trained: {len(view_codes)} / {len(all_codes)}"
        )

    tabs = getattr(self, "right_tabs", None)
    idx = int(getattr(self, "_trained_tab_index", -1))
    if tabs is not None and idx >= 0:
        tabs.setTabText(idx, f"Trained Stocks ({len(all_codes)})")

def _update_trained_stocks_ui(self, codes: list[str] | None = None) -> None:
    """Refresh trained-stock metadata section in AI panel."""
    # Always clear TTL cache so UI reflects current model artifacts.
    self._invalidate_trained_stock_cache()
    self._sync_trained_stock_last_train_from_model()
    stocks = list(codes) if isinstance(codes, list) else self._get_trained_stock_codes()
    self._trained_stock_codes_cache = list(stocks)

    if hasattr(self, "trained_stocks_label"):
        if not stocks:
            self.trained_stocks_label.setText("Trained Stocks: --")
        else:
            self.trained_stocks_label.setText(
                f"Trained Stocks: {len(stocks)}"
            )

    query = ""
    if hasattr(self, "trained_stock_search"):
        try:
            query = self.trained_stock_search.text()
        except _UI_RECOVERABLE_EXCEPTIONS:
            query = ""
    self._refresh_trained_stock_list(stocks, query=query)

    if not stocks:
        # User may remove model artifacts; clear stale metadata immediately.
        if getattr(self, "_trained_stock_last_train", None):
            self._trained_stock_last_train = {}
            self._save_trained_stock_last_train_meta()
        return

def _focus_trained_stocks_tab(self) -> None:
    """Focus the right-panel trained-stocks tab."""
    tabs = getattr(self, "right_tabs", None)
    idx = int(getattr(self, "_trained_tab_index", -1))
    if tabs is not None and idx >= 0:
        tabs.setCurrentIndex(idx)

def _get_infor_trained_stocks(self) -> None:
    self.log("Get Infor is disabled in this build.", "info")
    return

def _train_trained_stocks(self) -> None:
    self.log("Train trained stocks is disabled in this build.", "info")
    return

def _handle_training_drift_alarm(
    self,
    result: dict[str, object] | None,
    *,
    context: str,
) -> None:
    """Escalate trainer drift alarms and force auto-trade to MANUAL."""
    payload = result if isinstance(result, dict) else {}
    drift_guard = payload.get("drift_guard", {})
    quality_gate = payload.get("quality_gate", {})
    if not isinstance(drift_guard, dict):
        drift_guard = {}
    if not isinstance(quality_gate, dict):
        quality_gate = {}

    action = str(drift_guard.get("action", "") or "").strip().lower()
    failed_reasons = {
        str(x).strip().lower()
        for x in list(quality_gate.get("failed_reasons", []) or [])
        if str(x).strip()
    }
    if (
        action != "rollback_recommended"
        and "drift_guard_block" not in failed_reasons
    ):
        return

    try:
        score_drop = float(drift_guard.get("score_drop", 0.0) or 0.0)
    except (TypeError, ValueError):
        score_drop = 0.0
    try:
        acc_drop = float(drift_guard.get("accuracy_drop", 0.0) or 0.0)
    except (TypeError, ValueError):
        acc_drop = 0.0

    reason = (
        f"{context}: model drift guard triggered "
        f"(action={action or 'unknown'}, "
        f"score_drop={score_drop:.3f}, accuracy_drop={acc_drop:.3f})"
    )
    self.log(reason, "warning")

    try:
        ExecutionEngine = _lazy_get("trading.executor", "ExecutionEngine")
        handled = int(
            ExecutionEngine.trigger_model_drift_alarm(
                reason=reason,
                severity="critical",
                metadata={
                    "context": str(context),
                    "action": str(action),
                    "score_drop": float(score_drop),
                    "accuracy_drop": float(acc_drop),
                },
            )
        )
        if handled > 0:
            self._auto_trade_mode = AutoTradeMode.MANUAL
            self._apply_auto_trade_mode(AutoTradeMode.MANUAL)
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        self.log(f"Drift alarm escalation failed: {exc}", "warning")
