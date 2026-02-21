from __future__ import annotations

import time
from concurrent.futures import Future
from importlib import import_module
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem

from config.settings import CONFIG
from core.types import AutoTradeMode
from ui.background_tasks import WorkerThread
from utils.logger import get_logger

log = get_logger(__name__)

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
    """Normalized trained stock set from metadata cache/predictor."""
    raw = list(getattr(self, "_trained_stock_codes_cache", []) or [])
    if not raw and self.predictor is not None:
        raw = self._get_trained_stock_codes()
        if raw:
            self._trained_stock_codes_cache = list(raw)
    out: set[str] = set()
    for item in raw:
        code = self._ui_norm(item)
        if code:
            out.add(code)
    return out

def _is_trained_stock(self, symbol: str) -> bool:
    """Whether symbol is part of the currently loaded trained stock set."""
    code = self._ui_norm(symbol)
    if not code:
        return False
    return code in self._get_trained_stock_set()

def _persist_session_bar(
    self,
    symbol: str,
    interval: str,
    bar: dict[str, Any] | None,
    *,
    channel: str = "tick",
    min_gap_seconds: float = 0.9,
) -> None:
    """Persist latest live bar snapshot to session cache."""
    if self._session_bar_cache is None or not isinstance(bar, dict):
        return
    try:
        iv = self._normalize_interval_token(interval)
        # Persist session bars only in canonical 1m stream.
        # Coarser intervals are display-only and must be derived from 1m.
        if iv != "1m":
            return
        now_ts = time.time()
        key = f"{symbol}:{iv}:{channel}"
        min_gap = float(max(0.0, min_gap_seconds))
        lock = getattr(self, "_session_cache_write_lock", None)
        if lock is None:
            prev_ts = float(self._last_session_cache_write_ts.get(key, 0.0))
            if (now_ts - prev_ts) < min_gap:
                return
            self._last_session_cache_write_ts[key] = now_ts
        else:
            with lock:
                prev_ts = float(self._last_session_cache_write_ts.get(key, 0.0))
                if (now_ts - prev_ts) < min_gap:
                    return
                self._last_session_cache_write_ts[key] = now_ts
        payload = dict(bar)
        payload["interval"] = iv
        payload["source"] = str(payload.get("source", "") or "tencent_rt")
        submit = getattr(self, "_submit_session_cache_write", None)
        if callable(submit):
            submit(symbol, iv, payload)
        else:
            self._session_bar_cache.append_bar(symbol, iv, payload)
    except _UI_RECOVERABLE_EXCEPTIONS as e:
        log.debug(f"Session cache persist failed for {symbol}: {e}")

def _submit_session_cache_write(
    self,
    symbol: str,
    interval: str,
    payload: dict[str, Any],
) -> None:
    cache = self._session_bar_cache
    pool = getattr(self, "_session_cache_io_pool", None)
    if cache is None:
        return
    if pool is None:
        try:
            cache.append_bar(symbol, interval, dict(payload))
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Session cache write failed for %s: %s", symbol, exc)
        return
    lock = getattr(self, "_session_cache_io_lock", None)
    futures = getattr(self, "_session_cache_io_futures", None)
    if lock is None or futures is None:
        try:
            pool.submit(cache.append_bar, symbol, interval, dict(payload))
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Failed to enqueue session cache write for %s: %s", symbol, exc)
        return
    with lock:
        if len(futures) >= 256:
            log.debug(
                "Session cache write queue full; dropping %s (%s)",
                symbol,
                interval,
            )
            return
    try:
        future = pool.submit(cache.append_bar, symbol, interval, dict(payload))
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Failed to enqueue session cache write for %s: %s", symbol, exc)
        return
    with lock:
        futures.add(future)
    future.add_done_callback(self._on_session_cache_write_done)

def _on_session_cache_write_done(self, future: Future[object]) -> None:
    lock = getattr(self, "_session_cache_io_lock", None)
    futures = getattr(self, "_session_cache_io_futures", None)
    if lock is not None and futures is not None:
        with lock:
            futures.discard(future)
    try:
        future.result()
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Async session cache write failed: %s", exc)

def _shutdown_session_cache_writer(self) -> None:
    pool = getattr(self, "_session_cache_io_pool", None)
    if pool is None:
        return
    lock = getattr(self, "_session_cache_io_lock", None)
    futures = getattr(self, "_session_cache_io_futures", None)
    if lock is not None and futures is not None:
        with lock:
            pending = list(futures)
    else:
        pending = []
    for fut in pending:
        try:
            fut.result(timeout=0.3)
        except FuturesTimeout:
            break
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Session cache writer flush failed: %s", exc)
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Session cache writer shutdown failed: %s", exc)
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
        return

def _focus_trained_stocks_tab(self) -> None:
    """Focus the right-panel trained-stocks tab."""
    tabs = getattr(self, "right_tabs", None)
    idx = int(getattr(self, "_trained_tab_index", -1))
    if tabs is not None and idx >= 0:
        tabs.setCurrentIndex(idx)

def _get_infor_trained_stocks(self) -> None:
    """
    Refresh 29-day AKShare history for all trained stocks.

    If data already exists in the target window, only fetches from the
    last saved timestamp forward.
    """
    raw_codes = self._get_trained_stock_codes()
    codes = list(
        dict.fromkeys(
            self._ui_norm(x) for x in list(raw_codes or []) if self._ui_norm(x)
        )
    )
    if not codes:
        self.log("No trained stocks found. Load/train a model first.", "warning")
        return

    old_worker = self.workers.get("get_infor")
    if old_worker and old_worker.isRunning():
        self.log("Get Infor is already running.", "info")
        return

    if hasattr(self, "get_infor_btn"):
        self.get_infor_btn.setEnabled(False)
    self.progress.setRange(0, 0)
    self.progress.show()
    self.status_label.setText(
        f"Get Infor: syncing {len(codes)} trained stocks..."
    )
    self.log(
        (
            "Get Infor started: AKShare sync for "
            f"{len(codes)} trained stocks (last 29 days, incremental)."
        ),
        "info",
    )

    def _task() -> Any:
        from data.fetcher import get_fetcher

        fetcher = get_fetcher()
        return fetcher.refresh_trained_stock_history(
            codes,
            interval="1m",
            window_days=29,
            allow_online=True,
            sync_session_cache=True,
            replace_realtime_after_close=True,
        )

    worker = WorkerThread(
        _task,
        timeout_seconds=float(max(180, int(len(codes)) * 18)),
    )
    self._track_worker(worker)
    self.workers["get_infor"] = worker

    def _finalize() -> None:
        self.progress.hide()
        if hasattr(self, "get_infor_btn"):
            self.get_infor_btn.setEnabled(True)
        self.workers.pop("get_infor", None)

    def _on_done(res: object) -> None:
        _finalize()
        report = dict(res or {})
        total = int(report.get("total", 0) or 0)
        updated = int(report.get("updated", 0) or 0)
        cached = int(report.get("cached", 0) or 0)
        purged_map = dict(report.get("purged_realtime_rows", {}) or {})
        purged = int(
            sum(int(v or 0) for v in purged_map.values())
        )
        errors = dict(report.get("errors", {}) or {})
        if errors:
            self.log(
                (
                    "Get Infor completed with warnings: "
                    f"updated={updated}, cached={cached}, purged_rt={purged}, "
                    f"errors={len(errors)}, total={total}."
                ),
                "warning",
            )
            bad_codes = ", ".join(list(errors.keys())[:8])
            if bad_codes:
                self.log(f"Get Infor error codes: {bad_codes}", "warning")
        else:
            self.log(
                (
                    "Get Infor completed: "
                    f"updated={updated}, cached={cached}, purged_rt={purged}, "
                    f"total={total}."
                ),
                "success",
            )
        self.status_label.setText("Get Infor completed")

        try:
            sym = self._ui_norm(self.stock_input.text())
            iv = self._normalize_interval_token(self.interval_combo.currentText())
            if sym:
                self._queue_history_refresh(sym, iv)
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    def _on_error(err: str) -> None:
        _finalize()
        self.status_label.setText("Get Infor failed")
        self.log(f"Get Infor failed: {err}", "error")

    worker.result.connect(_on_done)
    worker.error.connect(_on_error)
    worker.start()

def _train_trained_stocks(self) -> None:
    """
    Train only already-trained stocks using latest cached data.

    A dialog asks for stock count (N). The model is retrained on the
    N stocks with the oldest last-train timestamps.
    """
    trained = list(
        dict.fromkeys(
            self._ui_norm(x) for x in self._get_trained_stock_codes()
            if self._ui_norm(x)
        )
    )
    self._sync_trained_stock_last_train_from_model()

    pending_codes: set[str] = set()
    try:
        fetcher = getattr(self.predictor, "fetcher", None)
        if fetcher is None:
            from data.fetcher import get_fetcher
            fetcher = get_fetcher()
        reconcile_fn = getattr(fetcher, "reconcile_pending_cache_sync", None)
        if callable(reconcile_fn):
            try:
                reconcile_fn(codes=list(trained), interval="1m")
            except TypeError:
                reconcile_fn()
        pending_fn = getattr(fetcher, "get_pending_reconcile_codes", None)
        if callable(pending_fn):
            pending_codes = {
                self._ui_norm(x)
                for x in list(pending_fn(interval="1m") or [])
                if self._ui_norm(x)
            }
    except _UI_RECOVERABLE_EXCEPTIONS:
        pending_codes = set()

    if pending_codes:
        before = int(len(trained))
        trained = [c for c in trained if c not in pending_codes]
        removed = int(max(0, before - len(trained)))
        if removed > 0:
            self.log(
                (
                    f"Skipped {removed} stock(s) with pending cache reconcile. "
                    "Press Get Infor to finish sync before training."
                ),
                "warning",
            )

    if not trained:
        if pending_codes:
            self.log(
                "All trained stocks are waiting for cache reconcile. Run Get Infor first.",
                "warning",
            )
        else:
            self.log("No trained stocks found. Load/train a model first.", "warning")
        return

    try:
        from .dialogs import TrainTrainedStocksDialog
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        self.log(f"Train trained stocks dialog unavailable: {exc}", "error")
        return

    dialog = TrainTrainedStocksDialog(
        trained_codes=trained,
        last_train_map=dict(self._trained_stock_last_train or {}),
        parent=self,
    )
    dialog.exec()

    result = getattr(dialog, "training_result", None)
    if not isinstance(result, dict):
        return

    if str(result.get("status", "")).strip().lower() != "complete":
        status = str(result.get("status", "cancelled")).strip().lower()
        if status == "cancelled":
            self.log("Train trained stocks cancelled.", "info")
        return
    self._handle_training_drift_alarm(
        result,
        context="train_trained_stocks",
    )

    trained_codes = list(
        dict.fromkeys(
            self._ui_norm(x)
            for x in list(
                result.get("trained_stock_codes")
                or result.get("selected_codes")
                or []
            )
            if self._ui_norm(x)
        )
    )
    if trained_codes:
        trained_at = str(
            result.get("trained_at") or datetime.now().isoformat(timespec="seconds")
        )
        self._record_trained_stock_last_train(
            trained_codes,
            trained_at=trained_at,
        )
        self._update_trained_stocks_ui()
        self.log(
            (
                "Train trained stocks completed: "
                f"{len(trained_codes)} stock(s)."
            ),
            "success",
        )

    self._init_components()

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

