from __future__ import annotations

from concurrent.futures import Future
from typing import Any

from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)
_UI_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS


def _get_trained_stock_codes(self: Any) -> list[str]:
    """Best-effort trained symbol metadata from predictor artifacts."""
    predictor = getattr(self, "predictor", None)
    if predictor is None:
        return []
    fn = getattr(predictor, "get_trained_stock_codes", None)
    if not callable(fn):
        return []
    try:
        out = fn()
    except _UI_RECOVERABLE_EXCEPTIONS:
        return []
    if not isinstance(out, list):
        return []
    return [str(x).strip() for x in out if str(x).strip()]


def _invalidate_trained_stock_cache(self: Any) -> None:
    _ = self
    return


def _sync_trained_stock_last_train_from_model(self: Any) -> None:
    _ = self
    return


def _get_trained_stock_set(self: Any) -> set[str]:
    out: set[str] = set()
    for item in _get_trained_stock_codes(self):
        code = self._ui_norm(item)
        if code:
            out.add(code)
    return out


def _is_trained_stock(self: Any, symbol: str) -> bool:
    code = self._ui_norm(symbol)
    return bool(code)


def _persist_session_bar(
    self: Any,
    symbol: str,
    interval: str,
    bar: dict[str, Any] | None,
    *,
    channel: str = "tick",
    min_gap_seconds: float = 0.9,
) -> None:
    """Persist latest 1m bar snapshot if session cache is configured."""
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
    self: Any,
    symbol: str,
    interval: str,
    payload: dict[str, Any],
) -> None:
    _ = (self, symbol, interval, payload)
    return


def _on_session_cache_write_done(self: Any, future: Future[object]) -> None:
    _ = (self, future)
    return


def _shutdown_session_cache_writer(self: Any) -> None:
    self._session_cache_io_futures = set()
    self._session_cache_io_pool = None


def _filter_trained_stocks_ui(self: Any, text: str) -> None:
    _ = (self, text)
    return


def _pin_watchlist_symbol(self: Any, code: str) -> None:
    """Ensure symbol exists in watchlist and is moved to the first row."""
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

    try:
        for row in range(self.watchlist.rowCount()):
            item = self.watchlist.item(row, 0)
            if item and self._ui_norm(item.text()) == normalized:
                self.watchlist.setCurrentCell(row, 0)
                break
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Suppressed exception in ui/app_training_ops.py", exc_info=exc)


def _on_trained_stock_activated(self: Any, item: Any) -> None:
    _ = item
    return


def _refresh_trained_stock_list(
    self: Any,
    stocks: list[str],
    query: str = "",
) -> None:
    _ = (self, stocks, query)
    return


def _update_trained_stocks_ui(
    self: Any,
    codes: list[str] | None = None,
) -> None:
    _ = (self, codes)
    return


def _focus_trained_stocks_tab(self: Any) -> None:
    _ = self
    return


def _get_infor_trained_stocks(self: Any) -> None:
    self.log("Get Infor is disabled in this build.", "info")


def _train_trained_stocks(self: Any) -> None:
    self.log("Train trained stocks is disabled in this build.", "info")


def _handle_training_drift_alarm(
    self: Any,
    result: dict[str, object] | None,
    *,
    context: str,
) -> None:
    """Log drift-guard warnings without execution-engine escalation."""
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
    if action != "rollback_recommended" and "drift_guard_block" not in failed_reasons:
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
