from __future__ import annotations

import time
from typing import Any

from PyQt6.QtWidgets import QApplication

from ui.background_tasks import WorkerThread, normalize_stock_code
from utils.logger import get_logger

log = get_logger(__name__)


class MainAppCommonMixin:
    """Shared utility helpers for MainApp UI/controller code."""

    def _ui_norm(self, text: str) -> str:
        """Normalize stock code for UI comparison."""
        return normalize_stock_code(text)

    @staticmethod
    def _safe_list(values: Any) -> list[Any]:
        """Convert optional iterables to list without truthiness checks."""
        if values is None:
            return []
        try:
            return list(values)
        except TypeError:
            return []

    def _set_initial_window_geometry(self) -> None:
        """Fit initial window to available screen so bottom controls remain visible."""
        try:
            screen = QApplication.primaryScreen()
            if screen is None:
                self.setGeometry(80, 60, 1360, 780)
                return
            avail = screen.availableGeometry()
            width = min(max(1120, int(avail.width() * 0.90)), int(avail.width()))
            height = min(max(700, int(avail.height() * 0.90)), int(avail.height()))
            x = int(avail.left() + ((avail.width() - width) / 2))
            y = int(avail.top() + ((avail.height() - height) / 2))
            self.setGeometry(x, y, width, height)
        except RuntimeError:
            self.setGeometry(80, 60, 1360, 780)

    def _track_worker(self, worker: WorkerThread) -> None:
        """Track worker lifecycle so threads are never orphaned."""
        self._active_workers.add(worker)

        def _drop(*_args: object) -> None:
            self._active_workers.discard(worker)

        worker.finished.connect(_drop)

    def _model_interval_to_ui_token(self, interval: str) -> str:
        """Normalize model interval metadata to available UI tokens."""
        iv = str(interval or "").strip().lower()
        if iv == "1h":
            return "60m"
        return iv

    def _normalize_interval_token(
        self,
        interval: str | None,
        *,
        fallback: str = "1m",
    ) -> str:
        """Normalize interval aliases used across feed/cache/UI paths."""
        iv = str(interval or "").strip().lower()
        if not iv:
            return str(fallback or "1m").strip().lower()

        aliases = {
            "1h": "60m",
            "60min": "60m",
            "60mins": "60m",
            "60minute": "60m",
            "60minutes": "60m",
            "1hour": "60m",
            "1day": "1d",
            "day": "1d",
            "daily": "1d",
            "1440m": "1d",
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "1minute": "1m",
            "5minute": "5m",
            "15minute": "15m",
            "30minute": "30m",
            "week": "1wk",
            "weekly": "1wk",
            "month": "1mo",
            "monthly": "1mo",
        }
        return aliases.get(iv, iv)

    def _debug_console(
        self,
        key: str,
        message: str,
        *,
        min_gap_seconds: float = 2.0,
        level: str = "warning",
    ) -> None:
        """Throttled debug message to UI system-log + backend logger."""
        if not bool(getattr(self, "_debug_console_enabled", False)):
            return
        try:
            now = time.monotonic()
            prev = float(self._debug_console_last_emit.get(key, 0.0))
            if (now - prev) < float(max(0.0, min_gap_seconds)):
                return
            self._debug_console_last_emit[key] = now
            txt = f"[DBG] {message}"
            if hasattr(self, "log"):
                self.log(txt, level=level)
            else:
                log.warning(txt)
        except (AttributeError, TypeError, ValueError) as exc:
            log.debug("Suppressed exception in ui/app_common.py", exc_info=exc)

    def _predictor_runtime_ready(self) -> bool:
        """Whether predictor has sufficient runtime artifacts for inference."""
        predictor = getattr(self, "predictor", None)
        if predictor is None:
            return False

        ready_fn = getattr(predictor, "_models_ready_for_runtime", None)
        if callable(ready_fn):
            try:
                return bool(ready_fn())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

        processor = getattr(predictor, "processor", None)
        scaler_ready = bool(
            processor is not None and getattr(processor, "is_fitted", True)
        )
        ensemble_ready = bool(getattr(predictor, "ensemble", None) is not None)
        forecaster_ready = bool(getattr(predictor, "forecaster", None) is not None)
        return bool(scaler_ready and (ensemble_ready or forecaster_ready))

    def _predictor_forecast_ready(self) -> bool:
        """Whether predictor can render guessed curve in fallback mode."""
        predictor = getattr(self, "predictor", None)
        if predictor is None:
            return False

        ready_fn = getattr(predictor, "_forecast_ready_for_runtime", None)
        if callable(ready_fn):
            try:
                return bool(ready_fn())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

        processor = getattr(predictor, "processor", None)
        return bool(
            processor is not None and getattr(processor, "is_fitted", True)
        )

    def _predictor_model_summary(self) -> dict[str, object]:
        """Best-effort predictor capability summary for UI status text."""
        predictor = getattr(self, "predictor", None)
        if predictor is None:
            return {
                "runtime_ready": False,
                "forecast_ready": False,
                "has_ensemble": False,
                "has_forecaster": False,
                "ensemble_models": 0,
            }

        has_ensemble = bool(getattr(predictor, "ensemble", None) is not None)
        has_forecaster = bool(getattr(predictor, "forecaster", None) is not None)
        ensemble_models = 0
        if has_ensemble:
            try:
                models_obj = getattr(predictor.ensemble, "models", {})
                ensemble_models = int(len(models_obj)) if models_obj is not None else 0
            except (AttributeError, RuntimeError, TypeError, ValueError):
                ensemble_models = 0

        return {
            "runtime_ready": bool(self._predictor_runtime_ready()),
            "forecast_ready": bool(self._predictor_forecast_ready()),
            "has_ensemble": bool(has_ensemble),
            "has_forecaster": bool(has_forecaster),
            "ensemble_models": int(max(0, ensemble_models)),
        }
