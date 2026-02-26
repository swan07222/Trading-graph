"""UI package exports with lazy imports.

Avoid importing heavy optional dependencies at package import time.
"""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "MainApp": (".app", "MainApp"),
    "run_app": (".app", "run_app"),
    "StockChart": (".charts", "StockChart"),
    "AutoLearnDialog": (".auto_learn_dialog", "AutoLearnDialog"),
    "show_auto_learn_dialog": (".auto_learn_dialog", "show_auto_learn_dialog"),
    "SignalPanel": (".widgets", "SignalPanel"),
    "PositionTable": (".widgets", "PositionTable"),
    "LogWidget": (".widgets", "LogWidget"),
    "TrainingDialog": (".dialogs", "TrainingDialog"),
    "BacktestDialog": (".dialogs", "BacktestDialog"),
    "ScreenerProfileDialog": (".dialogs", "ScreenerProfileDialog"),
    "NewsPanel": (".news_widget", "NewsPanel"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(str(name))
    if target is None:
        raise AttributeError(f"module 'ui' has no attribute {name!r}")
    mod_name, attr_name = target
    try:
        module = import_module(mod_name, __name__)
        value = getattr(module, attr_name)
    except Exception:
        value = None
    globals()[name] = value
    return value
