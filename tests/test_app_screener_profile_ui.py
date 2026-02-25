from __future__ import annotations

from types import SimpleNamespace

from ui import app_analysis_ops


class _ComboStub:
    def __init__(self) -> None:
        self.items: list[str] = []
        self.current = ""
        self.tooltip = ""
        self.blocked: list[bool] = []

    def blockSignals(self, value: bool) -> None:  # noqa: N802
        self.blocked.append(bool(value))

    def clear(self) -> None:
        self.items.clear()
        self.current = ""

    def addItems(self, rows: list[str]) -> None:
        self.items.extend([str(x) for x in rows])

    def setCurrentText(self, text: str) -> None:  # noqa: N802
        self.current = str(text)

    def setToolTip(self, text: str) -> None:  # noqa: N802
        self.tooltip = str(text)


def test_init_screener_profile_ui_populates_combo(monkeypatch) -> None:
    import analysis.screener as screener_mod

    monkeypatch.setattr(
        screener_mod,
        "list_screener_profiles",
        lambda: ["quality", "balanced", "momentum"],
    )
    monkeypatch.setattr(
        screener_mod,
        "get_active_screener_profile_name",
        lambda preferred=None: "quality",
    )

    combo = _ComboStub()
    stub = SimpleNamespace(screener_profile_combo=combo)
    app_analysis_ops._init_screener_profile_ui(stub)

    assert sorted(combo.items) == ["balanced", "momentum", "quality"]
    assert combo.current == "quality"
    assert "Scan profile" in combo.tooltip
    assert bool(getattr(stub, "_syncing_screener_profile_ui", True)) is False
    assert str(getattr(stub, "_active_screener_profile", "")) == "quality"


def test_on_screener_profile_changed_persists(monkeypatch) -> None:
    called: dict[str, object] = {}
    logs: list[tuple[str, str]] = []

    import analysis.screener as screener_mod

    def _set_active(name: str) -> bool:
        called["set_active"] = name
        return True

    def _build(name: str | None = None, *, force_reload: bool = False):
        called["build"] = (name, bool(force_reload))
        return object()

    monkeypatch.setattr(screener_mod, "set_active_screener_profile", _set_active)
    monkeypatch.setattr(screener_mod, "build_default_screener", _build)

    stub = SimpleNamespace(
        _syncing_screener_profile_ui=False,
        log=lambda msg, level="info": logs.append((str(msg), str(level))),
    )
    app_analysis_ops._on_screener_profile_changed(stub, "momentum")

    assert called["set_active"] == "momentum"
    assert called["build"] == ("momentum", True)
    assert str(getattr(stub, "_active_screener_profile", "")) == "momentum"
    assert any("momentum" in msg for msg, _ in logs)


def test_show_screener_profile_dialog_applies_selected_profile(monkeypatch) -> None:
    events: list[str] = []

    class _Dialog:
        def __init__(self, _parent=None) -> None:
            self.selected_profile_name = "value"

        def exec(self) -> int:
            return 1

    monkeypatch.setattr(app_analysis_ops, "_lazy_get", lambda module, name: _Dialog)

    combo = _ComboStub()
    stub = SimpleNamespace(
        screener_profile_combo=combo,
        _init_screener_profile_ui=lambda: events.append("init"),
        _on_screener_profile_changed=lambda name: events.append(f"set:{name}"),
    )
    app_analysis_ops._show_screener_profile_dialog(stub)

    assert "init" in events
    assert "set:value" in events
    assert combo.current == "value"
