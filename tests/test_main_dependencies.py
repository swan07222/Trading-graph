import types

import main


def test_check_dependencies_gui_mode_skips_ml_stack(monkeypatch):
    seen = []

    def fake_find_spec(name):
        seen.append(name)
        if name in {"psutil", "PyQt6"}:
            return types.SimpleNamespace()
        return None

    monkeypatch.setattr(main, "find_spec", fake_find_spec)

    ok = main.check_dependencies(require_gui=True, require_ml=False)
    assert ok is True
    assert "torch" not in seen
    assert "numpy" not in seen
    assert "pandas" not in seen
    assert "sklearn" not in seen


def test_check_dependencies_ml_mode_requires_ml_stack(monkeypatch, capsys):
    def fake_find_spec(name):
        if name in {"psutil", "numpy", "pandas", "sklearn"}:
            return types.SimpleNamespace()
        return None

    monkeypatch.setattr(main, "find_spec", fake_find_spec)

    ok = main.check_dependencies(require_gui=False, require_ml=True)
    assert ok is False
    out = capsys.readouterr().out
    assert "pytorch" in out
