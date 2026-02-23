from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType


def _load_theme_module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    target = root / "ui" / "modern_theme.py"
    spec = spec_from_file_location("modern_theme_under_test", target)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load ui/modern_theme.py for tests")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


theme = _load_theme_module()


def _clear_font_caches() -> None:
    theme._font_family_set.cache_clear()
    theme.get_primary_font_family.cache_clear()
    theme.get_display_font_family.cache_clear()
    theme.get_monospace_font_family.cache_clear()


def test_pick_font_family_prefers_first_available(monkeypatch) -> None:
    monkeypatch.setattr(theme, "_font_family_set", lambda: {"B", "C"})
    assert theme._pick_font_family(("A", "B", "C"), "Z") == "B"


def test_pick_font_family_falls_back_when_none_available(monkeypatch) -> None:
    monkeypatch.setattr(theme, "_font_family_set", lambda: set())
    assert theme._pick_font_family(("A", "B"), "Z") == "Z"


def test_public_font_resolvers_use_candidates(monkeypatch) -> None:
    _clear_font_caches()
    monkeypatch.setattr(theme, "_font_family_set", lambda: {"FontA", "MonoA"})
    monkeypatch.setattr(
        theme.ModernFonts,
        "PRIMARY_CANDIDATES",
        ("MissingA", "FontA"),
        raising=False,
    )
    monkeypatch.setattr(
        theme.ModernFonts,
        "DISPLAY_CANDIDATES",
        ("MissingB", "FontA"),
        raising=False,
    )
    monkeypatch.setattr(
        theme.ModernFonts,
        "MONOSPACE_CANDIDATES",
        ("MissingC", "MonoA"),
        raising=False,
    )
    monkeypatch.setattr(
        theme.ModernFonts,
        "FAMILY_PRIMARY",
        "FallbackPrimary",
        raising=False,
    )
    monkeypatch.setattr(
        theme.ModernFonts,
        "FAMILY_MONOSPACE",
        "FallbackMono",
        raising=False,
    )

    assert theme.get_primary_font_family() == "FontA"
    assert theme.get_display_font_family() == "FontA"
    assert theme.get_monospace_font_family() == "MonoA"
