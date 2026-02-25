from __future__ import annotations

from types import SimpleNamespace

from ui import app_universe_ops


class _ListStub:
    def __init__(self) -> None:
        self.items: list[object] = []
        self.signal_blocks: list[bool] = []

    def blockSignals(self, value: bool) -> None:  # noqa: N802
        self.signal_blocks.append(bool(value))

    def clear(self) -> None:
        self.items.clear()

    def addItem(self, item: object) -> None:  # noqa: N802
        self.items.append(item)


class _LabelStub:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def setText(self, text: str) -> None:  # noqa: N802
        self.texts.append(str(text))


def _make_catalog(size: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(int(size)):
        code = str(600000 + i)
        rows.append({"code": code, "name": f"Stock {i}", "is_new": False})
    return rows


def test_filter_universe_list_shows_all_when_query_empty() -> None:
    catalog = _make_catalog(1200)
    lst = _ListStub()
    status = _LabelStub()
    stub = SimpleNamespace(
        universe_list=lst,
        _universe_catalog=catalog,
        universe_status_label=status,
    )

    app_universe_ops._filter_universe_list(stub, "")

    assert len(lst.items) == 1200
    assert status.texts
    assert "showing 1,200/1,200" in status.texts[-1]


def test_filter_universe_list_applies_limit_for_query(monkeypatch) -> None:
    monkeypatch.setattr(app_universe_ops, "_UNIVERSE_RENDER_LIMIT", 5)
    catalog = _make_catalog(25)
    lst = _ListStub()
    status = _LabelStub()
    stub = SimpleNamespace(
        universe_list=lst,
        _universe_catalog=catalog,
        universe_status_label=status,
    )

    app_universe_ops._filter_universe_list(stub, "60")

    assert len(lst.items) == 5
    assert status.texts
    assert "showing 5/25" in status.texts[-1]
