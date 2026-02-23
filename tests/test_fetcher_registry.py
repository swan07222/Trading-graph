from __future__ import annotations

import threading

import data.fetcher as fetcher_mod
from data.fetcher_registry import FetcherRegistry


def _install_factory(monkeypatch) -> None:
    counter = {"n": 0}

    def _factory():
        counter["n"] += 1
        return {"id": counter["n"]}

    monkeypatch.setattr(fetcher_mod, "create_fetcher", _factory)


def test_get_fetcher_defaults_to_thread_scope(monkeypatch) -> None:
    monkeypatch.delenv("TRADING_DISABLE_SINGLETONS", raising=False)
    monkeypatch.delenv("TRADING_FETCHER_SCOPE", raising=False)
    fetcher_mod.reset_fetcher()
    _install_factory(monkeypatch)

    main_inst = fetcher_mod.get_fetcher()
    result: dict[str, object] = {}

    def _worker() -> None:
        result["worker"] = fetcher_mod.get_fetcher()

    t = threading.Thread(target=_worker)
    t.start()
    t.join(timeout=5)

    assert "worker" in result
    assert result["worker"] is not main_inst


def test_get_fetcher_process_scope_shares_instance(monkeypatch) -> None:
    monkeypatch.delenv("TRADING_DISABLE_SINGLETONS", raising=False)
    monkeypatch.setenv("TRADING_FETCHER_SCOPE", "process")
    fetcher_mod.reset_fetcher()
    _install_factory(monkeypatch)

    main_inst = fetcher_mod.get_fetcher()
    result: dict[str, object] = {}

    def _worker() -> None:
        result["worker"] = fetcher_mod.get_fetcher()

    t = threading.Thread(target=_worker)
    t.start()
    t.join(timeout=5)

    assert "worker" in result
    assert result["worker"] is main_inst


def test_use_fetcher_registry_isolates_context(monkeypatch) -> None:
    monkeypatch.delenv("TRADING_DISABLE_SINGLETONS", raising=False)
    monkeypatch.setenv("TRADING_FETCHER_SCOPE", "process")
    fetcher_mod.reset_fetcher()
    _install_factory(monkeypatch)

    base_inst = fetcher_mod.get_fetcher()
    custom_registry: FetcherRegistry[object] = FetcherRegistry()
    with fetcher_mod.use_fetcher_registry(custom_registry):
        scoped_a = fetcher_mod.get_fetcher()
        scoped_b = fetcher_mod.get_fetcher()

    after_inst = fetcher_mod.get_fetcher()

    assert scoped_a is scoped_b
    assert scoped_a is not base_inst
    assert after_inst is base_inst
