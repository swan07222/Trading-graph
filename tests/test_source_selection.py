import threading
from types import SimpleNamespace

from data.fetcher import AkShareSource, DataFetcher, DataSourceStatus
from data.universe import _can_use_akshare


def test_universe_tries_akshare_on_direct_china_even_if_probe_is_down(monkeypatch) -> None:
    """AkShare should be tried when China direct mode is active."""
    env = SimpleNamespace(
        eastmoney_ok=False,
        is_china_direct=True,
    )
    monkeypatch.setattr("data.universe.get_network_env", lambda: env)
    assert _can_use_akshare() is True


def test_universe_skips_akshare_when_offshore(monkeypatch) -> None:
    """AkShare should be skipped when not on China direct network."""
    env = SimpleNamespace(
        eastmoney_ok=False,
        is_china_direct=False,
    )
    monkeypatch.setattr("data.universe.get_network_env", lambda: env)
    assert _can_use_akshare() is False


def test_akshare_source_suitable_on_china_direct(monkeypatch) -> None:
    """AkShare is suitable whenever is_china_direct=True regardless of eastmoney probe."""
    src = AkShareSource()

    # China direct + eastmoney probe down: still suitable (probe is unreliable)
    env_cn_probe_down = SimpleNamespace(eastmoney_ok=False, is_china_direct=True)
    monkeypatch.setattr("core.network.get_network_env", lambda: env_cn_probe_down)
    assert src.is_suitable_for_network() is True

    # Offshore + eastmoney probe also down: not suitable
    env_offshore_bad = SimpleNamespace(eastmoney_ok=False, is_china_direct=False)
    monkeypatch.setattr("core.network.get_network_env", lambda: env_offshore_bad)
    assert src.is_suitable_for_network() is False

    # Offshore + eastmoney probe up (e.g. with eastmoney access): suitable
    env_offshore_good = SimpleNamespace(eastmoney_ok=True, is_china_direct=False)
    monkeypatch.setattr("core.network.get_network_env", lambda: env_offshore_good)
    assert src.is_suitable_for_network() is True


def test_source_health_prefers_tencent_when_eastmoney_down() -> None:
    fetcher = DataFetcher.__new__(DataFetcher)

    class _DummySource:
        def __init__(self, name: str) -> None:
            self.name = name
            self.priority = 0
            self.status = DataSourceStatus(name=name)

        def is_suitable_for_network(self) -> bool:
            return True

    env = SimpleNamespace(
        is_china_direct=True,
        eastmoney_ok=False,
    )
    tencent = _DummySource("tencent")
    akshare = _DummySource("akshare")

    t_score = fetcher._source_health_score(tencent, env)
    a_score = fetcher._source_health_score(akshare, env)
    assert t_score > a_score


def test_get_active_sources_filters_network_unsuitable_source(monkeypatch) -> None:
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._last_network_mode = (True, True, False)
    fetcher._rate_lock = threading.Lock()
    fetcher._request_times = {}

    class _Source:
        def __init__(self, name: str, suitable: bool) -> None:
            self.name = name
            self.priority = 0
            self.status = DataSourceStatus(name=name)
            self._suitable = bool(suitable)

        def is_available(self) -> bool:
            return True

        def is_suitable_for_network(self):
            return self._suitable

    good = _Source("tencent", True)
    bad = _Source("sina", False)
    fetcher._all_sources = [good, bad]

    env = SimpleNamespace(
        is_china_direct=True,
        eastmoney_ok=True,
    )
    monkeypatch.setattr("core.network.get_network_env", lambda: env)

    out = fetcher._get_active_sources()
    names = [s.name for s in out]
    assert names == ["tencent"]
