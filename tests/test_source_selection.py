import threading
from types import SimpleNamespace

from data.fetcher import AkShareSource, DataFetcher, DataSourceStatus
from data.universe import _can_use_akshare


def test_universe_tries_akshare_on_direct_china_even_if_probe_is_down(monkeypatch):
    env = SimpleNamespace(
        eastmoney_ok=False,
        is_vpn_active=False,
        is_china_direct=True,
    )
    monkeypatch.setattr("data.universe.get_network_env", lambda: env)
    assert _can_use_akshare() is True


def test_universe_skips_akshare_when_probe_is_down_offshore(monkeypatch):
    env = SimpleNamespace(
        eastmoney_ok=False,
        is_vpn_active=False,
        is_china_direct=False,
    )
    monkeypatch.setattr("data.universe.get_network_env", lambda: env)
    assert _can_use_akshare() is False


def test_akshare_source_requires_eastmoney_probe(monkeypatch):
    src = AkShareSource()

    env_bad = SimpleNamespace(eastmoney_ok=False, is_china_direct=True)
    monkeypatch.setattr("core.network.get_network_env", lambda: env_bad)
    assert src.is_suitable_for_network() is False

    env_good = SimpleNamespace(eastmoney_ok=True, is_china_direct=True)
    monkeypatch.setattr("core.network.get_network_env", lambda: env_good)
    assert src.is_suitable_for_network() is True


def test_source_health_prefers_tencent_when_eastmoney_down():
    fetcher = DataFetcher.__new__(DataFetcher)

    class _DummySource:
        def __init__(self, name: str):
            self.name = name
            self.priority = 0
            self.status = DataSourceStatus(name=name)

        def is_suitable_for_network(self):
            return True

    env = SimpleNamespace(
        is_china_direct=True,
        eastmoney_ok=False,
        yahoo_ok=False,
    )
    tencent = _DummySource("tencent")
    akshare = _DummySource("akshare")

    t_score = fetcher._source_health_score(tencent, env)
    a_score = fetcher._source_health_score(akshare, env)
    assert t_score > a_score


def test_get_active_sources_filters_network_unsuitable_source(monkeypatch):
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._last_network_mode = (True, True, False)
    fetcher._rate_lock = threading.Lock()
    fetcher._request_times = {}

    class _Source:
        def __init__(self, name: str, suitable: bool):
            self.name = name
            self.priority = 0
            self.status = DataSourceStatus(name=name)
            self._suitable = bool(suitable)

        def is_available(self):
            return True

        def is_suitable_for_network(self):
            return self._suitable

    good = _Source("tencent", True)
    bad = _Source("sina", False)
    fetcher._all_sources = [good, bad]

    env = SimpleNamespace(
        is_china_direct=True,
        eastmoney_ok=True,
        yahoo_ok=False,
    )
    monkeypatch.setattr("core.network.get_network_env", lambda: env)

    out = fetcher._get_active_sources()
    names = [s.name for s in out]
    assert names == ["tencent"]
