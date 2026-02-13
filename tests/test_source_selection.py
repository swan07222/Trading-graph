from types import SimpleNamespace

from data.fetcher import AkShareSource, DataFetcher, DataSourceStatus
from data.universe import _can_use_akshare


def test_universe_skips_akshare_when_eastmoney_unreachable(monkeypatch):
    env = SimpleNamespace(
        eastmoney_ok=False,
        is_vpn_active=False,
        is_china_direct=True,
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
