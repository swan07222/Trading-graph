from datetime import datetime

from core.network import NetworkDetector, NetworkEnv


def test_network_detector_peek_respects_ttl() -> None:
    det = NetworkDetector()

    old_env = det._env
    old_time = det._env_time
    old_ttl = det._ttl
    try:
        det._ttl = 10.0
        det._env = NetworkEnv(detected_at=datetime.now())
        det._env_time = 100.0
        assert det.peek_env() is None
    finally:
        det._env = old_env
        det._env_time = old_time
        det._ttl = old_ttl


def test_network_detector_get_env_uses_fresh_cache(monkeypatch) -> None:
    det = NetworkDetector()

    old_env = det._env
    old_time = det._env_time
    old_ttl = det._ttl
    try:
        cached = NetworkEnv(
            is_china_direct=True,
            detected_at=datetime.now(),
        )
        det._ttl = 3600.0
        det._env = cached
        det._env_time = 100.0
        monkeypatch.setattr("core.network.time.time", lambda: 120.0)

        calls = {"detect": 0}

        def _fake_detect():
            calls["detect"] += 1
            return NetworkEnv(detected_at=datetime.now())

        monkeypatch.setattr(det, "_detect", _fake_detect, raising=True)
        out = det.get_env(force_refresh=False)

        assert out is cached
        assert calls["detect"] == 0
    finally:
        det._env = old_env
        det._env_time = old_time
        det._ttl = old_ttl


def test_network_detector_force_refresh_bypasses_cache(monkeypatch) -> None:
    det = NetworkDetector()

    old_env = det._env
    old_time = det._env_time
    old_ttl = det._ttl
    try:
        det._ttl = 3600.0
        det._env = NetworkEnv(detected_at=datetime.now())
        det._env_time = 100.0
        monkeypatch.setattr("core.network.time.time", lambda: 120.0)

        calls = {"detect": 0}
        fresh = NetworkEnv(
            is_china_direct=True,
            detected_at=datetime.now(),
        )

        def _fake_detect():
            calls["detect"] += 1
            return fresh

        monkeypatch.setattr(det, "_detect", _fake_detect, raising=True)
        out = det.get_env(force_refresh=True)

        assert out is fresh
        assert calls["detect"] == 1
    finally:
        det._env = old_env
        det._env_time = old_time
        det._ttl = old_ttl


def test_network_detector_invalidate_clears_cache() -> None:
    det = NetworkDetector()

    old_env = det._env
    old_time = det._env_time
    try:
        det._env = NetworkEnv(detected_at=datetime.now())
        det._env_time = 123.0
        det.invalidate()
        assert det.peek_env() is None
    finally:
        det._env = old_env
        det._env_time = old_time


def test_network_detector_detect_does_not_depend_on_requests_session(monkeypatch) -> None:
    det = NetworkDetector()

    old_env = det._env
    old_time = det._env_time
    old_ttl = det._ttl
    try:
        # If _detect still instantiates requests.Session, this test should fail.
        monkeypatch.setattr(
            "core.network.requests.Session",
            lambda: (_ for _ in ()).throw(RuntimeError("session not allowed")),
        )

        class _Resp:
            def __init__(self, status_code: int) -> None:
                self.status_code = int(status_code)

        def _fake_get(url, timeout=0, headers=None):
            if "eastmoney" in str(url):
                return _Resp(200)
            if "tencent" in str(url):
                return _Resp(200)
            return _Resp(200)

        monkeypatch.setattr("core.network.requests.get", _fake_get)
        env = det.get_env(force_refresh=True)

        assert isinstance(env, NetworkEnv)
        assert env.eastmoney_ok is True
    finally:
        det._env = old_env
        det._env_time = old_time
        det._ttl = old_ttl
