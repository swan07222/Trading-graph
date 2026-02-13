from pathlib import Path

from data.cache import DiskCache


def test_disk_cache_clear_blocked_without_manual_override(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TRADING_MANUAL_CACHE_DELETE", raising=False)
    dc = DiskCache(tmp_path / "l2", compress=False)
    dc.set("k1", {"x": 1})
    files_before = list((tmp_path / "l2").glob("*.pkl"))
    assert files_before
    dc.clear()
    files_after = list((tmp_path / "l2").glob("*.pkl"))
    assert len(files_after) == len(files_before)


def test_disk_cache_clear_allowed_with_manual_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TRADING_MANUAL_CACHE_DELETE", "1")
    dc = DiskCache(tmp_path / "l2", compress=False)
    dc.set("k1", {"x": 1})
    assert list((tmp_path / "l2").glob("*.pkl"))
    dc.clear()
    assert not list((tmp_path / "l2").glob("*.pkl"))
