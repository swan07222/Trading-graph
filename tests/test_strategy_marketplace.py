import hashlib
import json
from pathlib import Path

from analysis.strategy_marketplace import StrategyMarketplace


def test_marketplace_enable_disable(tmp_path: Path) -> None:
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    script = strategies_dir / "demo.py"
    script.write_text("def generate_signal(df, indicators, context): return {'action':'hold','score':0}\n")

    manifest = {
        "version": 1,
        "strategies": [
            {
                "id": "demo",
                "name": "Demo",
                "version": "1.0",
                "file": "demo.py",
                "enabled_by_default": True,
            }
        ],
    }
    (strategies_dir / "marketplace.json").write_text(json.dumps(manifest), encoding="utf-8")

    m = StrategyMarketplace(strategies_dir=strategies_dir)
    entries = m.list_entries()
    assert len(entries) == 1
    assert entries[0]["enabled"] is True

    m.save_enabled_ids([])
    assert m.get_enabled_ids() == []


def test_marketplace_save_enabled_filters_unknown_and_mismatch(tmp_path: Path) -> None:
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)

    good = strategies_dir / "good.py"
    bad = strategies_dir / "bad.py"
    good.write_text("def generate_signal(df, indicators, context): return {'action':'hold','score':0}\n")
    bad.write_text("def generate_signal(df, indicators, context): return {'action':'buy','score':1}\n")
    good_hash = hashlib.sha256(good.read_bytes()).hexdigest()

    manifest = {
        "version": 1,
        "strategies": [
            {"id": "good", "name": "Good", "version": "1.0", "file": "good.py", "sha256": good_hash},
            {"id": "bad", "name": "Bad", "version": "1.0", "file": "bad.py", "sha256": "deadbeef"},
        ],
    }
    (strategies_dir / "marketplace.json").write_text(json.dumps(manifest), encoding="utf-8")

    m = StrategyMarketplace(strategies_dir=strategies_dir)
    m.save_enabled_ids(["good", "bad", "unknown"])
    assert m.get_enabled_ids() == ["good"]


def test_marketplace_integrity_summary_counts(tmp_path: Path) -> None:
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)

    script = strategies_dir / "demo.py"
    script.write_text("def generate_signal(df, indicators, context): return {'action':'hold','score':0}\n")
    script_hash = hashlib.sha256(script.read_bytes()).hexdigest()

    manifest = {
        "version": 1,
        "strategies": [
            {"id": "ok", "name": "OK", "version": "1.0", "file": "demo.py", "sha256": script_hash},
            {"id": "missing", "name": "Missing", "version": "1.0", "file": "missing.py"},
        ],
    }
    (strategies_dir / "marketplace.json").write_text(json.dumps(manifest), encoding="utf-8")

    m = StrategyMarketplace(strategies_dir=strategies_dir)
    summary = m.get_integrity_summary()
    assert summary["total"] == 2
    assert summary["ok"] == 1
    assert summary["missing"] == 1


def test_marketplace_rejects_unsafe_paths_and_dedupes_enabled(tmp_path: Path) -> None:
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)

    safe = strategies_dir / "safe.py"
    safe.write_text("def generate_signal(df, indicators, context): return {'action':'hold','score':0}\n")

    outside = tmp_path / "outside.py"
    outside.write_text("def generate_signal(df, indicators, context): return {'action':'buy','score':1}\n")
    assert outside.exists()

    manifest = {
        "version": 1,
        "strategies": [
            {"id": "safe", "name": "Safe", "version": "1.0", "file": "safe.py"},
            {"id": "escape", "name": "Escape", "version": "1.0", "file": "../outside.py"},
        ],
    }
    (strategies_dir / "marketplace.json").write_text(json.dumps(manifest), encoding="utf-8")
    (strategies_dir / "enabled.json").write_text(
        json.dumps({"enabled": ["safe", "safe", "escape"]}),
        encoding="utf-8",
    )

    m = StrategyMarketplace(strategies_dir=strategies_dir)

    assert m.get_enabled_ids() == ["safe", "escape"]
    entries = {e["id"]: e for e in m.list_entries()}
    assert entries["escape"]["integrity"] == "error"
    assert entries["safe"]["integrity"] in {"ok", "unverified"}

    m.save_enabled_ids(["safe", "escape"])
    assert m.get_enabled_ids() == ["safe"]
    files = m.get_enabled_files()
    assert files == [safe.resolve()]
