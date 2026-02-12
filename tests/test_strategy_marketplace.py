import json
from pathlib import Path

from analysis.strategy_marketplace import StrategyMarketplace


def test_marketplace_enable_disable(tmp_path: Path):
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
