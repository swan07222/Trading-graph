from pathlib import Path

from analysis.replay import MarketReplay


def test_replay_csv_load_and_order(tmp_path):
    csv_path = Path(tmp_path) / "replay.csv"
    csv_path.write_text(
        "\n".join(
            [
                "symbol,ts,open,high,low,close,volume,amount",
                "600519,2026-01-01T09:31:00,100,101,99,100.5,1000,100500",
                "000001,2026-01-01T09:30:00,10,10.2,9.9,10.1,2000,20200",
                "600519,2026-01-01T09:30:00,99,100,98.8,99.6,1500,149400",
            ]
        ),
        encoding="utf-8",
    )

    replay = MarketReplay.from_file(csv_path)
    bars = list(replay.iter_bars())

    assert len(replay) == 3
    assert [b.symbol for b in bars] == ["000001", "600519", "600519"]
    assert bars[0].ts.isoformat() == "2026-01-01T09:30:00"
    assert bars[-1].ts.isoformat() == "2026-01-01T09:31:00"
