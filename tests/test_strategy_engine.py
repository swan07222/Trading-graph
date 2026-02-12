from pathlib import Path

import pandas as pd

from analysis.strategy_engine import StrategyScriptEngine


def _make_df(rows: int = 80) -> pd.DataFrame:
    base = [100.0 + i * 0.1 for i in range(rows)]
    return pd.DataFrame(
        {
            "open": base,
            "high": [v + 1.0 for v in base],
            "low": [v - 1.0 for v in base],
            "close": [v + 0.2 for v in base],
            "volume": [100000 + i for i in range(rows)],
        }
    )


def test_strategy_engine_discovers_and_evaluates(tmp_path: Path):
    strategy = tmp_path / "test_rule.py"
    strategy.write_text(
        "\n".join(
            [
                "def generate_signal(df, indicators, context):",
                "    if indicators.get('rsi_14', 50) < 30:",
                "        return {'action': 'buy', 'score': 0.8, 'reason': 'RSI oversold'}",
                "    return {'action': 'hold', 'score': 0.0}",
            ]
        ),
        encoding="utf-8",
    )

    engine = StrategyScriptEngine(strategies_dir=tmp_path)
    bias, reasons = engine.evaluate(
        df=_make_df(),
        indicators={"rsi_14": 25.0},
        symbol="600519",
    )
    assert bias > 0
    assert any("RSI oversold" in r for r in reasons)
