import json
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


def test_strategy_engine_respects_marketplace_disable(tmp_path: Path):
    strategy = tmp_path / "rule.py"
    strategy.write_text(
        "\n".join(
            [
                "def generate_signal(df, indicators, context):",
                "    return {'action': 'buy', 'score': 1.0, 'reason': 'always'}",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "marketplace.json").write_text(
        json.dumps(
            {
                "version": 1,
                "strategies": [
                    {
                        "id": "rule",
                        "name": "Rule",
                        "version": "1.0",
                        "file": "rule.py",
                        "enabled_by_default": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "enabled.json").write_text(
        json.dumps({"enabled": []}),
        encoding="utf-8",
    )

    engine = StrategyScriptEngine(strategies_dir=tmp_path)
    assert engine.list_strategy_files() == []


def test_strategy_engine_isolates_script_side_effects(tmp_path: Path):
    (tmp_path / "a_mutator.py").write_text(
        "\n".join(
            [
                "def generate_signal(df, indicators, context):",
                "    indicators['rsi_14'] = 99.0",
                "    context['symbol'] = 'mutated'",
                "    df.loc[df.index[-1], 'close'] = 999999.0",
                "    return {'action': 'hold', 'score': 0.0, 'reason': 'mutator ran'}",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "b_reader.py").write_text(
        "\n".join(
            [
                "def generate_signal(df, indicators, context):",
                "    ok = (",
                "        indicators.get('rsi_14') == 25.0",
                "        and context.get('symbol') == '600519'",
                "        and float(df['close'].iloc[-1]) < 1000.0",
                "    )",
                "    if ok:",
                "        return {'action': 'buy', 'score': 0.6, 'reason': 'isolated'}",
                "    return {'action': 'hold', 'score': 0.0, 'reason': 'leaked'}",
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
    assert any("isolated" in r.lower() for r in reasons)


def test_strategy_engine_supports_batch_signals_and_weights(tmp_path: Path):
    (tmp_path / "batch_rule.py").write_text(
        "\n".join(
            [
                "def generate_signals(df, indicators, context):",
                "    return [",
                "        {'action': 'buy', 'score': 0.6, 'weight': 2.0, 'reason': 'weighted buy'},",
                "        {'action': 'sell', 'score': 0.2, 'weight': 0.5, 'reason': 'small hedge'},",
                "    ]",
            ]
        ),
        encoding="utf-8",
    )

    engine = StrategyScriptEngine(strategies_dir=tmp_path)
    bias, reasons = engine.evaluate(
        df=_make_df(),
        indicators={"rsi_14": 50.0},
        symbol="600519",
    )
    assert bias > 0
    assert any("weighted buy" in r.lower() for r in reasons)
