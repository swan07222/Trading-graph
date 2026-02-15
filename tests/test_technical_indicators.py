import pandas as pd
import pytest

from analysis.technical import TechnicalAnalyzer


def _df(rows: int = 220) -> pd.DataFrame:
    close = [100.0 + i * 0.2 for i in range(rows)]
    return pd.DataFrame(
        {
            "open": close,
            "high": [v + 1.0 for v in close],
            "low": [v - 1.0 for v in close],
            "close": close,
            "volume": [100000 + i * 10 for i in range(rows)],
        }
    )


def test_technical_analyzer_has_extended_indicators():
    ta = TechnicalAnalyzer()
    summary = ta.analyze(_df())
    keys = set(summary.indicators.keys())
    assert {
        "ema_9", "ema_21", "ema_55", "ema_100", "ema_200",
        "atr_14", "williams_r", "roc_10", "obv", "vwap",
        "ppo", "ppo_signal", "ppo_hist", "trix",
        "uo", "tsi", "kama", "cmf", "force_index",
        "atr_pct", "volatility_20", "momentum_20",
        "donchian_upper", "donchian_middle", "donchian_lower",
        "keltner_upper", "keltner_middle", "keltner_lower",
        "ichimoku_conv", "ichimoku_base", "stoch_rsi",
    } <= keys
    assert "ema_21" in ta.list_supported_indicators()


def test_technical_analyzer_sanitizes_dirty_ohlcv():
    ta = TechnicalAnalyzer()
    df = _df()
    df["close"] = df["close"].astype(object)
    df.loc[5, "close"] = "bad"
    df.loc[10, "high"] = float("inf")
    df.loc[11, "low"] = float("-inf")
    df.loc[20, "volume"] = None
    df.loc[25, "open"] = None

    summary = ta.analyze(df)
    assert isinstance(summary.overall_score, (int, float))
    assert all(pd.notna(v) for v in summary.indicators.values())


def test_technical_analyzer_requires_ohlcv_columns():
    ta = TechnicalAnalyzer()
    bad = pd.DataFrame({"close": [100.0] * 120})
    with pytest.raises(ValueError, match="Missing required columns"):
        ta.analyze(bad)
