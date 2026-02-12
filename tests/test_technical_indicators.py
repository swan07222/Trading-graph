import pandas as pd

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
    assert {"ema_9", "ema_21", "ema_55", "atr_14", "williams_r", "roc_10", "obv", "vwap"} <= keys
    assert "ema_21" in ta.list_supported_indicators()
