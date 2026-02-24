from __future__ import annotations

import pandas as pd

from data.fetcher_history_flow_ops import _synthesize_intraday_from_daily


def _daily_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [24000.0, 26000.0],
            "amount": [2_420_000.0, 2_652_000.0],
        },
        index=index,
    )


def test_synthesize_intraday_from_daily_generates_requested_count() -> None:
    daily = _daily_frame(pd.to_datetime(["2026-02-20", "2026-02-21"]))

    out = _synthesize_intraday_from_daily(
        daily_df=daily,
        interval="5m",
        count=48,
    )

    assert isinstance(out, pd.DataFrame)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert len(out) == 48
    assert (out["open"] > 0).all()
    assert (out["close"] > 0).all()
    assert (out["high"] >= out["low"]).all()


def test_synthesize_intraday_from_daily_preserves_timezone() -> None:
    daily = _daily_frame(
        pd.DatetimeIndex(
            ["2026-02-20", "2026-02-21"],
            tz="Asia/Shanghai",
        )
    )

    out = _synthesize_intraday_from_daily(
        daily_df=daily,
        interval="15m",
        count=16,
    )

    assert len(out) == 16
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is not None
