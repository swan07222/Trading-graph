from __future__ import annotations

import pandas as pd

from data.fetcher_history_flow_ops import _synthesize_intraday_from_daily


def _daily_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create a daily OHLCV DataFrame.
    
    Args:
        index: DatetimeIndex with one or more dates
        
    Returns:
        DataFrame with OHLCV data matching the index length
    """
    n = len(index)
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [102.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [101.0 + i for i in range(n)],
            "volume": [24000.0 + i * 2000 for i in range(n)],
            "amount": [2_420_000.0 + i * 232000 for i in range(n)],
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


def test_synthesize_intraday_from_daily_1m_correct_times() -> None:
    """Verify 1m bars have correct China A-share trading times.
    
    Morning session: 9:30-11:30 (bars 0-119)
    Afternoon session: 13:00-15:00 (bars 120-239)
    """
    daily = _daily_frame(pd.to_datetime(["2026-02-20"]))

    out = _synthesize_intraday_from_daily(
        daily_df=daily,
        interval="1m",
        count=240,
    )

    assert len(out) == 240

    # Bar 0 should be 9:30 (first bar of morning session)
    assert out.index[0].hour == 9
    assert out.index[0].minute == 30

    # Bar 59 should be 10:29
    assert out.index[59].hour == 10
    assert out.index[59].minute == 29

    # Bar 60 should be 10:30
    assert out.index[60].hour == 10
    assert out.index[60].minute == 30

    # Bar 119 should be 11:29 (last bar of morning session)
    assert out.index[119].hour == 11
    assert out.index[119].minute == 29

    # Bar 120 should be 13:00 (first bar of afternoon session)
    assert out.index[120].hour == 13
    assert out.index[120].minute == 0

    # Bar 179 should be 13:59
    assert out.index[179].hour == 13
    assert out.index[179].minute == 59

    # Bar 180 should be 14:00
    assert out.index[180].hour == 14
    assert out.index[180].minute == 0

    # Bar 239 should be 14:59 (last bar)
    assert out.index[239].hour == 14
    assert out.index[239].minute == 59


def test_synthesize_intraday_from_daily_5m_correct_times() -> None:
    """Verify 5m bars have correct China A-share trading times.
    
    Morning session: 9:30-11:30 (bars 0-23, 24 bars)
    Afternoon session: 13:00-15:00 (bars 24-47, 24 bars)
    """
    daily = _daily_frame(pd.to_datetime(["2026-02-20"]))

    out = _synthesize_intraday_from_daily(
        daily_df=daily,
        interval="5m",
        count=48,
    )

    assert len(out) == 48

    # Bar 0 should be 9:30 (first bar of morning session)
    assert out.index[0].hour == 9
    assert out.index[0].minute == 30

    # Bar 23 should be 11:25 (last bar of morning session)
    assert out.index[23].hour == 11
    assert out.index[23].minute == 25

    # Bar 24 should be 13:00 (first bar of afternoon session)
    assert out.index[24].hour == 13
    assert out.index[24].minute == 0

    # Bar 47 should be 14:55 (last bar)
    assert out.index[47].hour == 14
    assert out.index[47].minute == 55


def test_synthesize_intraday_from_daily_ohlc_consistency() -> None:
    """Verify synthesized OHLC values are consistent."""
    daily = _daily_frame(pd.to_datetime(["2026-02-20"]))

    out = _synthesize_intraday_from_daily(
        daily_df=daily,
        interval="1m",
        count=240,
    )

    # All bars should have valid OHLC
    assert (out["open"] > 0).all()
    assert (out["close"] > 0).all()
    assert (out["high"] >= out["low"]).all()

    # High should be >= both open and close
    assert (out["high"] >= out["open"]).all()
    assert (out["high"] >= out["close"]).all()

    # Low should be <= both open and close
    assert (out["low"] <= out["open"]).all()
    assert (out["low"] <= out["close"]).all()

    # All bars should be within daily range
    daily_low = daily.iloc[0]["low"]
    daily_high = daily.iloc[0]["high"]
    assert (out["low"] >= daily_low).all()
    assert (out["high"] <= daily_high).all()
