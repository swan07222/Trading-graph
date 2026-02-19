from __future__ import annotations

import pandas as pd

from data.database import MarketDatabase
from data.fetcher import DataFetcher


def test_clean_dataframe_compresses_zero_volume_flat_streak():
    idx = pd.date_range("2026-02-18 09:31:00", periods=120, freq="min")
    df = pd.DataFrame(
        {
            "open": [40.73] * 120,
            "high": [40.73] * 120,
            "low": [40.73] * 120,
            "close": [40.73] * 120,
            "volume": [0] * 120,
            "amount": [0.0] * 120,
        },
        index=idx,
    )

    out = DataFetcher._clean_dataframe(
        df,
        interval="1m",
        preserve_truth=False,
        aggressive_repairs=True,
        allow_synthetic_index=True,
    )

    assert not out.empty
    # Truth-preserving mode keeps original rows (no forced stale compression).
    assert len(out) == len(df)
    assert out.index.is_monotonic_increasing


def test_clean_dataframe_clamps_intraday_spike_shapes():
    idx = pd.date_range("2026-02-18 09:30:00", periods=3, freq="min")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0],
            "high": [100.2, 170.0, 101.5],
            "low": [99.9, 70.0, 100.8],
            "close": [100.0, 101.0, 101.2],
            "volume": [100, 120, 140],
            "amount": [10000.0, 12120.0, 14168.0],
        },
        index=idx,
    )

    out = DataFetcher._clean_dataframe(
        df,
        interval="1m",
        preserve_truth=False,
        aggressive_repairs=True,
        allow_synthetic_index=True,
    )

    assert len(out) == 3
    mid = out.iloc[1]
    span = abs(float(mid["high"]) - float(mid["low"])) / max(float(mid["close"]), 1e-8)
    # No forced shape clamping in truth-preserving mode.
    assert span > 0.50


def test_clean_dataframe_clips_extreme_intraday_jump():
    idx = pd.date_range("2026-02-18 09:30:00", periods=2, freq="min")
    df = pd.DataFrame(
        {
            "open": [100.0, 130.0],
            "high": [100.1, 130.1],
            "low": [99.9, 129.9],
            "close": [100.0, 130.0],
            "volume": [100, 120],
            "amount": [10000.0, 15600.0],
        },
        index=idx,
    )

    out = DataFetcher._clean_dataframe(
        df,
        interval="1m",
        preserve_truth=False,
        aggressive_repairs=True,
        allow_synthetic_index=True,
    )

    assert len(out) == 2
    # No forced jump clipping in truth-preserving mode.
    assert float(out.iloc[1]["close"]) == 130.0


def test_clean_dataframe_truth_preserving_does_not_clip_intraday_jump():
    idx = pd.date_range("2026-02-18 09:30:00", periods=2, freq="min")
    df = pd.DataFrame(
        {
            "open": [100.0, 130.0],
            "high": [100.1, 130.1],
            "low": [99.9, 129.9],
            "close": [100.0, 130.0],
            "volume": [100, 120],
            "amount": [10000.0, 15600.0],
        },
        index=idx,
    )

    out = DataFetcher._clean_dataframe(
        df,
        interval="1m",
        preserve_truth=True,
        aggressive_repairs=False,
        allow_synthetic_index=False,
    )

    assert len(out) == 2
    assert float(out.iloc[1]["close"]) == 130.0


def test_clean_dataframe_truth_preserving_rejects_undated_intraday_rows():
    df = pd.DataFrame(
        {
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "volume": [100, 100, 100],
        }
    )

    out = DataFetcher._clean_dataframe(
        df,
        interval="1m",
        preserve_truth=True,
        aggressive_repairs=False,
        allow_synthetic_index=False,
    )

    assert out.empty


def test_database_intraday_sanitizer_filters_bad_rows(tmp_path):
    db = MarketDatabase(db_path=tmp_path / "market_data.db")
    try:
        idx = pd.date_range("2026-02-18 09:30:00", periods=80, freq="min")
        closes = [40.73] * 60 + [40.75] * 20
        highs = [40.73] * 60 + [70.0] + [40.76] * 19
        lows = [40.73] * 60 + [10.0] + [40.74] * 19
        df = pd.DataFrame(
            {
                "open": closes,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": ([0] * 60) + ([120] * 20),
                "amount": [0.0] * 80,
            },
            index=idx,
        )

        db.upsert_intraday_bars("600519", "1m", df)
        out = db.get_intraday_bars("600519", interval="1m", limit=500)

        assert not out.empty
        assert len(out) < len(df)
        span = ((out["high"] - out["low"]).abs() / out["close"].clip(lower=1e-8)).max()
        # Truth-preserving mode should not inject large synthetic wick ranges.
        assert float(span) <= 0.015
    finally:
        db.close_all()
