from __future__ import annotations

import pandas as pd

from data.fetcher_frame_ops import merge_parts


def _clean(df: pd.DataFrame, _interval: str | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index()
    return out


def test_merge_parts_prefers_left_frame_on_duplicate_timestamp() -> None:
    idx = pd.to_datetime(["2026-02-24 09:31:00", "2026-02-24 09:32:00"])
    online = pd.DataFrame(
        {
            "open": [10.0, 10.2],
            "high": [10.3, 10.4],
            "low": [9.9, 10.1],
            "close": [10.2, 10.3],
            "volume": [120.0, 140.0],
        },
        index=idx,
    )
    db = pd.DataFrame(
        {
            "open": [9.8, 10.1],
            "high": [10.0, 10.2],
            "low": [9.7, 10.0],
            "close": [9.9, 10.15],
            "volume": [90.0, 100.0],
        },
        index=idx,
    )

    merged = merge_parts(online, db, interval="1m", clean_dataframe=_clean)

    assert len(merged) == 2
    assert float(merged.iloc[0]["close"]) == 10.2
    assert float(merged.iloc[1]["close"]) == 10.3
    assert float(merged.iloc[0]["volume"]) == 120.0
    assert float(merged.iloc[1]["volume"]) == 140.0


def test_merge_parts_respects_call_order_for_overlap() -> None:
    idx = pd.to_datetime(["2026-02-24 09:31:00"])
    left = pd.DataFrame({"close": [20.0], "volume": [500.0]}, index=idx)
    right = pd.DataFrame({"close": [19.5], "volume": [300.0]}, index=idx)

    merged_lr = merge_parts(left, right, interval="1m", clean_dataframe=_clean)
    merged_rl = merge_parts(right, left, interval="1m", clean_dataframe=_clean)

    assert float(merged_lr.iloc[0]["close"]) == 20.0
    assert float(merged_rl.iloc[0]["close"]) == 19.5
