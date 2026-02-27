from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from ui.app_chart_history_load_ops import _load_chart_history_bars


class _DummyFetcher:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def get_history(self, symbol: str, **kwargs) -> pd.DataFrame:
        _ = (symbol, kwargs)
        return self._frame


class _DummyPredictor:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.fetcher = _DummyFetcher(frame)


class _DummyApp:
    def __init__(self, frame: pd.DataFrame, selected_date: date) -> None:
        self.predictor = _DummyPredictor(frame)
        self._selected_chart_date = selected_date.isoformat()
        self._bars_by_symbol: dict[str, list[dict[str, object]]] = {}

    def _normalize_interval_token(self, interval: str, fallback: str = "1m") -> str:
        return str(interval or fallback or "1m").strip().lower() or str(fallback)

    def _recommended_lookback(self, interval: str) -> int:
        _ = interval
        return 32

    def _bars_needed_from_base_interval(
        self, target_interval: str, lookback: int, base_interval: str = "1m"
    ) -> int:
        _ = (target_interval, base_interval)
        return int(max(1, lookback))

    def _interval_seconds(self, interval: str) -> int:
        iv = self._normalize_interval_token(interval)
        if iv == "1m":
            return 60
        if iv == "5m":
            return 300
        return 60

    def _bar_bucket_epoch(self, ts_raw: object, interval: str) -> float:
        step = float(max(1, self._interval_seconds(interval)))
        epoch = float(self._ts_to_epoch(ts_raw))
        return float(int(epoch // step) * step)

    def _is_intraday_day_boundary(
        self, prev_epoch: float, next_epoch: float, interval: str
    ) -> bool:
        _ = interval
        prev_day = datetime.fromtimestamp(float(prev_epoch), tz=ZoneInfo("Asia/Shanghai")).date()
        next_day = datetime.fromtimestamp(float(next_epoch), tz=ZoneInfo("Asia/Shanghai")).date()
        return prev_day != next_day

    def _sanitize_ohlc(
        self,
        o: float,
        h: float,
        low: float,
        c: float,
        interval: str,
        ref_close: float | None = None,
    ) -> tuple[float, float, float, float] | None:
        _ = (interval, ref_close)
        if c <= 0:
            return None
        top = max(float(h), float(o), float(c))
        bot = min(float(low), float(o), float(c))
        return float(o), float(top), float(bot), float(c)

    def _epoch_to_iso(self, epoch: float) -> str:
        tz = ZoneInfo("Asia/Shanghai")
        return datetime.fromtimestamp(float(epoch), tz=tz).isoformat()

    def _ts_to_epoch(self, ts_raw: object) -> float:
        if isinstance(ts_raw, (int, float)):
            return float(ts_raw)
        if isinstance(ts_raw, datetime):
            return float(ts_raw.timestamp())
        text = str(ts_raw or "").strip()
        if not text:
            return float(datetime.now(tz=ZoneInfo("Asia/Shanghai")).timestamp())
        try:
            return float(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
        except ValueError:
            return float(datetime.now(tz=ZoneInfo("Asia/Shanghai")).timestamp())

    def _filter_bars_to_market_session(
        self, bars: list[dict[str, object]], interval: str
    ) -> list[dict[str, object]]:
        _ = interval
        return list(bars or [])

    def _merge_bars(
        self,
        base: list[dict[str, object]],
        extra: list[dict[str, object]],
        interval: str,
    ) -> list[dict[str, object]]:
        _ = (base, interval)
        return list(extra or [])

    def _resample_chart_bars(
        self,
        bars: list[dict[str, object]],
        source_interval: str,
        target_interval: str,
    ) -> list[dict[str, object]]:
        _ = (source_interval, target_interval)
        return list(bars or [])


def _build_intraday_frame(day_val: date, count: int = 4) -> pd.DataFrame:
    tz = ZoneInfo("Asia/Shanghai")
    rows: list[dict[str, object]] = []
    for i in range(count):
        ts = datetime(
            day_val.year,
            day_val.month,
            day_val.day,
            9,
            30 + i,
            tzinfo=tz,
        )
        price = 100.0 + float(i)
        rows.append(
            {
                "datetime": ts,
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price + 0.2,
                "volume": 1000 + i,
                "amount": (1000 + i) * (price + 0.2),
            }
        )
    return pd.DataFrame(rows)


def _row_day(row: dict[str, object]) -> date:
    text = str(row.get("timestamp", "") or "")
    dt_val = datetime.fromisoformat(text)
    return dt_val.astimezone(ZoneInfo("Asia/Shanghai")).date()


def test_load_chart_history_falls_back_to_latest_session_when_selected_date_empty() -> None:
    today = datetime.now(tz=ZoneInfo("Asia/Shanghai")).date()
    prev_day = today - timedelta(days=1)
    frame = _build_intraday_frame(prev_day, count=5)
    app = _DummyApp(frame=frame, selected_date=today)

    out = _load_chart_history_bars(app, "600519", "1m", 60)

    assert len(out) == 5
    assert {_row_day(r) for r in out} == {prev_day}


def test_load_chart_history_keeps_selected_date_when_data_exists() -> None:
    today = datetime.now(tz=ZoneInfo("Asia/Shanghai")).date()
    frame = _build_intraday_frame(today, count=3)
    app = _DummyApp(frame=frame, selected_date=today)

    out = _load_chart_history_bars(app, "600519", "1m", 60)

    assert len(out) == 3
    assert {_row_day(r) for r in out} == {today}
