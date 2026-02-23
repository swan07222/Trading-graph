from __future__ import annotations

import csv
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ReplayBar:
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    amount: float = 0.0


class MarketReplay:
    """Deterministic market replay utility for regression testing.

    Supports:
    - CSV files with columns: symbol, ts, open, high, low, close[, volume, amount]
    - JSONL files where each line is a dict with equivalent fields
    """

    def __init__(self, bars: list[ReplayBar]):
        self._bars = sorted(bars, key=lambda b: (b.ts, b.symbol))
        self._idx = 0

    @classmethod
    def from_file(cls, path: Path) -> MarketReplay:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Replay file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return cls(cls._load_csv(path))
        if suffix in (".jsonl", ".ndjson"):
            return cls(cls._load_jsonl(path))
        raise ValueError(f"Unsupported replay format: {path.suffix}")

    @staticmethod
    def _parse_bar(obj: dict) -> ReplayBar | None:
        try:
            ts_raw = obj.get("ts") or obj.get("timestamp")
            if ts_raw is None:
                return None
            ts = (
                ts_raw
                if isinstance(ts_raw, datetime)
                else datetime.fromisoformat(str(ts_raw))
            )
            symbol = str(obj.get("symbol") or obj.get("code") or "").strip()
            if not symbol:
                return None
            return ReplayBar(
                symbol=symbol.zfill(6) if symbol.isdigit() else symbol,
                ts=ts,
                open=float(obj.get("open", 0.0)),
                high=float(obj.get("high", 0.0)),
                low=float(obj.get("low", 0.0)),
                close=float(obj.get("close", 0.0)),
                volume=float(obj.get("volume", 0.0) or 0.0),
                amount=float(obj.get("amount", 0.0) or 0.0),
            )
        except Exception:
            return None

    @classmethod
    def _load_csv(cls, path: Path) -> list[ReplayBar]:
        out: list[ReplayBar] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bar = cls._parse_bar(row)
                if bar:
                    out.append(bar)
        log.info(f"Replay loaded from CSV: {path.name} ({len(out)} bars)")
        return out

    @classmethod
    def _load_jsonl(cls, path: Path) -> list[ReplayBar]:
        out: list[ReplayBar] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                bar = cls._parse_bar(obj)
                if bar:
                    out.append(bar)
        log.info(f"Replay loaded from JSONL: {path.name} ({len(out)} bars)")
        return out

    def __len__(self) -> int:
        return len(self._bars)

    def reset(self) -> None:
        self._idx = 0

    def next(self) -> ReplayBar | None:
        if self._idx >= len(self._bars):
            return None
        bar = self._bars[self._idx]
        self._idx += 1
        return bar

    def iter_bars(self) -> Iterator[ReplayBar]:
        self.reset()
        while True:
            b = self.next()
            if b is None:
                break
            yield b

    def play(self, speed: float = 1.0) -> Iterator[ReplayBar]:
        """Replay bars in chronological order.
        speed=1.0 keeps real time deltas, >1 accelerates.
        """
        self.reset()
        if not self._bars:
            return

        prev_ts: datetime | None = None
        for bar in self._bars:
            if prev_ts is not None:
                dt = (bar.ts - prev_ts).total_seconds()
                if dt > 0:
                    time.sleep(max(0.0, dt / max(0.001, speed)))
            prev_ts = bar.ts
            yield bar
