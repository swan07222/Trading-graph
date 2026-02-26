# analysis/advanced_charts.py
"""Advanced Chart Types for Professional Technical Analysis.

This module provides alternative chart types beyond standard candlestick/OHLC:
- Heikin-Ashi: Smoothed candlesticks for trend identification
- Renko: Brick-based charts filtered by price movement
- Kagi: Direction-based charts ignoring time
- Point & Figure: X/O charts for pure price action
- Three Line Break: Reversal-based charts
- Volume Profile: Volume distribution at price levels
- Market Profile: Time-price-volume analysis
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


class ChartType(Enum):
    """Supported advanced chart types."""
    CANDLESTICK = "candlestick"
    HEIKIN_ASHI = "heikin_ashi"
    RENKO = "renko"
    KAGI = "kagi"
    POINT_FIGURE = "point_figure"
    LINE_BREAK_3 = "line_break_3"
    LINE_BREAK_2 = "line_break_2"
    VOLUME_PROFILE = "volume_profile"
    MARKET_PROFILE = "market_profile"


@dataclass
class ChartBar:
    """Universal chart bar structure."""
    index: int
    timestamp: Any = None
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_tuple(self) -> tuple:
        """Convert to tuple for rendering."""
        return (self.index, self.open, self.close, self.low, self.high)

    def to_ohlc_dict(self) -> dict:
        """Convert to OHLCV dictionary."""
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "timestamp": self.timestamp,
        }


@dataclass
class RenkoBrick:
    """Renko brick representation."""
    index: int
    timestamp: Any
    price_open: float
    price_close: float
    price_high: float
    price_low: float
    is_up: bool
    brick_count: int = 1


@dataclass
class KagiLine:
    """Kagi line segment."""
    index: int
    direction: int  # 1 = up, -1 = down
    start_price: float
    end_price: float
    start_time: Any
    end_time: Any
    thickness: str = "thin"  # "thin" or "thick"


@dataclass
class PointFigureColumn:
    """Point & Figure column of X's or O's."""
    column_index: int
    is_x_column: bool  # True = X column (rising), False = O column (falling)
    start_price: float
    end_price: float
    box_count: int
    timestamp_start: Any
    timestamp_end: Any


@dataclass
class VolumeProfileLevel:
    """Volume at price level."""
    price: float
    volume: float
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    trades: int = 0
    percent_of_total: float = 0.0
    is_poc: bool = False  # Point of Control
    is_vah: bool = False  # Value Area High
    is_val: bool = False  # Value Area Low


class HeikinAshiCalculator:
    """Heikin-Ashi candlestick calculator.

    Heikin-Ashi (Japanese for "Average Bar") smooths price action by using
    modified OHLC formulas based on prior candle values.

    Formulas:
    - HA Close = (Open + High + Low + Close) / 4
    - HA Open = (Prior HA Open + Prior HA Close) / 2
    - HA High = max(High, HA Open, HA Close)
    - HA Low = min(Low, HA Open, HA Close)
    """

    def calculate(self, df: pd.DataFrame) -> list[ChartBar]:
        """Calculate Heikin-Ashi candles from OHLCV data."""
        if len(df) < 1:
            return []

        result = []
        prev_ha_open = 0.0
        prev_ha_close = 0.0

        for idx, (_, row) in enumerate(df.iterrows()):
            o = float(row.get("open", 0.0))
            h = float(row.get("high", 0.0))
            l = float(row.get("low", 0.0))
            c = float(row.get("close", 0.0))
            v = float(row.get("volume", 0.0))
            ts = row.get("timestamp", row.name)

            # Calculate HA Close (average of current OHLC)
            ha_close = (o + h + l + c) / 4.0

            # Calculate HA Open
            if idx == 0:
                ha_open = (o + c) / 2.0
            else:
                ha_open = (prev_ha_open + prev_ha_close) / 2.0

            # Calculate HA High and Low
            ha_high = max(h, ha_open, ha_close)
            ha_low = min(l, ha_open, ha_close)

            # Determine candle color and wick analysis
            is_bullish = ha_close > ha_open
            has_lower_wick = ha_low < min(ha_open, ha_close)
            has_upper_wick = ha_high > max(ha_open, ha_close)

            bar = ChartBar(
                index=idx,
                timestamp=ts,
                open=ha_open,
                high=ha_high,
                low=ha_low,
                close=ha_close,
                volume=v,
                extra={
                    "original_open": o,
                    "original_high": h,
                    "original_low": l,
                    "original_close": c,
                    "is_bullish": is_bullish,
                    "has_lower_wick": has_lower_wick,
                    "has_upper_wick": has_upper_wick,
                    "chart_type": "heikin_ashi",
                }
            )
            result.append(bar)

            prev_ha_open = ha_open
            prev_ha_close = ha_close

        return result


class RenkoCalculator:
    """Renko brick chart calculator.

    Renko charts filter out minor price movements and focus on significant trends.
    Each brick represents a fixed price movement (box size).

    Parameters:
    - box_size: Price movement required for new brick (fixed or ATR-based)
    - use_atr: Use Average True Range for dynamic box size
    - atr_period: Period for ATR calculation (default 14)
    """

    def __init__(
        self,
        box_size: float | None = None,
        use_atr: bool = True,
        atr_period: int = 14,
    ) -> None:
        self.box_size = box_size
        self.use_atr = use_atr
        self.atr_period = atr_period

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate ATR for dynamic box size."""
        if len(df) < self.atr_period + 1:
            # Fallback to fixed percentage of price range
            price_range = df["high"].max() - df["low"].min()
            return price_range / 100.0

        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean().iloc[-1]

        return float(atr) if pd.notna(atr) else 0.0

    def calculate(self, df: pd.DataFrame) -> list[RenkoBrick]:
        """Calculate Renko bricks from OHLCV data."""
        if len(df) < 2:
            return []

        # Determine box size
        if self.box_size is not None:
            box_size = self.box_size
        elif self.use_atr:
            box_size = self._calculate_atr(df)
            if box_size <= 0:
                box_size = (df["high"].max() - df["low"].min()) / 100.0
        else:
            # Default: 1% of current price
            box_size = df["close"].iloc[-1] * 0.01

        if box_size <= 0:
            log.warning("Invalid Renko box size, using default")
            box_size = 0.01

        result = []
        bricks = []

        # Initialize with first bar
        first_close = df["close"].iloc[0]
        first_ts = df.index[0] if hasattr(df.index[0], "to_pydatetime") else df.index[0]

        # Round to nearest box
        base_price = round(first_close / box_size) * box_size
        current_price = base_price
        direction = 1  # 1 = up, -1 = down
        brick_high = current_price
        brick_low = current_price

        for idx, (_, row) in enumerate(df.iterrows()):
            high = float(row["high"])
            low = float(row["low"])
            ts = row.get("timestamp", row.name)

            if direction == 1:
                # Rising brick
                if high >= current_price + box_size:
                    # Create new up brick
                    new_price = current_price + box_size
                    brick = RenkoBrick(
                        index=len(bricks),
                        timestamp=ts,
                        price_open=current_price,
                        price_close=new_price,
                        price_high=high,
                        price_low=current_price,
                        is_up=True,
                    )
                    bricks.append(brick)
                    current_price = new_price
                    brick_high = high
                elif low <= current_price - 2 * box_size:
                    # Reversal: create down bricks
                    direction = -1
                    while current_price - box_size >= low:
                        new_price = current_price - box_size
                        brick = RenkoBrick(
                            index=len(bricks),
                            timestamp=ts,
                            price_open=current_price,
                            price_close=new_price,
                            price_high=current_price,
                            price_low=low,
                            is_up=False,
                        )
                        bricks.append(brick)
                        current_price = new_price
                    brick_low = low
            else:
                # Falling brick
                if low <= current_price - box_size:
                    # Create new down brick
                    new_price = current_price - box_size
                    brick = RenkoBrick(
                        index=len(bricks),
                        timestamp=ts,
                        price_open=current_price,
                        price_close=new_price,
                        price_high=current_price,
                        price_low=low,
                        is_up=False,
                    )
                    bricks.append(brick)
                    current_price = new_price
                    brick_low = low
                elif high >= current_price + 2 * box_size:
                    # Reversal: create up bricks
                    direction = 1
                    while current_price + box_size <= high:
                        new_price = current_price + box_size
                        brick = RenkoBrick(
                            index=len(bricks),
                            timestamp=ts,
                            price_open=current_price,
                            price_close=new_price,
                            price_high=high,
                            price_low=current_price,
                            is_up=True,
                        )
                        bricks.append(brick)
                        current_price = new_price
                    brick_high = high

        # Convert to ChartBar format for rendering
        for idx, brick in enumerate(bricks):
            bar = ChartBar(
                index=idx,
                timestamp=brick.timestamp,
                open=brick.price_open,
                high=brick.price_high,
                low=brick.price_low,
                close=brick.price_close,
                volume=0.0,  # Renko doesn't track volume per brick
                extra={
                    "is_up": brick.is_up,
                    "chart_type": "renko",
                    "box_size": box_size,
                }
            )
            result.append(bar)

        return result


class KagiCalculator:
    """Kagi chart calculator.

    Kagi charts ignore time and focus on price reversals.
    A new line is drawn when price reverses by a specified amount.

    Parameters:
    - reversal_amount: Price reversal required to change direction (fixed or percentage)
    - use_percentage: Use percentage-based reversal (default True)
    """

    def __init__(
        self,
        reversal_amount: float = 0.02,
        use_percentage: bool = True,
    ) -> None:
        self.reversal_amount = reversal_amount
        self.use_percentage = use_percentage

    def calculate(self, df: pd.DataFrame) -> list[KagiLine]:
        """Calculate Kagi lines from OHLCV data."""
        if len(df) < 2:
            return []

        result = []
        lines = []

        # Initialize
        first_close = df["close"].iloc[0]
        first_ts = df.index[0] if hasattr(df.index[0], "to_pydatetime") else df.index[0]

        current_price = first_close
        direction = 1  # 1 = yang (up), -1 = yin (down)
        line_start = first_close
        line_start_time = first_ts
        line_high = first_close
        line_low = first_close

        # Calculate reversal threshold
        if self.use_percentage:
            reversal_threshold = abs(current_price * self.reversal_amount)
        else:
            reversal_threshold = self.reversal_amount

        for idx, (_, row) in enumerate(df.iterrows()):
            high = float(row["high"])
            low = float(row["low"])
            ts = row.get("timestamp", row.name)

            if direction == 1:
                # Yang line (rising)
                line_high = max(line_high, high)
                line_low = min(line_low, low)

                if low <= line_high - reversal_threshold:
                    # Complete yang line
                    line = KagiLine(
                        index=len(lines),
                        direction=1,
                        start_price=line_start,
                        end_price=line_high,
                        start_time=line_start_time,
                        end_time=ts,
                        thickness="thick" if line_high > line_start else "thin",
                    )
                    lines.append(line)

                    # Start yin line
                    direction = -1
                    line_start = line_high
                    line_start_time = ts
                    line_high = high
                    line_low = low
            else:
                # Yin line (falling)
                line_high = max(line_high, high)
                line_low = min(line_low, low)

                if high >= line_low + reversal_threshold:
                    # Complete yin line
                    line = KagiLine(
                        index=len(lines),
                        direction=-1,
                        start_price=line_start,
                        end_price=line_low,
                        start_time=line_start_time,
                        end_time=ts,
                        thickness="thick" if line_low < line_start else "thin",
                    )
                    lines.append(line)

                    # Start yang line
                    direction = 1
                    line_start = line_low
                    line_start_time = ts
                    line_high = high
                    line_low = low

        # Add final line
        if direction == 1:
            final_line = KagiLine(
                index=len(lines),
                direction=1,
                start_price=line_start,
                end_price=line_high,
                start_time=line_start_time,
                end_time=df.index[-1] if hasattr(df.index[-1], "to_pydatetime") else df.index[-1],
                thickness="thick" if line_high > line_start else "thin",
            )
        else:
            final_line = KagiLine(
                index=len(lines),
                direction=-1,
                start_price=line_start,
                end_price=line_low,
                start_time=line_start_time,
                end_time=df.index[-1] if hasattr(df.index[-1], "to_pydatetime") else df.index[-1],
                thickness="thick" if line_low < line_start else "thin",
            )
        lines.append(final_line)

        return lines


class PointFigureCalculator:
    """Point & Figure chart calculator.

    Point & Figure charts use X's and O's to represent price movements,
    filtering out time and minor price changes.

    Parameters:
    - box_size: Price per box (fixed or ATR-based)
    - reversal_boxes: Number of boxes for reversal (default 3)
    - use_atr: Use ATR for dynamic box size
    """

    def __init__(
        self,
        box_size: float | None = None,
        reversal_boxes: int = 3,
        use_atr: bool = False,
    ) -> None:
        self.box_size = box_size
        self.reversal_boxes = reversal_boxes
        self.use_atr = use_atr

    def calculate(self, df: pd.DataFrame) -> list[PointFigureColumn]:
        """Calculate Point & Figure columns from OHLCV data."""
        if len(df) < 2:
            return []

        # Determine box size
        if self.box_size is not None:
            box_size = self.box_size
        elif self.use_atr:
            # Calculate ATR-based box size
            high = df["high"]
            low = df["low"]
            close = df["close"]
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            box_size = float(atr) / 3.0 if pd.notna(atr) else 0.01
        else:
            box_size = df["close"].iloc[-1] * 0.01

        if box_size <= 0:
            box_size = 0.01

        result = []
        columns = []

        # Initialize
        first_price = df["close"].iloc[0]
        first_ts = df.index[0]
        current_price = round(first_price / box_size) * box_size
        column_type = "X"  # Start with rising column
        column_start = current_price
        column_high = current_price
        column_low = current_price
        column_start_time = first_ts

        for idx, (_, row) in enumerate(df.iterrows()):
            high = float(row["high"])
            low = float(row["low"])
            ts = row.get("timestamp", row.name)

            # Round to box
            high_box = round(high / box_size) * box_size
            low_box = round(low / box_size) * box_size

            if column_type == "X":
                # Rising column
                if high_box > column_high:
                    column_high = high_box
                elif column_high - low_box >= self.reversal_boxes * box_size:
                    # Complete X column and start O column
                    col = PointFigureColumn(
                        column_index=len(columns),
                        is_x_column=True,
                        start_price=column_start,
                        end_price=column_high,
                        box_count=int((column_high - column_start) / box_size) + 1,
                        timestamp_start=column_start_time,
                        timestamp_end=ts,
                    )
                    columns.append(col)

                    # Start O column
                    column_type = "O"
                    column_start = column_high
                    column_low = low_box
                    column_high = low_box
                    column_start_time = ts
            else:
                # Falling column
                if low_box < column_low:
                    column_low = low_box
                elif low_box - column_high >= self.reversal_boxes * box_size:
                    # Complete O column and start X column
                    col = PointFigureColumn(
                        column_index=len(columns),
                        is_x_column=False,
                        start_price=column_start,
                        end_price=column_low,
                        box_count=int((column_start - column_low) / box_size) + 1,
                        timestamp_start=column_start_time,
                        timestamp_end=ts,
                    )
                    columns.append(col)

                    # Start X column
                    column_type = "X"
                    column_start = column_low
                    column_high = high_box
                    column_low = high_box
                    column_start_time = ts

        # Add final column
        final_col = PointFigureColumn(
            column_index=len(columns),
            is_x_column=(column_type == "X"),
            start_price=column_start,
            end_price=column_high if column_type == "X" else column_low,
            box_count=int(abs(column_high - column_low) / box_size) + 1,
            timestamp_start=column_start_time,
            timestamp_end=df.index[-1],
        )
        columns.append(final_col)

        return columns


class LineBreakCalculator:
    """Three Line Break chart calculator.

    Line Break charts ignore time and draw new lines based on price breakouts.
    A reversal requires breaking the high/low of the previous N lines.

    Parameters:
    - break_count: Number of lines for reversal (2 or 3)
    """

    def __init__(self, break_count: int = 3) -> None:
        self.break_count = break_count

    def calculate(self, df: pd.DataFrame) -> list[ChartBar]:
        """Calculate Line Break bars from OHLCV data."""
        if len(df) < self.break_count + 1:
            return []

        result = []
        lines = []

        # Initialize with first bar
        first_close = df["close"].iloc[0]
        first_ts = df.index[0]

        current_direction = 1  # 1 = up, -1 = down
        lines.append({
            "high": float(df["high"].iloc[0]),
            "low": float(df["low"].iloc[0]),
            "close": first_close,
            "timestamp": first_ts,
            "direction": 1,
        })

        for idx in range(1, len(df)):
            row = df.iloc[idx]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            ts = row.get("timestamp", row.name)

            if current_direction == 1:
                # Rising trend
                if close > lines[-1]["high"]:
                    # Continue uptrend
                    lines.append({
                        "high": high,
                        "low": lines[-1]["high"],
                        "close": close,
                        "timestamp": ts,
                        "direction": 1,
                    })
                elif low < min(l["low"] for l in lines[-self.break_count:]):
                    # Reversal
                    lines.append({
                        "high": lines[-1]["close"],
                        "low": low,
                        "close": close,
                        "timestamp": ts,
                        "direction": -1,
                    })
                    current_direction = -1
            else:
                # Falling trend
                if close < lines[-1]["low"]:
                    # Continue downtrend
                    lines.append({
                        "high": lines[-1]["low"],
                        "low": low,
                        "close": close,
                        "timestamp": ts,
                        "direction": -1,
                    })
                elif high > max(l["high"] for l in lines[-self.break_count:]):
                    # Reversal
                    lines.append({
                        "high": high,
                        "low": lines[-1]["close"],
                        "close": close,
                        "timestamp": ts,
                        "direction": 1,
                    })
                    current_direction = 1

        # Convert to ChartBar format
        for idx, line in enumerate(lines):
            bar = ChartBar(
                index=idx,
                timestamp=line["timestamp"],
                open=line["close"],  # Simplified
                high=line["high"],
                low=line["low"],
                close=line["close"],
                volume=0.0,
                extra={
                    "direction": line["direction"],
                    "chart_type": f"line_break_{self.break_count}",
                }
            )
            result.append(bar)

        return result


class VolumeProfileCalculator:
    """Volume Profile calculator.

    Calculates volume distribution at price levels to identify:
    - Point of Control (POC): Price with highest volume
    - Value Area High (VAH): 70% value area upper bound
    - Value Area Low (VAL): 70% value area lower bound
    - Support/Resistance levels
    """

    def __init__(
        self,
        num_levels: int = 50,
        value_area_percent: float = 0.70,
    ) -> None:
        self.num_levels = num_levels
        self.value_area_percent = value_area_percent

    def calculate(self, df: pd.DataFrame) -> list[VolumeProfileLevel]:
        """Calculate volume profile from OHLCV data."""
        if len(df) < 2:
            return []

        # Get price range
        price_min = df["low"].min()
        price_max = df["high"].max()
        price_range = price_max - price_min

        if price_range <= 0:
            return []

        # Create price bins
        bin_size = price_range / self.num_levels
        levels = []

        total_volume = float(df["volume"].sum())

        for i in range(self.num_levels):
            level_price = price_min + (i + 0.5) * bin_size
            level_low = price_min + i * bin_size
            level_high = level_low + bin_size

            # Calculate volume at this price level
            # Using proportional allocation based on bar's price range
            level_volume = 0.0

            for _, row in df.iterrows():
                bar_low = float(row["low"])
                bar_high = float(row["high"])
                bar_volume = float(row["volume"])

                # Check if bar overlaps with this level
                overlap_low = max(bar_low, level_low)
                overlap_high = min(bar_high, level_high)

                if overlap_low < overlap_high:
                    # Proportional volume based on overlap
                    bar_range = bar_high - bar_low
                    if bar_range > 0:
                        overlap_ratio = (overlap_high - overlap_low) / bar_range
                        level_volume += bar_volume * overlap_ratio

            percent_of_total = (level_volume / total_volume * 100) if total_volume > 0 else 0.0

            level = VolumeProfileLevel(
                price=round(level_price, 2),
                volume=round(level_volume, 0),
                percent_of_total=round(percent_of_total, 2),
            )
            levels.append(level)

        # Find Point of Control (highest volume)
        if levels:
            poc_level = max(levels, key=lambda x: x.volume)
            poc_level.is_poc = True

            # Calculate Value Area (70% of volume around POC)
            sorted_levels = sorted(levels, key=lambda x: x.volume, reverse=True)
            cumulative_volume = 0.0
            value_area_volume = total_volume * self.value_area_percent

            for level in sorted_levels:
                cumulative_volume += level.volume
                level.is_vah = True
                level.is_val = True
                if cumulative_volume >= value_area_volume:
                    break

            # Find actual VAH and VAL
            value_area_levels = [l for l in levels if l.is_vah or l.is_val]
            if value_area_levels:
                vah_level = max(value_area_levels, key=lambda x: x.price)
                val_level = min(value_area_levels, key=lambda x: x.price)
                vah_level.is_vah = True
                val_level.is_val = True

        return levels


class AdvancedChartEngine:
    """Unified advanced chart engine.

    Provides all advanced chart types through a single interface.
    """

    def __init__(self) -> None:
        self.heikin_ashi = HeikinAshiCalculator()
        self.renko = RenkoCalculator()
        self.kagi = KagiCalculator()
        self.point_figure = PointFigureCalculator()
        self.line_break_3 = LineBreakCalculator(break_count=3)
        self.line_break_2 = LineBreakCalculator(break_count=2)
        self.volume_profile = VolumeProfileCalculator()

    def calculate(
        self,
        df: pd.DataFrame,
        chart_type: ChartType,
        **kwargs: Any,
    ) -> Any:
        """Calculate chart data for specified type.

        Args:
            df: OHLCV DataFrame
            chart_type: Type of chart to calculate
            **kwargs: Additional parameters for specific chart types

        Returns:
            Chart-specific data structure
        """
        if chart_type == ChartType.HEIKIN_ASHI:
            return self.heikin_ashi.calculate(df)
        elif chart_type == ChartType.RENKO:
            renko = RenkoCalculator(
                box_size=kwargs.get("box_size"),
                use_atr=kwargs.get("use_atr", True),
                atr_period=kwargs.get("atr_period", 14),
            )
            return renko.calculate(df)
        elif chart_type == ChartType.KAGI:
            kagi = KagiCalculator(
                reversal_amount=kwargs.get("reversal_amount", 0.02),
                use_percentage=kwargs.get("use_percentage", True),
            )
            return kagi.calculate(df)
        elif chart_type == ChartType.POINT_FIGURE:
            pf = PointFigureCalculator(
                box_size=kwargs.get("box_size"),
                reversal_boxes=kwargs.get("reversal_boxes", 3),
                use_atr=kwargs.get("use_atr", False),
            )
            return pf.calculate(df)
        elif chart_type in (ChartType.LINE_BREAK_3, ChartType.LINE_BREAK_2):
            break_count = 3 if chart_type == ChartType.LINE_BREAK_3 else 2
            lb = LineBreakCalculator(break_count=break_count)
            return lb.calculate(df)
        elif chart_type == ChartType.VOLUME_PROFILE:
            vp = VolumeProfileCalculator(
                num_levels=kwargs.get("num_levels", 50),
                value_area_percent=kwargs.get("value_area_percent", 0.70),
            )
            return vp.calculate(df)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")


def get_chart_engine() -> AdvancedChartEngine:
    """Get chart engine instance."""
    return AdvancedChartEngine()
