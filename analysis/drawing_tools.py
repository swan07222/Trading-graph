# analysis/drawing_tools.py
"""Professional Drawing Tools for Technical Analysis.

This module provides comprehensive drawing tools:
- Manual drawing tools (trendlines, channels, horizontal/vertical lines)
- Fibonacci tools (retracement, extension, projection, arcs, fans, time zones)
- Gann tools (Gann Fan, Gann Square, Gann Box)
- Andrews Pitchfork
- Geometric shapes (rectangles, triangles, circles)
- Auto-drawing features (auto trendlines, auto channels, auto fib)
- Pattern recognition (head & shoulders, triangles, flags, wedges)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import get_logger

log = get_logger(__name__)


class DrawingToolType(Enum):
    """Drawing tool types."""
    TRENDLINE = "trendline"
    HORIZONTAL_LINE = "horizontal_line"
    VERTICAL_LINE = "vertical_line"
    CHANNEL = "channel"
    FIB_RETRACEMENT = "fib_retracement"
    FIB_EXTENSION = "fib_extension"
    FIB_PROJECTION = "fib_projection"
    FIB_ARCS = "fib_arcs"
    FIB_FAN = "fib_fan"
    FIB_TIME_ZONES = "fib_time_zones"
    GANN_FAN = "gann_fan"
    GANN_BOX = "gann_box"
    GANN_SQUARE = "gann_square"
    ANDREWS_PITCHFORK = "andrews_pitchfork"
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"
    CIRCLE = "circle"
    RAY = "ray"
    SEGMENT = "segment"
    POLYLINE = "polyline"
    TEXT = "text"
    ARROW = "arrow"


class PatternType(Enum):
    """Chart pattern types."""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    BULL_PENNANT = "bull_pennant"
    BEAR_PENNANT = "bear_pennant"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    CUP_AND_HANDLE = "cup_and_handle"


@dataclass
class Point:
    """2D point for drawing."""
    x: float  # Time/index coordinate
    y: float  # Price coordinate
    label: str = ""

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class DrawingObject:
    """Base drawing object."""
    tool_type: DrawingToolType
    points: list[Point]
    color: str = "#00ff00"
    width: int = 2
    style: str = "solid"  # solid, dashed, dotted
    visible: bool = True
    label: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool_type": self.tool_type.value,
            "points": [(p.x, p.y, p.label) for p in self.points],
            "color": self.color,
            "width": self.width,
            "style": self.style,
            "visible": self.visible,
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass
class Trendline(DrawingObject):
    """Trendline drawing object."""
    def __init__(
        self,
        start: Point,
        end: Point,
        color: str = "#00ff00",
        width: int = 2,
        style: str = "solid",
        label: str = "",
    ) -> None:
        super().__init__(
            tool_type=DrawingToolType.TRENDLINE,
            points=[start, end],
            color=color,
            width=width,
            style=style,
            label=label,
        )

    def get_slope(self) -> float:
        if len(self.points) < 2:
            return 0.0
        p1, p2 = self.points[0], self.points[1]
        if p2.x - p1.x == 0:
            return float("inf")
        return (p2.y - p1.y) / (p2.x - p1.x)

    def get_intercept(self) -> float:
        if len(self.points) < 2:
            return 0.0
        p1 = self.points[0]
        return p1.y - self.get_slope() * p1.x

    def get_value_at(self, x: float) -> float:
        return self.get_slope() * x + self.get_intercept()

    def extend_to(self, x_start: float, x_end: float) -> list[tuple[float, float]]:
        """Extend trendline to specified x range."""
        slope = self.get_slope()
        intercept = self.get_intercept()
        y_start = slope * x_start + intercept
        y_end = slope * x_end + intercept
        return [(x_start, y_start), (x_end, y_end)]


@dataclass
class Channel(DrawingObject):
    """Channel drawing object (parallel lines)."""
    def __init__(
        self,
        trendline_start: Point,
        trendline_end: Point,
        channel_point: Point,
        color: str = "#00ff00",
        width: int = 2,
        label: str = "",
    ) -> None:
        super().__init__(
            tool_type=DrawingToolType.CHANNEL,
            points=[trendline_start, trendline_end, channel_point],
            color=color,
            width=width,
            label=label,
        )

    def get_parallel_line(self) -> list[tuple[float, float]]:
        """Calculate parallel channel line."""
        if len(self.points) < 3:
            return []

        tl = Trendline(self.points[0], self.points[1])
        slope = tl.get_slope()
        intercept = tl.get_intercept()

        # Calculate offset from channel point
        cp = self.points[2]
        expected_y = slope * cp.x + intercept
        offset = cp.y - expected_y

        # Create parallel line
        x_start = min(p.x for p in self.points)
        x_end = max(p.x for p in self.points)
        y1 = slope * x_start + intercept + offset
        y2 = slope * x_end + intercept + offset

        return [(x_start, y1), (x_end, y2)]


@dataclass
class FibonacciRetracement(DrawingObject):
    """Fibonacci retracement levels."""
    def __init__(
        self,
        high_point: Point,
        low_point: Point,
        color: str = "#ff9800",
        label: str = "",
    ) -> None:
        super().__init__(
            tool_type=DrawingToolType.FIB_RETRACEMENT,
            points=[high_point, low_point],
            color=color,
            label=label,
        )
        self.levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.level_colors = [
            "#ff0000", "#ffa500", "#ffff00", "#00ff00", "#00ffff", "#0000ff", "#800080"
        ]

    def get_levels(self) -> list[dict[str, Any]]:
        """Calculate Fibonacci retracement levels."""
        if len(self.points) < 2:
            return []

        high = max(self.points[0].y, self.points[1].y)
        low = min(self.points[0].y, self.points[1].y)
        diff = high - low

        x_start = min(self.points[0].x, self.points[1].x)
        x_end = max(self.points[0].x, self.points[1].x)

        result = []
        for i, level in enumerate(self.levels):
            price = high - diff * level
            result.append({
                "level": level,
                "price": price,
                "color": self.level_colors[i % len(self.level_colors)],
                "line": [(x_start, price), (x_end, price)],
            })

        return result


@dataclass
class FibonacciExtension(DrawingObject):
    """Fibonacci extension levels."""
    def __init__(
        self,
        point_a: Point,  # Start of move
        point_b: Point,  # End of move
        point_c: Point,  # Retracement
        color: str = "#ff9800",
        label: str = "",
    ) -> None:
        super().__init__(
            tool_type=DrawingToolType.FIB_EXTENSION,
            points=[point_a, point_b, point_c],
            color=color,
            label=label,
        )
        self.levels = [0.0, 0.618, 1.0, 1.618, 2.0, 2.618, 3.0]

    def get_levels(self) -> list[dict[str, Any]]:
        """Calculate Fibonacci extension levels."""
        if len(self.points) < 3:
            return []

        a, b, c = self.points
        ab_move = b.y - a.y
        x_start = min(p.x for p in self.points)
        x_end = max(p.x for p in self.points) * 1.5

        result = []
        for level in self.levels:
            if ab_move > 0:  # Uptrend
                price = c.y + ab_move * level
            else:  # Downtrend
                price = c.y + ab_move * level
            result.append({
                "level": level,
                "price": price,
                "line": [(x_start, price), (x_end, price)],
            })

        return result


@dataclass
class AndrewsPitchfork(DrawingObject):
    """Andrews Pitchfork (median line)."""
    def __init__(
        self,
        pivot_a: Point,
        pivot_b: Point,
        pivot_c: Point,
        color: str = "#00ff00",
        label: str = "",
    ) -> None:
        super().__init__(
            tool_type=DrawingToolType.ANDREWS_PITCHFORK,
            points=[pivot_a, pivot_b, pivot_c],
            color=color,
            label=label,
        )

    def get_lines(self) -> dict[str, list[tuple[float, float]]]:
        """Calculate pitchfork lines (median, upper, lower)."""
        if len(self.points) < 3:
            return {}

        a, b, c = self.points

        # Calculate midpoint between B and C
        mid_x = (b.x + c.x) / 2
        mid_y = (b.y + c.y) / 2
        midpoint = Point(mid_x, mid_y)

        # Median line from A through midpoint
        slope = (midpoint.y - a.y) / (midpoint.x - a.x) if midpoint.x != a.x else 0
        intercept = a.y - slope * a.x

        x_start = min(p.x for p in self.points)
        x_end = max(p.x for p in self.points) * 2

        # Median line
        median_line = [
            (x_start, slope * x_start + intercept),
            (x_end, slope * x_end + intercept),
        ]

        # Parallel lines through B and C
        upper_line = [
            (x_start, slope * x_start + (b.y - slope * b.x)),
            (x_end, slope * x_end + (b.y - slope * b.x)),
        ]
        lower_line = [
            (x_start, slope * x_start + (c.y - slope * c.x)),
            (x_end, slope * x_end + (c.y - slope * c.x)),
        ]

        return {
            "median": median_line,
            "upper": upper_line,
            "lower": lower_line,
        }


@dataclass
class GannFan(DrawingObject):
    """Gann Fan (geometric angles)."""
    def __init__(
        self,
        origin: Point,
        end_point: Point,
        color: str = "#00ff00",
        label: str = "",
    ) -> None:
        super().__init__(
            tool_type=DrawingToolType.GANN_FAN,
            points=[origin, end_point],
            color=color,
            label=label,
        )
        self.ratios = [1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, 1, 2, 3, 4]

    def get_lines(self) -> list[dict[str, Any]]:
        """Calculate Gann Fan lines."""
        if len(self.points) < 2:
            return []

        origin, end = self.points
        dx = end.x - origin.x
        dy = end.y - origin.y

        # Calculate base slope (1x1 line)
        base_slope = dy / dx if dx != 0 else 1

        x_start = origin.x - dx
        x_end = end.x + dx

        result = []
        for ratio in self.ratios:
            slope = base_slope * ratio
            intercept = origin.y - slope * origin.x
            result.append({
                "ratio": ratio,
                "line": [
                    (x_start, slope * x_start + intercept),
                    (x_end, slope * x_end + intercept),
                ],
            })

        return result


@dataclass
class ChartPattern:
    """Detected chart pattern."""
    pattern_type: PatternType
    points: list[Point]
    confidence: float
    breakout_price: float = 0.0
    target_price: float = 0.0
    stop_loss: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pattern_type": self.pattern_type.value,
            "points": [(p.x, p.y, p.label) for p in self.points],
            "confidence": self.confidence,
            "breakout_price": self.breakout_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "metadata": self.metadata,
        }


class DrawingToolsManager:
    """Manager for drawing tools."""

    def __init__(self) -> None:
        self.drawings: list[DrawingObject] = []
        self.patterns: list[ChartPattern] = []

    def add_drawing(self, drawing: DrawingObject) -> None:
        """Add a drawing object."""
        self.drawings.append(drawing)

    def remove_drawing(self, index: int) -> None:
        """Remove a drawing by index."""
        if 0 <= index < len(self.drawings):
            self.drawings.pop(index)

    def clear_drawings(self) -> None:
        """Clear all drawings."""
        self.drawings.clear()

    def get_drawings_by_type(self, tool_type: DrawingToolType) -> list[DrawingObject]:
        """Get drawings by type."""
        return [d for d in self.drawings if d.tool_type == tool_type]

    def export_drawings(self) -> list[dict]:
        """Export all drawings to dictionary."""
        return [d.to_dict() for d in self.drawings]

    def import_drawings(self, data: list[dict]) -> None:
        """Import drawings from dictionary."""
        self.drawings.clear()
        for item in data:
            tool_type = DrawingToolType(item["tool_type"])
            points = [Point(x=p[0], y=p[1], label=p[2] if len(p) > 2 else "") for p in item["points"]]
            drawing = DrawingObject(
                tool_type=tool_type,
                points=points,
                color=item.get("color", "#00ff00"),
                width=item.get("width", 2),
                style=item.get("style", "solid"),
                visible=item.get("visible", True),
                label=item.get("label", ""),
                metadata=item.get("metadata", {}),
            )
            self.drawings.append(drawing)


class AutoDrawingEngine:
    """Automatic drawing engine for technical analysis."""

    def __init__(self) -> None:
        self.min_points = 50

    def find_extrema(
        self,
        high: pd.Series,
        low: pd.Series,
        order: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find local extrema in price data."""
        from scipy.signal import argrelextrema

        high_extrema = argrelextrema(high.values, np.less, order=order)[0]
        low_extrema = argrelextrema(low.values, np.greater, order=order)[0]

        return high_extrema, low_extrema

    def fit_trendline(
        self,
        prices: pd.Series,
        indices: np.ndarray,
        min_r_squared: float = 0.7,
    ) -> tuple[float, float, float] | None:
        """Fit trendline to price points using linear regression."""
        if len(indices) < 2:
            return None

        x = indices
        y = prices.iloc[indices].values

        slope, intercept, r_value, _, _ = stats.linregress(x, y)

        if r_value ** 2 < min_r_squared:
            return None

        return slope, intercept, r_value ** 2

    def detect_auto_trendlines(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        min_touches: int = 3,
    ) -> list[Trendline]:
        """Automatically detect significant trendlines."""
        trendlines = []
        high_extrema, low_extrema = self.find_extrema(high, low, order=10)

        # Detect resistance lines (from highs)
        if len(high_extrema) >= min_touches:
            for i in range(len(high_extrema) - min_touches + 1):
                indices = high_extrema[i:i + min_touches]
                result = self.fit_trendline(high, indices)
                if result:
                    slope, intercept, r_squared = result
                    start_idx = indices[0]
                    end_idx = indices[-1]
                    trendlines.append(Trendline(
                        start=Point(x=float(start_idx), y=high.iloc[start_idx], label="R1"),
                        end=Point(x=float(end_idx), y=high.iloc[end_idx], label="R2"),
                        color="#ff0000",
                        label=f"Resistance (R²={r_squared:.2f})",
                    ))

        # Detect support lines (from lows)
        if len(low_extrema) >= min_touches:
            for i in range(len(low_extrema) - min_touches + 1):
                indices = low_extrema[i:i + min_touches]
                result = self.fit_trendline(low, indices)
                if result:
                    slope, intercept, r_squared = result
                    start_idx = indices[0]
                    end_idx = indices[-1]
                    trendlines.append(Trendline(
                        start=Point(x=float(start_idx), y=low.iloc[start_idx], label="S1"),
                        end=Point(x=float(end_idx), y=low.iloc[end_idx], label="S2"),
                        color="#00ff00",
                        label=f"Support (R²={r_squared:.2f})",
                    ))

        return trendlines

    def detect_auto_channels(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> list[Channel]:
        """Automatically detect price channels."""
        channels = []
        trendlines = self.detect_auto_trendlines(high, low, close)

        for tl in trendlines:
            slope = tl.get_slope()
            intercept = tl.get_intercept()

            # Find parallel touches
            if slope > 0:  # Uptrend - look for lower parallel
                prices = low
            else:  # Downtrend - look for upper parallel
                prices = high

            # Calculate expected values
            expected = slope * np.arange(len(prices)) + intercept
            actual = prices.values
            deviation = actual - expected

            # Find significant deviation for parallel line
            if len(deviation) > 0:
                if slope > 0:
                    offset = np.percentile(deviation[deviation < 0], 10)
                else:
                    offset = np.percentile(deviation[deviation > 0], 90)

                if abs(offset) > 0:
                    channels.append(Channel(
                        trendline_start=tl.points[0],
                        trendline_end=tl.points[1],
                        channel_point=Point(
                            x=tl.points[0].x,
                            y=tl.get_value_at(tl.points[0].x) + offset,
                        ),
                        color="#0000ff",
                        label="Auto Channel",
                    ))

        return channels

    def detect_auto_fibonacci(
        self,
        high: pd.Series,
        low: pd.Series,
        lookback: int = 100,
    ) -> list[FibonacciRetracement]:
        """Automatically detect Fibonacci retracement levels."""
        fibs = []

        # Find significant swing high and low
        recent_high_idx = high.rolling(window=lookback).max().idxmax()
        recent_low_idx = low.rolling(window=lookback).min().idxmin()

        if pd.isna(recent_high_idx) or pd.isna(recent_low_idx):
            return fibs

        recent_high_idx = int(recent_high_idx)
        recent_low_idx = int(recent_low_idx)

        # Determine trend direction
        if recent_high_idx > recent_low_idx:
            # Uptrend - fib from high to low
            high_point = Point(x=float(recent_high_idx), y=float(high.iloc[recent_high_idx]))
            low_point = Point(x=float(recent_low_idx), y=float(low.iloc[recent_low_idx]))
        else:
            # Downtrend - fib from low to high
            high_point = Point(x=float(recent_low_idx), y=float(high.iloc[recent_low_idx]))
            low_point = Point(x=float(recent_high_idx), y=float(low.iloc[recent_high_idx]))

        fibs.append(FibonacciRetracement(
            high_point=high_point,
            low_point=low_point,
            label="Auto Fib",
        ))

        return fibs

    def detect_pattern(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        pattern_type: PatternType,
    ) -> ChartPattern | None:
        """Detect specific chart pattern."""
        if pattern_type == PatternType.HEAD_AND_SHOULDERS:
            return self._detect_head_and_shoulders(high, low, close)
        elif pattern_type == PatternType.DOUBLE_TOP:
            return self._detect_double_top(high, low, close)
        elif pattern_type == PatternType.DOUBLE_BOTTOM:
            return self._detect_double_bottom(high, low, close)
        elif pattern_type in (PatternType.ASCENDING_TRIANGLE, PatternType.DESCENDING_TRIANGLE, PatternType.SYMMETRICAL_TRIANGLE):
            return self._detect_triangle(high, low, close, pattern_type)
        return None

    def _detect_head_and_shoulders(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> ChartPattern | None:
        """Detect head and shoulders pattern."""
        high_extrema, _ = self.find_extrema(high, low, order=15)

        if len(high_extrema) < 5:
            return None

        # Look for 3 peaks with middle one highest
        for i in range(len(high_extrema) - 4):
            left_idx = high_extrema[i]
            head_idx = high_extrema[i + 1]
            right_idx = high_extrema[i + 2]

            left_shoulder = high.iloc[left_idx]
            head = high.iloc[head_idx]
            right_shoulder = high.iloc[right_idx]

            # Head should be highest
            if head > left_shoulder and head > right_shoulder:
                # Shoulders should be roughly equal
                shoulder_diff = abs(left_shoulder - right_shoulder) / head
                if shoulder_diff < 0.1:  # Within 10%
                    # Find neckline (low between shoulders)
                    trough1_idx = low.iloc[left_idx:head_idx].idxmin()
                    trough2_idx = low.iloc[head_idx:right_idx].idxmin()

                    neckline_price = (low.iloc[trough1_idx] + low.iloc[trough2_idx]) / 2

                    pattern = ChartPattern(
                        pattern_type=PatternType.HEAD_AND_SHOULDERS,
                        points=[
                            Point(x=float(left_idx), y=left_shoulder, label="Left Shoulder"),
                            Point(x=float(head_idx), y=head, label="Head"),
                            Point(x=float(right_idx), y=right_shoulder, label="Right Shoulder"),
                            Point(x=float(trough1_idx), y=low.iloc[trough1_idx], label="Neckline 1"),
                            Point(x=float(trough2_idx), y=low.iloc[trough2_idx], label="Neckline 2"),
                        ],
                        confidence=1.0 - shoulder_diff,
                        breakout_price=neckline_price,
                        target_price=neckline_price - (head - neckline_price),
                        stop_loss=head,
                    )
                    return pattern

        return None

    def _detect_double_top(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> ChartPattern | None:
        """Detect double top pattern."""
        high_extrema, _ = self.find_extrema(high, low, order=15)

        if len(high_extrema) < 2:
            return None

        for i in range(len(high_extrema) - 1):
            peak1_idx = high_extrema[i]
            peak2_idx = high_extrema[i + 1]

            peak1 = high.iloc[peak1_idx]
            peak2 = high.iloc[peak2_idx]

            # Peaks should be roughly equal
            peak_diff = abs(peak1 - peak2) / peak1
            if peak_diff < 0.03:  # Within 3%
                # Find trough between peaks
                trough_idx = low.iloc[peak1_idx:peak2_idx].idxmin()
                neckline = low.iloc[trough_idx]

                pattern = ChartPattern(
                    pattern_type=PatternType.DOUBLE_TOP,
                    points=[
                        Point(x=float(peak1_idx), y=peak1, label="Peak 1"),
                        Point(x=float(peak2_idx), y=peak2, label="Peak 2"),
                        Point(x=float(trough_idx), y=neckline, label="Neckline"),
                    ],
                    confidence=1.0 - peak_diff,
                    breakout_price=neckline,
                    target_price=neckline - (peak1 - neckline),
                    stop_loss=peak1,
                )
                return pattern

        return None

    def _detect_double_bottom(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> ChartPattern | None:
        """Detect double bottom pattern."""
        _, low_extrema = self.find_extrema(high, low, order=15)

        if len(low_extrema) < 2:
            return None

        for i in range(len(low_extrema) - 1):
            trough1_idx = low_extrema[i]
            trough2_idx = low_extrema[i + 1]

            trough1 = low.iloc[trough1_idx]
            trough2 = low.iloc[trough2_idx]

            # Troughs should be roughly equal
            trough_diff = abs(trough1 - trough2) / trough1
            if trough_diff < 0.03:  # Within 3%
                # Find peak between troughs
                peak_idx = high.iloc[trough1_idx:trough2_idx].idxmax()
                neckline = high.iloc[peak_idx]

                pattern = ChartPattern(
                    pattern_type=PatternType.DOUBLE_BOTTOM,
                    points=[
                        Point(x=float(trough1_idx), y=trough1, label="Trough 1"),
                        Point(x=float(trough2_idx), y=trough2, label="Trough 2"),
                        Point(x=float(peak_idx), y=neckline, label="Neckline"),
                    ],
                    confidence=1.0 - trough_diff,
                    breakout_price=neckline,
                    target_price=neckline + (neckline - trough1),
                    stop_loss=trough1,
                )
                return pattern

        return None

    def _detect_triangle(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        pattern_type: PatternType,
    ) -> ChartPattern | None:
        """Detect triangle patterns."""
        high_extrema, low_extrema = self.find_extrema(high, low, order=10)

        if len(high_extrema) < 3 or len(low_extrema) < 2:
            return None

        # Get recent extrema points
        highs = [(idx, high.iloc[idx]) for idx in high_extrema[-5:]]
        lows = [(idx, low.iloc[idx]) for idx in low_extrema[-5:]]

        if len(highs) < 2 or len(lows) < 2:
            return None

        # Fit trendlines
        high_indices = np.array([h[0] for h in highs])
        low_indices = np.array([l[0] for l in lows])
        high_prices = pd.Series([h[1] for h in highs])
        low_prices = pd.Series([l[1] for l in lows])

        high_result = self.fit_trendline(pd.Series(high_prices), high_indices, min_r_squared=0.5)
        low_result = self.fit_trendline(pd.Series(low_prices), low_indices, min_r_squared=0.5)

        if not high_result or not low_result:
            return None

        high_slope, _, high_r2 = high_result
        low_slope, _, low_r2 = low_result

        # Determine triangle type
        confidence = (high_r2 + low_r2) / 2

        if pattern_type == PatternType.ASCENDING_TRIANGLE:
            # Flat resistance, rising support
            if abs(high_slope) < 0.01 and low_slope > 0:
                return ChartPattern(
                    pattern_type=PatternType.ASCENDING_TRIANGLE,
                    points=[Point(x=h[0], y=h[1]) for h in highs] + [Point(x=l[0], y=l[1]) for l in lows],
                    confidence=confidence,
                )
        elif pattern_type == PatternType.DESCENDING_TRIANGLE:
            # Flat support, falling resistance
            if abs(low_slope) < 0.01 and high_slope < 0:
                return ChartPattern(
                    pattern_type=PatternType.DESCENDING_TRIANGLE,
                    points=[Point(x=h[0], y=h[1]) for h in highs] + [Point(x=l[0], y=l[1]) for l in lows],
                    confidence=confidence,
                )
        elif pattern_type == PatternType.SYMMETRICAL_TRIANGLE:
            # Converging trendlines
            if high_slope < 0 and low_slope > 0:
                return ChartPattern(
                    pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
                    points=[Point(x=h[0], y=h[1]) for h in highs] + [Point(x=l[0], y=l[1]) for l in lows],
                    confidence=confidence,
                )

        return None

    def scan_all_patterns(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> list[ChartPattern]:
        """Scan for all supported patterns."""
        patterns = []

        for pattern_type in PatternType:
            pattern = self.detect_pattern(high, low, close, pattern_type)
            if pattern and pattern.confidence > 0.6:
                patterns.append(pattern)

        return patterns


def get_drawing_tools() -> DrawingToolsManager:
    """Get drawing tools manager instance."""
    return DrawingToolsManager()


def get_auto_drawing() -> AutoDrawingEngine:
    """Get auto drawing engine instance."""
    return AutoDrawingEngine()
