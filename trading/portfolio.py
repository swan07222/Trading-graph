# trading/portfolio.py

import copy
import json
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import CONFIG
from core.types import Position, Account
from utils.logger import get_logger

log = get_logger(__name__)

_MAX_EQUITY_HISTORY: int = 10_000
_TRADING_DAYS_PER_YEAR: int = 252
_SAVE_SLICE: int = _MAX_EQUITY_HISTORY  # save everything we keep in memory

@dataclass
class Trade:
    """
    Immutable record of a single executed trade.

    `id` is auto-generated when left empty.
    Basic validation runs in __post_init__.
    """
    id: str
    stock_code: str
    stock_name: str
    side: str  # "buy" | "sell"
    quantity: int
    price: float
    value: float
    commission: float
    stamp_tax: float
    timestamp: datetime
    pnl: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = (
                f"TRD_{datetime.now().strftime('%Y%m%d%H%M%S')}_"
                f"{uuid.uuid4().hex[:8].upper()}"
            )
        if self.side not in ("buy", "sell"):
            raise ValueError(f"Trade.side must be 'buy' or 'sell', got '{self.side}'")
        if self.quantity <= 0:
            raise ValueError(f"Trade.quantity must be > 0, got {self.quantity}")
        if self.price < 0:
            raise ValueError(f"Trade.price must be >= 0, got {self.price}")

@dataclass
class DailyPerformance:
    """Single-day performance snapshot."""
    record_date: date
    starting_equity: float
    ending_equity: float
    pnl: float
    pnl_pct: float
    trades: int
    wins: int
    losses: int

@dataclass
class PortfolioStats:
    """Comprehensive portfolio statistics returned by `get_stats()`."""
    total_value: float
    cash: float
    positions_value: float

    # P&L
    total_pnl: float
    total_pnl_pct: float
    realized_pnl: float
    unrealized_pnl: float
    exposure_pct: float
    cash_ratio_pct: float
    concentration_top1_pct: float
    concentration_top3_pct: float

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    expectancy: float
    payoff_ratio: float
    recovery_factor: float

    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Period P&L
    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl: float
    weekly_pnl_pct: float
    monthly_pnl: float
    monthly_pnl_pct: float

class Portfolio:
    """
    Thread-safe portfolio management system.

    Responsibilities:
        - Track positions, cash, and equity over time
        - Record trades and compute running P&L
        - Calculate risk-adjusted performance metrics
        - Persist / restore state across restarts
    """

    def __init__(self, initial_capital: Optional[float] = None):
        self._lock = threading.RLock()

        self.initial_capital: float = initial_capital or CONFIG.CAPITAL
        self.cash: float = self.initial_capital

        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_performance: List[DailyPerformance] = []

        now = datetime.now()
        self.equity_history: List[Tuple[datetime, float]] = [
            (now, self.initial_capital)
        ]

        self._peak_equity: float = self.initial_capital
        self._max_drawdown: float = 0.0

        self._daily_start_equity: float = self.initial_capital
        self._daily_start_date: date = date.today()

        self._load()

    # ------------------------------------------------------------------
    # Properties (all read under lock)
    # ------------------------------------------------------------------

    @property
    def positions_value(self) -> float:
        with self._lock:
            return sum(p.market_value for p in self.positions.values())

    @property
    def equity(self) -> float:
        with self._lock:
            return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.equity - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return (self.equity / self.initial_capital - 1.0) * 100.0

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def update_from_account(self, account: Account) -> None:
        """
        Sync portfolio state from broker account.

        Uses deep copy so Position objects are fully isolated.
        """
        with self._lock:
            self.cash = account.cash
            self.positions = copy.deepcopy(account.positions)
            self._update_equity_unlocked()

    def record_trade(self, trade: Trade) -> None:
        """Append a completed trade and persist."""
        with self._lock:
            self.trades.append(trade)
            self._update_equity_unlocked()
            self._save_unlocked()

    def get_stats(self) -> PortfolioStats:
        """Calculate comprehensive portfolio statistics (thread-safe)."""
        with self._lock:
            return self._compute_stats()

    def get_equity_curve(self) -> List[Tuple[datetime, float]]:
        """Return a copy of the equity curve."""
        with self._lock:
            return list(self.equity_history)

    def get_trade_history(self) -> List[Trade]:
        """Return a copy of all recorded trades."""
        with self._lock:
            return list(self.trades)

    def get_position_summary(self) -> List[Dict]:
        """Snapshot of current positions sorted by market value descending."""
        with self._lock:
            snapshot = dict(self.positions)

        summary = []
        for code, pos in snapshot.items():
            summary.append({
                "code": code,
                "name": pos.name,
                "quantity": pos.quantity,
                "available": pos.available_qty,
                "cost": pos.avg_cost,
                "price": pos.current_price,
                "value": pos.market_value,
                "pnl": pos.unrealized_pnl,
                "pnl_pct": pos.unrealized_pnl_pct,
            })

        summary.sort(key=lambda x: x["value"], reverse=True)
        return summary

    def reset(self) -> None:
        """Reset portfolio to initial state and remove persisted file."""
        with self._lock:
            self.cash = self.initial_capital
            self.positions.clear()
            self.trades.clear()
            self.daily_performance.clear()
            self.equity_history = [(datetime.now(), self.initial_capital)]
            self._peak_equity = self.initial_capital
            self._max_drawdown = 0.0
            self._daily_start_equity = self.initial_capital
            self._daily_start_date = date.today()

            path = self._save_path()
            if path.exists():
                try:
                    path.unlink()
                except OSError as exc:
                    log.warning("Failed to delete portfolio file: %s", exc)

        log.info("Portfolio reset to initial capital %.2f", self.initial_capital)

    # ------------------------------------------------------------------
    # Internal — equity tracking (caller MUST hold self._lock)
    # ------------------------------------------------------------------

    def _update_equity_unlocked(self) -> None:
        current_equity = self.cash + sum(
            p.market_value for p in self.positions.values()
        )

        self.equity_history.append((datetime.now(), current_equity))

        if len(self.equity_history) > _MAX_EQUITY_HISTORY:
            self.equity_history = self.equity_history[-_MAX_EQUITY_HISTORY:]

        # Peak / drawdown
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        if self._peak_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown

        # Daily roll-over
        today = date.today()
        if today != self._daily_start_date:
            self._roll_daily_performance(current_equity)
            self._daily_start_equity = current_equity
            self._daily_start_date = today

    def _roll_daily_performance(self, current_equity: float) -> None:
        """Close out the previous day's performance record."""
        daily_pnl = current_equity - self._daily_start_equity
        daily_pnl_pct = (
            (daily_pnl / self._daily_start_equity * 100.0)
            if self._daily_start_equity > 0
            else 0.0
        )

        day_trades = [
            t for t in self.trades
            if t.timestamp.date() == self._daily_start_date
        ]
        wins = sum(1 for t in day_trades if t.side == "sell" and t.pnl > 0)
        losses = sum(1 for t in day_trades if t.side == "sell" and t.pnl < 0)

        self.daily_performance.append(DailyPerformance(
            record_date=self._daily_start_date,
            starting_equity=self._daily_start_equity,
            ending_equity=current_equity,
            pnl=daily_pnl,
            pnl_pct=daily_pnl_pct,
            trades=len(day_trades),
            wins=wins,
            losses=losses,
        ))

    # ------------------------------------------------------------------
    # Internal — statistics computation (caller MUST hold self._lock)
    # ------------------------------------------------------------------

    def _compute_stats(self) -> PortfolioStats:
        current_equity = self.cash + sum(
            p.market_value for p in self.positions.values()
        )
        realized_pnl = sum(t.pnl for t in self.trades if t.side == "sell")
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())

        trade_stats = self._compute_trade_stats()
        risk_stats = self._compute_risk_metrics(current_equity)
        period_pnl = self._compute_period_pnl(current_equity)

        return PortfolioStats(
            total_value=current_equity,
            cash=self.cash,
            positions_value=current_equity - self.cash,
            total_pnl=current_equity - self.initial_capital,
            total_pnl_pct=(
                (current_equity / self.initial_capital - 1.0) * 100.0
                if self.initial_capital > 0 else 0.0
            ),
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            **self._compute_exposure_metrics(current_equity),
            **trade_stats,
            **risk_stats,
            **period_pnl,
        )

    def _compute_trade_stats(self) -> Dict:
        sell_trades = [t for t in self.trades if t.side == "sell"]
        total = len(sell_trades)

        if total == 0:
            return dict(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
            )

        wins = [t.pnl for t in sell_trades if t.pnl > 0]
        losses = [t.pnl for t in sell_trades if t.pnl < 0]

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0

        # profit_factor: inf when profitable with no losses
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        return dict(
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / total,
            profit_factor=profit_factor,
            avg_win=float(np.mean(wins)) if wins else 0.0,
            avg_loss=abs(float(np.mean(losses))) if losses else 0.0,
            largest_win=max(wins) if wins else 0.0,
            largest_loss=abs(min(losses)) if losses else 0.0,
            expectancy=(gross_profit - gross_loss) / total if total > 0 else 0.0,
            payoff_ratio=(
                (float(np.mean(wins)) / abs(float(np.mean(losses))))
                if wins and losses and abs(float(np.mean(losses))) > 0
                else 0.0
            ),
            recovery_factor=0.0,  # filled by risk metrics when drawdown is known
        )

    def _compute_risk_metrics(self, current_equity: float) -> Dict:
        returns = self._calculate_returns()

        sharpe_ratio = 0.0
        sortino_ratio = 0.0

        if len(returns) >= 2:
            std = float(np.std(returns, ddof=1))
            mean_ret = float(np.mean(returns))
            if std > 0:
                sharpe_ratio = mean_ret / std * np.sqrt(_TRADING_DAYS_PER_YEAR)

            downside = returns[returns < 0]
            if len(downside) >= 2:
                down_std = float(np.std(downside, ddof=1))
                if down_std > 0:
                    sortino_ratio = mean_ret / down_std * np.sqrt(_TRADING_DAYS_PER_YEAR)

        calmar_ratio = 0.0
        if self._max_drawdown > 0 and len(self.equity_history) >= 2:
            first_ts = self.equity_history[0][0]
            last_ts = self.equity_history[-1][0]
            elapsed_days = max((last_ts - first_ts).days, 1)
            years = elapsed_days / 365.25
            if years > 0 and self.initial_capital > 0:
                annualized_return = (
                    (current_equity / self.initial_capital) ** (1.0 / years) - 1.0
                ) * 100.0
                calmar_ratio = annualized_return / (self._max_drawdown * 100.0)

        max_drawdown_abs = self._max_drawdown * self._peak_equity
        recovery_factor = (
            (current_equity - self.initial_capital) / max_drawdown_abs
            if max_drawdown_abs > 0
            else 0.0
        )

        return dict(
            max_drawdown=max_drawdown_abs,
            max_drawdown_pct=self._max_drawdown * 100.0,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            recovery_factor=recovery_factor,
        )

    def _compute_exposure_metrics(self, current_equity: float) -> Dict:
        if current_equity <= 0:
            return dict(
                exposure_pct=0.0,
                cash_ratio_pct=0.0,
                concentration_top1_pct=0.0,
                concentration_top3_pct=0.0,
            )

        values = sorted(
            [float(p.market_value) for p in self.positions.values() if float(p.market_value) > 0.0],
            reverse=True,
        )
        positions_value = sum(values)
        top1 = values[0] if values else 0.0
        top3 = sum(values[:3]) if values else 0.0

        return dict(
            exposure_pct=(positions_value / current_equity) * 100.0,
            cash_ratio_pct=(self.cash / current_equity) * 100.0,
            concentration_top1_pct=(top1 / current_equity) * 100.0,
            concentration_top3_pct=(top3 / current_equity) * 100.0,
        )

    def _compute_period_pnl(self, current_equity: float) -> Dict:
        daily_pnl = current_equity - self._daily_start_equity
        daily_pnl_pct = (
            (daily_pnl / self._daily_start_equity * 100.0)
            if self._daily_start_equity > 0 else 0.0
        )

        week_ago = date.today() - timedelta(days=7)
        week_eq = self._get_equity_at_date(week_ago)
        weekly_pnl = current_equity - week_eq
        weekly_pnl_pct = (
            (weekly_pnl / week_eq * 100.0) if week_eq > 0 else 0.0
        )

        month_ago = date.today() - timedelta(days=30)
        month_eq = self._get_equity_at_date(month_ago)
        monthly_pnl = current_equity - month_eq
        monthly_pnl_pct = (
            (monthly_pnl / month_eq * 100.0) if month_eq > 0 else 0.0
        )

        return dict(
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl=weekly_pnl,
            weekly_pnl_pct=weekly_pnl_pct,
            monthly_pnl=monthly_pnl,
            monthly_pnl_pct=monthly_pnl_pct,
        )

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------

    def _calculate_returns(self) -> np.ndarray:
        """
        Daily returns from equity history.

        Returns empty array when fewer than 2 distinct trading days exist.
        """
        if len(self.equity_history) < 2:
            return np.array([], dtype=np.float64)

        series = pd.Series(
            [eq for _, eq in self.equity_history],
            index=pd.DatetimeIndex([ts for ts, _ in self.equity_history]),
        ).sort_index()

        daily = series.resample("1D").last().dropna()
        if len(daily) < 2:
            return np.array([], dtype=np.float64)

        # Filter out zero-equity days to avoid division by zero
        daily = daily[daily > 0]
        if len(daily) < 2:
            return np.array([], dtype=np.float64)

        return daily.pct_change().dropna().to_numpy(dtype=np.float64)

    def _get_equity_at_date(self, target_date: date) -> float:
        """
        Find the most recent equity value on or before *target_date*.

        Falls back to `initial_capital` if no history exists before that date.
        Uses binary-style reverse scan (short-circuit).
        """
        for ts, equity in reversed(self.equity_history):
            if ts.date() <= target_date:
                return equity
        return self.initial_capital

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def _save_path() -> Path:
        return CONFIG.DATA_DIR / "portfolio.json"

    def _save_unlocked(self) -> None:
        """
        Persist portfolio state to JSON (caller MUST hold self._lock).

        Saves the full bounded equity_history (consistent with _MAX_EQUITY_HISTORY).
        Also persists daily_performance so it survives restarts.
        """
        path = self._save_path()

        data = {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "peak_equity": self._peak_equity,
            "max_drawdown": self._max_drawdown,
            "daily_start_equity": self._daily_start_equity,
            "daily_start_date": self._daily_start_date.isoformat(),
            "trades": [
                {**asdict(t), "timestamp": t.timestamp.isoformat()}
                for t in self.trades[-_SAVE_SLICE:]
            ],
            "equity_history": [
                (ts.isoformat(), eq)
                for ts, eq in self.equity_history[-_SAVE_SLICE:]
            ],
            "daily_performance": [
                {
                    "record_date": dp.record_date.isoformat(),
                    "starting_equity": dp.starting_equity,
                    "ending_equity": dp.ending_equity,
                    "pnl": dp.pnl,
                    "pnl_pct": dp.pnl_pct,
                    "trades": dp.trades,
                    "wins": dp.wins,
                    "losses": dp.losses,
                }
                for dp in self.daily_performance[-_SAVE_SLICE:]
            ],
            "saved_at": datetime.now().isoformat(),
        }

        try:
            tmp_path = path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            tmp_path.replace(path)  # atomic on POSIX; near-atomic on Windows
        except Exception as exc:
            log.warning("Failed to save portfolio: %s", exc)

    def _load(self) -> None:
        """Load saved portfolio state from disk."""
        path = self._save_path()
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            log.warning("Failed to read portfolio file: %s", exc)
            return

        try:
            self.initial_capital = float(data.get("initial_capital", self.initial_capital))
            self.cash = float(data.get("cash", self.initial_capital))
            self._peak_equity = float(data.get("peak_equity", self.initial_capital))
            self._max_drawdown = float(data.get("max_drawdown", 0.0))
            self._daily_start_equity = float(
                data.get("daily_start_equity", self.initial_capital)
            )

            raw_date = data.get("daily_start_date")
            if raw_date:
                self._daily_start_date = date.fromisoformat(raw_date)

            self.trades = []
            for raw in data.get("trades", []):
                raw["timestamp"] = datetime.fromisoformat(raw["timestamp"])
                self.trades.append(Trade(**raw))

            self.equity_history = [
                (datetime.fromisoformat(ts), float(eq))
                for ts, eq in data.get("equity_history", [])
            ]
            if not self.equity_history:
                self.equity_history = [(datetime.now(), self.initial_capital)]

            self.daily_performance = []
            for raw in data.get("daily_performance", []):
                self.daily_performance.append(DailyPerformance(
                    record_date=date.fromisoformat(raw["record_date"]),
                    starting_equity=float(raw["starting_equity"]),
                    ending_equity=float(raw["ending_equity"]),
                    pnl=float(raw["pnl"]),
                    pnl_pct=float(raw["pnl_pct"]),
                    trades=int(raw["trades"]),
                    wins=int(raw["wins"]),
                    losses=int(raw["losses"]),
                ))

            log.info(
                "Portfolio loaded — capital=%.2f  cash=%.2f  trades=%d  "
                "equity_points=%d  daily_records=%d",
                self.initial_capital,
                self.cash,
                len(self.trades),
                len(self.equity_history),
                len(self.daily_performance),
            )

        except Exception as exc:
            log.warning("Failed to parse portfolio data: %s", exc)
