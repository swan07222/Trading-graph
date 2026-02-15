from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis.backtest import Backtester, BacktestResult
from config.settings import CONFIG


def _mk_result(
    *,
    total_return: float,
    excess_return: float,
    sharpe: float,
    sortino: float,
    max_dd_pct: float,
    win_rate: float,
    profit_factor: float,
    trades: int,
    fold_acc: float,
) -> BacktestResult:
    return BacktestResult(
        total_return=total_return,
        excess_return=excess_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        information_ratio=0.0,
        alpha=0.0,
        beta=0.0,
        max_drawdown=0.0,
        max_drawdown_pct=max_dd_pct,
        calmar_ratio=0.0,
        volatility=0.0,
        total_trades=trades,
        winning_trades=max(0, int(trades * win_rate)),
        losing_trades=max(0, trades - int(trades * win_rate)),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=0.0,
        avg_loss=0.0,
        avg_holding_days=0.0,
        num_folds=3,
        avg_fold_accuracy=fold_acc,
    )


def test_backtest_score_prefers_higher_quality():
    a = _mk_result(
        total_return=12.0,
        excess_return=9.0,
        sharpe=1.3,
        sortino=1.7,
        max_dd_pct=8.0,
        win_rate=0.58,
        profit_factor=1.4,
        trades=24,
        fold_acc=0.62,
    )
    b = _mk_result(
        total_return=7.0,
        excess_return=3.0,
        sharpe=0.7,
        sortino=0.9,
        max_dd_pct=13.0,
        win_rate=0.49,
        profit_factor=1.1,
        trades=18,
        fold_acc=0.54,
    )
    assert Backtester._score_result(a) > Backtester._score_result(b)


def test_backtest_optimize_returns_best_and_restores_confidence(monkeypatch):
    bt = Backtester.__new__(Backtester)

    old_conf = float(CONFIG.model.min_confidence)

    def fake_run(
        self,
        stock_codes=None,
        train_months=12,
        test_months=1,
        min_data_days=500,
        initial_capital=None,
    ):
        conf = float(CONFIG.model.min_confidence)
        # Make train=12/test=1/conf=0.7 the best combination.
        quality = (conf * 100.0) + (5.0 if int(train_months) == 12 else 0.0) - (2.0 if int(test_months) != 1 else 0.0)
        return _mk_result(
            total_return=quality / 10.0,
            excess_return=(quality - 50.0) / 10.0,
            sharpe=max(0.1, quality / 100.0),
            sortino=max(0.1, quality / 90.0),
            max_dd_pct=max(1.0, 25.0 - (quality / 5.0)),
            win_rate=min(0.90, max(0.30, quality / 120.0)),
            profit_factor=min(3.0, max(1.0, quality / 60.0)),
            trades=20,
            fold_acc=min(0.95, max(0.40, quality / 130.0)),
        )

    monkeypatch.setattr(Backtester, "run", fake_run, raising=True)

    summary = Backtester.optimize(
        bt,
        train_months_options=[9, 12],
        test_months_options=[1, 2],
        min_confidence_options=[0.6, 0.7],
        top_k=3,
    )

    assert summary["status"] == "ok"
    assert summary["best"]["train_months"] == 12
    assert summary["best"]["test_months"] == 1
    assert abs(float(summary["best"]["min_confidence"]) - 0.7) < 1e-9
    assert abs(float(CONFIG.model.min_confidence) - old_conf) < 1e-9


def test_backtest_resolve_horizon_falls_back_for_daily_interval():
    bt = Backtester.__new__(Backtester)

    old_h = int(getattr(CONFIG.model, "prediction_horizon", 1) or 1)
    try:
        CONFIG.model.prediction_horizon = 30
        assert bt._resolve_backtest_horizon("1d") == 1
        assert bt._resolve_backtest_horizon("1m") == 30
    finally:
        CONFIG.model.prediction_horizon = old_h


def test_backtest_fold_uses_configurable_backtest_epochs(monkeypatch):
    bt = Backtester.__new__(Backtester)

    class _DummyFeatureEngine:
        def get_feature_columns(self):
            return ["close"]

        def create_features(self, df):
            return df.copy()

    class _DummyProcessor:
        def fit_scaler(self, _arr):
            return None

        def create_labels(self, df, horizon=None):
            out = df.copy()
            out["label"] = 1.0
            out["future_return"] = 0.0
            return out

        def prepare_sequences(self, df, feature_cols, fit_scaler=False, return_index=False):
            n = len(df)
            X = np.zeros((n, 4, len(feature_cols)), dtype=np.float32)
            y = np.ones((n,), dtype=np.int64)
            r = np.zeros((n,), dtype=np.float64)
            if return_index:
                return X, y, r, pd.DatetimeIndex(df.index)
            return X, y, r

    train_calls = {"epochs": None}

    class _DummyModel:
        def __init__(self, input_size, model_names=None):
            self.input_size = input_size

        def train(self, X_train, y_train, X_val, y_val, epochs=None):
            train_calls["epochs"] = int(epochs)
            return {}

        def predict_batch(self, X):
            return [
                SimpleNamespace(predicted_class=1, confidence=0.8)
                for _ in range(len(X))
            ]

    bt.feature_engine = _DummyFeatureEngine()

    monkeypatch.setattr("analysis.backtest.DataProcessor", _DummyProcessor)
    monkeypatch.setattr("analysis.backtest.EnsembleModel", _DummyModel)

    dates = pd.date_range("2024-01-01", periods=260, freq="D")
    raw = pd.DataFrame(
        {
            "open": np.linspace(10.0, 12.0, len(dates)),
            "high": np.linspace(10.2, 12.2, len(dates)),
            "low": np.linspace(9.8, 11.8, len(dates)),
            "close": np.linspace(10.0, 12.0, len(dates)),
            "volume": np.full(len(dates), 100000.0),
        },
        index=dates,
    )
    all_data = {"600519": raw}

    old_backtest_epochs = getattr(CONFIG.model, "backtest_epochs", None)
    try:
        CONFIG.model.backtest_epochs = 3
        result = bt._run_fold(
            all_data=all_data,
            train_start=dates[0],
            train_end=dates[180],
            test_start=dates[190],
            test_end=dates[230],
            capital=100000.0,
        )
        assert result is not None
        assert train_calls["epochs"] == 3
    finally:
        if old_backtest_epochs is None:
            try:
                delattr(CONFIG.model, "backtest_epochs")
            except AttributeError:
                pass
        else:
            CONFIG.model.backtest_epochs = old_backtest_epochs


def test_backtest_run_fails_fast_when_train_window_cannot_build_sequences(monkeypatch):
    bt = Backtester.__new__(Backtester)

    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    raw = pd.DataFrame(
        {
            "open": np.linspace(10.0, 12.0, len(dates)),
            "high": np.linspace(10.2, 12.2, len(dates)),
            "low": np.linspace(9.8, 11.8, len(dates)),
            "close": np.linspace(10.0, 12.0, len(dates)),
            "volume": np.full(len(dates), 100000.0),
        },
        index=dates,
    )
    all_data = {"600519": raw}

    monkeypatch.setattr(
        Backtester,
        "_get_stock_list",
        lambda self, stock_codes=None: ["600519"],
        raising=True,
    )
    monkeypatch.setattr(
        Backtester,
        "_collect_data",
        lambda self, stocks, min_days: all_data,
        raising=True,
    )
    monkeypatch.setattr(
        Backtester,
        "_generate_folds",
        lambda self, min_date, max_date, train_months, test_months: [
            (dates[0], dates[20], dates[25], dates[40]),
            (dates[10], dates[30], dates[35], dates[50]),
        ],
        raising=True,
    )

    fold_calls = {"count": 0}

    def _run_fold_unexpected(*args, **kwargs):
        fold_calls["count"] += 1
        return None

    monkeypatch.setattr(Backtester, "_run_fold", _run_fold_unexpected, raising=True)

    old_seq = int(getattr(CONFIG.model, "sequence_length", 60) or 60)
    old_h = int(getattr(CONFIG.model, "prediction_horizon", 1) or 1)
    try:
        CONFIG.model.sequence_length = 60
        CONFIG.model.prediction_horizon = 30
        with pytest.raises(ValueError, match="train window too short"):
            bt.run(train_months=1, test_months=1, min_data_days=1)
        assert fold_calls["count"] == 0
    finally:
        CONFIG.model.sequence_length = old_seq
        CONFIG.model.prediction_horizon = old_h


def test_backtest_fold_skips_when_train_sequences_too_small(monkeypatch):
    bt = Backtester.__new__(Backtester)

    class _DummyFeatureEngine:
        def get_feature_columns(self):
            return ["close"]

        def create_features(self, df):
            return df.copy()

    class _DummyProcessor:
        def fit_scaler(self, _arr):
            return None

        def create_labels(self, df, horizon=None):  # noqa: ARG002
            out = df.copy()
            out["label"] = 1.0
            out["future_return"] = 0.0
            return out

        def prepare_sequences(
            self,
            df,
            feature_cols,
            fit_scaler=False,  # noqa: ARG002
            return_index=False,
        ):
            _ = (df, feature_cols)
            X = np.zeros((1, 4, 1), dtype=np.float32)
            y = np.ones((1,), dtype=np.int64)
            r = np.zeros((1,), dtype=np.float64)
            if return_index:
                idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01")])
                return X, y, r, idx
            return X, y, r

    def _unexpected_model(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("EnsembleModel must not be constructed for tiny train set")

    bt.feature_engine = _DummyFeatureEngine()
    monkeypatch.setattr("analysis.backtest.DataProcessor", _DummyProcessor)
    monkeypatch.setattr("analysis.backtest.EnsembleModel", _unexpected_model)
    monkeypatch.setattr(
        Backtester,
        "_backtest_train_row_requirement",
        lambda self, interval="1d": (5, 1, 5),  # noqa: ARG005
        raising=True,
    )

    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    raw = pd.DataFrame(
        {
            "open": np.linspace(10.0, 11.0, len(dates)),
            "high": np.linspace(10.2, 11.2, len(dates)),
            "low": np.linspace(9.8, 10.8, len(dates)),
            "close": np.linspace(10.0, 11.0, len(dates)),
            "volume": np.full(len(dates), 100000.0),
        },
        index=dates,
    )

    result = bt._run_fold(
        all_data={"600519": raw},
        train_start=dates[0],
        train_end=dates[20],
        test_start=dates[25],
        test_end=dates[35],
        capital=100000.0,
    )
    assert result is None
