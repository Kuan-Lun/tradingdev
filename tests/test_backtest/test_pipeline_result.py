"""Tests for PipelineResult construction and serialisation."""

from __future__ import annotations

import pickle
from datetime import UTC, datetime
from typing import Any

import numpy as np

from tradingdev.backtest.pipeline_result import PipelineResult
from tradingdev.backtest.result import BacktestResult
from tradingdev.validation.walk_forward import WalkForwardResult


def _make_backtest_result(**overrides: Any) -> BacktestResult:
    defaults: dict[str, Any] = {
        "metrics": {"total_return": 0.05, "sharpe_ratio": 1.2},
        "equity_curve": np.array([10000.0, 10050.0, 10100.0]),
        "trades": [{"net_pnl": 50.0}],
        "init_cash": 10_000.0,
        "mode": "signal",
    }
    defaults.update(overrides)
    return BacktestResult(**defaults)


class TestPipelineResultSimple:
    def test_simple_mode_construction(self) -> None:
        result = _make_backtest_result()
        pipeline = PipelineResult(
            mode="simple",
            backtest_result=result,
            config_snapshot={"strategy": {"name": "test"}},
        )
        assert pipeline.mode == "simple"
        assert pipeline.backtest_result is result
        assert pipeline.fold_results == []

    def test_simple_mode_pickle_roundtrip(self) -> None:
        result = _make_backtest_result()
        pipeline = PipelineResult(
            mode="simple",
            backtest_result=result,
            config_snapshot={"backtest": {"init_cash": 10000}},
        )
        data = pickle.dumps(pipeline)
        loaded: PipelineResult = pickle.loads(data)  # noqa: S301
        assert loaded.mode == "simple"
        assert loaded.backtest_result is not None
        np.testing.assert_array_equal(
            loaded.backtest_result.equity_curve,
            result.equity_curve,
        )
        assert loaded.config_snapshot == pipeline.config_snapshot


class TestPipelineResultWalkForward:
    def test_walk_forward_mode_construction(self) -> None:
        fold = WalkForwardResult(
            fold_index=0,
            train_start=datetime(2024, 1, 1, tzinfo=UTC),
            train_end=datetime(2024, 6, 1, tzinfo=UTC),
            test_start=datetime(2024, 6, 1, tzinfo=UTC),
            test_end=datetime(2024, 12, 1, tzinfo=UTC),
            train_metrics={"total_return": 0.03},
            test_metrics={"total_return": 0.02},
            train_backtest=_make_backtest_result(),
            test_backtest=_make_backtest_result(),
        )
        pipeline = PipelineResult(
            mode="walk_forward",
            fold_results=[fold],
            config_snapshot={"strategy": {"name": "xgb"}},
        )
        assert pipeline.mode == "walk_forward"
        assert pipeline.backtest_result is None
        assert len(pipeline.fold_results) == 1
        assert pipeline.fold_results[0].train_backtest is not None

    def test_walk_forward_pickle_roundtrip(self) -> None:
        fold = WalkForwardResult(
            fold_index=0,
            train_start=datetime(2024, 1, 1, tzinfo=UTC),
            train_end=datetime(2024, 6, 1, tzinfo=UTC),
            test_start=datetime(2024, 6, 1, tzinfo=UTC),
            test_end=datetime(2024, 12, 1, tzinfo=UTC),
            train_backtest=_make_backtest_result(),
            test_backtest=_make_backtest_result(),
        )
        pipeline = PipelineResult(
            mode="walk_forward",
            fold_results=[fold],
            config_snapshot={"validation": {"n_splits": 3}},
        )
        data = pickle.dumps(pipeline)
        loaded: PipelineResult = pickle.loads(data)  # noqa: S301
        assert loaded.mode == "walk_forward"
        assert len(loaded.fold_results) == 1
        f = loaded.fold_results[0]
        assert f.train_backtest is not None
        assert f.test_backtest is not None
        assert loaded.config_snapshot == pipeline.config_snapshot
