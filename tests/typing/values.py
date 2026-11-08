"""Static regressions for metric and parameter value preservation."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

from tradingdev.domain.optimization.grid_search import (
    GridSearchResult,
    best_result,
    parameter_grid,
    tuple_grid,
)
from tradingdev.domain.performance.catalog import MetricScalar, summarize_metrics

if TYPE_CHECKING:
    from tradingdev.domain.backtest.result import BacktestResult
    from tradingdev.domain.validation.walk_forward import WalkForwardResult


def value_contracts(backtest: BacktestResult, fold: WalkForwardResult) -> None:
    assert_type(backtest.metrics["sharpe_ratio"], MetricScalar)
    assert_type(fold.test_metrics["sharpe_ratio"], MetricScalar)
    assert_type(summarize_metrics(backtest.metrics, "signal"), dict[str, MetricScalar])
    assert_type(parameter_grid({"length": [10, 20]}), list[dict[str, int]])
    assert_type(tuple_grid([1, 2], [3, 4]), list[tuple[int, ...]])
    result = GridSearchResult(params={"length": 10}, metric_value=1.0, metrics={})
    assert_type(best_result([result], direction="maximize"), GridSearchResult[int])
    _text(backtest.metrics["sharpe_ratio"])  # type: ignore[arg-type]
    _text(parameter_grid({"length": [10]})[0]["length"])  # type: ignore[arg-type]


def _text(value: str) -> None:
    pass
