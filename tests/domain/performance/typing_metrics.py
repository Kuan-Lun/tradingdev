"""Static metric contracts checked by strict Mypy, not runtime pytest."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

from tradingdev.domain.performance.catalog import MetricScalar, summarize_metrics

if TYPE_CHECKING:
    from tradingdev.domain.backtest.result import BacktestResult
    from tradingdev.domain.validation.walk_forward import WalkForwardResult


def metric_contracts(backtest: BacktestResult, fold: WalkForwardResult) -> None:
    assert_type(backtest.metrics["sharpe_ratio"], MetricScalar)
    assert_type(fold.test_metrics["sharpe_ratio"], MetricScalar)
    assert_type(summarize_metrics(backtest.metrics, "signal"), dict[str, MetricScalar])
    _text(backtest.metrics["sharpe_ratio"])  # type: ignore[arg-type]


def _text(value: str) -> None:
    pass
