"""Fold reports retain unavailable values and identify descriptive aggregation."""

from datetime import UTC, datetime
from typing import Any

import pytest

from tradingdev.domain.validation.report import (
    format_walk_forward_report,
    summarize_results,
)
from tradingdev.domain.validation.walk_forward import WalkForwardResult


def _fold(index: int, metrics: dict[str, Any]) -> WalkForwardResult:
    moment = datetime(2024, 1, 1, tzinfo=UTC)
    return WalkForwardResult(
        fold_index=index,
        train_start=moment,
        train_end=moment,
        test_start=moment,
        test_end=moment,
        train_metrics=metrics,
        test_metrics=metrics,
    )


def test_empty_fold_summary() -> None:
    assert summarize_results([]) == {}


@pytest.mark.filterwarnings("error")
def test_all_unavailable_fold_metrics_are_null_without_warnings() -> None:
    results = [
        _fold(index, {"sharpe_ratio": value})
        for index, value in enumerate([None, float("inf"), float("nan"), -float("inf")])
    ]

    assert summarize_results(results) == {
        "n_folds": 4,
        "sharpe_ratio": {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "valid_count": 0,
        },
    }
    report = format_walk_forward_report(results)
    assert "mean=N/A std=N/A [N/A, N/A] valid=0/4" in report
    assert "full-period portfolio performance" in report


@pytest.mark.filterwarnings("error")
def test_summary_counts_only_finite_observations_and_uses_union_of_metric_keys() -> (
    None
):
    results = [
        _fold(0, {"total_return": 0.1}),
        _fold(1, {"total_return": None, "sharpe_ratio": 1.5}),
        _fold(2, {"total_return": 0.3, "sharpe_ratio": float("inf")}),
        _fold(3, {"total_return": float("nan"), "sharpe_ratio": True}),
    ]
    summary = summarize_results(results)

    assert summary["n_folds"] == 4
    assert summary["total_return"] == pytest.approx(
        {"mean": 0.2, "std": 0.1, "min": 0.1, "max": 0.3, "valid_count": 2}
    )
    assert summary["sharpe_ratio"] == {
        "mean": 1.5,
        "std": 0.0,
        "min": 1.5,
        "max": 1.5,
        "valid_count": 1,
    }
    assert "valid=2/4" in format_walk_forward_report(results)


def test_fold_report_keeps_positive_drawdown_magnitude_and_explicit_units() -> None:
    report = format_walk_forward_report(
        [
            _fold(0, {"max_drawdown": 0.05, "max_drawdown_amount": 500.0}),
            _fold(1, {"max_drawdown": 0.15, "max_drawdown_amount": 1500.0}),
        ]
    )

    assert "max_drawdown: mean=10.00% std=5.00% [5.00%, 15.00%] valid=2/2" in report
    assert "max_drawdown_amount: mean=1,000.00 std=500.00" in report
    assert "-5.00%" not in report
    assert "Descriptive test-fold statistics" in report


def test_volume_fold_report_keeps_unavailable_return_keys_explicit() -> None:
    report = format_walk_forward_report(
        [
            _fold(
                0,
                {
                    "total_return": None,
                    "sharpe_ratio": None,
                    "max_drawdown_amount": 25.5,
                },
            )
        ]
    )
    values = {
        line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
        for line in report.splitlines()
        if ":" in line
    }

    assert values["total_return"] == "N/A"
    assert values["sharpe_ratio"] == "N/A"
    assert values["max_drawdown_amount"] == "25.50"
