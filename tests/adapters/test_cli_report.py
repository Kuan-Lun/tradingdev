"""CLI report behavior for available and unavailable numerical metrics."""

from __future__ import annotations

from typing import Any

import pytest

from tradingdev.adapters.cli.report import format_metrics_report


def _metrics() -> dict[str, Any]:
    return {
        "total_pnl": 1250.0,
        "total_return": 0.125,
        "annual_return": 0.25,
        "max_drawdown": 0.05,
        "sharpe_ratio": 1.25,
        "win_rate": 0.75,
        "profit_factor": 3.0,
        "total_trades": 4,
        "total_volume": 2000.0,
        "daily_pnl_mean": 625.0,
        "daily_pnl_std": 125.0,
        "daily_pnl_min": -100.0,
        "daily_pnl_max": 1000.0,
        "daily_pnl_median": 625.0,
        "n_days": 2,
        "monthly_pnl_mean": 1250.0,
        "monthly_pnl_std": 0.0,
        "monthly_pnl_min": 1250.0,
        "monthly_pnl_max": 1250.0,
        "monthly_pnl_median": 1250.0,
        "monthly_trades_mean": 4.0,
        "monthly_volume_mean": 2000.0,
        "n_months": 1,
    }


@pytest.mark.parametrize("mode", ["signal", "volume"])
def test_report_preserves_finite_numbers_and_units(mode: str) -> None:
    metrics = _metrics()
    if mode == "volume":
        metrics["max_drawdown"] = 250.0

    lines = format_metrics_report(metrics, mode).splitlines()

    assert "  Total P&L:             +1,250" in lines
    assert "  Sharpe Ratio:          1.2500" in lines
    assert "  Win Rate:              75.00%" in lines
    assert "  Profit Factor:         3.0000" in lines
    assert "  Total Trades:               4" in lines
    assert "  Total Volume:              2,000" in lines
    assert "  Est. Monthly Vol:         30,000" in lines
    assert "  Daily P&L  min:       -100.00" in lines
    assert "  Monthly P&L  std:          0.00" in lines
    assert "  Period:                  2 days" in lines
    assert "  Period:                    1 months" in lines
    if mode == "volume":
        assert "  Max Drawdown:            -250 USDT" in lines
        assert not any("Return:" in line for line in lines)
    else:
        assert "  Total Return:          12.50%" in lines
        assert "  Annual Return:         25.00%" in lines
        assert "  Max Drawdown:          -5.00%" in lines


@pytest.mark.parametrize("mode", ["signal", "volume"])
@pytest.mark.parametrize("value", [None, float("inf"), float("-inf"), float("nan")])
def test_report_marks_unavailable_metrics_without_numeric_formatting(
    mode: str, value: float | None
) -> None:
    metrics = dict.fromkeys(_metrics(), value)

    report = format_metrics_report(metrics, mode)

    values = [
        line.split(":", 1)[1].strip() for line in report.splitlines() if ":" in line
    ]
    assert values
    assert all(
        value in {"N/A", "N/A days", "N/A months", "N/A USDT"} for value in values
    )


@pytest.mark.parametrize("mode", ["signal", "volume"])
def test_report_marks_missing_metrics_as_unavailable(mode: str) -> None:
    report = format_metrics_report({}, mode)

    values = [
        line.split(":", 1)[1].strip() for line in report.splitlines() if ":" in line
    ]
    assert values
    assert all(value.startswith("N/A") for value in values)


@pytest.mark.parametrize("key", ["total_volume", "n_days"])
@pytest.mark.parametrize("value", [None, float("inf"), float("-inf"), float("nan")])
def test_monthly_volume_is_unavailable_when_either_input_is_unavailable(
    key: str, value: float | None
) -> None:
    metrics = _metrics()
    metrics[key] = value

    report = format_metrics_report(metrics)

    line = next(line for line in report.splitlines() if "Est. Monthly Vol:" in line)
    assert line.split(":", 1)[1].strip() == "N/A"


def test_report_preserves_zero_day_monthly_volume_behavior() -> None:
    metrics = _metrics()
    metrics["n_days"] = 0

    report = format_metrics_report(metrics)

    assert "  Est. Monthly Vol:         60,000" in report.splitlines()
