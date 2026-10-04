"""Numerical regressions for metric semantics and execution ledger consistency."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tradingdev.domain.backtest.metrics import calculate_metrics_from_simulation
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.backtest.volume_engine import VolumeBacktestEngine
from tradingdev.domain.performance.catalog import METRIC_CATALOG, summarize_metrics


def _market(
    prices: list[float], signals: list[int], start: str = "2024-01-01"
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(prices), freq="h", tz="UTC"),
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "signal": signals,
        }
    )


def test_daily_return_risk_uses_sample_std_and_bar_drawdown() -> None:
    # Daily closes are 105, 100, 120. The 120 -> 100 intraday decline
    # is intentionally deeper than the daily-close 105 -> 100 decline.
    equity = np.array([110.0, 105.0, 120.0, 100.0, 110.0, 120.0])
    timestamps = pd.date_range("2024-01-01", periods=6, freq="12h", tz="UTC")
    analysis = calculate_metrics_from_simulation(
        equity,
        [],
        100.0,
        timestamps,
        periods_per_year=3,
        risk_free_rate=0.331,
        required_return=0.331,
    )
    daily_returns = np.array([0.05, -5 / 105, 0.2])
    excess = daily_returns - 0.1
    assert analysis.metrics["total_return"] == pytest.approx(0.2)
    assert analysis.metrics["annual_return"] == pytest.approx(0.2)
    assert analysis.metrics["annual_volatility"] == pytest.approx(
        math.sqrt(3) * float(daily_returns.std(ddof=1))
    )
    assert analysis.metrics["sharpe_ratio"] == pytest.approx(
        math.sqrt(3) * float(excess.mean() / excess.std(ddof=1))
    )
    assert analysis.metrics["sortino_ratio"] == pytest.approx(
        math.sqrt(3)
        * float(excess.mean())
        / math.sqrt(float(np.mean(np.minimum(excess, 0) ** 2)))
    )
    assert analysis.metrics["max_drawdown"] == pytest.approx(20 / 120)
    assert analysis.metrics["daily_max_drawdown"] == pytest.approx(5 / 105)
    assert analysis.metrics["max_drawdown_amount"] == pytest.approx(20)
    assert analysis.metrics["calmar_ratio"] == pytest.approx(4.2)
    assert analysis.metadata["settings"]["risk_free_rate_per_period"] == pytest.approx(
        0.1
    )
    assert analysis.returns is not None
    assert len(analysis.returns) == len(timestamps)


def test_date_aggregation_uses_market_timestamp_and_preserves_first_pnl() -> None:
    df = _market([100.0 + i for i in range(50)], [1] * 50, "2024-01-31")
    result = SignalBacktestEngine(init_cash=1000, fees=0.01, slippage=0).run(df)
    assert result.metrics["n_days"] == 3
    assert result.metrics["n_months"] == 2
    assert result.metrics["daily_pnl_mean"] * 3 == pytest.approx(
        result.metrics["total_pnl"]
    )
    assert result.metrics["monthly_pnl_mean"] * 2 == pytest.approx(
        result.metrics["total_pnl"]
    )
    np.testing.assert_array_equal(df.index.to_numpy(), np.arange(50))


def test_initial_loss_counts_as_drawdown_and_period_pnl() -> None:
    analysis = calculate_metrics_from_simulation(
        np.array([-5.0, -8.0, -3.0]),
        [],
        None,
        pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
    )
    assert analysis.metrics["max_drawdown_amount"] == 8
    assert analysis.metrics["daily_pnl_min"] == -5
    assert analysis.metrics["daily_pnl_mean"] == -1
    assert analysis.metrics["max_drawdown"] is None
    assert analysis.metadata["unavailable"]["max_drawdown"] == "not_applicable"


def test_volume_entry_fee_changes_winning_trade_to_losing_trade() -> None:
    result = VolumeBacktestEngine(
        fees=0.001,
        slippage=0,
        position_size=100,
        signal_as_position=True,
    ).run(_market([100.0, 100.0, 100.15], [1, 0, 0]))
    assert result.metrics["total_pnl"] == pytest.approx(-0.05015)
    assert result.trades[0]["net_pnl"] == pytest.approx(-0.05015)
    assert result.metrics["win_rate"] == 0
    assert result.metrics["profit_factor"] == 0
    assert result.metrics["trade_expectancy"] == pytest.approx(-0.05015)
    assert result.metrics["total_fees"] == pytest.approx(0.20015)
    assert result.metrics["total_volume"] == pytest.approx(200.15)
    for metric_id in ("sharpe_ratio", "annual_return", "total_return"):
        assert result.metrics[metric_id] is None
        assert result.metric_metadata["unavailable"][metric_id] == "not_applicable"


@pytest.mark.parametrize(
    ("prices", "signals", "kwargs"),
    [
        ([100.0, 100.0, 102.0], [1, 1, 1], {}),  # terminal close
        ([100.0, 100.0, 102.0, 101.0], [1, -1, -1, -1], {}),  # reversal
        ([100.0, 100.0, 98.0, 101.0], [1, 1, 1, 1], {"stop_loss": 0.01}),
        ([100.0, 100.0, 102.0, 101.0], [1, 1, 1, 1], {"take_profit": 0.01}),
        ([100.0, 100.0, 98.0, 101.0], [1, 1, 1, 1], {"monthly_max_loss": 1.0}),
        ([100.0, 100.0, 102.0, 101.0], [1, 0, 0, 0], {"signal_as_position": True}),
    ],
)
def test_volume_ledger_reconciles_all_close_paths(
    prices: list[float],
    signals: list[int],
    kwargs: dict[str, Any],
) -> None:
    result = VolumeBacktestEngine(
        fees=0.001,
        slippage=0.002,
        position_size=100,
        **kwargs,
    ).run(_market(prices, signals))
    net_pnl = sum(trade["net_pnl"] for trade in result.trades)
    gross_pnl = sum(trade["gross_pnl"] for trade in result.trades)
    assert result.equity_curve[-1] == pytest.approx(net_pnl)
    assert result.metrics["total_pnl"] == pytest.approx(net_pnl)
    assert result.metrics["total_fees"] == pytest.approx(
        result.metrics["total_volume"] * 0.001
    )
    assert result.metrics["total_slippage"] == pytest.approx(
        result.metrics["total_volume"] * 0.002
    )
    assert net_pnl == pytest.approx(gross_pnl - result.metrics["total_volume"] * 0.003)
    assert result.metrics["open_trades"] == 0


def test_stop_loss_circuit_breaker_does_not_reenter_on_trigger_bar() -> None:
    result = VolumeBacktestEngine(
        fees=0.001,
        slippage=0,
        position_size=100,
        stop_loss=0.01,
        monthly_max_loss=0.5,
    ).run(_market([100.0, 100.0, 98.0, 150.0], [1, 1, 1, 1]))
    assert result.metrics["total_trades"] == 1
    assert result.trades[0]["exit_idx"] == 2
    assert result.equity_curve[-1] == pytest.approx(result.trades[0]["net_pnl"])
    assert result.equity_curve[-2] == pytest.approx(result.equity_curve[-1])


def test_signal_closed_trade_scope_fees_and_executed_notional() -> None:
    df = _market([100.0, 100.0, 110.0, 100.0, 120.0], [1, 0, 1, 1, 1])
    result = SignalBacktestEngine(
        init_cash=1000,
        fees=0.01,
        slippage=0,
        position_size=100,
    ).run(df)
    assert result.metrics["total_trades"] == 1
    assert result.metrics["open_trades"] == 1
    assert result.metrics["win_rate"] == 1
    assert result.metrics["profit_factor"] is None
    assert result.metric_metadata["unavailable"]["profit_factor"] == "unbounded"
    assert result.metrics["trade_expectancy"] == pytest.approx(7.9)
    assert result.metrics["total_fees"] == pytest.approx(3.1)
    assert result.metrics["total_volume"] == pytest.approx(310)
    assert result.metrics["total_pnl"] == pytest.approx(
        sum(t["net_pnl"] for t in result.trades)
    )
    closed, opened = result.trades
    assert closed["fee"] == pytest.approx(2.1)
    assert closed["gross_pnl"] == pytest.approx(10)
    assert closed["exit_notional"] == pytest.approx(110)
    assert opened["exit_notional"] == 0
    assert opened["status"] == "open"


def test_empty_and_flat_mark_unavailable_without_fake_zeros() -> None:
    empty = SignalBacktestEngine(init_cash=100).run(_market([], []))
    assert empty.metrics["total_trades"] == 0
    assert empty.metrics["annual_return"] is None
    assert empty.metric_metadata["unavailable"]["annual_return"] == "insufficient_data"
    flat = SignalBacktestEngine(init_cash=100, periods_per_year=365).run(
        _market([100.0] * 50, [0] * 50)
    )
    assert flat.metrics["win_rate"] is None
    assert flat.metric_metadata["unavailable"]["win_rate"] == "no_trades"
    assert flat.metrics["total_return"] == 0
    assert flat.metrics["annual_volatility"] == 0
    assert flat.metrics["sharpe_ratio"] is None
    assert flat.metric_metadata["unavailable"]["sharpe_ratio"] == "zero_denominator"
    assert set(flat.metric_metadata["unavailable"]) == {
        key for key, value in flat.metrics.items() if value is None
    }


def test_missing_calendar_and_annualization_are_explicit() -> None:
    df = _market([100.0, 101.0, 102.0], [1, 1, 1])
    no_annualization = SignalBacktestEngine(init_cash=100).run(df)
    assert (
        no_annualization.metric_metadata["unavailable"]["sharpe_ratio"]
        == "missing_annualization"
    )
    no_dates = SignalBacktestEngine(init_cash=100, periods_per_year=365).run(
        df.drop(columns=["timestamp"])
    )
    assert no_dates.metrics["n_days"] is None
    assert no_dates.metric_metadata["unavailable"]["n_days"] == "missing_timestamps"
    assert (
        no_dates.metric_metadata["unavailable"]["sharpe_ratio"] == "missing_timestamps"
    )
    assert no_dates.metrics["total_return"] is not None


def test_timestamps_define_observed_days_without_filling_missing_dates() -> None:
    timestamps = pd.DatetimeIndex(["2024-01-01T00:00:00Z", "2024-01-04T00:00:00Z"])
    analysis = calculate_metrics_from_simulation(
        np.array([100.0, 110.0]),
        [],
        100.0,
        timestamps,
        periods_per_year=2,
    )
    assert analysis.metrics["n_days"] == 2
    assert analysis.metrics["annual_return"] == pytest.approx(0.1)
    assert analysis.metadata["settings"]["missing_dates"] == "not_filled"


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1.0, 0.0])
def test_nonpositive_or_nonfinite_prices_fail(bad_value: float) -> None:
    df = _market([100.0, bad_value], [1, 1])
    for engine in (SignalBacktestEngine(init_cash=100), VolumeBacktestEngine()):
        with pytest.raises(ValueError, match="Prices must be finite and positive"):
            engine.run(df)


def test_invalid_equity_or_timestamps_fail_instead_of_fabricating_metrics() -> None:
    with pytest.raises(ValueError, match="Equity must contain only finite values"):
        calculate_metrics_from_simulation(np.array([float("nan")]), [], None)
    df = _market([100.0, 101.0], [1, 1])
    df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
    with pytest.raises(
        ValueError, match="timestamps must be valid, unique, and increasing"
    ):
        SignalBacktestEngine(init_cash=100).run(df)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("periods_per_year", 0),
        ("periods_per_year", float("inf")),
        ("risk_free_rate", -1),
        ("required_return", float("nan")),
        ("init_cash", 0),
    ],
)
def test_invalid_performance_settings_rejected(key: str, value: float) -> None:
    config: dict[str, Any] = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-02",
        "init_cash": 100,
    }
    config[key] = value
    with pytest.raises(ValidationError):
        BacktestConfig.model_validate(config)


def test_catalog_summary_is_a_view_and_does_not_remove_research_values() -> None:
    metrics = {"total_pnl": 42, "daily_pnl_median": 3, "sharpe_ratio": None}
    summary = summarize_metrics(metrics, "volume")
    assert summary == {"total_pnl": 42}
    assert metrics["daily_pnl_median"] == 3
    assert METRIC_CATALOG["max_drawdown_amount"].optimization_direction == "minimize"
    assert METRIC_CATALOG["sharpe_ratio"].requires_annualization


def test_timestamp_length_mismatch_fails_even_for_empty_equity() -> None:
    timestamps = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    for equity in (np.array([1.0]), np.array([], dtype=np.float64)):
        with pytest.raises(ValueError, match="Timestamp and equity lengths must match"):
            calculate_metrics_from_simulation(equity, [], None, timestamps)


def test_negative_capital_retains_pnl_but_rejects_return_interpretation() -> None:
    analysis = calculate_metrics_from_simulation(
        np.array([100.0, -10.0, 20.0]),
        [],
        100.0,
        pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
        periods_per_year=365,
    )
    assert analysis.metrics["total_pnl"] == -80
    assert analysis.metrics["max_drawdown_amount"] == 110
    assert analysis.returns is None
    for metric_id in ("total_return", "sharpe_ratio", "max_drawdown"):
        assert analysis.metrics[metric_id] is None
        assert analysis.metadata["unavailable"][metric_id] == "nonpositive_capital"


def test_finite_equity_does_not_allow_nonfinite_derived_series() -> None:
    with pytest.raises(ValueError, match="Derived returns must be finite"):
        calculate_metrics_from_simulation(np.array([1e308]), [], 1e-320)
    with pytest.raises(ValueError, match="Equity differences must be finite"):
        calculate_metrics_from_simulation(np.array([-1e308]), [], 1e308)


def test_overflowing_annual_growth_is_unavailable_not_nonfinite_json() -> None:
    analysis = calculate_metrics_from_simulation(
        np.array([100.0, 100000.0]),
        [],
        100.0,
        pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
        periods_per_year=365,
    )
    assert analysis.metrics["annual_return"] is None
    assert analysis.metadata["unavailable"]["annual_return"] == "unbounded"
    assert all(
        value is None or math.isfinite(value) for value in analysis.metrics.values()
    )
