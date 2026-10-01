"""Tests for the engine-independent performance summary."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from tradingdev.domain.statistics.performance import summarize_performance
from tradingdev.domain.statistics.periods import period_pnl


def _hourly_2024() -> tuple[npt.NDArray[np.float64], pd.DatetimeIndex]:
    timestamps = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
    rng = np.random.default_rng(7)
    equity = 10_000.0 + np.cumsum(rng.normal(0.5, 5.0, len(timestamps)))
    return equity, timestamps


class TestFullYearSummary:
    def test_counts_366_days_and_12_months(self) -> None:
        equity, timestamps = _hourly_2024()
        summary = summarize_performance(equity, timestamps, [])

        assert summary.n_bars == 8784
        assert summary.n_days == 366
        assert summary.n_months == 12
        assert summary.start == pd.Timestamp("2024-01-01", tz="UTC")
        assert summary.end == pd.Timestamp("2024-12-31 23:00", tz="UTC")

    def test_daily_and_monthly_pnl_add_up_to_total_pnl(self) -> None:
        equity, timestamps = _hourly_2024()
        summary = summarize_performance(equity, timestamps, [])

        assert summary.total_pnl == pytest.approx(equity[-1] - equity[0])
        assert summary.daily_pnl_mean * summary.n_days == pytest.approx(
            summary.total_pnl
        )
        assert summary.monthly_pnl_mean * summary.n_months == pytest.approx(
            summary.total_pnl
        )

    def test_std_is_measured_on_daily_not_per_bar_pnl(self) -> None:
        equity, timestamps = _hourly_2024()
        summary = summarize_performance(equity, timestamps, [])

        daily = period_pnl(equity, timestamps, "daily")
        assert summary.daily_pnl_std == pytest.approx(float(daily.std(ddof=1)))
        per_bar_std = float(np.diff(equity).std(ddof=1))
        assert summary.daily_pnl_std is not None
        assert summary.daily_pnl_std > 3 * per_bar_std

    def test_sharpe_annualizes_daily_returns(self) -> None:
        equity, timestamps = _hourly_2024()
        daily_equity = pd.Series(equity, index=timestamps).resample("D").last()
        returns = daily_equity / daily_equity.shift(1).fillna(equity[0]) - 1.0
        expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)

        summary = summarize_performance(equity, timestamps, [], periods_per_year=252)

        assert summary.sharpe_ratio == pytest.approx(expected)


class TestEquityStatistics:
    def test_drawdown_and_return_from_known_curve(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=5, freq="D")
        summary = summarize_performance(
            [100.0, 120.0, 90.0, 110.0, 130.0], timestamps, []
        )

        assert summary.total_pnl == pytest.approx(30.0)
        assert summary.total_return == pytest.approx(0.3)
        assert summary.max_drawdown == pytest.approx(30.0)
        assert summary.max_drawdown_pct == pytest.approx(0.25)

    def test_cumulative_pnl_curve_has_no_return_figures(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=4, freq="D")
        summary = summarize_performance([0.0, 50.0, -20.0, 40.0], timestamps, [])

        assert summary.total_pnl == pytest.approx(40.0)
        assert summary.max_drawdown == pytest.approx(70.0)
        assert summary.total_return is None
        assert summary.max_drawdown_pct is None
        assert summary.sharpe_ratio is None

    def test_single_day_has_no_dispersion(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=3, freq="h")
        summary = summarize_performance([100.0, 101.0, 102.0], timestamps, [])

        assert summary.n_days == 1
        assert summary.daily_pnl_std is None
        assert summary.monthly_pnl_std is None
        assert summary.sharpe_ratio is None

    def test_flat_curve_has_no_sharpe(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=5, freq="D")
        summary = summarize_performance([100.0] * 5, timestamps, [])

        assert summary.sharpe_ratio is None
        assert summary.max_drawdown == 0.0


class TestTradeStatistics:
    _timestamps = pd.date_range("2024-01-01", periods=2, freq="D")

    def test_counts_wins_losses_and_breakeven(self) -> None:
        trades: list[dict[str, object]] = [
            {"net_pnl": 30.0},
            {"net_pnl": -10.0},
            {"net_pnl": 0.0},
            {"net_pnl": np.float64(20.0)},
            {"net_pnl": np.int64(-15)},
        ]
        summary = summarize_performance([100.0, 125.0], self._timestamps, trades)

        assert summary.total_trades == 5
        assert summary.winning_trades == 2
        assert summary.losing_trades == 2
        assert summary.win_rate == pytest.approx(0.4)
        assert summary.profit_factor == pytest.approx(50.0 / 25.0)
        assert summary.trade_net_pnl == pytest.approx(25.0)

    def test_no_trades(self) -> None:
        summary = summarize_performance([100.0, 100.0], self._timestamps, [])

        assert summary.total_trades == 0
        assert summary.win_rate is None
        assert summary.profit_factor is None
        assert summary.trade_net_pnl == 0.0

    def test_no_losing_trades_leaves_profit_factor_undefined(self) -> None:
        trades = [{"net_pnl": 5.0}, {"net_pnl": 0.0}]
        summary = summarize_performance([100.0, 105.0], self._timestamps, trades)

        assert summary.win_rate == pytest.approx(0.5)
        assert summary.profit_factor is None

    @pytest.mark.parametrize(
        "trade",
        [{}, {"net_pnl": None}, {"net_pnl": "1.0"}, {"net_pnl": True}],
    )
    def test_rejects_trade_without_numeric_net_pnl(
        self, trade: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError, match="trade 0 has no numeric net_pnl"):
            summarize_performance([100.0, 100.0], self._timestamps, [trade])

    def test_rejects_non_finite_net_pnl(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            summarize_performance(
                [100.0, 100.0], self._timestamps, [{"net_pnl": float("inf")}]
            )

    def test_rejects_non_positive_periods_per_year(self) -> None:
        with pytest.raises(ValueError, match="periods_per_year"):
            summarize_performance(
                [100.0, 100.0], self._timestamps, [], periods_per_year=0
            )
