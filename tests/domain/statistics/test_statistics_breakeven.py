"""Tests for the break-even win rate and turnover cost statistics."""

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.statistics.breakeven import summarize_breakeven

_START = pd.Timestamp("2024-01-01")
_ONE_YEAR = _START + pd.Timedelta(days=365.25)


def _trade(
    net_pnl: float,
    *,
    size_quote: float = 1_000.0,
    entry_price: float = 100.0,
    exit_price: float = 100.0,
    fee: float = 0.0,
) -> dict[str, object]:
    return {
        "net_pnl": net_pnl,
        "size_quote": size_quote,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "fee": fee,
    }


class TestBreakevenWinRate:
    def test_payoff_ratio_and_breakeven_from_known_trades(self) -> None:
        trades = [_trade(30.0), _trade(10.0), _trade(-10.0), _trade(-10.0)]
        summary = summarize_breakeven(trades, start=_START, end=_ONE_YEAR)

        assert summary.avg_win == pytest.approx(20.0)
        assert summary.avg_loss == pytest.approx(10.0)
        assert summary.payoff_ratio == pytest.approx(2.0)
        assert summary.breakeven_win_rate == pytest.approx(1.0 / 3.0)
        assert summary.win_rate == pytest.approx(0.5)
        assert summary.win_rate_edge == pytest.approx(0.5 - 1.0 / 3.0)

    def test_win_rate_ignores_breakeven_trades(self) -> None:
        trades = [_trade(10.0), _trade(-10.0), _trade(0.0), _trade(0.0)]
        summary = summarize_breakeven(trades, start=_START, end=_ONE_YEAR)

        assert summary.total_trades == 4
        assert summary.winning_trades == 1
        assert summary.losing_trades == 1
        assert summary.win_rate == pytest.approx(0.5)
        assert summary.breakeven_win_rate == pytest.approx(0.5)
        assert summary.win_rate_edge == pytest.approx(0.0)

    @pytest.mark.parametrize("seed", range(5))
    def test_edge_sign_matches_total_net_pnl(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        pnls = rng.normal(0.0, 10.0, 50)
        summary = summarize_breakeven(
            [_trade(float(p)) for p in pnls], start=_START, end=_ONE_YEAR
        )

        assert summary.win_rate_edge is not None
        assert np.sign(summary.win_rate_edge) == np.sign(pnls.sum())

    def test_without_losses_breakeven_is_undefined(self) -> None:
        summary = summarize_breakeven(
            [_trade(5.0), _trade(15.0)], start=_START, end=_ONE_YEAR
        )

        assert summary.avg_win == pytest.approx(10.0)
        assert summary.avg_loss is None
        assert summary.payoff_ratio is None
        assert summary.breakeven_win_rate is None
        assert summary.win_rate == pytest.approx(1.0)
        assert summary.win_rate_edge is None

    def test_without_trades_everything_trade_based_is_undefined(self) -> None:
        summary = summarize_breakeven([], start=_START, end=_ONE_YEAR, capital=1e4)

        assert summary.total_trades == 0
        assert summary.win_rate is None
        assert summary.breakeven_win_rate is None
        assert summary.traded_notional == 0.0
        assert summary.annual_turnover == 0.0
        assert summary.cost_per_turnover is None


class TestTurnoverCost:
    def test_counts_entry_and_exit_notional(self) -> None:
        trades = [
            _trade(100.0, size_quote=1_000.0, entry_price=100.0, exit_price=110.0),
            _trade(-50.0, size_quote=2_000.0, entry_price=50.0, exit_price=50.0),
        ]
        summary = summarize_breakeven(trades, start=_START, end=_ONE_YEAR)

        assert summary.traded_notional == pytest.approx(2_100.0 + 4_000.0)

    def test_annualizes_over_the_given_period(self) -> None:
        trades = [_trade(1.0, size_quote=5_000.0, fee=6.0)] * 4
        half_year = _START + pd.Timedelta(days=365.25 / 2)
        summary = summarize_breakeven(
            trades, start=_START, end=half_year, capital=10_000.0
        )

        assert summary.years == pytest.approx(0.5)
        assert summary.traded_notional == pytest.approx(40_000.0)
        assert summary.annual_traded_notional == pytest.approx(80_000.0)
        assert summary.annual_turnover == pytest.approx(8.0)
        assert summary.total_cost == pytest.approx(24.0)
        assert summary.annual_cost == pytest.approx(48.0)
        assert summary.cost_per_turnover == pytest.approx(0.0006)
        assert summary.annual_cost_drag == pytest.approx(0.0048)
        assert summary.annual_turnover is not None
        assert summary.annual_cost_drag == pytest.approx(
            summary.annual_turnover * 0.0006
        )

    def test_without_capital_relative_figures_are_undefined(self) -> None:
        summary = summarize_breakeven(
            [_trade(1.0, fee=1.0)], start=_START, end=_ONE_YEAR
        )

        assert summary.annual_traded_notional == pytest.approx(2_000.0)
        assert summary.cost_per_turnover == pytest.approx(0.0005)
        assert summary.annual_turnover is None
        assert summary.annual_cost_drag is None

    def test_timezone_aware_period_is_measured_in_utc(self) -> None:
        start = pd.Timestamp("2024-01-01 08:00", tz="Asia/Taipei")
        end = pd.Timestamp("2024-01-01 00:00") + pd.Timedelta(days=365.25)
        summary = summarize_breakeven([], start=start, end=end)

        assert summary.years == pytest.approx(1.0)


class TestValidation:
    def test_rejects_non_increasing_period(self) -> None:
        with pytest.raises(ValueError, match="end must be after start"):
            summarize_breakeven([], start=_ONE_YEAR, end=_START)

    @pytest.mark.parametrize("capital", [0.0, -1.0, float("inf")])
    def test_rejects_invalid_capital(self, capital: float) -> None:
        with pytest.raises(ValueError, match="capital"):
            summarize_breakeven([], start=_START, end=_ONE_YEAR, capital=capital)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("fee", None, "no numeric fee"),
            ("net_pnl", True, "no numeric net_pnl"),
            ("size_quote", float("nan"), "non-finite size_quote"),
            ("size_quote", -1.0, "negative size_quote"),
            ("entry_price", 0.0, "invalid entry or exit price"),
            ("exit_price", -1.0, "invalid entry or exit price"),
            ("fee", -0.1, "negative fee"),
        ],
    )
    def test_rejects_invalid_trade_fields(
        self, field: str, value: object, message: str
    ) -> None:
        trade = _trade(1.0)
        trade[field] = value
        with pytest.raises(ValueError, match=f"trade 1 has .*{message}"):
            summarize_breakeven([_trade(1.0), trade], start=_START, end=_ONE_YEAR)
