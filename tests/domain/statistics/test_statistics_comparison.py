"""Tests for the paired bootstrap comparison of strategies."""

import math
from functools import partial

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from tradingdev.domain.statistics.comparison import (
    align_returns,
    difference_interval,
    interval_table,
)
from tradingdev.domain.statistics.sharpe import (
    MIN_BOOTSTRAP_OBSERVATIONS,
    sharpe_ratio,
)

_SHARPE = partial(sharpe_ratio, periods_per_year=365.0)


def _daily(values: npt.NDArray[np.float64], start: str = "2024-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=index)


def _correlated_pair(
    n: int, rho: float, edge: float, seed: int
) -> tuple[pd.Series, pd.Series]:
    """Two daily return series with correlation ``rho``; ``a`` has a mean edge."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.01, n)
    own_a = rng.normal(0.0, 0.01, n)
    own_b = rng.normal(0.0, 0.01, n)
    shared, spread = math.sqrt(rho), math.sqrt(1.0 - rho)
    a = shared * common + spread * own_a + edge
    b = shared * common + spread * own_b
    return _daily(a), _daily(b)


class TestAlignReturns:
    def test_keeps_only_common_periods_and_reports_dropped(self) -> None:
        a = _daily(np.arange(10, dtype=float), start="2024-01-01")
        b = _daily(np.arange(10, dtype=float), start="2024-01-06")

        aligned = align_returns({"a": a, "b": b}, min_overlap=5)

        assert aligned.overlap == 5
        assert dict(aligned.lengths) == {"a": 10, "b": 10}
        assert aligned.sufficient
        assert list(aligned.frame.columns) == ["a", "b"]
        assert aligned.frame.index[0] == pd.Timestamp("2024-01-06", tz="UTC")
        assert aligned.frame["a"].tolist() == [5.0, 6.0, 7.0, 8.0, 9.0]
        assert aligned.frame["b"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_drops_gaps_inside_the_period(self) -> None:
        a = _daily(np.ones(30))
        b = a.drop(a.index[[3, 17]])

        aligned = align_returns({"a": a, "b": b})

        assert aligned.overlap == 28
        assert aligned.lengths["a"] - aligned.overlap == 2

    def test_reports_insufficient_overlap(self) -> None:
        a = _daily(np.ones(30), start="2024-01-01")
        b = _daily(np.ones(30), start="2024-01-21")

        aligned = align_returns({"a": a, "b": b})

        assert aligned.overlap == 10
        assert aligned.min_overlap == MIN_BOOTSTRAP_OBSERVATIONS
        assert not aligned.sufficient

    def test_sorts_the_common_index(self) -> None:
        a = _daily(np.arange(25, dtype=float))
        shuffled = a.iloc[::-1]

        aligned = align_returns({"a": shuffled, "b": a})

        assert aligned.frame.index.is_monotonic_increasing
        assert (aligned.frame["a"] == aligned.frame["b"]).all()

    @pytest.mark.parametrize(
        ("series", "message"),
        [
            ({}, "at least one"),
            ({"a": _daily(np.array([0.0, math.nan]))}, "finite"),
            (
                {"a": pd.Series([0.1, 0.2], index=[0, 0])},
                "duplicate",
            ),
        ],
    )
    def test_rejects_invalid_input(
        self, series: dict[str, pd.Series], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            align_returns(series)


class TestDifferenceInterval:
    def test_detects_a_known_difference(self) -> None:
        a, b = _correlated_pair(730, rho=0.8, edge=0.002, seed=1)

        result = difference_interval(a, b, _SHARPE, reps=1000, seed=0)

        assert result is not None
        assert result.lower > 0.0
        assert result.lower <= result.difference <= result.upper
        assert result.p_value < 0.05
        assert result.difference == pytest.approx(result.estimate_a - result.estimate_b)
        assert result.estimate_a == pytest.approx(_SHARPE(a.to_numpy()))
        assert result.estimate_b == pytest.approx(_SHARPE(b.to_numpy()))
        assert result.overlap == 730

    def test_is_antisymmetric_under_a_shared_seed(self) -> None:
        a, b = _correlated_pair(400, rho=0.5, edge=0.001, seed=2)

        forward = difference_interval(a, b, _SHARPE, reps=500, seed=3)
        backward = difference_interval(b, a, _SHARPE, reps=500, seed=3)

        assert forward is not None
        assert backward is not None
        assert forward.difference == pytest.approx(-backward.difference)
        assert forward.lower == pytest.approx(-backward.upper)
        assert forward.p_value == pytest.approx(backward.p_value)

    def test_identically_distributed_series_straddle_zero(self) -> None:
        a, b = _correlated_pair(500, rho=0.5, edge=0.0, seed=4)

        result = difference_interval(a, b, _SHARPE, reps=1000, seed=0)

        assert result is not None
        assert result.lower < 0.0 < result.upper
        assert result.p_value > 0.05

    def test_false_positive_rate_is_near_nominal(self) -> None:
        rng = np.random.default_rng(5)
        trials = 300
        rejections = 0
        for trial in range(trials):
            a = _daily(rng.normal(0.0, 0.01, 250))
            b = _daily(rng.normal(0.0, 0.01, 250))
            result = difference_interval(a, b, _SHARPE, reps=300, seed=trial)
            assert result is not None
            rejections += result.p_value < 0.05

        assert 0.02 <= rejections / trials <= 0.10

    def test_interval_and_p_value_agree(self) -> None:
        for seed in range(10):
            a, b = _correlated_pair(300, rho=0.7, edge=0.0008, seed=seed)
            result = difference_interval(a, b, _SHARPE, reps=400, seed=seed)
            assert result is not None
            excludes_zero = result.lower > 0.0 or result.upper < 0.0
            assert excludes_zero == (result.p_value < 0.05)

    def test_high_correlation_narrows_the_difference_interval(self) -> None:
        a, b = _correlated_pair(730, rho=0.95, edge=0.0005, seed=6)

        paired = difference_interval(a, b, _SHARPE, reps=1000, seed=0)
        table = interval_table({"a": a, "b": b}, _SHARPE, reps=1000, seed=0)

        assert paired is not None
        assert table is not None
        row_a, row_b = table["a"], table["b"]
        assert row_a is not None
        assert row_b is not None
        separate = (row_a.upper - row_a.lower) + (row_b.upper - row_b.lower)
        assert paired.upper - paired.lower < 0.25 * separate

    def test_reports_the_pearson_correlation(self) -> None:
        a, b = _correlated_pair(1000, rho=0.9, edge=0.0, seed=7)
        expected = float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])

        result = difference_interval(a, b, _SHARPE, reps=50, seed=0)

        assert result is not None
        assert result.correlation == pytest.approx(expected)
        assert result.correlation == pytest.approx(0.9, abs=0.02)

    def test_correlation_uses_aligned_periods_only(self) -> None:
        a, b = _correlated_pair(200, rho=0.6, edge=0.0, seed=8)
        trimmed = b.iloc[50:]
        expected = float(np.corrcoef(a.iloc[50:].to_numpy(), trimmed.to_numpy())[0, 1])

        result = difference_interval(a, trimmed, _SHARPE, reps=50, seed=0)

        assert result is not None
        assert result.overlap == 150
        assert result.correlation == pytest.approx(expected)

    def test_correlation_is_none_for_a_constant_series(self) -> None:
        a = _daily(np.random.default_rng(9).normal(0.0, 0.01, 60))
        flat = _daily(np.zeros(60))

        result = difference_interval(a, flat, lambda v: float(np.mean(v)), reps=50)

        assert result is not None
        assert result.correlation is None

    def test_insufficient_overlap_returns_none(self) -> None:
        rng = np.random.default_rng(10)
        a = _daily(rng.normal(0.0, 0.01, 100), start="2024-01-01")
        b = _daily(rng.normal(0.0, 0.01, 100), start="2024-03-31")

        assert align_returns({"a": a, "b": b}).overlap < MIN_BOOTSTRAP_OBSERVATIONS
        assert difference_interval(a, b, _SHARPE, reps=50) is None

    def test_undefined_statistic_returns_none(self) -> None:
        a = _daily(np.random.default_rng(11).normal(0.0, 0.01, 60))
        flat = _daily(np.zeros(60))

        assert difference_interval(a, flat, _SHARPE, reps=50) is None

    def test_accepts_any_statistic(self) -> None:
        a, b = _correlated_pair(400, rho=0.5, edge=0.0, seed=12)

        def volatility(values: npt.NDArray[np.float64]) -> float:
            return float(np.std(values, ddof=1) * math.sqrt(365.0))

        result = difference_interval(2.0 * a, b, volatility, reps=500, seed=0)

        assert result is not None
        assert result.lower > 0.0

    def test_is_reproducible_with_a_seed(self) -> None:
        a, b = _correlated_pair(200, rho=0.5, edge=0.0, seed=13)

        first = difference_interval(a, b, _SHARPE, reps=200, seed=42)
        second = difference_interval(a, b, _SHARPE, reps=200, seed=42)

        assert first == second

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"reps": 0}, "reps"),
            ({"reps": True}, "reps"),
            ({"confidence": 1.0}, "confidence"),
            ({"confidence": 0.0}, "confidence"),
        ],
    )
    def test_rejects_invalid_arguments(
        self, kwargs: dict[str, float], message: str
    ) -> None:
        a, b = _correlated_pair(50, rho=0.5, edge=0.0, seed=14)

        with pytest.raises(ValueError, match=message):
            difference_interval(a, b, _SHARPE, **kwargs)  # type: ignore[arg-type]


class TestIntervalTable:
    def test_rows_cover_the_common_period_in_input_order(self) -> None:
        rng = np.random.default_rng(15)
        series = {
            "late": _daily(rng.normal(0.001, 0.01, 300), start="2024-02-01"),
            "early": _daily(rng.normal(0.0, 0.01, 300), start="2024-01-01"),
        }

        table = interval_table(series, _SHARPE, reps=500, seed=0)

        assert table is not None
        assert list(table) == ["late", "early"]
        common = align_returns(series).frame
        for name, row in table.items():
            assert row is not None
            assert row.overlap == len(common)
            assert row.estimate == pytest.approx(_SHARPE(common[name].to_numpy()))
            assert row.lower <= row.estimate <= row.upper
            assert row.valid_reps == 500

    def test_undefined_rows_are_none(self) -> None:
        series = {
            "active": _daily(np.random.default_rng(16).normal(0.0, 0.01, 60)),
            "flat": _daily(np.zeros(60)),
        }

        table = interval_table(series, _SHARPE, reps=100, seed=0)

        assert table is not None
        assert table["active"] is not None
        assert table["flat"] is None

    def test_insufficient_overlap_returns_none(self) -> None:
        series = {"a": _daily(np.random.default_rng(17).normal(0.0, 0.01, 10))}

        assert interval_table(series, _SHARPE, reps=50) is None
