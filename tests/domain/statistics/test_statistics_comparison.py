"""Tests for the paired bootstrap comparison of strategies."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from tradingdev.domain.statistics.comparison import (
    align_returns,
    difference_interval,
    difference_interval_outcome,
    interval_table,
    interval_table_outcome,
)
from tradingdev.domain.statistics.periods import period_returns
from tradingdev.domain.statistics.sharpe import (
    MIN_BOOTSTRAP_OBSERVATIONS,
    IntervalUnavailable,
    sharpe_ratio,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class _Frequency:
    """A return frequency: its pandas period and its annualisation."""

    name: str
    pandas_freq: str
    periods_per_year: float

    def series(self, values: npt.NDArray[np.float64], offset: int = 0) -> pd.Series:
        """Returns indexed by consecutive periods, starting ``offset`` in."""
        index = pd.date_range(
            "2024-01-01", periods=len(values) + offset, freq=self.pandas_freq, tz="UTC"
        )
        return pd.Series(values, index=index[offset:])

    def sharpe(self, values: npt.NDArray[np.float64]) -> float | None:
        return sharpe_ratio(values, periods_per_year=self.periods_per_year)


_DAILY = _Frequency("daily", "D", 365.0)
_MONTHLY = _Frequency("monthly", "MS", 12.0)


@pytest.fixture(params=[_DAILY, _MONTHLY], ids=lambda frequency: frequency.name)
def frequency(request: pytest.FixtureRequest) -> _Frequency:
    """Each comparison test runs on daily and on monthly returns."""
    result: _Frequency = request.param
    return result


def _daily(values: npt.NDArray[np.float64], start: str = "2024-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=index)


def _correlated_pair(
    n: int, rho: float, edge: float, seed: int, frequency: _Frequency = _DAILY
) -> tuple[pd.Series, pd.Series]:
    """Two return series with correlation ``rho``; ``a`` has a mean edge."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.01, n)
    own_a = rng.normal(0.0, 0.01, n)
    own_b = rng.normal(0.0, 0.01, n)
    shared, spread = math.sqrt(rho), math.sqrt(1.0 - rho)
    a = shared * common + spread * own_a + edge
    b = shared * common + spread * own_b
    return frequency.series(a), frequency.series(b)


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

    def test_rejects_mixed_naive_and_aware_indexes(self) -> None:
        aware = _daily(np.ones(30))
        naive = aware.tz_localize(None)

        with pytest.raises(ValueError, match="timezone mismatch") as raised:
            align_returns({"strategy": aware, "benchmark": naive})

        assert "naive: ['benchmark']" in str(raised.value)
        assert "aware: ['strategy']" in str(raised.value)

    def test_mixed_timezones_reach_inference_as_an_error(self) -> None:
        a, b = _correlated_pair(60, 0.5, 0.0, seed=1)
        statistic = _DAILY.sharpe

        with pytest.raises(ValueError, match="timezone mismatch"):
            difference_interval(a, b.tz_localize(None), statistic, reps=50, seed=0)

    def test_aligns_aware_indexes_in_different_timezones(self) -> None:
        utc = _daily(np.arange(30, dtype=float))
        taipei = utc.tz_convert("Asia/Taipei")

        aligned = align_returns({"utc": utc, "taipei": taipei})

        assert aligned.overlap == 30
        assert (aligned.frame["utc"] == aligned.frame["taipei"]).all()

    def test_aligns_all_naive_indexes(self) -> None:
        a = _daily(np.arange(30, dtype=float)).tz_localize(None)

        assert align_returns({"a": a, "b": a.iloc[5:]}).overlap == 25


class TestDifferenceInterval:
    def test_detects_a_known_difference(self, frequency: _Frequency) -> None:
        a, b = _correlated_pair(730, rho=0.8, edge=0.002, seed=1, frequency=frequency)

        result = difference_interval(a, b, frequency.sharpe, reps=1000, seed=0)

        assert result is not None
        assert result.lower > 0.0
        assert result.lower <= result.difference <= result.upper
        assert result.p_value < 0.05
        assert result.difference == pytest.approx(result.estimate_a - result.estimate_b)
        assert result.estimate_a == pytest.approx(frequency.sharpe(a.to_numpy()))
        assert result.estimate_b == pytest.approx(frequency.sharpe(b.to_numpy()))
        assert result.overlap == 730

    def test_is_antisymmetric_under_a_shared_seed(self, frequency: _Frequency) -> None:
        a, b = _correlated_pair(400, rho=0.5, edge=0.001, seed=2, frequency=frequency)

        forward = difference_interval(a, b, frequency.sharpe, reps=500, seed=3)
        backward = difference_interval(b, a, frequency.sharpe, reps=500, seed=3)

        assert forward is not None
        assert backward is not None
        assert forward.difference == pytest.approx(-backward.difference)
        assert forward.lower == pytest.approx(-backward.upper)
        assert forward.p_value == pytest.approx(backward.p_value)

    def test_identically_distributed_series_straddle_zero(
        self, frequency: _Frequency
    ) -> None:
        a, b = _correlated_pair(500, rho=0.5, edge=0.0, seed=4, frequency=frequency)

        result = difference_interval(a, b, frequency.sharpe, reps=1000, seed=0)

        assert result is not None
        assert result.lower < 0.0 < result.upper
        assert result.p_value > 0.05

    def test_false_positive_rate_is_near_nominal(self, frequency: _Frequency) -> None:
        rng = np.random.default_rng(5)
        trials = 300
        rejections = 0
        for trial in range(trials):
            a = frequency.series(rng.normal(0.0, 0.01, 250))
            b = frequency.series(rng.normal(0.0, 0.01, 250))
            result = difference_interval(a, b, frequency.sharpe, reps=300, seed=trial)
            assert result is not None
            rejections += result.p_value < 0.05

        assert 0.02 <= rejections / trials <= 0.10

    def test_interval_and_p_value_agree(self, frequency: _Frequency) -> None:
        for seed in range(10):
            a, b = _correlated_pair(
                300, rho=0.7, edge=0.0008, seed=seed, frequency=frequency
            )
            result = difference_interval(a, b, frequency.sharpe, reps=400, seed=seed)
            assert result is not None
            excludes_zero = result.lower > 0.0 or result.upper < 0.0
            assert excludes_zero == (result.p_value < 0.05)

    def test_high_correlation_narrows_the_difference_interval(
        self, frequency: _Frequency
    ) -> None:
        a, b = _correlated_pair(730, rho=0.95, edge=0.0005, seed=6, frequency=frequency)

        paired = difference_interval(a, b, frequency.sharpe, reps=1000, seed=0)
        table = interval_table({"a": a, "b": b}, frequency.sharpe, reps=1000, seed=0)

        assert paired is not None
        assert table is not None
        row_a, row_b = table["a"], table["b"]
        assert row_a is not None
        assert row_b is not None
        separate = (row_a.upper - row_a.lower) + (row_b.upper - row_b.lower)
        assert paired.upper - paired.lower < 0.25 * separate

    def test_reports_the_pearson_correlation(self, frequency: _Frequency) -> None:
        a, b = _correlated_pair(1000, rho=0.9, edge=0.0, seed=7, frequency=frequency)
        expected = float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])

        result = difference_interval(a, b, frequency.sharpe, reps=50, seed=0)

        assert result is not None
        assert result.correlation == pytest.approx(expected)
        assert result.correlation == pytest.approx(0.9, abs=0.02)

    def test_correlation_uses_aligned_periods_only(self, frequency: _Frequency) -> None:
        a, b = _correlated_pair(200, rho=0.6, edge=0.0, seed=8, frequency=frequency)
        trimmed = b.iloc[50:]
        expected = float(np.corrcoef(a.iloc[50:].to_numpy(), trimmed.to_numpy())[0, 1])

        result = difference_interval(a, trimmed, frequency.sharpe, reps=50, seed=0)

        assert result is not None
        assert result.overlap == 150
        assert result.correlation == pytest.approx(expected)

    def test_correlation_is_none_for_a_constant_series(
        self, frequency: _Frequency
    ) -> None:
        a = frequency.series(np.random.default_rng(9).normal(0.0, 0.01, 60))
        flat = frequency.series(np.zeros(60))

        result = difference_interval(a, flat, lambda v: float(np.mean(v)), reps=50)

        assert result is not None
        assert result.correlation is None

    def test_insufficient_overlap_returns_none(self, frequency: _Frequency) -> None:
        rng = np.random.default_rng(10)
        a = frequency.series(rng.normal(0.0, 0.01, 100))
        b = frequency.series(rng.normal(0.0, 0.01, 100), offset=90)

        assert align_returns({"a": a, "b": b}).overlap < MIN_BOOTSTRAP_OBSERVATIONS
        assert difference_interval(a, b, frequency.sharpe, reps=50) is None

    def test_undefined_statistic_returns_none(self, frequency: _Frequency) -> None:
        a = frequency.series(np.random.default_rng(11).normal(0.0, 0.01, 60))
        flat = frequency.series(np.zeros(60))

        assert difference_interval(a, flat, frequency.sharpe, reps=50) is None

    def test_accepts_any_statistic(self, frequency: _Frequency) -> None:
        a, b = _correlated_pair(400, rho=0.5, edge=0.0, seed=12, frequency=frequency)

        def volatility(values: npt.NDArray[np.float64]) -> float:
            return float(np.std(values, ddof=1) * math.sqrt(frequency.periods_per_year))

        result = difference_interval(2.0 * a, b, volatility, reps=500, seed=0)

        assert result is not None
        assert result.lower > 0.0

    def test_is_reproducible_with_a_seed(self, frequency: _Frequency) -> None:
        a, b = _correlated_pair(200, rho=0.5, edge=0.0, seed=13, frequency=frequency)

        first = difference_interval(a, b, frequency.sharpe, reps=200, seed=42)
        second = difference_interval(a, b, frequency.sharpe, reps=200, seed=42)

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
        self, frequency: _Frequency, kwargs: dict[str, float], message: str
    ) -> None:
        a, b = _correlated_pair(50, rho=0.5, edge=0.0, seed=14, frequency=frequency)

        with pytest.raises(ValueError, match=message):
            difference_interval(a, b, frequency.sharpe, **kwargs)  # type: ignore[arg-type]


class TestIntervalTable:
    def test_rows_cover_the_common_period_in_input_order(
        self, frequency: _Frequency
    ) -> None:
        rng = np.random.default_rng(15)
        series = {
            "late": frequency.series(rng.normal(0.001, 0.01, 300), offset=31),
            "early": frequency.series(
                rng.normal(0.0, 0.01, 300),
            ),
        }

        table = interval_table(series, frequency.sharpe, reps=500, seed=0)

        assert table is not None
        assert list(table) == ["late", "early"]
        common = align_returns(series).frame
        for name, row in table.items():
            assert row is not None
            assert row.overlap == len(common)
            assert row.estimate == pytest.approx(
                frequency.sharpe(common[name].to_numpy())
            )
            assert row.lower <= row.estimate <= row.upper
            assert row.valid_reps == 500

    def test_undefined_rows_are_none(self, frequency: _Frequency) -> None:
        series = {
            "active": frequency.series(np.random.default_rng(16).normal(0.0, 0.01, 60)),
            "flat": frequency.series(np.zeros(60)),
        }

        table = interval_table(series, frequency.sharpe, reps=100, seed=0)

        assert table is not None
        assert table["active"] is not None
        assert table["flat"] is None

    def test_insufficient_overlap_returns_none(self, frequency: _Frequency) -> None:
        series = {
            "a": frequency.series(np.random.default_rng(17).normal(0.0, 0.01, 10))
        }

        assert interval_table(series, frequency.sharpe, reps=50) is None


def _only_on_originals(
    *originals: npt.NDArray[np.float64],
) -> Callable[[npt.NDArray[np.float64]], float | None]:
    """A statistic defined on the given series and undefined on any resample."""

    def statistic(values: npt.NDArray[np.float64]) -> float | None:
        if any(np.array_equal(values, original) for original in originals):
            return float(np.mean(values))
        return None

    return statistic


class TestUnavailableReasons:
    def test_short_overlap_reports_counts_and_the_point_estimate(
        self, frequency: _Frequency
    ) -> None:
        rng = np.random.default_rng(18)
        a = frequency.series(rng.normal(0.001, 0.01, 30))
        b = frequency.series(rng.normal(0.0, 0.01, 30), offset=20)

        result = difference_interval_outcome(a, b, frequency.sharpe, reps=50)

        assert isinstance(result, IntervalUnavailable)
        assert result.reason == "insufficient_observations"
        assert result.observations == 10
        assert result.min_observations == MIN_BOOTSTRAP_OBSERVATIONS
        assert result.reps == 0
        common = align_returns({"a": a, "b": b}).frame
        expected = frequency.sharpe(common["a"].to_numpy())
        other = frequency.sharpe(common["b"].to_numpy())
        assert expected is not None
        assert other is not None
        assert result.estimate == pytest.approx(expected - other)
        assert "10 observations" in result.message
        assert f"the {MIN_BOOTSTRAP_OBSERVATIONS} needed" in result.message

    def test_empty_overlap_never_calls_the_statistic(
        self, frequency: _Frequency
    ) -> None:
        a = frequency.series(np.full(10, 0.01))
        b = frequency.series(np.full(10, 0.02), offset=10)

        def statistic(values: npt.NDArray[np.float64]) -> float:
            raise AssertionError("statistic called on an empty overlap")

        result = difference_interval_outcome(a, b, statistic, reps=10)
        table = interval_table_outcome({"a": a, "b": b}, statistic, reps=10)

        assert isinstance(result, IntervalUnavailable)
        assert result.reason == "insufficient_observations"
        assert (result.observations, result.estimate) == (0, None)
        for row in table.values():
            assert isinstance(row, IntervalUnavailable)
            assert (row.reason, row.observations, row.estimate) == (
                "insufficient_observations",
                0,
                None,
            )

    def test_short_overlap_wins_over_an_undefined_statistic(
        self, frequency: _Frequency
    ) -> None:
        a = frequency.series(np.random.default_rng(24).normal(0.0, 0.01, 10))
        flat = frequency.series(np.zeros(10))

        result = difference_interval_outcome(a, flat, frequency.sharpe, reps=10)

        assert isinstance(result, IntervalUnavailable)
        assert result.reason == "insufficient_observations"
        assert result.estimate is None

    def test_undefined_statistic_is_named(self, frequency: _Frequency) -> None:
        a = frequency.series(np.random.default_rng(19).normal(0.0, 0.01, 60))
        flat = frequency.series(np.zeros(60))

        result = difference_interval_outcome(a, flat, frequency.sharpe, reps=50)
        table = interval_table_outcome(
            {"a": a, "flat": flat}, frequency.sharpe, reps=50
        )

        assert isinstance(result, IntervalUnavailable)
        assert result.reason == "undefined_statistic"
        assert result.estimate is None
        assert result.observations == 60
        assert result.reps == 0
        assert result.message == "the statistic is undefined on these returns"
        row = table["flat"]
        assert isinstance(row, IntervalUnavailable)
        assert row.reason == "undefined_statistic"
        assert row.estimate is None
        assert (row.observations, row.reps) == (60, 0)

    def test_no_valid_resample_is_named(self, frequency: _Frequency) -> None:
        rng = np.random.default_rng(20)
        values_a = rng.normal(0.0, 0.01, 40)
        values_b = rng.normal(0.0, 0.01, 40)
        a, b = frequency.series(values_a), frequency.series(values_b)
        statistic = _only_on_originals(values_a, values_b)

        paired = difference_interval_outcome(a, b, statistic, reps=30, seed=0)
        table = interval_table_outcome({"a": a}, statistic, reps=30, seed=0)

        assert isinstance(paired, IntervalUnavailable)
        assert paired.reason == "no_valid_resample"
        assert paired.estimate == pytest.approx(
            float(np.mean(values_a)) - float(np.mean(values_b))
        )
        assert paired.reps == 30
        row = table["a"]
        assert isinstance(row, IntervalUnavailable)
        assert row.reason == "no_valid_resample"
        assert row.estimate == pytest.approx(float(np.mean(values_a)))
        assert "30 resamples" in row.message
        assert difference_interval(a, b, statistic, reps=30, seed=0) is None
        assert interval_table({"a": a}, statistic, reps=30, seed=0) == {"a": None}

    def test_wrappers_keep_returning_none(self, frequency: _Frequency) -> None:
        series = {
            "a": frequency.series(np.random.default_rng(21).normal(0.0, 0.01, 10))
        }

        assert interval_table(series, frequency.sharpe, reps=50) is None
        outcome = interval_table_outcome(series, frequency.sharpe, reps=50)
        row = outcome["a"]
        assert isinstance(row, IntervalUnavailable)
        assert row.reason == "insufficient_observations"
        assert row.estimate == pytest.approx(frequency.sharpe(series["a"].to_numpy()))

    def test_wrappers_do_not_evaluate_the_statistic_on_a_short_overlap(
        self, frequency: _Frequency
    ) -> None:
        """Before the outcome functions, a short overlap returned ``None``
        without calling the statistic; a statistic that rejects short
        arrays must still get ``None``, not an exception."""
        rng = np.random.default_rng(25)
        a = frequency.series(rng.normal(0.0, 0.01, 10))
        b = frequency.series(rng.normal(0.0, 0.01, 10))

        def strict(values: npt.NDArray[np.float64]) -> float:
            if values.size < MIN_BOOTSTRAP_OBSERVATIONS:
                raise ValueError("too short for this statistic")
            return float(np.mean(values))

        assert difference_interval(a, b, strict, reps=10) is None
        assert interval_table({"a": a, "b": b}, strict, reps=10) is None

    def test_wrappers_validate_arguments_before_a_short_overlap(
        self, frequency: _Frequency
    ) -> None:
        a = frequency.series(np.random.default_rng(26).normal(0.0, 0.01, 10))

        with pytest.raises(ValueError, match="reps"):
            difference_interval(a, a, frequency.sharpe, reps=0)
        with pytest.raises(ValueError, match="confidence"):
            interval_table({"a": a}, frequency.sharpe, confidence=1.0)


def _monthly_strategy_returns(months: int, seed: int) -> dict[str, pd.Series]:
    """Monthly returns of two correlated strategies from hourly equity curves."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2024-01-01", periods=months * 24 * 31, freq="h", tz="UTC")
    hours = hours[
        hours < pd.Timestamp("2024-01-01", tz="UTC") + pd.DateOffset(months=months)
    ]
    common = rng.normal(0.0, 0.002, len(hours))
    curves = {
        "fast": 10_000 * np.cumprod(1.0 + common + rng.normal(5e-5, 0.001, len(hours))),
        "slow": 10_000 * np.cumprod(1.0 + common + rng.normal(0.0, 0.001, len(hours))),
    }
    return {
        name: period_returns(curve, hours, "monthly") for name, curve in curves.items()
    }


class TestMonthlyBootstrapBoundary:
    """Monthly returns reach ``MIN_BOOTSTRAP_OBSERVATIONS`` only after 20 months.

    One year of data has 12 monthly returns, so a monthly comparison over a
    year never has an interval. This is the documented behaviour, not a
    failure: the point estimates and the reason are still returned.
    """

    def test_one_year_has_estimates_and_a_reason_but_no_interval(self) -> None:
        returns = _monthly_strategy_returns(12, seed=22)
        fast, slow = returns["fast"], returns["slow"]
        assert len(fast) == 12

        assert difference_interval(fast, slow, _MONTHLY.sharpe, reps=200) is None
        assert interval_table(returns, _MONTHLY.sharpe, reps=200) is None

        paired = difference_interval_outcome(fast, slow, _MONTHLY.sharpe, reps=200)
        assert isinstance(paired, IntervalUnavailable)
        assert paired.reason == "insufficient_observations"
        assert (paired.observations, paired.min_observations) == (12, 20)
        fast_sharpe = _MONTHLY.sharpe(fast.to_numpy())
        slow_sharpe = _MONTHLY.sharpe(slow.to_numpy())
        assert fast_sharpe is not None
        assert slow_sharpe is not None
        assert paired.estimate == pytest.approx(fast_sharpe - slow_sharpe)

        table = interval_table_outcome(returns, _MONTHLY.sharpe, reps=200)
        for name, row in table.items():
            assert isinstance(row, IntervalUnavailable)
            assert row.reason == "insufficient_observations"
            assert row.estimate == pytest.approx(
                _MONTHLY.sharpe(returns[name].to_numpy())
            )

    @pytest.mark.parametrize(
        ("months", "has_interval"),
        [(MIN_BOOTSTRAP_OBSERVATIONS - 1, False), (MIN_BOOTSTRAP_OBSERVATIONS, True)],
        ids=["19-months", "20-months"],
    )
    def test_interval_appears_at_twenty_months(
        self, months: int, has_interval: bool
    ) -> None:
        returns = _monthly_strategy_returns(months, seed=23)
        assert len(returns["fast"]) == months

        paired = difference_interval_outcome(
            returns["fast"], returns["slow"], _MONTHLY.sharpe, reps=200, seed=0
        )
        table = interval_table_outcome(returns, _MONTHLY.sharpe, reps=200, seed=0)

        assert isinstance(paired, IntervalUnavailable) is not has_interval
        assert all(
            isinstance(row, IntervalUnavailable) is not has_interval
            for row in table.values()
        )
