"""Tests for the Sharpe ratio, its bootstrap interval, PSR, and DSR."""

import itertools
import math
from statistics import NormalDist

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from arch.bootstrap import IIDBootstrap

from tradingdev.domain.statistics.periods import period_returns
from tradingdev.domain.statistics.sharpe import (
    MIN_BOOTSTRAP_OBSERVATIONS,
    IntervalUnavailable,
    SharpeInterval,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_confidence_interval,
    sharpe_interval_outcome,
    sharpe_ratio,
)

_DAILY = 365.0
_MONTHLY = 12.0


def _ar1(n: int, phi: float, seed: int) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, 0.01, n)
    values = np.empty(n)
    values[0] = shocks[0]
    for t in range(1, n):
        values[t] = phi * values[t - 1] + shocks[t]
    return values + 0.0005


def _annual_sharpe(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([np.mean(values) / np.std(values, ddof=1) * math.sqrt(_DAILY)])


class TestSharpeRatio:
    def test_known_value(self) -> None:
        returns = [0.01, -0.01, 0.02, 0.0]
        expected = np.mean(returns) / np.std(returns, ddof=1) * math.sqrt(_DAILY)

        assert sharpe_ratio(returns, periods_per_year=_DAILY) == pytest.approx(expected)

    def test_accepts_numpy_arrays(self) -> None:
        returns = np.array([0.01, -0.01, 0.02, 0.0])

        assert sharpe_ratio(returns, periods_per_year=252.0) == pytest.approx(
            sharpe_ratio(list(returns), periods_per_year=252.0)
        )


class TestFrequencyDependence:
    def test_same_curve_gives_different_annualised_sharpe_per_frequency(
        self,
    ) -> None:
        """Daily and monthly Sharpe of one equity curve differ, by design.

        This is expected behaviour, not a bug. Annualising with
        ``sqrt(periods_per_year)`` assumes independent periods, but these
        daily returns are positively autocorrelated (AR(1), phi = 0.5), so a
        month's return varies more than 30 independent days would and the
        monthly figure comes out systematically lower, about
        ``sqrt((1 - phi) / (1 + phi)) = 0.58`` times the daily one in theory.
        Monthly returns are also compounded rather than summed. Thirty years
        keep the 360 monthly returns from drowning the effect in sampling
        noise: across 40 seeds the ratio stayed between 0.63 and 0.79, so
        the assertion does not hinge on this seed. The test exists so that nobody
        "fixes" the two numbers into agreement: report the frequency with
        every Sharpe ratio and compare Sharpes only at the same frequency.
        """
        daily_returns = _ar1(30 * 365, 0.5, seed=3) + 0.0003
        equity = 10_000.0 * np.cumprod(1.0 + daily_returns)
        days = pd.date_range("1990-01-01", periods=len(equity), freq="D", tz="UTC")

        daily = sharpe_ratio(
            period_returns(equity, days, "daily"), periods_per_year=_DAILY
        )
        monthly = sharpe_ratio(
            period_returns(equity, days, "monthly"), periods_per_year=_MONTHLY
        )

        assert daily is not None
        assert monthly is not None
        assert monthly != pytest.approx(daily, rel=0.1)
        assert 0.0 < monthly < 0.85 * daily


class TestProbabilisticSharpeRatio:
    def test_benchmark_equal_to_estimate_gives_one_half(self) -> None:
        returns = np.random.default_rng(1).normal(0.001, 0.01, 500)
        per_period = float(np.mean(returns) / np.std(returns, ddof=1))

        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=per_period)
        assert psr == pytest.approx(0.5)

    def test_matches_normal_closed_form_on_iid_normal_returns(self) -> None:
        returns = np.random.default_rng(2).normal(0.0005, 0.01, 2_000)
        sr = float(np.mean(returns) / np.std(returns, ddof=1))
        expected = NormalDist().cdf(
            sr * math.sqrt(len(returns) - 1) / math.sqrt(1.0 + sr**2 / 2.0)
        )

        psr = probabilistic_sharpe_ratio(returns)
        assert psr == pytest.approx(expected, abs=0.02)

    def test_strong_edges_are_near_certain(self) -> None:
        rng = np.random.default_rng(3)
        winning = rng.normal(0.002, 0.01, 1_000)

        assert probabilistic_sharpe_ratio(winning) == pytest.approx(1.0, abs=1e-3)
        assert probabilistic_sharpe_ratio(-winning) == pytest.approx(0.0, abs=1e-3)

    def test_is_calibrated_under_a_zero_true_sharpe(self) -> None:
        rng = np.random.default_rng(4)
        psrs = [
            probabilistic_sharpe_ratio(rng.normal(0.0, 0.01, 250)) for _ in range(800)
        ]
        values = np.array([p for p in psrs if p is not None])

        assert values.size == 800
        assert np.mean(values > 0.95) == pytest.approx(0.05, abs=0.025)
        assert np.mean(values) == pytest.approx(0.5, abs=0.05)

    def test_decreases_with_benchmark(self) -> None:
        returns = np.random.default_rng(5).normal(0.001, 0.01, 500)
        psrs = [
            probabilistic_sharpe_ratio(returns, benchmark_sr=b)
            for b in (-0.05, 0.0, 0.05, 0.1)
        ]

        assert all(p is not None for p in psrs)
        assert psrs == sorted(psrs, reverse=True)  # type: ignore[type-var]

    def test_negative_skew_lowers_probability(self) -> None:
        noise = np.random.default_rng(6).exponential(0.01, 1_000)
        noise -= noise.mean()
        right_skewed = 0.001 + noise
        left_skewed = 0.001 - noise

        right = probabilistic_sharpe_ratio(right_skewed)
        left = probabilistic_sharpe_ratio(left_skewed)
        assert right is not None
        assert left is not None
        assert left < right


class TestConfidenceInterval:
    def test_autocorrelated_returns_widen_the_interval(self) -> None:
        returns = _ar1(1_000, 0.6, seed=7)
        interval = sharpe_confidence_interval(returns, periods_per_year=_DAILY, seed=11)
        assert interval is not None

        iid = IIDBootstrap(returns, seed=np.random.default_rng(11))
        draws = iid.apply(_annual_sharpe, 1_000)[:, 0]
        iid_lower, iid_upper = np.quantile(draws, [0.025, 0.975])

        assert interval.block_size > 1
        assert interval.upper - interval.lower > 1.5 * (iid_upper - iid_lower)
        assert interval.lower < interval.sharpe < interval.upper

    def test_interval_fields(self) -> None:
        returns = np.random.default_rng(8).normal(0.001, 0.01, 400)
        interval = sharpe_confidence_interval(
            returns, periods_per_year=_DAILY, confidence=0.9, reps=200, seed=1
        )

        assert interval is not None
        assert interval.sharpe == pytest.approx(
            sharpe_ratio(returns, periods_per_year=_DAILY)
        )
        assert interval.confidence == pytest.approx(0.9)
        assert interval.reps == 200
        assert interval.valid_reps == 200
        assert interval.block_size >= 1
        assert interval.lower < interval.upper

    def test_seed_makes_the_interval_reproducible(self) -> None:
        returns = _ar1(300, 0.3, seed=9)
        first = sharpe_confidence_interval(returns, periods_per_year=_DAILY, seed=5)
        second = sharpe_confidence_interval(returns, periods_per_year=_DAILY, seed=5)

        assert first == second


class TestDegenerateReturns:
    @pytest.mark.parametrize("returns", [[], [0.01]], ids=["empty", "single"])
    def test_too_few_returns(self, returns: list[float]) -> None:
        assert sharpe_ratio(returns, periods_per_year=_DAILY) is None
        assert probabilistic_sharpe_ratio(returns) is None
        assert sharpe_confidence_interval(returns, periods_per_year=_DAILY) is None

    def test_too_few_returns_to_bootstrap(self) -> None:
        returns = np.random.default_rng(10).normal(
            0.001, 0.01, MIN_BOOTSTRAP_OBSERVATIONS - 1
        )

        assert sharpe_ratio(returns, periods_per_year=_DAILY) is not None
        assert probabilistic_sharpe_ratio(returns) is not None
        assert sharpe_confidence_interval(returns, periods_per_year=_DAILY) is None

    @pytest.mark.parametrize("value", [0.0, 0.1, -0.003], ids=["zero", "pos", "neg"])
    def test_constant_returns(self, value: float) -> None:
        returns = [value] * 100

        assert sharpe_ratio(returns, periods_per_year=_DAILY) is None
        assert probabilistic_sharpe_ratio(returns) is None
        assert sharpe_confidence_interval(returns, periods_per_year=_DAILY) is None


class TestIntervalOutcome:
    def test_short_monthly_sample_reports_counts_and_the_estimate(self) -> None:
        returns = np.random.default_rng(12).normal(0.01, 0.03, 12)

        outcome = sharpe_interval_outcome(returns, periods_per_year=_MONTHLY)

        assert isinstance(outcome, IntervalUnavailable)
        assert outcome.reason == "insufficient_observations"
        assert outcome.observations == 12
        assert outcome.min_observations == MIN_BOOTSTRAP_OBSERVATIONS
        assert outcome.reps == 0
        assert outcome.estimate == pytest.approx(
            sharpe_ratio(returns, periods_per_year=_MONTHLY)
        )
        assert "12 observations" in outcome.message
        assert f"{MIN_BOOTSTRAP_OBSERVATIONS} needed" in outcome.message
        assert sharpe_confidence_interval(returns, periods_per_year=_MONTHLY) is None

    def test_constant_returns_are_an_undefined_statistic(self) -> None:
        outcome = sharpe_interval_outcome([0.01] * 100, periods_per_year=_DAILY)

        assert isinstance(outcome, IntervalUnavailable)
        assert outcome.reason == "undefined_statistic"
        assert outcome.estimate is None
        assert outcome.observations == 100

    def test_short_sample_wins_over_an_undefined_statistic(self) -> None:
        outcome = sharpe_interval_outcome([0.01], periods_per_year=_DAILY)

        assert isinstance(outcome, IntervalUnavailable)
        assert outcome.reason == "insufficient_observations"
        assert outcome.estimate is None

    def test_all_constant_resamples_are_no_valid_resample(self) -> None:
        # One non-zero return among zeros: with this seed, all three
        # resamples miss it, so every resampled Sharpe is undefined.
        returns = [0.0] * (MIN_BOOTSTRAP_OBSERVATIONS - 1) + [0.01]

        outcome = sharpe_interval_outcome(
            returns, periods_per_year=_DAILY, reps=3, seed=24
        )

        assert isinstance(outcome, IntervalUnavailable)
        assert outcome.reason == "no_valid_resample"
        assert outcome.reps == 3
        assert outcome.estimate == pytest.approx(
            sharpe_ratio(returns, periods_per_year=_DAILY)
        )
        assert (
            sharpe_confidence_interval(
                returns, periods_per_year=_DAILY, reps=3, seed=24
            )
            is None
        )

    def test_twenty_months_give_an_interval(self) -> None:
        returns = np.random.default_rng(13).normal(
            0.01, 0.03, MIN_BOOTSTRAP_OBSERVATIONS
        )

        outcome = sharpe_interval_outcome(
            returns, periods_per_year=_MONTHLY, reps=200, seed=0
        )

        assert isinstance(outcome, SharpeInterval)
        assert outcome == sharpe_confidence_interval(
            returns, periods_per_year=_MONTHLY, reps=200, seed=0
        )


class TestValidation:
    @pytest.mark.parametrize("periods_per_year", [0.0, -1.0, math.inf, math.nan])
    def test_rejects_invalid_periods_per_year(self, periods_per_year: float) -> None:
        with pytest.raises(ValueError, match="periods_per_year"):
            sharpe_ratio([0.01, 0.02], periods_per_year=periods_per_year)

    def test_rejects_non_finite_returns(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            probabilistic_sharpe_ratio([0.01, math.nan, 0.02])

    def test_rejects_non_finite_benchmark(self) -> None:
        with pytest.raises(ValueError, match="benchmark_sr"):
            probabilistic_sharpe_ratio([0.01, 0.02], benchmark_sr=math.inf)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"confidence": 0.0}, "confidence"),
            ({"confidence": 1.0}, "confidence"),
            ({"reps": 0}, "reps"),
            ({"reps": True}, "reps"),
        ],
    )
    def test_rejects_invalid_bootstrap_settings(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            sharpe_confidence_interval(
                [0.01, 0.02],
                periods_per_year=_DAILY,
                **kwargs,  # type: ignore[arg-type]
            )


def _selected_returns() -> npt.NDArray[np.float64]:
    return np.random.default_rng(7).normal(0.001, 0.01, 365)


class TestDeflatedSharpeRatio:
    def test_single_trial_equals_psr_against_zero(self) -> None:
        returns = _selected_returns()

        result = deflated_sharpe_ratio(returns, [1.8], periods_per_year=_DAILY)

        assert result is not None
        assert result.n_trials == 1
        assert result.hurdle_annualised == 0.0
        assert result.trial_sharpe_std_annualised is None
        assert result.dsr == pytest.approx(probabilistic_sharpe_ratio(returns))
        assert result.dsr == pytest.approx(result.psr_vs_zero)

    def test_equal_trial_sharpes_equal_psr_against_zero(self) -> None:
        returns = _selected_returns()

        result = deflated_sharpe_ratio(returns, [1.2] * 5, periods_per_year=_DAILY)

        assert result is not None
        assert result.n_trials == 5
        assert result.trial_sharpe_std_annualised == 0.0
        assert result.hurdle_annualised == 0.0
        assert result.dsr == pytest.approx(probabilistic_sharpe_ratio(returns))

    def test_hurdle_grows_with_the_number_of_trials(self) -> None:
        hurdles = []
        for n_trials in (2, 5, 10, 50, 200):
            result = deflated_sharpe_ratio(
                _selected_returns(),
                [0.5, 1.5],
                periods_per_year=_DAILY,
                n_trials=n_trials,
            )
            assert result is not None
            assert result.n_trials == n_trials
            hurdles.append(result.hurdle_annualised)

        assert all(a < b for a, b in itertools.pairwise(hurdles))

    def test_hurdle_grows_with_trial_sharpe_dispersion(self) -> None:
        hurdles = []
        for width in (0.25, 0.5, 1.0, 2.0):
            result = deflated_sharpe_ratio(
                _selected_returns(),
                [1.0 - width, 1.0, 1.0 + width],
                periods_per_year=_DAILY,
            )
            assert result is not None
            hurdles.append(result.hurdle_annualised)

        assert all(a < b for a, b in itertools.pairwise(hurdles))

    @pytest.mark.parametrize(
        ("trial_sharpes", "n_trials"),
        [([1.0], None), ([0.5, 1.5], None), ([-1.0, 0.0, 2.0], 100), ([3.0, 3.0], 2)],
    )
    def test_dsr_never_exceeds_psr_against_zero(
        self, trial_sharpes: list[float], n_trials: int | None
    ) -> None:
        result = deflated_sharpe_ratio(
            _selected_returns(),
            trial_sharpes,
            periods_per_year=_DAILY,
            n_trials=n_trials,
        )

        assert result is not None
        assert result.hurdle_annualised >= 0.0
        assert result.dsr <= result.psr_vs_zero

    def test_annualised_trial_sharpes_are_converted_per_period(self) -> None:
        """Regression test for feeding annualised Sharpes as per-period ones.

        The same per-period trial Sharpes expressed at 365 and at 252
        periods per year must give the same per-period hurdle, so the DSR
        is unchanged and the annualised hurdles differ by sqrt(252 / 365).
        """
        returns = _selected_returns()
        daily = [0.4, 1.1, 1.9, 2.6]
        trading = [value * math.sqrt(252) / math.sqrt(365) for value in daily]

        at_365 = deflated_sharpe_ratio(returns, daily, periods_per_year=365)
        at_252 = deflated_sharpe_ratio(returns, trading, periods_per_year=252)

        assert at_365 is not None
        assert at_252 is not None
        assert at_252.hurdle_annualised / at_365.hurdle_annualised == pytest.approx(
            math.sqrt(252 / 365)
        )
        assert 0.01 < at_365.dsr < 0.99
        assert at_252.dsr == pytest.approx(at_365.dsr)

    def test_hurdle_matches_hand_calculation(self) -> None:
        result = deflated_sharpe_ratio(
            _selected_returns(),
            [-6.00, -6.67, -7.57],
            periods_per_year=365,
            n_trials=3,
        )

        assert result is not None
        assert 0.6 <= result.hurdle_annualised <= 0.75
        normal = NormalDist()
        gamma = 0.5772156649015329
        expected_max = (1 - gamma) * normal.inv_cdf(2 / 3) + gamma * normal.inv_cdf(
            1 - 1 / (3 * math.e)
        )
        std = float(np.std([-6.00, -6.67, -7.57], ddof=1))
        assert result.trial_sharpe_std_annualised == pytest.approx(std)
        assert result.hurdle_annualised == pytest.approx(std * expected_max)
        per_period_hurdle = std * expected_max / math.sqrt(365)
        assert result.dsr == pytest.approx(
            probabilistic_sharpe_ratio(
                _selected_returns(), benchmark_sr=per_period_hurdle
            )
        )

    def test_accepts_numpy_integer_trial_count(self) -> None:
        result = deflated_sharpe_ratio(
            _selected_returns(),
            [0.5, 1.5],
            periods_per_year=_DAILY,
            n_trials=np.int64(5),  # type: ignore[arg-type]
        )

        assert result is not None
        assert result.n_trials == 5

    def test_huge_trial_count_keeps_a_finite_hurdle(self) -> None:
        result = deflated_sharpe_ratio(
            _selected_returns(),
            [0.5, 1.5],
            periods_per_year=_DAILY,
            n_trials=10**18,
        )

        assert result is not None
        assert math.isfinite(result.hurdle_annualised)
        assert result.hurdle_annualised > 0.0

    def test_trial_count_near_float_limit_keeps_a_finite_hurdle(self) -> None:
        # N * e would overflow to infinity here; 1/N divided by e does not.
        result = deflated_sharpe_ratio(
            _selected_returns(),
            [0.5, 1.5],
            periods_per_year=_DAILY,
            n_trials=10**308,
        )

        assert result is not None
        assert math.isfinite(result.hurdle_annualised)

    def test_trial_count_beyond_float_range_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="n_trials is too large"):
            deflated_sharpe_ratio(
                _selected_returns(),
                [0.5, 1.5],
                periods_per_year=_DAILY,
                n_trials=10**400,
            )

    def test_trial_count_beyond_float_range_raises_before_returning_none(
        self,
    ) -> None:
        # One trial Sharpe with N >= 2 would otherwise return None first.
        with pytest.raises(ValueError, match="n_trials is too large"):
            deflated_sharpe_ratio(
                _selected_returns(),
                [0.5],
                periods_per_year=_DAILY,
                n_trials=10**400,
            )

    def test_single_trial_count_without_trial_sharpes(self) -> None:
        returns = _selected_returns()

        result = deflated_sharpe_ratio(returns, [], periods_per_year=_DAILY, n_trials=1)

        assert result is not None
        assert result.hurdle_annualised == 0.0
        assert result.dsr == pytest.approx(probabilistic_sharpe_ratio(returns))

    def test_too_few_trial_sharpes_for_several_trials(self) -> None:
        assert (
            deflated_sharpe_ratio(
                _selected_returns(), [1.0], periods_per_year=_DAILY, n_trials=3
            )
            is None
        )

    def test_undefined_psr_returns_none(self) -> None:
        assert (
            deflated_sharpe_ratio([0.01] * 30, [0.5, 1.5], periods_per_year=_DAILY)
            is None
        )

    @pytest.mark.parametrize(
        ("returns", "trial_sharpes", "kwargs", "match"),
        [
            ([0.01, math.nan, 0.02], [1.0, 2.0], {}, "returns must be finite"),
            ([0.01, 0.02], [1.0, math.inf], {}, "trial Sharpe 1 is not finite"),
            ([0.01, 0.02], [1.0, None], {}, "trial Sharpe 1 is not numeric"),
            ([0.01, 0.02], [], {}, "empty; give n_trials"),
            ([0.01, 0.02], [1.0], {"n_trials": 0}, "n_trials"),
            ([0.01, 0.02], [1.0], {"n_trials": True}, "n_trials"),
            ([0.01, 0.02], [1.0], {"n_trials": 2.0}, "n_trials"),
            ([0.01, 0.02], [1.0, 2.0, 3.0], {"n_trials": 2}, "smaller"),
        ],
        ids=[
            "nan-return",
            "inf-trial",
            "none-trial",
            "no-trials",
            "zero-n",
            "bool-n",
            "float-n",
            "n-below-trials",
        ],
    )
    def test_rejects_invalid_inputs(
        self,
        returns: list[float],
        trial_sharpes: list[object],
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            deflated_sharpe_ratio(
                returns,
                trial_sharpes,  # type: ignore[arg-type]
                periods_per_year=_DAILY,
                **kwargs,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("periods_per_year", [0.0, -1.0, math.inf, math.nan])
    def test_rejects_invalid_periods_per_year(self, periods_per_year: float) -> None:
        with pytest.raises(ValueError, match="periods_per_year"):
            deflated_sharpe_ratio(
                [0.01, 0.02], [1.0, 2.0], periods_per_year=periods_per_year
            )
