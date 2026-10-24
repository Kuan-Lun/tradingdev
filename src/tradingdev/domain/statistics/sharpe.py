"""Sharpe ratio, its bootstrap interval, and probabilistic and deflated Sharpe.

All functions take a return series supplied by the caller, such as the
daily returns from :func:`tradingdev.domain.statistics.periods.period_returns`,
and never read runs, artifacts, or caches.

Frequency: compute these statistics on daily returns. Annualising
intraday returns with ``sqrt(periods_per_year)`` assumes independent
periods, but intraday strategy returns are autocorrelated, so an hourly
Sharpe is systematically biased against the daily one; a KD strategy
measured -8.70 annualised from hourly returns versus -7.57 from daily
returns. ``periods_per_year`` is therefore normally 365 for crypto or 252
for exchange-traded markets.

Conventions:

* The Sharpe ratio is the mean over the sample standard deviation
  (``ddof=1``) of the returns, with a zero risk-free rate, as in
  :mod:`tradingdev.domain.statistics.performance`.
* Undefined figures (fewer than two returns, zero dispersion, too few
  returns to bootstrap) are ``None`` rather than infinities or zeros.
  :func:`sharpe_interval_outcome` returns an :class:`IntervalUnavailable`
  instead of ``None``, saying why the interval is missing; monthly returns
  need ``MIN_BOOTSTRAP_OBSERVATIONS`` months, so one year never has one.
* Non-finite returns raise ``ValueError``; the caller decides how to treat
  missing periods before calling.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING, Literal

import numpy as np
from arch.bootstrap import StationaryBootstrap, optimal_block_length

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

MIN_BOOTSTRAP_OBSERVATIONS = 20
EULER_MASCHERONI = 0.5772156649015329

UnavailableReason = Literal[
    "insufficient_observations", "undefined_statistic", "no_valid_resample"
]


@dataclass(frozen=True)
class IntervalUnavailable:
    """Why a bootstrap interval is missing, with what could still be computed.

    The reasons are checked in this order, so the first that applies wins:

    * ``"insufficient_observations"``: fewer than ``min_observations``
      returns (after alignment, for comparisons), so the block length
      cannot be estimated. Monthly returns over one year always land here.
    * ``"undefined_statistic"``: the statistic of the returns themselves
      is undefined, such as the Sharpe ratio of constant returns.
    * ``"no_valid_resample"``: no resample defines the statistic.

    Attributes:
        reason: Which of the cases above applies.
        estimate: The statistic of the returns, or ``None`` when it is
            undefined; it is reported even when the sample is too short to
            bootstrap. For a paired comparison it is the difference.
        observations: Number of returns, after alignment for comparisons.
        min_observations: Number of returns the bootstrap requires.
        reps: Resamples drawn; ``0`` when the bootstrap did not run.
    """

    reason: UnavailableReason
    estimate: float | None
    observations: int
    min_observations: int
    reps: int

    @property
    def message(self) -> str:
        """One-sentence explanation for a report or tool response."""
        if self.reason == "insufficient_observations":
            return (
                f"{self.observations} observations, fewer than the "
                f"{self.min_observations} needed to bootstrap an interval"
            )
        if self.reason == "undefined_statistic":
            return "the statistic is undefined on these returns"
        return f"none of the {self.reps} resamples has a defined statistic"


@dataclass(frozen=True)
class SharpeInterval:
    """Bootstrap confidence interval of an annualised Sharpe ratio.

    Attributes:
        sharpe: Annualised Sharpe ratio of the original returns.
        lower: Lower percentile bound of the bootstrap distribution.
        upper: Upper percentile bound of the bootstrap distribution.
        confidence: Two-sided confidence level of ``[lower, upper]``.
        block_size: Mean block length of the stationary bootstrap.
        reps: Number of bootstrap resamples drawn.
        valid_reps: Resamples with a defined Sharpe ratio; resamples whose
            returns are all equal are discarded.
    """

    sharpe: float
    lower: float
    upper: float
    confidence: float
    block_size: int
    reps: int
    valid_reps: int


@dataclass(frozen=True)
class DeflatedSharpe:
    """Deflated Sharpe ratio of a strategy selected from several trials.

    Attributes:
        dsr: Probability that the true Sharpe ratio exceeds the selection
            hurdle ``SR*_0``.
        psr_vs_zero: Probabilistic Sharpe ratio against a zero benchmark,
            for comparison; ``psr_vs_zero - dsr`` is what the
            selection-bias correction costs.
        hurdle_annualised: The hurdle ``SR*_0`` annualised, the expected
            maximum Sharpe of ``n_trials`` skill-less trials.
        n_trials: Number of trials ``N`` used for the hurdle.
        trial_sharpe_std_annualised: Sample standard deviation
            (``ddof=1``) of the annualised trial Sharpe ratios; ``None``
            when ``N`` is 1 and fewer than two trial Sharpes are given.
    """

    dsr: float
    psr_vs_zero: float
    hurdle_annualised: float
    n_trials: int
    trial_sharpe_std_annualised: float | None


def sharpe_ratio(returns: Iterable[float], *, periods_per_year: float) -> float | None:
    """Annualised Sharpe ratio: mean over sample std times ``sqrt(ppy)``.

    Returns ``None`` with fewer than two returns or zero dispersion.

    Raises:
        ValueError: If ``periods_per_year`` is not positive and finite or a
            return is not a finite number.
    """
    scale = _annualisation(periods_per_year)
    values = _returns(returns)
    sharpe = _period_sharpe(values)
    return sharpe * scale if sharpe is not None else None


def sharpe_confidence_interval(
    returns: Iterable[float],
    *,
    periods_per_year: float,
    confidence: float = 0.95,
    reps: int = 1000,
    seed: int | None = None,
) -> SharpeInterval | None:
    """Stationary-bootstrap percentile interval of the annualised Sharpe.

    The same as :func:`sharpe_interval_outcome`, with every
    :class:`IntervalUnavailable` collapsed to ``None``.
    """
    outcome = sharpe_interval_outcome(
        returns,
        periods_per_year=periods_per_year,
        confidence=confidence,
        reps=reps,
        seed=seed,
    )
    return outcome if isinstance(outcome, SharpeInterval) else None


def sharpe_interval_outcome(
    returns: Iterable[float],
    *,
    periods_per_year: float,
    confidence: float = 0.95,
    reps: int = 1000,
    seed: int | None = None,
) -> SharpeInterval | IntervalUnavailable:
    """Stationary-bootstrap percentile interval of the annualised Sharpe.

    Strategy returns are autocorrelated, and an iid bootstrap that shuffles
    single periods destroys that dependence and understates the sampling
    variability. The stationary bootstrap of Politis and Romano (1994)
    resamples blocks with geometrically distributed lengths, keeping the
    short-range dependence while producing a stationary resample.

    Block size: the mean block length is the Politis and White (2004)
    automatic estimate, with the correction of Patton, Politis and White
    (2009), as computed by :func:`arch.bootstrap.optimal_block_length`,
    rounded up to a whole number of periods and at least one. The estimate
    adapts to the measured autocorrelation: nearly independent returns get
    a block length near one, persistent returns get longer blocks. It needs
    a reasonable sample, so fewer than ``MIN_BOOTSTRAP_OBSERVATIONS``
    returns give no interval. That is 20 periods at any frequency: 20 days
    of daily returns, but 20 months of monthly ones.

    Returns an :class:`IntervalUnavailable` when the sample is too short,
    the Sharpe ratio itself is undefined, or no resample has a defined
    Sharpe ratio; its ``estimate`` is the annualised Sharpe ratio whenever
    that is defined.

    Raises:
        ValueError: If ``periods_per_year`` is not positive and finite,
            ``confidence`` is not strictly between 0 and 1, ``reps`` is not
            a positive integer, or a return is not a finite number.
    """
    scale = _annualisation(periods_per_year)
    if not (isinstance(confidence, numbers.Real) and 0.0 < confidence < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1")
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
        raise ValueError("reps must be a positive integer")
    values = _returns(returns)
    sharpe = _period_sharpe(values)

    def unavailable(reason: UnavailableReason, drawn: int) -> IntervalUnavailable:
        return IntervalUnavailable(
            reason=reason,
            estimate=sharpe * scale if sharpe is not None else None,
            observations=int(values.size),
            min_observations=MIN_BOOTSTRAP_OBSERVATIONS,
            reps=drawn,
        )

    if values.size < MIN_BOOTSTRAP_OBSERVATIONS:
        return unavailable("insufficient_observations", 0)
    if sharpe is None:
        return unavailable("undefined_statistic", 0)

    estimate = float(optimal_block_length(values)["stationary"].iloc[0])
    block_size = max(1, math.ceil(estimate)) if math.isfinite(estimate) else 1
    bootstrap = StationaryBootstrap(
        block_size, values, seed=np.random.default_rng(seed)
    )
    draws = bootstrap.apply(_resample_sharpe, reps)[:, 0] * scale
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return unavailable("no_valid_resample", reps)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return SharpeInterval(
        sharpe=sharpe * scale,
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        block_size=block_size,
        reps=reps,
        valid_reps=int(draws.size),
    )


def probabilistic_sharpe_ratio(
    returns: Iterable[float], *, benchmark_sr: float = 0.0
) -> float | None:
    """Probability that the true Sharpe ratio exceeds ``benchmark_sr``.

    Bailey and Lopez de Prado (2012):

    ``PSR = Phi((SR - SR*) * sqrt(n - 1) / sqrt(1 - g3 * SR + (g4 - 1) / 4
    * SR**2))``

    where ``SR`` is the per-period (not annualised) Sharpe ratio, ``g3`` the
    skewness and ``g4`` the (non-excess) kurtosis of the returns, both
    computed from population moments. Negative skew and fat tails widen
    the Sharpe estimator's standard error and lower the probability.

    ``benchmark_sr`` is per period as well; divide an annual benchmark by
    ``sqrt(periods_per_year)``.

    Returns ``None`` with fewer than two returns, zero dispersion, or a
    non-positive variance term.

    Raises:
        ValueError: If ``benchmark_sr`` or a return is not a finite number.
    """
    benchmark = _finite(benchmark_sr, "benchmark_sr")
    values = _returns(returns)
    sharpe = _period_sharpe(values)
    if sharpe is None:
        return None
    deviations = values - values.mean()
    variance = float(np.mean(deviations**2))
    skewness = float(np.mean(deviations**3)) / variance**1.5
    kurtosis = float(np.mean(deviations**4)) / variance**2
    spread = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if not spread > 0.0:
        return None
    z = (sharpe - benchmark) * math.sqrt(values.size - 1) / math.sqrt(spread)
    return NormalDist().cdf(z)


def deflated_sharpe_ratio(
    returns: Iterable[float],
    annualised_trial_sharpes: Iterable[float],
    *,
    periods_per_year: float,
    n_trials: int | None = None,
) -> DeflatedSharpe | None:
    """Probabilistic Sharpe against the best Sharpe expected from luck alone.

    Bailey and Lopez de Prado (2014). Selecting the best of ``N`` trials
    inflates its Sharpe ratio even without any edge, so the benchmark of
    :func:`probabilistic_sharpe_ratio` is raised from 0 to the expected
    maximum Sharpe of ``N`` skill-less trials:

    ``SR*_0 = std(SR_n) * ((1 - gamma) * Z^-1(1 - 1/N)
    + gamma * Z^-1(1 - 1/(N * e)))``

    ``DSR = PSR(benchmark_sr=SR*_0)``

    where ``std(SR_n)`` is the sample standard deviation (``ddof=1``) of
    the trials' per-period Sharpe ratios, ``gamma`` the Euler-Mascheroni
    constant, and ``Z^-1`` the standard normal quantile function.

    Units: ``returns`` are the selected strategy's per-period returns, as
    for :func:`sharpe_ratio`. ``annualised_trial_sharpes`` are annualised
    Sharpe ratios, as :func:`sharpe_ratio` returns them, all computed with
    the same ``periods_per_year``; they are converted to per-period values
    here, so the hurdle is never inflated by ``sqrt(periods_per_year)``.

    ``n_trials`` must be the total number of trials searched, including
    failed trials that were dropped because their metric was undefined. It
    defaults to the number of trial Sharpes, which is correct only when no
    trial was dropped. With plateau analysis, :func:`trials_from_metrics
    <tradingdev.domain.statistics.plateau.trials_from_metrics>` returns
    the dropped count for this purpose; pass
    ``n_trials = len(result.trials) + result.dropped_count``. Passing only
    the surviving count underestimates ``N`` and under-corrects the
    selection bias, without any error. With ``N = 1`` there is no
    selection, and with equal trial Sharpes there is no dispersion; both
    give ``SR*_0 = 0``, so the DSR equals the PSR against zero.

    Limitation: the formula assumes ``N`` independent trials. Grid-search
    trials are strongly correlated (a lookback of 20 and of 21 is almost
    the same strategy), so the effective number of trials is far smaller
    than the actual ``N``. Using the actual ``N`` over-corrects and leans
    towards rejecting a real edge. This version deliberately takes that
    conservative side, preferring a missed edge to a false positive.
    Estimating the effective ``N`` requires clustering the trials by the
    correlation of their returns and is out of scope.

    Two selection corrections: this project corrects for multiple
    comparisons in two ways, one per metric family.

    * This function corrects the Sharpe family: strategy return series of
      backtest or optimisation trials. Its hurdle is the expected maximum
      Sharpe of ``N`` skill-less trials, and the DSR is the probability
      that the true Sharpe clears it.
    * ``required_t_stat`` in the ML leaderboard
      (``tradingdev.domain.ml.leaderboard``)
      corrects the IC family:
      t-statistics of signal ICs. It is a two-sided Bonferroni threshold,
      ``Phi^-1(1 - alpha / (2 N))``, controlling the family-wise error
      rate, with a normal approximation the leaderboard itself flags as
      optimistic below 30 effective periods.

    They answer different questions: the DSR asks how likely the selected
    strategy's Sharpe beats what luck alone would produce in ``N`` trials;
    Bonferroni asks whether any of ``N`` IC tests would reject at
    family-wise ``alpha``. They are not interchangeable, and never both
    apply to the same number: each already accounts for the whole
    selection, so stacking them corrects one selection twice. When reports
    show both side by side, label each figure with the correction it
    carries. Both share one weakness: they count ``N`` as
    independent candidates, but neighbouring candidates are correlated, so
    ``N`` overstates the effective number of trials and both over-correct.

    Returns ``None`` when :func:`probabilistic_sharpe_ratio` is undefined
    for ``returns``, or when ``N >= 2`` but fewer than two trial Sharpes
    are given, so their dispersion cannot be estimated.

    Raises:
        ValueError: If ``periods_per_year`` is not positive and finite, a
            return or trial Sharpe is not a finite number, ``n_trials`` is
            not a positive integer or too large for a float, ``n_trials``
            is smaller than the number of trial Sharpes, or no trial
            Sharpes are given without ``n_trials``.
    """
    scale = _annualisation(periods_per_year)
    trial_sharpes = np.array(
        [
            _finite(value, f"trial Sharpe {position}")
            for position, value in enumerate(annualised_trial_sharpes)
        ],
        dtype=np.float64,
    )
    if n_trials is None:
        if trial_sharpes.size == 0:
            raise ValueError("annualised_trial_sharpes is empty; give n_trials")
        n = trial_sharpes.size
    elif (
        isinstance(n_trials, bool)
        or not isinstance(n_trials, numbers.Integral)
        or n_trials < 1
    ):
        raise ValueError("n_trials must be a positive integer")
    else:
        n = int(n_trials)
    if n < trial_sharpes.size:
        raise ValueError("n_trials must not be smaller than the trial Sharpes")
    try:
        tail = 1.0 / n
    except OverflowError:
        raise ValueError("n_trials is too large for a float") from None
    values = _returns(returns)

    std_annualised: float | None = None
    if trial_sharpes.size >= 2:
        # Equal Sharpes can still yield a rounding-error std, so test spread exactly.
        spread = np.ptp(trial_sharpes) != 0.0
        std_annualised = float(trial_sharpes.std(ddof=1)) if spread else 0.0
    if n == 1:
        hurdle = 0.0
    elif std_annualised is None:
        return None
    else:
        # Z^-1(1 - p) as -Z^-1(p) keeps precision when 1 - 1/N rounds to 1;
        # dividing 1/N by e avoids N * e overflowing to infinity.
        normal = NormalDist()
        expected_max = -(1.0 - EULER_MASCHERONI) * normal.inv_cdf(
            tail
        ) - EULER_MASCHERONI * normal.inv_cdf(tail / math.e)
        hurdle = std_annualised / scale * expected_max

    psr_vs_zero = probabilistic_sharpe_ratio(values)
    dsr = probabilistic_sharpe_ratio(values, benchmark_sr=hurdle)
    if psr_vs_zero is None or dsr is None:
        return None
    return DeflatedSharpe(
        dsr=dsr,
        psr_vs_zero=psr_vs_zero,
        hurdle_annualised=hurdle * scale,
        n_trials=n,
        trial_sharpe_std_annualised=std_annualised,
    )


def _period_sharpe(values: npt.NDArray[np.float64]) -> float | None:
    # Equal values can still yield a rounding-error std, so test spread exactly.
    if values.size < 2 or np.ptp(values) == 0.0:
        return None
    std = float(values.std(ddof=1))
    return float(values.mean()) / std


def _resample_sharpe(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    sharpe = _period_sharpe(values)
    return np.array([sharpe if sharpe is not None else math.nan])


def _returns(returns: Iterable[float]) -> npt.NDArray[np.float64]:
    values = np.asarray(list(returns), dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must be finite")
    return values


def _annualisation(periods_per_year: float) -> float:
    value = _finite(periods_per_year, "periods_per_year")
    if value <= 0:
        raise ValueError("periods_per_year must be positive")
    return math.sqrt(value)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} is not numeric")
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite")
    return float(value)
