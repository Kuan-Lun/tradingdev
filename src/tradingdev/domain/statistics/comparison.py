"""Paired bootstrap comparison of strategies evaluated on the same data.

All functions take return series supplied by the caller, such as the daily
returns from :func:`tradingdev.domain.statistics.periods.period_returns`,
and never read runs, artifacts, or caches.

Why paired: two strategies run on the same market data share its shocks, so
their returns are usually strongly correlated. The variance of a difference
is ``var(A) + var(B) - 2 cov(A, B)``, far below ``var(A) + var(B)`` when the
correlation is high. Comparing two independently bootstrapped intervals
ignores that covariance. :func:`difference_interval` therefore aligns both
series on one timeline and resamples the same block indices for both, so
each resample keeps the pairing of their returns.

Reading the results:

* Non-overlapping individual intervals (from :func:`interval_table`) imply
  a significant difference, but overlapping intervals do not imply an
  insignificant one. The individual intervals are a conservative screen;
  decide with :func:`difference_interval`.
* Compare out-of-sample returns only, such as walk-forward test folds.
  In-sample returns of tuned parameters measure which strategy overfits
  more thoroughly, not which one is better.
* Strategies run over different periods, instruments, or cost assumptions
  are not comparable. That is an experimental design problem that no
  statistic can repair; this module only guarantees the common period.
* Statistical significance is not economic significance. A tiny but
  precisely estimated edge may not survive costs or capacity, so the
  difference itself is always reported together with its interval.

Conventions:

* ``statistic`` maps a one-dimensional array of returns to a float, or
  ``None`` when undefined; for example
  ``functools.partial(sharpe_ratio, periods_per_year=365)``.
* Undefined results (insufficient overlap, undefined statistic, no valid
  resample) are ``None``.
* Non-finite returns or duplicate index labels raise ``ValueError``; the
  caller decides how to treat missing periods before calling.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length

from tradingdev.domain.statistics.sharpe import MIN_BOOTSTRAP_OBSERVATIONS

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpy.typing as npt

    Statistic = Callable[[npt.NDArray[np.float64]], float | None]


@dataclass(frozen=True)
class AlignedReturns:
    """Return series restricted to the periods they all cover.

    Attributes:
        frame: One column per series, indexed by the common periods in
            ascending order.
        overlap: Number of common periods.
        lengths: Number of observations of each series before alignment;
            ``lengths[name] - overlap`` observations of that series were
            dropped.
        min_overlap: Minimum overlap required for inference.
        sufficient: Whether ``overlap`` reaches ``min_overlap``.
    """

    frame: pd.DataFrame
    overlap: int
    lengths: Mapping[str, int]
    min_overlap: int
    sufficient: bool


@dataclass(frozen=True)
class StatisticInterval:
    """Bootstrap percentile interval of one strategy's statistic.

    Attributes:
        estimate: Statistic of the aligned returns.
        lower: Lower percentile bound of the bootstrap distribution.
        upper: Upper percentile bound of the bootstrap distribution.
        confidence: Two-sided confidence level of ``[lower, upper]``.
        block_size: Mean block length of the stationary bootstrap, shared
            by every series of the table.
        overlap: Number of aligned observations.
        reps: Number of bootstrap resamples drawn.
        valid_reps: Resamples with a defined statistic.
    """

    estimate: float
    lower: float
    upper: float
    confidence: float
    block_size: int
    overlap: int
    reps: int
    valid_reps: int


@dataclass(frozen=True)
class DifferenceInterval:
    """Paired bootstrap inference on ``statistic(a) - statistic(b)``.

    Attributes:
        difference: ``estimate_a - estimate_b`` on the aligned returns.
        lower: Lower percentile bound of the bootstrap differences.
        upper: Upper percentile bound of the bootstrap differences.
        p_value: Two-sided bootstrap p-value of a zero difference.
        estimate_a: Statistic of ``a`` on the aligned returns.
        estimate_b: Statistic of ``b`` on the aligned returns.
        confidence: Two-sided confidence level of ``[lower, upper]``.
        block_size: Mean block length of the stationary bootstrap.
        correlation: Pearson correlation of the aligned returns; ``None``
            when either series is constant.
        overlap: Number of aligned observations.
        reps: Number of bootstrap resamples drawn.
        valid_reps: Resamples where both statistics are defined.
    """

    difference: float
    lower: float
    upper: float
    p_value: float
    estimate_a: float
    estimate_b: float
    confidence: float
    block_size: int
    correlation: float | None
    overlap: int
    reps: int
    valid_reps: int


def align_returns(
    series_by_name: Mapping[str, pd.Series],
    *,
    min_overlap: int = MIN_BOOTSTRAP_OBSERVATIONS,
) -> AlignedReturns:
    """Restrict return series to the index labels they all share.

    The intersection is taken on index labels, so a period missing from any
    series is dropped from all of them. Nothing is truncated silently: the
    result carries each series' original length and whether the overlap
    reaches ``min_overlap``; inference functions return ``None`` when it
    does not.

    Raises:
        ValueError: If no series is given, ``min_overlap`` is not a
            positive integer, an index has duplicate labels, or a return is
            not a finite number.
    """
    if not series_by_name:
        raise ValueError("at least one return series is required")
    _positive_int(min_overlap, "min_overlap")
    columns: dict[str, pd.Series] = {}
    for name, series in series_by_name.items():
        if not series.index.is_unique:
            raise ValueError(f"returns {name!r} have duplicate index labels")
        values = pd.to_numeric(series, errors="coerce").astype(np.float64)
        if not np.all(np.isfinite(values.to_numpy())):
            raise ValueError(f"returns {name!r} must be finite")
        columns[name] = values
    frame = pd.concat(columns, axis=1, join="inner").sort_index()
    overlap = len(frame)
    return AlignedReturns(
        frame=frame,
        overlap=overlap,
        lengths={name: len(series) for name, series in columns.items()},
        min_overlap=min_overlap,
        sufficient=overlap >= min_overlap,
    )


def difference_interval(
    a: pd.Series,
    b: pd.Series,
    statistic: Statistic,
    *,
    reps: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> DifferenceInterval | None:
    """Paired stationary-bootstrap interval of ``statistic(a) - statistic(b)``.

    Both series are aligned with :func:`align_returns`, then resampled with
    one :class:`arch.bootstrap.StationaryBootstrap` so every resample draws
    the same period indices for ``a`` and ``b``. This keeps their
    correlation, which is what makes the difference interval narrower than
    the individual intervals suggest. This is the formal test; see the
    module docstring for why overlapping individual intervals are not.

    Block size: the Politis and White (2004) automatic stationary block
    length with the Patton, Politis and White (2009) correction, computed
    for each series by :func:`arch.bootstrap.optimal_block_length`; the
    larger estimate is used, rounded up to a whole period and at least one,
    so neither series' dependence is cut short.

    p-value: twice the smaller tail share of the bootstrap differences on
    either side of zero, capped at one. It is the smallest ``1 -
    confidence`` at which the percentile interval would exclude zero, so
    the interval and the p-value always agree.

    Returns ``None`` when the overlap is below ``MIN_BOOTSTRAP_OBSERVATIONS``,
    either statistic is undefined on the aligned returns, or no resample
    defines both statistics.

    Raises:
        ValueError: If ``reps`` is not a positive integer, ``confidence``
            is not strictly between 0 and 1, an index has duplicate labels,
            or a return is not a finite number.
    """
    _positive_int(reps, "reps")
    tail = _tail(confidence)
    aligned = align_returns({"a": a, "b": b})
    if not aligned.sufficient:
        return None
    values_a = aligned.frame["a"].to_numpy()
    values_b = aligned.frame["b"].to_numpy()
    estimate_a = _evaluate(statistic, values_a)
    estimate_b = _evaluate(statistic, values_b)
    if not (math.isfinite(estimate_a) and math.isfinite(estimate_b)):
        return None

    def resample(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.array([_evaluate(statistic, x) - _evaluate(statistic, y)])

    block_size = _block_size(aligned.frame)
    bootstrap = StationaryBootstrap(
        block_size, values_a, values_b, seed=np.random.default_rng(seed)
    )
    draws = bootstrap.apply(resample, reps)[:, 0]
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return None
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    smaller_tail = min(float(np.mean(draws <= 0.0)), float(np.mean(draws >= 0.0)))
    return DifferenceInterval(
        difference=estimate_a - estimate_b,
        lower=float(lower),
        upper=float(upper),
        p_value=min(1.0, 2.0 * smaller_tail),
        estimate_a=estimate_a,
        estimate_b=estimate_b,
        confidence=float(confidence),
        block_size=block_size,
        correlation=_correlation(values_a, values_b),
        overlap=aligned.overlap,
        reps=reps,
        valid_reps=int(draws.size),
    )


def interval_table(
    series_by_name: Mapping[str, pd.Series],
    statistic: Statistic,
    *,
    reps: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict[str, StatisticInterval | None] | None:
    """Individual stationary-bootstrap intervals for a forest plot.

    The series are aligned with :func:`align_returns` first, so every row
    covers the same periods, and resampled jointly with one block length
    (the largest per-series estimate, as in :func:`difference_interval`).

    These intervals answer "how precise is each estimate", not "do two
    strategies differ". Non-overlapping rows imply a significant difference;
    overlapping rows do not imply an insignificant one, because the
    intervals ignore the correlation between strategies. Use
    :func:`difference_interval` for the comparison itself.

    Returns ``None`` when the overlap is below ``MIN_BOOTSTRAP_OBSERVATIONS``.
    Otherwise the result maps each name, in input order, to its interval or
    to ``None`` when its statistic is undefined on the aligned returns or
    on every resample.

    Raises:
        ValueError: If no series is given, ``reps`` is not a positive
            integer, ``confidence`` is not strictly between 0 and 1, an
            index has duplicate labels, or a return is not a finite number.
    """
    _positive_int(reps, "reps")
    tail = _tail(confidence)
    aligned = align_returns(series_by_name)
    if not aligned.sufficient:
        return None
    columns = [aligned.frame[name].to_numpy() for name in aligned.frame.columns]

    def resample(*values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([_evaluate(statistic, column) for column in values])

    block_size = _block_size(aligned.frame)
    bootstrap = StationaryBootstrap(
        block_size, *columns, seed=np.random.default_rng(seed)
    )
    draws = bootstrap.apply(resample, reps)

    table: dict[str, StatisticInterval | None] = {}
    for position, name in enumerate(aligned.frame.columns):
        estimate = _evaluate(statistic, columns[position])
        valid = draws[:, position][np.isfinite(draws[:, position])]
        if not math.isfinite(estimate) or valid.size == 0:
            table[name] = None
            continue
        lower, upper = np.quantile(valid, [tail, 1.0 - tail])
        table[name] = StatisticInterval(
            estimate=estimate,
            lower=float(lower),
            upper=float(upper),
            confidence=float(confidence),
            block_size=block_size,
            overlap=aligned.overlap,
            reps=reps,
            valid_reps=int(valid.size),
        )
    return table


def _evaluate(statistic: Statistic, values: npt.NDArray[np.float64]) -> float:
    result = statistic(values)
    if result is None:
        return math.nan
    return float(result)


def _block_size(frame: pd.DataFrame) -> int:
    # A constant series has no dependence to preserve and breaks the estimator.
    varying = frame.loc[:, np.ptp(frame.to_numpy(), axis=0) > 0.0]
    if varying.empty:
        return 1
    estimates = optimal_block_length(varying)["stationary"].to_numpy()
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return 1
    return max(1, math.ceil(float(finite.max())))


def _correlation(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> float | None:
    # Equal values can still yield a rounding-error std, so test spread exactly.
    if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _tail(confidence: float) -> float:
    if isinstance(confidence, bool) or not isinstance(confidence, numbers.Real):
        raise ValueError("confidence must be strictly between 0 and 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    return (1.0 - float(confidence)) / 2.0


def _positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
