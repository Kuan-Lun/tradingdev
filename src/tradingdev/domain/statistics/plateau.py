"""Parameter plateau analysis over the trials of a parameter grid search.

The question is whether the best parameter set is a real signal, sitting on
a broad region where nearby parameters also perform well, or a lucky spike
in the grid whose neighbours perform poorly. The module only inspects the
trials it is given and never reads jobs, runs, artifacts, or caches.

Inputs come from :func:`trials_from_metrics`, which applies the metric's
optimization direction and drops failed trials (see the integration contract
below). Every analysis function takes its ``trials`` and its ``axes``: the
grid is a property of what was searched, not of which trials survived, so
the axes are taken from every searched row, failed ones included. No
per-trial return series is needed by this version.

First-version design decisions (adjustable):

* Neighbourhoods use grid indices, not raw parameter distances. The distinct
  searched values of each parameter axis are sorted and replaced by their
  rank ``r_k``; ``theta'`` is a neighbour of ``theta`` when
  ``max_k |r_k(theta') - r_k(theta)| <= radius`` (Chebyshev distance) and
  ``theta' != theta``. The default radius is 1. This sidesteps parameters
  on different scales, but on a non-uniform grid adjacent grid points span
  different raw parameter distances, so one step of radius is not one fixed
  amount of parameter change.
* A point's plateau score is the mean metric of its neighbours, excluding
  the point itself, so a lone spike gets no credit from its own value.
* With more than three parameters a radius-1 neighbourhood holds up to
  ``3**d - 1`` points and stops being local; results flag this with
  ``dimension_warning``.
* The primary statistic is the metric gap between the best trial and the
  trial with the highest plateau score. Spikiness, the best metric's
  distance from its neighbourhood mean in neighbourhood standard
  deviations, is secondary: its denominator is a sample standard deviation
  estimated from a handful of neighbours and is unstable in small
  neighbourhoods.

Conventions:

* Incomplete grids are fine: neighbourhoods contain only trials that exist,
  and every result reports the actual neighbourhood size. A failed trial is
  a missing grid point that keeps its position on the axes, so a whole
  failed slice still separates its neighbours and shrinks their
  neighbourhoods instead of collapsing the axis.
* Ties for the best metric or best plateau score go to the earliest trial
  in input order. Plateau scores are means over neighbourhoods of different
  sizes, so equal metrics can average to scores a few rounding errors
  apart; scores within a relative ``1e-12`` of the highest one count as
  tied. Metrics, which are inputs, tie only when exactly equal.
* Standard deviations are sample standard deviations (``ddof=1``). A
  neighbourhood whose metrics agree within a relative ``1e-12`` has zero
  dispersion: its standard deviation is ``0.0`` rather than the rounding
  error of the mean, and the spikiness is ``None``.
* Undefined figures (no neighbours, fewer than two neighbours for a
  standard deviation, zero dispersion) are ``None`` rather than infinities
  or zeros.

Limitation: a broad plateau is not evidence against overfitting when the
whole parameter region is fitted to the same market regime. Every
neighbour then looks good for the same reason and the analysis shows a
false plateau, so plateau analysis cannot replace out-of-sample validation.

Integration contract:

* Build trials with :func:`trials_from_metrics` rather than constructing
  :class:`Trial` by hand, and pass its ``trials`` and ``axes`` together:
  ``plateau_summary(result.trials, axes=result.axes)``. It applies the
  metric direction, so callers never negate metrics themselves. Report its
  ``dropped_count`` alongside the plateau results.
* Pass every searched parameter set as a row, failed ones included, with
  ``params`` holding every searched parameter as a finite number; each
  parameter set is unique. A failed trial carries ``None``, NaN, or an
  infinite metric; it is dropped from the trials but its parameter values
  stay on the axes, so it becomes a missing grid point. Rows omitted
  upstream cannot be recovered and shrink the axes.
* Each metric value is passed together with its ``optimization_direction``
  as the ``direction`` argument (``"maximize"`` or ``"minimize"``); the
  direction is never inferred. Results are in that higher-is-better scale,
  so a minimised metric is reported negated.
* All trials come from the same evaluation window, dataset, and cost model.
* Upstream currently persists only ``trial/{index}/train``, so the
  resulting plateau is in-sample. Label it as in-sample; it is not
  out-of-sample evidence.
* Plateau scores depend on the radius and on the grid density, so they are
  comparable only within one run, never across runs or radii.
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

DEFAULT_RADIUS = 1
MAX_LOCAL_DIMENSIONS = 3
# Relative gap below which two plateau figures differ only by rounding.
_RELATIVE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Trial:
    """One evaluated parameter set and its score (higher is better)."""

    params: Mapping[str, float]
    metric: float


@dataclass(frozen=True)
class MetricTrials:
    """Trials built by :func:`trials_from_metrics`.

    Attributes:
        trials: Higher-is-better trials, in input order.
        axes: Sorted distinct searched values per parameter, taken from
            every row, failed rows included; pass it as the ``axes`` of the
            plateau functions.
        dropped_count: Number of failed rows dropped because their metric
            was ``None``, NaN, or infinite.
    """

    trials: tuple[Trial, ...]
    axes: Mapping[str, tuple[float, ...]]
    dropped_count: int


class DroppedTrialsWarning(UserWarning):
    """Rows with a non-finite metric were dropped as failed trials."""

    def __init__(self, dropped_count: int) -> None:
        super().__init__(f"dropped {dropped_count} trial(s) with a non-finite metric")
        self.dropped_count = dropped_count


def trials_from_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    params_key: str,
    metric_key: str,
    direction: Literal["maximize", "minimize"],
) -> MetricTrials:
    """Build higher-is-better trials from raw metric rows.

    ``direction`` is the metric's optimization direction and has no default;
    ``"minimize"`` metrics are negated. Rows whose metric is ``None``, NaN,
    or infinite are failed trials and are dropped from ``trials``, but their
    parameter values stay in ``axes``, which makes them missing grid points.
    A dropped row still occupies its parameter set, so a later row repeating
    it is rejected rather than treated as a retry.

    The number of dropped rows is returned as
    :attr:`MetricTrials.dropped_count`; callers must surface it, because
    dropped rows shrink the grid. A :class:`DroppedTrialsWarning` is also
    issued as a secondary signal for interactive use; under an ``"error"``
    warnings filter it is raised instead of returning.

    Raises:
        ValueError: If ``direction`` is unknown, a row lacks ``params_key``
            or ``metric_key``, ``params`` is not a non-empty mapping, rows
            have different parameter names, a parameter is not a finite
            number, a metric is neither ``None`` nor a number, or a
            parameter set repeats.
    """
    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be 'maximize' or 'minimize'")
    sign = -1.0 if direction == "minimize" else 1.0
    trials: list[Trial] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    names: frozenset[str] | None = None
    dropped_count = 0
    for position, row in enumerate(rows):
        for key in (params_key, metric_key):
            if key not in row:
                raise ValueError(f"row {position} has no {key!r}")
        raw_params = row[params_key]
        if not isinstance(raw_params, Mapping) or not raw_params:
            raise ValueError(f"row {position} params must be a non-empty mapping")
        params = {
            name: _finite(value, f"row {position} parameter {name!r}")
            for name, value in raw_params.items()
        }
        if names is None:
            names = frozenset(params)
        elif frozenset(params) != names:
            raise ValueError(f"row {position} has different parameter names")
        identity = tuple(sorted(params.items()))
        if identity in seen:
            raise ValueError(f"row {position} repeats an earlier parameter set")
        seen.add(identity)
        raw_metric = row[metric_key]
        metric = (
            math.nan
            if raw_metric is None
            else _numeric(raw_metric, f"row {position} metric")
        )
        if not math.isfinite(metric):
            dropped_count += 1
            continue
        trials.append(Trial(params=MappingProxyType(params), metric=sign * metric))
    if dropped_count:
        warnings.warn(DroppedTrialsWarning(dropped_count), stacklevel=2)
    axes = {
        name: tuple(sorted({dict(identity)[name] for identity in seen}))
        for name in sorted(names or ())
    }
    return MetricTrials(
        trials=tuple(trials),
        axes=MappingProxyType(axes),
        dropped_count=dropped_count,
    )


@dataclass(frozen=True)
class PlateauScore:
    """Neighbourhood performance of one trial.

    Attributes:
        params: The trial's parameter set.
        metric: The trial's own metric.
        score: Mean metric of the trial's neighbours; ``None`` without
            neighbours.
        neighborhood_size: Number of existing neighbours, excluding the
            trial itself.
    """

    params: Mapping[str, float]
    metric: float
    score: float | None
    neighborhood_size: int


@dataclass(frozen=True)
class PlateauSummary:
    """Comparison of the best trial with the best plateau.

    Plateau scores, and so ``plateau_best_score``, change with ``radius``
    and with the grid density in ``axis_sizes``. They are meaningful only
    within one run and must not be compared across runs or radii.

    Attributes:
        n_trials: Number of trials.
        n_dimensions: Number of searched parameters.
        axis_sizes: Number of searched values per parameter, from ``axes``,
            including values whose trials all failed.
        radius: Grid-index radius used for neighbourhoods.
        min_neighborhood: Minimum neighbourhood size a trial needs to
            compete for the plateau best; ``None`` applies no threshold.
        n_below_min_neighborhood: Number of trials excluded from the
            plateau best by ``min_neighborhood``, including trials without
            neighbours; ``0`` without a threshold.
        dimension_warning: ``True`` when there are more than three
            parameters, so the neighbourhood is too large to be local.
        best_params: Parameters of the trial with the highest metric.
        best_metric: Highest metric.
        plateau_best_params: Parameters of the trial with the highest
            plateau score among the trials meeting ``min_neighborhood``;
            ``None`` when no trial qualifies.
        plateau_best_metric: That trial's own metric.
        plateau_best_score: That trial's plateau score.
        metric_gap: ``best_metric - plateau_best_metric``, the primary
            statistic; a large gap means the best trial is not on the best
            plateau.
        grid_distance: Chebyshev grid-index distance between the best trial
            and the plateau best trial.
        best_neighborhood_mean: Mean metric of the best trial's neighbours.
        best_neighborhood_std: Sample standard deviation of those metrics.
        best_neighborhood_size: Number of the best trial's neighbours.
        spikiness: ``(best_metric - best_neighborhood_mean) /
            best_neighborhood_std``; secondary and unstable in small
            neighbourhoods.
    """

    n_trials: int
    n_dimensions: int
    axis_sizes: Mapping[str, int]
    radius: int
    min_neighborhood: int | None
    n_below_min_neighborhood: int
    dimension_warning: bool
    best_params: Mapping[str, float] | None
    best_metric: float | None
    plateau_best_params: Mapping[str, float] | None
    plateau_best_metric: float | None
    plateau_best_score: float | None
    metric_gap: float | None
    grid_distance: int | None
    best_neighborhood_mean: float | None
    best_neighborhood_std: float | None
    best_neighborhood_size: int | None
    spikiness: float | None


@dataclass(frozen=True)
class PlateauGrid:
    """A two-parameter slice of metrics and plateau scores for plotting.

    Cells are indexed ``[y_index][x_index]``; missing grid points are
    ``None``. Plateau scores use the full-dimensional neighbourhood, not
    only the neighbours inside the slice.

    Attributes:
        x: Parameter on the x axis.
        y: Parameter on the y axis.
        x_values: Sorted searched values of ``x``, from ``axes``.
        y_values: Sorted searched values of ``y``, from ``axes``.
        fixed_params: Values of the other parameters, fixed at the best
            trial's values; empty without trials.
        metric: Trial metric per cell.
        score: Plateau score per cell.
        neighborhood_size: Neighbourhood size per cell.
        radius: Grid-index radius used for neighbourhoods.
        dimension_warning: Same flag as :class:`PlateauSummary`.
    """

    x: str
    y: str
    x_values: tuple[float, ...]
    y_values: tuple[float, ...]
    fixed_params: Mapping[str, float]
    metric: tuple[tuple[float | None, ...], ...]
    score: tuple[tuple[float | None, ...], ...]
    neighborhood_size: tuple[tuple[int | None, ...], ...]
    radius: int
    dimension_warning: bool


@dataclass(frozen=True)
class _Grid:
    names: tuple[str, ...]
    axes: tuple[tuple[float, ...], ...]
    params: tuple[Mapping[str, float], ...]
    ranks: npt.NDArray[np.int64]
    metrics: npt.NDArray[np.float64]

    @property
    def dimension_warning(self) -> bool:
        return len(self.names) > MAX_LOCAL_DIMENSIONS

    @property
    def axis_sizes(self) -> Mapping[str, int]:
        return MappingProxyType(
            {name: len(axis) for name, axis in zip(self.names, self.axes, strict=True)}
        )

    def neighbours(self, index: int, radius: int) -> npt.NDArray[np.intp]:
        distances = np.abs(self.ranks - self.ranks[index]).max(axis=1)
        mask = distances <= radius
        mask[index] = False
        return np.flatnonzero(mask)

    def distance(self, first: int, second: int) -> int:
        return int(np.abs(self.ranks[first] - self.ranks[second]).max())


def plateau_scores(
    trials: Sequence[Trial],
    *,
    axes: Mapping[str, Iterable[float]],
    radius: int = DEFAULT_RADIUS,
) -> tuple[PlateauScore, ...]:
    """Compute every trial's neighbourhood mean metric, in input order.

    ``axes`` holds the searched values of every parameter, normally
    :attr:`MetricTrials.axes`; grid indices come from it, not from the
    trials, so failed trials stay missing grid points. Empty ``axes`` are
    allowed only without trials, as an empty search.

    Raises:
        ValueError: If ``radius`` is not a positive integer, ``axes`` is
            invalid, or the trials are inconsistent with each other or with
            ``axes``.
    """
    grid = _build_grid(trials, axes)
    return _scores(grid, _checked_radius(radius))


def plateau_summary(
    trials: Sequence[Trial],
    *,
    axes: Mapping[str, Iterable[float]],
    radius: int = DEFAULT_RADIUS,
    min_neighborhood: int | None = None,
) -> PlateauSummary:
    """Compare the best trial with the trial on the best plateau.

    ``axes`` is as for :func:`plateau_scores`. Trials with fewer than
    ``min_neighborhood`` neighbours keep their plateau scores but cannot
    become the plateau best, because a mean of few neighbours is noisy and
    favours false plateaus and spikes alike.

    Raises:
        ValueError: If ``radius`` or ``min_neighborhood`` is not a positive
            integer, ``axes`` is invalid, or the trials are inconsistent
            with each other or with ``axes``.
    """
    radius = _checked_radius(radius)
    if min_neighborhood is not None:
        min_neighborhood = _positive_int(min_neighborhood, "min_neighborhood")
    grid = _build_grid(trials, axes)
    scores = _scores(grid, radius)
    n_trials = len(scores)
    if n_trials == 0:
        return PlateauSummary(
            n_trials=0,
            n_dimensions=len(grid.names),
            axis_sizes=grid.axis_sizes,
            radius=radius,
            min_neighborhood=min_neighborhood,
            n_below_min_neighborhood=0,
            dimension_warning=grid.dimension_warning,
            best_params=None,
            best_metric=None,
            plateau_best_params=None,
            plateau_best_metric=None,
            plateau_best_score=None,
            metric_gap=None,
            grid_distance=None,
            best_neighborhood_mean=None,
            best_neighborhood_std=None,
            best_neighborhood_size=None,
            spikiness=None,
        )

    best = int(np.argmax(grid.metrics))
    best_metric = float(grid.metrics[best])
    eligible = [
        min_neighborhood is None or entry.neighborhood_size >= min_neighborhood
        for entry in scores
    ]
    plateau_best = _argmax_score(scores, eligible)
    neighbour_metrics = grid.metrics[grid.neighbours(best, radius)]
    mean = float(neighbour_metrics.mean()) if neighbour_metrics.size else None
    std = _sample_std(neighbour_metrics)
    spikiness = (
        (best_metric - mean) / std
        if mean is not None and std is not None and std > 0
        else None
    )
    plateau_best_metric = (
        float(grid.metrics[plateau_best]) if plateau_best is not None else None
    )
    return PlateauSummary(
        n_trials=n_trials,
        n_dimensions=len(grid.names),
        axis_sizes=grid.axis_sizes,
        radius=radius,
        min_neighborhood=min_neighborhood,
        n_below_min_neighborhood=eligible.count(False),
        dimension_warning=grid.dimension_warning,
        best_params=grid.params[best],
        best_metric=best_metric,
        plateau_best_params=(
            grid.params[plateau_best] if plateau_best is not None else None
        ),
        plateau_best_metric=plateau_best_metric,
        plateau_best_score=(
            scores[plateau_best].score if plateau_best is not None else None
        ),
        metric_gap=(
            best_metric - plateau_best_metric
            if plateau_best_metric is not None
            else None
        ),
        grid_distance=(
            grid.distance(best, plateau_best) if plateau_best is not None else None
        ),
        best_neighborhood_mean=mean,
        best_neighborhood_std=std,
        best_neighborhood_size=int(neighbour_metrics.size),
        spikiness=spikiness,
    )


def plateau_grid(
    trials: Sequence[Trial],
    *,
    axes: Mapping[str, Iterable[float]],
    x: str,
    y: str,
    radius: int = DEFAULT_RADIUS,
) -> PlateauGrid:
    """Lay out metrics and plateau scores on two parameter axes.

    ``axes`` is as for :func:`plateau_scores` and sets the cells, so a
    failed slice shows as a row or column of ``None``. Parameters other
    than ``x`` and ``y`` are fixed at the best trial's values. Empty
    ``trials`` yield a grid of ``None`` cells, or an empty grid when
    ``axes`` is empty too.

    Raises:
        ValueError: If ``x`` equals ``y``, either is not a searched
            parameter, ``radius`` is not a positive integer, ``axes`` is
            invalid, or the trials are inconsistent with each other or with
            ``axes``.
    """
    radius = _checked_radius(radius)
    if x == y:
        raise ValueError("x and y must be different parameters")
    grid = _build_grid(trials, axes)
    if not grid.names:
        return PlateauGrid(
            x=x,
            y=y,
            x_values=(),
            y_values=(),
            fixed_params=MappingProxyType({}),
            metric=(),
            score=(),
            neighborhood_size=(),
            radius=radius,
            dimension_warning=False,
        )
    for name in (x, y):
        if name not in grid.names:
            raise ValueError(f"{name!r} is not a searched parameter")

    scores = _scores(grid, radius)
    fixed: dict[str, float] = {}
    if grid.params:
        best_params = grid.params[int(np.argmax(grid.metrics))]
        fixed = {k: v for k, v in best_params.items() if k not in (x, y)}
    x_values = grid.axes[grid.names.index(x)]
    y_values = grid.axes[grid.names.index(y)]
    x_index = {value: i for i, value in enumerate(x_values)}
    y_index = {value: i for i, value in enumerate(y_values)}

    metric: list[list[float | None]] = [[None] * len(x_values) for _ in y_values]
    score: list[list[float | None]] = [[None] * len(x_values) for _ in y_values]
    sizes: list[list[int | None]] = [[None] * len(x_values) for _ in y_values]
    for entry in scores:
        if any(entry.params[k] != v for k, v in fixed.items()):
            continue
        row = y_index[entry.params[y]]
        column = x_index[entry.params[x]]
        metric[row][column] = entry.metric
        score[row][column] = entry.score
        sizes[row][column] = entry.neighborhood_size

    return PlateauGrid(
        x=x,
        y=y,
        x_values=x_values,
        y_values=y_values,
        fixed_params=MappingProxyType(fixed),
        metric=tuple(tuple(row) for row in metric),
        score=tuple(tuple(row) for row in score),
        neighborhood_size=tuple(tuple(row) for row in sizes),
        radius=radius,
        dimension_warning=grid.dimension_warning,
    )


def _scores(grid: _Grid, radius: int) -> tuple[PlateauScore, ...]:
    results = []
    for index, params in enumerate(grid.params):
        neighbour_metrics = grid.metrics[grid.neighbours(index, radius)]
        results.append(
            PlateauScore(
                params=params,
                metric=float(grid.metrics[index]),
                score=(
                    float(neighbour_metrics.mean()) if neighbour_metrics.size else None
                ),
                neighborhood_size=int(neighbour_metrics.size),
            )
        )
    return tuple(results)


def _sample_std(values: npt.NDArray[np.float64]) -> float | None:
    if values.size < 2:
        return None
    # Equal values give a rounding-error std from their inexact mean (about
    # 1e-16 for 0.7), which would turn the spikiness into a huge number.
    if _within_tolerance(float(np.ptp(values)), float(np.abs(values).max())):
        return 0.0
    return float(values.std(ddof=1))


def _argmax_score(
    scores: Sequence[PlateauScore], eligible: Sequence[bool]
) -> int | None:
    candidates = [
        (index, entry.score)
        for index, (entry, allowed) in enumerate(zip(scores, eligible, strict=True))
        if allowed and entry.score is not None
    ]
    if not candidates:
        return None
    top = max(score for _, score in candidates)
    # Means of equal metrics over different neighbourhood sizes round
    # differently, so an exact comparison would pick a winner by rounding.
    # The tolerance is relative to the two scores compared, not to the
    # largest score anywhere, which a far-off failing region would inflate.
    return next(
        index
        for index, score in candidates
        if _within_tolerance(top - score, max(abs(top), abs(score)))
    )


def _within_tolerance(gap: float, scale: float) -> bool:
    return gap <= _RELATIVE_TOLERANCE * scale


def _build_grid(trials: Sequence[Trial], axes: Mapping[str, Iterable[float]]) -> _Grid:
    if not isinstance(axes, Mapping):
        raise ValueError("axes must be a mapping of parameter names to values")
    if not all(isinstance(name, str) for name in axes):
        raise ValueError("axes keys must be parameter names")
    names = tuple(sorted(axes))
    if not names:
        if trials:
            raise ValueError("axes must name at least one parameter")
        # An empty search, such as trials_from_metrics of no rows.
        return _Grid(
            names=(),
            axes=(),
            params=(),
            ranks=np.empty((0, 0), dtype=np.int64),
            metrics=np.empty(0, dtype=np.float64),
        )
    sorted_axes = []
    for name in names:
        values = axes[name]
        if isinstance(values, str) or not isinstance(values, Iterable):
            raise ValueError(f"axis {name!r} must be a collection of values")
        axis = [_finite(value, f"axis {name!r} value") for value in values]
        if not axis:
            raise ValueError(f"axis {name!r} has no values")
        if len(set(axis)) != len(axis):
            raise ValueError(f"axis {name!r} repeats a value")
        sorted_axes.append(tuple(sorted(axis)))
    rank_of = [{value: rank for rank, value in enumerate(axis)} for axis in sorted_axes]

    params: list[Mapping[str, float]] = []
    ranks: list[list[int]] = []
    seen: set[tuple[float, ...]] = set()
    metrics: list[float] = []
    for position, trial in enumerate(trials):
        if set(trial.params) != set(names):
            raise ValueError(f"trial {position} parameter names differ from the axes")
        values = tuple(
            _finite(trial.params[name], f"trial {position} parameter {name!r}")
            for name in names
        )
        if values in seen:
            raise ValueError(f"trial {position} repeats an earlier parameter set")
        seen.add(values)
        row = []
        for k, (name, value) in enumerate(zip(names, values, strict=True)):
            if value not in rank_of[k]:
                raise ValueError(
                    f"trial {position} parameter {name!r} is not on its axis"
                )
            row.append(rank_of[k][value])
        ranks.append(row)
        params.append(MappingProxyType(dict(zip(names, values, strict=True))))
        metrics.append(_finite(trial.metric, f"trial {position} metric"))

    return _Grid(
        names=names,
        axes=tuple(sorted_axes),
        params=tuple(params),
        ranks=np.array(ranks, dtype=np.int64).reshape(len(params), len(names)),
        metrics=np.array(metrics, dtype=np.float64),
    )


def _numeric(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} is not numeric")
    try:
        return float(value)
    except OverflowError:
        # Too large for a float; every caller treats any infinity alike.
        return math.inf


def _finite(value: object, label: str) -> float:
    number = _numeric(value, label)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _checked_radius(radius: int) -> int:
    return _positive_int(radius, "radius")


def _positive_int(value: int, name: str) -> int:
    # numbers.Integral admits numpy integers, which grid indices usually are.
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
