"""Tests for the parameter plateau analysis."""

import itertools
import math
import warnings

import numpy as np
import pytest

from tradingdev.domain.statistics.plateau import (
    DroppedTrialsWarning,
    MetricTrials,
    Trial,
    plateau_grid,
    plateau_scores,
    plateau_summary,
    trials_from_metrics,
)

_A_VALUES = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
_B_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def _hill_grid(
    peak: tuple[int, int], *, height: float, spike: tuple[int, int, float] | None
) -> list[Trial]:
    trials = []
    for i, a in enumerate(_A_VALUES):
        for j, b in enumerate(_B_VALUES):
            metric = height - 0.05 * ((i - peak[0]) ** 2 + (j - peak[1]) ** 2)
            if spike is not None and (i, j) == spike[:2]:
                metric = spike[2]
            trials.append(Trial(params={"a": a, "b": b}, metric=metric))
    return trials


def _by_params(trials: list[Trial], **params: float) -> int:
    return next(i for i, t in enumerate(trials) if dict(t.params) == params)


class TestPlateauVersusSpike:
    def test_plateau_best_coincides_with_best_on_a_broad_plateau(self) -> None:
        summary = plateau_summary(_hill_grid((3, 3), height=1.0, spike=None))

        assert summary.best_params == {"a": 40.0, "b": 0.4}
        assert summary.plateau_best_params == summary.best_params
        assert summary.metric_gap == pytest.approx(0.0)
        assert summary.grid_distance == 0
        assert summary.best_neighborhood_size == 8
        assert summary.best_neighborhood_mean == pytest.approx(0.925)

    def test_plateau_best_leaves_a_single_spike(self) -> None:
        summary = plateau_summary(_hill_grid((2, 2), height=0.5, spike=(5, 5, 2.0)))

        assert summary.best_params == {"a": 60.0, "b": 0.6}
        assert summary.best_metric == pytest.approx(2.0)
        assert summary.plateau_best_params == {"a": 30.0, "b": 0.3}
        assert summary.plateau_best_params != summary.best_params
        assert summary.plateau_best_metric == pytest.approx(0.5)
        assert summary.metric_gap == pytest.approx(1.5)
        assert summary.grid_distance == 3
        assert summary.spikiness is not None
        assert summary.spikiness > 3.0


class TestDimensions:
    def test_single_parameter(self) -> None:
        trials = [
            Trial(params={"n": float(n)}, metric=m)
            for n, m in zip(range(1, 6), [0.0, 1.0, 5.0, 2.0, 0.0], strict=True)
        ]
        scores = plateau_scores(trials)

        assert [s.neighborhood_size for s in scores] == [1, 2, 2, 2, 1]
        assert [s.score for s in scores] == pytest.approx([1.0, 2.5, 1.5, 2.5, 2.0])

        summary = plateau_summary(trials)
        assert summary.n_dimensions == 1
        assert summary.dimension_warning is False
        assert summary.best_params == {"n": 3.0}
        assert summary.plateau_best_params == {"n": 2.0}
        assert summary.metric_gap == pytest.approx(4.0)
        assert summary.grid_distance == 1
        assert summary.best_neighborhood_mean == pytest.approx(1.5)
        assert summary.best_neighborhood_std == pytest.approx(math.sqrt(0.5))
        assert summary.spikiness == pytest.approx(3.5 / math.sqrt(0.5))

    def test_two_parameters_have_no_dimension_warning(self) -> None:
        summary = plateau_summary(_hill_grid((3, 3), height=1.0, spike=None))

        assert summary.n_dimensions == 2
        assert summary.dimension_warning is False

    def test_three_parameters_have_no_dimension_warning(self) -> None:
        trials = [
            Trial(params={"a": a, "b": b, "c": c}, metric=a + b + c)
            for a, b, c in itertools.product([0.0, 1.0, 2.0], repeat=3)
        ]
        summary = plateau_summary(trials)

        assert summary.dimension_warning is False
        assert max(s.neighborhood_size for s in plateau_scores(trials)) == 26

    def test_four_parameters_raise_dimension_warning(self) -> None:
        trials = [
            Trial(params=dict(zip("abcd", p, strict=True)), metric=-sum(p))
            for p in itertools.product([0.0, 1.0, 2.0], repeat=4)
        ]
        summary = plateau_summary(trials)
        scores = plateau_scores(trials)

        assert summary.n_dimensions == 4
        assert summary.dimension_warning is True
        center = _by_params(trials, a=1.0, b=1.0, c=1.0, d=1.0)
        assert scores[center].neighborhood_size == 3**4 - 1
        grid = plateau_grid(trials, x="a", y="b")
        assert grid.dimension_warning is True


class TestIncompleteGrid:
    def test_neighbourhood_counts_only_existing_points(self) -> None:
        missing = {(1.0, 1.0), (2.0, 2.0)}
        trials = [
            Trial(params={"a": a, "b": b}, metric=a - b)
            for a, b in itertools.product([0.0, 1.0, 2.0], repeat=2)
            if (a, b) not in missing
        ]
        scores = plateau_scores(trials)
        sizes = {(s.params["a"], s.params["b"]): s.neighborhood_size for s in scores}

        assert sizes == {
            (0.0, 0.0): 2,
            (0.0, 1.0): 4,
            (0.0, 2.0): 2,
            (1.0, 0.0): 4,
            (1.0, 2.0): 3,
            (2.0, 0.0): 2,
            (2.0, 1.0): 3,
        }
        corner = scores[_by_params(trials, a=0.0, b=0.0)]
        assert corner.score == pytest.approx(((0.0 - 1.0) + (1.0 - 0.0)) / 2)

    def test_isolated_point_has_no_score(self) -> None:
        trials = [
            Trial(params={"a": 0.0, "b": 0.0}, metric=1.0),
            Trial(params={"a": 2.0, "b": 2.0}, metric=2.0),
            Trial(params={"a": 1.0, "b": 0.0}, metric=0.0),
            Trial(params={"a": 0.0, "b": 1.0}, metric=0.5),
        ]
        scores = plateau_scores(trials)

        assert scores[1].neighborhood_size == 0
        assert scores[1].score is None
        summary = plateau_summary(trials)
        assert summary.best_params == {"a": 2.0, "b": 2.0}
        assert summary.best_neighborhood_size == 0
        assert summary.best_neighborhood_mean is None
        assert summary.spikiness is None
        assert summary.plateau_best_params == {"a": 1.0, "b": 0.0}

    def test_larger_radius_widens_the_neighbourhood(self) -> None:
        trials = [Trial(params={"n": float(n)}, metric=float(n)) for n in range(5)]

        assert [s.neighborhood_size for s in plateau_scores(trials, radius=2)] == [
            2,
            3,
            4,
            3,
            2,
        ]
        assert plateau_summary(trials, radius=2).radius == 2


class TestDegenerateInputs:
    def test_empty_trials(self) -> None:
        assert plateau_scores([]) == ()

        summary = plateau_summary([])
        assert summary.n_trials == 0
        assert summary.dimension_warning is False
        assert summary.best_params is None
        assert summary.best_metric is None
        assert summary.plateau_best_params is None
        assert summary.metric_gap is None
        assert summary.grid_distance is None
        assert summary.best_neighborhood_size is None
        assert summary.spikiness is None

        grid = plateau_grid([], x="a", y="b")
        assert grid.x_values == ()
        assert grid.metric == ()

    def test_single_trial(self) -> None:
        trials = [Trial(params={"a": 1.0, "b": 2.0}, metric=0.7)]
        (score,) = plateau_scores(trials)

        assert score.score is None
        assert score.neighborhood_size == 0

        summary = plateau_summary(trials)
        assert summary.best_params == {"a": 1.0, "b": 2.0}
        assert summary.best_metric == pytest.approx(0.7)
        assert summary.plateau_best_params is None
        assert summary.plateau_best_score is None
        assert summary.metric_gap is None
        assert summary.grid_distance is None
        assert summary.best_neighborhood_mean is None
        assert summary.best_neighborhood_std is None
        assert summary.best_neighborhood_size == 0
        assert summary.spikiness is None

    def test_constant_metric(self) -> None:
        trials = [
            Trial(params={"a": a, "b": b}, metric=0.3)
            for a, b in itertools.product([1.0, 2.0, 3.0], repeat=2)
        ]
        summary = plateau_summary(trials)

        assert summary.best_params == {"a": 1.0, "b": 1.0}
        assert summary.plateau_best_params == {"a": 1.0, "b": 1.0}
        assert summary.metric_gap == pytest.approx(0.0)
        assert summary.grid_distance == 0
        assert summary.best_neighborhood_mean == pytest.approx(0.3)
        assert summary.best_neighborhood_std == pytest.approx(0.0)
        assert summary.spikiness is None


class TestGridIndexNeighbourhood:
    def test_parameters_a_thousand_times_apart_use_grid_steps(self) -> None:
        fast = [0.001, 0.002, 0.004, 0.008]
        slow = [1_000.0, 2_000.0, 3_000.0, 4_000.0]
        rng = np.random.default_rng(7)
        metrics = rng.normal(0.0, 1.0, (4, 4))
        metrics[1, 2] = 10.0
        trials = [
            Trial(params={"fast": f, "slow": s}, metric=float(metrics[i, j]))
            for i, f in enumerate(fast)
            for j, s in enumerate(slow)
        ]
        scores = plateau_scores(trials)

        center = scores[_by_params(trials, fast=0.002, slow=3_000.0)]
        assert center.neighborhood_size == 8
        corner = scores[_by_params(trials, fast=0.008, slow=4_000.0)]
        assert corner.neighborhood_size == 3

        expected = np.delete(metrics[0:3, 1:4].ravel(), 4)
        summary = plateau_summary(trials)
        assert summary.best_params == {"fast": 0.002, "slow": 3_000.0}
        assert summary.best_neighborhood_size == 8
        assert summary.best_neighborhood_mean == pytest.approx(expected.mean())
        assert summary.best_neighborhood_std == pytest.approx(expected.std(ddof=1))


class TestPlateauGrid:
    def test_two_parameter_grid_layout(self) -> None:
        trials = [
            Trial(params={"a": a, "b": b}, metric=a * 10 + b)
            for a, b in itertools.product([1.0, 2.0, 3.0], [5.0, 6.0])
            if (a, b) != (3.0, 5.0)
        ]
        grid = plateau_grid(trials, x="a", y="b")

        assert grid.x_values == (1.0, 2.0, 3.0)
        assert grid.y_values == (5.0, 6.0)
        assert grid.fixed_params == {}
        assert grid.metric == ((15.0, 25.0, None), (16.0, 26.0, 36.0))
        assert grid.neighborhood_size == ((3, 4, None), (3, 4, 2))
        assert grid.score[0][0] == pytest.approx((25.0 + 16.0 + 26.0) / 3)
        assert grid.score[0][2] is None

    def test_extra_parameters_are_fixed_at_best(self) -> None:
        trials = [
            Trial(params={"a": a, "b": b, "c": c}, metric=-abs(c - 1.0) + a + b)
            for a, b, c in itertools.product([0.0, 1.0, 2.0], repeat=3)
        ]
        grid = plateau_grid(trials, x="b", y="a")

        assert grid.fixed_params == {"c": 1.0}
        assert grid.metric[2][1] == pytest.approx(3.0)
        assert grid.metric[0][0] == pytest.approx(0.0)
        assert grid.neighborhood_size[1][1] == 26

    @pytest.mark.parametrize(
        ("x", "y"), [("a", "a"), ("a", "z")], ids=["same", "unknown"]
    )
    def test_rejects_invalid_axes(self, x: str, y: str) -> None:
        trials = [Trial(params={"a": 1.0, "b": 2.0}, metric=0.0)]

        with pytest.raises(ValueError):
            plateau_grid(trials, x=x, y=y)


class TestValidation:
    @pytest.mark.parametrize("radius", [0, -1, True, 1.5])
    def test_rejects_invalid_radius(self, radius: object) -> None:
        trials = [Trial(params={"a": 1.0}, metric=0.0)]

        with pytest.raises(ValueError, match="radius"):
            plateau_summary(trials, radius=radius)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("trials", "match"),
        [
            (
                [Trial({"a": 1.0}, 0.0), Trial({"b": 1.0}, 0.0)],
                "different parameter names",
            ),
            ([Trial({"a": 1.0}, 0.0), Trial({"a": 1.0}, 1.0)], "repeats"),
            ([Trial({"a": 1.0}, math.nan)], "metric is not finite"),
            ([Trial({"a": math.inf}, 0.0)], "parameter 'a' is not finite"),
            ([Trial({}, 0.0)], "at least one parameter"),
        ],
        ids=["names", "duplicate", "nan-metric", "inf-param", "no-params"],
    )
    def test_rejects_inconsistent_trials(self, trials: list[Trial], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            plateau_scores(trials)


def _corner_grid() -> list[Trial]:
    """5x5 grid whose metric falls with the grid index sum.

    Without a threshold the corner (0, 0) has the best plateau score; its
    three neighbours average -4/3, ahead of the edge (0, 1) at -1.6 and the
    interior (1, 1) at -2.
    """
    return [
        Trial(params={"x": float(i), "y": float(j)}, metric=-float(i + j))
        for i, j in itertools.product(range(5), repeat=2)
    ]


def _metric_rows() -> list[dict[str, object]]:
    return [
        {"params": {"a": 1.0}, "max_drawdown": 0.30},
        {"params": {"a": 2.0}, "max_drawdown": 0.05},
        {"params": {"a": 3.0}, "max_drawdown": 0.20},
        {"params": {"a": 4.0}, "max_drawdown": 0.40},
    ]


class TestTrialsFromMetrics:
    def test_minimize_selects_the_smallest_raw_metric(self) -> None:
        result = trials_from_metrics(
            _metric_rows(),
            params_key="params",
            metric_key="max_drawdown",
            direction="minimize",
        )

        metrics = [t.metric for t in result.trials]
        assert metrics == pytest.approx([-0.30, -0.05, -0.20, -0.40])
        assert result.dropped_count == 0
        summary = plateau_summary(result.trials)
        assert summary.best_params == {"a": 2.0}
        assert summary.best_metric == pytest.approx(-0.05)

    def test_direction_changes_the_best_params(self) -> None:
        def best(direction: str) -> object:
            result = trials_from_metrics(
                _metric_rows(),
                params_key="params",
                metric_key="max_drawdown",
                direction=direction,  # type: ignore[arg-type]
            )
            return plateau_summary(result.trials).best_params

        assert best("maximize") == {"a": 4.0}
        assert best("minimize") == {"a": 2.0}

    def test_rejects_repeated_params(self) -> None:
        rows = [
            {"params": {"a": 1.0, "b": 2.0}, "m": 0.1},
            {"params": {"b": 2.0, "a": 1}, "m": 0.2},
        ]

        with pytest.raises(ValueError, match="row 1 repeats"):
            trials_from_metrics(
                rows, params_key="params", metric_key="m", direction="maximize"
            )

    def test_drops_non_finite_metrics_and_reports_the_count(self) -> None:
        rows = [
            {"params": {"a": 1.0}, "m": 0.1},
            {"params": {"a": 2.0}, "m": math.nan},
            {"params": {"a": 3.0}, "m": math.inf},
            {"params": {"a": 4.0}, "m": -math.inf},
            {"params": {"a": 5.0}, "m": 0.5},
        ]

        with pytest.warns(DroppedTrialsWarning) as record:
            result = trials_from_metrics(
                rows, params_key="params", metric_key="m", direction="maximize"
            )

        assert [dict(t.params) for t in result.trials] == [{"a": 1.0}, {"a": 5.0}]
        assert result.dropped_count == 3
        assert len(record) == 1
        warning = record[0].message
        assert isinstance(warning, DroppedTrialsWarning)
        assert warning.dropped_count == 3

    def test_dropped_count_is_returned_when_warnings_are_ignored(self) -> None:
        rows = [
            {"params": {"a": 1.0}, "m": math.nan},
            {"params": {"a": 2.0}, "m": 0.2},
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = trials_from_metrics(
                rows, params_key="params", metric_key="m", direction="maximize"
            )

        assert result == MetricTrials(trials=(result.trials[0],), dropped_count=1)
        assert dict(result.trials[0].params) == {"a": 2.0}

    def test_none_and_overflowing_metrics_are_failed_trials(self) -> None:
        rows: list[dict[str, object]] = [
            {"params": {"a": 1.0}, "m": None},
            {"params": {"a": 2.0}, "m": 10**400},
            {"params": {"a": 3.0}, "m": 0.3},
        ]

        with pytest.warns(DroppedTrialsWarning):
            result = trials_from_metrics(
                rows, params_key="params", metric_key="m", direction="minimize"
            )

        assert [dict(t.params) for t in result.trials] == [{"a": 3.0}]
        assert result.dropped_count == 2

    def test_dropping_every_row_returns_no_trials(self) -> None:
        rows = [{"params": {"a": 1.0}, "m": math.nan}]

        with pytest.warns(DroppedTrialsWarning):
            result = trials_from_metrics(
                rows, params_key="params", metric_key="m", direction="maximize"
            )

        assert result.trials == ()
        assert result.dropped_count == 1

    def test_dropped_row_still_occupies_its_params(self) -> None:
        rows = [
            {"params": {"a": 1.0}, "m": math.nan},
            {"params": {"a": 1.0}, "m": 0.2},
        ]

        with pytest.raises(ValueError, match="row 1 repeats"):
            trials_from_metrics(
                rows, params_key="params", metric_key="m", direction="maximize"
            )

    def test_no_warning_when_nothing_is_dropped(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        result = trials_from_metrics(
            _metric_rows(),
            params_key="params",
            metric_key="max_drawdown",
            direction="maximize",
        )

        assert len(result.trials) == 4
        assert result.dropped_count == 0
        assert not [w for w in recwarn if w.category is DroppedTrialsWarning]

    @pytest.mark.parametrize(
        ("rows", "direction", "match"),
        [
            ([{"params": {"a": 1.0}, "m": 0.1}], "max", "direction"),
            ([{"params": {"a": 1.0}}], "maximize", "no 'm'"),
            ([{"m": 0.1}], "maximize", "no 'params'"),
            ([{"params": {"a": 1.0}, "m": "0.1"}], "maximize", "metric is not numeric"),
            ([{"params": {"a": math.nan}, "m": 0.1}], "maximize", "not finite"),
            ([{"params": [("a", 1.0)], "m": 0.1}], "maximize", "non-empty mapping"),
            ([{"params": {}, "m": 0.1}], "maximize", "non-empty mapping"),
        ],
        ids=[
            "direction",
            "no-metric",
            "no-params",
            "str-metric",
            "nan-param",
            "list-params",
            "empty-params",
        ],
    )
    def test_rejects_invalid_rows(
        self, rows: list[dict[str, object]], direction: str, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            trials_from_metrics(
                rows,
                params_key="params",
                metric_key="m",
                direction=direction,  # type: ignore[arg-type]
            )


class TestMinNeighborhood:
    def test_default_keeps_existing_behaviour(self) -> None:
        trials = _corner_grid()

        default = plateau_summary(trials)
        explicit = plateau_summary(trials, min_neighborhood=None)

        assert explicit == default
        assert default.min_neighborhood is None
        assert default.n_below_min_neighborhood == 0
        assert default.plateau_best_params == {"x": 0.0, "y": 0.0}
        assert default.plateau_best_score == pytest.approx(-4 / 3)

    def test_four_excludes_only_the_corners_of_a_5x5_grid(self) -> None:
        summary = plateau_summary(_corner_grid(), min_neighborhood=4)

        assert summary.min_neighborhood == 4
        assert summary.n_below_min_neighborhood == 4
        assert summary.plateau_best_params == {"x": 0.0, "y": 1.0}
        assert summary.plateau_best_score == pytest.approx(-1.6)

    def test_eight_leaves_only_the_interior_of_a_5x5_grid(self) -> None:
        summary = plateau_summary(_corner_grid(), min_neighborhood=8)

        assert summary.n_below_min_neighborhood == 16
        assert summary.plateau_best_params == {"x": 1.0, "y": 1.0}
        assert summary.plateau_best_score == pytest.approx(-2.0)
        assert summary.grid_distance == 1

    def test_excluded_trials_still_report_scores(self) -> None:
        trials = _corner_grid()

        scores = plateau_scores(trials)
        corner = scores[_by_params(trials, x=0.0, y=0.0)]

        assert corner.neighborhood_size == 3
        assert corner.score == pytest.approx(-4 / 3)

    def test_threshold_excluding_everything_returns_none(self) -> None:
        summary = plateau_summary(_corner_grid(), min_neighborhood=9)

        assert summary.n_below_min_neighborhood == 25
        assert summary.plateau_best_params is None
        assert summary.plateau_best_metric is None
        assert summary.plateau_best_score is None
        assert summary.metric_gap is None
        assert summary.grid_distance is None
        assert summary.best_params == {"x": 0.0, "y": 0.0}

    def test_threshold_applies_to_wider_radius(self) -> None:
        summary = plateau_summary(_corner_grid(), radius=2, min_neighborhood=24)

        assert summary.n_below_min_neighborhood == 24
        assert summary.plateau_best_params == {"x": 2.0, "y": 2.0}

    def test_empty_trials_record_the_threshold(self) -> None:
        summary = plateau_summary([], min_neighborhood=4)

        assert summary.min_neighborhood == 4
        assert summary.n_below_min_neighborhood == 0
        assert summary.plateau_best_params is None

    @pytest.mark.parametrize("min_neighborhood", [0, -1, True, 2.5])
    def test_rejects_invalid_threshold(self, min_neighborhood: object) -> None:
        with pytest.raises(ValueError, match="min_neighborhood"):
            plateau_summary(
                _corner_grid(),
                min_neighborhood=min_neighborhood,  # type: ignore[arg-type]
            )


class TestAxisSizes:
    def test_reports_grid_points_per_axis(self) -> None:
        trials = [
            Trial(params={"lookback": float(i), "threshold": float(j)}, metric=0.0)
            for i, j in itertools.product(range(5), range(3))
        ]

        summary = plateau_summary(trials)

        assert summary.axis_sizes == {"lookback": 5, "threshold": 3}

    def test_empty_trials_have_no_axes(self) -> None:
        assert plateau_summary([]).axis_sizes == {}
