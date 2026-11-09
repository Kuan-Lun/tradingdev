"""Negative calls ensure test helpers do not bypass keyword checking."""

from tests.domain.statistics.test_statistics_plateau import _grid, _scores, _summary


def invalid_helper_arguments() -> None:
    _scores([], raduis=2)  # type: ignore[call-arg]
    _summary([], min_neighborhood="8")  # type: ignore[arg-type]
    _grid([])  # type: ignore[call-arg]
