"""Static grid contracts checked by strict Mypy, not runtime pytest."""

from __future__ import annotations

from typing import assert_type

from tradingdev.domain.optimization.grid_search import (
    GridSearchResult,
    best_result,
    parameter_grid,
    tuple_grid,
)


def grid_contracts() -> None:
    assert_type(parameter_grid({"length": [10, 20]}), list[dict[str, int]])
    assert_type(tuple_grid([1, 2], [3, 4]), list[tuple[int, ...]])
    result = GridSearchResult(params={"length": 10}, metric_value=1.0, metrics={})
    assert_type(best_result([result], direction="maximize"), GridSearchResult[int])
    _text(parameter_grid({"length": [10]})[0]["length"])  # type: ignore[arg-type]


def _text(value: str) -> None:
    pass
