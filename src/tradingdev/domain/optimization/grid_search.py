"""Generic grid-search primitives used by strategies and workers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class GridSearchResult:
    """One evaluated grid-search result."""

    params: dict[str, Any]
    metric_value: float
    metrics: dict[str, Any]


def parameter_grid(param_ranges: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Expand named parameter ranges into dictionaries."""
    names = list(param_ranges.keys())
    values = [list(param_ranges[name]) for name in names]
    return [
        dict(zip(names, combo, strict=True)) for combo in itertools.product(*values)
    ]


def tuple_grid(*ranges: Iterable[Any]) -> list[tuple[Any, ...]]:
    """Expand positional parameter ranges into tuples."""
    return list(itertools.product(*ranges))


def best_result(results: Iterable[GridSearchResult]) -> GridSearchResult:
    """Return the result with the highest target metric value."""
    return max(results, key=lambda item: item.metric_value)
