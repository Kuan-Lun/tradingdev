"""Static split-builder contracts; negative calls must remain errors."""

from __future__ import annotations

from typing import assert_type

from tradingdev.domain.ml.validation import (
    Split,
    ValidationScheme,
    get_validation_scheme,
)


def split_builders() -> None:
    scheme = get_validation_scheme("purged_walk_forward")
    assert_type(scheme.build(120, horizon=3, n_folds=2), list[Split])
    scheme.build(120, horizon="3")  # type: ignore[arg-type]
    scheme.build(120, horizon=3, n_fold=2)  # type: ignore[call-arg]
    ValidationScheme("invalid", "missing fold options", _incomplete_builder)  # type: ignore[arg-type]


def _incomplete_builder(n_bars: int, *, horizon: int) -> list[Split]:
    return []
