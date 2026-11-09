"""Static model-artifact value contracts; negative calls must remain errors."""

from __future__ import annotations

from typing import assert_type

from tradingdev.domain.ml.registry import feature_importance_from_gain


def importance_values() -> None:
    assert_type(
        feature_importance_from_gain(["x"], ["0.25"]), tuple[tuple[str, float], ...]
    )
    feature_importance_from_gain(["x"], [object()])  # type: ignore[list-item]
