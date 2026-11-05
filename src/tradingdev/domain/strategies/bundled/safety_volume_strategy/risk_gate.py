"""Translate risk estimates into decisions independently of position management."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


class RiskDecision(IntEnum):
    UNAVAILABLE = -1
    BLOCK = 0
    ALLOW = 1


def evaluate_risk(
    scores: npt.NDArray[np.floating[Any]], threshold: float
) -> npt.NDArray[np.intp]:
    """NaN denotes no prediction; invalid finite probabilities are errors."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Risk threshold must be finite and in [0, 1]")
    available = ~np.isnan(scores)
    values = scores[available]
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Risk probabilities must be finite and in [0, 1]")
    decisions = np.full(len(scores), RiskDecision.UNAVAILABLE, dtype=np.intp)
    decisions[available] = np.where(
        values >= threshold, RiskDecision.ALLOW, RiskDecision.BLOCK
    )
    return decisions
