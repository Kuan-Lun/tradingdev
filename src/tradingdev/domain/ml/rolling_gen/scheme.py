"""Resolve feature-aware, causal rolling geometry before execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradingdev.domain.ml.features.context import FeatureContext


class RollType(StrEnum):
    ROLL_SD = "ROLL_SD"
    ROLL_EX = "ROLL_EX"


@dataclass(frozen=True)
class RollingScheme:
    """A named training shape with optional overrides, all measured in bars."""

    name: str
    description: str
    rtype: RollType
    step: int | None = None
    window_size: int | None = None
    min_train_bars: int | None = None
    predict_lookback: int | None = None
    """Prior bars; the scored bar is added. Unset uses the feature preference."""


@dataclass(frozen=True)
class RollingGeometry:
    """Resolved once for schedule enumeration, execution and provenance."""

    name: str
    rtype: RollType
    step: int
    min_train_bars: int
    predict_lookback: int
    window_size: int | None

    def train_window(self, combined_idx: int) -> tuple[int, int]:
        if self.rtype is RollType.ROLL_EX:
            return 0, combined_idx
        if self.window_size is None:
            raise ValueError("ROLL_SD requires a window_size")
        return max(0, combined_idx - self.window_size), combined_idx

    def predict_context(self, combined_idx: int) -> tuple[int, int]:
        return max(0, combined_idx - self.predict_lookback), combined_idx + 1


def resolve_geometry(
    scheme: RollingScheme | None,
    *,
    lookback: int,
    retrain_interval: int,
    context: FeatureContext,
    history_bars: int = 0,
    label_horizon: int = 1,
) -> RollingGeometry:
    """Resolve automatic or declared geometry against actual feature needs.

    Automatic training slides a window covering the initial history, ten times
    lookback, or enough raw rows for ten mature observations, whichever is larger.
    Named schemes retain their declared shape and minimum labelled sample count.
    Every mode defaults to the feature's preferred prediction history. An explicit
    context below its minimum is rejected before fitting or scoring.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be at least 1 bar, got {lookback}")
    if history_bars < 0:
        raise ValueError("history_bars must be nonnegative")
    if label_horizon < 1:
        raise ValueError("label_horizon must be at least 1")
    if scheme is None:
        scheme = RollingScheme(
            "automatic",
            "Feature-aware sliding window",
            RollType.ROLL_SD,
            window_size=max(
                history_bars,
                10 * lookback,
                context.minimum_rows + label_horizon + 10 - 1,
            ),
            min_train_bars=10,
        )
    step = retrain_interval if scheme.step is None else scheme.step
    if step < 1:
        raise ValueError(f"rolling step must be at least 1 bar, got {step}")
    minimum = lookback + 10 if scheme.min_train_bars is None else scheme.min_train_bars
    if minimum < 1:
        raise ValueError(f"min_train_bars must be at least 1, got {minimum}")
    prior = (
        context.preferred_rows - 1
        if scheme.predict_lookback is None
        else scheme.predict_lookback
    )
    if prior < context.minimum_rows - 1:
        raise ValueError(
            f"predict_lookback must be at least {context.minimum_rows - 1} prior bars "
            f"for these features ({context.minimum_rows} rows including "
            "the scored bar); "
            f"got {prior}"
        )
    window: int | None = None
    if scheme.rtype is RollType.ROLL_SD:
        window = lookback * 10 if scheme.window_size is None else scheme.window_size
        if window < 1:
            raise ValueError(f"window_size must be at least 1 bar, got {window}")
    elif scheme.window_size is not None:
        raise ValueError("window_size only applies to ROLL_SD")
    return RollingGeometry(scheme.name, scheme.rtype, step, minimum, prior, window)
