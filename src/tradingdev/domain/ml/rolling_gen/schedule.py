"""Enumerate the resolved geometry consumed by the prediction engine.

The same windows drive execution, so preview and trace cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradingdev.domain.ml.rolling_gen.scheme import RollingGeometry


@dataclass(frozen=True)
class RollWindow:
    """One retrain and the bars that model scores.

    Training bounds and ``combined_idx`` use the concatenated history/scoring
    frame. ``retrain_bar`` and test bounds use the scored frame's positions.
    Both intervals are half-open. ``train_end`` is the bar at the scheduled
    retrain attempt, so the training window excludes it by construction.
    """

    index: int
    """Roll number, counted from zero."""

    retrain_bar: int
    """Scheduled retrain attempt in the scored frame; fitting can be skipped."""

    combined_idx: int
    """The same bar, as a position in the concatenated frame."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_bars(self) -> int:
        """Raw window length before feature warm-up and label maturity checks."""
        return self.train_end - self.train_start

    @property
    def test_bars(self) -> int:
        """Planned bars for this roll, including any that cannot be scored."""
        return self.test_end - self.test_start


def enumerate_rolls(
    geometry: RollingGeometry,
    *,
    n_scored_bars: int,
    history_bars: int,
) -> tuple[RollWindow, ...]:
    """Return every roll the scheme would run, in order.

    Args:
        geometry: The already resolved geometry shared with execution.
        n_scored_bars: Bars to score.
        history_bars: Training rows retained after overlap with the scored frame
            is dropped, i.e. the scored frame's offset in the concatenated one.

    Returns:
        One entry per retrain. Rolls are contiguous and cover the scored frame
        exactly once; an empty scored frame yields no rolls.

    Raises:
        ValueError: If a count is negative, or the scheme cannot form a usable
            window.
    """
    if n_scored_bars < 0:
        msg = f"n_scored_bars must not be negative, got {n_scored_bars}"
        raise ValueError(msg)
    if history_bars < 0:
        msg = f"history_bars must not be negative, got {history_bars}"
        raise ValueError(msg)

    rolls: list[RollWindow] = []
    for index, retrain_bar in enumerate(range(0, n_scored_bars, geometry.step)):
        combined_idx = history_bars + retrain_bar
        train_start, train_end = geometry.train_window(combined_idx)
        rolls.append(
            RollWindow(
                index=index,
                retrain_bar=retrain_bar,
                combined_idx=combined_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=retrain_bar,
                test_end=min(retrain_bar + geometry.step, n_scored_bars),
            )
        )
    return tuple(rolls)
