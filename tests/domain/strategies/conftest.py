"""Shared fixtures for strategy tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from tradingdev.domain.strategies.base import BaseStrategy


class LookAheadAssertion(Protocol):
    def __call__(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        check_points: Iterable[int],
    ) -> None: ...


@pytest.fixture
def assert_no_look_ahead() -> LookAheadAssertion:
    """Truncation-based no-look-ahead check.

    The signal at bar ``t`` computed from data truncated at ``t`` must equal
    the signal at bar ``t`` computed from the full series; otherwise the
    strategy is using future information.
    """

    def _assert(
        strategy: BaseStrategy,
        df: pd.DataFrame,
        check_points: Iterable[int],
    ) -> None:
        full = strategy.generate_signals(df)["signal"].reset_index(drop=True)
        for t in check_points:
            truncated_df = df.iloc[: t + 1].reset_index(drop=True)
            truncated = strategy.generate_signals(truncated_df)["signal"]
            truncated = truncated.reset_index(drop=True)
            assert truncated.iloc[t] == full.iloc[t], (
                f"Look-ahead bias: signal at bar {t} changes when future data "
                f"is removed (truncated={truncated.iloc[t]!r}, "
                f"full={full.iloc[t]!r})"
            )

    return _assert
