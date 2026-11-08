"""Static contracts checked by strict Mypy, without runtime execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

import pandas as pd

from tradingdev.domain.statistics.periods import (
    equity_series,
    period_closing_equity,
    period_pnl,
    period_returns,
)

if TYPE_CHECKING:
    import numpy.typing as npt


def period_types(equity: npt.ArrayLike, timestamps: npt.ArrayLike) -> None:
    assert_type(equity_series(equity, timestamps), pd.Series[float])
    assert_type(period_closing_equity(equity, timestamps, "daily"), pd.Series[float])
    assert_type(period_pnl(equity, timestamps, "monthly"), pd.Series[float])
    assert_type(period_returns(equity, timestamps, "daily"), pd.Series[float])
    _text(period_pnl(equity, timestamps, "daily").iloc[0])  # type: ignore[arg-type]


def _text(value: str) -> None:
    pass
