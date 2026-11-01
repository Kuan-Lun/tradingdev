"""Pin three KD backtest figures on the versioned fixture.

The figures are deterministic, so they are pinned at rel 1e-9 to the values
computed from the source pickle the fixture README names.
"""

import numpy as np
import pandas as pd
import pytest
from tests.domain.statistics.kd_fixture import KDFixture, load_kd_fixture

from tradingdev.domain.statistics.breakeven import summarize_breakeven
from tradingdev.domain.statistics.periods import period_returns
from tradingdev.domain.statistics.sharpe import sharpe_ratio

_REL = 1e-9
_DAYS = 366
_SHARPE = -7.565103143673017
_TRADED_NOTIONAL = 5014735.715506879
_COST_SHARE = 0.863481526386825


@pytest.fixture(scope="module")
def kd() -> KDFixture:
    return load_kd_fixture()


def _daily_returns(kd: KDFixture) -> pd.Series:
    return period_returns(kd.equity, kd.timestamps, "daily")


def _traded_notional(kd: KDFixture) -> float:
    return summarize_breakeven(
        kd.trades,
        start=kd.timestamps[0],
        end=kd.timestamps[-1],
        capital=kd.run["init_cash"],
    ).traded_notional


def _cost_share(kd: KDFixture) -> float:
    rate = float(kd.run["fees"]) + float(kd.run["slippage"])
    return _traded_notional(kd) * rate / abs(float(kd.equity[-1] - kd.equity[0]))


class TestResample:
    def test_hourly_equity_resamples_to_366_utc_days(self, kd: KDFixture) -> None:
        returns = _daily_returns(kd)

        assert len(kd.equity) == 8761
        assert len(returns) == _DAYS
        assert returns.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
        assert returns.index[-1] == pd.Timestamp("2024-12-31", tz="UTC")


class TestPinnedFigures:
    def test_annualised_daily_sharpe(self, kd: KDFixture) -> None:
        returns = _daily_returns(kd).to_numpy()

        sharpe = sharpe_ratio(returns, periods_per_year=365.0)

        assert sharpe == pytest.approx(_SHARPE, rel=_REL)

    def test_traded_notional(self, kd: KDFixture) -> None:
        engine_volume = float(
            np.sum([t["size_quote"] + t["exit_notional"] for t in kd.trades])
        )

        assert _traded_notional(kd) == pytest.approx(_TRADED_NOTIONAL, rel=_REL)
        assert engine_volume == pytest.approx(_TRADED_NOTIONAL, rel=_REL)

    def test_cost_share_of_net_pnl(self, kd: KDFixture) -> None:
        assert _cost_share(kd) == pytest.approx(_COST_SHARE, rel=_REL)
