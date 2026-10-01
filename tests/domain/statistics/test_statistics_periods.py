"""Tests for UTC daily/monthly resampling of equity curves."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from tradingdev.domain.statistics.periods import (
    period_closing_equity,
    period_pnl,
    period_returns,
    to_utc_index,
)


def _hourly_2024() -> tuple[npt.NDArray[np.float64], pd.DatetimeIndex]:
    timestamps = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
    rng = np.random.default_rng(42)
    equity = 10_000.0 + np.cumsum(rng.normal(0.0, 5.0, len(timestamps)))
    return equity, timestamps


class TestFullYearHourly:
    def test_leap_year_has_366_days_and_12_months(self) -> None:
        equity, timestamps = _hourly_2024()
        assert len(timestamps) == 8784

        assert len(period_pnl(equity, timestamps, "daily")) == 366
        assert len(period_pnl(equity, timestamps, "monthly")) == 12

    def test_period_pnl_sums_to_total_pnl(self) -> None:
        equity, timestamps = _hourly_2024()
        total = equity[-1] - equity[0]

        assert period_pnl(equity, timestamps, "daily").sum() == pytest.approx(total)
        assert period_pnl(equity, timestamps, "monthly").sum() == pytest.approx(total)

    def test_daily_pnl_sums_into_monthly_pnl(self) -> None:
        equity, timestamps = _hourly_2024()
        daily = period_pnl(equity, timestamps, "daily")
        monthly = period_pnl(equity, timestamps, "monthly")

        regrouped = daily.groupby(pd.DatetimeIndex(daily.index).strftime("%Y-%m")).sum()
        np.testing.assert_allclose(regrouped.to_numpy(), monthly.to_numpy())

    def test_returns_compound_to_total_growth(self) -> None:
        equity, timestamps = _hourly_2024()
        growth = equity[-1] / equity[0]

        for frequency in ("daily", "monthly"):
            returns = period_returns(equity, timestamps, frequency)
            assert float(np.prod(1.0 + returns)) == pytest.approx(growth)

    def test_accepts_numpy_datetime64_timestamps(self) -> None:
        equity, timestamps = _hourly_2024()
        raw = timestamps.to_numpy(dtype="datetime64[ns]")

        pd.testing.assert_series_equal(
            period_pnl(equity, raw, "daily"),
            period_pnl(equity, timestamps, "daily"),
        )


class TestUtcBoundaries:
    def test_index_labels_period_start_in_utc(self) -> None:
        timestamps = pd.date_range("2024-01-31 22:00", periods=4, freq="h")
        closing = period_closing_equity([1.0, 2.0, 3.0, 4.0], timestamps, "monthly")

        assert list(closing.index) == [
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-02-01", tz="UTC"),
        ]
        assert closing.tolist() == [2.0, 4.0]

    def test_bar_at_2300_belongs_to_its_own_day(self) -> None:
        timestamps = pd.to_datetime(["2024-03-01 23:00", "2024-03-02 00:00"])
        pnl = period_pnl([100.0, 105.0], timestamps, "daily")

        assert pnl.index.tolist() == [
            pd.Timestamp("2024-03-01", tz="UTC"),
            pd.Timestamp("2024-03-02", tz="UTC"),
        ]
        assert pnl.tolist() == [0.0, 5.0]

    def test_aware_timestamps_are_bucketed_by_utc_not_local_date(self) -> None:
        # 07:00 and 09:00 in Taipei fall on different UTC days.
        timestamps = pd.DatetimeIndex(
            ["2024-01-01 07:00", "2024-01-01 09:00"], tz="Asia/Taipei"
        )
        pnl = period_pnl([100.0, 110.0], timestamps, "daily")

        assert pnl.index.tolist() == [
            pd.Timestamp("2023-12-31", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC"),
        ]
        assert pnl.tolist() == [0.0, 10.0]

    def test_naive_timestamps_are_treated_as_utc(self) -> None:
        naive = to_utc_index(pd.to_datetime(["2024-01-01 12:00"]))

        assert naive[0] == pd.Timestamp("2024-01-01 12:00", tz="UTC")

    def test_days_without_observations_are_omitted(self) -> None:
        timestamps = pd.to_datetime(
            ["2024-01-05 15:00", "2024-01-05 20:00", "2024-01-08 15:00"]
        )
        pnl = period_pnl([100.0, 102.0, 99.0], timestamps, "daily")

        assert pnl.index.tolist() == [
            pd.Timestamp("2024-01-05", tz="UTC"),
            pd.Timestamp("2024-01-08", tz="UTC"),
        ]
        assert pnl.tolist() == [2.0, -3.0]

    def test_single_observation_yields_one_flat_period(self) -> None:
        pnl = period_pnl([100.0], pd.to_datetime(["2024-01-01"]), "daily")

        assert pnl.tolist() == [0.0]


class TestInvalidInputs:
    def test_rejects_length_mismatch(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=3, freq="h")
        with pytest.raises(ValueError, match="does not match"):
            period_pnl([1.0, 2.0], timestamps, "daily")

    def test_rejects_empty_curve(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            period_pnl([], pd.DatetimeIndex([]), "daily")

    def test_rejects_non_increasing_timestamps(self) -> None:
        timestamps = pd.to_datetime(["2024-01-01 01:00", "2024-01-01 01:00"])
        with pytest.raises(ValueError, match="strictly increasing"):
            period_pnl([1.0, 2.0], timestamps, "daily")

    def test_rejects_non_finite_equity(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=2, freq="h")
        with pytest.raises(ValueError, match="finite"):
            period_pnl([1.0, np.nan], timestamps, "daily")

    def test_returns_require_positive_equity(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=2, freq="D")
        with pytest.raises(ValueError, match="strictly positive"):
            period_returns([0.0, 5.0], timestamps, "daily")
