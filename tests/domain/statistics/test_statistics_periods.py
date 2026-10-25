"""Tests for UTC daily/monthly resampling of equity curves."""

from typing import Literal

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


class TestTimestampResolution:
    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_any_input_resolution_gives_a_nanosecond_index(
        self, unit: Literal["s", "ms", "us", "ns"]
    ) -> None:
        equity, timestamps = _hourly_2024()

        returns = period_returns(equity, timestamps.as_unit(unit), "monthly")

        assert isinstance(returns.index, pd.DatetimeIndex)
        assert returns.index.unit == "ns"
        pd.testing.assert_series_equal(
            returns, period_returns(equity, timestamps.as_unit("ns"), "monthly")
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

    @pytest.mark.parametrize(
        "timestamps",
        [
            ["2024-03-30T12:00:00+01:00", "2024-03-31T12:00:00+02:00"],
            [
                pd.Timestamp("2024-03-30T12:00:00+01:00"),
                pd.Timestamp("2024-03-31T12:00:00+02:00"),
            ],
            list(
                pd.date_range(
                    "2024-03-30 12:00", periods=2, freq="D", tz="Europe/Berlin"
                )
            ),
        ],
        ids=["strings", "fixed-offset-timestamps", "zone-timestamps"],
    )
    def test_aware_timestamps_across_a_dst_change_are_bucketed_in_utc(
        self, timestamps: npt.ArrayLike
    ) -> None:
        # Berlin moves from +01:00 to +02:00 on 2024-03-31, so the two
        # observations carry different offsets.
        pnl = period_pnl([100.0, 101.0], timestamps, "daily")

        assert pnl.index.tolist() == [
            pd.Timestamp("2024-03-30", tz="UTC"),
            pd.Timestamp("2024-03-31", tz="UTC"),
        ]
        assert pnl.tolist() == [0.0, 1.0]

    @pytest.mark.parametrize(
        "timestamps",
        [
            ["2024-01-01", "2024-01-01 01:00"],
            # datetime.isoformat() drops a zero fraction.
            ["2024-01-01T00:00:00", "2024-01-01T01:00:00.500"],
            ["2024-01-01 00:00:00", "2024-01-01T01:00:00"],
        ],
        ids=["date-then-time", "isoformat-fraction", "space-then-t"],
    )
    def test_each_string_is_parsed_on_its_own(self, timestamps: list[str]) -> None:
        index = to_utc_index(timestamps)

        assert index.tolist() == [pd.Timestamp(value, tz="UTC") for value in timestamps]

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

    @pytest.mark.parametrize(
        "timestamps",
        [
            [
                pd.Timestamp("2024-03-30T12:00:00"),
                pd.Timestamp("2024-03-31T12:00+02:00"),
            ],
            [
                pd.Timestamp("2024-03-30T12:00+01:00"),
                pd.Timestamp("2024-03-31T12:00:00"),
            ],
            ["2024-03-30T12:00:00", "2024-03-31T12:00:00+02:00"],
            ["2024-03-30T12:00:00+01:00", "2024-03-31T12:00:00"],
            np.array(
                [
                    np.datetime64("2024-03-30T12:00"),
                    pd.Timestamp("2024-03-31", tz="UTC"),
                ],
                dtype=object,
            ),
            np.array(
                ["2024-03-30", pd.Timestamp("2024-03-31", tz="UTC")], dtype=object
            ),
        ],
        ids=[
            "naive-first",
            "aware-first",
            "naive-first-str",
            "aware-first-str",
            "datetime64-and-aware",
            "str-and-aware",
        ],
    )
    def test_rejects_naive_and_aware_timestamps_in_one_sequence(
        self, timestamps: npt.ArrayLike
    ) -> None:
        with pytest.raises(ValueError, match="mix timezone-naive and timezone-aware"):
            period_pnl([100.0, 101.0], timestamps, "daily")

    def test_rejects_two_dimensional_timestamps(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            period_pnl([100.0, 101.0], np.array([[0, 1]]), "daily")

    def test_rejects_non_finite_equity(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=2, freq="h")
        with pytest.raises(ValueError, match="finite"):
            period_pnl([1.0, np.nan], timestamps, "daily")

    def test_returns_require_positive_equity(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=2, freq="D")
        with pytest.raises(ValueError, match="strictly positive"):
            period_returns([0.0, 5.0], timestamps, "daily")
