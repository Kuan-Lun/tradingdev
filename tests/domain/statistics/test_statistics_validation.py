"""Inputs of the wrong kind are rejected at each boundary, never converted.

Each case used to be accepted silently: NumPy read booleans as 0/1 and
numeric strings as numbers, pandas read integer timestamps as nanoseconds,
and the modules disagreed on whether a NumPy integer is a count.
"""

from __future__ import annotations

from functools import partial
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.statistics.benchmark import buy_and_hold_returns
from tradingdev.domain.statistics.comparison import (
    align_returns,
    difference_interval,
    difference_interval_outcome,
    interval_table,
    interval_table_outcome,
)
from tradingdev.domain.statistics.performance import summarize_performance
from tradingdev.domain.statistics.periods import (
    equity_series,
    has_positive_equity,
    period_returns,
    to_utc_index,
)
from tradingdev.domain.statistics.plateau import (
    Trial,
    plateau_grid,
    plateau_scores,
    plateau_summary,
)
from tradingdev.domain.statistics.sharpe import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_confidence_interval,
    sharpe_interval_outcome,
    sharpe_ratio,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

_N = 40
_INDEX = pd.date_range("2024-01-01", periods=_N, freq="D", tz="UTC")
_RNG = np.random.default_rng(0)
_A = pd.Series(_RNG.normal(0.001, 0.01, _N), index=_INDEX)
_B = pd.Series(_RNG.normal(0.0, 0.01, _N), index=_INDEX)
_TRIALS = [Trial({"x": float(x)}, float(x)) for x in range(3)]
_AXES = {"x": (0.0, 1.0, 2.0)}
_GRID = [
    Trial({"x": float(x), "y": float(y)}, 0.0) for x, y in product(range(3), repeat=2)
]
_GRID_AXES = {"x": (0.0, 1.0, 2.0), "y": (0.0, 1.0, 2.0)}
_UNIX_SECONDS = np.arange(1_704_067_200, 1_704_067_200 + 86_400 * 30, 3_600)


def _mean(values: npt.NDArray[np.float64]) -> float:
    return float(np.mean(values))


_SHARPE = partial(sharpe_ratio, periods_per_year=365)


class TestTimestamps:
    @pytest.mark.parametrize(
        "timestamps",
        [
            _UNIX_SECONDS,
            _UNIX_SECONDS * 1_000_000_000,
            _UNIX_SECONDS.astype(float),
            np.array([True, False]),
            list(_UNIX_SECONDS[:3]),
            np.array([1_704_067_200, "2024-01-01"], dtype=object),
            pd.Index(_UNIX_SECONDS),
        ],
        ids=[
            "unix-seconds",
            "int-nanoseconds",
            "float",
            "bool",
            "int-list",
            "object-with-int",
            "int-index",
        ],
    )
    def test_numbers_are_rejected(self, timestamps: object) -> None:
        with pytest.raises(ValueError, match="not numbers"):
            to_utc_index(timestamps)  # type: ignore[arg-type]

    def test_unix_seconds_no_longer_collapse_into_one_day(self) -> None:
        # 30 days of hourly bars used to become one daily period, silently.
        equity = np.linspace(100.0, 110.0, _UNIX_SECONDS.size)
        with pytest.raises(ValueError, match="pd.to_datetime"):
            summarize_performance(equity, _UNIX_SECONDS, [])

    def test_unix_seconds_work_once_the_caller_states_the_unit(self) -> None:
        equity = np.linspace(100.0, 110.0, _UNIX_SECONDS.size)
        timestamps = pd.to_datetime(_UNIX_SECONDS, unit="s")

        assert summarize_performance(equity, timestamps, []).n_days == 30

    @pytest.mark.parametrize(
        "timestamps",
        [
            pd.date_range("2024-01-01", periods=3, freq="D"),
            pd.date_range("2024-01-01", periods=3, freq="D").to_numpy(),
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
        ],
        ids=["index", "datetime64", "strings", "timestamps"],
    )
    def test_datetimes_and_date_strings_are_accepted(self, timestamps: object) -> None:
        assert len(to_utc_index(timestamps)) >= 2  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("timestamps", "match"),
        [
            (np.array([1, 2], dtype="timedelta64[D]"), "got timedelta64"),
            (np.array([b"2024-01-01"]), "got |S10"),
            (np.array([np.True_], dtype=object), "not numbers"),
            ([pd.Timedelta("1D")], "got Timedelta"),
            (["not a date"], "cannot be parsed"),
            ([object()], "cannot be parsed"),
        ],
        ids=[
            "timedelta64",
            "bytes",
            "object-numpy-bool",
            "timedelta",
            "text",
            "object",
        ],
    )
    def test_wrong_kinds_raise_value_error(
        self, timestamps: object, match: str
    ) -> None:
        # These used to escape as TypeError (or DateParseError) from pandas.
        with pytest.raises(ValueError, match=match):
            to_utc_index(timestamps)  # type: ignore[arg-type]

    def test_numpy_strings_are_parsed_like_strings(self) -> None:
        values = np.array([np.str_("2024-01-01"), np.str_("2024-01-02")], dtype=object)

        assert list(to_utc_index(values)) == list(
            pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
        )

    def test_an_unhashable_frequency_raises_value_error(self) -> None:
        days = pd.date_range("2024-01-01", periods=2, freq="D")
        with pytest.raises(ValueError, match="frequency must be one of"):
            period_returns([100.0, 101.0], days, ["daily"])  # type: ignore[arg-type]


class TestEquity:
    _DAYS = pd.date_range("2024-01-01", periods=3, freq="D")

    @pytest.mark.parametrize(
        ("equity", "match"),
        [
            (np.array([True, True, True]), "not booleans"),
            (["100", "101", "102"], "got '100'"),
            (np.array([100.0, None, 102.0], dtype=object), "got None"),
            (np.array([100.0, True, 102.0], dtype=object), "got True"),
            # NumPy promotes these lists to floats before a dtype can show it.
            ([100.0, True, 102.0], "got True"),
            ([np.True_, 100.0, 102.0], "got np.True_"),
        ],
        ids=[
            "bool",
            "numeric-strings",
            "object-none",
            "object-bool",
            "list-with-bool",
            "list-with-numpy-bool",
        ],
    )
    def test_non_numbers_are_rejected(self, equity: object, match: str) -> None:
        with pytest.raises(ValueError, match=f"equity_curve must be numbers, {match}"):
            equity_series(equity, self._DAYS)  # type: ignore[arg-type]

    def test_has_positive_equity_rejects_non_numbers(self) -> None:
        with pytest.raises(ValueError, match="got True"):
            has_positive_equity([True, True])
        with pytest.raises(ValueError, match="got '1'"):
            has_positive_equity(["1", "2"])

    def test_summarize_performance_rejects_a_boolean_in_a_list(self) -> None:
        with pytest.raises(ValueError, match="got True"):
            summarize_performance([100.0, True, 102.0], self._DAYS, [])

    def test_buy_and_hold_rejects_boolean_prices(self) -> None:
        with pytest.raises(ValueError, match="not booleans"):
            buy_and_hold_returns(
                np.array([True, True, True]), timestamps=self._DAYS, frequency="daily"
            )

    def test_integer_and_nullable_float_equity_are_accepted(self) -> None:
        assert period_returns([100, 101, 102], self._DAYS, "daily").size == 3
        nullable = pd.Series([100.0, 101.0, 102.0], dtype="Float64")
        assert period_returns(nullable, self._DAYS, "daily").size == 3


class TestReturns:
    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            (pd.Series([True, False] * (_N // 2), index=_INDEX), "not booleans"),
            (pd.Series(["0.01"] * _N, index=_INDEX), "got '0.01'"),
            (
                pd.Series([0.01] * (_N - 1) + [True], index=_INDEX, dtype=object),
                "got True",
            ),
        ],
        ids=["bool", "numeric-strings", "object-with-bool"],
    )
    def test_align_returns_rejects_non_numbers(
        self, bad: pd.Series, match: str
    ) -> None:
        with pytest.raises(ValueError, match=f"returns 'a' must be numbers, {match}"):
            align_returns({"a": bad, "b": _B})

    def test_paired_interval_rejects_a_boolean_signal(self) -> None:
        # A 0/1 signal passed instead of returns used to pair as returns.
        signal = pd.Series([True, False] * (_N // 2), index=_INDEX)
        with pytest.raises(ValueError, match="not booleans"):
            difference_interval(signal, _B, _mean, reps=20, seed=0)

    @pytest.mark.parametrize(
        "bad",
        [[True, False, True], ["0.01", "0.02", "0.03"]],
        ids=["bool", "numeric-strings"],
    )
    def test_sharpe_functions_reject_non_numbers(self, bad: list[object]) -> None:
        with pytest.raises(ValueError, match="returns must be numbers"):
            sharpe_ratio(bad, periods_per_year=365)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="returns must be numbers"):
            probabilistic_sharpe_ratio(bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad",
        [
            [True, 0.5, 0.1],
            np.array([0.01, True, 0.0], dtype=object),
            pd.Series([True, 0.01, 0.02, 0.0], dtype=object),
        ],
        ids=["list-with-bool", "object-array", "object-series"],
    )
    def test_sharpe_functions_reject_hidden_booleans(self, bad: object) -> None:
        # align_returns already refused these; the Sharpe entry points listed
        # them first and so let NumPy turn True into 1.0.
        with pytest.raises(ValueError, match="returns must be numbers, got True"):
            sharpe_ratio(bad, periods_per_year=365)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="returns must be numbers, got True"):
            sharpe_interval_outcome(bad, periods_per_year=365)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="returns must be numbers, got True"):
            deflated_sharpe_ratio(
                bad,  # type: ignore[arg-type]
                [0.5, 1.5],
                periods_per_year=365,
            )

    def test_sharpe_accepts_a_generator_and_a_float_series(self) -> None:
        values = [0.01, 0.02, -0.01, 0.0]
        expected = sharpe_ratio(values, periods_per_year=365)

        assert sharpe_ratio((v for v in values), periods_per_year=365) == expected
        assert sharpe_ratio(pd.Series(values), periods_per_year=365) == expected

    def test_integer_returns_are_accepted(self) -> None:
        integers = pd.Series([0, 1] * (_N // 2), index=_INDEX)

        assert align_returns({"a": integers, "b": _B}).sufficient
        assert sharpe_ratio([0, 1, 0, 1], periods_per_year=365) is not None


class TestPairing:
    @pytest.mark.parametrize(
        "bad",
        [[0.01] * _N, np.zeros(_N), pd.DataFrame({"x": np.zeros(_N)})],
        ids=["list", "ndarray", "dataframe"],
    )
    def test_unlabelled_returns_raise_value_error(self, bad: object) -> None:
        # These used to fail with AttributeError on series.index.
        with pytest.raises(ValueError, match="must be a pandas Series"):
            align_returns({"a": bad})  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="must be a pandas Series"):
            interval_table({"a": bad}, _mean, reps=20)  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="must be a pandas Series"):
            difference_interval(bad, _B, _mean, reps=20)  # type: ignore[arg-type]


def _seeded_calls() -> list[Callable[[Any], object]]:
    return [
        lambda s: difference_interval(_A, _B, _mean, reps=20, seed=s),
        lambda s: difference_interval_outcome(_A, _B, _mean, reps=20, seed=s),
        lambda s: interval_table({"a": _A}, _mean, reps=20, seed=s),
        lambda s: interval_table_outcome({"a": _A}, _mean, reps=20, seed=s),
        lambda s: sharpe_confidence_interval(_A, periods_per_year=365, reps=20, seed=s),
        lambda s: sharpe_interval_outcome(_A, periods_per_year=365, reps=20, seed=s),
    ]


_SEEDED = _seeded_calls()
_SEEDED_IDS = [
    "difference",
    "difference-outcome",
    "table",
    "table-outcome",
    "sharpe",
    "sharpe-outcome",
]


class TestSeed:
    """A seed follows the count rule with zero allowed."""

    @pytest.mark.parametrize(
        "seed",
        [True, np.int64(1), 1.5, "1", -1],
        ids=["bool", "numpy-int", "float", "str", "negative"],
    )
    @pytest.mark.parametrize("call", _SEEDED, ids=_SEEDED_IDS)
    def test_rejects(self, call: Callable[[Any], object], seed: object) -> None:
        # True used to be seed 1; 1.5 and "1" escaped as TypeError from NumPy.
        with pytest.raises(ValueError, match="seed must be None or a non-negative"):
            call(seed)

    @pytest.mark.parametrize("seed", [None, 0, 7], ids=["none", "zero", "int"])
    @pytest.mark.parametrize("call", _SEEDED, ids=_SEEDED_IDS)
    def test_accepts(self, call: Callable[[Any], object], seed: object) -> None:
        call(seed)


def _count_parameters() -> list[tuple[str, Callable[[int], object]]]:
    return [
        ("min_overlap", lambda v: align_returns({"a": _A}, min_overlap=v)),
        ("reps", lambda v: difference_interval(_A, _B, _mean, reps=v)),
        ("reps", lambda v: difference_interval_outcome(_A, _B, _mean, reps=v)),
        (
            "block_size",
            lambda v: difference_interval(_A, _B, _mean, reps=20, block_size=v),
        ),
        (
            "block_size",
            lambda v: difference_interval_outcome(_A, _B, _mean, reps=20, block_size=v),
        ),
        ("reps", lambda v: interval_table({"a": _A}, _mean, reps=v)),
        ("reps", lambda v: interval_table_outcome({"a": _A}, _mean, reps=v)),
        (
            "reps",
            lambda v: sharpe_confidence_interval(_A, periods_per_year=365, reps=v),
        ),
        ("reps", lambda v: sharpe_interval_outcome(_A, periods_per_year=365, reps=v)),
        (
            "n_trials",
            lambda v: deflated_sharpe_ratio(
                _A, [0.5, 1.5], periods_per_year=365, n_trials=v
            ),
        ),
        ("radius", lambda v: plateau_scores(_TRIALS, axes=_AXES, radius=v)),
        ("radius", lambda v: plateau_summary(_TRIALS, axes=_AXES, radius=v)),
        (
            "radius",
            lambda v: plateau_grid(_GRID, axes=_GRID_AXES, x="x", y="y", radius=v),
        ),
        (
            "min_neighborhood",
            lambda v: plateau_summary(_TRIALS, axes=_AXES, min_neighborhood=v),
        ),
    ]


_COUNTS = _count_parameters()
_COUNT_IDS = [f"{name}-{position}" for position, (name, _) in enumerate(_COUNTS)]


class TestCountParameters:
    """Every count follows one rule: a built-in int of at least one."""

    @pytest.mark.parametrize(
        "value",
        [np.int64(2), True, 0, -1, 2.0],
        ids=["numpy-int", "bool", "zero", "negative", "float"],
    )
    @pytest.mark.parametrize(("name", "call"), _COUNTS, ids=_COUNT_IDS)
    def test_rejects(
        self, name: str, call: Callable[[int], object], value: object
    ) -> None:
        with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
            call(value)  # type: ignore[arg-type]

    def test_the_message_names_the_type_that_was_passed(self) -> None:
        with pytest.raises(ValueError, match=r"got np\.int64\(5\) of type int64"):
            deflated_sharpe_ratio(
                _A,
                [0.5, 1.5],
                periods_per_year=365,
                n_trials=np.int64(5),  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(("name", "call"), _COUNTS, ids=_COUNT_IDS)
    def test_accepts_a_built_in_int(
        self, name: str, call: Callable[[int], object]
    ) -> None:
        call(2)
