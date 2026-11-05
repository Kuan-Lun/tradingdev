"""The standalone MAE prototype reports only complete future observations."""

from __future__ import annotations

import os
import sys
from inspect import unwrap
from pathlib import Path

import pandas as pd
import pytest
from research import mae_analysis
from tooling.process_guard import process_workspace, run_captured


def _prices(rows: int, *, frequency: str = "min") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=rows, freq=frequency, tz="UTC"
            ),
            "close": 100.0,
            "low": 99.0,
            "high": 101.0,
        }
    )


@pytest.mark.parametrize("rows, entries", [(0, 0), (2, 0), (3, 0), (4, 1), (6, 3)])
def test_only_entries_with_a_full_future_window_survive(
    rows: int, entries: int
) -> None:
    frame = _prices(rows)
    before = frame.copy(deep=True)
    result = unwrap(mae_analysis.compute_mae)(frame, 3, "2024-01-01", "2024-01-02")
    assert list(result.columns) == ["timestamp", "mae_long", "mae_short", "mae_worst"]
    assert len(result) == entries
    pd.testing.assert_series_equal(result["timestamp"], frame["timestamp"][:entries])
    assert result.index.equals(pd.RangeIndex(entries))
    pd.testing.assert_frame_equal(frame, before)


def test_window_uses_future_bars_and_excludes_entry_and_incomplete_tail() -> None:
    frame = _prices(5).assign(
        low=[1.0, 97.0, 93.0, 95.0, 96.0],
        high=[1000.0, 101.0, 105.0, 109.0, 103.0],
    )
    result = unwrap(mae_analysis.compute_mae)(frame, 2, "2024-01-01", "2024-01-02")
    assert result["mae_long"].tolist() == pytest.approx([0.07, 0.07, 0.05])
    assert result["mae_short"].tolist() == pytest.approx([0.05, 0.09, 0.09])
    assert result["mae_worst"].tolist() == pytest.approx([0.07, 0.09, 0.09])


def test_window_maturity_uses_only_the_selected_data() -> None:
    result = unwrap(mae_analysis.compute_mae)(
        _prices(10, frequency="D"), 3, "2024-01-03", "2024-01-04"
    )
    assert result.empty


def _check_app() -> None:
    from streamlit.testing.v1 import AppTest

    # Real parquet and app, isolated from the user's data and pytest's __main__.
    mae_analysis.DATA_PATH.parent.mkdir(parents=True)
    _prices(31, frequency="D").to_parquet(mae_analysis.DATA_PATH)
    app = AppTest.from_file(mae_analysis.__file__, default_timeout=15).run()
    assert not app.exception
    assert len(app.success) == 1
    assert len(app.metric) == 1
    assert len(app.dataframe) == 1
    assert len(app.get("plotly_chart")) == 3

    for holding in (31, 32):
        app.sidebar.slider[0].set_value(holding).run()
        assert not app.exception
        assert len(app.warning) == 1
        assert "完整持倉視窗" in app.warning[0].value
        assert not app.success
        assert not app.metric
        assert not app.dataframe
        assert not app.get("plotly_chart")

    app.sidebar.slider[0].set_value(30).run()
    assert not app.exception
    assert not app.warning
    assert len(app.success) == 1
    assert len(app.metric) == 1


def _check_empty_app() -> None:
    from streamlit.testing.v1 import AppTest

    mae_analysis.DATA_PATH.parent.mkdir(parents=True)
    _prices(0).to_parquet(mae_analysis.DATA_PATH)
    app = AppTest.from_file(mae_analysis.__file__, default_timeout=15).run()
    assert not app.exception
    assert len(app.warning) == 1
    assert "完整持倉視窗" in app.warning[0].value
    assert not app.sidebar.date_input
    assert not app.success
    assert not app.metric
    assert not app.dataframe
    assert not app.get("plotly_chart")


@pytest.mark.parametrize("check", ["_check_app", "_check_empty_app"])
def test_app_handles_data_availability(check: str) -> None:
    # Reuse the supervised runner's tested success/failure/timeout cleanup.
    with process_workspace(prefix="mae-app-test-", label="MAE AppTest") as root:
        temporary = root / "temporary"
        temporary.mkdir()
        result = run_captured(
            [
                sys.executable,
                "-B",
                "-c",
                f"from tests.test_mae_analysis import {check}; {check}()",
            ],
            cwd=root,
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                "TMPDIR": str(temporary),
                "MPLCONFIGDIR": str(root / "matplotlib"),
            },
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
