"""Tests for scripts/validate_run.py on real engine results.

Every case runs the real signal or volume engine on a few synthetic bars and
checks a pickle written to pytest's temporary directory, so the checks see
the same metrics and metadata a saved run has.
"""

import copy
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.backtest.volume_engine import VolumeBacktestEngine

if TYPE_CHECKING:
    from types import ModuleType

    from tradingdev.domain.backtest.result import BacktestResult

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_run.py"
# The review's six bars: two closed trades and one still open at the end.
_PRICES = [100.0, 110.0, 100.0, 90.0, 100.0, 110.0]
_OPEN_TRADE = [1, 0, 1, 0, 1, 1]


@pytest.fixture(scope="module")
def validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_run", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bars(
    signals: list[int], prices: list[float] | None = None, freq: str = "D"
) -> pd.DataFrame:
    prices = prices or _PRICES
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=len(prices), freq=freq, tz="UTC"
            ),
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "signal": signals,
        }
    )


def _signal_run(
    signals: list[int],
    prices: list[float] | None = None,
    *,
    freq: str = "D",
    engine_freq: str | None = None,
    **engine: Any,
) -> PipelineResult:
    settings: dict[str, Any] = {
        "init_cash": 1000.0,
        "fees": 0.0,
        "slippage": 0.0,
        "freq": engine_freq or ("1d" if freq == "D" else "1h"),
        "signal_as_position": True,
        "periods_per_year": 365.0,
    } | engine
    result = SignalBacktestEngine(**settings).run(_bars(signals, prices, freq))
    return PipelineResult(
        mode="simple",
        backtest_result=result,
        config_snapshot={
            "backtest": {
                "init_cash": settings["init_cash"],
                "fees": settings["fees"],
                "slippage": settings["slippage"],
            }
        },
    )


def _volume_run() -> PipelineResult:
    result = VolumeBacktestEngine(
        fees=0.0,
        slippage=0.0,
        freq="1d",
        signal_as_position=True,
        periods_per_year=365.0,
    ).run(_bars(_OPEN_TRADE))
    return PipelineResult(
        mode="simple",
        backtest_result=result,
        config_snapshot={"backtest": {"mode": "volume", "fees": 0.0}},
    )


def _check(
    validator: ModuleType, run: PipelineResult, tmp_path: Path
) -> tuple[int, dict[str, tuple[str, str]]]:
    """Validate a pickled run; map each check's code to (status, detail)."""
    path = tmp_path / "pipeline_result.pkl"
    path.write_bytes(pickle.dumps(run))
    code = validator.main(str(path))
    return code, {
        name.split()[0]: (status, detail) for status, name, detail in validator.results
    }


def _statuses(checks: dict[str, tuple[str, str]], status: str) -> set[str]:
    return {code for code, (found, _) in checks.items() if found == status}


def _result(run: PipelineResult) -> BacktestResult:
    assert run.backtest_result is not None
    return run.backtest_result


class TestUnavailableMetrics:
    """Item 2: legitimate missing values are N/A with the engine's reason."""

    def test_zero_trades_complete_the_report(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        # abs(None) used to stop the report with a TypeError at A4.
        code, checks = _check(validator, _signal_run([0] * 6), tmp_path)

        assert code == 0
        assert _statuses(checks, "FAIL") == set()
        assert checks["A4"] == ("N/A", "引擎無值，原因 no_trades")
        assert checks["A5"] == ("N/A", "引擎無值，原因 no_trades")
        assert checks["V2"] == ("N/A", "引擎無值，原因 no_trades")
        assert checks["T6"] == ("N/A", "引擎無值，原因 zero_denominator")

    def test_all_winning_trades_have_an_unbounded_profit_factor(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        run = _signal_run(
            [1, 1, 0, 0, 0, 0], [100.0, 110.0, 120.0, 120.0, 120.0, 120.0]
        )
        assert _result(run).metric_metadata["unavailable"]["profit_factor"] == (
            "unbounded"
        )

        code, checks = _check(validator, run, tmp_path)

        assert code == 0
        assert checks["A5"] == ("N/A", "引擎無值，原因 unbounded")
        assert checks["A4"][0] == "PASS"

    def test_missing_annualization_leaves_sharpe_unverified(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        run = _signal_run(_OPEN_TRADE, periods_per_year=None)

        code, checks = _check(validator, run, tmp_path)

        assert code == 0
        assert checks["T6"] == ("N/A", "引擎無值，原因 missing_annualization")

    def test_volume_mode_return_metrics_are_not_applicable(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        code, checks = _check(validator, _volume_run(), tmp_path)

        assert code == 0
        for name in ("A6", "A7", "T6", "V3"):
            assert checks[name] == ("N/A", "引擎無值，原因 not_applicable"), name
        assert checks["A1"][0] == "PASS"
        assert checks["T3"][0] == "PASS"


class TestDerivedReasons:
    """Reasons outside trade statistics are derived from the saved run too."""

    @pytest.mark.parametrize(
        ("frequency", "reason"),
        [("1w", "unsupported_daily_sampling"), ("1x", "unknown_bar_frequency")],
    )
    def test_daily_metrics_without_daily_sampling_are_not_applicable(
        self, validator: ModuleType, tmp_path: Path, frequency: str, reason: str
    ) -> None:
        # Weekly bars: the engine reports no daily or monthly P&L statistics.
        run = _signal_run(_OPEN_TRADE, freq="W-MON", engine_freq=frequency)

        code, checks = _check(validator, run, tmp_path)

        assert code == 0, checks
        for name in ("T3", "T4", "T6"):
            assert checks[name] == ("N/A", f"引擎無值，原因 {reason}"), name

    def test_nonpositive_capital_leaves_return_metrics_unavailable(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        # A short position through a tripling price takes equity below zero.
        run = _signal_run(
            [-1, -1, -1, 0, 0, 0], [100.0, 100.0, 300.0, 300.0, 300.0, 300.0]
        )
        assert min(_result(run).equity_curve) < 0

        code, checks = _check(validator, run, tmp_path)

        assert code == 0, checks
        for name in ("A6", "A7", "T6"):
            assert checks[name] == ("N/A", "引擎無值，原因 nonpositive_capital"), name

    @pytest.mark.parametrize(
        ("metric", "reason", "failing"),
        [
            ("sharpe_ratio", "missing_annualization", "T6"),
            ("max_drawdown", "no_trades", "A6"),
            ("daily_pnl_std", "unsupported_daily_sampling", "T3"),
        ],
    )
    def test_reason_the_saved_run_contradicts_fails(
        self,
        validator: ModuleType,
        tmp_path: Path,
        metric: str,
        reason: str,
        failing: str,
    ) -> None:
        run = _signal_run(_OPEN_TRADE)
        _result(run).metrics[metric] = None
        _result(run).metric_metadata["unavailable"][metric] = reason

        code, checks = _check(validator, run, tmp_path)

        assert code == 1
        assert checks[failing][0] == "FAIL"
        assert reason in checks[failing][1]

    def test_run_without_settings_keeps_sharpe_not_applicable_in_volume_mode(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        # Older runs kept no settings; volume mode still has no Sharpe.
        run = _volume_run()
        del _result(run).metric_metadata["settings"]

        _, checks = _check(validator, run, tmp_path)

        assert checks["T6"] == ("N/A", "引擎無值，原因 not_applicable")

    def test_run_without_settings_reads_the_configured_timeframe(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        # BacktestConfig names the bar frequency ``timeframe``.
        run = _signal_run(_OPEN_TRADE, freq="W-MON", engine_freq="1w")
        del _result(run).metric_metadata["settings"]
        assert run.config_snapshot is not None
        run.config_snapshot["backtest"]["timeframe"] = "1w"

        _, checks = _check(validator, run, tmp_path)

        for name in ("T3", "T4", "T5"):
            assert checks[name] == (
                "N/A",
                "引擎無值，原因 unsupported_daily_sampling",
            ), name

    def test_declared_ddof_other_than_the_catalog_fails(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        run = _signal_run(_OPEN_TRADE)
        _result(run).metric_metadata["settings"]["period_pnl_std_ddof"] = 1

        code, checks = _check(validator, run, tmp_path)

        assert code == 1
        assert checks["T3"][0] == "FAIL"
        assert "period_pnl_std_ddof=1" in checks["T3"][1]


class TestClosedTradeScope:
    """Item 3: trade statistics count closed trades; the P&L sum all trades."""

    def test_open_trade_is_left_out_of_closed_only_metrics(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        run = _signal_run(_OPEN_TRADE)
        assert [t["status"] for t in _result(run).trades] == [
            "closed",
            "closed",
            "open",
        ]

        code, checks = _check(validator, run, tmp_path)

        assert code == 0
        assert checks["A3"] == ("PASS", "2 vs 2（未平倉 1 筆不計入）")
        assert checks["A4"] == ("PASS", "報告 0.5000 vs 重算 0.5000")
        assert checks["A5"][0] == "PASS"
        assert checks["A2"][0] == "PASS"

    def test_reconciliation_includes_the_open_trade_pnl(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        # One more bar moves the open position, so its marked-to-market P&L
        # is needed for the trades to add up to the equity change.
        run = _signal_run(_OPEN_TRADE + [1], _PRICES + [121.0])
        trades = _result(run).trades
        open_pnl = sum(t["net_pnl"] for t in trades if t["status"] == "open")
        assert open_pnl != 0.0

        code, checks = _check(validator, run, tmp_path)

        assert code == 0
        assert checks["A2"][0] == "PASS"
        assert checks["A3"][0] == "PASS"

    def test_closed_count_mismatch_still_fails(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        run = _signal_run(_OPEN_TRADE)
        _result(run).metrics["total_trades"] = 3

        code, checks = _check(validator, run, tmp_path)

        assert code == 1
        assert checks["A3"][0] == "FAIL"


class TestEngineDefinitions:
    """Item 4: daily volatility and Sharpe follow the engine's definitions."""

    def test_untampered_open_trade_run_passes_t3_and_t6(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        # Previously 55.4534 vs 67.9100 and 1.0036 vs a daily 0.0000.
        code, checks = _check(validator, _signal_run(_OPEN_TRADE), tmp_path)

        assert code == 0
        assert checks["T3"][0] == "PASS"
        assert checks["T6"][0] == "PASS"
        assert checks["T6"][1].startswith("報告 1.0036 vs 重算 1.0036")

    def test_other_annualization_and_risk_free_rate(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        run = _signal_run(_OPEN_TRADE, periods_per_year=252.0, risk_free_rate=0.05)

        code, checks = _check(validator, run, tmp_path)

        assert code == 0
        assert checks["T6"][0] == "PASS"
        assert "periods_per_year=252, risk_free_rate=0.05" in checks["T6"][1]

    def test_hourly_bars_with_costs_measure_from_initial_capital(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        # Four days of hourly bars with costs: daily and bar statistics
        # differ, and the first day already moves away from the capital.
        prices = [100.0 + 5.0 * ((i * 7) % 11) for i in range(96)]
        signals = [1 if (i // 5) % 2 == 0 else 0 for i in range(96)]
        run = _signal_run(signals, prices, freq="h", fees=0.001)
        assert abs(_result(run).equity_curve[23] - 1000.0) > 1.0

        code, checks = _check(validator, run, tmp_path)

        assert code == 0, checks
        for name in ("A1", "T3", "T4", "T5", "T6"):
            assert checks[name][0] == "PASS", (name, checks[name])

    @pytest.mark.parametrize(
        ("metric", "factor", "failing"),
        [
            ("daily_pnl_std", 24**0.5, "T3"),
            ("sharpe_ratio", 1.1, "T6"),
            ("monthly_pnl_mean", 0.5, "T4"),
        ],
    )
    def test_tampered_values_fail(
        self,
        validator: ModuleType,
        tmp_path: Path,
        metric: str,
        factor: float,
        failing: str,
    ) -> None:
        run = _signal_run(_OPEN_TRADE)
        _result(run).metrics[metric] *= factor

        code, checks = _check(validator, run, tmp_path)

        assert code == 1
        assert _statuses(checks, "FAIL") == {failing}

    @pytest.mark.parametrize(
        ("metric", "failing"),
        [
            ("win_rate", "A4"),
            ("profit_factor", "A5"),
            ("max_drawdown", "A6"),
            ("total_return", "A7"),
            ("daily_pnl_mean", "T5"),
        ],
    )
    def test_one_percent_tampering_fails(
        self,
        validator: ModuleType,
        tmp_path: Path,
        metric: str,
        failing: str,
    ) -> None:
        # A 2% tolerance used to let these through; the values are exact.
        run = _signal_run(_OPEN_TRADE)
        _result(run).metrics[metric] *= 1.01

        code, checks = _check(validator, run, tmp_path)

        assert code == 1
        assert _statuses(checks, "FAIL") == {failing}

    def test_calendar_gaps_only_warn(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        # Business-day bars skip the weekend, which the engine allows.
        run = _signal_run(_OPEN_TRADE * 2, _PRICES * 2, freq="B", engine_freq="1d")

        code, checks = _check(validator, run, tmp_path)

        assert code == 0, checks
        assert checks["S4"][0] == "WARN"
        assert checks["S5"][0] == "WARN"
        for name in ("T1", "T3", "T5", "T6"):
            assert checks[name][0] == "PASS", (name, checks[name])


class TestCorruptRecords:
    """Missing fields and corrupt records fail a check instead of crashing."""

    def test_none_without_a_reason_fails(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        run = _signal_run([0] * 6)
        del _result(run).metric_metadata["unavailable"]["win_rate"]

        code, checks = _check(validator, run, tmp_path)

        assert code == 1
        assert checks["A4"] == (
            "FAIL",
            "資料缺漏或損壞：win_rate 為 None，但 metadata 沒有原因",
        )

    def test_reason_that_contradicts_the_trades_fails(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        run = _signal_run(
            [1, 1, 0, 0, 0, 0], [100.0, 110.0, 120.0, 120.0, 120.0, 120.0]
        )
        _result(run).metric_metadata["unavailable"]["profit_factor"] = (
            "zero_denominator"
        )

        code, checks = _check(validator, run, tmp_path)

        assert code == 1
        assert checks["A5"] == (
            "FAIL",
            "引擎原因 zero_denominator，但原始資料推得 unbounded",
        )

    def test_missing_metric_and_trade_field_fail(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        run = _signal_run(_OPEN_TRADE)
        del _result(run).metrics["total_trades"]
        trades = copy.deepcopy(_result(run).trades)
        del trades[0]["net_pnl"]
        _result(run).trades = trades

        code, checks = _check(validator, run, tmp_path)

        assert code == 1
        assert checks["A3"] == ("FAIL", "資料缺漏或損壞：metrics 缺少 total_trades")
        assert checks["A2"][0] == "FAIL"
        assert checks["V5"][0] == "FAIL"

    def test_duplicate_timestamps_fail_without_a_crash(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        run = _signal_run(_OPEN_TRADE)
        stamps = _result(run).timestamps
        assert stamps is not None
        _result(run).timestamps = stamps[[0, 0, 0, 0, 0, 0]]

        code, checks = _check(validator, run, tmp_path)

        assert code == 1
        assert checks["S3"][0] == "FAIL"
        assert checks["S4"][0] == "FAIL"

    def test_non_mapping_trade_fails_without_a_crash(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        run = _signal_run(_OPEN_TRADE)
        _result(run).trades = [None]  # type: ignore[list-item]

        code, checks = _check(validator, run, tmp_path)

        assert code == 1
        assert checks["A3"][0] == "FAIL"

    def test_unreadable_equity_fails_without_a_crash(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        run = _signal_run(_OPEN_TRADE)
        _result(run).equity_curve = None  # type: ignore[assignment]

        code, checks = _check(validator, run, tmp_path)

        assert code == 1
        assert checks["X1"][0] == "FAIL"

    def test_missing_explicit_file_fails(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        assert validator.main(str(tmp_path / "missing.pkl")) == 1
        assert validator.results[0][:2] == ("FAIL", "L1 讀取 pickle")

    def test_simple_mode_without_a_backtest_result_is_not_checked(
        self, validator: ModuleType, tmp_path: Path
    ) -> None:
        path = tmp_path / "pipeline_result.pkl"
        path.write_bytes(pickle.dumps(SimpleNamespace(mode="simple")))

        assert validator.main(str(path)) == 2

    def test_unreadable_file_fails(self, validator: ModuleType, tmp_path: Path) -> None:
        path = tmp_path / "broken.pkl"
        path.write_bytes(b"not a pickle")

        assert validator.main(str(path)) == 1
        assert validator.results[0][:2] == ("FAIL", "L1 讀取 pickle")


@pytest.mark.parametrize(
    ("signals", "corrupt", "exit_code"),
    [([0] * 6, False, 0), (_OPEN_TRADE, True, 1)],
    ids=["zero-trades", "corrupt"],
)
def test_script_exits_without_a_traceback(
    tmp_path: Path, signals: list[int], corrupt: bool, exit_code: int
) -> None:
    run = _signal_run(signals)
    if corrupt:
        _result(run).metrics["win_rate"] = None
    path = tmp_path / "pipeline_result.pkl"
    path.write_bytes(pickle.dumps(run))

    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(path)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert completed.returncode == exit_code, completed.stdout + completed.stderr
    assert "Traceback" not in completed.stderr


def _run_script(cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_without_a_path_the_last_run_by_name_wins_over_the_cache(
    tmp_path: Path,
) -> None:
    good = pickle.dumps(_signal_run([0] * 6))
    files = {
        "workspace/runs/a/pipeline_result.pkl": b"not a pickle",
        "workspace/runs/b/pipeline_result.pkl": good,
        "workspace/data/processed/cache/z.pkl": b"not a pickle",
    }
    for name, content in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(content)

    completed = _run_script(tmp_path)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "workspace/runs/b/pipeline_result.pkl" in completed.stdout


def test_without_a_path_the_cache_is_used_when_there_is_no_run(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "workspace" / "data" / "processed" / "cache"
    cache.mkdir(parents=True)
    (cache / "a.pkl").write_bytes(b"not a pickle")
    (cache / "b.pkl").write_bytes(pickle.dumps(_signal_run([0] * 6)))

    completed = _run_script(tmp_path)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "cache/b.pkl" in completed.stdout


def test_without_a_path_and_no_file_exits_2(tmp_path: Path) -> None:
    completed = _run_script(tmp_path)

    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
