"""Independent generated-strategy semantics check, executed as a subprocess."""

from __future__ import annotations

import hashlib
import math
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

from tests.e2e.strategy_scenarios import SCENARIOS, expected_signals, market_frame
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.artifact_service import ArtifactService
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.run_service import RunService
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.shared.utils.config import load_config


def verify(root: Path) -> None:
    """Verify runnable state, configurable periods and independent expected signals."""
    workspace = WorkspacePaths(root / "workspace")
    service = StrategyService(workspace)
    spec = service.resolve_executable("codex_sma_integration")
    assert spec.status.value == "runnable"
    assert Path(spec.source_path).is_relative_to(workspace.generated_strategies)
    assert Path(spec.config_path).is_relative_to(workspace.configs)
    raw_config = load_config(Path(spec.config_path))
    parameters = raw_config["strategy"]["parameters"]
    assert parameters["fast_period"] == 5
    assert parameters["slow_period"] == 20

    close = pd.Series([*range(1, 41), *range(40, 0, -1), *([1.0] * 30)], dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(close), freq="h"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100.0,
        }
    )
    before = frame.copy(deep=True)
    for fast_period, slow_period in ((5, 20), (3, 8)):
        config = deepcopy(raw_config)
        config["strategy"]["parameters"].update(
            fast_period=fast_period, slow_period=slow_period
        )
        strategy = StrategyLoader(workspace_root=workspace.root).create_from_config(
            config, engine=None
        )
        result = strategy.generate_signals(frame)
        fast = close.rolling(fast_period).mean()
        slow = close.rolling(slow_period).mean()
        expected = pd.Series(0, index=frame.index, name="signal")
        expected.loc[fast > slow] = 1
        expected.loc[fast < slow] = -1
        pd.testing.assert_series_equal(result["signal"], expected, check_dtype=False)
        pd.testing.assert_frame_equal(frame, before)


def verify_workflow(root: Path, scenario_name: str) -> None:
    """Compare generated signals and stored backtest metrics with independent inputs."""
    scenario = SCENARIOS[scenario_name]
    workspace = WorkspacePaths(root / "workspace")
    spec = StrategyService(workspace).resolve_executable(scenario.strategy_id)
    assert spec.status.value == "runnable"
    config = load_config(Path(spec.config_path))
    assert config["strategy"]["parameters"] == scenario.parameters
    bt = BacktestConfig(**config["backtest"])
    assert (bt.symbol, bt.timeframe, bt.mode) == ("BTC/USDT", "1h", "signal")
    assert str(bt.start_date.date()) == "2024-01-01"
    assert str(bt.end_date.date()) == "2024-01-08"
    assert bt.init_cash == 10000 and bt.fees == 0 and bt.slippage == 0
    assert bt.random_seed == 42
    frame = market_frame()
    original = frame.copy(deep=True)
    for parameters in (scenario.parameters, scenario.overrides):
        overridden = deepcopy(config)
        overridden["strategy"]["parameters"] = parameters
        strategy = StrategyLoader(workspace_root=workspace.root).create_from_config(
            overridden, engine=None
        )
        actual = strategy.generate_signals(frame)
        expected = expected_signals(frame, scenario, parameters)
        pd.testing.assert_series_equal(actual["signal"], expected, check_dtype=False)
        pd.testing.assert_frame_equal(frame, original)
        # Prefixes include warmup, a reversal, and a flat interval. A strategy
        # cannot use future rows to rewrite the earlier signal history.
        for length in (4, 24, 55, 92):
            prefix = frame.iloc[:length].copy()
            result = strategy.generate_signals(prefix)
            pd.testing.assert_series_equal(
                result["signal"], expected.iloc[:length], check_dtype=False
            )
            pd.testing.assert_frame_equal(prefix, original.iloc[:length])

    runs = RunService(workspace=workspace).list_runs()
    assert len(runs) == 1, runs
    run = runs[0]
    assert run["strategy_id"] == scenario.strategy_id
    # Ordinary backtests treat end_date as a timestamp (midnight here).
    selected = frame.loc[
        frame["timestamp"] <= pd.Timestamp("2024-01-08", tz="UTC")
    ].copy()
    selected["signal"] = expected_signals(selected, scenario, scenario.parameters)
    reference = BacktestService().create_engine(bt).run(selected).metrics
    assert reference["total_trades"] > 0
    for metric in ("total_return", "total_trades", "max_drawdown"):
        assert math.isclose(
            run["metrics"][metric], reference[metric], rel_tol=1e-9, abs_tol=1e-9
        ), (metric, run["metrics"][metric], reference[metric])
    artifacts = ArtifactService(workspace=workspace).list_artifacts(run["run_id"])
    by_type = {artifact["artifact_type"]: artifact for artifact in artifacts}
    assert {
        "strategy_source",
        "config_snapshot",
        "result_json",
        "dataset_fingerprint",
    } <= by_type.keys()
    for artifact in artifacts:
        path = Path(artifact["path"])
        assert path.is_relative_to(workspace.runs)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    assert (
        Path(by_type["strategy_source"]["path"]).read_bytes()
        == Path(spec.source_path).read_bytes()
    )
    snapshot = load_config(Path(by_type["config_snapshot"]["path"]))
    assert snapshot["strategy"]["parameters"] == scenario.parameters


if __name__ == "__main__":
    root = Path(sys.argv[1])
    (root / "verifier.ready").write_text("ready", encoding="utf-8")
    if len(sys.argv) > 2:
        verify_workflow(root, sys.argv[2])
    else:
        verify(root)
