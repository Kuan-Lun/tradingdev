"""Optimization workers execute only the accepted, verified specification."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest

from tradingdev.adapters.execution.process_runner import WorkerHandle
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService, LoadedDataset
from tradingdev.app.job_store import JobStore
from tradingdev.domain.execution import ExecutionManifest, OptimizationSpec
from tradingdev.domain.strategies.bundled.kd_strategy.config import KDStrategyConfig
from tradingdev.domain.strategies.bundled.kd_strategy.strategy import KDStrategy
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.mcp.workers import optimization
from tradingdev.mcp.workers.optimization import _run_optimization

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
    from tradingdev.domain.backtest.schemas import BacktestConfig, ParallelConfig


def _queued_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[JobStore, ExecutionManifest]:
    from tradingdev.domain.backtest.schemas import BacktestConfig

    workspace = WorkspacePaths(tmp_path / "workspace")
    store = JobStore(workspace=workspace)
    monkeypatch.setenv("TRADINGDEV_WORKSPACE", str(workspace.root))
    monkeypatch.setenv(
        "TRADINGDEV_WORKER_IDENTITY",
        json.dumps(WorkerHandle(4321, 100.0, "a" * 32).job_fields()),
    )
    config: dict[str, Any] = {
        "strategy": {"id": "fixture"},
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-07T23:59:59.999999",
            "init_cash": 10000,
        },
    }
    config["data"] = DataService(workspace).execution_config(
        config, BacktestConfig(**config["backtest"])
    )
    manifest = ExecutionManifest.create(
        kind="optimization",
        strategy_execution=StrategyExecution(
            kind="generated", constructor_kwargs={"direction": -1}
        ),
        config=config,
        optimization=OptimizationSpec.model_validate(
            {
                "param_ranges": {"direction": [-1, 1]},
                "optimization_metric": "total_return",
                "train_start": "2024-01-01",
                "train_end": "2024-01-03",
                "test_start": "2024-01-04",
                "test_end": "2024-01-07",
            }
        ),
    )
    store.create_job(job_id="fixture", job_type="optimization", manifest=manifest)
    return store, manifest


@pytest.mark.parametrize("corruption", ["missing", "search", "kind"])
def test_invalid_optimization_manifest_fails_before_data_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    store, _ = _queued_job(tmp_path, monkeypatch)
    path = store.workspace.runs / "fixture" / "manifest.json"
    if corruption == "missing":
        path.unlink()
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if corruption == "search":
            payload["optimization"]["param_ranges"] = {"direction": [99]}
        else:
            payload["kind"] = "backtest"
        path.write_text(json.dumps(payload), encoding="utf-8")
    data_calls: list[dict[str, Any]] = []

    def unexpected_data_load(
        self: DataService, config: dict[str, Any], bt: BacktestConfig
    ) -> LoadedDataset:
        data_calls.append(config)
        raise AssertionError("Invalid manifest reached data loading")

    monkeypatch.setattr(DataService, "load", unexpected_data_load)
    _run_optimization("fixture")

    job = store.get_job("fixture")
    assert job is not None
    assert job["status"] == "failed"
    assert "Execution manifest error" in job["error"]
    assert data_calls == []
    assert not (path.parent / "result.json").exists()


def test_optimization_worker_ignores_mutable_config_and_search_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, manifest = _queued_job(tmp_path, monkeypatch)
    (store.workspace.runs / "fixture" / "config.yaml").write_text(
        "invalid: config projection changed after submission\n", encoding="utf-8"
    )
    store.update_job(
        "fixture",
        param_ranges={"direction": [99]},
        optimization_metric="profit_factor",
        confirmed=True,
    )
    captured: list[dict[str, Any]] = []
    evaluations: list[tuple[dict[str, Any], str, str, str, dict[str, Any]]] = []

    def capture_data_load(
        self: DataService, config: dict[str, Any], bt: BacktestConfig
    ) -> LoadedDataset:
        captured.append(config)
        return LoadedDataset(
            frame=pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2024-01-01", periods=7, freq="D", tz="UTC"
                    )
                }
            ),
            processed_path=tmp_path / "unused.parquet",
            dataset_id="fixture-data",
        )

    def evaluate(
        strategy_cfg: dict[str, Any],
        strategy_execution: StrategyExecution,
        bt_cfg: BacktestConfig,
        frame: pd.DataFrame,
        params: dict[str, Any],
        metric: str,
        parallel_cfg: ParallelConfig,
    ) -> tuple[dict[str, Any], float, dict[str, Any]]:
        assert strategy_execution == manifest.strategy_execution
        evaluations.append(
            (
                params,
                metric,
                bt_cfg.start_date.date().isoformat(),
                bt_cfg.end_date.date().isoformat(),
                parallel_cfg.model_dump(),
            )
        )
        return params, 1.0, {metric: 1.0}

    # Stub only external data/strategy evaluation. The real worker still loads
    # the manifest, expands the grid, selects its winner, performs OOS and saves.
    monkeypatch.setattr(BacktestService, "prepare_strategy", lambda *args: None)
    monkeypatch.setattr(DataService, "load", capture_data_load)
    monkeypatch.setattr(optimization, "_run_single_combo", evaluate)
    monkeypatch.setattr(optimization, "estimate_n_jobs", lambda *args, **kwargs: 1)
    _run_optimization("fixture")

    assert captured == [manifest.config_copy()]
    assert [item[:4] for item in evaluations] == [
        ({"direction": -1}, "total_return", "2024-01-01", "2024-01-03"),
        ({"direction": 1}, "total_return", "2024-01-01", "2024-01-03"),
        ({"direction": -1}, "total_return", "2024-01-04", "2024-01-07"),
    ]
    assert all(item[4] == manifest.config["parallel"] for item in evaluations)
    job = store.get_job("fixture")
    assert job is not None
    assert job["status"] == "done"
    assert job["best_params"] == {"direction": -1}
    assert store.load_manifest("fixture") == manifest


def test_optimization_evaluations_keep_unsearched_strategy_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tradingdev.domain.backtest.schemas import BacktestConfig, ParallelConfig

    strategy_config = {"id": "kd_crossover", "parameters": {"oversold": 15.0}}
    loader = StrategyLoader()
    execution = loader.resolve_execution(strategy_config)
    captured: list[dict[str, Any]] = []

    class ChangedDefaultConfig(KDStrategyConfig):
        k_period: int = 21

    monkeypatch.setattr(
        StrategyLoader,
        "_bundled_config_model",
        lambda *args, **kwargs: ChangedDefaultConfig,
    )

    def capture_signals(self: KDStrategy, frame: pd.DataFrame) -> pd.DataFrame:
        captured.append(self.get_parameters())
        return frame.copy()

    engine = cast(
        "BaseBacktestEngine",
        SimpleNamespace(
            run=lambda frame: SimpleNamespace(metrics={"total_return": 1.0})
        ),
    )
    monkeypatch.setattr(BacktestService, "prepare_strategy", lambda *args: None)
    monkeypatch.setattr(BacktestService, "create_engine", lambda *args: engine)
    monkeypatch.setattr(KDStrategy, "generate_signals", capture_signals)
    config = BacktestConfig(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-07",
        init_cash=10000,
    )
    frame = pd.DataFrame({"close": [1.0, 2.0]})
    parallel = ParallelConfig()
    optimization._run_single_combo(
        strategy_config,
        execution,
        config,
        frame,
        {"d_period": 5},
        "total_return",
        parallel,
    )
    optimization._evaluate_combo(
        strategy_config,
        execution,
        config.model_dump(),
        frame.to_json(orient="split"),
        {"d_period": 7},
        "total_return",
        parallel.model_dump(),
    )

    assert [parameters["k_period"] for parameters in captured] == [14, 14]
    assert [parameters["d_period"] for parameters in captured] == [5, 7]
    assert [parameters["oversold"] for parameters in captured] == [15.0, 15.0]
