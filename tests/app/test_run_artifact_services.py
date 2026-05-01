"""Run and artifact service tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app import job_store
from tradingdev.app.artifact_service import ArtifactService
from tradingdev.app.run_service import RunService
from tradingdev.domain.backtest.pipeline_result import PipelineResult

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


def test_run_service_compare_and_artifact_lookup(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    run_dir = workspace.runs / "run_a"
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "result.json"
    artifact_path.write_text('{"total_return": 0.1}', encoding="utf-8")
    store.create_run(
        run_id="run_a",
        job_id="job_a",
        strategy_id="fixture",
        artifact_dir=run_dir,
        metrics={"total_return": 0.1, "sharpe_ratio": 1.2},
    )
    store.create_run(
        run_id="run_b",
        job_id="job_b",
        strategy_id="fixture",
        artifact_dir=workspace.runs / "run_b",
        metrics={"total_return": 0.2, "sharpe_ratio": 1.5},
    )
    store.create_artifact(
        artifact_id="run_a:result_json",
        run_id="run_a",
        artifact_type="result_json",
        path=artifact_path,
        sha256=sha256_file(artifact_path),
        metadata={"job_id": "job_a"},
    )

    runs = RunService(workspace=workspace, store=store)
    artifacts = ArtifactService(workspace=workspace, store=store)

    comparison = runs.compare_runs(["run_a", "run_b"])
    assert comparison["success"] is True
    assert comparison["runs"][0]["metrics"]["total_return"] == 0.1
    artifact = artifacts.get_artifact("run_a:result_json", include_content=True)
    assert artifact["success"] is True
    assert artifact["content"] == '{"total_return": 0.1}'


def test_job_store_save_result_records_run_lineage(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    monkeypatch.setattr(job_store, "_WORKSPACE", workspace)
    monkeypatch.setattr(job_store, "_STORE", store)

    strategy_source = tmp_path / "fixture_strategy.py"
    strategy_source.write_text(
        "from tradingdev.domain.strategies.base import BaseStrategy\n"
        "class FixtureStrategy(BaseStrategy):\n"
        "    pass\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "fixture.yaml"
    config_path.write_text(
        f"""\
strategy:
  id: fixture
  class_name: FixtureStrategy
  source_path: "{strategy_source}"
  parameters:
    random_state: 7
backtest:
  symbol: BTC/USDT
  timeframe: 1h
  start_date: "2024-01-01"
  end_date: "2024-01-02"
""",
        encoding="utf-8",
    )

    created = job_store.create_job(
        job_id="job_lineage",
        strategy_name="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
        config_path=str(config_path),
    )
    job_store.update_job(
        "job_lineage",
        status="running_backtest",
        dataset_id="dataset-fixture",
    )

    pipeline = PipelineResult(mode="simple", config_snapshot={"strategy": {}})
    result_path = job_store.save_result(
        "job_lineage",
        {"total_return": 0.1},
        pipeline=pipeline,
    )

    assert created["created_at"]
    assert result_path == workspace.runs / "job_lineage" / "result.json"
    assert result_path.exists()
    assert (workspace.runs / "job_lineage" / "config.yaml").exists()
    assert (workspace.runs / "job_lineage" / "strategy.py").exists()
    assert (workspace.runs / "job_lineage" / "dataset_fingerprint.json").exists()
    assert (workspace.runs / "job_lineage" / "pipeline_result.pkl").exists()

    run = job_store.get_run("job_lineage")
    assert run is not None
    assert run["job_id"] == "job_lineage"
    assert run["strategy_id"] == "fixture"
    assert run["dataset_id"] == "dataset-fixture"
    assert run["artifact_dir"] == str(workspace.runs / "job_lineage")
    assert run["source_hash"] == sha256_file(strategy_source)
    assert run["random_seed"] == 7

    artifacts = {
        item["artifact_type"]: item for item in job_store.list_artifacts("job_lineage")
    }
    assert set(artifacts) == {
        "config_snapshot",
        "dataset_fingerprint",
        "pipeline_result",
        "result_json",
        "strategy_source",
    }
    assert artifacts["config_snapshot"]["metadata"]["config_hash"] == run["config_hash"]
    assert artifacts["strategy_source"]["metadata"]["source_path"] == str(
        strategy_source
    )
    assert artifacts["strategy_source"]["metadata"]["source_hash"] == run["source_hash"]
    assert artifacts["dataset_fingerprint"]["metadata"]["fingerprint"]

    loaded = ArtifactService(workspace=workspace, store=store).load_pipeline_result(
        "job_lineage"
    )
    assert loaded["success"] is True
    assert loaded["pipeline"].mode == "simple"


def test_artifact_service_cache_pipeline_records_run_lineage(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    monkeypatch.setenv("TRADINGDEV_DATA_ROOT", str(workspace.root / "data"))

    strategy_source = tmp_path / "cli_strategy.py"
    strategy_source.write_text("class CliStrategy: pass\n", encoding="utf-8")
    config_path = tmp_path / "cli.yaml"
    config_path.write_text(
        f"""\
strategy:
  id: cli_fixture
  source_path: "{strategy_source}"
  parameters:
    random_seed: 11
backtest:
  symbol: BTC/USDT
  timeframe: 1h
  start_date: "2024-01-01"
  end_date: "2024-01-02"
""",
        encoding="utf-8",
    )
    processed_path = tmp_path / "processed.parquet"
    processed_path.write_text("fixture", encoding="utf-8")

    service = ArtifactService(workspace=workspace, store=store)
    pipeline = PipelineResult(mode="simple", config_snapshot={})

    service.cache_pipeline_result(
        pipeline=pipeline,
        config_path=config_path,
        processed_path=processed_path,
        metrics={"total_return": 0.2},
        strategy_id="cli_fixture",
    )

    run = store.list_runs()[0]
    assert run["source_hash"] == sha256_file(strategy_source)
    assert run["random_seed"] == 11
