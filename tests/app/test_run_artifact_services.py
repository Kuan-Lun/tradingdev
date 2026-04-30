"""Run and artifact service tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.artifact_service import ArtifactService
from tradingdev.app.run_service import RunService

if TYPE_CHECKING:
    from pathlib import Path


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
