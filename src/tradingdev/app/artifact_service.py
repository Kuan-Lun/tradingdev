"""Application service for artifact metadata and content."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from tradingdev.adapters.storage.filesystem import (
    WorkspacePaths,
    sha256_file,
    sha256_text,
)
from tradingdev.adapters.storage.sqlite import SQLiteStore, get_sqlite_store
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.shared.utils.cache import cache_dir, compute_cache_key


class ArtifactService:
    """Read artifact metadata from SQLite and content from disk."""

    def __init__(
        self,
        *,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
        strategy_service: StrategyService | None = None,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._workspace.ensure()
        self._store = store or get_sqlite_store(self._workspace)
        self._strategy_service = strategy_service or StrategyService(self._workspace)

    def list_artifacts(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """List artifact metadata."""
        return self._store.list_artifacts(run_id)

    def get_artifact(
        self, artifact_id: str, *, include_content: bool = False
    ) -> dict[str, Any]:
        """Return artifact metadata and optional text content."""
        artifact = self._store.get_artifact(artifact_id)
        if artifact is None:
            return {"success": False, "error": f"Unknown artifact: {artifact_id}"}
        result: dict[str, Any] = {"success": True, "artifact": artifact}
        path = Path(str(artifact["path"]))
        if include_content:
            if not path.exists():
                return {"success": False, "error": f"Artifact file missing: {path}"}
            try:
                result["content"] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return {
                    "success": False,
                    "error": f"Artifact is not UTF-8 text: {artifact_id}",
                }
        return result

    def load_pipeline_result(self, run_id: str) -> dict[str, Any]:
        """Load a run's pickled PipelineResult artifact."""
        artifact = next(
            (
                item
                for item in self._store.list_artifacts(run_id)
                if item["artifact_type"] == "pipeline_result"
            ),
            None,
        )
        if artifact is None:
            return {
                "success": False,
                "error": f"No pipeline_result artifact for run: {run_id}",
            }
        path = Path(str(artifact["path"]))
        if not path.exists():
            return {"success": False, "error": f"Artifact file missing: {path}"}
        with path.open("rb") as handle:
            pipeline = pickle.load(handle)  # noqa: S301
        if not isinstance(pipeline, PipelineResult):
            return {
                "success": False,
                "error": f"Artifact is not a PipelineResult: {artifact['artifact_id']}",
            }
        return {
            "success": True,
            "artifact": artifact,
            "pipeline": pipeline,
        }

    def cache_pipeline_result(
        self,
        *,
        pipeline: PipelineResult,
        config_path: Path,
        processed_path: Path,
        metrics: dict[str, Any],
        strategy_id: str,
    ) -> Path:
        """Persist a CLI pipeline result and track it as a SQLite artifact."""
        key = compute_cache_key(config_path, processed_path)
        directory = cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        cache_path = directory / f"{key}.pkl"
        with cache_path.open("wb") as handle:
            pickle.dump(pipeline, handle)

        run_id = f"cli_{key}"
        config_hash = sha256_file(config_path) if config_path.exists() else None
        dataset_id = (
            sha256_file(processed_path)
            if processed_path.exists()
            else sha256_text(str(processed_path))
        )
        self._store.create_run(
            run_id=run_id,
            job_id=run_id,
            strategy_id=strategy_id,
            artifact_dir=directory,
            metrics=metrics,
            config_hash=config_hash,
            dataset_id=dataset_id,
        )
        self._store.create_artifact(
            artifact_id=f"{run_id}:pipeline_result",
            run_id=run_id,
            artifact_type="pipeline_result",
            path=cache_path,
            sha256=sha256_file(cache_path),
            metadata={
                "source": "cli_cache",
                "config_path": str(config_path),
                "processed_path": str(processed_path),
                "cache_key": key,
            },
        )
        return cache_path

    def promote_strategy(self, strategy_id: str) -> dict[str, Any]:
        """Promote a runnable generated strategy artifact."""
        return self._strategy_service.promote(strategy_id)
