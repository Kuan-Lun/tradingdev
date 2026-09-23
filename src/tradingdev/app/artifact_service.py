"""Application service for artifact metadata and content."""

from __future__ import annotations

import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from tradingdev.adapters.storage.filesystem import (
    WorkspacePaths,
    sha256_file,
    sha256_text,
)
from tradingdev.adapters.storage.sqlite import SQLiteStore, get_sqlite_store
from tradingdev.app.run_lineage import (
    extract_random_seed,
    read_strategy_snapshot,
)
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.shared.utils.cache import cache_dir, compute_cache_key


class ArtifactService:
    """Read artifact metadata from SQLite and content from disk."""

    def __init__(
        self,
        *,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._workspace.ensure()
        self._store = store or get_sqlite_store(self._workspace)

    def list_artifacts(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """List artifact metadata."""
        return self._store.list_artifacts(run_id)

    def get_artifact(
        self, artifact_id: str, *, include_content: bool = False
    ) -> dict[str, Any]:
        """Return artifact metadata and optional text content."""
        artifact = self._store.get_artifact(artifact_id)
        if artifact is None:
            return {
                "success": False,
                "error": f"Unknown artifact: {artifact_id}",
                "code": "artifact_not_found",
            }
        result: dict[str, Any] = {"success": True, "artifact": artifact}
        path = Path(str(artifact["path"]))
        if include_content:
            if not path.exists():
                return {
                    "success": False,
                    "error": f"Artifact file missing: {path}",
                    "code": "artifact_file_missing",
                }
            try:
                result["content"] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return {
                    "success": False,
                    "error": f"Artifact is not UTF-8 text: {artifact_id}",
                    "code": "artifact_not_text",
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
        config_payload = pipeline.config_snapshot
        source = read_strategy_snapshot(
            config_payload, self._workspace, strategy_id=strategy_id
        )
        disk_content = config_path.read_bytes()
        disk_config = yaml.safe_load(disk_content)
        executed_config = deepcopy(config_payload)
        for config in (disk_config, executed_config):
            if isinstance(config, dict) and isinstance(config.get("strategy"), dict):
                config["strategy"].pop("source_hash", None)
        if disk_config != executed_config:
            msg = "Config changed after the CLI run; result cannot be cached"
            raise ValueError(msg)
        key = compute_cache_key(
            config_path, processed_path, config_content=disk_content
        )
        directory = cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        cache_path = directory / f"{key}.pkl"
        with cache_path.open("wb") as handle:
            pickle.dump(pipeline, handle)

        run_id = f"cli_{key}"
        config_hash = sha256_text(
            yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True)
        )
        dataset_id = (
            sha256_file(processed_path)
            if processed_path.exists()
            else sha256_text(str(processed_path))
        )
        self._store.create_run(
            run_id=run_id,
            job_id=run_id,
            strategy_id=strategy_id,
            revision_id=source.revision_id,
            artifact_dir=directory,
            metrics=metrics,
            config_hash=config_hash,
            source_hash=source.source_hash,
            random_seed=extract_random_seed(config_payload),
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
