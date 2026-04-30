"""Application service for artifact metadata and content."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.backtest.pipeline_result import PipelineResult


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
        self._store = store or SQLiteStore(self._workspace)
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

    def promote_strategy(self, strategy_id: str) -> dict[str, Any]:
        """Promote a runnable generated strategy artifact."""
        return self._strategy_service.promote(strategy_id)
