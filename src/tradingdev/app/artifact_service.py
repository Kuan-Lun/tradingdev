"""Application service for artifact metadata and content."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore


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
        self._store = store or SQLiteStore(self._workspace)

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
            result["content"] = path.read_text(encoding="utf-8")
        return result
