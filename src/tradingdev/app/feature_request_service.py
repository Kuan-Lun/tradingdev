"""Application service for unsupported capability requests."""

from __future__ import annotations

from uuid import uuid4

from tradingdev.adapters.storage.filesystem import (
    WorkspacePaths,
    now_iso,
    sha256_file,
    write_json,
)
from tradingdev.adapters.storage.sqlite import SQLiteStore


class FeatureRequestService:
    """Persist feature requests as workspace artifacts."""

    def __init__(
        self,
        *,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._workspace.ensure()
        self._store = store or SQLiteStore(self._workspace)

    def record(
        self,
        *,
        title: str,
        description: str,
        source_tool: str = "",
    ) -> dict[str, str | bool]:
        """Write a feature request artifact."""
        request_id = uuid4().hex[:12]
        path = self._workspace.feature_requests / f"{request_id}.json"
        payload = {
            "request_id": request_id,
            "title": title,
            "description": description,
            "source_tool": source_tool,
            "created_at": now_iso(),
            "status": "open",
        }
        write_json(path, payload)
        self._store.create_artifact(
            artifact_id=f"feature_request:{request_id}",
            run_id=None,
            artifact_type="feature_request",
            path=path,
            sha256=sha256_file(path),
            metadata=payload,
        )
        return {"success": True, "request_id": request_id, "path": str(path)}

    def list_requests(self) -> list[dict[str, object]]:
        """List feature request artifacts."""
        return [
            artifact
            for artifact in self._store.list_artifacts()
            if artifact.get("artifact_type") == "feature_request"
        ]
