"""Application service for completed run lookup and comparison."""

from __future__ import annotations

from typing import Any

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore


class RunService:
    """Read and compare run metadata."""

    def __init__(
        self,
        *,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._workspace.ensure()
        self._store = store or SQLiteStore(self._workspace)

    def list_runs(self) -> list[dict[str, Any]]:
        """List completed runs."""
        return self._store.list_runs()

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Return one completed run."""
        run = self._store.get_run(run_id)
        if run is None:
            return {"success": False, "error": f"Unknown run: {run_id}"}
        return {"success": True, "run": run}

    def compare_runs(self, run_ids: list[str]) -> dict[str, Any]:
        """Compare numeric metrics across selected runs."""
        if len(run_ids) < 2:
            return {
                "success": False,
                "error": "compare_runs requires at least two run_ids",
            }

        rows = []
        metric_names: set[str] = set()
        for run_id in run_ids:
            run = self._store.get_run(run_id)
            if run is None:
                return {"success": False, "error": f"Unknown run: {run_id}"}
            metrics = run.get("metrics", {})
            if isinstance(metrics, dict):
                metric_names.update(
                    key
                    for key, value in metrics.items()
                    if isinstance(value, int | float)
                )
            rows.append(run)

        comparison = []
        for run in rows:
            metrics = run.get("metrics", {})
            comparison.append(
                {
                    "run_id": run["run_id"],
                    "strategy_id": run["strategy_id"],
                    "metrics": {
                        name: metrics.get(name)
                        for name in sorted(metric_names)
                        if isinstance(metrics, dict)
                    },
                }
            )
        return {"success": True, "runs": comparison}
