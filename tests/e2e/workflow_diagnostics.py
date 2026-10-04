"""Retain useful failure evidence in pytest output before workspace cleanup."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.strategy_revisions import StrategyRevisionStore
from tradingdev.shared.utils.config import load_config

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.e2e.strategy_scenarios import Scenario
    from tests.integration.mcp_harness import MCPWorkspace


@contextmanager
def workflow_diagnostics(workspace: MCPWorkspace, scenario: Scenario) -> Iterator[None]:
    """Annotate the original exception without retaining test files or processes."""
    try:
        yield
    except Exception as error:
        evidence: dict[str, Any] = {"scenario": scenario.name}
        try:
            paths = WorkspacePaths(workspace.workspace)
            revision = StrategyRevisionStore(paths).load(scenario.strategy_id)
            if revision is not None:
                evidence["revision_id"] = revision.revision_id
                evidence["status"] = revision.status.value
                evidence["saved_config"] = load_config(
                    paths.root / revision.config_path
                )
            evidence["execution_configs"] = [
                {
                    "run_id": path.parent.name,
                    "manifest": json.loads(path.read_text(encoding="utf-8")),
                }
                for path in sorted(paths.runs.glob("*/manifest.json"))
            ]
        except Exception as diagnostic_error:
            evidence["diagnostic_error"] = str(diagnostic_error)
        try:
            serialized = json.dumps(
                evidence, ensure_ascii=False, default=str, sort_keys=True
            )
        except Exception as serialization_error:
            serialized = f"Cannot serialize workflow evidence: {serialization_error}"
        error.add_note("Workflow evidence before cleanup:\n" + serialized)
        raise
