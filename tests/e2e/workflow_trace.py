"""Typed evidence passed between model-workflow verification stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.e2e.llm_client import ToolCall
    from tests.e2e.strategy_scenarios import Scenario


@dataclass(frozen=True)
class WorkflowTrace:
    calls: list[ToolCall]
    scenario: Scenario

    @property
    def target(self) -> str:
        return f"strategy_id={self.scenario.strategy_id!r}"

    def require_index(self, description: str, indices: Iterator[int]) -> int:
        index = next(indices, None)
        if index is None:
            raise AssertionError(f"Missing required MCP step: {description}")
        return index


@dataclass(frozen=True)
class RevisionEvidence:
    save_index: int
    dry_run_index: int
    revision_id: str
    checked: dict[str, ToolCall]


@dataclass(frozen=True)
class PlanEvidence:
    prepare_index: int
    manifest_hash: str
    plan_id: str


@dataclass(frozen=True)
class RunEvidence:
    done_index: int
    queried_index: int
    run_id: str
    queried: ToolCall
