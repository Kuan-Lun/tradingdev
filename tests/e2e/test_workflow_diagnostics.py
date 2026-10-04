"""Failures report saved inputs before normal, error, and timeout cleanup."""

from __future__ import annotations

import pytest

from tests.e2e.strategy_scenarios import SCENARIOS
from tests.e2e.workflow_diagnostics import workflow_diagnostics
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.strategy_revisions import StrategyRevisionStore


@pytest.mark.parametrize(
    "failure", [AssertionError("seed mismatch"), TimeoutError("model timeout")]
)
def test_saved_settings_are_reported_and_original_failure_survives_cleanup(
    failure: Exception,
) -> None:
    with pytest.raises(type(failure)) as caught, temporary_mcp_workspace() as workspace:
        paths = WorkspacePaths(workspace.workspace)
        saved = StrategyRevisionStore(paths).create(
            "llm_repair",
            "# fixture",
            {
                "strategy": {"class_name": "Fixture"},
                "random_seed": 42,
                "backtest": {"random_sead": 7},
            },
        )
        with workflow_diagnostics(workspace, SCENARIOS["repair"]):
            raise failure
    assert caught.value is failure
    evidence = "\n".join(failure.__notes__)
    assert saved.revision_id in evidence
    assert '"random_seed": 42' in evidence
    assert '"random_sead": 7' in evidence
    assert not workspace.root.exists()


def test_diagnostic_failure_does_not_mask_original_error() -> None:
    failure = RuntimeError("provider failed")
    with pytest.raises(RuntimeError) as caught, temporary_mcp_workspace() as workspace:
        directory = workspace.workspace / "generated_strategies" / "llm_repair"
        directory.mkdir(parents=True)
        (directory / "current.json").write_text("invalid json", encoding="utf-8")
        with workflow_diagnostics(workspace, SCENARIOS["repair"]):
            raise failure
    assert caught.value is failure
    assert "diagnostic_error" in "\n".join(failure.__notes__)
    assert not workspace.root.exists()


@pytest.mark.parametrize("cyclic", [False, True])
def test_non_json_draft_evidence_cannot_replace_original_failure(cyclic: bool) -> None:
    failure = AssertionError("original seed mismatch")
    data: dict[object, object] = {"a": 1, 2: "mixed"}
    if cyclic:
        data = {}
        data["cycle"] = data
    with (
        pytest.raises(AssertionError) as caught,
        temporary_mcp_workspace() as workspace,
    ):
        StrategyRevisionStore(WorkspacePaths(workspace.workspace)).create(
            "llm_repair",
            "# fixture",
            {"strategy": {"class_name": "Fixture"}, "unknown": data},
        )
        with workflow_diagnostics(workspace, SCENARIOS["repair"]):
            raise failure
    assert caught.value is failure
    assert "Cannot serialize workflow evidence" in "\n".join(failure.__notes__)
    assert not workspace.root.exists()


def test_success_does_not_read_or_retain_diagnostics() -> None:
    with (
        temporary_mcp_workspace() as workspace,
        workflow_diagnostics(workspace, SCENARIOS["repair"]),
    ):
        assert not workspace.workspace.exists()
    assert not workspace.root.exists()
