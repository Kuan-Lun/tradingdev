"""Offline checks that legacy model recovery requires actual discovery evidence."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tests.e2e.llm_client import ToolCall
from tests.e2e.strategy_scenarios import SCENARIOS
from tests.e2e.test_llm_workflows import assert_workflow


@pytest.fixture
def legacy_calls() -> list[ToolCall]:
    strategy_id = SCENARIOS["legacy"].strategy_id
    identity = {"strategy_id": strategy_id, "revision_id": "new-revision"}
    legacy = {
        "strategy_id": strategy_id,
        "revision_id": None,
        "kind": "legacy",
        "status": "revision_required",
        "code": "strategy_revision_required",
    }
    return [
        ToolCall("get_strategy_contract", {}, {"example_strategy_code": "source"}),
        ToolCall("list_strategies", {}, [legacy]),
        ToolCall(
            "get_strategy",
            {"strategy_id": strategy_id},
            {
                **legacy,
                "success": True,
                "source_code": "legacy source",
                "yaml_config": "legacy config",
            },
        ),
        ToolCall(
            "save_strategy",
            {"strategy_id": strategy_id},
            {**identity, "success": True, "status": "draft"},
        ),
        ToolCall(
            "validate_strategy",
            identity,
            {**identity, "success": True, "status": "validated"},
        ),
        ToolCall(
            "dry_run_strategy",
            identity,
            {**identity, "success": True, "status": "runnable"},
        ),
        ToolCall(
            "start_backtest",
            {
                **identity,
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-08",
            },
            {**identity, "job_id": "job-1"},
        ),
        ToolCall(
            "get_job_status",
            {"job_id": "job-1"},
            {**identity, "status": "done", "run_id": "run-1", "metrics": {}},
        ),
        ToolCall(
            "get_run",
            {"run_id": "run-1"},
            {"success": True, "run": {**identity, "metrics": {}}},
        ),
        ToolCall("list_artifacts", {"run_id": "run-1"}, [{"path": "result.json"}]),
    ]


def test_legacy_evidence_accepts_discovery_read_and_fresh_revision(
    legacy_calls: list[ToolCall],
) -> None:
    assert_workflow(legacy_calls, SCENARIOS["legacy"])


@pytest.mark.parametrize("missing", ["list_strategies", "get_strategy"])
def test_legacy_evidence_rejects_skipping_existing_contents(
    legacy_calls: list[ToolCall], missing: str
) -> None:
    calls = [call for call in legacy_calls if call.name != missing]
    with pytest.raises(AssertionError, match="Missing required MCP step"):
        assert_workflow(calls, SCENARIOS["legacy"])


@pytest.mark.parametrize("name", ["list_strategies", "get_strategy"])
def test_legacy_evidence_rejects_nonlegacy_responses(
    legacy_calls: list[ToolCall], name: str
) -> None:
    calls = deepcopy(legacy_calls)
    call = next(call for call in calls if call.name == name)
    payload = call.result[0] if name == "list_strategies" else call.result
    payload["status"] = "runnable"
    with pytest.raises(AssertionError, match="Missing required MCP step"):
        assert_workflow(calls, SCENARIOS["legacy"])


def test_legacy_evidence_rejects_reading_only_after_resaving(
    legacy_calls: list[ToolCall],
) -> None:
    legacy_calls[2], legacy_calls[3] = legacy_calls[3], legacy_calls[2]
    with pytest.raises(AssertionError, match="Invalid MCP legacy recovery order"):
        assert_workflow(legacy_calls, SCENARIOS["legacy"])


def test_legacy_evidence_rejects_inheriting_runnable_status(
    legacy_calls: list[ToolCall],
) -> None:
    legacy_calls[3].result["status"] = "runnable"
    with pytest.raises(AssertionError, match="without inherited evidence"):
        assert_workflow(legacy_calls, SCENARIOS["legacy"])
