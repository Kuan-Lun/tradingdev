"""Offline checks that model evidence belongs to the requested repair target."""

from __future__ import annotations

import pytest

from tests.e2e.llm_client import ToolCall
from tests.e2e.strategy_scenarios import SCENARIOS
from tests.e2e.test_llm_workflows import assert_workflow


def _repair_evidence() -> list[ToolCall]:
    strategy_id = SCENARIOS["repair"].strategy_id
    target = {"strategy_id": strategy_id}
    metrics = {"total_return": 0.25}
    return [
        ToolCall("get_strategy_contract", {}, {"lifecycle": "fixture"}),
        ToolCall("get_strategy", target, {"success": True}),
        ToolCall(
            "validate_strategy",
            target,
            {"success": False, "diagnostics": [{"code": "invalid_signal_values"}]},
        ),
        ToolCall("save_strategy", target, {"success": True, "status": "draft"}),
        ToolCall("validate_strategy", target, {"success": True, "status": "validated"}),
        ToolCall("dry_run_strategy", target, {"success": True, "status": "runnable"}),
        ToolCall(
            "start_backtest",
            {
                **target,
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-08",
            },
            {"job_id": "job"},
        ),
        ToolCall(
            "get_job_status",
            {"job_id": "job"},
            {"status": "done", "run_id": "run", "metrics": metrics},
        ),
        ToolCall(
            "get_run",
            {"run_id": "run"},
            {"success": True, "run": {"strategy_id": strategy_id, "metrics": metrics}},
        ),
        ToolCall("list_artifacts", {"run_id": "run"}, [{"artifact_id": "artifact"}]),
    ]


@pytest.mark.parametrize("event_index", [1, 2, 3], ids=["read", "diagnosis", "save"])
def test_other_strategy_cannot_supply_repair_evidence(event_index: int) -> None:
    calls = _repair_evidence()
    assert_workflow(calls, SCENARIOS["repair"])
    original = calls[event_index]
    calls[event_index] = ToolCall(
        original.name, {"strategy_id": "unrelated"}, original.result
    )
    with pytest.raises((AssertionError, StopIteration)):
        assert_workflow(calls, SCENARIOS["repair"])


def test_unrelated_events_do_not_hide_correct_target_diagnosis() -> None:
    target = {"strategy_id": SCENARIOS["repair"].strategy_id}
    unrelated = {"strategy_id": "unrelated"}
    distractors = [
        ToolCall("save_strategy", unrelated, {"success": True, "status": "draft"}),
        ToolCall("get_strategy", unrelated, {"success": False}),
        ToolCall(
            "validate_strategy",
            unrelated,
            {"success": False, "diagnostics": [{"code": "invalid_signal_values"}]},
        ),
        ToolCall(
            "validate_strategy",
            target,
            {"success": False, "diagnostics": [{"code": "ruff_failed"}]},
        ),
    ]
    assert_workflow([*distractors, *_repair_evidence()], SCENARIOS["repair"])


def test_repair_requires_the_seeded_signal_error() -> None:
    calls = _repair_evidence()
    calls[2].result = {"success": False, "diagnostics": [{"code": "ruff_failed"}]}
    with pytest.raises((AssertionError, StopIteration)):
        assert_workflow(calls, SCENARIOS["repair"])


def test_failed_lookup_does_not_count_as_reading_the_draft() -> None:
    calls = _repair_evidence()
    calls[1].result = {"success": False, "error": "Unknown strategy"}
    with pytest.raises((AssertionError, StopIteration)):
        assert_workflow(calls, SCENARIOS["repair"])
