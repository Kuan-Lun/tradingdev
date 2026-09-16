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
    with pytest.raises(AssertionError, match=SCENARIOS["repair"].strategy_id):
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
    with pytest.raises(AssertionError, match="diagnostic code='invalid_signal_values'"):
        assert_workflow(calls, SCENARIOS["repair"])


def test_failed_lookup_does_not_count_as_reading_the_draft() -> None:
    calls = _repair_evidence()
    calls[1].result = {"success": False, "error": "Unknown strategy"}
    with pytest.raises(AssertionError, match="get_strategy.*success=True"):
        assert_workflow(calls, SCENARIOS["repair"])


@pytest.mark.parametrize(
    ("event_index", "expected_details"),
    [
        (0, ("get_strategy_contract", "strategy_id='llm_repair'")),
        (1, ("get_strategy", "strategy_id='llm_repair'", "success=True")),
        (2, ("validate_strategy", "success=False", "invalid_signal_values")),
        (3, ("save_strategy", "strategy_id='llm_repair'", "success=True")),
        (4, ("validate_strategy", "status='validated'", "after save_strategy")),
        (5, ("dry_run_strategy", "status='runnable'", "after validate_strategy")),
        (6, ("start_backtest", "nonempty job_id", "after successful dry_run_strategy")),
        (
            7,
            ("get_job_status", "job_id='job'", "status='done'", "after start_backtest"),
        ),
        (8, ("get_run", "run_id='run'", "after get_job_status reported done")),
        (9, ("list_artifacts", "run_id='run'", "nonempty result")),
    ],
    ids=[
        "contract",
        "read",
        "diagnosis",
        "save",
        "validate",
        "dry-run",
        "start",
        "done",
        "query",
        "artifacts",
    ],
)
def test_missing_step_reports_required_tool_target_and_outcome(
    event_index: int, expected_details: tuple[str, ...]
) -> None:
    calls = _repair_evidence()
    del calls[event_index]
    with pytest.raises(AssertionError, match="Missing required MCP step:") as error:
        assert_workflow(calls, SCENARIOS["repair"])
    message = str(error.value)
    for detail in expected_details:
        assert detail in message


@pytest.mark.parametrize(
    ("event_index", "actual_status", "expected_status"),
    [(4, "draft", "validated"), (5, "validated", "runnable"), (7, "failed", "done")],
)
def test_wrong_status_reports_the_required_status(
    event_index: int, actual_status: str, expected_status: str
) -> None:
    calls = _repair_evidence()
    calls[event_index].result["status"] = actual_status
    with pytest.raises(AssertionError, match=f"status='{expected_status}'"):
        assert_workflow(calls, SCENARIOS["repair"])


def test_contract_order_failure_explains_required_order() -> None:
    calls = _repair_evidence()
    calls[0], calls[3] = calls[3], calls[0]
    with pytest.raises(AssertionError, match="get_strategy_contract must precede"):
        assert_workflow(calls, SCENARIOS["repair"])


def test_dry_run_before_validation_does_not_satisfy_required_step() -> None:
    calls = _repair_evidence()
    calls[4], calls[5] = calls[5], calls[4]
    with pytest.raises(
        AssertionError, match="dry_run_strategy.*after validate_strategy"
    ):
        assert_workflow(calls, SCENARIOS["repair"])


def test_repair_order_failure_explains_read_diagnose_save_sequence() -> None:
    calls = _repair_evidence()
    calls[1], calls[2] = calls[2], calls[1]
    with pytest.raises(AssertionError, match="Invalid MCP repair order") as error:
        assert_workflow(calls, SCENARIOS["repair"])
    assert "strategy_id='llm_repair'" in str(error.value)
    assert "get_strategy must precede validate_strategy" in str(error.value)
    assert "precede successful save_strategy" in str(error.value)
