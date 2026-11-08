"""Verify preparation, user decisions, and the resulting run's identity."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from tests.e2e.research_evidence import assert_saved_research
from tests.e2e.workflow_research import assert_research_reused
from tests.e2e.workflow_trace import PlanEvidence, RunEvidence
from tradingdev.domain.preflight import PreflightReceipt

if TYPE_CHECKING:
    from tests.e2e.workflow_trace import RevisionEvidence, WorkflowTrace


def assert_plan(trace: WorkflowTrace, revision: RevisionEvidence) -> PlanEvidence:
    """Verify captured research, experiment overrides, and offline preflight."""
    calls, scenario, target = trace.calls, trace.scenario, trace.target
    require_index = trace.require_index
    previous, revision_id, checked = (
        revision.dry_run_index,
        revision.revision_id,
        revision.checked,
    )
    prepare_index = require_index(
        f"prepare_backtest for {target} returning a ready plan_id "
        "after successful dry_run_strategy",
        (
            index
            for index, call in enumerate(calls)
            if index > previous
            and call.name == "prepare_backtest"
            and call.result
            and call.result.get("success") is True
            and call.result.get("status") == "ready"
            and call.result.get("plan_id")
        ),
    )
    prepared = calls[prepare_index]
    expected_arguments: dict[str, Any] = {
        "strategy_id": scenario.strategy_id,
        "revision_id": revision_id,
    }
    if scenario.experiment:
        expected_arguments["parameters"] = scenario.overrides
        assert not any(
            call.name == "save_strategy" for call in calls[previous + 1 :]
        ), "Parameter experiments must reuse the runnable revision"
    actual_arguments = dict(prepared.arguments)
    assert_research_reused(
        actual_arguments.pop("research", None),
        validated=checked["validate_strategy"],
        dry_run=checked["dry_run_strategy"],
    )
    draft = assert_saved_research(calls, prepare_index)
    actual_arguments.pop("evidence_id")
    presentation = draft.presentation.model_dump(mode="json")
    assert presentation.get("title") and presentation.get("summary")
    assert set(presentation.get("parameter_descriptions", {})) == {
        f"/{name}" for name in scenario.parameters
    }, "Presentation must explain every captured parameter"
    for optional in ("parameters", "backtest_overrides"):
        if actual_arguments.get(optional) is None:
            actual_arguments.pop(optional, None)
    assert actual_arguments == expected_arguments, (
        f"prepare_backtest arguments differ for {target}: "
        f"expected {expected_arguments!r}; got {prepared.arguments!r}"
    )
    manifest_hash = prepared.result.get("manifest_hash")
    assert isinstance(manifest_hash, str) and len(manifest_hash) == 64
    plan_id = prepared.result["plan_id"]
    assert prepared.result.get("confirmation_summary")
    assert prepared.result.get("confirmation_artifact")
    assert prepared.result.get("preflight"), (
        "Preparation must retain preflight evidence"
    )
    try:
        receipt = PreflightReceipt.model_validate(prepared.result["preflight"])
    except ValidationError as exc:
        raise AssertionError("Preparation must retain valid offline evidence") from exc
    assert receipt.manifest_hash == manifest_hash
    assert receipt.signal_check == "synthetic_contract_passed"
    assert receipt.synthetic_fixture_rows == [80, 240]
    assert set(receipt.not_checked) == {
        "historical_data",
        "backtest_engine",
        "training",
        "performance",
    }
    assert not prepared.result.get("job_id"), "Preparing must not launch a formal job"
    return PlanEvidence(prepare_index, manifest_hash, plan_id)


def assert_execution(
    trace: WorkflowTrace, revision: RevisionEvidence, plan: PlanEvidence
) -> RunEvidence:
    """Follow the approved plan through completion and persisted run readback."""
    calls, scenario, target = trace.calls, trace.scenario, trace.target
    require_index = trace.require_index
    revision_id = revision.revision_id
    prepare_index, manifest_hash, plan_id = (
        plan.prepare_index,
        plan.manifest_hash,
        plan.plan_id,
    )
    start_index = require_index(
        f"submit_execution_decision(plan_id={plan_id!r}) returning a nonempty "
        "job_id after prepare_backtest",
        (
            index
            for index, call in enumerate(calls)
            if index > prepare_index
            and call.name == "submit_execution_decision"
            and call.arguments == {"plan_id": plan_id, "decision": "approve"}
            and call.result
            and call.result.get("success") is True
            and call.result.get("plan_id") == plan_id
            and call.result.get("job_id")
        ),
    )
    started = calls[start_index]
    assert started.result["manifest_hash"] == manifest_hash
    job_id = started.result["job_id"]
    done_index = require_index(
        f"get_job_status(job_id={job_id!r}) returning status='done' "
        f"after submit_execution_decision for {target}",
        (
            index
            for index, call in enumerate(calls)
            if index > start_index
            and call.name == "get_job_status"
            and call.result
            and call.arguments.get("job_id") == job_id
            and call.result.get("status") == "done"
        ),
    )
    done = calls[done_index]
    assert done.result["revision_id"] == revision_id
    assert done.result["manifest_hash"] == manifest_hash
    run_id = done.result["run_id"]
    queried_index = require_index(
        f"get_run(run_id={run_id!r}) returning a result after "
        f"get_job_status reported done for {target}",
        (
            index
            for index, call in enumerate(calls)
            if index > done_index
            and call.name == "get_run"
            and call.result
            and call.arguments.get("run_id") == run_id
        ),
    )
    queried = calls[queried_index]
    assert queried.result["run"]["revision_id"] == revision_id
    assert queried.result["run"]["manifest_hash"] == manifest_hash
    assert queried.result["success"], (
        f"get_run(run_id={run_id!r}) did not return success=True: {queried.result!r}"
    )
    assert queried.result["run"]["metrics"] == done.result["metrics"], (
        f"get_run(run_id={run_id!r}) metrics differ from "
        f"get_job_status(job_id={job_id!r}) metrics"
    )
    assert queried.result["run"]["strategy_id"] == scenario.strategy_id, (
        f"get_run(run_id={run_id!r}) belongs to "
        f"strategy_id={queried.result['run']['strategy_id']!r}, expected {target}"
    )
    return RunEvidence(done_index, queried_index, run_id, queried)
