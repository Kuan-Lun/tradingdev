"""Verify complete model workflows from linked revision, plan, and run evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.e2e.workflow_arguments import effective_workflow_calls
from tests.e2e.workflow_execution import assert_execution, assert_plan
from tests.e2e.workflow_lifecycle import assert_repair, assert_revision
from tests.e2e.workflow_trace import WorkflowTrace

if TYPE_CHECKING:
    from tests.e2e.llm_client import ToolCall
    from tests.e2e.strategy_scenarios import Scenario
    from tests.e2e.workflow_trace import RunEvidence

DETAIL_METRIC_IDS = {"daily_pnl_mean", "total_volume", "n_days"}


def assert_workflow(calls: list[ToolCall], scenario: Scenario) -> None:
    """Check successful evidence and necessary revision/execution ordering."""
    trace = WorkflowTrace(effective_workflow_calls(calls), scenario)
    revision = assert_revision(trace)
    plan = assert_plan(trace, revision)
    run = assert_execution(trace, revision, plan)
    _assert_results(trace, run)
    if scenario.history:
        _assert_history(trace, run)
    if scenario.repair:
        assert_repair(trace, revision.save_index)


def _assert_results(trace: WorkflowTrace, run: RunEvidence) -> None:
    """Require artifacts and requested metric details from the completed run."""
    calls, scenario, target = trace.calls, trace.scenario, trace.target
    require_index = trace.require_index
    done_index, queried_index, run_id, queried = (
        run.done_index,
        run.queried_index,
        run.run_id,
        run.queried,
    )
    if scenario.runtime_exit:
        # A natural request may obtain artifact IDs directly from the report
        # result. The outer verifier checks the delivered file, hash, sections
        # and saved analysis before the temporary workspace is removed.
        require_index(
            f"generate_report returning a report for run_id={run_id!r}",
            (
                index
                for index, call in enumerate(calls)
                if index > done_index
                and call.name == "generate_report"
                and call.result
                and call.result.get("success") is True
                and call.result.get("run_ids") == [run_id]
                and call.result.get("artifact_id")
            ),
        )
    else:
        require_index(
            f"list_artifacts(run_id={run_id!r}) returning a nonempty result "
            f"after get_job_status reported done for {target}",
            (
                index
                for index, call in enumerate(calls)
                if index > done_index
                and call.name == "list_artifacts"
                and isinstance(call.result, dict)
                and call.result.get("items")
                and call.arguments.get("run_id") == run_id
            ),
        )
    catalog_indices = [
        index
        for index, call in enumerate(calls)
        if call.name == "get_metric_catalog"
        and call.result.get("success") is True
        and {definition["id"] for definition in call.result["definitions"]}
        >= DETAIL_METRIC_IDS
    ]
    catalog_index = (
        done_index
        if scenario.runtime_exit
        else require_index(
            "get_metric_catalog returning the additional metrics",
            iter(catalog_indices),
        )
    )
    detail_index = require_index(
        f"get_run_metrics(run_id={run_id!r}) returning the requested extra metrics",
        (
            index
            for index, call in enumerate(calls)
            if index
            > (
                done_index
                if scenario.runtime_exit
                else max(queried_index, catalog_index)
            )
            and call.name == "get_run_metrics"
            and call.arguments.get("run_id") == run_id
            and call.result
            and call.result.get("success") is True
            and set(call.result.get("metrics", {})) >= DETAIL_METRIC_IDS
        ),
    )
    assert calls[detail_index].result["scope"] == "full"
    if scenario.runtime_exit:
        assert (
            any(index < detail_index for index in catalog_indices)
            or set(calls[detail_index].result.get("definitions", {}))
            >= DETAIL_METRIC_IDS
        ), "Requested metrics must include their definitions"
    summary = queried.result["run"]
    assert set(summary["available_metric_ids"]) >= DETAIL_METRIC_IDS
    assert DETAIL_METRIC_IDS.isdisjoint(summary["metrics"])


def _assert_history(trace: WorkflowTrace, run: RunEvidence) -> None:
    """Require saved history and paging without launching another execution."""
    calls, scenario = trace.calls, trace.scenario
    require_index = trace.require_index
    done_index, run_id = run.done_index, run.run_id
    for tool in (
        "find_runs",
        "get_run_trades",
        "get_run_equity",
        "get_run_executions",
        "get_run_account_history",
        "get_report_sections",
        "generate_report",
    ):
        require_index(
            f"{tool} returning saved history/report after completion",
            (
                index
                for index, call in enumerate(calls)
                if index > done_index
                and call.name == tool
                and call.result
                and call.result.get("success") is True
                and (
                    tool in {"find_runs", "get_report_sections"}
                    or call.arguments.get("run_id") == run_id
                    or call.arguments.get("run_ids") == [run_id]
                )
            ),
        )
    trade_calls = [
        call
        for call in calls
        if call.name == "get_run_trades" and call.result.get("success")
    ]
    assert trade_calls[0].arguments.get("limit") == 1
    assert trade_calls[0].result["next_offset"] == 1
    assert any(call.arguments.get("offset") == 1 for call in trade_calls[1:])
    found = next(call for call in calls if call.name == "find_runs")
    assert found.arguments.get("strategy_id") == scenario.strategy_id
    assert found.arguments.get("parameters") == scenario.parameters
    assert any(
        row["run_id"] == run_id and row["scope"] == "full"
        for row in found.result["runs"]
    )
    for name, count in (("get_run_executions", 1), ("get_run_account_history", 2)):
        queried_history = next(call for call in calls if call.name == name)
        assert queried_history.arguments.get("limit") == count
        assert queried_history.result["availability"] == "available"
    assert sum(call.name == "prepare_backtest" for call in calls) == 1
    assert sum(call.name == "submit_execution_decision" for call in calls) == 1
