"""Verify revision checks, legacy recovery, and repair evidence ordering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.e2e.workflow_trace import RevisionEvidence

if TYPE_CHECKING:
    from tests.e2e.llm_client import ToolCall
    from tests.e2e.workflow_trace import WorkflowTrace


def assert_revision(trace: WorkflowTrace) -> RevisionEvidence:
    """Bind successful validation and dry-run to a previously saved revision."""
    calls, scenario, target = trace.calls, trace.scenario, trace.target
    require_index = trace.require_index
    contract_index = require_index(
        f"get_strategy_contract before saving {target}",
        (
            index
            for index, call in enumerate(calls)
            if call.name == "get_strategy_contract"
        ),
    )
    save_index = require_index(
        f"save_strategy({target}) returning success=True",
        (
            index
            for index, call in enumerate(calls)
            if call.name == "save_strategy"
            and call.arguments.get("strategy_id") == scenario.strategy_id
            and call.result
            and call.result.get("success")
        ),
    )
    assert contract_index < save_index, (
        f"Invalid MCP step order for {target}: get_strategy_contract must precede "
        f"the first successful save_strategy (events {contract_index}, {save_index})"
    )
    if scenario.legacy:
        _assert_legacy_recovery(trace, save_index)
    previous = save_index
    for name, status in (
        ("validate_strategy", "validated"),
        ("dry_run_strategy", "runnable"),
    ):
        previous = require_index(
            f"{name}({target}) returning success=True, status={status!r} "
            f"after {calls[previous].name} (event {previous})",
            (
                index
                for index, call in enumerate(calls)
                if index > previous
                and call.name == name
                and call.result
                and call.result.get("success")
                and call.result.get("status") == status
                and call.arguments.get("strategy_id") == scenario.strategy_id
            ),
        )
    revision_id = calls[previous].result.get("revision_id")
    assert isinstance(revision_id, str) and revision_id, (
        "Dry-run must identify revision_id"
    )
    require_index(
        f"save_strategy returning revision_id={revision_id!r} before dry-run",
        (
            index
            for index, call in enumerate(calls)
            if index < previous
            and call.name == "save_strategy"
            and call.result
            and call.arguments.get("strategy_id") == scenario.strategy_id
            and call.result.get("success")
            and call.result.get("revision_id") == revision_id
        ),
    )
    checked: dict[str, ToolCall] = {}
    for name, status in (
        ("validate_strategy", "validated"),
        ("dry_run_strategy", "runnable"),
    ):
        checked_index = require_index(
            f"{name} explicitly checking revision_id={revision_id!r}",
            (
                index
                for index, call in enumerate(calls)
                if save_index < index <= previous
                and call.name == name
                and call.result
                and call.arguments.get("strategy_id") == scenario.strategy_id
                and call.arguments.get("revision_id") == revision_id
                and call.result.get("success")
                and call.result.get("status") == status
                and call.result.get("revision_id") == revision_id
            ),
        )
        checked[name] = calls[checked_index]
    return RevisionEvidence(save_index, previous, revision_id, checked)


def _assert_legacy_recovery(trace: WorkflowTrace, save_index: int) -> None:
    """Require reading old source before creating a fresh draft identity."""
    calls, scenario, target = trace.calls, trace.scenario, trace.target
    require_index = trace.require_index
    legacy_identity = {
        "strategy_id": scenario.legacy_strategy_id,
        "revision_id": None,
        "kind": "legacy",
        "status": "revision_required",
        "code": "strategy_revision_required",
    }
    list_index = require_index(
        f"list_strategies identifying {target} as legacy/revision_required",
        (
            index
            for index, call in enumerate(calls)
            if call.name == "list_strategies"
            and isinstance(call.result, dict)
            and any(
                all(item.get(key) == value for key, value in legacy_identity.items())
                for item in call.result.get("items", [])
            )
        ),
    )
    read_index = require_index(
        f"get_strategy({target}) returning legacy source/config before resaving",
        (
            index
            for index, call in enumerate(calls)
            if call.name == "get_strategy"
            and call.arguments.get("strategy_id") == scenario.legacy_strategy_id
            and call.result
            and call.result.get("success") is True
            and all(
                call.result.get(key) == value for key, value in legacy_identity.items()
            )
            and call.result.get("source_code")
            and call.result.get("yaml_config")
        ),
    )
    assert list_index < read_index < save_index, (
        f"Invalid MCP legacy recovery order for {target}: "
        "discovery and source/config "
        "retrieval must precede successful save_strategy "
        f"(events {list_index}, {read_index}, {save_index})"
    )
    assert calls[save_index].result.get("status") == "draft", (
        "Resaving a legacy strategy must create a draft without inherited evidence"
    )


def assert_repair(trace: WorkflowTrace, save_index: int) -> None:
    """Bind the repair save to the failed revision the model actually retrieved."""
    calls, scenario, target = trace.calls, trace.scenario, trace.target
    require_index = trace.require_index
    failure_code = (
        "strategy_check_failed" if scenario.runtime_exit else "invalid_signal_values"
    )
    read_index = require_index(
        f"get_strategy({target}) returning success=True before repairing the draft",
        (
            index
            for index, call in enumerate(calls)
            if call.name == "get_strategy"
            and call.arguments.get("strategy_id") == scenario.strategy_id
            and call.result
            and call.result.get("success")
        ),
    )
    failure_index = require_index(
        f"validate_strategy({target}) returning success=False with "
        f"diagnostic code='{failure_code}' before repairing the draft",
        (
            index
            for index, call in enumerate(calls)
            if call.name == "validate_strategy"
            and call.arguments.get("strategy_id") == scenario.strategy_id
            and call.result
            and call.result.get("success") is False
            and any(
                diagnostic.get("code") == failure_code
                for diagnostic in call.result.get("diagnostics", [])
            )
        ),
    )
    assert read_index < failure_index < save_index, (
        f"Invalid MCP repair order for {target}: successful get_strategy must "
        f"precede validate_strategy with {failure_code}, which must "
        f"precede successful save_strategy (events {read_index}, "
        f"{failure_index}, {save_index})"
    )
    selected_revision = calls[read_index].result.get("revision_id")
    assert isinstance(selected_revision, str) and selected_revision, (
        "Repair retrieval must identify an immutable revision"
    )
    assert (
        calls[save_index].arguments.get("expected_revision_id") == selected_revision
    ), "Repair save_strategy must bind expected_revision_id to the retrieved draft"
