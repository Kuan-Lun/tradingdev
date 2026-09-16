"""Offline model-adapter tests, including a real stdio MCP round trip."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, create_autospec, patch

import httpx
import pytest
from mcp import ClientSession
from mcp.types import CallToolResult, ListToolsResult, Tool

from tests.e2e.codex_harness import LIFECYCLE_TOOLS
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.integration.mcp_harness import MCPClient, temporary_mcp_workspace


def _call(
    name: str = "list_strategies",
    arguments: str = "{}",
    call_id: str = "call-1",
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _reply(
    calls: list[dict[str, Any]] | None = None,
    *,
    finish_reason: str | None = None,
) -> httpx.Response:
    message: dict[str, Any] = {"role": "assistant", "content": None}
    if calls is not None:
        message["tool_calls"] = calls
    else:
        message["content"] = "Complete."
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": message,
                    "finish_reason": finish_reason
                    or ("tool_calls" if calls else "stop"),
                }
            ]
        },
    )


def _mock_client() -> tuple[MCPClient, AsyncMock]:
    session = create_autospec(ClientSession, instance=True)
    session.list_tools.return_value = ListToolsResult(
        tools=[
            Tool(
                name="list_strategies",
                description="List strategies.",
                inputSchema={"type": "object", "properties": {}},
            )
        ]
    )
    session.call_tool.return_value = CallToolResult(
        content=[], structuredContent={"result": []}
    )
    return MCPClient(session, "Use MCP tools."), session.call_tool


def test_local_model_receives_real_mcp_schemas_errors_and_repaired_result() -> None:
    async def exercise() -> None:
        with temporary_mcp_workspace() as workspace:
            async with workspace.connect() as client:
                advertised = {
                    tool.name: tool
                    for tool in (await client.session.list_tools()).tools
                }
                requests: list[dict[str, Any]] = []
                repaired_id = ""

                def respond(request: httpx.Request) -> httpx.Response:
                    nonlocal repaired_id
                    assert (
                        str(request.url) == "http://localhost:11434/v1/chat/completions"
                    )
                    body = json.loads(request.content)
                    requests.append(body)
                    assert body["model"] == "offline-fixture-model"
                    assert "temperature" not in body
                    assert "reasoning_effort" not in body
                    assert body["messages"][0] == {
                        "role": "system",
                        "content": client.instructions,
                    }
                    schemas = {
                        item["function"]["name"]: item["function"]
                        for item in body["tools"]
                    }
                    assert set(schemas) == set(LIFECYCLE_TOOLS) & advertised.keys()
                    for name, schema in schemas.items():
                        assert schema["parameters"] == advertised[name].inputSchema
                    if len(requests) == 1:
                        assert (
                            body["messages"][1]["content"] == "Find a valid strategy."
                        )
                        return _reply(
                            [
                                _call(call_id="list"),
                                _call(
                                    "get_strategy",
                                    '{"strategy_id": "missing-strategy"}',
                                    "missing",
                                ),
                                _call("get_strategy", "{}", "invalid-schema"),
                            ]
                        )
                    if len(requests) == 2:
                        responses = {
                            item["tool_call_id"]: json.loads(item["content"])
                            for item in body["messages"]
                            if item["role"] == "tool"
                        }
                        listed = responses["list"]["structuredContent"]["result"]
                        assert listed and all(
                            item["kind"] == "bundled" for item in listed
                        )
                        semantic_error = responses["missing"]["structuredContent"]
                        assert semantic_error["success"] is False
                        assert "missing-strategy" in semantic_error["error"]
                        assert responses["invalid-schema"]["isError"] is True
                        assert responses["invalid-schema"]["content"]
                        repaired_id = listed[0]["strategy_id"]
                        return _reply(
                            [
                                _call(
                                    "get_strategy",
                                    json.dumps({"strategy_id": repaired_id}),
                                    "repair",
                                )
                            ]
                        )
                    assert len(requests) == 3
                    repaired = json.loads(body["messages"][-1]["content"])
                    assert body["messages"][-1]["tool_call_id"] == "repair"
                    assert repaired["structuredContent"]["success"] is True
                    assert repaired["structuredContent"]["strategy_id"] == repaired_id
                    assert repaired["structuredContent"]["source_code"]
                    return _reply()

                calls = await run_local_model(
                    client,
                    "Find a valid strategy.",
                    model="offline-fixture-model",
                    base_url="http://localhost:11434/v1/",
                    timeout_seconds=20,
                    transport=httpx.MockTransport(respond),
                )
                assert len(calls) == 4
                assert isinstance(calls[0].result, list)
                assert calls[1].result["success"] is False
                assert calls[-1].arguments == {"strategy_id": repaired_id}
                assert calls[-1].result["success"] is True

    asyncio.run(exercise())


@pytest.mark.parametrize("temperature", [0.0, 0.7, 2.0])
def test_explicit_temperature_is_preserved_across_model_tool_rounds(
    temperature: float,
) -> None:
    client, invoke = _mock_client()
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert body["temperature"] == temperature
        return _reply([_call()] if len(requests) == 1 else None)

    calls = asyncio.run(
        run_local_model(
            client,
            "Use the selected sampling temperature",
            model="fixture",
            base_url="http://localhost/v1",
            timeout_seconds=1,
            temperature=temperature,
            transport=httpx.MockTransport(respond),
        )
    )
    assert len(requests) == 2
    assert calls == [ToolCall("list_strategies", {}, [])]
    invoke.assert_awaited_once_with("list_strategies", {})


def test_explicit_reasoning_effort_is_preserved_across_model_tool_rounds() -> None:
    client, invoke = _mock_client()
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert body["reasoning_effort"] == "low"
        return _reply([_call()] if len(requests) == 1 else None)

    calls = asyncio.run(
        run_local_model(
            client,
            "Use the selected reasoning effort",
            model="fixture",
            base_url="http://localhost/v1",
            timeout_seconds=1,
            reasoning_effort="low",
            transport=httpx.MockTransport(respond),
        )
    )
    assert len(requests) == 2
    assert calls == [ToolCall("list_strategies", {}, [])]
    invoke.assert_awaited_once_with("list_strategies", {})


def test_progress_reports_usage_and_results_without_source_or_reasoning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client, invoke = _mock_client()
    invoke.return_value = CallToolResult(
        content=[],
        structuredContent={"success": True, "source_code": "PRIVATE_SOURCE"},
    )
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        payload = _reply([_call()] if requests == 1 else None).json()
        if requests == 1:
            payload["choices"][0]["message"]["reasoning"] = "PRIVATE_REASONING"
            payload["usage"] = {
                "prompt_tokens": 103,
                "completion_tokens": 17,
                "completion_tokens_details": {"reasoning_tokens": 7},
            }
        else:
            messages = json.loads(request.content)["messages"]
            assert messages[-2]["reasoning"] == "PRIVATE_REASONING"
            assert (
                json.loads(messages[-1]["content"])["structuredContent"]["source_code"]
                == "PRIVATE_SOURCE"
            )
            payload["choices"][0]["message"]["content"] = (
                "Task complete. " + "More detail. " * 100 + "UNPRINTED_TAIL"
            )
        return httpx.Response(200, json=payload)

    calls = asyncio.run(
        run_local_model(
            client,
            "Report progress",
            model="fixture",
            base_url="http://localhost/v1",
            timeout_seconds=1,
            transport=httpx.MockTransport(respond),
        )
    )
    output = capsys.readouterr().out
    assert calls[0].result["source_code"] == "PRIVATE_SOURCE"
    assert "request started" in output and "response received" in output
    assert "MCP tool list_strategies" in output and "success=True" in output
    assert "input_tokens=103" in output and "output_tokens=17" in output
    assert "reasoning_tokens=7" in output and "reasoning_chars=17" in output
    assert "Task complete." in output
    assert not any(
        secret in output
        for secret in ("PRIVATE_SOURCE", "PRIVATE_REASONING", "UNPRINTED_TAIL")
    )


def test_http_failure_keeps_status_error_and_bounded_response_diagnostic() -> None:
    client, invoke = _mock_client()
    with pytest.raises(httpx.HTTPStatusError) as failure:
        asyncio.run(
            run_local_model(
                client,
                "Unavailable model",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=1,
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(
                        400,
                        json={
                            "error": "Unsupported reasoning effort",
                            "detail": "x" * 2000 + "UNPRINTED_TAIL",
                        },
                    )
                ),
            )
        )
    error = failure.value
    assert error.response.status_code == 400
    assert str(error.request.url) == "http://localhost/v1/chat/completions"
    assert isinstance(error.__cause__, httpx.HTTPStatusError)
    assert error.__cause__.response is error.response
    assert "Unsupported reasoning effort" in str(error)
    assert "UNPRINTED_TAIL" not in str(error)
    invoke.assert_not_awaited()


def test_model_timeout_includes_recent_tool_failure_and_preserves_cause() -> None:
    client, invoke = _mock_client()
    invoke.return_value = CallToolResult(
        content=[],
        structuredContent={"success": False, "error": "Storage unavailable"},
    )
    requests = 0
    original_error: httpx.ReadTimeout | None = None

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests, original_error
        requests += 1
        if requests == 1:
            return _reply([_call()])
        original_error = httpx.ReadTimeout("Model stalled", request=request)
        raise original_error

    with pytest.raises(AssertionError, match="after 1 MCP calls") as failure:
        asyncio.run(
            run_local_model(
                client,
                "Retry after tool failure",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=1,
                transport=httpx.MockTransport(respond),
            )
        )
    assert failure.value.__cause__ is original_error
    diagnostic = str(failure.value)
    assert "model round 2" in diagnostic
    assert "list_strategies" in diagnostic and "success=False" in diagnostic
    assert "Storage unavailable" in diagnostic
    invoke.assert_awaited_once_with("list_strategies", {})


@pytest.mark.parametrize(
    "invalid_call",
    [
        _call("shell", '{"command": "touch should-not-exist"}', "bad"),
        _call(arguments="not-json", call_id="bad"),
        _call(arguments="[]", call_id="bad"),
        _call(call_id="valid"),
    ],
    ids=["unadvertised-shell", "malformed-json", "non-object", "duplicate-id"],
)
def test_invalid_batch_is_rejected_before_any_mcp_tool_runs(
    invalid_call: dict[str, Any],
) -> None:
    client, invoke = _mock_client()
    with pytest.raises((AssertionError, ValueError)):
        asyncio.run(
            run_local_model(
                client,
                "Bad batch",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=1,
                transport=httpx.MockTransport(
                    lambda _request: _reply([_call(call_id="valid"), invalid_call])
                ),
            )
        )
    invoke.assert_not_awaited()


def test_tool_budget_counts_prior_rounds_and_rejects_next_batch_atomically() -> None:
    client, invoke = _mock_client()
    requests = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return _reply(
            [_call()]
            if requests == 1
            else [_call(call_id="two"), _call(call_id="three")]
        )

    with pytest.raises(AssertionError, match="exceeded 2 MCP tool calls"):
        asyncio.run(
            run_local_model(
                client,
                "Keep calling",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=1,
                max_tool_calls=2,
                transport=httpx.MockTransport(respond),
            )
        )
    assert requests == 2
    invoke.assert_awaited_once_with("list_strategies", {})


def test_truncated_model_response_does_not_count_as_completion() -> None:
    client, invoke = _mock_client()
    with pytest.raises(AssertionError):
        asyncio.run(
            run_local_model(
                client,
                "Finish the task",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=1,
                transport=httpx.MockTransport(
                    lambda _request: _reply(finish_reason="length")
                ),
            )
        )
    invoke.assert_not_awaited()


@pytest.mark.parametrize("failure", ["deadline", "http-timeout"])
def test_model_timeout_is_bounded_and_reports_completed_tool_count(
    failure: str,
) -> None:
    client, invoke = _mock_client()
    transport_finished = False

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal transport_finished
        try:
            if failure == "http-timeout":
                raise httpx.ReadTimeout("No response", request=request)
            await asyncio.Event().wait()
            raise AssertionError("Unreachable")
        finally:
            transport_finished = True

    with pytest.raises(AssertionError, match="after 0 MCP calls") as error:
        asyncio.run(
            run_local_model(
                client,
                "Wait forever",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=0.02,
                transport=httpx.MockTransport(respond),
            )
        )
    assert transport_finished
    assert "model round 1" in str(error.value)
    assert isinstance(error.value.__cause__, (TimeoutError, httpx.ReadTimeout))
    invoke.assert_not_awaited()


def test_model_deadline_also_cancels_stalled_mcp_tool_discovery() -> None:
    client, invoke = _mock_client()
    discovery_finished = False

    async def stalled_discovery() -> ListToolsResult:
        nonlocal discovery_finished
        try:
            await asyncio.Event().wait()
            raise AssertionError("Unreachable")
        finally:
            discovery_finished = True

    def reject_model_request(_request: httpx.Request) -> httpx.Response:
        pytest.fail("The model cannot run before MCP tool discovery completes")

    async def exercise() -> None:
        # This outer bound prevents a regression from hanging the test suite.
        await asyncio.wait_for(
            run_local_model(
                client,
                "Discover tools",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=0.02,
                transport=httpx.MockTransport(reject_model_request),
            ),
            timeout=1,
        )

    with (
        patch.object(
            client.session, "list_tools", new=AsyncMock(side_effect=stalled_discovery)
        ),
        pytest.raises(AssertionError, match="after 0 MCP calls") as error,
    ):
        asyncio.run(exercise())
    assert discovery_finished
    assert "MCP tool discovery" in str(error.value)
    invoke.assert_not_awaited()


def test_codex_events_normalize_arguments_and_both_mcp_result_encodings() -> None:
    completed: dict[str, Any] = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "status": "completed",
            "tool": "get_strategy",
            "arguments": '{"strategy_id": "example"}',
            "result": {"structured_content": {"success": False, "error": "Repair"}},
        },
    }
    wrapped = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "status": "completed",
            "tool": "list_strategies",
            "arguments": {},
            "result": {"structuredContent": {"result": [{"strategy_id": "example"}]}},
        },
    }
    text_only = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "status": "completed",
            "tool": "get_job_status",
            "arguments": {"job_id": "job-1"},
            "result": {
                "content": [
                    {"type": "text", "text": "Diagnostic before payload"},
                    {"type": "text", "text": '{"status": "done"}'},
                ]
            },
        },
    }
    calls = codex_calls(
        [
            {"type": "item.started", "item": completed["item"]},
            {"type": "item.completed", "item": {"type": "agent_message"}},
            {
                "type": "item.completed",
                "item": {**completed["item"], "status": "failed"},
            },
            completed,
            wrapped,
            text_only,
        ]
    )
    assert calls == [
        ToolCall(
            "get_strategy",
            {"strategy_id": "example"},
            {"success": False, "error": "Repair"},
        ),
        ToolCall("list_strategies", {}, [{"strategy_id": "example"}]),
        ToolCall("get_job_status", {"job_id": "job-1"}, {"status": "done"}),
    ]
