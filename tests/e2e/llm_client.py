"""Two model clients, one MCP conversation contract and one assertion format."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from tests.e2e.codex_harness import LIFECYCLE_TOOLS, successful_mcp_calls

if TYPE_CHECKING:
    from tests.integration.mcp_harness import MCPClient


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: Any


def tool_payload(result: dict[str, Any]) -> Any:
    """Accept MCP wire format and Codex's snake_case JSON event representation."""
    structured = result.get("structuredContent", result.get("structured_content"))
    if structured is None:
        for block in result.get("content", []):
            if block.get("type") == "text":
                try:
                    structured = json.loads(block["text"])
                except (ValueError, KeyError):
                    continue
                break
    if isinstance(structured, dict) and set(structured) == {"result"}:
        return structured["result"]
    return structured


def codex_calls(events: list[dict[str, Any]]) -> list[ToolCall]:
    calls = []
    for item in successful_mcp_calls(events):
        arguments = item.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        assert isinstance(arguments, dict), item
        calls.append(
            ToolCall(item["tool"], arguments, tool_payload(item.get("result") or {}))
        )
    return calls


async def run_local_model(
    client: MCPClient,
    prompt: str,
    *,
    model: str,
    base_url: str,
    timeout_seconds: float,
    max_tool_calls: int = 64,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[ToolCall]:
    """Bridge local Chat Completions tool calls to real stdio MCP, with bounds.

    No shell or file tool is exposed. MCP errors are returned to the model so it
    can repair drafts; malformed model responses and budget exhaustion fail.
    The caller owns the MCP session and its worker/file cleanup.
    """
    calls: list[ToolCall] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": client.instructions},
        {"role": "user", "content": prompt},
    ]
    try:
        async with (
            asyncio.timeout(timeout_seconds),
            httpx.AsyncClient(
                timeout=timeout_seconds, transport=transport, trust_env=False
            ) as http,
        ):
            tools: list[dict[str, Any]] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in (await client.session.list_tools()).tools
                if tool.name in LIFECYCLE_TOOLS
            ]
            allowed = {tool["function"]["name"] for tool in tools}
            while True:
                response = await http.post(
                    base_url.rstrip("/") + "/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "auto",
                        "stream": False,
                        "temperature": 0,
                    },
                )
                response.raise_for_status()
                choice = response.json()["choices"][0]
                message = choice["message"]
                assert message.get("role") == "assistant", message
                requested = message.get("tool_calls") or []
                if not requested:
                    assert choice.get("finish_reason") == "stop", choice
                    return calls
                assert choice.get("finish_reason") in {"tool_calls", "stop"}, choice
                assert len(calls) + len(requested) <= max_tool_calls, (
                    f"Model exceeded {max_tool_calls} MCP tool calls"
                )
                # Validate the entire batch before allowing any side effect.
                parsed = []
                ids: set[str] = set()
                for call in requested:
                    name = call["function"]["name"]
                    assert call["type"] == "function" and name in allowed, name
                    assert call["id"] and call["id"] not in ids, call
                    ids.add(call["id"])
                    arguments = json.loads(call["function"]["arguments"])
                    assert isinstance(arguments, dict), arguments
                    parsed.append((call["id"], name, arguments))
                messages.append(message)
                for call_id, name, arguments in parsed:
                    result = await client.session.call_tool(name, arguments)
                    payload = result.model_dump(mode="json")
                    calls.append(ToolCall(name, arguments, tool_payload(payload)))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps(payload, ensure_ascii=False),
                        }
                    )
                # Avoid a tight polling loop while a real backtest worker starts.
                if parsed and all(name == "get_job_status" for _, name, _ in parsed):
                    await asyncio.sleep(0.5)
    except (TimeoutError, httpx.TimeoutException) as exc:
        msg = f"Local model exceeded {timeout_seconds:g}s after {len(calls)} MCP calls"
        raise AssertionError(msg) from exc
