"""Run an isolated Codex client and inspect its actual MCP tool events."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import psutil

from tests.e2e.codex_binary import resolve_codex_binary
from tests.integration.mcp_harness import MCPWorkspace

LIFECYCLE_TOOLS = [
    "list_strategies",
    "get_strategy_contract",
    "get_strategy",
    "save_strategy",
    "validate_strategy",
    "dry_run_strategy",
    "start_backtest",
    "get_job_status",
    "list_runs",
    "get_run",
    "list_artifacts",
    "get_artifact",
]


def runtime_environment(root: Path) -> dict[str, str]:
    """Keep server, verifier and CLI runtime caches inside the temporary root."""
    repo = Path(__file__).resolve().parents[2]
    environment = MCPWorkspace(root).environment()
    environment["PYTHONPATH"] = os.pathsep.join((environment["PYTHONPATH"], str(repo)))
    return environment


def _command(root: Path, model: str | None = None) -> list[str]:
    binary = resolve_codex_binary()
    command = [
        binary,
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--json",
        "--color",
        "never",
        "--cd",
        str(root),
    ]
    config: dict[str, Any] = {
        "features.shell_tool": False,
        "features.shell_snapshot": False,
        "web_search": "disabled",
        "history.persistence": "none",
        "log_dir": str(root / "codex_logs"),
        "sqlite_home": str(root / "codex_state"),
        "mcp_servers.tradingdev.command": sys.executable,
        "mcp_servers.tradingdev.args": [
            "-m",
            "tradingdev.mcp.server",
            "--workspace",
            str(root / "workspace"),
        ],
        "mcp_servers.tradingdev.cwd": str(root),
        "mcp_servers.tradingdev.required": True,
        "mcp_servers.tradingdev.startup_timeout_sec": 30,
        "mcp_servers.tradingdev.tool_timeout_sec": 60,
        "mcp_servers.tradingdev.enabled_tools": LIFECYCLE_TOOLS,
        "mcp_servers.tradingdev.default_tools_approval_mode": "approve",
    }
    config.update(
        {
            f"mcp_servers.tradingdev.env.{key}": value
            for key, value in runtime_environment(root).items()
        }
    )
    for key, value in config.items():
        command.extend(["-c", f"{key}={json.dumps(value)}"])
    if model := model or os.environ.get("TRADINGDEV_CODEX_MODEL"):
        command.extend(["--model", model])
    return command


def _process_identity(pid: int) -> psutil.Process | None:
    try:
        return psutil.Process(pid)
    except psutil.NoSuchProcess:
        return None


def _alive(process: psutil.Process) -> bool:
    try:
        return bool(process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, PermissionError):
        return bool(process.is_running())


def _descendants(process: psutil.Process | None) -> set[psutil.Process]:
    if process is None or not _alive(process):
        return set()
    try:
        return set(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
        return set()


def _terminate_tree(
    leader: psutil.Process | None,
    descendants: set[psutil.Process] | None = None,
    *,
    wait_seconds: float = 3,
) -> None:
    """Terminate known process identities; never signal an exited leader's group."""
    targets = (descendants or set()) | _descendants(leader)
    if leader is not None:
        targets.add(leader)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive = {process for process in targets if _alive(process)}
        if not alive:
            return
        if leader is not None and _alive(leader):
            with suppress(ProcessLookupError, PermissionError):
                if os.getpgid(leader.pid) == leader.pid:
                    os.killpg(leader.pid, sig)
        for process in alive:
            with suppress(psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                process.send_signal(sig)
        deadline = time.monotonic() + wait_seconds
        while any(_alive(process) for process in alive):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
    remaining = sorted(process.pid for process in targets if _alive(process))
    if remaining:
        msg = f"Could not terminate test subprocesses: {remaining}"
        raise RuntimeError(msg)


async def run_codex(
    root: Path,
    prompt: str,
    *,
    timeout_seconds: float = 300,
    max_tool_calls: int = 64,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Run one prompt, fail on bypasses/limits, and always stop the subprocess tree."""
    command = [*_command(root, model), prompt]
    events: list[dict[str, Any]] = []
    seen_calls: set[str] = set()
    stderr_path = root / "codex_stderr.log"
    with stderr_path.open("w", encoding="utf-8") as stderr:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=root,
            env={**os.environ, **runtime_environment(root)},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr,
            start_new_session=True,
            limit=1024 * 1024,
        )
        identity = _process_identity(process.pid)
        descendants: set[psutil.Process] = set()
        try:
            async with asyncio.timeout(timeout_seconds):
                assert process.stdout is not None
                while line := await process.stdout.readline():
                    descendants.update(_descendants(identity))
                    event = json.loads(line)
                    assert isinstance(event, dict), "Codex event must be a JSON object"
                    events.append(event)
                    item = event.get("item", {})
                    if not isinstance(item, dict):
                        continue
                    assert item.get("type") not in {
                        "command_execution",
                        "file_change",
                    }, "Codex bypassed MCP by using shell commands or editing files"
                    if item.get("type") == "mcp_tool_call":
                        assert item.get("server") == "tradingdev", item.get("server")
                        assert item.get("tool") in LIFECYCLE_TOOLS, item.get("tool")
                        seen_calls.add(str(item["id"]))
                        assert len(seen_calls) <= max_tool_calls, (
                            f"Codex exceeded {max_tool_calls} MCP tool calls"
                        )
                returncode = await process.wait()
                errors = [
                    str(event.get("message") or event.get("error"))
                    for event in events
                    if event.get("type") in {"error", "turn.failed"}
                ]
                assert returncode == 0, (
                    f"Codex exited {returncode}; errors: {errors[-3:]}\n"
                    + stderr_path.read_text(encoding="utf-8")[-4000:]
                )
        except TimeoutError as exc:
            msg = (
                f"Codex exceeded {timeout_seconds:g}s after {len(seen_calls)} MCP calls"
            )
            raise AssertionError(msg) from exc
        finally:
            _terminate_tree(identity, descendants)
            await asyncio.wait_for(process.wait(), timeout=3)
    assert any(event.get("type") == "turn.completed" for event in events), (
        "Codex did not complete its turn"
    )
    return events


def verify_generated_strategy(
    root: Path,
    *,
    timeout_seconds: float = 30,
    startup_timeout_seconds: float = 30,
    scenario: str | None = None,
) -> None:
    """Check generated code in a bounded subprocess and always reap it."""
    ready = root / "verifier.ready"
    ready.unlink(missing_ok=True)
    process = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "tests.e2e.verify_strategy",
            str(root),
            *([scenario] if scenario else []),
        ],
        cwd=root,
        env={**os.environ, **runtime_environment(root)},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    identity = _process_identity(process.pid)
    try:
        deadline = time.monotonic() + startup_timeout_seconds
        while not ready.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                msg = (
                    "Generated strategy verifier startup exceeded "
                    f"{startup_timeout_seconds:g}s"
                )
                raise AssertionError(msg)
            time.sleep(0.02)
        output, _ = process.communicate(timeout=timeout_seconds)
        assert process.returncode == 0, (
            f"Generated strategy verification failed:\n{output[-4000:]}"
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"Generated strategy verification exceeded {timeout_seconds:g}s"
        raise AssertionError(msg) from exc
    finally:
        _terminate_tree(identity)
        process.communicate(timeout=3)


def successful_mcp_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract completed, successful MCP invocations from the CLI JSONL stream."""
    return [
        event["item"]
        for event in events
        if event.get("type") == "item.completed"
        and isinstance(event.get("item"), dict)
        and event["item"].get("type") == "mcp_tool_call"
        and event["item"].get("status") == "completed"
    ]
