"""Real MCP subprocess fixtures with eager cleanup, including failed tests."""

from __future__ import annotations

import json
import os
import shutil
import signal
import sqlite3
import sys
import time
from contextlib import asynccontextmanager, closing, contextmanager, suppress
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

import anyio
import psutil
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tradingdev.adapters.execution.process_runner import ProcessIdentity

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    import pandas as pd


@dataclass
class MCPClient:
    session: ClientSession
    instructions: str = ""

    async def call(self, name: str, **arguments: Any) -> Any:
        result = await self.session.call_tool(name, arguments)
        assert not result.isError, result.model_dump(mode="json")
        assert result.structuredContent is not None, result
        payload = result.structuredContent
        return payload["result"] if set(payload) == {"result"} else payload

    async def wait_for_job(self, job_id: str, timeout: float = 120) -> dict[str, Any]:
        with anyio.fail_after(timeout):
            while True:
                status = await self.call("get_job_status", job_id=job_id)
                if status["status"] in {"done", "failed", "cancelled"}:
                    return dict(status)
                await anyio.sleep(0.1)


@dataclass
class MCPWorkspace:
    root: Path

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def data_root(self) -> Path:
        # Deliberately different from workspace/data to check propagation.
        return self.root / "market-data"

    def seed_market(self, frame: pd.DataFrame) -> Path:
        path = self.data_root / "processed" / "btcusdt_1h_2024.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return path

    def environment(self) -> dict[str, str]:
        """Confine caches and block market network access in the backend."""
        source_root = Path(__file__).resolve().parents[2] / "src"
        env = {
            "PYTHONPATH": str(source_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TRADINGDEV_WORKSPACE": str(self.workspace),
            "TRADINGDEV_DATA_ROOT": str(self.data_root),
            "TRADINGDEV_PROJECT_ROOT": str(self.root),
            "MYPY_CACHE_DIR": str(self.root / "mypy-cache"),
            "RUFF_CACHE_DIR": str(self.root / "ruff-cache"),
            "NUMBA_CACHE_DIR": str(self.root / "numba-cache"),
            "MPLCONFIGDIR": str(self.root / "matplotlib"),
            "XDG_CACHE_HOME": str(self.root / "cache"),
            "UV_OFFLINE": "1",
        }
        # Test-only network guard; the MCP server and workers remain real.
        guard = self.root / "python-guard"
        guard.mkdir(exist_ok=True)
        (guard / "sitecustomize.py").write_text(
            "import sys\n"
            "def deny_network(event, args):\n"
            "    if event in {'socket.connect', 'socket.getaddrinfo'}:\n"
            "        raise RuntimeError('Network disabled in MCP integration tests')\n"
            "sys.addaudithook(deny_network)\n",
            encoding="utf-8",
        )
        env["PYTHONPATH"] = os.pathsep.join((str(guard), str(source_root)))
        return env

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[MCPClient]:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "tradingdev.mcp.server"],
            cwd=str(self.root),
            env={**os.environ, **self.environment()},
        )
        log_path = self.root / "server.log"
        try:
            with log_path.open("a", encoding="utf-8") as errlog:
                async with stdio_client(params, errlog=errlog) as (read, write):
                    async with ClientSession(
                        read, write, read_timeout_seconds=timedelta(seconds=100)
                    ) as session:
                        initialized = await session.initialize()
                        assert initialized.serverInfo.name == "tradingdev"
                        assert "save_strategy" in (initialized.instructions or "")
                        try:
                            yield MCPClient(session, initialized.instructions or "")
                        finally:
                            self.stop_workers()
        except BaseException:
            # Capture the failure log before successful workspace cleanup.
            print(log_path.read_text(encoding="utf-8"))
            raise

    def stop_workers(self) -> None:
        """Stop only this workspace's detached worker groups.

        Target individual PIDs from our database; global process enumeration is
        unavailable in some sandboxes. The MCP server owns/reaps these children,
        so a zombie already counts as exited while its parent is shutting down.
        """
        database = self.workspace / "tradingdev.sqlite"
        if not database.exists():
            return
        with closing(sqlite3.connect(database)) as connection:
            workers = connection.execute(
                "SELECT pid, payload FROM jobs WHERE pid IS NOT NULL"
            ).fetchall()
        processes: list[psutil.Process] = []
        failures: list[Exception] = []
        for pid, payload in workers:
            identity = ProcessIdentity.from_values(
                pid, json.loads(payload).get("process_create_time")
            )
            if identity is None:
                failures.append(
                    AssertionError(f"Worker creation identity missing: PID {pid}")
                )
                continue
            try:
                # Constructing Process(pid) alone accepts an already-reused PID.
                process = identity.get_process()
                if process is None:
                    continue
                command = process.cmdline()
                # Also reject another workspace's process or a non-worker.
                if not any("tradingdev.mcp.workers." in item for item in command):
                    continue
                if not any(str(self.workspace) in item for item in command) and (
                    Path(process.cwd()).resolve() != self.root
                ):
                    continue
                if os.getpgid(pid) != pid:
                    continue
                processes.append(process)
            except (psutil.NoSuchProcess, ProcessLookupError):
                continue
            except (psutil.AccessDenied, PermissionError) as error:
                failures.append(error)
        for process in processes:
            try:
                self._signal_worker_group(process, signal.SIGTERM)
            except (psutil.AccessDenied, PermissionError) as error:
                failures.append(error)
        alive = self._wait_for_workers(processes)
        for process in alive:
            try:
                self._signal_worker_group(process, signal.SIGKILL)
            except (psutil.AccessDenied, PermissionError) as error:
                failures.append(error)
        alive = self._wait_for_workers(alive)
        if alive:
            failures.append(AssertionError(f"Test workers did not exit: {alive}"))
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ExceptionGroup("Test worker cleanup failed", failures)

    @staticmethod
    def _signal_worker_group(process: psutil.Process, sig: signal.Signals) -> None:
        with suppress(psutil.NoSuchProcess, ProcessLookupError):
            # Recheck the retained identity for reuse after initial inspection.
            if process.is_running() and os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, sig)

    @staticmethod
    def _wait_for_workers(processes: list[psutil.Process]) -> list[psutil.Process]:
        deadline = time.monotonic() + 5
        alive = processes
        while alive and time.monotonic() < deadline:
            alive = [process for process in alive if worker_is_alive(process)]
            if alive:
                time.sleep(0.05)
        return alive


def worker_is_alive(process: psutil.Process) -> bool:
    """Check the recorded process identity without treating zombies as active."""
    try:
        return bool(process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


@contextmanager
def temporary_mcp_workspace() -> Iterator[MCPWorkspace]:
    # Also used outside pytest; stop detached workers before removing files.
    directory = Path(mkdtemp(prefix="tradingdev-mcp-test-")).resolve()
    workspace = MCPWorkspace(directory)
    try:
        yield workspace
    finally:
        try:
            workspace.stop_workers()
        except Exception as error:
            raise RuntimeError(
                f"Worker cleanup failed; temporary directory retained: {directory}"
            ) from error
        shutil.rmtree(directory)
        assert not directory.exists()
