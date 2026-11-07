"""Bounded stdio transport with an unreaped child anchoring process-group cleanup."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.clients.codex_protocol import CodexProtocolError, object_value
from tradingdev.adapters.execution.worker_supervisor import terminate_owned_group

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

MAX_FRAME_BYTES = 64 * 1024 * 1024


def _invalid_constant(value: str) -> None:
    raise CodexProtocolError(f"Non-finite JSON value: {value}")


class CodexTransport:
    def __init__(self, reader: asyncio.StreamReader, descriptor: int) -> None:
        self.reader, self.descriptor = reader, descriptor

    async def read(self) -> dict[str, Any]:
        try:
            line = await self.reader.readline()
        except (ValueError, asyncio.LimitOverrunError) as exc:
            raise CodexProtocolError(
                "Codex frame exceeds the transport byte limit"
            ) from exc
        if not line:
            raise CodexProtocolError(
                "Codex closed the stream before completing the request"
            )
        if len(line) > MAX_FRAME_BYTES:
            raise CodexProtocolError("Codex frame exceeds the transport byte limit")
        return object_value(
            json.loads(line, parse_constant=_invalid_constant), "Codex frame"
        )

    async def send(self, message: dict[str, Any]) -> None:
        data = memoryview(
            (json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n").encode()
        )
        if len(data) > MAX_FRAME_BYTES:
            raise ValueError("Codex request exceeds the transport byte limit")
        loop = asyncio.get_running_loop()
        while data:
            try:
                written = os.write(self.descriptor, data)
            except BlockingIOError:
                ready: asyncio.Future[None] = loop.create_future()

                def writable(future: asyncio.Future[None] = ready) -> None:
                    if not future.done():
                        future.set_result(None)

                loop.add_writer(self.descriptor, writable)
                try:
                    await ready
                finally:
                    loop.remove_writer(self.descriptor)
            else:
                if written == 0:
                    raise CodexProtocolError("Codex input pipe stopped accepting data")
                data = data[written:]


@asynccontextmanager
async def codex_transport(
    command: list[str], *, cwd: Path, stderr_path: Path
) -> AsyncIterator[CodexTransport]:
    """Stop the client group on success/error/timeout, leaving managed jobs separate.

    TradingDev research workers own independent sessions and persisted control
    identities. Closing a chat client must not cancel those background research jobs.
    This function is the sole waiter for its direct child; no poll()/asyncio child
    watcher may reap it before terminate_owned_group verifies group cleanup.
    """
    pipe_transport: asyncio.ReadTransport | None = None
    with stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            start_new_session=True,
            bufsize=0,
        )
        try:
            if process.stdout is None or process.stdin is None:
                raise RuntimeError("Codex stdio pipes were not created")
            os.set_blocking(process.stdin.fileno(), False)
            reader = asyncio.StreamReader(limit=MAX_FRAME_BYTES)
            protocol = asyncio.StreamReaderProtocol(reader)
            pipe_transport, _ = await asyncio.get_running_loop().connect_read_pipe(
                lambda: protocol, process.stdout
            )
            yield CodexTransport(reader, process.stdin.fileno())
        finally:
            try:
                if process.stdin is not None:
                    process.stdin.close()
                terminate_owned_group(process)
            finally:
                if pipe_transport is not None:
                    pipe_transport.close()
                elif process.stdout is not None:
                    process.stdout.close()
