"""Own bounded Linux execution units through the configured Docker daemon."""

from __future__ import annotations

import json
import math
import os
import re
import selectors
import subprocess
import time
from contextlib import contextmanager, suppress
from importlib.metadata import version
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from uuid import uuid4

from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field

from tradingdev.domain.runtime import IsolatedRuntimeError, RuntimeLimits, RuntimeSpec

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

_OWNER_LABEL = "org.tradingdev.execution.owner"
_IMAGE_PYTHON = "/app/.venv/bin/python"
_CONTROL_TIMEOUT = 15


class _ControlRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    owner: str = Field(pattern=r"^[0-9a-f]{32}$")
    container_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    daemon_id: str = Field(min_length=1)
    spec: RuntimeSpec


class DockerRuntime:
    """Resolve immutable images and create one private unit per execution session."""

    def __init__(
        self,
        control_root: Path,
        *,
        executable: str = "docker",
        endpoint: str | None = None,
    ) -> None:
        self.control_root = control_root
        self.executable = executable
        self._environment = dict(os.environ)
        selected = endpoint or self._local_endpoint()
        parsed = urlsplit(selected)
        if (
            parsed.scheme != "unix"
            or parsed.netloc
            or not parsed.path.startswith("/")
            or parsed.query
            or parsed.fragment
        ):
            raise IsolatedRuntimeError(
                "runtime_unavailable",
                "Strategy execution requires a local Docker Unix socket",
                cleanup_verified=True,
            )
        self._prefix = [self.executable, "--host", selected]
        # Freeze the chosen endpoint; later changes to the current Docker context
        # must not direct cleanup at another daemon or spawn an SSH helper.
        self._environment.pop("DOCKER_CONTEXT", None)
        self._environment.pop("DOCKER_TLS_VERIFY", None)
        self._environment["DOCKER_HOST"] = selected

    def _local_endpoint(self) -> str:
        context = self._environment.get("DOCKER_CONTEXT")
        host = self._environment.get("DOCKER_HOST")
        if host and not context:
            return host

        def read(*arguments: str) -> str:
            try:
                result = subprocess.run(
                    [self.executable, *arguments],
                    env=self._environment,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=_CONTROL_TIMEOUT,
                )
                return result.stdout.strip()
            except (OSError, subprocess.SubprocessError) as error:
                raise IsolatedRuntimeError(
                    "runtime_unavailable",
                    "Cannot resolve the local Docker endpoint",
                    cleanup_verified=True,
                ) from error

        selected_context = context or read("context", "show")
        return read(
            "context",
            "inspect",
            selected_context,
            "--format",
            "{{.Endpoints.docker.Host}}",
        )

    def recover_abandoned(self) -> None:
        """Recover released ownership leases, including across PID namespaces."""
        for path in sorted(self.control_root.glob("*.json")):
            if path.is_symlink():
                raise IsolatedRuntimeError(
                    "runtime_control_invalid", "Invalid execution control identity"
                )
            lease = FileLock(path.with_suffix(".lock"), thread_local=False)
            try:
                lease.acquire(timeout=0)
            except Timeout:
                continue
            try:
                if not path.exists():
                    continue
                record = _ControlRecord.model_validate_json(path.read_bytes())
                if path.name != f"{record.owner}.json":
                    raise ValueError(
                        "Execution control filename does not match its owner"
                    )
                session = ContainerSession(self, record.spec, ())
                session.owner = record.owner
                session.name = f"tradingdev-strategy-{record.owner}"
                session.container_id = record.container_id
                session.control_path = path
                session._daemon_id = record.daemon_id
                session._lease = lease
                session._record_owned = True
                session._create_attempted = True
                session._create_acknowledged = record.container_id is not None
                session.close()
            except (OSError, ValueError) as error:
                raise IsolatedRuntimeError(
                    "runtime_control_invalid", str(error), control_path=str(path)
                ) from error
            finally:
                lease.release()
                if not path.exists():
                    path.with_suffix(".lock").unlink(missing_ok=True)

    def command(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Run a bounded daemon-control command, never strategy code on the host."""
        try:
            return subprocess.run(
                [*self._prefix, *arguments],
                env=self._environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=_CONTROL_TIMEOUT,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise IsolatedRuntimeError("runtime_unavailable", str(error)) from error

    def engine_identity(self) -> str:
        """Validate capabilities and identify the daemon owning execution units."""
        try:
            info = self.command("info", "--format", "{{json .}}")
            if info.returncode:
                raise ValueError(info.stderr.strip())
            engine = json.loads(info.stdout)
            if (
                not isinstance(engine, dict)
                or engine.get("OSType") != "linux"
                or engine.get("CgroupVersion") != "2"
                or not all(
                    engine.get(name) is True
                    for name in ("MemoryLimit", "CpuCfsQuota", "PidsLimit")
                )
            ):
                raise ValueError(
                    "Linux Docker with cgroup v2 resource controls is required"
                )
            identity = engine.get("ID")
            if not isinstance(identity, str) or not identity:
                raise ValueError("The Docker daemon must expose its stable identity")
            return identity
        except (ValueError, TypeError, IsolatedRuntimeError) as error:
            raise IsolatedRuntimeError(
                "runtime_unavailable", str(error), cleanup_verified=True
            ) from error

    def resolve(
        self, *, image: str | None = None, limits: RuntimeLimits | None = None
    ) -> RuntimeSpec:
        """Require the supported Linux controls; never pull or fall back silently."""
        try:
            self.engine_identity()
            selected = image or self._environment.get(
                "TRADINGDEV_STRATEGY_IMAGE",
                f"tradingdev-strategy-runtime:{version('tradingdev')}",
            )
            inspected = self.command(
                "image", "inspect", "--format", "{{.Id}}", selected
            )
            if inspected.returncode:
                raise ValueError(
                    f"Strategy runtime image {selected!r} is unavailable; "
                    "build the Dockerfile strategy-runtime target first"
                )
            effective_limits = limits
            if effective_limits is None:
                effective_limits = RuntimeLimits.model_validate_json(
                    self._environment.get("TRADINGDEV_RUNTIME_LIMITS", "{}")
                )
            return RuntimeSpec(
                image_id=inspected.stdout.strip(), limits=effective_limits
            )
        except (ValueError, TypeError, IsolatedRuntimeError) as error:
            raise IsolatedRuntimeError(
                "runtime_unavailable", str(error), cleanup_verified=True
            ) from error

    @contextmanager
    def open(
        self, spec: RuntimeSpec, command: Sequence[str]
    ) -> Iterator[ContainerSession]:
        """Yield a session and verify cleanup on success, failure and interruption."""
        session = ContainerSession(self, spec, command)
        try:
            session.start()
            yield session
        except BaseException as error:
            try:
                session.close()
            except IsolatedRuntimeError as cleanup_error:
                raise cleanup_error from error
            if isinstance(error, IsolatedRuntimeError):
                error.cleanup_verified = True
            raise
        else:
            session.close()


class ContainerSession:
    """A private PID namespace with bounded, sequential line-oriented messages."""

    def __init__(
        self, runtime: DockerRuntime, spec: RuntimeSpec, command: Sequence[str]
    ) -> None:
        self._runtime = runtime
        self.spec = spec
        self.owner = uuid4().hex
        self.name = f"tradingdev-strategy-{self.owner}"
        self.container_id: str | None = None
        self.control_path = runtime.control_root / f"{self.owner}.json"
        self._daemon_id = runtime.engine_identity()
        self._lease = FileLock(
            self.control_path.with_suffix(".lock"), thread_local=False
        )
        self._command = list(command)
        self._create_attempted = False
        self._create_acknowledged = False
        self._record_owned = False
        self._closed = False
        self._process: subprocess.Popen[bytes] | None = None
        self._selector = selectors.DefaultSelector()
        self._diagnostics = bytearray()
        self._diagnostics_truncated = False
        self._deadline = 0.0

    def _record(self) -> None:
        self.control_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._lease.is_locked:
            try:
                self._lease.acquire(timeout=0)
            except Timeout as error:
                raise IsolatedRuntimeError(
                    "runtime_identity_busy",
                    "Execution identity is already in use",
                    cleanup_verified=True,
                ) from error
        temporary = self.control_path.with_suffix(".tmp")
        if not self._record_owned:
            if self.control_path.exists() or temporary.exists():
                self._lease.release()
                raise IsolatedRuntimeError(
                    "runtime_identity_busy",
                    "Execution identity already has a record",
                    cleanup_verified=True,
                )
            self._record_owned = True
        temporary.write_text(
            _ControlRecord(
                owner=self.owner,
                container_id=self.container_id,
                daemon_id=self._daemon_id,
                spec=self.spec,
            ).model_dump_json(),
            encoding="utf-8",
        )
        temporary.replace(self.control_path)

    def start(self) -> None:
        if not self._command or self._create_attempted or self._closed:
            raise ValueError("A new execution session and command are required")
        limits = self.spec.limits
        self._record()
        arguments = [
            "create",
            "--pull",
            "never",
            "--rm",
            "--interactive",
            "--name",
            self.name,
            "--label",
            f"{_OWNER_LABEL}={self.owner}",
            "--user",
            "0:0",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "SETUID",
            "--cap-add",
            "SETGID",
            "--security-opt",
            "no-new-privileges=true",
            "--read-only",
            "--network",
            "none",
            "--ipc",
            "private",
            "--cgroupns",
            "private",
            "--no-healthcheck",
            "--log-driver",
            "none",
            "--memory",
            str(limits.memory_bytes),
            "--memory-swap",
            str(limits.memory_bytes),
            "--cpus",
            str(limits.cpus),
            "--pids-limit",
            str(limits.processes),
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,nodev,size={limits.temporary_bytes},mode=1777",
            "--shm-size",
            "16777216",
            "--ulimit",
            "core=0:0",
            "--workdir",
            "/tmp",
            "--entrypoint",
            _IMAGE_PYTHON,
        ]
        threads = str(max(1, math.ceil(limits.cpus)))
        environment = {
            "HOME": "/tmp",
            "TMPDIR": "/tmp",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "NUMBA_CACHE_DIR": "/tmp/numba",
            "XDG_CACHE_HOME": "/tmp/cache",
            "OPENBLAS_NUM_THREADS": threads,
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
        }
        for key, value in environment.items():
            arguments.extend(("--env", f"{key}={value}"))
        arguments.extend(
            (
                self.spec.image_id,
                "-I",
                "-u",
                "-m",
                "tradingdev.adapters.execution.container_init",
                "--timeout",
                str(limits.max_seconds),
                "--",
                *self._command,
            )
        )
        self._create_attempted = True
        created = self._runtime.command(*arguments)
        if created.returncode or not re.fullmatch(
            r"[0-9a-f]{64}", created.stdout.strip()
        ):
            raise IsolatedRuntimeError("runtime_start_failed", created.stderr.strip())
        self.container_id = created.stdout.strip()
        self._create_acknowledged = True
        self._record()
        self._deadline = time.monotonic() + limits.max_seconds
        self._process = subprocess.Popen(
            [
                *self._runtime._prefix,
                "start",
                "--attach",
                "--interactive",
                self.container_id,
            ],
            env=self._runtime._environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self._process.stdin and self._process.stdout and self._process.stderr
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            os.set_blocking(stream.fileno(), False)
        self._selector.register(self._process.stdout, selectors.EVENT_READ, "output")
        self._selector.register(
            self._process.stderr, selectors.EVENT_READ, "diagnostic"
        )

    def _diagnostic(self, value: bytes) -> None:
        self._diagnostics.extend(value)
        excess = len(self._diagnostics) - self.spec.limits.diagnostic_bytes
        if excess > 0:
            del self._diagnostics[:excess]
            self._diagnostics_truncated = True

    def _failure(self, code: str, message: str) -> IsolatedRuntimeError:
        diagnostic = self._diagnostics.decode("utf-8", errors="replace").strip()
        if diagnostic:
            prefix = (
                "[earlier diagnostics truncated] "
                if self._diagnostics_truncated
                else ""
            )
            message += f"; {prefix}{diagnostic}"
        return IsolatedRuntimeError(code, message, control_path=str(self.control_path))

    def exchange(self, message: bytes) -> bytes:
        """Write and read without blocking on a full input pipe or unbounded output."""
        process = self._process
        if self._closed or process is None or process.stdin is None:
            raise RuntimeError("Execution session is not open")
        if b"\n" in message or len(message) > self.spec.limits.message_bytes:
            raise ValueError("An execution message must be one bounded line")
        outgoing = memoryview(message + b"\n")
        output = bytearray()
        self._selector.register(process.stdin, selectors.EVENT_WRITE, "input")
        try:
            while True:
                remaining = self._deadline - time.monotonic()
                if remaining <= 0:
                    raise self._failure(
                        "runtime_timeout", "Strategy execution timed out"
                    )
                events = self._selector.select(remaining)
                # Preserve the final diagnostic when stderr and stdout EOF arrive
                # together, without waiting for an unbounded diagnostic stream.
                events.sort(key=lambda event: event[0].data != "diagnostic")
                for key, _events in events:
                    if key.data == "input":
                        try:
                            count = os.write(key.fd, outgoing[:65536])
                        except BrokenPipeError as error:
                            raise self._failure(
                                "runtime_exit", "Worker input closed"
                            ) from error
                        outgoing = outgoing[count:]
                        if not outgoing:
                            self._selector.unregister(process.stdin)
                        continue
                    data = os.read(key.fd, 65536)
                    if key.data == "diagnostic":
                        if data:
                            self._diagnostic(data)
                        else:
                            self._selector.unregister(key.fileobj)
                        continue
                    if not data:
                        raise self._failure(
                            "runtime_exit", "Worker exited without a complete response"
                        )
                    output.extend(data)
                    if len(output) > self.spec.limits.message_bytes + 1 or (
                        len(output) > self.spec.limits.message_bytes
                        and b"\n" not in output
                    ):
                        raise self._failure(
                            "runtime_output_limit",
                            "Worker response exceeds its byte limit",
                        )
                    if b"\n" in output:
                        line, rest = output.split(b"\n", maxsplit=1)
                        if outgoing or rest:
                            raise self._failure(
                                "invalid_runtime_response", "Unsolicited worker output"
                            )
                        return bytes(line)
        finally:
            with suppress(KeyError):
                self._selector.unregister(process.stdin)

    def _remove_owned(self) -> None:
        if self._runtime.engine_identity() != self._daemon_id:
            raise RuntimeError("Docker daemon changed; cannot verify the original unit")
        listing = self._runtime.command(
            "ps",
            "--all",
            "--quiet",
            "--no-trunc",
            "--filter",
            f"label={_OWNER_LABEL}={self.owner}",
        )
        if listing.returncode:
            raise RuntimeError(listing.stderr.strip())
        identifiers = listing.stdout.split()
        if len(identifiers) > 1 or any(
            not re.fullmatch(r"[0-9a-f]{64}", item) for item in identifiers
        ):
            raise RuntimeError("Ambiguous execution-unit ownership")
        if identifiers:
            identity = identifiers[0]
            if self.container_id is not None and identity != self.container_id:
                raise RuntimeError("Execution-unit identity changed")
            self._create_acknowledged = True
            # IDs are immutable. The label lookup never authorizes deleting a name
            # that might have been reassigned after this inspection.
            self._runtime.command("rm", "--force", "--volumes", identity)
        deadline = time.monotonic() + _CONTROL_TIMEOUT
        while True:
            verified = self._runtime.command(
                "ps",
                "--all",
                "--quiet",
                "--no-trunc",
                "--filter",
                f"label={_OWNER_LABEL}={self.owner}",
            )
            if verified.returncode:
                raise RuntimeError("Execution-unit removal could not be verified")
            if not verified.stdout.strip():
                break
            if verified.stdout.split() != identifiers:
                raise RuntimeError("Execution-unit identity changed during cleanup")
            if time.monotonic() >= deadline:
                raise RuntimeError("Execution-unit removal did not finish")
            # Auto-remove may already be in progress when the explicit rm arrives.
            time.sleep(0.05)
        if not self._create_acknowledged:
            raise RuntimeError(
                "The create request was not acknowledged; retain its control record"
            )

    def close(self) -> None:
        """Release attach backpressure, then remove and verify the owned unit."""
        if self._closed:
            return
        if not self._record_owned and not self._create_attempted:
            self._selector.close()
            self._closed = True
            return
        if self._create_attempted and not self._lease.is_locked:
            try:
                self._lease.acquire(timeout=0)
            except Timeout as error:
                raise IsolatedRuntimeError(
                    "runtime_cleanup_failed",
                    "Another controller owns cleanup",
                    control_path=str(self.control_path),
                ) from error
        errors: list[BaseException] = []
        self._selector.close()
        if self._process is not None:
            # An attach client blocked writing to our undrained pipe can keep
            # Docker's remove operation waiting for I/O even after SIGKILL.
            # Only this trusted, directly owned local client runs on the host.
            try:
                for stream in (
                    self._process.stdin,
                    self._process.stdout,
                    self._process.stderr,
                ):
                    if stream is not None:
                        stream.close()
                if self._process.poll() is None:
                    self._process.terminate()
                try:
                    self._process.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
            except BaseException as error:
                errors.append(error)
        try:
            if self._create_attempted:
                self._remove_owned()
        except BaseException as error:
            errors.append(error)
        if not errors:
            try:
                self.control_path.with_suffix(".tmp").unlink(missing_ok=True)
                self.control_path.unlink(missing_ok=True)
                self._lease.release()
                self.control_path.with_suffix(".lock").unlink(missing_ok=True)
            except OSError as error:
                errors.append(error)
        self._lease.release()
        if errors:
            raise IsolatedRuntimeError(
                "runtime_cleanup_failed",
                "; ".join(f"{type(error).__name__}: {error}" for error in errors)
                + f"; retained {self.control_path}",
                control_path=str(self.control_path),
            ) from BaseExceptionGroup("Isolated runtime cleanup failed", errors)
        self._closed = True
