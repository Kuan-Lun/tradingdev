"""Local successful snapshot evidence; never accept a hand-written test summary."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir

from tooling.environment import require_strategy_runtime
from tooling.presentations import _toolchain
from tooling.process_guard import run_captured

_VERSION = 1
# These are supplied by the snapshot runner, not inherited test settings.
_MANAGED_ENV = frozenset(
    {
        "PWD",
        "OLDPWD",
        "SHLVL",
        "_",
        "TMPDIR",
        "TMP",
        "TEMP",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "TRADINGDEV_CHECK_PYTHON",
        "TRADINGDEV_CHECK_SNAPSHOT",
        "TRADINGDEV_CHECK_SNAPSHOT_TIMEOUT",
    }
)
# Host tracing identities change between invocations without changing offline
# test behavior. Keep permission/sandbox flags and all unknown settings.
_TRACE_ENV = frozenset({"CODEX_TOOL_CALL_ID", "CODEX_SESSION_ID", "CODEX_THREAD_ID"})


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_identity(path: Path) -> tuple[str, int, int]:
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    return str(resolved), info.st_size, info.st_mtime_ns


def _installed_file(path: Path) -> tuple[str, int, int, int]:
    info = path.stat()
    return str(path), info.st_mode, info.st_size, info.st_mtime_ns


def _node_installation(root: Path) -> str:
    entries = []
    for directory, directories, files in os.walk(root):
        for name in directories:
            if (Path(directory) / name).is_symlink():
                raise RuntimeError("Cannot identify linked package directories")
        for name in files:
            entries.append(_installed_file(Path(directory) / name))
    return _digest(sorted(entries))


def environment_fingerprint(root: Path, profile: str, env: dict[str, str]) -> str:
    """Measure the local interpreter, installed tools, browser and runtime image.

    Values of ambient settings are hashed, never written to the receipt. This is
    local check evidence, not a signature or a claim of hermetic execution.
    """

    def command(arguments: list[str], cwd: Path = root) -> str:
        result = run_captured(arguments, cwd=cwd, env=env, timeout=15)
        result.check_returncode()
        return result.stdout.strip()

    binaries = {}
    for name in ("git", "uv", "node", *(("docker",) if profile == "full" else ())):
        executable = shutil.which(name, path=env.get("PATH"))
        if executable is None:
            raise RuntimeError(f"Cannot identify {name} for check reuse")
        binaries[name] = (
            _file_identity(Path(executable)),
            command([executable, "--version"]),
        )
    toolchain = _toolchain(root)
    npm_lock = (toolchain / "node_modules/.package-lock.json").read_bytes()
    if env.get("WORKERS_CI") == "1":
        # That runtime extracts a separate browser; no local reuse proof yet.
        raise RuntimeError("Check reuse is unavailable for the cloud browser")
    browser = env.get("TRADINGDEV_PRESENTATION_BROWSER") or command(
        [
            "node",
            "--input-type=module",
            "-e",
            "import p from 'puppeteer-core'; "
            "process.stdout.write(p.executablePath('chrome'));",
        ],
        cwd=toolchain,
    )
    packages = []
    for dist in importlib.metadata.distributions():
        if dist.files is None:
            raise RuntimeError("An installed Python package has no file manifest")
        files = [
            _installed_file(Path(str(dist.locate_file(file))))
            for file in dist.files
            if "__pycache__" not in file.parts and file.suffix != ".pyc"
        ]
        packages.append((dist.metadata["Name"], dist.version, _digest(sorted(files))))
    runtime = None
    if profile == "full":
        image = require_strategy_runtime()
        output = command(["docker", "info", "--format", "{{json .}}"])
        info = json.loads(output)
        engine_fields = (
            "ID",
            "ServerVersion",
            "OSType",
            "Architecture",
            "KernelVersion",
            "Driver",
            "CgroupDriver",
            "CgroupVersion",
            "SecurityOptions",
            "NCPU",
            "MemTotal",
        )
        if not isinstance(info, dict) or any(key not in info for key in engine_fields):
            raise RuntimeError("Cannot identify the Docker engine for check reuse")
        runtime = {
            "image": image,
            "engine": {key: info[key] for key in engine_fields},
        }
    controller = hashlib.sha256()
    for source in sorted(Path(__file__).resolve().parents[1].rglob("*.py")):
        if "node_modules" not in source.parts:
            controller.update(
                str(source.relative_to(Path(__file__).resolve().parents[2])).encode()
            )
            controller.update(source.read_bytes())
    return _digest(
        {
            "python": (sys.version, _file_identity(Path(sys.executable))),
            "platform": (
                platform.platform(),
                platform.machine(),
                os.cpu_count(),
                os.process_cpu_count(),
            ),
            "packages": sorted(packages),
            "controller": controller.hexdigest(),
            "temporary_filesystem": Path(gettempdir()).stat().st_dev,
            "commands": binaries,
            "npm_lock": hashlib.sha256(npm_lock).hexdigest(),
            "node_installation": _node_installation(toolchain / "node_modules"),
            "browser": (_file_identity(Path(browser)), command([browser, "--version"])),
            "runtime": runtime,
            "settings": {
                key: value
                for key, value in env.items()
                if key not in _MANAGED_ENV | _TRACE_ENV
            },
        }
    )


def receipt_path(root: Path, profile: str) -> Path:
    if profile not in {"full", "tooling"}:
        raise ValueError(f"No reusable test profile: {profile}")
    common = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        text=True,
    ).strip()
    return Path(common) / "tradingdev-checks" / f"{profile}.json"


def read_receipt(path: Path) -> dict[str, object] | None:
    """Load untrusted prior data; matching_receipt must verify applicability."""
    try:
        value = json.loads(path.read_text())
    except OSError, ValueError:
        return None
    return dict(value) if isinstance(value, dict) else None


def matching_receipt(
    value: dict[str, object] | None, *, profile: str, inputs: str, environment: str
) -> dict[str, object] | None:
    if value is None or any(
        value.get(key) != expected
        for key, expected in {
            "version": _VERSION,
            "state": "passed",
            "profile": profile,
            "inputs": inputs,
            "environment": environment,
            "command": ["bash", f"scripts/check-{profile}.sh"],
        }.items()
    ):
        return None
    tree = value.get("tree")
    if (
        not isinstance(tree, str)
        or len(tree) not in {40, 64}
        or any(c not in "0123456789abcdef" for c in tree)
    ):
        return None
    return dict(value)


def write_receipt(
    path: Path,
    *,
    state: str,
    tree: str,
    profile: str,
    inputs: str,
    environment: str | None,
) -> None:
    value = {
        "version": _VERSION,
        "state": state,
        "tree": tree,
        "profile": profile,
        "inputs": inputs,
        "environment": environment,
        "command": ["bash", f"scripts/check-{profile}.sh"],
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
