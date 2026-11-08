"""Check only the prerequisites needed by a development operation."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit


class EnvironmentCheckError(RuntimeError):
    """A prerequisite is unavailable; the requested work must not start."""


def require_packages(*names: str) -> None:
    """Check installed distributions without importing their application code."""
    missing = []
    for name in dict.fromkeys(names):
        try:
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise EnvironmentCheckError(
            f"Missing Python packages: {', '.join(missing)}. "
            "Run uv sync --locked --all-extras in this repository."
        )


def require_command(name: str, *, help_text: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise EnvironmentCheckError(f"{name} is unavailable. {help_text}")
    return executable


def require_node(*, npm: bool = False) -> str:
    """Check the compiler's host before npm installation or rendering starts."""
    help_text = "Install Node.js 18 or newer and npm before presentation setup."
    node = require_command("node", help_text=help_text)
    if npm:
        require_command("npm", help_text=help_text)
    from tooling.process_guard import run_captured

    try:
        result = run_captured(
            [node, "--version"],
            cwd=Path.cwd(),
            timeout=10,
            env=node_environment(),
        )
        result.check_returncode()
        major = int(result.stdout.strip().removeprefix("v").split(".")[0])
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise EnvironmentCheckError(f"Cannot verify Node.js. {help_text}") from error
    if major < 18:
        raise EnvironmentCheckError(f"Node.js {major} is unsupported. {help_text}")
    return node


def node_environment() -> dict[str, str]:
    """Do not let ambient Node/npm options execute code during a check."""
    return {
        key: value
        for key, value in os.environ.items()
        if key != "DEBUG"
        and not key.upper().startswith(("NODE_", "NPM_CONFIG_", "MARP_"))
    }


def require_strategy_runtime() -> str:
    """Reuse product policy without launching a container or touching user data."""
    require_command(
        "docker", help_text="Install Docker CLI and start the local daemon."
    )
    require_packages("tradingdev", "pydantic", "filelock", "psutil")
    from tradingdev.adapters.execution.container_runtime import DockerRuntime
    from tradingdev.domain.runtime import IsolatedRuntimeError

    try:
        with TemporaryDirectory(prefix="tradingdev-environment-") as directory:
            spec = DockerRuntime(Path(directory)).resolve()
    except IsolatedRuntimeError as error:
        raise EnvironmentCheckError(
            f"Strategy runtime is unavailable: {error}. "
            "Prepare the local Docker Engine and Dockerfile strategy-runtime image; "
            "TRADINGDEV_STRATEGY_IMAGE selects a custom image. "
            "See docs/development.md (container build and verification)."
        ) from error
    return spec.image_id


def require_codex() -> str:
    """Check the selected CLI's login without making a model request."""
    from tooling.codex_binary import resolve_codex_binary
    from tooling.process_guard import process_workspace, run_captured

    try:
        binary = resolve_codex_binary()
    except RuntimeError as error:
        raise EnvironmentCheckError(str(error)) from error
    try:
        with process_workspace(
            prefix="tradingdev-codex-environment-", label="Codex login probe"
        ) as root:
            result = run_captured(
                [binary, "login", "status"],
                cwd=root,
                env={**os.environ, "XDG_CACHE_HOME": str(root / "cache")},
                timeout=15,
            )
    except (OSError, subprocess.SubprocessError) as error:
        raise EnvironmentCheckError(
            "Cannot check Codex login status; check the selected CLI installation."
        ) from error
    if result.returncode:
        raise EnvironmentCheckError(
            "Codex is not logged in. "
            "Run the selected CLI with 'login', then 'login status'."
        )
    return binary


def local_service_address(base_url: str) -> tuple[str, int]:
    """Validate the same local-only endpoint for options and readiness checks."""
    diagnostic = "--llm-base-url must be a local loopback URL"
    try:
        url = urlsplit(base_url)
        port = url.port
    except ValueError as error:
        raise EnvironmentCheckError(diagnostic) from error
    if (
        url.scheme not in {"http", "https"}
        or url.hostname not in {"localhost", "127.0.0.1", "::1"}
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise EnvironmentCheckError(diagnostic)
    assert url.hostname is not None
    return url.hostname, port if port is not None else (
        443 if url.scheme == "https" else 80
    )


def require_local_service(base_url: str) -> None:
    """Probe connectivity only; model/API/tool support belongs to live scenarios."""
    address = local_service_address(base_url)
    try:
        with socket.create_connection(address, timeout=5):
            pass
    except OSError as error:
        raise EnvironmentCheckError(
            "Local model service is unreachable. "
            "Start the selected service and check --llm-base-url."
        ) from error


def check_environment(profile: str) -> None:
    """Compose prerequisites; lightweight checks never require Docker or models."""
    if profile not in {"docs", "format", "fast", "tooling", "full"}:
        raise ValueError(f"Unknown environment profile: {profile}")
    packages = ["pymarkdownlnt"]
    if profile != "format":
        packages.append("markdown-it-py")
    if profile != "docs":
        packages.append("ruff")
    if profile in {"fast", "tooling", "full"}:
        packages.extend(["mypy", "streamlit", "plotly", "jsonschema"])
    if profile in {"tooling", "full"}:
        packages.extend(["pytest", "psutil"])
    require_packages(*packages)
    if profile in {"tooling", "full"}:
        require_command("git", help_text="Install Git before running repository tests.")
        require_command("uv", help_text="Install uv before running environment tests.")
    if profile == "full":
        image = require_strategy_runtime()
        print(f"Strategy runtime prerequisites passed: {image}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile", choices=("docs", "format", "fast", "tooling", "full")
    )
    args = parser.parse_args()
    try:
        check_environment(args.profile)
    except EnvironmentCheckError as error:
        print(f"Environment check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
