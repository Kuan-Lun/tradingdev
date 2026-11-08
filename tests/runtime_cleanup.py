"""Keep generated-unit ownership evidence until temporary workspaces are safe."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.adapters.execution.container_runtime import DockerRuntime

if TYPE_CHECKING:
    from pathlib import Path


def recover_test_strategy_units(root: Path) -> None:
    """Recover abandoned direct API units; active or uncertain ownership fails."""
    for controls in root.rglob(".strategy-runtime"):
        if controls.is_symlink():
            raise RuntimeError(
                f"Symlinked strategy ownership directory; retained {root}"
            )
        if not controls.is_dir() or not any(controls.glob("*.json")):
            continue
        try:
            DockerRuntime(controls).recover_abandoned()
            if any(controls.glob("*.json")):
                raise RuntimeError("Strategy units still have active ownership leases")
        except Exception as error:
            raise RuntimeError(
                f"Generated strategy cleanup unverified; retained {root}: {error}"
            ) from error
