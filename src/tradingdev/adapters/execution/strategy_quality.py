"""Execute generated-strategy quality tools and translate their failures."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

from tradingdev.domain.strategies.validator import diagnostic

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.domain.strategies.schemas import StrategyDiagnostic


def run_quality_gates(source_path: Path, config: Path) -> list[StrategyDiagnostic]:
    """Run the server interpreter with the supplied canonical quality policy."""
    diagnostics: list[StrategyDiagnostic] = []
    config_path = str(config)
    for command, label, timeout in (
        (
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--config",
                config_path,
                str(source_path),
            ],
            "ruff",
            30,
        ),
        (
            [
                sys.executable,
                "-m",
                "mypy",
                "--config-file",
                config_path,
                "--follow-imports=silent",
                str(source_path),
            ],
            "mypy",
            60,
        ),
    ):
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            diagnostics.append(
                diagnostic(
                    code=f"{label}_unavailable",
                    phase="quality_gate",
                    message=f"{label} quality gate could not start: {exc}",
                    fix=(
                        "Install ruff and mypy in the server's Python "
                        "environment before running "
                        "validate_strategy."
                    ),
                )
            )
            continue
        except subprocess.TimeoutExpired as exc:
            diagnostics.append(
                diagnostic(
                    code=f"{label}_timeout",
                    phase="quality_gate",
                    message=f"{label} quality gate timed out: {exc}",
                    fix=(
                        f"Make the generated strategy analyzable within "
                        f"{timeout} seconds or investigate the quality gate."
                    ),
                )
            )
            continue
        if result.returncode != 0:
            output = (result.stdout or result.stderr).strip()
            diagnostics.append(
                diagnostic(
                    code=f"{label}_failed",
                    phase="quality_gate",
                    message=f"{label} failed: {output}",
                    fix=f"Fix the {label} diagnostics and save the strategy again.",
                )
            )
    return diagnostics
