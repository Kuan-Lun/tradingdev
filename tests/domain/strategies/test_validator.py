"""Import policy and actionable diagnostics for generated strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tradingdev.domain.strategies.validator import StrategyValidator

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("root", "member"),
    [
        ("__future__", "annotations"),
        ("collections", "deque"),
        ("dataclasses", "dataclass"),
        ("enum", "Enum"),
        ("statistics", "mean"),
        ("typing_extensions", "override"),
    ],
)
@pytest.mark.parametrize("from_import", [False, True])
def test_import_guidance_includes_previously_omitted_allowed_modules(
    tmp_path: Path, root: str, member: str, from_import: bool
) -> None:
    source = tmp_path / "strategy.py"
    validator = StrategyValidator()
    statement = f"from {root} import {member}" if from_import else f"import {root}"
    source.write_text(statement + "\n", encoding="utf-8")
    assert validator.static_policy_scan(source) == []

    source.write_text(statement + "\nimport pandas_ta\n", encoding="utf-8")
    diagnostics = validator.static_policy_scan(source)
    assert len(diagnostics) == 1
    rejected = diagnostics[0]
    assert rejected.code == "import_not_allowed"
    assert rejected.line == 2
    assert rejected.fix is not None
    assert root in rejected.fix
