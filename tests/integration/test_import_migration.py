"""Import migration guardrails."""

from __future__ import annotations

import ast
from pathlib import Path


def test_source_and_tests_do_not_reference_legacy_imports() -> None:
    legacy_modules = {"quant_backtest", "mcp_server"}
    offenders: list[str] = []

    for root in (Path("src"), Path("tests")):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                else:
                    continue
                if any(name.split(".")[0] in legacy_modules for name in modules):
                    offenders.append(f"{path}:{node.lineno}")

    assert offenders == []
