"""Import migration guardrails."""

from __future__ import annotations

from pathlib import Path


def test_source_and_tests_do_not_reference_legacy_imports() -> None:
    legacy_package = "quant_" + "backtest"
    legacy_server = "mcp_" + "server"
    offenders: list[str] = []

    for root in (Path("src"), Path("tests")):
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if legacy_package in text or legacy_server in text:
                offenders.append(str(path))

    assert offenders == []
