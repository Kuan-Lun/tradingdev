"""Test temporary workspace cleanup without contacting a real Docker daemon."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import mkdtemp

import pytest

from tests import conftest, runtime_cleanup
from tests.integration.mcp_harness import MCPWorkspace


@pytest.mark.parametrize("outcome", ["normal", "failure", "timeout"])
def test_temporary_fixture_recovers_units_before_removing_files(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    fixture = conftest._temporary_path()
    root = next(fixture)
    controls = root / "workspace/.strategy-runtime"
    controls.mkdir(parents=True)
    record = controls / "fixture.json"
    record.write_text("substitute ownership evidence")

    class Runtime:
        def __init__(self, control_root: Path) -> None:
            assert control_root == controls

        def recover_abandoned(self) -> None:
            assert root.exists() and record.exists()
            record.unlink()

    monkeypatch.setattr(runtime_cleanup, "DockerRuntime", Runtime)
    try:
        if outcome == "normal":
            with pytest.raises(StopIteration):
                next(fixture)
        else:
            error = TimeoutError if outcome == "timeout" else ValueError
            with pytest.raises(error):
                fixture.throw(error("Test body failure"))
        assert not root.exists()
    finally:
        fixture.close()
        if root.exists():
            shutil.rmtree(root)  # Only substitute records exist in this test.


@pytest.mark.parametrize("uncertain", ["daemon_failure", "active_lease"])
def test_temporary_fixture_retains_unverified_ownership(
    monkeypatch: pytest.MonkeyPatch, uncertain: str
) -> None:
    fixture = conftest._temporary_path()
    root = next(fixture)
    controls = root / ".strategy-runtime"
    controls.mkdir()
    record = controls / "fixture.json"
    record.write_text("substitute ownership evidence")

    class Runtime:
        def __init__(self, control_root: Path) -> None:
            assert control_root == controls

        def recover_abandoned(self) -> None:
            if uncertain == "daemon_failure":
                raise RuntimeError("Substitute daemon unavailable")

    monkeypatch.setattr(runtime_cleanup, "DockerRuntime", Runtime)
    try:
        with pytest.raises(RuntimeError, match="cleanup unverified; retained"):
            next(fixture)
        assert record.read_text() == "substitute ownership evidence"
    finally:
        fixture.close()
        shutil.rmtree(root)  # No real execution unit was created.


def test_model_server_inherits_operator_runtime_but_confines_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(mkdtemp(prefix="tradingdev-runtime-env-test-"))
    monkeypatch.setenv("TRADINGDEV_STRATEGY_IMAGE", "strategy:test")
    monkeypatch.setenv("TRADINGDEV_RUNTIME_CONTROL_ROOT", "/must-not-write-here")
    monkeypatch.setenv("DOCKER_CONTEXT", "local-test-daemon")
    try:
        env = MCPWorkspace(root).environment()
        assert env["TRADINGDEV_STRATEGY_IMAGE"] == "strategy:test"
        assert env["DOCKER_CONTEXT"] == "local-test-daemon"
        assert Path(env["TRADINGDEV_RUNTIME_CONTROL_ROOT"]).is_relative_to(root)
    finally:
        shutil.rmtree(root)
