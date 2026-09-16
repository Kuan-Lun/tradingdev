"""Verify explicit model-test selection in isolated, entirely offline pytest runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SUITE = """\
from pathlib import Path

import pytest


def test_offline():
    Path("offline.ran").write_text("done", encoding="utf-8")


@pytest.mark.live_llm
def test_model(pytestconfig):
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}
    Path("model.ran").write_text(provider, encoding="utf-8")
"""


def _run_selection(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    (root / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    live_llm: explicit model test\n",
        encoding="utf-8",
    )
    (root / "test_selection.py").write_text(_SUITE, encoding="utf-8")
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
        "PYTEST_PLUGINS": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.llm_plugin",
            "-p",
            "no:cacheprovider",
            "-q",
            *arguments,
            "test_selection.py",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_default_run_deselects_live_models_without_skipping(tmp_path: Path) -> None:
    result = _run_selection(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed, 1 deselected" in result.stdout
    assert "skipped" not in result.stdout
    assert (tmp_path / "offline.ran").exists()
    assert not (tmp_path / "model.ran").exists()


@pytest.mark.parametrize(
    ("provider", "options"),
    [("codex", []), ("local", ["--llm-model", "fixture-model"])],
)
def test_explicit_provider_includes_live_tests(
    tmp_path: Path, provider: str, options: list[str]
) -> None:
    result = _run_selection(tmp_path, "--llm-provider", provider, *options)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert (tmp_path / "model.ran").read_text(encoding="utf-8") == provider
    assert (tmp_path / "offline.ran").exists()


@pytest.mark.parametrize(
    ("options", "diagnostic"),
    [
        ([], "Local LLM tests require --llm-model"),
        (
            ["--llm-model", "fixture", "--llm-base-url", "https://example.com/v1"],
            "local loopback",
        ),
        (
            [
                "--llm-model",
                "fixture",
                "--llm-base-url",
                "http://localhost.example.com/v1",
            ],
            "local loopback",
        ),
        (
            [
                "--llm-model",
                "fixture",
                "--llm-base-url",
                "http://user:secret@localhost/v1",
            ],
            "local loopback",
        ),
        (
            [
                "--llm-model",
                "fixture",
                "--llm-base-url",
                "http://localhost/v1?target=remote",
            ],
            "local loopback",
        ),
    ],
    ids=["missing-model", "remote-host", "deceptive-host", "credentials", "query"],
)
def test_invalid_local_configuration_fails_before_any_test_executes(
    tmp_path: Path, options: list[str], diagnostic: str
) -> None:
    result = _run_selection(tmp_path, "--llm-provider", "local", *options)
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert diagnostic in result.stderr
    assert not (tmp_path / "offline.ran").exists()
    assert not (tmp_path / "model.ran").exists()


@pytest.mark.parametrize("timeout", ["nan", "inf", "-inf", "0", "-1"])
def test_nonfinite_or_nonpositive_model_timeout_is_rejected(
    tmp_path: Path, timeout: str
) -> None:
    result = _run_selection(
        tmp_path, "--llm-provider", "codex", f"--llm-timeout={timeout}"
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "--llm-timeout must be finite and positive" in result.stderr
    assert not (tmp_path / "offline.ran").exists()
    assert not (tmp_path / "model.ran").exists()
