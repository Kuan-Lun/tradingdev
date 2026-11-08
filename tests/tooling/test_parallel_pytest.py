"""Verify CPU policy and real xdist execution without product workloads."""

from __future__ import annotations

import getpass
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from tests import parallel_plugin
from tooling.process_guard import run_captured

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def auto_config(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config
) -> pytest.Config:
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    monkeypatch.setattr(pytestconfig.option, "numprocesses", "auto")
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    return pytestconfig


def test_apple_silicon_uses_performance_cores(
    auto_config: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = Mock(return_value=subprocess.CompletedProcess([], 0, "10\n"))
    monkeypatch.setattr(subprocess, "run", probe)
    assert parallel_plugin.pytest_xdist_auto_num_workers(auto_config) == 10
    assert probe.call_args.args[0] == [
        "/usr/sbin/sysctl",
        "-n",
        "hw.perflevel0.physicalcpu",
    ]
    assert probe.call_args.kwargs["timeout"] > 0


@pytest.mark.parametrize(
    "failure",
    [OSError("denied"), subprocess.TimeoutExpired("sysctl", 5), "0", "invalid"],
)
def test_failed_performance_probe_warns_and_uses_one_worker(
    auto_config: pytest.Config,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception | str,
) -> None:
    probe = (
        Mock(side_effect=failure)
        if isinstance(failure, Exception)
        else Mock(return_value=subprocess.CompletedProcess([], 0, failure))
    )
    monkeypatch.setattr(subprocess, "run", probe)
    with pytest.warns(pytest.PytestConfigWarning, match="using one pytest worker"):
        assert parallel_plugin.pytest_xdist_auto_num_workers(auto_config) == 1


@pytest.mark.parametrize("system", ["Linux", "Windows"])
@pytest.mark.parametrize("available, expected", [(12, 10), (2, 1), (1, 1), (None, 1)])
def test_reserve_two_available_logical_cpus(
    auto_config: pytest.Config,
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    available: int | None,
    expected: int,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(os, "process_cpu_count", lambda: available)
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: pytest.fail("sysctl"))
    assert parallel_plugin.pytest_xdist_auto_num_workers(auto_config) == expected


@pytest.mark.parametrize("override, expected", [("10", 10), ("0", 0)])
def test_explicit_environment_count_precedes_detection(
    auto_config: pytest.Config,
    monkeypatch: pytest.MonkeyPatch,
    override: str,
    expected: int,
) -> None:
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", override)
    monkeypatch.setattr(platform, "system", lambda: pytest.fail("CPU detection"))
    assert parallel_plugin.pytest_xdist_auto_num_workers(auto_config) == expected


@pytest.mark.parametrize("override", ["", "-2", "1.5", "many"])
def test_invalid_environment_count_is_rejected(
    auto_config: pytest.Config, monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", override)
    with pytest.raises(pytest.UsageError, match="nonnegative integer"):
        parallel_plugin.pytest_xdist_auto_num_workers(auto_config)


@pytest.mark.parametrize("mode, machine", [("auto", "x86_64"), ("logical", "arm64")])
def test_preserve_xdist_policy_for_intel_mac_and_explicit_logical_mode(
    auto_config: pytest.Config,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    machine: str,
) -> None:
    monkeypatch.setattr(auto_config.option, "numprocesses", mode)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    assert parallel_plugin.pytest_xdist_auto_num_workers(auto_config) is None


@pytest.fixture
def run_suite(tmp_path: Path) -> Callable[..., subprocess.CompletedProcess[str]]:
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    retained = temporary / f"pytest-of-{getpass.getuser()}" / "pytest-0"
    retained.mkdir(parents=True)
    (retained / "keep.txt").write_text("another session", encoding="utf-8")
    (tmp_path / "test_parallel.py").write_text(
        "import os, time\nfrom pathlib import Path\nimport pytest\n"
        "@pytest.mark.parametrize('case', range(8))\n"
        "def test_case(case, worker_id, pytestconfig):\n"
        "    with Path(f'{case}.ran').open('x') as output:\n"
        "        output.write(worker_id)\n"
        "    if worker_id != 'master':\n"
        "        Path(f'{worker_id}.base').write_text(\n"
        "            pytestconfig.getoption('basetemp'))\n"
        "    if case == 0 and os.environ['STALL_CASE'] == '1': time.sleep(60)\n"
        "    assert not (case == 0 and os.environ['FAIL_CASE'] == '1')\n",
        encoding="utf-8",
    )

    def run(
        mode: str,
        override: str,
        *,
        fail: bool = False,
        stall: bool = False,
        timeout: float = 30,
    ) -> subprocess.CompletedProcess[str]:
        return run_captured(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "xdist.plugin",
                "-p",
                "tests.parallel_plugin",
                "-p",
                "no:cacheprovider",
                "-n",
                mode,
                "--dist=worksteal",
                "--max-worker-restart=0",
                "-c",
                str(Path(__file__).resolve().parents[2] / "pyproject.toml"),
                "-q",
                str(tmp_path / "test_parallel.py"),
            ],
            cwd=tmp_path,
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "PYTEST_PLUGINS": "",
                "PYTEST_ADDOPTS": "",
                "PYTEST_XDIST_AUTO_NUM_WORKERS": override,
                "PYTHONDONTWRITEBYTECODE": "1",
                "FAIL_CASE": str(int(fail)),
                "STALL_CASE": str(int(stall)),
                "TMPDIR": str(temporary),
                "TMP": str(temporary),
                "TEMP": str(temporary),
            },
            timeout=timeout,
        )

    return run


@pytest.mark.parametrize(
    "mode, override, fail, workers",
    [
        ("auto", "2", False, {"gw0", "gw1"}),
        ("2", "invalid", True, {"gw0", "gw1"}),
        ("0", "invalid", False, {"master"}),
        ("auto", "0", False, {"master"}),
    ],
)
def test_real_pytest_executes_every_case_once_and_propagates_failures(
    tmp_path: Path,
    run_suite: Callable[..., subprocess.CompletedProcess[str]],
    mode: str,
    override: str,
    fail: bool,
    workers: set[str],
) -> None:
    result = run_suite(mode, override, fail=fail)
    assert result.returncode == int(fail), result.stdout + result.stderr
    records = list(tmp_path.glob("*.ran"))
    assert {path.stem for path in records} == {str(case) for case in range(8)}
    assert {path.read_text() for path in records} == workers
    _assert_session_cleanup(tmp_path, parallel=workers != {"master"})


def test_invalid_count_prevents_actual_test_execution(
    tmp_path: Path, run_suite: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    result = run_suite("auto", "invalid")
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "nonnegative integer" in result.stderr
    assert not list(tmp_path.glob("*.ran"))


def test_timed_out_workers_stop_and_remove_pytest_session_directory(
    tmp_path: Path, run_suite: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        run_suite("2", "2", stall=True, timeout=10)
    assert (tmp_path / "0.ran").exists(), "Timeout must exercise a running worker"
    _assert_session_cleanup(tmp_path, parallel=True)


def _assert_session_cleanup(root: Path, *, parallel: bool) -> None:
    basetemps = list(root.glob("*.base"))
    assert bool(basetemps) == parallel
    for record in basetemps:
        path = Path(record.read_text())
        assert path.is_relative_to(root / "temporary")
        assert not path.exists()
    assert not list((root / "temporary").glob("tradingdev-pytest-*"))
    retained = (
        root / "temporary" / f"pytest-of-{getpass.getuser()}" / "pytest-0" / "keep.txt"
    )
    assert retained.read_text() == "another session"
