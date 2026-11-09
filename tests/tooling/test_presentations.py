"""Presentation routing and validation are deterministic and clean up processes."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING
from xml.etree import ElementTree
from zipfile import ZipFile

import psutil
import pytest
from filelock import FileLock, Timeout
from tooling import presentation_html, presentations
from tooling.environment import EnvironmentCheckError
from tooling.presentation_errors import PresentationError
from tooling.process_guard import (
    CLEANUP_UNVERIFIED_EXIT_CODE,
    ProcessCleanupError,
    run_captured,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest import MonkeyPatch


def _tasks(*tasks: dict[str, object], **settings: object) -> bytes:
    return json.dumps({"version": "2.0.0", "tasks": tasks, **settings}).encode()


@pytest.mark.parametrize("action", ["build", "watch"])
def test_vscode_task_installs_missing_python_dependency_before_entrypoint(
    tmp_path: Path, action: str
) -> None:
    repository = Path(__file__).resolve().parents[2]
    tasks = json.loads((repository / ".vscode/tasks.json").read_text())["tasks"]
    task = next(task for task in tasks if task["label"] == f"slides: {action}")
    assert task == presentations.presentation_task(action)
    wheel = tmp_path / "slides_fixture-1.0-py3-none-any.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("slides_fixture.py", "READY = True\n")
        archive.writestr(
            "slides_fixture-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: slides-fixture\nVersion: 1.0\n",
        )
        archive.writestr(
            "slides_fixture-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr("slides_fixture-1.0.dist-info/RECORD", "")
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        '[project]\nname = "slides-task-fixture"\nversion = "1.0"\n'
        'requires-python = ">=3.14,<3.15"\ndependencies = ["slides-fixture"]\n'
        f'[tool.uv.sources]\nslides-fixture = {{path = "{wheel.name}"}}\n'
    )
    module = tmp_path / "tooling/presentations.py"
    module.parent.mkdir()
    module.write_text(
        "import sys\nfrom pathlib import Path\nfrom slides_fixture import READY\n"
        "assert READY\nPath('started').write_text(sys.argv[1])\n"
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("UV_") and key not in {"VIRTUAL_ENV", "PYTHONPATH"}
    }
    environment.update(
        UV_OFFLINE="1",
        UV_CACHE_DIR=str(tmp_path / "cache"),
        UV_PYTHON=sys.executable,
        UV_PYTHON_DOWNLOADS="never",
    )
    for arguments in (["lock"], ["venv", ".venv"]):
        run_captured(
            ["uv", *arguments], cwd=tmp_path, env=environment, timeout=30
        ).check_returncode()
    lock = tmp_path / "uv.lock"
    locked = lock.read_bytes()
    command = [task["command"], *task["args"]]
    for _ in range(2):
        result = run_captured(command, cwd=tmp_path, env=environment, timeout=30)
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "started").read_text() == action
        assert lock.read_bytes() == locked
        (tmp_path / "started").unlink()
    manifest.write_text(
        manifest.read_text().replace('version = "1.0"', 'version = "2.0"')
    )
    result = run_captured(command, cwd=tmp_path, env=environment, timeout=30)
    assert result.returncode != 0
    assert not (tmp_path / "started").exists()
    assert lock.read_bytes() == locked


def test_registered_tasks_allow_only_known_changes_and_unchanged_other_tasks() -> None:
    other: dict[str, object] = {"label": "tests", "command": "existing"}
    watch = presentations.presentation_task("watch")
    build = presentations.presentation_task("build")
    assert presentations.presentation_change_allowed(
        ".vscode/tasks.json", _tasks(other, watch), _tasks(other, watch, build)
    )
    assert presentations.presentation_change_allowed(
        ".vscode/tasks.json", None, _tasks(watch, build)
    )
    assert not presentations.presentation_change_allowed(
        ".vscode/tasks.json", _tasks(other, watch), _tasks(watch, build)
    )
    assert not presentations.presentation_change_allowed(
        ".vscode/tasks.json", _tasks(watch), _tasks(watch, {"label": "unknown"})
    )


@pytest.mark.parametrize(
    "update",
    [
        {"command": "sh"},
        {"args": ["-c", "touch marker"]},
        {"type": "shell"},
        {"dependsOn": "danger"},
        {"linux": {"command": "danger"}},
        {"windows": {"command": "danger"}},
        {"osx": {"command": "danger"}},
        {"options": {"cwd": "${workspaceFolder}", "env": {"NODE_OPTIONS": "danger"}}},
        {"command": "${command:evil}"},
        {"unknown": True},
        {"runOptions": {"runOn": "folderOpen", "instanceLimit": 2}},
    ],
)
def test_task_label_does_not_authorize_arbitrary_execution(
    update: dict[str, object],
) -> None:
    valid = presentations.presentation_task("watch")
    assert not presentations.presentation_change_allowed(
        ".vscode/tasks.json", _tasks(valid), _tasks({**valid, **update})
    )


@pytest.mark.parametrize(
    "content",
    [
        b'{"version":"2.0.0","tasks":[],"tasks":[]}',
        b'{"version":"2.0.0","tasks":[],"options":{"env":{"PATH":"bad"}}}',
        b'{"version":"2.0.0","tasks":[],"command":"danger"}',
        b'{"version":"2.0.0","tasks":[],"inputs":[]}',
        b"\xff",
        b"[]",
    ],
)
def test_ambiguous_or_global_task_configuration_stays_code(content: bytes) -> None:
    assert not presentations.presentation_change_allowed(
        ".vscode/tasks.json", None, content
    )


def test_duplicate_task_labels_and_unsafe_old_task_are_rejected() -> None:
    watch = presentations.presentation_task("watch")
    assert not presentations.presentation_change_allowed(
        ".vscode/tasks.json", None, _tasks(watch, watch)
    )
    assert not presentations.presentation_change_allowed(
        ".vscode/tasks.json", _tasks({**watch, "command": "danger"}), _tasks(watch)
    )


def test_registered_paths_do_not_allow_arbitrary_decks_or_compiler_changes() -> None:
    for path in presentations.PRESENTATION_PATHS:
        assert presentations.is_presentation_path(path)
        assert presentations.presentation_change_allowed(path, b"before", b"after")
    assert presentations.presentation_change_allowed("docs/.nojekyll", None, b"")
    assert not presentations.presentation_change_allowed(
        "docs/.nojekyll", None, b"code"
    )
    for path in (
        "tooling/marp/package-lock.json",
        "tooling/presentations.py",
        "tooling/presentation_html.py",
        "tooling/presentation_bundle.py",
        "tooling/presentation_errors.py",
        "scripts/check-presentations.sh",
        "scripts/presentations.sh",
    ):
        assert presentations.is_presentation_path(path)
        assert not presentations.presentation_change_allowed(path, b"old", b"new")
    assert not presentations.presentation_change_allowed(
        "docs/other/index.html", None, b"x"
    )


def _html(extra: str = "") -> str:
    return (
        "<html><head><style>section {color:red}</style></head><body>"
        '<section id="1" data-marpit-pagination="1" data-marpit-pagination-total="2">'
        '<a href="https://external.invalid/citation">Citation</a></section>'
        '<section id="2" data-marpit-pagination="2" data-marpit-pagination-total="2">'
        f"{extra}</section></body></html>"
    )


_ENTRY = "# Editing the deck\n\nUse Tasks: Run Task → slides: watch.\n"


def test_static_html_accepts_embedded_resources_and_external_citations() -> None:
    assert (
        presentation_html.check_html(
            _html(
                '<img src="data:image/png;base64,AA==">'
                "<style>a {background:url(#a)}</style>"
            )
        )
        == 2
    )


@pytest.mark.parametrize(
    "extra",
    [
        '<img src="https://external.invalid/a.png">',
        '<img src="//external.invalid/a.png">',
        '<img srcset="data:image/png;base64,AA== 1x, '
        'https://external.invalid/a.png 2x">',
        '<img src="relative.png">',
        '<script src="https://external.invalid/a.js"></script>',
        '<link rel="stylesheet" href="https://external.invalid/a.css">',
        '<style>@import "https://external.invalid/a.css";</style>',
        "<style>section {background:url(https://external.invalid/a.png)}</style>",
        "<style>section {background:image-set("
        '"https://external.invalid/a.png" 1x)}</style>',
        r"<style>section {background:u\72l(\68ttps://external.invalid/a.png)}</style>",
        '<svg><image href="https://external.invalid/a.svg"/></svg>',
        '<script>new WebSocket("ws://localhost:1234")</script>',
        "<script>location.reload()</script>",
        '<img src="data:x" onload="fetch(1)">',
        '<iframe srcdoc="code"></iframe>',
        '<meta http-equiv="refresh" content="0;url=https://external.invalid">',
        '<a href="javascript:alert(1)">bad</a>',
    ],
)
def test_loaded_resources_and_watch_injections_are_rejected(extra: str) -> None:
    with pytest.raises(PresentationError):
        presentation_html.check_html(_html(extra))


@pytest.mark.parametrize(
    "content",
    [
        _html().replace('id="2"', 'id="3"'),
        _html().replace('pagination="2"', 'pagination="1"'),
        _html().replace('total="2"', 'total="3"'),
        "<html></html>",
    ],
)
def test_html_page_counts_and_section_identity_must_agree(content: str) -> None:
    with pytest.raises(PresentationError):
        presentation_html.check_html(content)


def _deck(root: Path) -> Path:
    deck = root / presentations.DECK
    deck.mkdir(parents=True)
    (deck / "slides.md").write_text("# Slides\n")
    (deck / "theme.css").write_text("section {color:red}")
    (deck / "README.md").write_text(_ENTRY)
    (deck / "index.html").write_text(_html())
    return deck


def _fake_toolchain(root: Path) -> Path:
    toolchain = root / "tooling/marp"
    cli = "node_modules/@marp-team/marp-cli"
    package: dict[str, object] = {
        "version": "4.5.1",
        "resolved": "https://registry.npmjs.org/cli.tgz",
        "integrity": "sha512-test",
    }
    (toolchain / cli).mkdir(parents=True)
    identity = {"name": "tradingdev-presentations", "version": "1.0.0"}
    manifest = {**identity, "dependencies": {"@marp-team/marp-cli": "4.5.1"}}
    (toolchain / "package.json").write_text(json.dumps(manifest))
    for name in ("build.mjs", "render-diagrams.mjs"):
        (toolchain / name).write_text("// test compiler\n")
    (toolchain / "package-lock.json").write_text(
        json.dumps(
            {**identity, "lockfileVersion": 3, "packages": {"": manifest, cli: package}}
        )
    )
    (toolchain / "node_modules/.package-lock.json").write_text(
        json.dumps({"packages": {cli: package}})
    )
    (toolchain / cli / "package.json").write_text('{"version":"4.5.1"}')
    return toolchain


@pytest.mark.parametrize("action", ["build", "watch"])
@pytest.mark.parametrize("state", ["ready", "missing", "stale"])
def test_task_prepares_compiler_once_before_build_or_watch(
    tmp_path: Path, monkeypatch: MonkeyPatch, action: str, state: str
) -> None:
    installed = _fake_toolchain(tmp_path)
    deck = _deck(tmp_path)
    if state == "missing":
        shutil.rmtree(installed / "node_modules")
    elif state == "stale":
        (installed / "node_modules/@marp-team/marp-cli/package.json").write_text(
            '{"version":"0.0.0"}'
        )
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    monkeypatch.setattr(
        presentations, "__file__", str(tmp_path / "tooling/presentations.py")
    )
    monkeypatch.setattr(presentations, "require_node", lambda **_: "/fixture/node")
    monkeypatch.setattr(shutil, "which", lambda _: "/fixture/npm")
    monkeypatch.setattr(sys, "argv", ["presentations", action, "--ensure-toolchain"])
    installations: list[Path] = []
    builds: list[str] = []

    def install(
        command: list[str], *, cwd: Path, temporary: Path, timeout: float
    ) -> None:
        assert command[1:] == ["ci", "--ignore-scripts", "--no-audit", "--no-fund"]
        assert cwd == installed
        with (
            pytest.raises(Timeout),
            FileLock(tmp_path / ".venv/.presentation-setup.lock", timeout=0),
        ):
            pytest.fail("Another installer acquired the same lock")
        installations.append(temporary)
        if (installed / "node_modules").exists():
            shutil.rmtree(installed / "node_modules")
        _fake_toolchain(tmp_path)

    def build(root: Path) -> None:
        assert presentations._toolchain(root) == installed
        builds.append((deck / "slides.md").read_text())

    def edit(seconds: float) -> None:
        if len(builds) == 1:
            (deck / "slides.md").write_text("# Updated slides\n")
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(presentations, "_execute", install)
    monkeypatch.setattr(presentations, "build", build)
    monkeypatch.setattr(time, "sleep", edit)
    assert presentations.main() == 0
    assert len(builds) == (2 if action == "watch" else 1)
    presentations.setup(tmp_path, only_if_needed=True)
    assert len(installations) == (0 if state == "ready" else 1)
    assert all(not path.exists() for path in installations)


@pytest.mark.parametrize("reason", ["invalid_lock", "external", "missing_node"])
def test_automatic_setup_does_not_install_for_unrepairable_configuration(
    tmp_path: Path, monkeypatch: MonkeyPatch, reason: str
) -> None:
    installed = _fake_toolchain(tmp_path)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    monkeypatch.setattr(presentations, "require_node", lambda **_: "/fixture/node")
    if reason == "invalid_lock":
        manifest = installed / "package.json"
        manifest.write_text(manifest.read_text().replace("4.5.1", "4.6.0"))
        message = "differs from the npm lock"
    elif reason == "external":
        external = tmp_path / "shared-compiler"
        shutil.copytree(installed, external)
        shutil.rmtree(external / "node_modules")
        monkeypatch.setenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", str(external))
        message = "external TRADINGDEV_PRESENTATION_TOOLCHAIN"
    else:

        def missing(**kwargs: object) -> str:
            raise EnvironmentCheckError("Node.js is unavailable")

        monkeypatch.setattr(presentations, "require_node", missing)
        message = "Node.js is unavailable"
    monkeypatch.setattr(
        presentations, "_execute", lambda *_, **__: pytest.fail("Installer started")
    )
    with pytest.raises(PresentationError, match=message):
        presentations.setup(tmp_path, only_if_needed=True)


@pytest.mark.parametrize("failure", ["npm", "timeout", "incomplete"])
def test_failed_automatic_install_stops_before_build_and_releases_workspace(
    tmp_path: Path, monkeypatch: MonkeyPatch, failure: str
) -> None:
    installed = _fake_toolchain(tmp_path)
    shutil.rmtree(installed / "node_modules")
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    monkeypatch.setattr(
        presentations, "__file__", str(tmp_path / "tooling/presentations.py")
    )
    monkeypatch.setattr(presentations, "require_node", lambda **_: "/fixture/node")
    monkeypatch.setattr(shutil, "which", lambda _: "/fixture/npm")
    monkeypatch.setattr(sys, "argv", ["presentations", "build", "--ensure-toolchain"])
    workspaces: list[Path] = []

    def install(
        command: list[str], *, cwd: Path, temporary: Path, timeout: float
    ) -> None:
        workspaces.append(temporary)
        if failure == "npm":
            raise PresentationError("npm failed")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(presentations, "_execute", install)
    monkeypatch.setattr(presentations, "build", lambda _: pytest.fail("Build started"))
    assert presentations.main() == 1
    assert len(workspaces) == 1 and not workspaces[0].exists()
    with FileLock(tmp_path / ".venv/.presentation-setup.lock", timeout=0):
        pass


@pytest.mark.parametrize("action", ["setup", "check", "package"])
def test_automatic_install_option_is_restricted_to_editor_actions(
    monkeypatch: MonkeyPatch, action: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["presentations", action, "--ensure-toolchain"])
    monkeypatch.setattr(
        presentations, "setup", lambda *_: pytest.fail("Installer started")
    )
    with pytest.raises(SystemExit) as error:
        presentations.main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "update",
    [
        {"dependencies": {"@marp-team/marp-cli": "4.6.0"}},
        {"dependencies": {}},
        {"devDependencies": {"new-package": "1.0.0"}},
        {"optionalDependencies": {"new-package": "1.0.0"}},
        {"peerDependencies": {"new-package": "1.0.0"}},
        {"peerDependenciesMeta": {"new-package": {"optional": True}}},
        {"name": "different-toolchain"},
        {"version": "2.0.0"},
        {"engines": {"node": ">=30"}},
        {"os": ["linux"]},
        {"cpu": ["x64"]},
        {"libc": ["glibc"]},
        {"dependencies": ["invalid"]},
    ],
)
def test_changed_root_manifest_cannot_reuse_stale_lock_and_installation(
    tmp_path: Path, monkeypatch: MonkeyPatch, update: dict[str, object]
) -> None:
    installed = _fake_toolchain(tmp_path)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    manifest = json.loads((installed / "package.json").read_text())
    (installed / "package.json").write_text(json.dumps({**manifest, **update}))
    with pytest.raises(PresentationError, match="[Rr]oot manifest"):
        presentations._toolchain(tmp_path)


@pytest.mark.parametrize("field", ["name", "version"])
@pytest.mark.parametrize("record", ["root", "top"])
def test_lock_identity_must_match_root_manifest(
    tmp_path: Path, monkeypatch: MonkeyPatch, field: str, record: str
) -> None:
    installed = _fake_toolchain(tmp_path)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    lock = json.loads((installed / "package-lock.json").read_text())
    target = lock["packages"][""] if record == "root" else lock
    target[field] = "different"
    (installed / "package-lock.json").write_text(json.dumps(lock))
    with pytest.raises(PresentationError, match=f"manifest {field}"):
        presentations._toolchain(tmp_path)


@pytest.mark.parametrize(
    "field", ["overrides", "workspaces", "bundleDependencies", "bundledDependencies"]
)
def test_unverified_root_resolution_modes_are_rejected(
    tmp_path: Path, monkeypatch: MonkeyPatch, field: str
) -> None:
    installed = _fake_toolchain(tmp_path)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    manifest = json.loads((installed / "package.json").read_text())
    manifest[field] = {}
    (installed / "package.json").write_text(json.dumps(manifest))
    with pytest.raises(PresentationError, match="Unsupported root npm"):
        presentations._toolchain(tmp_path)


def test_matching_manifest_and_lock_can_change_dependency_category(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    installed = _fake_toolchain(tmp_path)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    manifest = json.loads((installed / "package.json").read_text())
    manifest["devDependencies"] = manifest.pop("dependencies")
    lock = json.loads((installed / "package-lock.json").read_text())
    lock["packages"][""] = manifest
    (installed / "package.json").write_text(json.dumps(manifest))
    (installed / "package-lock.json").write_text(json.dumps(lock))
    assert presentations._toolchain(tmp_path) == installed


@pytest.mark.parametrize("specification", [None, "^4.5.1", "4.6.0"])
def test_root_must_declare_exact_cli_version_even_with_matching_root_lock(
    tmp_path: Path, monkeypatch: MonkeyPatch, specification: str | None
) -> None:
    installed = _fake_toolchain(tmp_path)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    manifest = json.loads((installed / "package.json").read_text())
    manifest["dependencies"] = (
        {} if specification is None else {"@marp-team/marp-cli": specification}
    )
    lock = json.loads((installed / "package-lock.json").read_text())
    lock["packages"][""] = manifest
    (installed / "package.json").write_text(json.dumps(manifest))
    (installed / "package-lock.json").write_text(json.dumps(lock))
    with pytest.raises(PresentationError, match="root manifest must"):
        presentations._toolchain(tmp_path)


def test_toolchain_requires_lock_matching_snapshot_and_installed_versions(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    root = tmp_path / "candidate"
    installed = _fake_toolchain(root)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    assert presentations._toolchain(root) == installed
    copy = tmp_path / "installed"
    shutil.copytree(installed, copy)
    monkeypatch.setenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", str(copy))
    assert presentations._toolchain(root) == copy
    (copy / "render-diagrams.mjs").write_text("// changed renderer\n")
    with pytest.raises(PresentationError, match="render-diagrams.mjs differs"):
        presentations._toolchain(root)
    shutil.copyfile(installed / "render-diagrams.mjs", copy / "render-diagrams.mjs")
    (copy / "package-lock.json").write_text("{}")
    with pytest.raises(PresentationError, match="differs from the checked snapshot"):
        presentations._toolchain(root)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN")
    (installed / "node_modules/@marp-team/marp-cli/package.json").write_text(
        '{"version":"4.6.0"}'
    )
    with pytest.raises(PresentationError, match="version differs"):
        presentations._toolchain(root)


def test_missing_dependency_fails_without_implicit_install(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _deck(tmp_path)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)

    def no_launch(*args: object, **kwargs: object) -> None:
        pytest.fail("Missing dependencies must not trigger npm or network installation")

    monkeypatch.setattr(presentations, "run_captured", no_launch)
    with pytest.raises(PresentationError, match="scripts/presentations.sh setup"):
        presentations.check_presentations(tmp_path)


def test_missing_node_directs_to_installing_node_before_setup(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _deck(tmp_path)
    monkeypatch.setattr(presentations, "_toolchain", lambda root: root)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(PresentationError) as error:
        presentations.build(tmp_path)

    message = str(error.value)
    assert "Node.js 22.17" in message
    assert "npm" in message
    assert "scripts/presentations.sh setup" in message
    assert message.index("Node.js 22.17") < message.index("scripts/presentations.sh")


@pytest.mark.parametrize(
    ("present", "node_step_required"),
    [(set(), True), ({"node"}, True), ({"npm"}, True), ({"node", "npm"}, False)],
)
def test_missing_toolchain_reports_the_install_step_that_can_succeed(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    present: set[str],
    node_step_required: bool,
) -> None:
    _deck(tmp_path)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    monkeypatch.setattr(
        shutil, "which", lambda name: f"/usr/bin/{name}" if name in present else None
    )

    with pytest.raises(PresentationError) as error:
        presentations.build(tmp_path)

    message = str(error.value)
    assert "Locked presentation dependencies are unavailable" in message
    assert ("Node.js 22.17" in message) is node_step_required


def test_build_command_cannot_load_candidate_config_or_local_engine(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(presentations, "require_node", lambda **_: "/usr/bin/node")
    command = presentations._command(
        tmp_path / "tools",
        tmp_path / "slides.md",
        tmp_path / "theme.css",
        tmp_path / "index.html",
    )
    assert command == [
        "/usr/bin/node",
        str(tmp_path / "tools/build.mjs"),
        str(tmp_path / "slides.md"),
        str(tmp_path / "theme.css"),
        str(tmp_path / "index.html"),
    ]
    monkeypatch.setenv("NODE_OPTIONS", "--require bad.js")
    monkeypatch.setenv("MARP_ENGINE", "bad.js")
    monkeypatch.setenv("NPM_CONFIG_IGNORE_SCRIPTS", "false")
    environment = presentations._environment(tmp_path)
    assert (
        not {"NODE_OPTIONS", "MARP_ENGINE", "NPM_CONFIG_IGNORE_SCRIPTS"}
        & environment.keys()
    )


@pytest.fixture
def mermaid_deck(tmp_path: Path, monkeypatch: MonkeyPatch) -> Iterator[Path]:
    """Exercise the real locked compiler and Chromium in a disposable deck."""
    repository = Path(__file__).resolve().parents[2]
    installed = Path(
        os.environ.get(
            "TRADINGDEV_PRESENTATION_TOOLCHAIN", str(repository / "tooling/marp")
        )
    )
    toolchain = tmp_path / "tooling/marp"
    toolchain.mkdir(parents=True)
    for name in presentations.TOOLCHAIN_FILES:
        shutil.copyfile(repository / "tooling/marp" / name, toolchain / name)
    monkeypatch.setenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", str(installed))
    deck = _deck(tmp_path)
    shutil.copyfile(repository / presentations.DECK / "theme.css", deck / "theme.css")
    spaces: list[Path] = []

    def make_space(*, prefix: str) -> str:
        directory = mkdtemp(prefix=prefix, dir=tmp_path)
        spaces.append(Path(directory))
        return directory

    monkeypatch.setattr(presentations, "mkdtemp", make_space)
    yield deck
    assert spaces and all(not path.exists() for path in spaces)


def test_mermaid_build_embeds_repeatable_svg_fonts_and_distinct_diagram_ids(
    tmp_path: Path, mermaid_deck: Path
) -> None:
    diagram = (
        "```mermaid\nsequenceDiagram\n"
        "    accTitle: 使用者與 AI\n"
        "    participant U as 使用者\n    participant A as AI 代理\n"
        "    U->>A: 策略 & 設定\n    A-->>U: 確認單\n```\n"
    )
    (mermaid_deck / "slides.md").write_text(
        f"---\nmarp: true\npaginate: true\n---\n# First\n\n{diagram}"
        f"\n---\n\n# Second\n\n{diagram}\n```python\nx = 1\n```\n"
    )
    presentations.build(tmp_path)
    output = mermaid_deck / "index.html"
    original = output.read_bytes()
    content = original.decode()
    assert 'class="language-mermaid"' not in content
    assert 'class="language-python"' in content
    assert "data:font/woff2;base64," in content
    assert "SIL OPEN FONT LICENSE" in content
    assert presentation_html.check_html(content, root=mermaid_deck) == 2
    # Inspect the actual SVG text and references, not the Markdown or a mock renderer.
    diagrams = content.split('<div class="mermaid-diagram">')[1:]
    assert len(diagrams) == 2
    all_ids: set[str] = set()
    for diagram_html in diagrams:
        svg = ElementTree.fromstring(diagram_html.split("</div>", 1)[0])
        labels = " ".join(svg.itertext())
        assert all(
            label in labels for label in ("使用者", "AI 代理", "策略 & 設定", "確認單")
        )
        ids = [node.attrib["id"] for node in svg.iter() if "id" in node.attrib]
        assert len(ids) == len(set(ids))
        assert not all_ids.intersection(ids)
        all_ids.update(ids)
        for node in svg.iter():
            for value in node.attrib.values():
                assert all(
                    target in ids for target in re.findall(r"url\(#([^)]+)\)", value)
                )
    presentations.build(tmp_path)
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "source",
    [
        "sequenceDiagram\nU->>A missing colon\n",
        "flowchart LR\nA-->B\n",
        'sequenceDiagram\n%%{init: {"securityLevel": "loose"}}%%\nU->>A: Hello\n',
    ],
)
def test_mermaid_failure_preserves_last_successful_html(
    tmp_path: Path, mermaid_deck: Path, source: str
) -> None:
    original = (mermaid_deck / "index.html").read_bytes()
    (mermaid_deck / "slides.md").write_text(f"# Diagram\n\n```mermaid\n{source}```\n")
    with pytest.raises(PresentationError, match="Mermaid diagram 1"):
        presentations.build(tmp_path)
    assert (mermaid_deck / "index.html").read_bytes() == original


def test_mermaid_missing_browser_has_actionable_error(
    tmp_path: Path, mermaid_deck: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("TRADINGDEV_PRESENTATION_BROWSER", str(tmp_path / "missing"))
    (mermaid_deck / "slides.md").write_text(
        "# Diagram\n\n```mermaid\nsequenceDiagram\nU->>A: Hello\n```\n"
    )
    with pytest.raises(PresentationError, match="TRADINGDEV_PRESENTATION_BROWSER"):
        presentations.build(tmp_path)


def test_mermaid_renderer_timeout_preserves_html_and_cleans_workspace(
    tmp_path: Path, mermaid_deck: Path, monkeypatch: MonkeyPatch
) -> None:
    original = (mermaid_deck / "index.html").read_bytes()
    (mermaid_deck / "slides.md").write_text(
        "# Diagram\n\n```mermaid\nsequenceDiagram\nU->>A: Hello\n```\n"
    )
    execute = presentations._execute
    browser_marker = tmp_path / "browser.pid"
    wrapper = tmp_path / "stalled-renderer.mjs"

    def stall_after_browser_launch(
        command: list[str], *, cwd: Path, temporary: Path, timeout: float
    ) -> None:
        if command[1].endswith("build.mjs"):
            renderer = Path(command[1])
            package = renderer.parent / "node_modules/puppeteer-core"
            manifest = json.loads((package / "package.json").read_text())
            module = (package / manifest["exports"]["."]["import"]).as_uri()
            wrapper.write_text(
                "import { writeFileSync } from 'node:fs';\n"
                f"import puppeteer from {json.dumps(module)};\n"
                "const launch = puppeteer.launch.bind(puppeteer);\n"
                "puppeteer.launch = async (...args) => {\n"
                "  const browser = await launch(...args);\n"
                f"  writeFileSync({json.dumps(str(browser_marker))}, "
                "String(browser.process().pid));\n"
                "  await new Promise(() => {});\n"
                "};\n"
                f"await import({json.dumps(renderer.as_uri())});\n"
            )
            command = [command[0], str(wrapper), *command[2:]]
            timeout = 5
        execute(command, cwd=cwd, temporary=temporary, timeout=timeout)

    observed: dict[int, float] = {}
    children = psutil.Process.children

    def record_children(
        process: psutil.Process, recursive: bool = False
    ) -> list[psutil.Process]:
        descendants: list[psutil.Process] = children(process, recursive=recursive)
        if browser_marker.exists():
            for child in descendants:
                with suppress(psutil.NoSuchProcess):
                    observed[child.pid] = child.create_time()
        return descendants

    monkeypatch.setattr(psutil.Process, "children", record_children)
    monkeypatch.setattr(presentations, "_execute", stall_after_browser_launch)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            presentations.build(tmp_path)
        assert browser_marker.exists()
        assert int(browser_marker.read_text()) in observed
        for pid, created_at in observed.items():
            try:
                process = psutil.Process(pid)
                assert (
                    process.create_time() != created_at
                    or process.status() == psutil.STATUS_ZOMBIE
                )
            except psutil.NoSuchProcess:
                pass
        assert (mermaid_deck / "index.html").read_bytes() == original
    finally:
        # A detected cleanup regression must fail before this test-only fallback.
        remaining: list[psutil.Process] = []
        for pid, created_at in observed.items():
            with suppress(psutil.NoSuchProcess, psutil.ZombieProcess):
                process = psutil.Process(pid)
                if process.create_time() == created_at:
                    process.kill()
                    remaining.append(process)
        _, alive = psutil.wait_procs(remaining, timeout=5)
        if alive:
            raise ProcessCleanupError("Test browser cleanup could not be verified")


@pytest.mark.parametrize(
    "mode, observation_delay",
    [
        ("success", 0),
        ("failure", 0),
        ("timeout", 0),
        ("invalid", 0),
        pytest.param("success", 0.35, id="delayed-observation"),
    ],
)
def test_check_cleans_temporary_files_and_child_processes(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    mode: str,
    observation_delay: float,
) -> None:
    deck = _deck(tmp_path)
    original = (deck / "index.html").read_bytes()
    spaces: list[Path] = []
    real_mkdtemp = mkdtemp

    def make_space(*, prefix: str) -> str:
        directory = real_mkdtemp(prefix=prefix, dir=tmp_path)
        spaces.append(Path(directory))
        return directory

    monkeypatch.setattr(presentations, "mkdtemp", make_space)
    monkeypatch.setattr(presentations, "_toolchain", lambda root: root)
    monkeypatch.setattr(presentations, "require_node", lambda **_: "/fixture/node")
    marker = tmp_path / "child.pid"
    observed = tmp_path / "child.observed"
    child_code = "import time; time.sleep(60)"
    original_children = psutil.Process.children

    # Test cleanup of a discovered child; a fixed delay races process inspection.
    def observe_children(
        process: psutil.Process, recursive: bool = False
    ) -> list[psutil.Process]:
        if observation_delay and marker.exists() and not observed.exists():
            time.sleep(observation_delay)
        children: list[psutil.Process] = original_children(process, recursive=recursive)
        if marker.exists() and int(marker.read_text()) in {p.pid for p in children}:
            observed.touch()
        return children

    monkeypatch.setattr(psutil.Process, "children", observe_children)

    def command(toolchain: Path, source: Path, theme: Path, output: Path) -> list[str]:
        code = (
            "import os,subprocess,sys,time\nfrom pathlib import Path\n"
            "child=subprocess.Popen([sys.executable,'-c',sys.argv[4]],start_new_session=True)\n"
            "pending=Path(sys.argv[1]+'.pending')\n"
            "pending.write_text(str(child.pid))\n"
            "pending.replace(sys.argv[1])\n"
            "while not Path(sys.argv[6]).exists(): time.sleep(0.01)\n"
            "if sys.argv[3]=='timeout': time.sleep(60)\n"
            "text='<html></html>' if sys.argv[3]=='invalid' else sys.argv[5]\n"
            "Path(sys.argv[2]).write_text(text)\n"
            "raise SystemExit(9 if sys.argv[3]=='failure' else 0)\n"
        )
        return [
            sys.executable,
            "-c",
            code,
            str(marker),
            str(output),
            mode,
            child_code,
            _html(),
            str(observed),
        ]

    monkeypatch.setattr(presentations, "_command", command)
    try:
        if mode == "success":
            presentations.check_presentations(tmp_path, timeout=5)
        else:
            expected = (
                subprocess.TimeoutExpired if mode == "timeout" else PresentationError
            )
            with pytest.raises(expected):
                presentations.check_presentations(
                    tmp_path, timeout=0.6 if mode == "timeout" else 5
                )
        assert observed.exists()
        assert spaces and all(not path.exists() for path in spaces)
        assert (deck / "index.html").read_bytes() == original
        process = psutil.Process(int(marker.read_text()))
        assert process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
    finally:
        if marker.exists():
            try:
                child = psutil.Process(int(marker.read_text()))
                if child_code in child.cmdline():
                    child.kill()
                    child.wait(timeout=5)
            except psutil.NoSuchProcess, psutil.ZombieProcess:
                pass


def test_unverified_process_cleanup_preserves_workspace(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _deck(tmp_path)
    spaces: list[Path] = []
    real_mkdtemp = mkdtemp

    def make_space(*, prefix: str) -> str:
        directory = real_mkdtemp(prefix=prefix, dir=tmp_path)
        spaces.append(Path(directory))
        return directory

    monkeypatch.setattr(presentations, "mkdtemp", make_space)
    monkeypatch.setattr(presentations, "_toolchain", lambda root: root)
    monkeypatch.setattr(presentations, "require_node", lambda **_: "/fixture/node")

    def fail(*args: object, **kwargs: object) -> None:
        raise ProcessCleanupError("Could not terminate subprocesses")

    monkeypatch.setattr(presentations, "run_captured", fail)
    with pytest.raises(ProcessCleanupError, match="Cleanup unverified; retained"):
        presentations.check_presentations(tmp_path)
    assert len(spaces) == 2 and all(path.exists() for path in spaces)


@pytest.mark.parametrize("failure_kind", ["cleanup", "runtime", "build", "timeout"])
def test_cli_propagates_only_typed_cleanup_failure_and_retains_its_workspace(
    tmp_path: Path, monkeypatch: MonkeyPatch, failure_kind: str
) -> None:
    errors: dict[str, Exception] = {
        "cleanup": ProcessCleanupError("opaque failure"),
        "runtime": RuntimeError("Could not terminate subprocesses"),
        "build": PresentationError("compiler failed"),
        "timeout": subprocess.TimeoutExpired(["compiler"], 1),
    }
    spaces: list[Path] = []

    def make_space(*, prefix: str) -> str:
        return mkdtemp(prefix=prefix, dir=tmp_path)

    def fail(root: Path) -> None:
        with presentations._workspace() as temporary:
            spaces.append(temporary)
            (temporary / "evidence").write_text("diagnostics")
            raise errors[failure_kind]

    monkeypatch.setattr(presentations, "mkdtemp", make_space)
    monkeypatch.setattr(presentations, "check_presentations", fail)
    monkeypatch.setattr(sys, "argv", ["presentations", "check"])
    expected_status = CLEANUP_UNVERIFIED_EXIT_CODE if failure_kind == "cleanup" else 1
    assert presentations.main() == expected_status
    assert len(spaces) == 1
    assert spaces[0].exists() is (failure_kind == "cleanup")


def test_check_reuses_markdown_local_link_validation(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(presentations, "_toolchain", lambda root: root)
    monkeypatch.setattr(presentations, "require_node", lambda **_: "/fixture/node")
    deck = _deck(tmp_path)
    (deck / "slides.md").write_text("# Slides\n\n[broken](missing.md)\n")

    def no_build(*args: object, **kwargs: object) -> None:
        pytest.fail("Broken Markdown links must fail before rebuilding")

    monkeypatch.setattr(presentations, "build", no_build)
    from tooling.documentation import DocumentationError

    with pytest.raises(DocumentationError, match="missing.md"):
        presentations.check_presentations(tmp_path)


def test_watch_rebuilds_static_output_for_source_and_theme_changes(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(presentations, "_toolchain", lambda root: root)
    monkeypatch.setattr(presentations, "require_node", lambda **_: "/fixture/node")
    deck = _deck(tmp_path)
    built: list[tuple[str, str]] = []

    def record(root: Path) -> None:
        built.append(
            ((deck / "slides.md").read_text(), (deck / "theme.css").read_text())
        )

    monkeypatch.setattr(presentations, "build", record)
    pauses = 0

    def change(seconds: float) -> None:
        nonlocal pauses
        pauses += 1
        if pauses == 1:
            (deck / "theme.css").write_text("updated theme")
        elif pauses == 2:
            (deck / "slides.md").write_text("updated source")
        elif pauses == 4:
            raise KeyboardInterrupt

    monkeypatch.setattr(time, "sleep", change)
    presentations.watch(tmp_path)
    assert len(built) == 3
    assert built[1][1] == "updated theme"
    assert built[2][0] == "updated source"


def test_check_reuses_markdown_formatting_validation(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(presentations, "_toolchain", lambda root: root)
    monkeypatch.setattr(presentations, "require_node", lambda **_: "/fixture/node")
    deck = _deck(tmp_path)
    (deck / "slides.md").write_text("# Slides\nMissing heading spacing\n")

    def no_build(*args: object, **kwargs: object) -> None:
        pytest.fail("Markdown formatting errors must fail before rebuilding")

    monkeypatch.setattr(presentations, "build", no_build)
    with pytest.raises(subprocess.CalledProcessError):
        presentations.check_presentations(tmp_path)


@pytest.mark.parametrize("timed_out", [False, True])
def test_watch_recovers_from_build_failure_when_input_is_fixed(
    tmp_path: Path, monkeypatch: MonkeyPatch, timed_out: bool
) -> None:
    monkeypatch.setattr(presentations, "_toolchain", lambda root: root)
    monkeypatch.setattr(presentations, "require_node", lambda **_: "/fixture/node")
    deck = _deck(tmp_path)
    calls = 0

    def compile_deck(root: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            if timed_out:
                raise subprocess.TimeoutExpired(["compiler"], 120)
            raise PresentationError("invalid source")

    monkeypatch.setattr(presentations, "build", compile_deck)
    pauses = 0

    def change(seconds: float) -> None:
        nonlocal pauses
        pauses += 1
        if pauses == 1:
            (deck / "index.html").write_text("unrelated output edit")
            (deck / "README.md").write_text("unrelated entry edit")
        elif pauses == 2:
            (deck / "slides.md").write_text("fixed source")
        elif pauses == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(time, "sleep", change)
    presentations.watch(tmp_path)
    assert calls == 2


def test_source_only_check_builds_without_touching_preview(
    tmp_path: Path,
    mermaid_deck: Path,
) -> None:
    (mermaid_deck / "slides.md").write_text(
        "# Source\n\n<!-- paginate: true -->\n\n"
        "[Next](index.html#2)\n\n---\n\n## Next\n"
    )
    preview = mermaid_deck / "index.html"
    preview.unlink()
    presentations.check_presentations(tmp_path)
    assert not preview.exists()
    preview.write_text("outdated local preview")
    presentations.check_presentations(tmp_path)
    assert preview.read_text() == "outdated local preview"
    slides = mermaid_deck / "slides.md"
    slides.write_text(slides.read_text().replace("index.html#2", "index.html#999"))
    with pytest.raises(PresentationError, match="fragment"):
        presentations.check_presentations(tmp_path)


def test_shared_compiler_disables_candidate_config_and_raw_html(
    tmp_path: Path,
    mermaid_deck: Path,
) -> None:
    (tmp_path / "marp.config.js").write_text("throw new Error('candidate config ran');")
    (mermaid_deck / "slides.md").write_text(
        "---\nmarp: true\npaginate: true\n---\n# Source\n\n"
        '<iframe src="https://invalid.example/"></iframe>\n'
    )
    presentations.build(tmp_path)
    output = (mermaid_deck / "index.html").read_text()
    assert "<iframe" not in output
    assert "&lt;iframe" in output


@pytest.mark.parametrize("inside_source", [True, False])
def test_package_cli_keeps_output_outside_authoring_sources(
    tmp_path: Path, monkeypatch: MonkeyPatch, inside_source: bool
) -> None:
    monkeypatch.setattr(presentations, "_toolchain", lambda root: root)
    monkeypatch.setattr(presentations, "require_node", lambda **_: "/fixture/node")
    deck = _deck(tmp_path)
    (deck / "assets").mkdir()
    (deck / "evidence").mkdir()
    output = (deck / "assets" if inside_source else tmp_path) / "delivery"
    built: list[Path] = []

    def compile_deck(root: Path, *, output: Path, timeout: float) -> None:
        built.append(root)
        output.write_text(_html())

    monkeypatch.setattr(presentations, "build", compile_deck)
    monkeypatch.setattr(
        presentations, "__file__", str(tmp_path / "tooling/presentations.py")
    )
    monkeypatch.setattr(
        sys, "argv", ["presentations", "package", "--output", str(output)]
    )
    assert presentations.main() == (1 if inside_source else 0)
    assert bool(built) is not inside_source
    assert output.exists() is not inside_source
    assert output.with_suffix(".zip").exists() is not inside_source


@pytest.mark.parametrize("failure", ["launch", "close"])
def test_workers_browser_prepares_libraries_and_handles_cleanup_failures(
    tmp_path: Path,
    failure: str,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    toolchain = presentations._toolchain(repository)
    renderer = toolchain / "render-diagrams.mjs"
    package = toolchain / "node_modules/puppeteer-core"
    manifest = json.loads((package / "package.json").read_text())
    module = (package / manifest["exports"]["."]["import"]).as_uri()
    wrapper = tmp_path / "cloud-browser.mjs"
    wrapper.write_text(
        "import assert from 'node:assert/strict';\n"
        "import { existsSync, readFileSync, readdirSync, rmSync } from 'node:fs';\n"
        "import { join, resolve } from 'node:path';\n"
        f"import puppeteer from {json.dumps(module)};\n"
        "Object.defineProperty(process, 'platform', { value: 'linux' });\n"
        "Object.defineProperty(process, 'arch', { value: 'x64' });\n"
        "process.env.WORKERS_CI = '1';\n"
        f"const failure = {json.dumps(failure)};\n"
        "const previous = process.env.TMPDIR;\n"
        "let temporary;\n"
        "puppeteer.launch = async (options) => {\n"
        "  temporary = options.env.TMPDIR;\n"
        "  assert.notEqual(temporary, previous);\n"
        "  assert.equal(options.headless, 'shell');\n"
        "  assert.ok(options.args.includes('--no-sandbox'));\n"
        "  const library = join(temporary, 'al2023/lib');\n"
        "  assert.ok(existsSync(join(library, 'libnss3.so')));\n"
        "  assert.ok(options.env.LD_LIBRARY_PATH.startsWith(library));\n"
        "  const fontRoot = options.env.FONTCONFIG_PATH;\n"
        "  assert.equal(options.env.FONTCONFIG_FILE, join(fontRoot, 'fonts.conf'));\n"
        "  assert.ok(options.env.XDG_CACHE_HOME.startsWith(temporary + '/'));\n"
        "  const config = readFileSync(join(fontRoot, 'fonts.conf'), 'utf8');\n"
        "  const fontDirectories = [...config.matchAll("
        '/<dir(?: prefix="(relative)")?>([^<]+)<\\/dir>/g)]\n'
        "    .map(([, prefix, path]) => prefix ? resolve(fontRoot, path) : path);\n"
        "  assert.ok(fontDirectories.some((path) => existsSync(path) && "
        "readdirSync(path, { recursive: true }).some((file) => "
        "file.endsWith('.ttf'))), 'Fontconfig must find bundled fallback fonts');\n"
        "  assert.equal(readFileSync(options.executablePath).subarray(0, 4)"
        ".toString('hex'), '7f454c46');\n"
        "  if (failure === 'launch') throw new Error('expected launch failure');\n"
        "  return {\n"
        "    newPage: async () => { throw new Error('expected page failure'); },\n"
        "    close: async () => { throw new Error('expected close failure'); },\n"
        "  };\n"
        "};\n"
        f"const {{ renderDiagrams }} = await import({json.dumps(renderer.as_uri())});\n"
        "await assert.rejects(renderDiagrams(['sequenceDiagram']), "
        "(error) => error.cause?.message === `expected ${failure} failure` "
        "&& (failure !== 'close' || "
        "error.message.includes(`retained ${temporary}`)));\n"
        "assert.ok(temporary);\n"
        "assert.equal(existsSync(temporary), failure === 'close');\n"
        "assert.equal(process.env.TMPDIR, previous);\n"
        "// No process was launched; retained test files are safe to remove.\n"
        "rmSync(temporary, { recursive: true, force: true });\n"
    )
    node = shutil.which("node")
    assert node is not None
    presentations._execute(
        [node, str(wrapper)], cwd=tmp_path, temporary=tmp_path, timeout=30
    )
