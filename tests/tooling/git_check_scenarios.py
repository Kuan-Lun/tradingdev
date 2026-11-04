"""Disposable Git repositories and fixtures shared by candidate-check tests."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import TYPE_CHECKING

import pytest
from tooling import process_guard
from tooling.git import check_pr
from tooling.git.pr_context import PullRequest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest import FixtureRequest, MonkeyPatch


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            f"core.hooksPath={os.devnull}",
            "-c",
            "commit.gpgSign=false",
            *arguments,
        ],
        text=True,
    ).strip()


def _commit(repo: Path, path: str, text: str) -> str:
    (repo / path).write_text(text, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "--quiet", "-m", f"test: change {path}")
    return _git(repo, "rev-parse", "HEAD")


def _source_state(repo: Path) -> tuple[str, str, str, bytes]:
    return (
        _git(repo, "rev-parse", "HEAD"),
        _git(repo, "show-ref"),
        _git(repo, "status", "--porcelain", "--untracked-files=all"),
        (repo / ".git/config").read_bytes(),
    )


@dataclass(frozen=True)
class Scenario:
    source: Path
    request: PullRequest
    temporary_paths: list[Path]


@pytest.fixture
def scenario(monkeypatch: MonkeyPatch) -> Iterator[Scenario]:
    with TemporaryDirectory(prefix="tradingdev-check-pr-test-") as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        _git(source, "init", "--quiet", "--initial-branch=main", "--template=")
        _git(source, "config", "user.name", "PR tests")
        _git(source, "config", "user.email", "checks@example.invalid")
        _commit(source, "shared.txt", "common\n")
        _git(source, "branch", "feature/work")
        base = _commit(source, "base.txt", "new main behavior\n")
        _git(source, "switch", "--quiet", "feature/work")
        head = _commit(source, "head.txt", "new PR behavior\n")
        _git(source, "update-ref", "refs/pull/7/head", head)
        request = PullRequest(
            repository="owner/project",
            number=7,
            base_ref="main",
            base_sha=base,
            head_repository="contributor/project",
            head_ref="feature/work",
            head_sha=head,
            merged=False,
            state="open",
            merge_commit_sha=None,
            url="https://github.com/owner/project/pull/7",
        )
        monkeypatch.chdir(source)
        monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: request)
        monkeypatch.setattr(check_pr, "remote_url", lambda *_: str(source))
        temporary_paths: list[Path] = []

        def snapshot_directory(*, prefix: str) -> str:
            result = mkdtemp(prefix=prefix, dir=root)
            temporary_paths.append(Path(result))
            return result

        monkeypatch.setattr(process_guard, "mkdtemp", snapshot_directory)
        yield Scenario(source, request, temporary_paths)


def _assert_cleaned(scenario: Scenario) -> None:
    assert scenario.temporary_paths
    assert all(not path.exists() for path in scenario.temporary_paths)


@pytest.fixture(params=["docs", "presentations"])
def lightweight_scenario(
    scenario: Scenario, monkeypatch: MonkeyPatch, request: FixtureRequest
) -> Scenario:
    profile = "presentations" if request.param == "presentations" else "docs"
    name = (
        "docs/talks/tradingdev-introduction/slides.md"
        if profile == "presentations"
        else "README.md"
    )
    _git(
        scenario.source,
        "switch",
        "--quiet",
        "-c",
        f"feature/{profile}",
        f"{scenario.request.base_sha}^",
    )
    (scenario.source / name).parent.mkdir(parents=True, exist_ok=True)
    head = _commit(scenario.source, name, "# Documentation change\n")
    _git(scenario.source, "update-ref", "refs/pull/7/head", head)
    pull_request = replace(
        scenario.request, head_sha=head, head_ref=f"feature/{profile}"
    )
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: pull_request)
    return replace(scenario, request=pull_request)
