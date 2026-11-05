"""Only independent presentation advances may refresh a checked PR candidate."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
from tooling import presentations
from tooling.git import candidates, check_pr
from tooling.process_guard import ProcessCleanupError

if TYPE_CHECKING:
    from typing import Any

    from pytest import CaptureFixture, MonkeyPatch
    from tests.tooling.git_check_scenarios import Scenario

from tests.tooling.git_check_scenarios import (
    _assert_cleaned,
    _commit,
    _git,
    _source_state,
)
from tests.tooling.git_check_scenarios import (
    scenario as scenario,
)


def _advance_main(
    source: Path,
    changes: dict[str, str],
    *,
    mode: str = "100644",
    parent: str | None = None,
) -> str:
    """Model another user's merge without changing the runner checkout/index."""
    previous = _git(source, "rev-parse", "refs/heads/main")
    with TemporaryDirectory(prefix="tradingdev-main-update-") as temporary:
        environment = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index")}

        def git(*args: str, input_text: str | None = None) -> str:
            return subprocess.run(
                ["git", "-C", str(source), "-c", "commit.gpgSign=false", *args],
                input=input_text,
                text=True,
                capture_output=True,
                check=True,
                env=environment,
            ).stdout.strip()

        git("read-tree", previous)
        for name, content in changes.items():
            blob = git("hash-object", "-w", "--stdin", input_text=content)
            git("update-index", "--add", "--cacheinfo", mode, blob, name)
        tree = git("write-tree")
        commit = git(
            "commit-tree", tree, "-p", parent or previous, "-m", "test: advance main"
        )
        git("update-ref", "refs/heads/main", commit, previous)
        return commit


@pytest.mark.parametrize("name", sorted(presentations.PRESENTATION_PATHS))
def test_disjoint_main_deck_update_runs_only_fresh_presentation_snapshot(
    scenario: Scenario,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    name: str,
) -> None:
    before = _source_state(scenario.source)
    index = _git(scenario.source, "write-tree")
    verified: list[candidates.MergeCandidate] = []
    snapshots: list[str] = []

    def verify(candidate: candidates.MergeCandidate, **kwargs: Any) -> None:
        verified.append(candidate)
        _advance_main(scenario.source, {name: "new presentation"})

    def snapshot(tree: str, profile: str, *, repo: Path, timeout_seconds: int) -> None:
        assert profile == "presentations"
        assert timeout_seconds == 45
        assert _git(repo, "show", f"{tree}:{name}") == "new presentation"
        assert _git(repo, "show", f"{tree}:head.txt") == "new PR behavior"
        snapshots.append(tree)

    monkeypatch.setattr(check_pr, "verify_candidate", verify)
    monkeypatch.setattr(check_pr, "check_snapshot", snapshot)
    check_pr.check_pr(7, snapshot_timeout_seconds=45)
    output = capsys.readouterr().out
    event = json.loads(
        next(line for line in output.splitlines() if line.startswith("{"))
    )
    assert len(verified) == len(snapshots) == 1
    assert event == {
        "code": "pr_base_refreshed",
        "previous_base": scenario.request.base_sha,
        "reused_tree": verified[0].tree,
        "base": _git(scenario.source, "rev-parse", "main"),
        "head": scenario.request.head_sha,
        "tree": snapshots[0],
        "checks": ["presentations"],
    }
    assert (
        f"Base: {event['base']}\nHead: {event['head']}\nTree: {event['tree']}" in output
    )
    after = _source_state(scenario.source)
    assert (after[0], after[2:]) == (before[0], before[2:])
    assert _git(scenario.source, "write-tree") == index
    _assert_cleaned(scenario)


@pytest.mark.parametrize(
    "case",
    [
        "product",
        "docs",
        "config",
        "marker",
        "mode",
        "symlink",
        "overlap",
        "rewrite",
        "pr_compiler",
        "pr_deck",
    ],
)
def test_main_updates_outside_disjoint_regular_deck_files_require_recheck(
    scenario: Scenario, monkeypatch: MonkeyPatch, case: str
) -> None:
    name = "docs/talks/tradingdev-introduction/slides.md"
    if case in {"overlap", "pr_compiler", "pr_deck"}:
        pr_name = {
            "pr_compiler": "tooling/presentations.py",
            "pr_deck": "docs/talks/tradingdev-introduction/theme.css",
        }.get(case, name)
        (scenario.source / pr_name).parent.mkdir(parents=True)
        head = _commit(scenario.source, pr_name, "new presentation")
        _git(scenario.source, "update-ref", "refs/pull/7/head", head)
        monkeypatch.setattr(
            check_pr,
            "get_pull_request",
            lambda *_: replace(scenario.request, head_sha=head),
        )

    def verify(candidate: candidates.MergeCandidate, **kwargs: Any) -> None:
        changed = {
            "product": "product.py",
            "docs": "README.md",
            "config": ".vscode/tasks.json",
            "marker": "docs/.nojekyll",
        }.get(case, name)
        _advance_main(
            scenario.source,
            {changed: "" if case == "marker" else "new presentation"},
            mode="100755"
            if case == "mode"
            else "120000"
            if case == "symlink"
            else "100644",
            parent=_git(scenario.source, "rev-parse", "main^")
            if case == "rewrite"
            else None,
        )

    monkeypatch.setattr(check_pr, "verify_candidate", verify)
    monkeypatch.setattr(
        check_pr, "check_snapshot", lambda *_, **__: pytest.fail("Unsafe refresh")
    )
    with pytest.raises(RuntimeError, match="changed during checks"):
        check_pr.check_pr(7)
    _assert_cleaned(scenario)


@pytest.mark.parametrize(
    "mutation",
    ["fetch_base", "fetch_head", "candidate_tree", "base", "head", "state", "checkout"],
)
def test_presentation_refresh_rejects_further_identity_or_content_changes(
    scenario: Scenario, monkeypatch: MonkeyPatch, mutation: str
) -> None:
    name = "docs/talks/tradingdev-introduction/slides.md"
    monkeypatch.setattr(
        check_pr,
        "verify_candidate",
        lambda *_, **__: _advance_main(scenario.source, {name: "first update"}),
    )
    real_fetch = candidates.fetch_merge_candidate
    snapshots: list[str] = []

    def fetch(*args: Any, **kwargs: Any) -> candidates.MergeCandidate:
        if mutation == "fetch_base":
            _advance_main(scenario.source, {name: "second update"})
        if mutation == "fetch_head":
            _git(
                scenario.source,
                "update-ref",
                "refs/pull/7/head",
                scenario.request.base_sha,
            )
        candidate = real_fetch(*args, **kwargs)
        return (
            replace(candidate, tree=candidate.base)
            if mutation == "candidate_tree"
            else candidate
        )

    def snapshot(tree: str, *args: Any, **kwargs: Any) -> None:
        snapshots.append(tree)
        if mutation == "base":
            _advance_main(scenario.source, {name: "second update"})
        elif mutation == "head":
            _git(
                scenario.source,
                "update-ref",
                "refs/pull/7/head",
                scenario.request.base_sha,
            )
        elif mutation == "state":
            monkeypatch.setattr(
                check_pr,
                "get_pull_request",
                lambda *_: replace(scenario.request, state="closed"),
            )
        elif mutation == "checkout":
            (scenario.source / "unexpected").touch()

    monkeypatch.setattr(check_pr, "fetch_merge_candidate", fetch)
    monkeypatch.setattr(check_pr, "check_snapshot", snapshot)
    with pytest.raises(RuntimeError, match="changed during"):
        check_pr.check_pr(7)
    assert len(snapshots) == (
        0 if mutation.startswith("fetch_") or mutation == "candidate_tree" else 1
    )
    _assert_cleaned(scenario)


@pytest.mark.parametrize("failure", ["rejected", "timeout", "interrupt", "cleanup"])
def test_presentation_refresh_failure_preserves_cleanup_policy(
    scenario: Scenario, monkeypatch: MonkeyPatch, failure: str
) -> None:
    name = "docs/talks/tradingdev-introduction/slides.md"
    monkeypatch.setattr(
        check_pr,
        "verify_candidate",
        lambda *_, **__: _advance_main(scenario.source, {name: "first update"}),
    )
    error = {
        "rejected": subprocess.CalledProcessError(1, "presentation check"),
        "timeout": subprocess.TimeoutExpired("presentation check", 1),
        "interrupt": KeyboardInterrupt(),
        "cleanup": ProcessCleanupError("injected uncertainty; no check child started"),
    }[failure]

    def snapshot(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(check_pr, "check_snapshot", snapshot)
    try:
        with pytest.raises(type(error)):
            check_pr.check_pr(7)
        assert all(
            root.exists() == (failure == "cleanup") for root in scenario.temporary_paths
        )
    finally:
        # All real Git commands returned; the injected snapshot starts no child.
        for root in scenario.temporary_paths:
            if root.exists():
                shutil.rmtree(root)
    _assert_cleaned(scenario)
