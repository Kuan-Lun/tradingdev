"""PR checks validate fetched integration content and discard all temporary state."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
from scripts import check_pr, git_gate
from scripts.pr_context import PullRequest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest import CaptureFixture, MonkeyPatch


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

        def directory(*, prefix: str) -> TemporaryDirectory[str]:
            result = TemporaryDirectory(prefix=prefix, dir=root)
            temporary_paths.append(Path(result.name))
            return result

        monkeypatch.setattr(git_gate, "TemporaryDirectory", directory)
        yield Scenario(source, request, temporary_paths)


def _assert_cleaned(scenario: Scenario) -> None:
    assert scenario.temporary_paths
    assert all(not path.exists() for path in scenario.temporary_paths)


def test_check_pr_validates_both_sides_and_removes_candidate_and_snapshot(
    scenario: Scenario, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    before = _source_state(scenario.source)
    calls: list[str] = []

    def run(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 900,
        temporary_root: Path | None = None,
    ) -> None:
        assert timeout == 900
        if command[0] == "bash":
            calls.append("full")
            assert command[1] == "scripts/check-full.sh"
            assert (cwd / "base.txt").read_text() == "new main behavior\n"
            assert (cwd / "head.txt").read_text() == "new PR behavior\n"
            assert temporary_root is not None
            temporary_root.mkdir()
            (temporary_root / "model-output.txt").write_text("temporary output")
        else:
            calls.append("docs")
            assert Path(command[1]).name == "review_docs.py"
            assert command[command.index("--base") + 1] == scenario.request.base_sha
            assert Path(command[command.index("--repo") + 1]).exists()

    monkeypatch.setattr(git_gate, "run", run)
    check_pr.check_pr(7)
    assert calls == ["docs", "full"]
    assert _source_state(scenario.source) == before
    assert len(scenario.temporary_paths) == 2
    _assert_cleaned(scenario)
    output = capsys.readouterr().out
    assert "PR checks passed" in output
    assert scenario.request.base_sha in output and scenario.request.head_sha in output


def test_conflicting_candidate_fails_before_verification_and_cleans_up(
    scenario: Scenario, monkeypatch: MonkeyPatch
) -> None:
    _git(scenario.source, "switch", "--quiet", "main")
    base = _commit(scenario.source, "shared.txt", "main changed\n")
    _git(scenario.source, "switch", "--quiet", "feature/work")
    head = _commit(scenario.source, "shared.txt", "PR changed\n")
    _git(scenario.source, "update-ref", "refs/pull/7/head", head)
    request = replace(scenario.request, base_sha=base, head_sha=head)
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: request)
    monkeypatch.setattr(
        check_pr, "verify_candidate", lambda _: pytest.fail("conflict reached checks")
    )
    before = _source_state(scenario.source)
    with pytest.raises(RuntimeError, match="Merge conflicts"):
        check_pr.check_pr(7)
    assert _source_state(scenario.source) == before
    _assert_cleaned(scenario)


def test_fetched_head_must_match_api_before_running_checks(
    scenario: Scenario, monkeypatch: MonkeyPatch
) -> None:
    request = replace(scenario.request, head_sha="a" * 40)
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: request)
    monkeypatch.setattr(
        check_pr, "verify_candidate", lambda _: pytest.fail("mismatch reached checks")
    )
    with pytest.raises(RuntimeError, match="moved while fetching"):
        check_pr.check_pr(7)
    _assert_cleaned(scenario)


@pytest.mark.parametrize("field", ["head_sha", "state", "base_ref"])
def test_pr_movement_during_checks_invalidates_result(
    scenario: Scenario,
    monkeypatch: MonkeyPatch,
    field: str,
    capsys: CaptureFixture[str],
) -> None:
    changed = {
        "state": replace(scenario.request, state="closed"),
        "base_ref": replace(scenario.request, base_ref="release"),
        "head_sha": replace(scenario.request, head_sha="a" * 40),
    }[field]
    responses = iter((scenario.request, changed))
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: next(responses))
    monkeypatch.setattr(check_pr, "verify_candidate", lambda _: None)
    with pytest.raises(RuntimeError, match="changed during checks"):
        check_pr.check_pr(7)
    assert "PR checks passed" not in capsys.readouterr().out
    _assert_cleaned(scenario)


def test_github_provisional_merge_sha_update_does_not_invalidate_same_content(
    scenario: Scenario, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    responses = iter(
        (scenario.request, replace(scenario.request, merge_commit_sha="a" * 40))
    )
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: next(responses))
    monkeypatch.setattr(check_pr, "verify_candidate", lambda _: None)
    check_pr.check_pr(7)
    assert "PR checks passed" in capsys.readouterr().out
    _assert_cleaned(scenario)


@pytest.mark.parametrize("mode", ["closed", "merged", "wrong_base", "deleted_fork"])
def test_ineligible_pr_never_fetches_or_checks(
    scenario: Scenario, monkeypatch: MonkeyPatch, mode: str
) -> None:
    request = {
        "closed": replace(scenario.request, state="closed"),
        "merged": replace(scenario.request, state="closed", merged=True),
        "wrong_base": replace(scenario.request, base_ref="release"),
        "deleted_fork": replace(scenario.request, head_repository=None),
    }[mode]
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: request)
    monkeypatch.setattr(
        check_pr,
        "verify_candidate",
        lambda _: pytest.fail("ineligible PR reached checks"),
    )
    with pytest.raises(RuntimeError):
        check_pr.check_pr(7)
    assert not scenario.temporary_paths


@pytest.mark.parametrize("failure", [RuntimeError("failed"), KeyboardInterrupt()])
def test_verifier_failure_or_interrupt_cleans_up_and_preserves_source(
    scenario: Scenario, monkeypatch: MonkeyPatch, failure: BaseException
) -> None:
    before = _source_state(scenario.source)

    def fail(candidate: git_gate.MergeCandidate) -> None:
        (candidate.repository / "partial-model-output").write_text("temporary result")
        raise failure

    monkeypatch.setattr(check_pr, "verify_candidate", fail)
    with pytest.raises(type(failure)):
        check_pr.check_pr(7)
    assert _source_state(scenario.source) == before
    _assert_cleaned(scenario)


def test_full_snapshot_failure_cleans_both_temporary_repositories(
    scenario: Scenario, monkeypatch: MonkeyPatch
) -> None:
    before = _source_state(scenario.source)

    def run(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 900,
        temporary_root: Path | None = None,
    ) -> None:
        if command[0] == "bash":
            (cwd / "partial-test-output").write_text("temporary")
            raise RuntimeError("full suite failed")

    monkeypatch.setattr(git_gate, "run", run)
    with pytest.raises(RuntimeError, match="full suite failed"):
        check_pr.check_pr(7)
    assert _source_state(scenario.source) == before
    assert len(scenario.temporary_paths) == 2
    _assert_cleaned(scenario)


@pytest.mark.parametrize("runner", ["pr", "full"])
def test_dirty_runner_is_rejected_before_fetch(
    scenario: Scenario, monkeypatch: MonkeyPatch, runner: str
) -> None:
    (scenario.source / "uncommitted.txt").write_text("user work")
    if runner == "pr":
        with pytest.raises(RuntimeError, match="clean, committed"):
            check_pr.check_pr(7)
    else:
        with pytest.raises(RuntimeError, match="Commit the task stages"):
            git_gate.check_full("main", "HEAD")
    assert not scenario.temporary_paths
    assert (scenario.source / "uncommitted.txt").read_text() == "user work"


@pytest.mark.parametrize("mutation", ["new_file", "new_commit"])
def test_runner_changes_during_pr_check_are_detected(
    scenario: Scenario, monkeypatch: MonkeyPatch, mutation: str
) -> None:
    def mutate(_: git_gate.MergeCandidate) -> None:
        if mutation == "new_file":
            (scenario.source / "user-work.txt").write_text("keep this")
        else:
            _commit(scenario.source, "user-work.txt", "keep this")

    monkeypatch.setattr(check_pr, "verify_candidate", mutate)
    with pytest.raises(RuntimeError, match="runner checkout changed"):
        check_pr.check_pr(7)
    assert (scenario.source / "user-work.txt").read_text() == "keep this"
    _assert_cleaned(scenario)


def test_manual_full_uses_explicit_base_and_preserves_source(
    scenario: Scenario, monkeypatch: MonkeyPatch
) -> None:
    before = _source_state(scenario.source)
    observed: list[git_gate.MergeCandidate] = []
    monkeypatch.setattr(git_gate, "verify_candidate", observed.append)
    git_gate.check_full("main", "feature/work")
    assert len(observed) == 1
    assert observed[0].base == scenario.request.base_sha
    assert observed[0].head == scenario.request.head_sha
    assert _source_state(scenario.source) == before
    _assert_cleaned(scenario)


@pytest.mark.parametrize("ref", ["main", "feature/work"])
def test_manual_full_rejects_ref_movement(
    scenario: Scenario, monkeypatch: MonkeyPatch, ref: str
) -> None:
    def mutate(_: git_gate.MergeCandidate) -> None:
        replacement = (
            scenario.request.head_sha if ref == "main" else scenario.request.base_sha
        )
        _git(scenario.source, "update-ref", f"refs/heads/{ref}", replacement)

    monkeypatch.setattr(git_gate, "verify_candidate", mutate)
    with pytest.raises(RuntimeError, match="compared refs changed"):
        git_gate.check_full("main", "feature/work")
    _assert_cleaned(scenario)


def test_fetch_failure_removes_partial_repository_without_touching_source(
    scenario: Scenario, monkeypatch: MonkeyPatch
) -> None:
    _git(scenario.source, "update-ref", "-d", "refs/pull/7/head")
    before = _source_state(scenario.source)
    monkeypatch.setattr(
        check_pr,
        "verify_candidate",
        lambda _: pytest.fail("failed fetch reached checks"),
    )
    with pytest.raises(RuntimeError, match="Candidate git fetch failed"):
        check_pr.check_pr(7)
    assert _source_state(scenario.source) == before
    _assert_cleaned(scenario)


@pytest.mark.parametrize("lag", ["initial", "final", "both"])
def test_api_base_lag_does_not_override_live_remote_base(
    scenario: Scenario, monkeypatch: MonkeyPatch, lag: str, capsys: CaptureFixture[str]
) -> None:
    first = (
        replace(scenario.request, base_sha="a" * 40)
        if lag in ("initial", "both")
        else scenario.request
    )
    final = (
        replace(scenario.request, base_sha="b" * 40)
        if lag in ("final", "both")
        else scenario.request
    )
    responses = iter((first, final))
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: next(responses))
    checked: list[git_gate.MergeCandidate] = []
    monkeypatch.setattr(check_pr, "verify_candidate", checked.append)
    remote_calls: list[str] = []

    def source(remote: str) -> str:
        remote_calls.append(remote)
        return str(scenario.source)

    monkeypatch.setattr(check_pr, "remote_url", source)
    check_pr.check_pr(7)
    assert checked[0].base == scenario.request.base_sha
    assert remote_calls == ["origin"]
    assert "PR checks passed" in capsys.readouterr().out
    _assert_cleaned(scenario)


@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/pull/7/head"])
@pytest.mark.parametrize("change", ["move", "delete"])
def test_actual_remote_ref_movement_invalidates_unchanged_api_response(
    scenario: Scenario,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    ref: str,
    change: str,
) -> None:
    def mutate(_: git_gate.MergeCandidate) -> None:
        if change == "delete":
            _git(scenario.source, "update-ref", "-d", ref)
        else:
            replacement = (
                scenario.request.head_sha
                if ref == "refs/heads/main"
                else scenario.request.base_sha
            )
            _git(scenario.source, "update-ref", ref, replacement)

    monkeypatch.setattr(check_pr, "verify_candidate", mutate)
    with pytest.raises(RuntimeError, match="changed during checks|disappeared"):
        check_pr.check_pr(7)
    assert "PR checks passed" not in capsys.readouterr().out
    _assert_cleaned(scenario)


@pytest.mark.parametrize(
    "output",
    [
        "",
        "a" * 40 + "\trefs/heads/main\n",
        "not-a-sha\trefs/heads/main\n",
        "a" * 40 + "\trefs/heads/main\n" + "b" * 40 + "\trefs/heads/main\n",
        "a" * 40 + "\trefs/unexpected/main\n",
        "a" * 40 + " refs/heads/main\n",
    ],
)
def test_advertised_ref_response_must_contain_two_valid_unique_refs(
    scenario: Scenario, monkeypatch: MonkeyPatch, output: str
) -> None:
    def captured(
        command: list[str], *, cwd: Path, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        assert command == [
            "git",
            "ls-remote",
            "--refs",
            "--",
            str(scenario.source),
            "refs/heads/main",
            "refs/pull/7/head",
        ]
        assert cwd.resolve() == scenario.source.resolve()
        assert timeout == 120
        assert "GIT_DIR" not in env
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(check_pr, "run_captured", captured)
    with pytest.raises(RuntimeError, match="invalid Git data|disappeared"):
        check_pr._remote_refs(
            str(scenario.source), "refs/heads/main", "refs/pull/7/head"
        )


def test_remote_ref_verification_failure_hides_credential_bearing_command(
    scenario: Scenario, monkeypatch: MonkeyPatch
) -> None:
    def captured(
        command: list[str], *, cwd: Path, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout, stderr="private-token")

    monkeypatch.setattr(check_pr, "run_captured", captured)
    with pytest.raises(RuntimeError, match="Remote ref verification failed") as failure:
        check_pr._remote_refs(
            "https://private-token@github.com/owner/project",
            "refs/heads/main",
            "refs/pull/7/head",
        )
    assert "private-token" not in str(failure.value)
