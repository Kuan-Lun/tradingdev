"""PR and local checks reject changes to eligibility, identities and versions."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from tooling.git import candidates, check_pr, gate
from tooling.process_guard import ProcessCleanupError

if TYPE_CHECKING:
    from pathlib import Path
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
    lightweight_scenario as lightweight_scenario,
)
from tests.tooling.git_check_scenarios import (
    scenario as scenario,
)


@pytest.mark.parametrize("mutation", ["remote_ref", "checkout"])
def test_lightweight_pr_still_rejects_version_changes(
    lightweight_scenario: Scenario,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    mutation: str,
) -> None:
    scenario = lightweight_scenario
    profile = scenario.request.head_ref.removeprefix("feature/")

    def run(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 900,
        temporary_root: Path | None = None,
    ) -> None:
        assert command == ["bash", f"scripts/check-{profile}.sh"]
        if mutation == "remote_ref":
            _git(
                scenario.source,
                "update-ref",
                "refs/heads/main",
                scenario.request.head_sha,
            )
        else:
            (scenario.source / "user-work.txt").write_text("keep this")

    monkeypatch.setattr(candidates, "run", run)
    with pytest.raises(RuntimeError, match="refs changed|checkout changed"):
        check_pr.check_pr(7)
    assert "PR checks passed" not in capsys.readouterr().out
    _assert_cleaned(scenario)


def test_fetched_head_must_match_api_before_running_checks(
    scenario: Scenario, monkeypatch: MonkeyPatch
) -> None:
    request = replace(scenario.request, head_sha="a" * 40)
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: request)
    monkeypatch.setattr(
        check_pr,
        "verify_candidate",
        lambda _, **__: pytest.fail("mismatch reached checks"),
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
    monkeypatch.setattr(check_pr, "verify_candidate", lambda _, **__: None)
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
    monkeypatch.setattr(check_pr, "verify_candidate", lambda _, **__: None)
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
        lambda _, **__: pytest.fail("ineligible PR reached checks"),
    )
    with pytest.raises(RuntimeError):
        check_pr.check_pr(7)
    assert not scenario.temporary_paths


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
            gate.check_full("main", "HEAD")
    assert not scenario.temporary_paths
    assert (scenario.source / "uncommitted.txt").read_text() == "user work"


@pytest.mark.parametrize("mutation", ["new_file", "new_commit"])
def test_runner_changes_during_pr_check_are_detected(
    scenario: Scenario, monkeypatch: MonkeyPatch, mutation: str
) -> None:
    def mutate(
        _: candidates.MergeCandidate,
        *,
        max_evidence_bytes: int | None,
        snapshot_timeout_seconds: int,
        documentation_timeout_seconds: int,
        **review_options: object,
    ) -> None:
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
    observed: list[candidates.MergeCandidate] = []
    monkeypatch.setattr(
        gate, "verify_candidate", lambda candidate, **_: observed.append(candidate)
    )
    gate.check_full("main", "feature/work")
    assert len(observed) == 1
    assert observed[0].base == scenario.request.base_sha
    assert observed[0].head == scenario.request.head_sha
    assert _source_state(scenario.source) == before
    _assert_cleaned(scenario)


@pytest.mark.parametrize("ref", ["main", "feature/work"])
def test_manual_full_rejects_ref_movement(
    scenario: Scenario, monkeypatch: MonkeyPatch, ref: str
) -> None:
    def mutate(
        _: candidates.MergeCandidate,
        *,
        max_evidence_bytes: int | None,
        snapshot_timeout_seconds: int,
        documentation_timeout_seconds: int,
        **review_options: object,
    ) -> None:
        replacement = (
            scenario.request.head_sha if ref == "main" else scenario.request.base_sha
        )
        _git(scenario.source, "update-ref", f"refs/heads/{ref}", replacement)

    monkeypatch.setattr(gate, "verify_candidate", mutate)
    with pytest.raises(RuntimeError, match="compared refs changed"):
        gate.check_full("main", "feature/work")
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
    checked: list[candidates.MergeCandidate] = []
    monkeypatch.setattr(
        check_pr, "verify_candidate", lambda candidate, **_: checked.append(candidate)
    )
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
    def mutate(
        _: candidates.MergeCandidate,
        *,
        max_evidence_bytes: int | None,
        snapshot_timeout_seconds: int,
        documentation_timeout_seconds: int,
        **review_options: object,
    ) -> None:
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


def test_remote_ref_cleanup_failure_keeps_type_without_exposing_credentials(
    monkeypatch: MonkeyPatch,
) -> None:
    def unverified(*args: Any, **kwargs: Any) -> None:
        raise ProcessCleanupError("https://secret@github.com/repository")

    monkeypatch.setattr(check_pr, "run_captured", unverified)
    with pytest.raises(ProcessCleanupError, match="cleanup is unverified") as error:
        check_pr._remote_refs("unused", "refs/heads/main", "refs/pull/7/head")
    assert "secret" not in str(error.value)
