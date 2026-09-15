"""Check a specified GitHub PR locally; publishing and merging remain manual."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.git_gate import git, merge_candidate, verify_candidate  # noqa: E402
from scripts.pr_context import get_pull_request, remote_url  # noqa: E402


def check_pr(number: int, remote: str = "origin") -> None:
    request = get_pull_request(number, remote)
    primary = subprocess.check_output(
        ["bash", str(Path(__file__).with_name("detect-primary-branch.sh"))], text=True
    ).strip()
    if request.merged or request.state != "open":
        raise RuntimeError("Choose an open, unmerged PR.")
    if request.base_ref != primary or request.head_repository is None:
        raise RuntimeError(
            "The PR must target the primary branch and have a source repository."
        )
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Use a clean, committed checkout for the PR check runner.")
    starting_head = git("rev-parse", "HEAD")
    with merge_candidate(
        remote_url(remote), f"refs/heads/{request.base_ref}", f"refs/pull/{number}/head"
    ) as candidate:
        if (candidate.base, candidate.head) != (request.base_sha, request.head_sha):
            raise RuntimeError("The PR/base moved while fetching; run the check again.")
        print(
            f"Checking {request.url}\nBase: {candidate.base}\n"
            f"Head: {candidate.head}\nTree: {candidate.tree}",
            flush=True,
        )
        verify_candidate(candidate)
        current = get_pull_request(number, remote)
        # GitHub computes this provisional merge SHA asynchronously for open PRs.
        # Only the source/base identity and lifecycle need to remain unchanged.
        if replace(current, merge_commit_sha=request.merge_commit_sha) != request:
            raise RuntimeError(
                "The PR/base changed during checks; run the check again."
            )
        if git("rev-parse", "HEAD") != starting_head or git(
            "status", "--porcelain", "--untracked-files=all"
        ):
            raise RuntimeError("The check runner checkout changed during checks.")
        print(
            f"PR checks passed: {request.url}\n"
            f"Base: {candidate.base}\nHead: {candidate.head}\nTree: {candidate.tree}\n"
            "Codex documentation review and full pytest passed.\n"
            "Before merging on GitHub, compare the current base/head with these SHAs. "
            "If either changed, rerun. This local result does not lock GitHub merging.",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int)
    parser.add_argument("--remote", default="origin")
    args = parser.parse_args()
    os.chdir(git("rev-parse", "--show-toplevel"))
    check_pr(args.number, args.remote)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"PR checks failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
