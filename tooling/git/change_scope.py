"""Classify immutable changes by their required verification boundary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from tooling.presentations import (
    PRESENTATION_PATHS,
    is_presentation_path,
    presentation_change_allowed,
)
from tooling.requirements import is_requirements_path

# These tools do not implement product behavior or shared product-test facilities.
# New/unknown tools remain on the full path until their consumers are reviewed.
_TOOLING_FILES = frozenset(
    {
        "AGENTS.md",
        "CLAUDE.md",
        "tooling/documentation.py",
        "tooling/requirements.py",
        "tooling/quality.py",
        "tooling/presentations.py",
        "tooling/presentation_html.py",
        "tooling/presentation_errors.py",
        "tooling/review_docs.py",
        "tooling/review_evidence.py",
        "tooling/review_options.py",
        "tooling/review_plan.py",
        "tooling/review_scope.py",
        "tooling/git/__init__.py",
        "tooling/git/candidates.py",
        "tooling/git/change_scope.py",
        "tooling/git/check_pr.py",
        "tooling/git/claim_issue.py",
        "tooling/git/cleanup_pr.py",
        "tooling/git/gate.py",
        "tooling/git/pr_context.py",
    }
)
_TOOLING_SCRIPTS = frozenset(
    {
        ".githooks/commit-msg",
        ".githooks/pre-commit",
        ".githooks/pre-push",
        "scripts/check-tooling.sh",
        "scripts/check-docs.sh",
        "scripts/check-fast.sh",
        "scripts/check-pr.sh",
        "scripts/check-presentations.sh",
        "scripts/claim-issue.sh",
        "scripts/cleanup-pr.sh",
        "scripts/detect-primary-branch.sh",
        "scripts/format.sh",
        "scripts/git-flow-commit.sh",
        "scripts/install-git-hooks.sh",
        "scripts/presentations.sh",
    }
)


@dataclass(frozen=True)
class ChangeScope:
    kind: Literal["docs", "presentations", "tooling", "code"]
    presentations_changed: bool = False
    requirements_changed: bool = False

    @property
    def commit_profile(self) -> str:
        return "fast" if self.kind in {"tooling", "code"} else self.kind

    @property
    def delivery_profile(self) -> str:
        return "full" if self.kind == "code" else self.kind

    @property
    def candidate_checks(self) -> tuple[str, ...]:
        if self.kind in {"docs", "presentations"}:
            return (
                self.kind,
                *(("documentation_review",) if self.requirements_changed else ()),
            )
        return (
            *(("presentations",) if self.presentations_changed else ()),
            "documentation_review",
            self.delivery_profile,
        )

    def payload(self) -> dict[str, object]:
        """One executable plan for staged commits, delivery and PR candidates."""
        return {
            "scope": self.kind,
            "commit_profile": self.commit_profile,
            "delivery_profile": self.delivery_profile,
            "candidate_checks": list(self.candidate_checks),
        }


def classify_changes(repository: Path, base: str, tree: str) -> ChangeScope:
    """Inspect both Git sides; unknown changes keep code checks and related checks."""
    return _classify_diff(repository, _raw_diff(repository, base, tree))


def presentation_only_advance(
    repository: Path,
    *,
    base: str,
    updated_base: str,
    tree: str,
    updated_tree: str,
) -> bool:
    """Allow only a disjoint deck delta copied unchanged into the new candidate.

    The caller must separately prove an unchanged PR identity/head and a forward
    base update. Compiler, configuration and general documentation changes are
    ineligible, even when they use a lightweight check profile on their own.
    """
    delta = _raw_diff(repository, base, updated_base)
    if _classify_diff(repository, delta).kind != "presentations":
        return False
    paths = set(delta.split("\0")[1::2])
    if not paths <= PRESENTATION_PATHS:
        return False
    pr_delta = _raw_diff(repository, base, tree)
    # Even disjoint deck/compiler edits can change the original review context.
    if _classify_diff(repository, pr_delta).presentations_changed:
        return False
    pr_paths = set(pr_delta.split("\0")[1::2])
    return not paths.intersection(pr_paths) and delta == _raw_diff(
        repository, tree, updated_tree
    )


def _raw_diff(repository: Path, base: str, tree: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "diff-tree",
            "--no-commit-id",
            "--raw",
            "--no-abbrev",
            "--no-renames",
            "--no-ext-diff",
            "--no-textconv",
            "-r",
            "-z",
            base,
            tree,
            "--",
        ],
        text=True,
        stderr=subprocess.PIPE,
    )


def _classify_diff(repository: Path, output: str) -> ChangeScope:
    if not output:
        return ChangeScope("code")
    fields = output.split("\0")
    if fields.pop() != "" or len(fields) % 2:
        raise RuntimeError("Cannot classify the Git candidate changes.")
    code_changed = presentations_changed = tooling_changed = False
    requirements_changed = False
    for metadata, name in zip(fields[::2], fields[1::2], strict=True):
        parts = metadata.split()
        if len(parts) != 5 or not parts[0].startswith(":"):
            raise RuntimeError("Cannot classify the Git candidate changes.")
        old_mode, new_mode, old_blob, new_blob, status = parts
        requirements_changed |= is_requirements_path(name)
        presentation = is_presentation_path(name)
        presentations_changed |= presentation
        modes = {"000000", "100644"}
        if name in _TOOLING_SCRIPTS:
            modes.add("100755")
        if (
            old_mode[1:] not in modes
            or new_mode not in modes
            or status not in {"A", "M", "D"}
        ):
            code_changed = True
            continue
        if _tooling_path(name):
            tooling_changed = True
        elif presentation:
            before = _read_blob(repository, old_blob) if old_mode != ":000000" else None
            after = _read_blob(repository, new_blob) if new_mode != "000000" else None
            if not presentation_change_allowed(name, before, after):
                code_changed = True
        elif not _documentation_path(name):
            code_changed = True
    if code_changed:
        return ChangeScope("code", presentations_changed, requirements_changed)
    if tooling_changed:
        return ChangeScope("tooling", presentations_changed, requirements_changed)
    return ChangeScope(
        "presentations" if presentations_changed else "docs",
        presentations_changed,
        requirements_changed,
    )


def _read_blob(repository: Path, oid: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repository), "cat-file", "blob", oid],
        stderr=subprocess.PIPE,
    )


def _tooling_path(name: str) -> bool:
    path = PurePosixPath(name)
    return name in _TOOLING_FILES | _TOOLING_SCRIPTS or (
        path.parent == PurePosixPath("tests/tooling")
        and path.name.startswith("test_")
        and path.suffix == ".py"
    )


def _documentation_path(name: str) -> bool:
    path = PurePosixPath(name)
    return name in {"README.md", "ARCHITECTURE.md"} or (
        len(path.parts) > 1 and path.parts[0] == "docs" and path.suffix == ".md"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", required=True, help="Explicit base commit or tree")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--tree", help="Explicit candidate commit or tree")
    target.add_argument("--staged", action="store_true", help="Only the current index")
    args = parser.parse_args(argv)

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(args.repo), *arguments],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()

    def tree(reference: str) -> str:
        return git("rev-parse", "--verify", "--end-of-options", f"{reference}^{{tree}}")

    try:
        base = tree(args.base)
        candidate = git("write-tree") if args.staged else tree(args.tree)
        scope = classify_changes(args.repo, base, candidate)
        current = git("write-tree") if args.staged else tree(args.tree)
        if tree(args.base) != base or current != candidate:
            raise RuntimeError("The requested Git trees changed while planning checks.")
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        message = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            diagnostic = error.stderr
            if isinstance(diagnostic, bytes):
                diagnostic = diagnostic.decode("utf-8", errors="replace")
            if isinstance(diagnostic, str) and diagnostic.strip():
                message = diagnostic.strip()
        print(
            json.dumps({"code": "verification_plan_failed", "error": message}),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "code": "verification_planned",
                "base_tree": base,
                "candidate_tree": candidate,
                "source": "staged" if args.staged else "tree",
                **scope.payload(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
