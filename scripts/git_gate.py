"""Check immutable Git candidates and retain successful local check receipts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.process_guard import run_checked  # noqa: E402 - standalone entry point


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def environment_key() -> str:
    packages = sorted(
        (distribution.metadata["Name"], distribution.version)
        for distribution in importlib.metadata.distributions()
    )
    encoded = json.dumps(
        [
            sys.version,
            sys.executable,
            sys.platform,
            packages,
            codex_identity(),
            {
                name: os.environ.get(name)
                for name in (
                    "TRADINGDEV_CODEX_BIN",
                    "TRADINGDEV_CODEX_MODEL",
                    "TRADINGDEV_CODEX_TIMEOUT",
                    "CODEX_HOME",
                )
            },
        ],
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def codex_identity() -> tuple[str, str] | None:
    binary = os.environ.get("TRADINGDEV_CODEX_BIN") or shutil.which("codex")
    if (Path(__file__).resolve().parents[1] / "tests/e2e/codex_binary.py").is_file():
        from tests.e2e.codex_binary import resolve_codex_binary

        try:
            binary = resolve_codex_binary()
        except RuntimeError:
            return None
    if not binary or not Path(binary).is_file():
        return None
    with Path(binary).open("rb") as source:
        return str(Path(binary).resolve()), hashlib.file_digest(
            source, "sha256"
        ).hexdigest()


def receipt_path(tree: str, profile: str) -> Path:
    directory = Path(git("rev-parse", "--git-common-dir")).resolve()
    return directory / "tradingdev-checks" / f"{profile}-{tree}.json"


def receipt_matches(tree: str, profile: str) -> bool:
    try:
        receipt = json.loads(receipt_path(tree, profile).read_text())
    except (OSError, ValueError):
        return False
    return bool(
        receipt
        == {
            "schema": 1,
            "tree": tree,
            "profile": profile,
            "environment": environment_key(),
            "result": "passed",
        }
    )


def save_receipt(tree: str, profile: str) -> None:
    path = receipt_path(tree, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "tree": tree,
                "profile": profile,
                "environment": environment_key(),
                "result": "passed",
            },
            indent=2,
        )
        + "\n"
    )


def execution_environment(cwd: Path) -> dict[str, str]:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join([str(cwd / "src"), str(cwd)]),
        "TRADINGDEV_CHECK_PYTHON": sys.executable,
    }
    for name in git("rev-parse", "--local-env-vars").splitlines():
        environment.pop(name, None)
    environment.pop("PYTEST_ADDOPTS", None)
    return environment


def run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 900,
    temporary_root: Path | None = None,
) -> None:
    environment = execution_environment(cwd)
    if temporary_root is not None:
        temporary_root.mkdir()
        environment.update(
            {name: str(temporary_root) for name in ("TMPDIR", "TMP", "TEMP")}
        )
    run_checked(command, cwd=cwd, env=environment, timeout=timeout)


def check_snapshot(tree: str, profile: str) -> None:
    with TemporaryDirectory(prefix="tradingdev-git-check-") as temporary:
        root = Path(temporary)
        archive = root / "candidate.tar"
        with archive.open("wb") as output:
            subprocess.run(["git", "archive", tree], stdout=output, check=True)
        candidate = root / "candidate"
        candidate.mkdir()
        with tarfile.open(archive) as contents:
            contents.extractall(candidate, filter="data")
        environment = {
            **execution_environment(candidate),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
        }

        def snapshot_git(*arguments: str) -> str:
            return subprocess.check_output(
                ["git", *arguments], cwd=candidate, env=environment, text=True
            ).strip()

        snapshot_git("init", "--quiet")
        snapshot_git("add", "--force", "--all")
        if snapshot_git("write-tree") != tree:
            raise RuntimeError("The exported files differ from the Git candidate.")
        run(
            ["bash", f"scripts/check-{profile}.sh"],
            cwd=candidate,
            temporary_root=root / "runtime",
        )
        if (
            snapshot_git("write-tree") != tree
            or snapshot_git("diff", "--name-only")
            or snapshot_git("ls-files", "--others", "--exclude-standard")
        ):
            raise RuntimeError("A check modified the candidate snapshot.")


def assert_index_unchanged(tree: str) -> None:
    if git("write-tree") != tree:
        raise RuntimeError("The staged content changed during checks; stage and retry.")


def check_commit() -> None:
    git("diff", "--cached", "--check")
    tree = git("write-tree")
    check_snapshot(tree, "fast")
    assert_index_unchanged(tree)


def check_full(*, merging: bool) -> None:
    git("diff", "--cached", "--check")
    if git("status", "--porcelain", "--untracked-files=all") and not merging:
        raise RuntimeError("Commit the task stages before running the full gate.")
    if git("diff", "--name-only") or git("ls-files", "--others", "--exclude-standard"):
        raise RuntimeError("Merge checks require no unstaged or untracked files.")
    tree = git("write-tree")
    if receipt_matches(tree, "full"):
        print(f"Reusing successful full checks for {tree[:12]}.", flush=True)
        return
    checked_environment = environment_key()
    primary = subprocess.check_output(
        ["bash", "scripts/detect-primary-branch.sh"], text=True
    ).strip()
    if merging:
        base = git("rev-parse", "HEAD^{tree}")
    elif git("branch", "--show-current") == primary:
        base = git("rev-parse", "HEAD^1^{tree}")
    else:
        base = git("merge-base", primary, "HEAD")
    print("Reviewing code/documentation consistency with Codex...", flush=True)
    run(
        [
            sys.executable,
            "scripts/review_docs.py",
            "--base",
            base,
            "--tree",
            tree,
        ],
        cwd=Path.cwd(),
    )
    check_snapshot(tree, "full")
    assert_index_unchanged(tree)
    if git("diff", "--name-only") or git("ls-files", "--others", "--exclude-standard"):
        raise RuntimeError("The worktree changed during checks; commit and retry.")
    if environment_key() != checked_environment:
        raise RuntimeError("The check environment changed during verification.")
    save_receipt(tree, "full")


def check_push() -> None:
    primary = subprocess.check_output(
        ["bash", "scripts/detect-primary-branch.sh"], text=True
    ).strip()
    for line in sys.stdin:
        _, oid, remote_ref, _ = line.split()
        if remote_ref != f"refs/heads/{primary}" or set(oid) == {"0"}:
            continue
        tree = git("rev-parse", f"{oid}^{{tree}}")
        if not receipt_matches(tree, "full"):
            raise RuntimeError(
                f"No valid full-check receipt for {oid[:12]}; run the full gate first."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["commit", "merge", "full", "push"])
    action = parser.parse_args().action
    os.chdir(git("rev-parse", "--show-toplevel"))
    if action == "commit":
        check_commit()
    elif action == "push":
        check_push()
    else:
        check_full(merging=action == "merge")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        print(f"Git checks failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
