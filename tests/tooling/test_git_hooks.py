"""Exercise the installed hooks and task scripts against isolated Git repositories."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

_PROJECT = Path(__file__).resolve().parents[2]
_CHECKER = """import json
import os
import subprocess
import sys
from pathlib import Path

mode = sys.argv[1]
root = Path.cwd()
if mode == 'docs':
    base = sys.argv[sys.argv.index('--base') + 1]
    tree = sys.argv[sys.argv.index('--tree') + 1]
    content = subprocess.check_output(
        ['git', 'show', f'{tree}:src/app.py'], text=True
    )
else:
    base = tree = None
    content = (root / 'src/app.py').read_text()
record = {'mode': mode, 'root': str(root), 'content': content,
          'base': base, 'tree': tree}
with Path(os.environ['HOOK_TEST_RECORD']).open('a') as stream:
    stream.write(json.dumps(record) + '\\n')
failure = os.environ.get('HOOK_TEST_FAIL')
if mode == 'full' and failure in {'mutation', 'staged_mutation'}:
    (root / 'src/app.py').write_text("VALUE = 'modified by checker'\\n")
    if os.environ['HOOK_TEST_FAIL'] == 'staged_mutation':
        subprocess.run(['git', 'add', 'src/app.py'], check=True)
if os.environ.get('HOOK_TEST_FAIL') == mode or 'REJECT_FAST' in content:
    print(f'{mode} rejected candidate', file=sys.stderr)
    sys.exit(1)
"""


@dataclass
class GitRepository:
    root: Path
    environment: dict[str, str]
    record: Path

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            args,
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if check:
            assert result.returncode == 0, result.stdout + result.stderr
        return result

    def git(self, *args: str) -> str:
        return self.run("git", *args).stdout.strip()

    def write(self, name: str, content: str) -> None:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def events(self, mode: str) -> list[dict[str, str | None]]:
        if not self.record.exists():
            return []
        events = [json.loads(line) for line in self.record.read_text().splitlines()]
        return [event for event in events if event["mode"] == mode]

    def start_task(self) -> None:
        self.git("switch", "-c", "feature/test-hooks")

    def commit_task(self) -> str:
        self.start_task()
        self.write("src/app.py", "VALUE = 'feature'\n")
        self.run("scripts/git-flow-commit.sh", "feat: update behavior", "src/app.py")
        return self.git("rev-parse", "HEAD")

    def task_worktree(self) -> GitRepository:
        path = self.root.parent / "task-worktree"
        self.git("worktree", "add", "-b", "feature/test-hooks", str(path))
        task = GitRepository(path, self.environment, self.record)
        task.write("src/app.py", "VALUE = 'feature'\n")
        task.run("scripts/git-flow-commit.sh", "feat: update behavior", "src/app.py")
        return task


@contextmanager
def _repository(primary: str = "main") -> Iterator[GitRepository]:
    with TemporaryDirectory(prefix="tradingdev-git-hook-test-") as temporary:
        root = Path(temporary) / "repository"
        root.mkdir()
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        }
        record = Path(temporary) / "checks.jsonl"
        environment.update(
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0",
                "TRADINGDEV_CHECK_PYTHON": sys.executable,
                "HOOK_TEST_RECORD": str(record),
                "HOOK_TEST_FAIL": "",
            }
        )
        repo = GitRepository(root, environment, record)
        repo.git("init", "-q", "-b", primary)
        repo.git("config", "user.name", "Hook Test")
        repo.git("config", "user.email", "hook-test@example.invalid")
        if primary not in {"main", "master"}:
            repo.git("config", "tradingdev.primaryBranch", primary)
        shutil.copytree(_PROJECT / ".githooks", root / ".githooks")
        (root / "scripts").mkdir()
        for name in (
            "__init__.py",
            "git_gate.py",
            "process_guard.py",
            "detect-primary-branch.sh",
            "git-flow-commit.sh",
            "git-flow-merge.sh",
        ):
            shutil.copy2(_PROJECT / "scripts" / name, root / "scripts" / name)
        repo.write("scripts/test_checker.py", _CHECKER)
        for mode in ("fast", "full"):
            repo.write(
                f"scripts/check-{mode}.sh",
                "#!/usr/bin/env bash\n"
                f"exec {shlex.quote(sys.executable)} scripts/test_checker.py {mode}\n",
            )
        repo.write(
            "scripts/review_docs.py",
            "import sys\n" + _CHECKER.replace("mode = sys.argv[1]", "mode = 'docs'"),
        )
        repo.write("src/app.py", "VALUE = 'initial'\n")
        repo.write("README.md", "# Test repository\n")
        repo.write(".gitignore", ".venv/\n__pycache__/\n")
        repo.git("add", ".")
        repo.git("commit", "-qm", "chore: initialize fixture")
        repo.git("config", "core.hooksPath", ".githooks")
        yield repo


def test_primary_branch_rejects_direct_commit_without_losing_staged_changes() -> None:
    with _repository() as repo:
        before = repo.git("rev-parse", "HEAD")
        repo.write("src/app.py", "VALUE = 'direct'\n")
        repo.git("add", "src/app.py")
        result = repo.run("git", "commit", "-m", "feat: direct commit", check=False)
        assert result.returncode != 0
        assert repo.git("rev-parse", "HEAD") == before
        assert repo.git("diff", "--cached", "--name-only") == "src/app.py"
        assert not repo.events("full")
        assert not repo.events("docs")


def test_whitespace_failure_reports_the_file_and_preserves_the_index() -> None:
    with _repository() as repo:
        repo.start_task()
        before = repo.git("rev-parse", "HEAD")
        repo.write("src/app.py", "VALUE = 'feature'  \n")
        repo.git("add", "src/app.py")
        result = repo.run("git", "commit", "-m", "feat: candidate", check=False)
        assert result.returncode != 0
        assert "src/app.py:1: trailing whitespace" in result.stdout + result.stderr
        assert repo.git("rev-parse", "HEAD") == before
        assert repo.git("diff", "--cached", "--name-only") == "src/app.py"


def test_commit_message_hook_rejects_invalid_message_and_accepts_conventional() -> None:
    with _repository() as repo:
        repo.start_task()
        repo.write("src/app.py", "VALUE = 'feature'\n")
        repo.git("add", "src/app.py")
        before = repo.git("rev-parse", "HEAD")
        result = repo.run("git", "commit", "-m", "miscellaneous work", check=False)
        assert result.returncode != 0
        assert repo.git("rev-parse", "HEAD") == before
        repo.run("scripts/git-flow-commit.sh", "feat(mcp)!: revise contract")
        assert repo.git("log", "-1", "--format=%s") == "feat(mcp)!: revise contract"
        assert repo.git("branch", "--show-current") == "feature/test-hooks"
        assert not repo.events("full")
        assert not repo.events("docs")


def test_fast_check_uses_staged_content_and_leaves_unstaged_edits_alone() -> None:
    with _repository() as repo:
        repo.start_task()
        before = repo.git("rev-parse", "HEAD")
        repo.write("src/app.py", "VALUE = 'REJECT_FAST'\n")
        repo.git("add", "src/app.py")
        repo.write("src/app.py", "VALUE = 'working tree passes'\n")
        rejected = repo.run(
            "scripts/git-flow-commit.sh", "feat: rejected snapshot", check=False
        )
        assert rejected.returncode != 0
        assert repo.git("rev-parse", "HEAD") == before
        assert "REJECT_FAST" in str(repo.events("fast")[-1]["content"])

        repo.write("src/app.py", "VALUE = 'staged candidate'\n")
        repo.git("add", "src/app.py")
        repo.write("src/app.py", "VALUE = 'REJECT_FAST unstaged'\n")
        repo.run("scripts/git-flow-commit.sh", "feat: accepted snapshot")
        assert repo.git("show", "HEAD:src/app.py") == "VALUE = 'staged candidate'"
        assert "REJECT_FAST unstaged" in (repo.root / "src/app.py").read_text()
        events = repo.events("fast")
        assert len(events) == 2
        assert events[-1]["content"] == "VALUE = 'staged candidate'\n"
        assert all(not Path(str(event["root"])).exists() for event in events)


def test_stage_commit_script_keeps_unrelated_edits_and_branch() -> None:
    with _repository() as repo:
        primary = repo.git("rev-parse", "main")
        repo.start_task()
        repo.write("src/app.py", "VALUE = 'stage one'\n")
        repo.write("README.md", "# Unfinished documentation\n")
        repo.run("scripts/git-flow-commit.sh", "feat: complete stage one", "src/app.py")
        assert repo.git("show", "--format=", "--name-only", "HEAD") == "src/app.py"
        assert repo.git("status", "--porcelain") == "M README.md"
        assert repo.git("rev-parse", "main") == primary
        assert repo.git("branch", "--show-current") == "feature/test-hooks"
        repo.run("scripts/git-flow-commit.sh", "docs: complete stage two", "README.md")
        assert repo.git("rev-list", "--count", "main..HEAD") == "2"
        assert repo.git("status", "--porcelain") == ""
        assert not repo.events("full")
        assert not repo.events("docs")


@pytest.mark.parametrize("failure", ["full", "docs", "mutation", "staged_mutation"])
def test_merge_gate_failure_aborts_merge_and_preserves_task(failure: str) -> None:
    with _repository() as repo:
        primary = repo.git("rev-parse", "main")
        task = repo.commit_task()
        repo.environment["HOOK_TEST_FAIL"] = failure
        result = repo.run("scripts/git-flow-merge.sh", check=False)
        assert result.returncode != 0
        assert repo.git("rev-parse", "main") == primary
        assert repo.git("rev-parse", "feature/test-hooks") == task
        assert repo.git("branch", "--show-current") == "feature/test-hooks"
        assert repo.git("status", "--porcelain") == ""
        assert not (repo.root / ".git/MERGE_HEAD").exists()
        assert len(repo.events("docs" if failure == "docs" else "full")) == 1
        for event in repo.events("full"):
            assert not Path(str(event["root"])).exists()


@pytest.mark.parametrize("primary", ["main", "master", "trunk"])
def test_successful_merge_checks_candidate_once_and_deletes_task(primary: str) -> None:
    with _repository(primary) as repo:
        previous = repo.git("rev-parse", primary)
        task = repo.commit_task()
        tree = repo.git("rev-parse", "HEAD^{tree}")
        repo.run("scripts/git-flow-merge.sh")
        assert repo.git("branch", "--show-current") == primary
        assert repo.git("show", "-s", "--format=%P", "HEAD") == f"{previous} {task}"
        assert repo.git("rev-parse", "HEAD^{tree}") == tree
        assert repo.git("status", "--porcelain") == ""
        assert repo.git("branch", "--list", "feature/test-hooks") == ""
        repo.run(sys.executable, "scripts/git_gate.py", "full")
        assert len(repo.events("full")) == 1
        assert len(repo.events("docs")) == 1
        assert repo.events("docs")[0]["tree"] == tree
        assert repo.events("full")[0]["content"] == "VALUE = 'feature'\n"
        assert not Path(str(repo.events("full")[0]["root"])).exists()


def test_primary_rebase_is_rejected_without_moving_primary() -> None:
    with _repository() as repo:
        primary = repo.git("rev-parse", "main")
        task = repo.commit_task()
        repo.git("switch", "main")
        result = repo.run("git", "rebase", "feature/test-hooks", check=False)
        assert result.returncode != 0
        assert repo.git("rev-parse", "main") == primary
        assert repo.git("rev-parse", "feature/test-hooks") == task
        assert repo.git("status", "--porcelain") == ""


def test_task_branch_rejects_commit_after_resolving_merge_conflicts() -> None:
    with _repository() as repo:
        task = repo.commit_task()
        repo.git("switch", "-c", "feature/upstream", "main")
        repo.write("src/app.py", "VALUE = 'upstream'\n")
        repo.run("scripts/git-flow-commit.sh", "feat: change upstream", "src/app.py")
        repo.run("scripts/git-flow-merge.sh")
        primary = repo.git("rev-parse", "main")
        repo.git("switch", "feature/test-hooks")

        conflict = repo.run("git", "merge", "main", check=False)
        assert conflict.returncode != 0
        assert repo.git("diff", "--name-only", "--diff-filter=U") == "src/app.py"
        repo.write("src/app.py", "VALUE = 'resolved'\n")
        repo.git("add", "src/app.py")
        assert repo.git("diff", "--name-only", "--diff-filter=U") == ""
        checks = repo.record.read_text()

        result = repo.run(
            "git",
            "commit",
            "-m",
            "Merge branch 'main' into feature/test-hooks",
            check=False,
        )
        assert result.returncode != 0
        assert "Merge commits must be made on main" in result.stdout + result.stderr
        assert repo.git("rev-parse", "HEAD") == task
        assert repo.git("rev-parse", "main") == primary
        assert repo.git("branch", "--show-current") == "feature/test-hooks"
        assert repo.git("rev-parse", "MERGE_HEAD") == primary
        assert repo.git("show", ":src/app.py") == "VALUE = 'resolved'"
        assert repo.record.read_text() == checks


def test_failed_cross_worktree_merge_aborts_primary_and_retains_task() -> None:
    with _repository() as repo:
        previous = repo.git("rev-parse", "HEAD")
        task = repo.task_worktree()
        task_commit = task.git("rev-parse", "HEAD")
        task.environment["HOOK_TEST_FAIL"] = "docs"
        result = task.run("scripts/git-flow-merge.sh", check=False)
        assert result.returncode != 0
        assert len(repo.events("docs")) == 1
        assert repo.git("rev-parse", "main") == previous
        assert repo.git("status", "--porcelain") == ""
        assert not (repo.root / ".git/MERGE_HEAD").exists()
        assert task.root.exists()
        assert task.git("branch", "--show-current") == "feature/test-hooks"
        assert task.git("rev-parse", "HEAD") == task_commit
        assert task.git("status", "--porcelain") == ""


def test_successful_cross_worktree_merge_removes_only_task_worktree() -> None:
    with _repository() as repo:
        previous = repo.git("rev-parse", "HEAD")
        task = repo.task_worktree()
        task_commit = task.git("rev-parse", "HEAD")
        candidate = task.git("rev-parse", "HEAD^{tree}")
        task.run("scripts/git-flow-merge.sh")
        assert repo.root.exists()
        assert not task.root.exists()
        assert repo.git("branch", "--show-current") == "main"
        assert repo.git("show", "-s", "--format=%P", "HEAD") == (
            f"{previous} {task_commit}"
        )
        assert repo.git("rev-parse", "HEAD^{tree}") == candidate
        assert repo.git("status", "--porcelain") == ""
        assert repo.git("branch", "--list", "feature/test-hooks") == ""
        assert len(repo.events("full")) == 1
        assert len(repo.events("docs")) == 1
