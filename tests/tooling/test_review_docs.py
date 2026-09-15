"""Documentation review must use the candidate tree and fail closed."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
from scripts import review_docs

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest import MonkeyPatch


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def repository() -> Iterator[tuple[Path, str, str]]:
    with TemporaryDirectory(prefix="tradingdev-review-test-") as temporary:
        root = Path(temporary)
        _git(root, "init", "-q")
        (root / "src").mkdir()
        (root / "README.md").write_text("# App\nOld behavior.\n", encoding="utf-8")
        (root / "src/app.py").write_text("VALUE = 'old'\n", encoding="utf-8")
        _git(root, "add", ".")
        base = _git(root, "write-tree")
        (root / "src/app.py").write_text("VALUE = 'staged'\n", encoding="utf-8")
        _git(root, "add", "src/app.py")
        tree = _git(root, "write-tree")
        yield root, base, tree


@pytest.fixture
def fake_codex(monkeypatch: MonkeyPatch) -> Iterator[Path]:
    with TemporaryDirectory(prefix="tradingdev-fake-reviewer-") as temporary:
        root = Path(temporary)
        binary = root / "codex"
        record = root / "record.json"
        binary.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, signal, sys, time\n"
            "prompt = sys.stdin.read()\n"
            "mode = os.environ.get('REVIEW_FAKE_MODE', 'pass')\n"
            "record = {'root': str(pathlib.Path.cwd()), 'pid': os.getpid(), "
            "'prompt': prompt, 'args': sys.argv[1:]}\n"
            "pathlib.Path(os.environ['REVIEW_FAKE_RECORD']).write_text(json.dumps(record))\n"
            "if mode == 'timeout':\n"
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "    while True: time.sleep(0.05)\n"
            "if mode == 'exit':\n"
            "    print('authentication failed', file=sys.stderr)\n"
            "    sys.exit(9)\n"
            "verdict = {'passed': True, 'findings': []}\n"
            "if mode == 'outdated':\n"
            "    verdict = {'passed': False, 'findings': [{'file': 'README.md', "
            "'explanation': 'Update old behavior to staged behavior.'}]}\n"
            "if mode == 'inconsistent': verdict['passed'] = False\n"
            "result = 'broken json' if mode == 'malformed' else json.dumps(verdict)\n"
            "output = pathlib.Path(sys.argv["
            "sys.argv.index('--output-last-message') + 1])\n"
            "if mode != 'missing': output.write_text(result)\n"
            "if mode == 'failed_event': print(json.dumps({'type': 'turn.failed'}))\n"
            "elif mode == 'tool':\n"
            "    print(json.dumps({'type': 'item.completed', "
            "'item': {'type': 'command_execution'}}))\n"
            "else: print(json.dumps({'type': 'turn.completed'}))\n",
            encoding="utf-8",
        )
        binary.chmod(0o700)
        monkeypatch.setenv("TRADINGDEV_CODEX_BIN", str(binary))
        monkeypatch.setenv("REVIEW_FAKE_RECORD", str(record))
        monkeypatch.delenv("REVIEW_FAKE_MODE", raising=False)
        yield record


def test_review_uses_immutable_candidate_and_removes_temporary_files(
    repository: tuple[Path, str, str], fake_codex: Path
) -> None:
    repo, base, tree = repository
    (repo / "src/app.py").write_text("VALUE = 'UNSTAGED_SECRET'\n", encoding="utf-8")
    (repo / "README.md").write_text("UNSTAGED_DOC\n", encoding="utf-8")
    review_docs.review(base, tree, repo=repo)
    record = json.loads(fake_codex.read_text(encoding="utf-8"))
    assert "staged" in record["prompt"]
    assert "Old behavior" in record["prompt"]
    assert "UNSTAGED" not in record["prompt"]
    assert "ignore-user-config" in " ".join(record["args"])
    assert "features.shell_tool=false" in record["args"]
    assert "features.shell_snapshot=false" in record["args"]
    assert 'web_search="disabled"' in record["args"]
    assert "--ephemeral" in record["args"]
    assert "read-only" in record["args"]
    assert not Path(record["root"]).exists()


def test_evidence_includes_deleted_files_and_all_existing_docs(
    repository: tuple[Path, str, str],
) -> None:
    repo, base, _ = repository
    (repo / "docs/strategies").mkdir(parents=True)
    (repo / "docs/strategies/example.md").write_text("# Strategy\n", encoding="utf-8")
    (repo / "src/app.py").unlink()
    _git(repo, "add", ".")
    tree = _git(repo, "write-tree")
    evidence = json.loads(review_docs.build_evidence(repo, base, tree))
    assert "deleted file" in evidence["diff"]
    assert "src/app.py" in evidence["changed_paths"]
    assert "src/app.py" not in evidence["candidate_files"]
    assert "docs/strategies/example.md" in evidence["candidate_files"]


@pytest.mark.parametrize(
    "tooling_path",
    [
        "scripts/hooks/pre-commit",
        ".githooks/pre-commit",
        ".Codex/hooks/finalize-python",
        ".vscode/settings.json",
    ],
)
def test_evidence_includes_tooling_but_excludes_runtime_state(
    repository: tuple[Path, str, str],
    tooling_path: str,
) -> None:
    repo, base, _ = repository
    tooling_file = repo / tooling_path
    tooling_file.parent.mkdir(parents=True)
    tooling_file.write_text("tooling content\n", encoding="utf-8")
    (repo / "workspace").mkdir()
    (repo / "workspace/strategy.py").write_text("RUNTIME_SECRET\n", encoding="utf-8")
    _git(repo, "add", ".")
    tree = _git(repo, "write-tree")
    evidence = review_docs.build_evidence(repo, base, tree)
    assert tooling_path in json.loads(evidence)["candidate_files"]
    assert "RUNTIME_SECRET" not in evidence


def test_evidence_includes_deleted_codex_hook(
    repository: tuple[Path, str, str],
) -> None:
    repo, _, _ = repository
    hook = repo / ".Codex/hooks/finalize-python"
    hook.parent.mkdir(parents=True)
    hook.write_text("legacy hook\n", encoding="utf-8")
    _git(repo, "add", ".")
    base = _git(repo, "write-tree")
    hook.unlink()
    _git(repo, "add", ".")
    tree = _git(repo, "write-tree")
    evidence = json.loads(review_docs.build_evidence(repo, base, tree))
    assert "deleted file" in evidence["diff"]
    assert "legacy hook" in evidence["diff"]
    assert ".Codex/hooks/finalize-python" not in evidence["candidate_files"]


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("outdated", "README.md: Update old behavior"),
        ("malformed", "could not complete"),
        ("inconsistent", "inconsistent documentation verdict"),
        ("missing", "could not complete"),
        ("exit", "exited 9: authentication failed"),
        ("failed_event", "failed review"),
        ("tool", "attempted a tool call"),
    ],
)
def test_failed_review_always_removes_temporary_files(
    repository: tuple[Path, str, str],
    fake_codex: Path,
    monkeypatch: MonkeyPatch,
    mode: str,
    message: str,
) -> None:
    repo, base, tree = repository
    monkeypatch.setenv("REVIEW_FAKE_MODE", mode)
    with pytest.raises(review_docs.ReviewError, match=message):
        review_docs.review(base, tree, repo=repo)
    record = json.loads(fake_codex.read_text(encoding="utf-8"))
    assert not Path(record["root"]).exists()
    with pytest.raises(ProcessLookupError):
        os.kill(record["pid"], 0)


def test_timeout_terminates_process_and_cleans_files(
    repository: tuple[Path, str, str], fake_codex: Path, monkeypatch: MonkeyPatch
) -> None:
    repo, base, tree = repository
    monkeypatch.setenv("REVIEW_FAKE_MODE", "timeout")
    with pytest.raises(review_docs.ReviewError, match="timed out"):
        review_docs.review(base, tree, repo=repo, timeout_seconds=0.5)
    record = json.loads(fake_codex.read_text(encoding="utf-8"))
    assert not Path(record["root"]).exists()
    with pytest.raises(ProcessLookupError):
        os.kill(record["pid"], 0)


def test_excessive_evidence_fails_before_model_call(
    repository: tuple[Path, str, str], fake_codex: Path
) -> None:
    repo, base, tree = repository
    with pytest.raises(review_docs.ReviewError, match="no evidence was truncated"):
        review_docs.review(base, tree, repo=repo, max_evidence_bytes=10)
    assert not fake_codex.exists()


def test_unavailable_codex_is_an_actionable_failure(
    repository: tuple[Path, str, str], monkeypatch: MonkeyPatch
) -> None:
    repo, base, tree = repository
    monkeypatch.setenv("TRADINGDEV_CODEX_BIN", "/missing/documentation-review-codex")
    with pytest.raises(
        review_docs.ReviewError, match="TRADINGDEV_CODEX_BIN is not executable"
    ):
        review_docs.review(base, tree, repo=repo)


def test_script_is_executable_from_another_working_directory(
    repository: tuple[Path, str, str], fake_codex: Path
) -> None:
    repo, base, tree = repository
    result = subprocess.run(
        [
            sys.executable,
            str(Path(review_docs.__file__).resolve()),
            "--base",
            base,
            "--tree",
            tree,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Documentation review passed" in result.stdout
    assert fake_codex.exists()
