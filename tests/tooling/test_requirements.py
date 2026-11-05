"""SRS maintenance catches broken records without claiming behavioral coverage."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
from tooling import documentation
from tooling.requirements import RequirementsError, check_requirements

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


RECORD = """## FR-EXEC-001 固定執行

- **需求**：保存有效設定。
- **驗收**：預設值更新後，查詢仍回傳原設定。
- **契約**：[執行](../contract.md#設定)。
- **驗證**：[服務測試](../../tests/test_execution.py)，驗證歷史設定。
"""


def _srs(root: Path, record: str = RECORD) -> Path:
    (root / "docs/requirements").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "docs/requirements.md").write_text(
        "# SRS\n\n[執行](requirements/execution.md)\n", encoding="utf-8"
    )
    (root / "docs/contract.md").write_text("# 設定\n", encoding="utf-8")
    # Evidence must only be read as a reference, never imported or executed.
    (root / "tests/test_execution.py").write_text(
        "raise RuntimeError('never execute')\n"
    )
    chapter = root / "docs/requirements/execution.md"
    chapter.write_text("# 執行需求\n\n" + record, encoding="utf-8")
    return chapter


def test_documentation_entry_checks_srs_before_formatter(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _srs(tmp_path, RECORD.replace("- **驗收**：預設值更新後，查詢仍回傳原設定。\n", ""))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("invalid SRS"))
    with pytest.raises(RequirementsError, match="missing fields 驗收"):
        documentation.check_documents(tmp_path)


def test_multiple_chapters_and_explicit_manual_verification(tmp_path: Path) -> None:
    _srs(tmp_path)
    (tmp_path / "docs/requirements/quality.md").write_text(
        "# 品質\n\n"
        + RECORD.replace("FR-EXEC-001", "NFR-UI-001").replace(
            "[服務測試](../../tests/test_execution.py)，驗證歷史設定。",
            "人工驗收：在離線瀏覽器開啟報告，確認可閱讀；瀏覽器差異尚待實測。",
        ),
        encoding="utf-8",
    )
    index = tmp_path / "docs/requirements.md"
    index.write_text(index.read_text() + "\n[品質](requirements/quality.md)\n")
    assert check_requirements(tmp_path) == 2


@pytest.mark.parametrize(
    "before, after, error",
    [
        ("FR-EXEC-001", "FR-EXEC-1", "malformed requirement"),
        ("## FR-EXEC-001", "### FR-EXEC-001", "malformed requirement"),
        ("保存有效設定。", "", "empty or duplicate 需求"),
        ("- **需求**：保存有效設定。", "", "missing fields 需求"),
        ("- **驗收**：", "- **驗収**：", "missing fields 驗收"),
        ("[執行](../contract.md#設定)", "詳見程式碼", "missing 契約 evidence"),
        (
            "[服務測試](../../tests/test_execution.py)",
            "[範例](https://example.invalid/test.py)",
            "missing 驗證 evidence",
        ),
        ("tests/test_execution.py", "tests/missing.py", "invalid local evidence"),
    ],
)
def test_invalid_requirement_records(
    tmp_path: Path, before: str, after: str, error: str
) -> None:
    _srs(tmp_path, RECORD.replace(before, after))
    with pytest.raises(RequirementsError, match=error):
        check_requirements(tmp_path)


def test_duplicate_identifiers_across_chapters(tmp_path: Path) -> None:
    _srs(tmp_path)
    (tmp_path / "docs/requirements/quality.md").write_text("# Other\n\n" + RECORD)
    index = tmp_path / "docs/requirements.md"
    index.write_text(index.read_text() + "\n[Other](requirements/quality.md)\n")
    with pytest.raises(RequirementsError, match="duplicate FR-EXEC-001"):
        check_requirements(tmp_path)


def test_duplicate_fields_and_code_examples_do_not_supply_evidence(
    tmp_path: Path,
) -> None:
    chapter = _srs(tmp_path, RECORD + "\n- **需求**：重複。\n")
    with pytest.raises(RequirementsError, match="duplicate 需求"):
        check_requirements(tmp_path)
    chapter.write_text("# Examples\n\n```markdown\n" + RECORD + "```\n")
    with pytest.raises(RequirementsError, match="no requirement records"):
        check_requirements(tmp_path)


@pytest.mark.parametrize("removed", ["entry", "navigation", "chapter", "test"])
def test_missing_navigation_and_deleted_evidence(tmp_path: Path, removed: str) -> None:
    chapter = _srs(tmp_path)
    if removed == "entry":
        (tmp_path / "docs/requirements.md").unlink()
    elif removed == "navigation":
        (tmp_path / "docs/requirements.md").write_text("# SRS\n")
    elif removed == "chapter":
        chapter.unlink()
    else:
        (tmp_path / "tests/test_execution.py").unlink()
    with pytest.raises(RequirementsError):
        check_requirements(tmp_path)


def test_repositories_before_srs_introduction_remain_valid(tmp_path: Path) -> None:
    assert check_requirements(tmp_path) == 0


def test_evidence_cannot_escape_repository(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _srs(root)
    outside = tmp_path / "outside.py"
    outside.write_text("raise RuntimeError('do not read')\n")
    target = root / "tests/test_execution.py"
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(RequirementsError, match="invalid local evidence"):
        check_requirements(root)


def test_new_chapter_requires_a_review_responsibility(tmp_path: Path) -> None:
    _srs(tmp_path)
    (tmp_path / "docs/requirements/new.md").write_text(RECORD)
    index = tmp_path / "docs/requirements.md"
    index.write_text(index.read_text() + "\n[New](requirements/new.md)\n")
    with pytest.raises(
        RequirementsError, match="unregistered SRS review responsibility"
    ):
        check_requirements(tmp_path)
