"""Dashboard reruns never offer unverified or stale report downloads."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tradingdev.adapters.dashboard import app
from tradingdev.adapters.reporting.storage import publish_report
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.report_service import ReportService


@dataclass
class FakeStreamlit:
    session_state: dict[str, Any] = field(default_factory=dict)
    pressed: bool = False
    downloads: list[bytes] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def button(self, label: str) -> bool:
        return self.pressed

    def download_button(
        self, label: str, data: bytes, *, file_name: str, mime: str
    ) -> None:
        assert file_name == "tradingdev-run.html" and mime == "text/html"
        self.downloads.append(data)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def rerun(self, service: ReportService, *, pressed: bool = False) -> None:
        self.pressed = pressed
        self.downloads.clear()
        self.errors.clear()
        app._render_report_download("run", service)


@pytest.fixture
def report_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[FakeStreamlit, ReportService, dict[str, Any]]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    store.create_run(
        run_id="run",
        job_id="job",
        strategy_id="fixture",
        artifact_dir=workspace.runs / "run",
        metrics={},
    )
    report = publish_report(
        workspace,
        store,
        "a" * 64,
        b"<html>saved report</html>",
        b"{}",
        run_ids=["run"],
        scope_count=1,
    )
    service = ReportService(workspace=workspace, store=store)
    # Generation is a separate tested workflow; these tests exercise the real
    # download service and storage on repeated dashboard renders.
    monkeypatch.setattr(service, "generate_report", lambda *_args: report)
    ui = FakeStreamlit()
    monkeypatch.setattr(app, "st", ui)
    return ui, service, report


def test_report_download_survives_normal_reruns_with_verified_bytes(
    report_context: tuple[FakeStreamlit, ReportService, dict[str, Any]],
) -> None:
    ui, service, report = report_context
    ui.rerun(service, pressed=True)
    ui.rerun(service)

    assert not ui.errors and len(ui.downloads) == 1
    assert hashlib.sha256(ui.downloads[0]).hexdigest() == report["sha256"]
    assert ui.session_state[app._REPORT_KEY] == {
        "run_id": "run",
        "report_id": report["report_id"],
        "sha256": report["sha256"],
    }


@pytest.mark.parametrize("change", ["modified", "deleted"])
def test_report_changed_between_reruns_removes_download_and_shows_error(
    report_context: tuple[FakeStreamlit, ReportService, dict[str, Any]], change: str
) -> None:
    ui, service, report = report_context
    ui.rerun(service, pressed=True)
    path = Path(report["path"])
    if change == "modified":
        path.write_bytes(b"modified")
    else:
        path.unlink()

    ui.rerun(service)

    assert ui.errors and not ui.downloads
    assert app._REPORT_KEY not in ui.session_state
    ui.rerun(service)
    assert not ui.downloads


@pytest.mark.parametrize("change", ["modified", "deleted"])
def test_failed_regeneration_does_not_fall_back_to_previous_report(
    report_context: tuple[FakeStreamlit, ReportService, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    ui, service, report = report_context
    ui.rerun(service, pressed=True)
    path = Path(report["path"])
    if change == "modified":
        path.write_bytes(b"modified")
    else:
        path.unlink()
    monkeypatch.setattr(
        service,
        "generate_report",
        lambda *_args: {
            "success": False,
            "code": "report_artifact_invalid",
            "error": "Registered report is missing or corrupt",
        },
    )
    monkeypatch.setattr(
        service,
        "get_report_download",
        lambda *_args, **_kwargs: pytest.fail("must not read a stale download"),
    )

    ui.rerun(service, pressed=True)

    assert ui.errors == ["Registered report is missing or corrupt"]
    assert not ui.downloads and app._REPORT_KEY not in ui.session_state
    ui.rerun(service)
    assert not ui.downloads


def test_switching_run_does_not_offer_another_runs_report(
    report_context: tuple[FakeStreamlit, ReportService, dict[str, Any]],
) -> None:
    ui, service, _ = report_context
    ui.rerun(service, pressed=True)
    ui.pressed = False
    ui.downloads.clear()

    app._render_report_download("different-run", service)

    assert not ui.downloads and not ui.errors
