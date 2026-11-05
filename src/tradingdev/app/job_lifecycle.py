"""Reconcile and cancel workers without overwriting a completed job."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.execution.process_runner import (
    ProcessIdentity,
    WorkerHandle,
    request_worker_stop,
)
from tradingdev.app.comparison_service import ComparisonService
from tradingdev.app.job_store import ACTIVE_JOB_STATUSES, TERMINAL_JOB_STATUSES

if TYPE_CHECKING:
    from tradingdev.app.job_store import JobStore


@dataclass(frozen=True)
class JobObservation:
    """The latest persisted state and any failure to verify worker cleanup."""

    job: dict[str, Any] | None
    cleanup_error: dict[str, Any] | None = None


class JobLifecycle:
    """Own worker control and lifecycle recovery under the store's protections."""

    _ACTIVE_STATUSES = ACTIVE_JOB_STATUSES
    _TERMINAL_STATUSES = TERMINAL_JOB_STATUSES

    def __init__(self, job_store: JobStore) -> None:
        self._job_store = job_store

    def refresh(self, job_id: str) -> JobObservation:
        """Reconcile stopped workers, then read back the persisted lifecycle."""
        job = self._job_store.get_job(job_id)
        if job is None:
            return JobObservation(job=None)

        status = str(job["status"])
        cleanup_error: dict[str, Any] | None = None
        if (
            status in self._ACTIVE_STATUSES
            and (status != "queued" or job.get("pid") is not None)
            and not self._is_process_alive(job)
        ):
            latest = self._job_store.get_job(job_id)
            if latest is not None and latest["status"] in self._TERMINAL_STATUSES:
                return self.refresh(job_id)
            if job.get("job_type") == "comparison":
                handle = WorkerHandle.from_job(job)
                try:
                    if handle is None:
                        raise RuntimeError("Worker control identity is unavailable")
                    request_worker_stop(self._job_store.workspace.root, handle)
                except (OSError, ValueError, RuntimeError) as exc:
                    cleanup_error = {
                        "success": False,
                        "code": "worker_cleanup_failed",
                        "error": f"Worker cleanup could not be confirmed: {exc}",
                    }
                if cleanup_error is None:
                    latest = self._job_store.get_job(job_id)
                    if (
                        latest is not None
                        and latest["status"] in self._TERMINAL_STATUSES
                    ):
                        # The worker may finish between the first read and its
                        # supervisor's exit. Preserve that completed lifecycle.
                        return self.refresh(job_id)
                    try:
                        ComparisonService(job_store=self._job_store).reconcile(
                            job_id, reason="Worker process terminated unexpectedly."
                        )
                    except (OSError, TypeError, ValueError) as exc:
                        cleanup_error = {
                            "success": False,
                            "code": "execution_manifest_invalid",
                            "error": str(exc),
                        }
            if (
                cleanup_error is None
                or cleanup_error["code"] == "execution_manifest_invalid"
            ):
                status = "failed"
                failure = {
                    "status": status,
                    "error": "Worker process terminated unexpectedly.",
                    "ended_at": datetime.now(UTC).isoformat(),
                }
                latest = self._job_store.update_job(job_id, **failure)
                if latest is not None:
                    job = latest
                    status = str(job["status"])

        # A cleanup attempt can fail after the worker commits its final state.
        # Preserve that state while still reporting the cleanup diagnostic.
        job = self._job_store.get_job(job_id) or job
        return JobObservation(job=job, cleanup_error=cleanup_error)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued or running background job."""
        job = self._job_store.get_job(job_id)
        if job is not None and job.get("job_type") == "comparison":
            with self._job_store.submission_lock(job_id):
                return self._cancel_job(job_id)
        return self._cancel_job(job_id)

    def _cancel_job(self, job_id: str) -> dict[str, Any]:
        job = self._job_store.get_job(job_id)
        if job is None:
            return {
                "success": False,
                "error": f"No job with ID: {job_id}",
                "code": "job_not_found",
            }

        status = str(job["status"])
        if status in self._TERMINAL_STATUSES:
            return {
                "success": False,
                "error": f"Job is already terminal: {status}",
                "code": "invalid_job_state",
                "status": status,
            }
        if status not in self._ACTIVE_STATUSES:
            return {
                "success": False,
                "error": f"Job status is not cancellable: {status}",
                "code": "invalid_job_state",
                "status": status,
            }

        process_terminated = False
        handle = WorkerHandle.from_job(job)
        if handle is None:
            if status != "queued" or job.get("pid") is not None:
                return {
                    "success": False,
                    "error": (
                        "Worker control identity is unavailable; "
                        "cannot confirm process cleanup."
                    ),
                    "code": "worker_identity_unavailable",
                    "status": status,
                    "pid": job.get("pid"),
                }
        else:
            try:
                process_terminated = request_worker_stop(
                    self._job_store.workspace.root, handle
                )
            except (OSError, ValueError, RuntimeError) as exc:
                cleanup_job = self._job_store.get_job(job_id) or job
                return {
                    "success": False,
                    "error": f"Worker cleanup could not be confirmed: {exc}",
                    "code": "worker_cleanup_failed",
                    "status": cleanup_job["status"],
                    "pid": handle.pid,
                }

        latest = self._job_store.get_job(job_id)
        if latest is not None and latest["status"] in self._TERMINAL_STATUSES:
            return {
                "success": False,
                "code": "invalid_job_state",
                "error": "Job finished before cancellation was recorded.",
                "status": latest["status"],
            }

        if job.get("job_type") == "comparison":
            try:
                ComparisonService(job_store=self._job_store).reconcile(
                    job_id,
                    reason="Cancelled by user.",
                    active_status="cancelled",
                )
            except (OSError, TypeError, ValueError) as exc:
                latest = self._job_store.update_job(
                    job_id,
                    status="failed",
                    error=str(exc),
                )
                return {
                    "success": False,
                    "error": f"Comparison evidence could not be finalized: {exc}",
                    "code": "execution_manifest_invalid",
                    "status": latest["status"] if latest else status,
                }

        latest = self._job_store.update_job(
            job_id,
            status="cancelled",
            error="Cancelled by user.",
            ended_at=datetime.now(UTC).isoformat(),
        )
        if latest is None or latest["status"] != "cancelled":
            return {
                "success": False,
                "code": "invalid_job_state",
                "error": "Job finished before cancellation was recorded.",
                "status": latest["status"] if latest else status,
            }
        return {
            "success": True,
            "job_id": job_id,
            "status": "cancelled",
            "process_terminated": process_terminated,
        }

    @staticmethod
    def _process_identity(job: dict[str, Any]) -> ProcessIdentity | None:
        return ProcessIdentity.from_values(
            job.get("pid"), job.get("process_create_time")
        )

    def _is_process_alive(self, job: dict[str, Any]) -> bool:
        identity = self._process_identity(job)
        return identity is not None and identity.get_process() is not None
