"""Job MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.types import ToolAnnotations
from pydantic import TypeAdapter

from tradingdev.app.contracts.jobs import (
    JobActionFailed,
    JobCancelled,
    JobNotFound,
    JobStatus,
    JobSummary,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.job_service import JobService


def register(mcp: FastMCP, service: JobService) -> None:
    """Register job tools."""

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def get_job_status(job_id: str) -> JobStatus | JobNotFound:
        """Check progress, reconciling a vanished worker to persisted failed status.

        When done, use run_id with get_run. pending_confirmation requires user
        approval before confirm_optimization. failed is a job state, not a
        lookup error; not_found means the job ID does not exist.
        """
        return TypeAdapter(JobStatus | JobNotFound).validate_python(
            service.get_job_status(job_id)
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def list_jobs() -> list[JobSummary]:
        """List persisted job summaries; use get_job_status to check worker liveness."""
        return [JobSummary.model_validate(job) for job in service.list_jobs()]

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def cancel_job(job_id: str) -> JobCancelled | JobActionFailed:
        """Stop an active job and acknowledge cleanup before marking it cancelled.

        success=False means cancellation was not confirmed; inspect code and
        get_job_status. Completed jobs cannot be cancelled.
        """
        return TypeAdapter(JobCancelled | JobActionFailed).validate_python(
            service.cancel_job(job_id)
        )
