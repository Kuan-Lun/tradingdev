"""Public job responses, distinct from persisted worker control records."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from tradingdev.app.contracts.common import ContractModel, ErrorResponse

type JobState = Literal[
    "queued",
    "downloading_data",
    "running_backtest",
    "estimating",
    "pending_confirmation",
    "optimizing",
    "testing_oos",
    "done",
    "failed",
    "cancelled",
    "estimation_timeout",
]


class BacktestStarted(ContractModel):
    """A worker was accepted; poll job status before reading a completed run."""

    job_id: Annotated[str, Field(min_length=1)]
    message: str
    data_available: bool


class BacktestRejected(ContractModel):
    """No job was created; correct the request before trying again."""

    job_id: Literal[""]
    message: str
    data_available: Literal[False]
    code: Literal["strategy_not_executable", "invalid_run_mode"]


class OptimizationStarted(ContractModel):
    """An estimation worker was accepted; full search requires confirmation."""

    job_id: Annotated[str, Field(min_length=1)]
    message: str
    total_combinations: Annotated[int, Field(gt=0)]


class OptimizationRejected(ContractModel):
    """No optimization job was created."""

    job_id: Literal[""]
    message: str
    total_combinations: Literal[0]
    code: Literal["strategy_not_executable", "invalid_optimization_request"]


class JobSummary(ContractModel):
    """Public job identity and progress, without process control secrets."""

    job_id: str
    job_type: str
    status: JobState
    strategy_name: str | None
    symbol: str | None
    timeframe: str | None
    start_date: str | None
    end_date: str | None
    elapsed_seconds: float
    data_downloaded: bool
    total_combinations: int | None = None
    completed: int | None = None


class JobStatus(ContractModel):
    """Lifecycle status; stage-specific fields are null when unavailable.

    A failed worker is a successfully retrieved job with status='failed',
    not a failed lookup. Metrics may contain nested optimization results.
    """

    status: JobState
    job_type: str
    strategy_name: str | None
    symbol: str | None
    timeframe: str | None
    start_date: str | None
    end_date: str | None
    elapsed_seconds: float
    ended_at: str | None = None
    run_id: str | None = None
    metrics: dict[str, JsonValue] | None = None
    error: str | None = None
    message: str | None = None
    data_downloaded: bool | None = None
    best_params: dict[str, JsonValue] | None = None
    train_metrics: dict[str, JsonValue] | None = None
    test_metrics: dict[str, JsonValue] | None = None
    optimization_metric: str | None = None
    total_combinations: int | None = None
    time_per_combo: float | None = None
    estimated_total_seconds: float | None = None
    n_parallel_workers: int | None = None
    completed: int | None = None
    estimated_remaining_seconds: float | None = None


class JobNotFound(ContractModel):
    """No persisted job matches the requested identifier."""

    status: Literal["not_found"]
    error: str
    code: Literal["job_not_found"]


class OptimizationConfirmed(ContractModel):
    """The existing estimated job has permission to begin full search."""

    success: Literal[True]
    message: str


class JobActionFailed(ErrorResponse):
    """An expected lifecycle or cleanup failure; no successful action is claimed."""

    code: Literal[
        "job_not_found",
        "invalid_job_state",
        "estimation_timeout",
        "worker_identity_unavailable",
        "worker_cleanup_failed",
    ]
    status: JobState | None = None
    pid: int | None = None


class JobCancelled(ContractModel):
    """Cancellation recorded after any required process cleanup acknowledgement."""

    success: Literal[True]
    job_id: str
    status: Literal["cancelled"]
    process_terminated: bool
