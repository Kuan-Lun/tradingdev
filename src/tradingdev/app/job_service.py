"""Application service for background jobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from tradingdev.adapters.execution.process_runner import (
    ProcessRunner,
)
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.contracts.common import DEFAULT_PAGE_LIMIT, Page
from tradingdev.app.data_service import DataService
from tradingdev.app.execution_submission import (
    ExecutionSubmissionService,
)
from tradingdev.app.job_config import (
    apply_backtest_overrides,
    apply_run_overrides,
    bind_strategy_revision,
)
from tradingdev.app.job_lifecycle import JobLifecycle
from tradingdev.app.job_queries import JobQueries
from tradingdev.app.job_store import (
    JobStore,
    get_default_job_store,
)
from tradingdev.app.prepared_execution import PreparedExecution
from tradingdev.app.strategy_service import (
    StrategyNotExecutableError,
    StrategyService,
)
from tradingdev.domain.backtest.schemas import BacktestRunConfig
from tradingdev.domain.strategies.capabilities import (
    StrategyExecutionModeError,
    ensure_execution_mode,
)
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.shared.utils.config import load_config

if TYPE_CHECKING:
    from tradingdev.app.contracts.jobs import JobState, JobSummary
    from tradingdev.domain.strategies.schemas import StrategySpec


class JobService:
    """Create and query background jobs."""

    def __init__(
        self,
        *,
        strategy_service: StrategyService | None = None,
        data_service: DataService | None = None,
        strategy_loader: StrategyLoader | None = None,
        job_store: JobStore | None = None,
        process_runner: ProcessRunner | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._job_store = job_store or get_default_job_store()
        self._lifecycle = JobLifecycle(self._job_store)
        self._queries = JobQueries(self._job_store)
        self._strategy_service = strategy_service or StrategyService(
            self._job_store.workspace
        )
        self._data_service = data_service or DataService(self._job_store.workspace)
        self._strategy_loader = strategy_loader or StrategyLoader(
            workspace_root=self._job_store.workspace.root
        )
        self._project_root = (project_root or self._default_project_root()).resolve()
        self._process_runner = process_runner or ProcessRunner(
            self._project_root, workspace=self._job_store.workspace
        )
        self._submission = ExecutionSubmissionService(
            self._job_store, self._process_runner
        )

    def start_backtest(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        revision_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        backtest_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare and submit a simple backtest job."""
        prepared = self.prepare_backtest(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            revision_id=revision_id,
            parameters=parameters,
            backtest_overrides=backtest_overrides,
        )
        return self._submit_prepared(prepared)

    def prepare_backtest(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        revision_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        backtest_overrides: dict[str, Any] | None = None,
    ) -> PreparedExecution | dict[str, Any]:
        """Capture a simple backtest without creating a job or launching a worker."""
        try:
            spec, error = self._resolve_strategy_run_config(strategy_id, revision_id)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            return self._invalid_execution_request(exc)
        if spec is None:
            return {
                "job_id": "",
                "message": error,
                "data_available": False,
                "code": "strategy_not_executable",
            }
        config_path = Path(spec.config_path)
        try:
            raw_config = load_config(config_path)
            run_config = BacktestRunConfig.model_validate(raw_config)
            ensure_execution_mode(
                spec.strategy_id, spec.execution_capabilities, "backtest"
            )
        except StrategyExecutionModeError as exc:
            return {
                "job_id": "",
                "message": str(exc),
                "data_available": False,
                "code": exc.code,
                "execution_guidance": exc.guidance.model_dump(mode="json"),
            }
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            return self._invalid_execution_request(exc)
        if run_config.is_walk_forward:
            return {
                "job_id": "",
                "message": (
                    "Config contains validation settings; use prepare_walk_forward."
                ),
                "data_available": False,
                "code": "invalid_run_mode",
            }
        return self._prepare_execution(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            config_path=config_path,
            raw_config=raw_config,
            walk_forward=False,
            spec=spec,
            parameters=parameters,
            backtest_overrides=backtest_overrides,
        )

    def start_walk_forward(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        revision_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        backtest_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare and submit a walk-forward job."""
        prepared = self.prepare_walk_forward(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            revision_id=revision_id,
            parameters=parameters,
            backtest_overrides=backtest_overrides,
        )
        return self._submit_prepared(prepared)

    def prepare_walk_forward(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        revision_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        backtest_overrides: dict[str, Any] | None = None,
    ) -> PreparedExecution | dict[str, Any]:
        """Capture a walk-forward without creating a job or launching a worker."""
        try:
            spec, error = self._resolve_strategy_run_config(strategy_id, revision_id)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            return self._invalid_execution_request(exc)
        if spec is None:
            return {
                "job_id": "",
                "message": error,
                "data_available": False,
                "code": "strategy_not_executable",
            }
        try:
            resolved = self._resolve_walk_forward_config(
                strategy_id=strategy_id,
                config_path=Path(spec.config_path),
                allow_fallback=spec.kind == "bundled",
            )
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            return self._invalid_execution_request(exc)
        if resolved is None:
            return {
                "job_id": "",
                "message": "Config has no validation section for walk-forward.",
                "data_available": False,
                "code": "invalid_run_mode",
            }
        config_path, raw_config = resolved
        return self._prepare_execution(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            config_path=config_path,
            raw_config=raw_config,
            walk_forward=True,
            spec=spec,
            parameters=parameters,
            backtest_overrides=backtest_overrides,
        )

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Recover stopped workers and return current status with result details."""
        observation = self._lifecycle.refresh(job_id)
        return self._queries.status(job_id, observation)

    def list_jobs(self) -> list[dict[str, Any]]:
        """List job summaries newest first without polling worker state."""
        return self._queries.list_jobs()

    def list_jobs_page(
        self,
        *,
        status: JobState | None = None,
        job_type: str | None = None,
        strategy_name: str | None = None,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> Page[JobSummary]:
        """Page saved job progress without polling or changing worker state."""
        return self._queries.list_jobs_page(
            status=status,
            job_type=job_type,
            strategy_name=strategy_name,
            offset=offset,
            limit=limit,
        )

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued or running background job."""
        return self._lifecycle.cancel_job(job_id)

    def _prepare_execution(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        config_path: Path,
        raw_config: dict[str, Any],
        walk_forward: bool,
        spec: StrategySpec,
        parameters: dict[str, Any] | None = None,
        backtest_overrides: dict[str, Any] | None = None,
    ) -> PreparedExecution | dict[str, Any]:
        kind: Literal["backtest", "walk_forward"] = (
            "walk_forward" if walk_forward else "backtest"
        )
        try:
            bind_strategy_revision(raw_config, spec)
            effective_config = apply_run_overrides(
                raw_config,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
            apply_backtest_overrides(effective_config, backtest_overrides)
            checked = BacktestService(
                data_service=self._data_service,
                strategy_gate=self._strategy_service,
                strategy_loader=self._strategy_loader,
            ).capture_execution(effective_config, kind=kind, parameters=parameters)
        except (
            OSError,
            TypeError,
            ValueError,
            yaml.YAMLError,
            StrategyNotExecutableError,
        ) as exc:
            return self._invalid_execution_request(exc)

        return PreparedExecution(
            checked.manifest, config_path, checked.synthetic_fixture_rows
        )

    def _submit_prepared(
        self, prepared: PreparedExecution | dict[str, Any]
    ) -> dict[str, Any]:
        if isinstance(prepared, dict):
            return prepared
        backtest = prepared.manifest.config_copy()["backtest"]
        data_available = self._data_service.data_available(
            backtest["symbol"],
            backtest["timeframe"],
            backtest["start_date"],
            backtest["end_date"],
            raw_config=prepared.manifest.config_copy(),
        )
        submitted = self._submission.submit(prepared)
        data_msg = (
            "A reusable market query is cached locally; observations may have gaps."
            if data_available
            else "No reusable market query; worker will query the provider."
        )
        return {
            **submitted,
            "message": f"Job started. Job ID: {submitted['job_id']}. {data_msg}",
            "data_available": data_available,
        }

    def _resolve_walk_forward_config(
        self,
        *,
        strategy_id: str,
        config_path: Path,
        allow_fallback: bool,
    ) -> tuple[Path, dict[str, Any]] | None:
        raw_config = load_config(config_path)
        run_config = BacktestRunConfig.model_validate(raw_config)
        if run_config.is_walk_forward:
            return config_path, raw_config

        fallback = config_path.with_name("walkforward_config.yaml")
        if allow_fallback and fallback.exists():
            fallback_raw = load_config(fallback)
            fallback_run_config = BacktestRunConfig.model_validate(fallback_raw)
            fallback_strategy = fallback_raw.get("strategy", {})
            fallback_id = (
                fallback_strategy.get("id")
                if isinstance(fallback_strategy, dict)
                else None
            )
            if fallback_id == strategy_id and fallback_run_config.is_walk_forward:
                return fallback, fallback_raw
        return None

    @staticmethod
    def _invalid_execution_request(exc: Exception) -> dict[str, Any]:
        """Reject preparation errors before creating a job or starting a worker."""
        return {
            "job_id": "",
            "message": str(exc),
            "data_available": False,
            "code": "invalid_execution_request",
        }

    def _resolve_strategy_run_config(
        self, strategy_id: str, revision_id: str | None
    ) -> tuple[StrategySpec | None, str]:
        try:
            spec = self._strategy_service.resolve_executable(strategy_id, revision_id)
        except StrategyNotExecutableError as exc:
            return None, str(exc)
        path = Path(spec.config_path)
        return (spec, "") if path.exists() else (None, f"Config not found: {path}")

    def _default_project_root(self) -> Path:
        configured = os.environ.get("TRADINGDEV_PROJECT_ROOT")
        if configured:
            return Path(configured).expanduser().resolve()
        return Path.cwd().resolve()
