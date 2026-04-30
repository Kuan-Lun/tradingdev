"""Subprocess worker for backtest and walk-forward jobs."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from tradingdev.app import job_store
from tradingdev.app.backtest_service import BacktestService
from tradingdev.shared.utils.logger import setup_logger

logger = setup_logger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _run_backtest(
    job_id: str, config_path: Path, *, walk_forward: bool = False
) -> None:
    """Run a background job and persist metrics."""
    logger.info(
        "Worker started: job=%s config=%s walk_forward=%s",
        job_id,
        config_path,
        walk_forward,
    )
    job_store.update_job(job_id, status="downloading_data", pid=os.getpid())

    try:
        service = BacktestService()
        job_store.update_job(job_id, status="running_backtest")
        run = service.run_config(config_path, walk_forward=walk_forward)
        job_store.update_job(job_id, data_downloaded=True, dataset_id=run.dataset_id)
        result_path = job_store.save_result(job_id, run.metrics)
        job_store.update_job(
            job_id,
            status="done",
            end_time=_now_iso(),
            result_path=str(result_path),
        )
        logger.info("Job %s done -> %s", job_id, result_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Worker failed")
        job_store.update_job(
            job_id,
            status="failed",
            error=str(exc),
            end_time=_now_iso(),
        )


def main() -> None:
    """Run the worker."""
    parser = argparse.ArgumentParser(
        description="MCP backtest worker (run as subprocess)"
    )
    parser.add_argument("job_id", help="Job ID assigned by MCP server")
    parser.add_argument("config_path", type=Path, help="Path to YAML config file")
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run the config validation section as walk-forward.",
    )
    args = parser.parse_args()
    _run_backtest(args.job_id, args.config_path, walk_forward=args.walk_forward)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
