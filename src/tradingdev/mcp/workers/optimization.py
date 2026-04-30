"""Subprocess worker that executes a parameter optimization job.

Invoked by the MCP server as a detached subprocess:

    uv run python -m tradingdev.mcp.workers.optimization <job_id>

The worker reads all configuration from job_store (populated by
start_optimization), then:

1. Downloads / loads OHLCV data
2. Runs a single trial combo with a 5-minute timeout
3. Reports estimated total time and waits for user confirmation
4. Runs all remaining combos in parallel batches
5. Selects the best params and runs an out-of-sample test
6. Persists results to job_store
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import logging
import os
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed

from tradingdev.data.data_manager import DataManager
from tradingdev.data.schemas import BacktestConfig, DataConfig
from tradingdev.mcp import job_store
from tradingdev.mcp.workers.backtest import (
    _create_engine,
    _load_strategy_class,
    _serialize_metrics,
)
from tradingdev.utils.config import load_config
from tradingdev.utils.logger import setup_logger
from tradingdev.utils.parallel import estimate_n_jobs

logger = setup_logger(__name__)


def _default_project_root() -> Path:
    configured = os.environ.get("TRADINGDEV_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd().resolve()


_PROJECT_ROOT = _default_project_root()

# Trial run timeout in seconds
_TRIAL_TIMEOUT_SECONDS = 300

# Max time to wait for user confirmation (seconds)
_CONFIRMATION_TIMEOUT_SECONDS = 1800

# Polling interval for confirmation check (seconds)
_CONFIRMATION_POLL_INTERVAL = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _fail(job_id: str, error: str) -> None:
    """Mark a job as failed and log the error."""
    logger.error("Job %s failed: %s", job_id, error)
    job_store.update_job(
        job_id,
        status="failed",
        error=error,
        end_time=_now_iso(),
    )


class _TrialTimeoutError(Exception):
    """Raised when the trial run exceeds the timeout."""


def _trial_timeout_handler(signum: int, frame: Any) -> None:
    raise _TrialTimeoutError("Trial run exceeded 5-minute timeout")


# ---------------------------------------------------------------------------
# Module-level evaluation function (must be picklable for joblib)
# ---------------------------------------------------------------------------
def _evaluate_combo(
    strategy_file: str,
    strategy_class_name: str,
    bt_cfg_dict: dict[str, Any],
    df_json: str,
    param_dict: dict[str, Any],
    metric_name: str,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Evaluate a single parameter combination.

    This function is called by joblib workers.  It re-loads the strategy
    class from source to avoid pickle issues with dynamically loaded modules.

    Args:
        strategy_file: Absolute path to the strategy .py file.
        strategy_class_name: Name of the strategy class.
        bt_cfg_dict: BacktestConfig fields as a plain dict.
        df_json: Training DataFrame serialised as JSON string.
        param_dict: Parameter combination to evaluate.
        metric_name: Target metric to extract.

    Returns:
        (param_dict, target_metric_value, all_serialised_metrics)
    """
    import pandas as pd

    from tradingdev.data.schemas import BacktestConfig
    from tradingdev.mcp.workers.backtest import (
        _create_engine,
        _load_strategy_class,
        _serialize_metrics,
    )

    # Reconstruct DataFrame
    df = pd.read_json(df_json, orient="split")

    # Load strategy class
    strategy_cfg = {"file": strategy_file, "class": strategy_class_name}
    cls = _load_strategy_class(strategy_cfg)

    # Create engine
    bt_cfg = BacktestConfig(**bt_cfg_dict)
    engine = _create_engine(bt_cfg)

    # Instantiate strategy with params
    sig = inspect.signature(cls)
    kwargs: dict[str, Any] = {}
    if "backtest_engine" in sig.parameters:
        kwargs["backtest_engine"] = engine
    for k, v in param_dict.items():
        if k in sig.parameters:
            kwargs[k] = v
    strategy = cls(**kwargs)

    # Run backtest
    signals_df = strategy.generate_signals(df)
    result = engine.run(signals_df)

    target_val = result.metrics.get(metric_name, float("-inf"))
    serialized = _serialize_metrics(result.metrics)

    return param_dict, float(target_val), serialized


def _run_single_combo(
    strategy_file: str,
    strategy_class_name: str,
    bt_cfg: BacktestConfig,
    df: Any,
    param_dict: dict[str, Any],
    metric_name: str,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Run a single combo in the main process (for trial run)."""
    engine = _create_engine(bt_cfg)

    cls = _load_strategy_class({"file": strategy_file, "class": strategy_class_name})

    sig = inspect.signature(cls)
    kwargs: dict[str, Any] = {}
    if "backtest_engine" in sig.parameters:
        kwargs["backtest_engine"] = engine
    for k, v in param_dict.items():
        if k in sig.parameters:
            kwargs[k] = v
    strategy = cls(**kwargs)

    signals_df = strategy.generate_signals(df)
    result = engine.run(signals_df)

    target_val = result.metrics.get(metric_name, float("-inf"))
    serialized = _serialize_metrics(result.metrics)

    return param_dict, float(target_val), serialized


# ---------------------------------------------------------------------------
# Main worker logic
# ---------------------------------------------------------------------------
def _run_optimization(job_id: str) -> None:  # noqa: C901, PLR0912, PLR0915
    logger.info("Optimization worker started: job=%s", job_id)

    job = job_store.get_job(job_id)
    if job is None:
        logger.error("Job %s not found in store", job_id)
        return

    # Extract optimization parameters from job record
    config_path = Path(job["config_path"])
    param_ranges: dict[str, list[Any]] = job["param_ranges"]
    optimization_metric: str = job["optimization_metric"]
    train_start: str = job["train_start"]
    train_end: str = job["train_end"]
    test_start: str = job["test_start"]
    test_end: str = job["test_end"]

    # --- Phase 1: mark running & record PID ---
    job_store.update_job(job_id, status="downloading_data", pid=os.getpid())

    # --- Phase 2: load config ---
    try:
        raw_config: dict[str, Any] = load_config(config_path)
        # Override date range to cover full period
        bt_raw = dict(raw_config["backtest"])
        bt_raw["start_date"] = train_start
        bt_raw["end_date"] = test_end
        bt_cfg = BacktestConfig(**bt_raw)
        data_cfg = DataConfig(**raw_config.get("data", {}))
    except Exception as exc:
        _fail(job_id, f"Config error: {exc}")
        return

    # --- Phase 3: load / download data ---
    try:
        manager = DataManager(data_config=data_cfg, backtest_config=bt_cfg)
        full_df, _ = manager.load()
        job_store.update_job(job_id, data_downloaded=True)
        logger.info("Data loaded: %d rows", len(full_df))
    except Exception as exc:
        _fail(job_id, f"Data error: {exc}")
        return

    # Split into train / test
    import pandas as pd

    train_df = full_df[
        (full_df["timestamp"] >= pd.Timestamp(train_start, tz="UTC"))
        & (
            full_df["timestamp"]
            < pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(days=1)
        )
    ].copy()
    test_df = full_df[
        (full_df["timestamp"] >= pd.Timestamp(test_start, tz="UTC"))
        & (
            full_df["timestamp"]
            < pd.Timestamp(test_end, tz="UTC") + pd.Timedelta(days=1)
        )
    ].copy()

    if train_df.empty:
        _fail(job_id, f"No training data found for {train_start} ~ {train_end}")
        return
    if test_df.empty:
        _fail(job_id, f"No test data found for {test_start} ~ {test_end}")
        return

    logger.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    # --- Phase 4: load strategy class & validate params ---
    try:
        strategy_cfg: dict[str, Any] = raw_config["strategy"]
        strategy_file = str((_PROJECT_ROOT / strategy_cfg["file"]).resolve())
        strategy_class_name = strategy_cfg["class"]
        cls = _load_strategy_class(strategy_cfg)
    except Exception as exc:
        _fail(job_id, f"Strategy load error: {exc}")
        return

    # Validate param_ranges keys against __init__ signature
    sig = inspect.signature(cls)
    invalid_params = [k for k in param_ranges if k not in sig.parameters]
    if invalid_params:
        _fail(
            job_id,
            f"Parameters not found in {strategy_class_name}.__init__: "
            f"{invalid_params}. Available: {list(sig.parameters.keys())}",
        )
        return

    # Build all combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    all_combos = [
        dict(zip(param_names, combo, strict=True))
        for combo in itertools.product(*param_values)
    ]
    total_combinations = len(all_combos)
    logger.info("Total combinations: %d", total_combinations)

    # --- Phase 5: trial run with timeout ---
    job_store.update_job(
        job_id,
        status="estimating",
        total_combinations=total_combinations,
    )

    # Use BacktestConfig with training period for evaluation
    train_bt_raw = dict(raw_config["backtest"])
    train_bt_raw["start_date"] = train_start
    train_bt_raw["end_date"] = train_end
    train_bt_cfg = BacktestConfig(**train_bt_raw)

    first_combo = all_combos[0]
    trial_result: tuple[dict[str, Any], float, dict[str, Any]] | None = None

    # Set up SIGALRM timeout
    old_handler = signal.signal(signal.SIGALRM, _trial_timeout_handler)
    signal.alarm(_TRIAL_TIMEOUT_SECONDS)
    try:
        t0 = time.monotonic()
        trial_result = _run_single_combo(
            strategy_file,
            strategy_class_name,
            train_bt_cfg,
            train_df,
            first_combo,
            optimization_metric,
        )
        time_per_combo = time.monotonic() - t0
    except _TrialTimeoutError:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        job_store.update_job(
            job_id,
            status="estimation_timeout",
            error=(
                f"Trial run exceeded {_TRIAL_TIMEOUT_SECONDS}s timeout. "
                "This strategy is too slow for parameter optimization."
            ),
            end_time=_now_iso(),
        )
        logger.warning("Job %s: trial run timed out", job_id)
        return
    except Exception as exc:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        _fail(job_id, f"Trial run error: {exc}")
        return
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    # Estimate total time
    n_jobs = estimate_n_jobs(train_df)
    # Remaining combos after trial (first already done)
    remaining_count = total_combinations - 1
    estimated_total_seconds = round(
        time_per_combo + (time_per_combo * remaining_count / max(n_jobs, 1)),
        1,
    )

    logger.info(
        "Trial: %.2fs/combo, %d combos, %d workers → est %.1fs total",
        time_per_combo,
        total_combinations,
        n_jobs,
        estimated_total_seconds,
    )

    job_store.update_job(
        job_id,
        status="pending_confirmation",
        time_per_combo=round(time_per_combo, 2),
        estimated_total_seconds=estimated_total_seconds,
        n_parallel_workers=n_jobs,
    )

    # --- Phase 6: wait for confirmation ---
    confirmation_start = time.monotonic()
    confirmed = False
    while time.monotonic() - confirmation_start < _CONFIRMATION_TIMEOUT_SECONDS:
        current_job = job_store.get_job(job_id)
        if current_job is None:
            logger.error("Job %s disappeared from store", job_id)
            return
        if current_job.get("confirmed"):
            confirmed = True
            break
        time.sleep(_CONFIRMATION_POLL_INTERVAL)

    if not confirmed:
        job_store.update_job(
            job_id,
            status="failed",
            error="No confirmation received within 30 minutes. Job cancelled.",
            end_time=_now_iso(),
        )
        logger.warning("Job %s: confirmation timeout", job_id)
        return

    logger.info("Job %s confirmed, running optimization", job_id)

    # --- Phase 7: run all combos in parallel batches ---
    job_store.update_job(
        job_id,
        status="optimizing",
        completed=1,
        total_combinations=total_combinations,
    )

    # Collect all results (including trial)
    all_results: list[tuple[dict[str, Any], float, dict[str, Any]]] = []
    if trial_result is not None:
        all_results.append(trial_result)

    remaining_combos = all_combos[1:]
    optimization_start = time.monotonic()

    # Serialise train_df once for joblib workers
    train_df_json = train_df.to_json(orient="split", date_format="iso")

    batch_size = max(n_jobs * 2, 10)
    for batch_start in range(0, len(remaining_combos), batch_size):
        batch = remaining_combos[batch_start : batch_start + batch_size]

        try:
            batch_results: list[
                tuple[dict[str, Any], float, dict[str, Any]]
            ] = Parallel(n_jobs=n_jobs)(
                delayed(_evaluate_combo)(
                    strategy_file,
                    strategy_class_name,
                    train_bt_cfg.model_dump(),
                    train_df_json,
                    combo,
                    optimization_metric,
                )
                for combo in batch
            )
        except Exception as exc:
            _fail(job_id, f"Optimization error at batch {batch_start}: {exc}")
            return

        all_results.extend(batch_results)

        # Update progress
        completed = 1 + batch_start + len(batch)
        elapsed = time.monotonic() - optimization_start
        rate = completed / elapsed if elapsed > 0 else 1.0
        remaining_seconds = (total_combinations - completed) / rate

        job_store.update_job(
            job_id,
            completed=completed,
            estimated_remaining_seconds=round(remaining_seconds, 1),
        )

    logger.info("All %d combos evaluated", total_combinations)

    # --- Phase 8: select best params ---
    best_params, best_metric_value, best_train_metrics = max(
        all_results, key=lambda x: x[1]
    )
    logger.info(
        "Best params: %s (%s=%.4f)",
        json.dumps(best_params),
        optimization_metric,
        best_metric_value,
    )

    # --- Phase 9: out-of-sample test ---
    job_store.update_job(job_id, status="testing_oos")

    test_bt_raw = dict(raw_config["backtest"])
    test_bt_raw["start_date"] = test_start
    test_bt_raw["end_date"] = test_end
    test_bt_cfg = BacktestConfig(**test_bt_raw)

    try:
        _, oos_metric_value, oos_metrics = _run_single_combo(
            strategy_file,
            strategy_class_name,
            test_bt_cfg,
            test_df,
            best_params,
            optimization_metric,
        )
    except Exception as exc:
        _fail(job_id, f"Out-of-sample test error: {exc}")
        return

    logger.info("OOS result: %s=%.4f", optimization_metric, oos_metric_value)

    # --- Phase 10: persist results ---
    try:
        optimization_result: dict[str, Any] = {
            "best_params": best_params,
            "optimization_metric": optimization_metric,
            "best_train_metric_value": best_metric_value,
            "best_oos_metric_value": oos_metric_value,
            "train_metrics": best_train_metrics,
            "test_metrics": oos_metrics,
            "total_combinations": total_combinations,
            "time_per_combo": round(time_per_combo, 2),
            "n_parallel_workers": n_jobs,
        }
        result_path = job_store.save_result(job_id, optimization_result)
        job_store.update_job(
            job_id,
            status="done",
            end_time=_now_iso(),
            result_path=str(result_path),
            best_params=best_params,
        )
        logger.info("Job %s done → %s", job_id, result_path)
    except Exception as exc:
        _fail(job_id, f"Result save error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCP optimization worker (run as subprocess)"
    )
    parser.add_argument("job_id", help="Job ID assigned by MCP server")
    args = parser.parse_args()
    _run_optimization(args.job_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
