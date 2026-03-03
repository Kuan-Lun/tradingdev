"""Subprocess worker that executes a single backtest job.

Invoked by the MCP server as a detached subprocess:

    python mcp_server/backtest_worker.py <job_id> <config_path>

The worker updates .backtest_jobs/jobs.json at each stage so the MCP
server can report progress via get_job_status.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so 'quant_backtest' and top-level
# 'strategies/' are importable regardless of the working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp_server import job_store  # noqa: E402 (must come after sys.path fix)
from quant_backtest.backtest.signal_engine import SignalBacktestEngine  # noqa: E402
from quant_backtest.backtest.volume_engine import VolumeBacktestEngine  # noqa: E402
from quant_backtest.data.data_manager import DataManager  # noqa: E402
from quant_backtest.data.schemas import BacktestConfig, DataConfig  # noqa: E402
from quant_backtest.strategies.base import BaseStrategy  # noqa: E402
from quant_backtest.utils.config import load_config  # noqa: E402
from quant_backtest.utils.logger import setup_logger  # noqa: E402

if TYPE_CHECKING:
    from types import ModuleType

    from quant_backtest.backtest.base_engine import BaseBacktestEngine

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Metrics subset to persist (JSON-serialisable keys only)
# ---------------------------------------------------------------------------
_RESULT_METRICS_KEYS = [
    "total_return",
    "total_pnl",
    "annual_return",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "total_trades",
    "monthly_pnl_mean",
    "monthly_pnl_std",
    "monthly_pnl_min",
    "monthly_pnl_max",
    "monthly_pnl_median",
    "n_months",
    "monthly_trades_mean",
]


# ---------------------------------------------------------------------------
# Engine factory — mirrors main.py _create_engine() without importing main.py
# (importing main.py would trigger `from strategies.registry import ...`
# which requires the top-level strategies/ package to be fully populated).
# ---------------------------------------------------------------------------
def _create_engine(config: BacktestConfig) -> BaseBacktestEngine:
    if config.mode == "volume":
        return VolumeBacktestEngine(
            fees=config.fees,
            slippage=config.slippage,
            freq=config.timeframe,
            position_size=config.position_size,
            stop_loss=config.stop_loss,
            take_profit=config.take_profit,
            signal_as_position=config.signal_as_position,
            re_entry_after_sl=config.re_entry_after_sl,
            monthly_max_loss=config.monthly_max_loss,
        )
    return SignalBacktestEngine(
        init_cash=config.init_cash,
        fees=config.fees,
        slippage=config.slippage,
        freq=config.timeframe,
        position_size=config.position_size,
        stop_loss=config.stop_loss,
        take_profit=config.take_profit,
        signal_as_position=config.signal_as_position,
        re_entry_after_sl=config.re_entry_after_sl,
    )


# ---------------------------------------------------------------------------
# Dynamic strategy class loader
# ---------------------------------------------------------------------------
def _load_strategy_class(strategy_cfg: dict[str, Any]) -> type:
    """Load a strategy class from the path specified in the YAML config."""
    file_key = strategy_cfg.get("file")
    class_name: str = strategy_cfg["class"]

    if not file_key:
        raise ValueError(
            "YAML 'strategy.file' is required for dynamically loaded strategies."
        )

    file_path = (_PROJECT_ROOT / file_key).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {file_path}")

    spec = importlib.util.spec_from_file_location("_mcp_dynamic_strategy", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from {file_path}")

    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise AttributeError(
            f"Class '{class_name}' not found in {file_path}. "
            f"Available names: {[n for n in dir(module) if not n.startswith('_')]}"
        )

    return getattr(module, class_name)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Strategy instantiation — adapts to whatever __init__ the class exposes
# ---------------------------------------------------------------------------
def _instantiate_strategy(
    cls: type,
    engine: BaseBacktestEngine,
) -> BaseStrategy:
    """Instantiate a strategy class, injecting backtest_engine if accepted."""
    sig = inspect.signature(cls)
    kwargs: dict[str, Any] = {}
    if "backtest_engine" in sig.parameters:
        kwargs["backtest_engine"] = engine
    instance = cls(**kwargs)
    if not isinstance(instance, BaseStrategy):
        raise TypeError(f"{cls.__name__} must inherit from BaseStrategy.")
    return instance


# ---------------------------------------------------------------------------
# Metrics serialisation
# ---------------------------------------------------------------------------
def _serialize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Extract and JSON-serialise the LLM-relevant subset of metrics."""
    return {k: metrics[k] for k in _RESULT_METRICS_KEYS if k in metrics}


# ---------------------------------------------------------------------------
# Main worker logic
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _run_backtest(job_id: str, config_path: Path) -> None:
    logger.info("Worker started: job=%s config=%s", job_id, config_path)

    # --- Phase 1: mark running & record PID ---
    job_store.update_job(job_id, status="downloading_data", pid=os.getpid())

    # --- Phase 2: load config ---
    try:
        raw_config: dict[str, Any] = load_config(config_path)
        bt_cfg = BacktestConfig(**raw_config["backtest"])
        data_cfg = DataConfig(**raw_config.get("data", {}))
    except Exception as exc:
        logger.exception("Config load failed")
        job_store.update_job(
            job_id,
            status="failed",
            error=f"Config error: {exc}",
            end_time=_now_iso(),
        )
        return

    # --- Phase 3: load / download data ---
    try:
        manager = DataManager(data_config=data_cfg, backtest_config=bt_cfg)
        df, _ = manager.load()
        job_store.update_job(job_id, data_downloaded=True, status="running_backtest")
        logger.info("Data loaded: %d rows", len(df))
    except Exception as exc:
        logger.exception("Data load failed")
        job_store.update_job(
            job_id,
            status="failed",
            error=f"Data error: {exc}",
            end_time=_now_iso(),
        )
        return

    # --- Phase 4: load strategy class ---
    try:
        strategy_cfg: dict[str, Any] = raw_config["strategy"]
        strategy_cls = _load_strategy_class(strategy_cfg)
    except Exception as exc:
        logger.exception("Strategy load failed")
        job_store.update_job(
            job_id,
            status="failed",
            error=f"Strategy load error: {exc}",
            end_time=_now_iso(),
        )
        return

    # --- Phase 5: create engine & instantiate strategy ---
    try:
        engine = _create_engine(bt_cfg)
        strategy = _instantiate_strategy(strategy_cls, engine)
    except Exception as exc:
        logger.exception("Strategy instantiation failed")
        job_store.update_job(
            job_id,
            status="failed",
            error=f"Strategy init error: {exc}",
            end_time=_now_iso(),
        )
        return

    # --- Phase 6: generate signals & run backtest ---
    try:
        signals_df = strategy.generate_signals(df)
        result = engine.run(signals_df)
        n_trades = result.metrics.get("total_trades", 0)
        logger.info("Backtest complete: %d trades", n_trades)
    except Exception as exc:
        logger.exception("Backtest execution failed")
        job_store.update_job(
            job_id,
            status="failed",
            error=f"Backtest error: {exc}",
            end_time=_now_iso(),
        )
        return

    # --- Phase 7: persist results ---
    try:
        metrics = _serialize_metrics(result.metrics)
        result_path = job_store.save_result(job_id, metrics)
        job_store.update_job(
            job_id,
            status="done",
            end_time=_now_iso(),
            result_path=str(result_path),
        )
        logger.info("Job %s done → %s", job_id, result_path)
    except Exception as exc:
        logger.exception("Result save failed")
        job_store.update_job(
            job_id,
            status="failed",
            error=f"Result save error: {exc}",
            end_time=_now_iso(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCP backtest worker (run as subprocess)"
    )
    parser.add_argument("job_id", help="Job ID assigned by MCP server")
    parser.add_argument("config_path", type=Path, help="Path to YAML config file")
    args = parser.parse_args()
    _run_backtest(args.job_id, args.config_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
