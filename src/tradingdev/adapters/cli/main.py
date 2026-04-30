"""CLI adapter for running TradingDev backtests."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tradingdev.adapters.cli.report import format_metrics_report
from tradingdev.app.backtest_service import BacktestService
from tradingdev.domain.validation.report import format_walk_forward_report
from tradingdev.shared.utils.cache import save_cached_result
from tradingdev.shared.utils.config import load_config
from tradingdev.shared.utils.logger import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    """Run a config from the command line."""
    parser = argparse.ArgumentParser(description="TradingDev backtest runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run the config validation section as walk-forward.",
    )
    args = parser.parse_args()

    t_start = time.monotonic()
    raw_config = load_config(args.config)
    strategy_cfg = raw_config["strategy"]
    logger.info("Strategy: %s", strategy_cfg.get("id"))

    service = BacktestService()
    run = service.run_config(args.config, walk_forward=args.walk_forward)
    if run.mode == "walk_forward":
        report = format_walk_forward_report(run.pipeline.fold_results)
    else:
        mode = (
            run.pipeline.backtest_result.mode
            if run.pipeline.backtest_result
            else "signal"
        )
        report = format_metrics_report(run.metrics, mode=mode)
    logger.info("Backtest results:\n%s", report)

    elapsed = time.monotonic() - t_start
    minutes, seconds = divmod(elapsed, 60)
    elapsed_str = (
        f"{int(minutes)}m {seconds:.1f}s" if minutes >= 1 else f"{seconds:.1f}s"
    )
    logger.info("Elapsed time: %s", elapsed_str)

    cache_file = save_cached_result(run.pipeline, args.config, run.processed_path)
    logger.info("Result cached -> %s", cache_file)


if __name__ == "__main__":
    main()
