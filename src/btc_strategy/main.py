"""Main entry point for running the full backtest pipeline.

Usage::

    uv run python -m btc_strategy.main --config configs/kd_strategy.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from btc_strategy.backtest.engine import BacktestEngine
from btc_strategy.backtest.metrics import format_metrics_report
from btc_strategy.crawlers.binance_api import BinanceAPICrawler
from btc_strategy.data.loader import DataLoader
from btc_strategy.data.processor import DataProcessor
from btc_strategy.data.schemas import (
    BacktestConfig,
    KDFitConfig,
    KDStrategyConfig,
    WalkForwardConfig,
    XGBoostStrategyConfig,
)
from btc_strategy.strategies.kd_strategy import KDStrategy
from btc_strategy.strategies.xgboost_strategy import XGBoostStrategy
from btc_strategy.utils.config import load_config
from btc_strategy.utils.logger import setup_logger
from btc_strategy.validation.walk_forward import (
    WalkForwardValidator,
    format_walk_forward_report,
)

if TYPE_CHECKING:
    from btc_strategy.strategies.base import BaseStrategy

logger = setup_logger(__name__)


def _create_strategy(
    raw_config: dict[str, Any],
    engine: BacktestEngine,
) -> BaseStrategy:
    """Build a strategy instance from YAML configuration."""
    name: str = raw_config["strategy"]["name"]
    params = raw_config["strategy"]["parameters"]

    if name == "kd_crossover":
        strategy_config = KDStrategyConfig(**params)
        fit_config: KDFitConfig | None = None
        if "fit" in raw_config["strategy"]:
            fit_config = KDFitConfig(
                **raw_config["strategy"]["fit"]
            )
        return KDStrategy(
            config=strategy_config,
            fit_config=fit_config,
            backtest_engine=engine,
        )

    if name == "xgboost_direction":
        strategy_config_xgb = XGBoostStrategyConfig(**params)
        return XGBoostStrategy(config=strategy_config_xgb)

    msg = f"Unknown strategy: {name}"
    raise ValueError(msg)


def main() -> None:
    """Run the full backtest pipeline."""
    parser = argparse.ArgumentParser(
        description="BTC/USDT Strategy Backtester",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to strategy YAML config",
    )
    args = parser.parse_args()

    # 1. Load configuration
    raw_config = load_config(args.config)
    backtest_config = BacktestConfig(**raw_config["backtest"])

    logger.info("Strategy: %s", raw_config["strategy"]["name"])
    logger.info(
        "Backtest period: %s to %s",
        backtest_config.start_date,
        backtest_config.end_date,
    )

    # 2. Fetch data (skip if processed file already exists)
    processed_path = Path(raw_config["data"]["processed_path"])
    if not processed_path.exists():
        logger.info("Fetching data from Binance API...")
        crawler = BinanceAPICrawler()
        raw_df = crawler.fetch(
            symbol=backtest_config.symbol,
            timeframe=backtest_config.timeframe,
            start=backtest_config.start_date,
            end=backtest_config.end_date,
        )

        raw_path = Path(raw_config["data"]["raw_path"])
        crawler.save_raw(raw_df, raw_path)

        logger.info("Processing data...")
        processor = DataProcessor()
        processed_df = processor.process(raw_df)
        processor.save_processed(processed_df, processed_path)
    else:
        logger.info(
            "Found existing processed data at %s",
            processed_path,
        )

    # 3. Load processed data
    loader = DataLoader()
    df = loader.load_parquet(processed_path)

    # 4. Create engine and strategy
    engine = BacktestEngine(
        init_cash=backtest_config.init_cash,
        fees=backtest_config.fees,
        slippage=backtest_config.slippage,
        freq=backtest_config.timeframe,
    )
    strategy = _create_strategy(raw_config, engine)

    # 5. Walk-forward validation or simple backtest
    if "validation" in raw_config:
        wf_config = WalkForwardConfig(
            **raw_config["validation"]
        )
        validator = WalkForwardValidator(
            config=wf_config, engine=engine
        )
        results = validator.validate(strategy, df)
        report = format_walk_forward_report(results)
        logger.info(
            "Walk-forward validation results:\n%s", report
        )
    else:
        # Simple single-run backtest (original flow)
        logger.info("Generating signals...")
        df_with_signals = strategy.generate_signals(df)

        n_long = (df_with_signals["signal"] == 1).sum()
        n_short = (df_with_signals["signal"] == -1).sum()
        logger.info(
            "Signals generated: %d long, %d short",
            n_long,
            n_short,
        )

        logger.info("Running backtest...")
        metrics = engine.run(df_with_signals)
        report = format_metrics_report(metrics)
        logger.info("Backtest results:\n%s", report)


if __name__ == "__main__":
    main()
