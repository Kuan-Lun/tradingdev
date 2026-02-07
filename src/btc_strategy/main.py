"""Main entry point for running the full backtest pipeline.

Usage::

    uv run python -m btc_strategy.main --config configs/kd_strategy.yaml
"""

import argparse
from pathlib import Path

from btc_strategy.backtest.engine import BacktestEngine
from btc_strategy.backtest.metrics import format_metrics_report
from btc_strategy.crawlers.binance_api import BinanceAPICrawler
from btc_strategy.data.loader import DataLoader
from btc_strategy.data.processor import DataProcessor
from btc_strategy.data.schemas import BacktestConfig, KDStrategyConfig
from btc_strategy.strategies.kd_strategy import KDStrategy
from btc_strategy.utils.config import load_config
from btc_strategy.utils.logger import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    """Run the full backtest pipeline: fetch → process → signals → backtest."""
    parser = argparse.ArgumentParser(description="BTC/USDT Strategy Backtester")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to strategy YAML config"
    )
    args = parser.parse_args()

    # 1. Load configuration
    raw_config = load_config(args.config)
    backtest_config = BacktestConfig(**raw_config["backtest"])
    strategy_config = KDStrategyConfig(**raw_config["strategy"]["parameters"])

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
        logger.info("Found existing processed data at %s", processed_path)

    # 3. Load processed data
    loader = DataLoader()
    df = loader.load_parquet(processed_path)

    # 4. Generate signals
    logger.info("Generating KD strategy signals...")
    strategy = KDStrategy(config=strategy_config)
    df_with_signals = strategy.generate_signals(df)

    n_long = (df_with_signals["signal"] == 1).sum()
    n_short = (df_with_signals["signal"] == -1).sum()
    logger.info("Signals generated: %d long, %d short", n_long, n_short)

    # 5. Run backtest
    logger.info("Running backtest...")
    engine = BacktestEngine(
        init_cash=backtest_config.init_cash,
        fees=backtest_config.fees,
        slippage=backtest_config.slippage,
        freq=backtest_config.timeframe,
    )
    metrics = engine.run(df_with_signals)

    # 6. Report results
    report = format_metrics_report(metrics)
    logger.info("Backtest results:\n%s", report)


if __name__ == "__main__":
    main()
