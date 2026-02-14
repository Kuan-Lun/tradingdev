"""Main entry point for the backtest pipeline.

Usage::

    uv run python -m btc_strategy.main --config configs/kd_strategy.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from btc_strategy.backtest.pipeline_result import (
    PipelineResult,
)
from btc_strategy.backtest.report import (
    format_metrics_report,
)
from btc_strategy.backtest.signal_engine import (
    SignalBacktestEngine,
)
from btc_strategy.backtest.volume_engine import (
    VolumeBacktestEngine,
)
from btc_strategy.crawlers.binance_api import (
    BinanceAPICrawler,
)
from btc_strategy.data.loader import DataLoader
from btc_strategy.data.processor import DataProcessor
from btc_strategy.data.schemas import (
    BacktestConfig,
    WalkForwardConfig,
)
from btc_strategy.strategies.registry import create_strategy
from btc_strategy.utils.cache import save_cached_result
from btc_strategy.utils.config import load_config
from btc_strategy.utils.logger import setup_logger
from btc_strategy.validation.report import (
    format_walk_forward_report,
)
from btc_strategy.validation.walk_forward import (
    WalkForwardValidator,
)

if TYPE_CHECKING:
    import pandas as pd

    from btc_strategy.backtest.base_engine import (
        BaseBacktestEngine,
    )

logger = setup_logger(__name__)


def main() -> None:
    """Run the full backtest pipeline."""
    parser = argparse.ArgumentParser(
        description="BTC/USDT Strategy Backtester",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    raw_config = load_config(args.config)
    bt_cfg = BacktestConfig(**raw_config["backtest"])

    logger.info("Strategy: %s", raw_config["strategy"]["name"])

    df = _load_data(raw_config, bt_cfg)
    engine = _create_engine(bt_cfg)
    strategy = create_strategy(raw_config, engine)

    if "validation" in raw_config:
        wf_cfg = WalkForwardConfig(**raw_config["validation"])
        validator = WalkForwardValidator(config=wf_cfg, engine=engine)
        fold_results = validator.validate(strategy, df)
        report = format_walk_forward_report(fold_results)
        logger.info("Walk-forward results:\n%s", report)
        pipeline = PipelineResult(
            mode="walk_forward",
            fold_results=fold_results,
            config_snapshot=raw_config,
        )
    else:
        signals = strategy.generate_signals(df)
        result = engine.run(signals)
        report = format_metrics_report(result.metrics)
        logger.info("Backtest results:\n%s", report)
        pipeline = PipelineResult(
            mode="simple",
            backtest_result=result,
            config_snapshot=raw_config,
        )

    processed_path = Path(
        str(raw_config["data"]["processed_path"])
    )
    cache_file = save_cached_result(
        pipeline, args.config, processed_path
    )
    logger.info("Result cached → %s", cache_file)


def _create_engine(
    config: BacktestConfig,
) -> BaseBacktestEngine:
    """Create the appropriate engine based on mode."""
    cls = (
        VolumeBacktestEngine
        if config.mode == "volume"
        else SignalBacktestEngine
    )
    return cls(
        init_cash=config.init_cash,
        fees=config.fees,
        slippage=config.slippage,
        freq=config.timeframe,
        position_size_usdt=config.position_size_usdt,
        stop_loss=config.stop_loss,
        take_profit=config.take_profit,
    )


def _load_data(
    raw_config: dict[str, object],
    bt_cfg: BacktestConfig,
) -> pd.DataFrame:
    """Load or fetch+process data."""
    processed_path = Path(
        str(raw_config["data"]["processed_path"])  # type: ignore[index]
    )
    if not processed_path.exists():
        logger.info("Fetching data from Binance API...")
        crawler = BinanceAPICrawler()
        raw_df = crawler.fetch(
            symbol=bt_cfg.symbol,
            timeframe=bt_cfg.timeframe,
            start=bt_cfg.start_date,
            end=bt_cfg.end_date,
        )
        raw_path = Path(
            str(raw_config["data"]["raw_path"])  # type: ignore[index]
        )
        crawler.save_raw(raw_df, raw_path)
        processor = DataProcessor()
        processed_df = processor.process(raw_df)
        processor.save_processed(processed_df, processed_path)

    loader = DataLoader()
    return loader.load_parquet(processed_path)


if __name__ == "__main__":
    main()
