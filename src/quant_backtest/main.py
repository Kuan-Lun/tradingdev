"""Main entry point for the backtest pipeline.

Usage::

    uv run python -m quant_backtest.main --config configs/kd_strategy.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

from strategies.registry import create_strategy

from quant_backtest.backtest.pipeline_result import (
    PipelineResult,
)
from quant_backtest.backtest.report import (
    format_metrics_report,
)
from quant_backtest.backtest.signal_engine import (
    SignalBacktestEngine,
)
from quant_backtest.backtest.volume_engine import (
    VolumeBacktestEngine,
)
from quant_backtest.data.data_manager import DataManager
from quant_backtest.data.loader import DataLoader
from quant_backtest.data.schemas import (
    BacktestConfig,
    DataConfig,
    ParallelConfig,
    WalkForwardConfig,
)
from quant_backtest.utils.cache import save_cached_result
from quant_backtest.utils.config import load_config
from quant_backtest.utils.logger import setup_logger
from quant_backtest.validation.report import (
    format_walk_forward_report,
)
from quant_backtest.validation.walk_forward import (
    WalkForwardValidator,
)

if TYPE_CHECKING:
    import pandas as pd

    from quant_backtest.backtest.base_engine import (
        BaseBacktestEngine,
    )

logger = setup_logger(__name__)


def main() -> None:
    """Run the full backtest pipeline."""
    parser = argparse.ArgumentParser(
        description="Strategy Backtester",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    t_start = time.monotonic()

    raw_config = load_config(args.config)
    bt_cfg = BacktestConfig(**raw_config["backtest"])
    parallel_cfg = ParallelConfig(**raw_config.get("parallel", {}))

    logger.info("Strategy: %s", raw_config["strategy"]["name"])

    df, processed_path = _load_data(raw_config, bt_cfg)
    engine = _create_engine(bt_cfg)
    strategy = create_strategy(raw_config, engine, parallel_cfg)

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
        report = format_metrics_report(result.metrics, mode=result.mode)
        logger.info("Backtest results:\n%s", report)
        pipeline = PipelineResult(
            mode="simple",
            backtest_result=result,
            config_snapshot=raw_config,
        )

    elapsed = time.monotonic() - t_start
    minutes, seconds = divmod(elapsed, 60)
    if minutes >= 1:
        elapsed_str = f"{int(minutes)}m {seconds:.1f}s"
    else:
        elapsed_str = f"{seconds:.1f}s"
    logger.info("Elapsed time: %s", elapsed_str)

    cache_file = save_cached_result(pipeline, args.config, processed_path)
    logger.info("Result cached → %s", cache_file)


def _create_engine(
    config: BacktestConfig,
) -> BaseBacktestEngine:
    """Create the appropriate engine based on mode."""
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


def _load_data(
    raw_config: dict[str, object],
    bt_cfg: BacktestConfig,
) -> tuple[pd.DataFrame, Path]:
    """Load or fetch+process data via yearly-cached DataManager.

    Returns:
        Tuple of (OHLCV DataFrame, effective processed path for cache key).
    """
    data_cfg = DataConfig(**raw_config.get("data", {}))  # type: ignore[arg-type]
    manager = DataManager(data_config=data_cfg, backtest_config=bt_cfg)
    df, processed_path = manager.load()

    # Merge DVOL data when strategy uses implied volatility
    # or when dvol_processed_path is explicitly set
    strategy_cfg = raw_config["strategy"]
    params = strategy_cfg["parameters"]  # type: ignore[index]
    needs_dvol = params.get("vol_type") == "implied" or (
        params.get("dvol_processed_path") and params["dvol_processed_path"] != ""
    )
    if needs_dvol:
        loader = DataLoader()
        df = _merge_dvol(df, params, bt_cfg, loader)

    return df, processed_path


def _merge_dvol(
    df: pd.DataFrame,
    params: dict[str, object],
    bt_cfg: BacktestConfig,
    loader: DataLoader,
) -> pd.DataFrame:
    """Fetch (if needed) and merge DVOL data into the OHLCV frame."""
    dvol_processed_str = params.get("dvol_processed_path")
    if dvol_processed_str is None:
        msg = (
            "dvol_processed_path must be set in strategy "
            "parameters when vol_type is 'implied'"
        )
        raise ValueError(msg)
    dvol_processed = Path(str(dvol_processed_str))
    if not dvol_processed.exists():
        from quant_backtest.crawlers.deribit_dvol import (
            DeribitDVOLCrawler,
        )

        logger.info("Fetching DVOL data from Deribit API...")
        dvol_crawler = DeribitDVOLCrawler()
        # DVOL uses currency (e.g. "BTC"), not pair
        currency = bt_cfg.symbol.split("/")[0]
        dvol_raw_df = dvol_crawler.fetch(
            symbol=currency,
            timeframe=bt_cfg.timeframe,
            start=bt_cfg.start_date,
            end=bt_cfg.end_date,
        )
        dvol_raw_path = Path(
            str(
                params.get(
                    "dvol_raw_path",
                    "data/raw/btc_dvol.csv",
                )
            )
        )
        dvol_crawler.save_raw(dvol_raw_df, dvol_raw_path)
        # Skip DataProcessor.process() — it expects OHLCV columns.
        # The crawler already sorts, deduplicates, and filters.
        # Just ensure UTC and save as parquet.
        if dvol_raw_df["timestamp"].dt.tz is None:
            dvol_raw_df["timestamp"] = dvol_raw_df["timestamp"].dt.tz_localize("UTC")
        dvol_processed.parent.mkdir(parents=True, exist_ok=True)
        dvol_raw_df.to_parquet(dvol_processed, index=False)

    dvol_df = loader.load_parquet(dvol_processed)
    dvol_df = dvol_df[["timestamp", "dvol_close"]].rename(
        columns={"dvol_close": "dvol"}
    )
    df = df.merge(dvol_df, on="timestamp", how="left")

    n_missing = int(df["dvol"].isna().sum())
    if n_missing > 0:
        logger.warning("Forward-filling %d missing DVOL values", n_missing)
        df["dvol"] = df["dvol"].ffill().bfill()

    return df


if __name__ == "__main__":
    main()
