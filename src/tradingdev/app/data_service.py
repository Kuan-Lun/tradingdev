"""Application service for market data access."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.domain.data.crawlers.deribit_dvol import DeribitDVOLCrawler
from tradingdev.domain.data.data_manager import DataManager
from tradingdev.domain.data.loader import DataLoader
from tradingdev.domain.data.requirements import (
    DataRequirement,
    FeatureSpec,
    MarketDataSpec,
)
from tradingdev.domain.data.schemas import DataConfig

if TYPE_CHECKING:
    import pandas as pd

    from tradingdev.domain.backtest.schemas import BacktestConfig


@dataclass(frozen=True)
class LoadedDataset:
    """Loaded market data and cache metadata."""

    frame: pd.DataFrame
    processed_path: Path
    dataset_id: str


class DataService:
    """Load market data from explicit data requirements."""

    _DATA_FILE = re.compile(r"^([a-z0-9]+)_([a-z0-9]+)_(\d{4})(_partial)?\.parquet$")

    def __init__(self, workspace: WorkspacePaths | None = None) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._workspace.ensure()
        self._loader = DataLoader()

    def data_config(self, raw_config: dict[str, Any]) -> DataConfig:
        """Build DataConfig with workspace defaults."""
        raw = raw_config.get("data", {})
        data = dict(raw) if isinstance(raw, dict) else {}
        data_root = os.environ.get("TRADINGDEV_DATA_ROOT")
        if data_root and "raw_dir" not in data and "processed_dir" not in data:
            root = Path(data_root).expanduser()
            data["raw_dir"] = str(root / "raw")
            data["processed_dir"] = str(root / "processed")
        return DataConfig(**data)

    def requirements(
        self,
        raw_config: dict[str, Any],
        backtest_config: BacktestConfig,
    ) -> DataRequirement:
        """Parse explicit data requirements, defaulting to market data only."""
        raw = raw_config.get("data", {})
        data = raw if isinstance(raw, dict) else {}
        req = data.get("requirements") if isinstance(data, dict) else None
        if isinstance(req, dict):
            return DataRequirement(**req)
        return DataRequirement(
            market=MarketDataSpec(
                source=str(data.get("source", "binance_api")),
                symbol=backtest_config.symbol,
                timeframe=backtest_config.timeframe,
            ),
            features=[],
        )

    def load(
        self,
        raw_config: dict[str, Any],
        backtest_config: BacktestConfig,
    ) -> LoadedDataset:
        """Load OHLCV data and merge declared feature sources."""
        data_config = self.data_config(raw_config)
        manager = DataManager(data_config=data_config, backtest_config=backtest_config)
        frame, processed_path = manager.load()
        requirements = self.requirements(raw_config, backtest_config)

        for feature in requirements.features:
            frame = self._merge_feature(frame, feature, backtest_config)

        dataset_id = self._dataset_id(backtest_config, requirements)
        return LoadedDataset(
            frame=frame, processed_path=processed_path, dataset_id=dataset_id
        )

    def list_available_data(self) -> list[dict[str, Any]]:
        """List cached OHLCV parquet files from the workspace data root."""
        processed_dir = self._workspace.processed_data
        if not processed_dir.exists():
            return []

        grouped: dict[tuple[str, str], set[int]] = {}
        for path in sorted(processed_dir.glob("*.parquet")):
            match = self._DATA_FILE.match(path.name)
            if not match:
                continue
            symbol, timeframe = match.group(1), match.group(2)
            grouped.setdefault((symbol, timeframe), set()).add(int(match.group(3)))
        return [
            {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "years_available": sorted(years),
            }
            for (symbol, timeframe), years in sorted(grouped.items())
        ]

    def data_available(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> bool:
        """Return whether yearly processed parquet files are already cached."""
        processed_dir = self._workspace.processed_data
        if not processed_dir.exists():
            return False
        normalized = symbol.replace("/", "").lower()
        try:
            start_year = int(start_date[:4])
            end_year = int(end_date[:4])
        except (ValueError, IndexError):
            return False
        for year in range(start_year, end_year + 1):
            complete = processed_dir / f"{normalized}_{timeframe}_{year}.parquet"
            partial = processed_dir / f"{normalized}_{timeframe}_{year}_partial.parquet"
            if not complete.exists() and not partial.exists():
                return False
        return True

    def inspect_dataset(self) -> dict[str, Any]:
        """Return a lightweight snapshot of cached datasets."""
        files = []
        for item in self.list_available_data():
            files.append(item)
        return {
            "data_root": str(self._workspace.root / "data"),
            "processed_dir": str(self._workspace.processed_data),
            "datasets": files,
        }

    def _merge_feature(
        self,
        frame: pd.DataFrame,
        feature: FeatureSpec,
        backtest_config: BacktestConfig,
    ) -> pd.DataFrame:
        if feature.type == "dvol":
            feature_df = self._load_or_fetch_dvol(feature, backtest_config)
            value_column = feature.column
            if (
                value_column not in feature_df.columns
                and "dvol_close" in feature_df.columns
            ):
                value_column = "dvol_close"
            merged = feature_df[["timestamp", value_column]].rename(
                columns={value_column: feature.column}
            )
        elif feature.type == "funding_rate":
            if not feature.path:
                msg = "funding_rate feature requires data.requirements.features[].path"
                raise ValueError(msg)
            feature_df = self._loader.load_parquet(Path(feature.path))
            merged = feature_df[["timestamp", feature.column]]
        else:
            if not feature.path:
                msg = "custom feature requires data.requirements.features[].path"
                raise ValueError(msg)
            feature_df = self._loader.load_parquet(Path(feature.path))
            merged = feature_df[["timestamp", feature.column]]

        result = frame.merge(merged, on="timestamp", how="left")
        result[feature.column] = result[feature.column].ffill().bfill()
        return result

    def _load_or_fetch_dvol(
        self,
        feature: FeatureSpec,
        backtest_config: BacktestConfig,
    ) -> pd.DataFrame:
        processed_path = (
            Path(feature.path)
            if feature.path
            else self._default_dvol_path(backtest_config)
        )
        if processed_path.exists():
            return self._loader.load_parquet(processed_path)

        crawler = DeribitDVOLCrawler()
        currency = backtest_config.symbol.split("/")[0]
        raw = crawler.fetch(
            symbol=currency,
            timeframe=backtest_config.timeframe,
            start=backtest_config.start_date,
            end=backtest_config.end_date,
        )
        raw_path = (
            Path(feature.raw_path)
            if feature.raw_path
            else processed_path.with_suffix(".csv")
        )
        crawler.save_raw(raw, raw_path)
        if raw["timestamp"].dt.tz is None:
            raw["timestamp"] = raw["timestamp"].dt.tz_localize("UTC")
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(processed_path, index=False)
        return raw

    def _default_dvol_path(self, backtest_config: BacktestConfig) -> Path:
        symbol = backtest_config.symbol.split("/")[0].lower()
        start_year = backtest_config.start_date.year
        end_year = backtest_config.end_date.year
        return self._workspace.processed_data / (
            f"{symbol}_dvol_{backtest_config.timeframe}_"
            f"{start_year}_{end_year}.parquet"
        )

    def _dataset_id(
        self,
        backtest_config: BacktestConfig,
        requirements: DataRequirement,
    ) -> str:
        feature_key = ",".join(
            f"{feature.type}:{feature.source}:{feature.column}:{feature.path or ''}"
            for feature in requirements.features
        )
        return (
            f"{backtest_config.symbol}:{backtest_config.timeframe}:"
            f"{backtest_config.start_date.date()}:{backtest_config.end_date.date()}:"
            f"{feature_key}"
        )
