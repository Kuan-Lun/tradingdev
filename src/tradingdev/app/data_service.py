"""Application service for market data access."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tradingdev.adapters.storage.filesystem import (
    WorkspacePaths,
    sha256_file,
    sha256_text,
)
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.data.crawlers.deribit_dvol import DeribitDVOLCrawler
from tradingdev.domain.data.data_manager import DataManager
from tradingdev.domain.data.loader import DataLoader
from tradingdev.domain.data.requirements import (
    DataRequirement,
    FeatureSpec,
    MarketDataSpec,
)
from tradingdev.domain.data.schemas import DataConfig
from tradingdev.shared.utils.config import load_config


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
        raw_dir, processed_dir = self._data_dirs()
        data.setdefault("raw_dir", str(raw_dir))
        data.setdefault("processed_dir", str(processed_dir))
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
                source=str(data.get("source", "binance_vision")),
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
        processed_dir = self._processed_data_dir()
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
        processed_dir = self._processed_data_dir()
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

    def inspect_dataset(self, config_path: Path | None = None) -> dict[str, Any]:
        """Return cached data plus feature-source status for an optional config."""
        raw_dir, processed_dir = self._data_dirs()
        response: dict[str, Any] = {
            "data_root": str(raw_dir.parent),
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "datasets": self.list_available_data(),
            "requirements": None,
            "market_available": None,
            "market": None,
            "dataset_fingerprint": None,
            "features": [],
        }
        if config_path is None:
            return response

        raw_config = load_config(config_path)
        backtest_config = BacktestConfig(**raw_config["backtest"])
        requirements = self.requirements(raw_config, backtest_config)
        response["requirements"] = requirements.model_dump(mode="json")
        response["market_available"] = self.data_available(
            backtest_config.symbol,
            backtest_config.timeframe,
            backtest_config.start_date.date().isoformat(),
            backtest_config.end_date.date().isoformat(),
        )
        response["market"] = self._inspect_market(backtest_config)
        response["dataset_fingerprint"] = self._dataset_fingerprint_report(
            backtest_config,
            requirements,
            response["market"],
        )
        response["features"] = [
            self._inspect_feature(feature, backtest_config)
            for feature in requirements.features
        ]
        return response

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
            feature_df = self._loader.load_parquet(
                self._resolve_data_path(feature.path)
            )
            merged = feature_df[["timestamp", feature.column]]
        else:
            if not feature.path:
                msg = "custom feature requires data.requirements.features[].path"
                raise ValueError(msg)
            feature_df = self._loader.load_parquet(
                self._resolve_data_path(feature.path)
            )
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
            self._resolve_data_path(feature.path)
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
            self._resolve_data_path(feature.raw_path)
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
        return self._processed_data_dir() / (
            f"{symbol}_dvol_{backtest_config.timeframe}_"
            f"{start_year}_{end_year}.parquet"
        )

    def _data_dirs(self) -> tuple[Path, Path]:
        data_root = os.environ.get("TRADINGDEV_DATA_ROOT")
        if data_root:
            root = Path(data_root).expanduser().resolve()
            return root / "raw", root / "processed"
        return self._workspace.raw_data, self._workspace.processed_data

    def _processed_data_dir(self) -> Path:
        return self._data_dirs()[1]

    def _market_paths(self, backtest_config: BacktestConfig) -> list[Path]:
        normalized = backtest_config.symbol.replace("/", "").lower()
        return [
            self._processed_data_dir()
            / f"{normalized}_{backtest_config.timeframe}_{year}.parquet"
            for year in range(
                backtest_config.start_date.year,
                backtest_config.end_date.year + 1,
            )
        ]

    def _resolve_data_path(self, value: str | None) -> Path:
        if not value:
            msg = "data path is required"
            raise ValueError(msg)
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()

    def _inspect_market(self, backtest_config: BacktestConfig) -> dict[str, Any]:
        paths = self._market_paths(backtest_config)
        report: dict[str, Any] = {
            "symbol": backtest_config.symbol,
            "timeframe": backtest_config.timeframe,
            "paths": [str(path) for path in paths],
            "exists": all(path.exists() for path in paths),
            "rows": 0,
            "columns": [],
            "start_timestamp": None,
            "end_timestamp": None,
            "timezone": None,
            "missing_values": {},
            "errors": [],
        }
        frames: list[pd.DataFrame] = []
        for path in paths:
            if not path.exists():
                continue
            try:
                frames.append(self._loader.load_parquet(path))
            except Exception as exc:  # noqa: BLE001
                report["errors"].append(f"{path}: {exc}")
        if not frames:
            return report

        frame = pd.concat(frames, ignore_index=True)
        report["rows"] = int(len(frame))
        report["columns"] = [str(column) for column in frame.columns]
        report["missing_values"] = {
            str(column): int(count)
            for column, count in frame.isna().sum().to_dict().items()
        }
        if "timestamp" in frame.columns and not frame.empty:
            timestamps = frame["timestamp"]
            report["start_timestamp"] = str(timestamps.min())
            report["end_timestamp"] = str(timestamps.max())
            tz = getattr(timestamps.dt, "tz", None)
            report["timezone"] = str(tz) if tz is not None else None
        return report

    def _inspect_feature(
        self,
        feature: FeatureSpec,
        backtest_config: BacktestConfig,
    ) -> dict[str, Any]:
        path = (
            self._default_dvol_path(backtest_config)
            if feature.type == "dvol" and not feature.path
            else self._resolve_data_path(feature.path) if feature.path else None
        )
        exists = bool(path and path.exists())
        missing_values: int | None = None
        rows: int | None = None
        if exists and path is not None:
            try:
                frame = self._loader.load_parquet(path)
                rows = len(frame)
                if feature.column in frame.columns:
                    missing_values = int(frame[feature.column].isna().sum())
                elif feature.type == "dvol" and "dvol_close" in frame.columns:
                    missing_values = int(frame["dvol_close"].isna().sum())
            except Exception:  # noqa: BLE001
                missing_values = None
        return {
            "type": feature.type,
            "source": feature.source,
            "column": feature.column,
            "path": str(path) if path is not None else None,
            "exists": exists,
            "rows": rows,
            "missing_values": missing_values,
        }

    def _dataset_fingerprint_report(
        self,
        backtest_config: BacktestConfig,
        requirements: DataRequirement,
        market_report: object,
    ) -> dict[str, str]:
        file_hashes = []
        if isinstance(market_report, dict):
            for value in market_report.get("paths", []):
                path = Path(str(value))
                if path.exists():
                    file_hashes.append({"path": str(path), "sha256": sha256_file(path)})
        dataset_id = self._dataset_id(backtest_config, requirements)
        payload = {
            "dataset_id": dataset_id,
            "files": file_hashes,
            "requirements": requirements.model_dump(mode="json"),
        }
        return {
            "dataset_id": dataset_id,
            "fingerprint": sha256_text(json.dumps(payload, sort_keys=True)),
        }

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
