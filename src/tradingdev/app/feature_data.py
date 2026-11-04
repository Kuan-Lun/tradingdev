"""Load, inspect, and align declared feature sources with market observations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from tradingdev.adapters.storage.feature_samples import read_feature_sample
from tradingdev.app.dvol_data import DVOLData
from tradingdev.domain.data.crawlers.deribit_dvol import DeribitDVOLCrawler

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tradingdev.domain.data.loader import DataLoader
    from tradingdev.domain.data.requirements import FeatureSpec
    from tradingdev.domain.data.schemas import DataConfig, MarketDataRequest


class FeatureData:
    """Keep normal and bounded feature reads under the same source semantics."""

    def __init__(
        self, loader: DataLoader, resolve_path: Callable[[str | None], Path]
    ) -> None:
        self._loader = loader
        self._resolve_data_path = resolve_path

    def merge(
        self,
        frame: pd.DataFrame,
        feature: FeatureSpec,
        request: MarketDataRequest,
        *,
        data_config: DataConfig,
        sample_max_rows: int | None = None,
    ) -> pd.DataFrame:
        path = self._resolve_data_path(feature.path) if feature.path else None
        if sample_max_rows is not None and path is not None and path.exists():
            merged = read_feature_sample(
                path,
                timestamps=frame["timestamp"],
                column=feature.column,
                max_rows=sample_max_rows,
                fallback_column="dvol_close" if feature.type == "dvol" else None,
            )
        elif feature.type == "dvol":
            feature_df = self._load_or_fetch_dvol(
                feature,
                request,
                data_config=data_config,
                sample_max_rows=sample_max_rows,
                sample_timestamps=pd.DatetimeIndex(frame["timestamp"].unique())
                if sample_max_rows is not None
                else None,
            )
            value_column = feature.column
            if (
                value_column not in feature_df.columns
                and "dvol_close" in feature_df.columns
            ):
                value_column = "dvol_close"
            merged = feature_df[["timestamp", value_column]].rename(
                columns={value_column: feature.column}
            )
        else:
            if not feature.path:
                msg = (
                    f"{feature.type} feature requires data.requirements.features[].path"
                )
                raise ValueError(msg)
            feature_df = self._loader.load_parquet(
                self._resolve_data_path(feature.path)
            )
            merged = feature_df[["timestamp", feature.column]]

        if sample_max_rows is not None:
            merged = merged.loc[merged["timestamp"].isin(frame["timestamp"])]
            counts = merged["timestamp"].value_counts()
            expected_rows = sum(
                max(int(counts.get(timestamp, 0)), 1)
                for timestamp in frame["timestamp"]
            )
            if expected_rows > sample_max_rows:
                raise ValueError("Feature joins exceeded the sample row budget")
        result = frame.merge(merged, on="timestamp", how="left")
        # Leading gaps are unknown; later observations cannot fill earlier bars.
        result[feature.column] = result[feature.column].ffill()
        return result

    def _load_or_fetch_dvol(
        self,
        feature: FeatureSpec,
        request: MarketDataRequest,
        *,
        data_config: DataConfig,
        sample_max_rows: int | None = None,
        sample_timestamps: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        if not feature.path:
            if sample_max_rows is not None:
                raise ValueError("Sample DVOL requires an isolated output path")
            return self.managed(feature, request, data_config).load(DeribitDVOLCrawler)
        processed_path = self._resolve_data_path(feature.path)
        if processed_path.exists():
            if sample_max_rows is not None and sample_timestamps is not None:
                return read_feature_sample(
                    processed_path,
                    timestamps=pd.Series(sample_timestamps),
                    column=feature.column,
                    fallback_column="dvol_close",
                    max_rows=sample_max_rows,
                )
            return self._loader.load_parquet(processed_path)

        if feature.source != "deribit":
            raise ValueError("Automatic DVOL downloads require source='deribit'")
        crawler = DeribitDVOLCrawler()
        currency = request.symbol.split("/")[0]
        if sample_max_rows is None:
            raw = crawler.fetch(
                symbol=currency,
                timeframe=request.timeframe,
                start=request.start_date,
                end=request.end_date,
            )
        else:
            raw = crawler.fetch_sample(
                symbol=currency,
                timeframe=request.timeframe,
                start=request.start_date,
                end=request.end_date,
                max_rows=sample_max_rows,
                timestamps=sample_timestamps,
            )
            if len(raw) > sample_max_rows:
                raise ValueError("Feature provider exceeded the sample row budget")
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

    @staticmethod
    def managed(
        feature: FeatureSpec, request: MarketDataRequest, config: DataConfig
    ) -> DVOLData:
        if feature.source != "deribit":
            raise ValueError("Automatic DVOL downloads require source='deribit'")
        return DVOLData(config, request)

    def inspect(
        self,
        feature: FeatureSpec,
        request: MarketDataRequest,
        config: DataConfig,
    ) -> dict[str, Any]:
        path = self._resolve_data_path(feature.path) if feature.path else None
        query = None
        query_reusable = None
        if feature.type == "dvol" and not feature.path:
            managed = self.managed(feature, request, config)
            receipt = managed.receipt()
            query_reusable = False
            if receipt is not None:
                path = managed.cache.processed_path(receipt)
                query = receipt.model_dump(
                    mode="json", exclude={"identity", "generation", "version"}
                )
                query_reusable = receipt.reusable(request.start_date, request.end_date)
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
            "query": query,
            "query_reusable": query_reusable,
        }
