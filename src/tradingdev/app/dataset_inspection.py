"""Describe cached market observations and their content fingerprints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from tradingdev.adapters.storage.filesystem import sha256_file, sha256_text

if TYPE_CHECKING:
    from tradingdev.adapters.storage.data_cache import DataCache
    from tradingdev.domain.data.loader import DataLoader
    from tradingdev.domain.data.requirements import DataRequirement
    from tradingdev.domain.data.schemas import MarketDataRequest


class DatasetInspection:
    """Read cache contents without fetching, publishing, or changing query identity."""

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def market(self, request: MarketDataRequest, cache: DataCache) -> dict[str, Any]:
        paths: list[Path] = []
        errors: list[str] = []
        queries: list[dict[str, Any]] = []
        for year in range(request.start_date.year, request.end_date.year + 1):
            try:
                receipt = cache.receipt(str(year))
                if receipt is not None:
                    paths.append(cache.processed_path(receipt))
                    queries.append(
                        receipt.model_dump(
                            mode="json", exclude={"identity", "generation", "version"}
                        )
                    )
            except (OSError, ValueError) as error:
                errors.append(str(error))
        report: dict[str, Any] = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "source": cache.identity.source,
            "market_type": cache.identity.market_type,
            "queries": queries,
            "paths": [str(path) for path in paths],
            "exists": len(paths) == request.end_date.year - request.start_date.year + 1,
            "rows": None,
            "columns": [],
            "start_timestamp": None,
            "end_timestamp": None,
            "timezone": None,
            "missing_values": {},
            "errors": errors,
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

    @staticmethod
    def fingerprint(
        requirements: DataRequirement,
        market_report: object,
        dataset_id: str,
    ) -> dict[str, str]:
        file_hashes = []
        if isinstance(market_report, dict):
            for value in market_report.get("paths", []):
                path = Path(str(value))
                if path.exists():
                    file_hashes.append({"path": str(path), "sha256": sha256_file(path)})
        payload = {
            "dataset_id": dataset_id,
            "files": file_hashes,
            "requirements": requirements.model_dump(mode="json"),
        }
        return {
            "dataset_id": dataset_id,
            "fingerprint": sha256_text(json.dumps(payload, sort_keys=True)),
        }
