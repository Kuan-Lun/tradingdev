"""Explicit data requirement schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MarketDataSpec(BaseModel):
    """Primary market data requirement."""

    source: str = "binance_api"
    symbol: str
    timeframe: str


class FeatureSpec(BaseModel):
    """Additional feature data requirement."""

    type: Literal["dvol", "funding_rate", "custom"]
    source: str
    column: str
    path: str | None = None
    raw_path: str | None = None


class DataSourceSpec(BaseModel):
    """External data source descriptor."""

    name: str
    kind: str
    metadata: dict[str, str] = Field(default_factory=dict)


class DataRequirement(BaseModel):
    """Full data requirement section for a strategy run."""

    market: MarketDataSpec
    features: list[FeatureSpec] = []
