"""Crawler registry tests."""

from __future__ import annotations

import pytest

from tradingdev.domain.data.crawlers.base import BaseCrawler
from tradingdev.domain.data.crawlers.binance_vision import BinanceVisionCrawler
from tradingdev.domain.data.crawlers.registry import (
    available_sources,
    create_crawler,
)
from tradingdev.domain.data.schemas import DataConfig


def test_builtin_sources_are_registered() -> None:
    sources = available_sources()

    assert "binance_vision" in sources
    assert "binance_api" in sources


def test_create_crawler_builds_binance_vision_with_market_type() -> None:
    crawler = create_crawler(
        "binance_vision",
        DataConfig(source="binance_vision", market_type="spot"),
    )

    assert isinstance(crawler, BinanceVisionCrawler)
    assert isinstance(crawler, BaseCrawler)
    assert crawler._market_type == "spot"


def test_create_crawler_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="Unknown data source 'nope'"):
        create_crawler("nope", DataConfig())
