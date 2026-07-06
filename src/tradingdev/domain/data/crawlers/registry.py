"""Registry mapping data source names to crawler factories.

Adding a new market data source only requires implementing BaseCrawler and
registering a factory here (or calling register_crawler from the new module);
no strategy-layer or backtest-pipeline change is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from tradingdev.domain.data.crawlers.base import BaseCrawler
    from tradingdev.domain.data.schemas import DataConfig

_REGISTRY: dict[str, Callable[[DataConfig], BaseCrawler]] = {}


def register_crawler(
    name: str,
    factory: Callable[[DataConfig], BaseCrawler],
) -> None:
    """Register a crawler factory under a data source name."""
    _REGISTRY[name] = factory


def create_crawler(source: str, data_config: DataConfig) -> BaseCrawler:
    """Create the crawler registered for a data source name."""
    factory = _REGISTRY.get(source)
    if factory is None:
        known = ", ".join(available_sources())
        msg = f"Unknown data source {source!r}. Known sources: {known}"
        raise ValueError(msg)
    return factory(data_config)


def available_sources() -> list[str]:
    """Return the registered data source names."""
    return sorted(_REGISTRY)


def _binance_vision_factory(data_config: DataConfig) -> BaseCrawler:
    from tradingdev.domain.data.crawlers.binance_vision import BinanceVisionCrawler

    return BinanceVisionCrawler(market_type=data_config.market_type)


def _binance_api_factory(_data_config: DataConfig) -> BaseCrawler:
    from tradingdev.domain.data.crawlers.binance_api import BinanceAPICrawler

    return BinanceAPICrawler()


def _yahoo_finance_factory(_data_config: DataConfig) -> BaseCrawler:
    from tradingdev.domain.data.crawlers.yahoo_finance import YahooFinanceCrawler

    return YahooFinanceCrawler()


register_crawler("binance_vision", _binance_vision_factory)
register_crawler("binance_api", _binance_api_factory)
register_crawler("yahoo_finance", _yahoo_finance_factory)
