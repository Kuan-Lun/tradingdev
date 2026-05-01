"""Data requirement schema tests."""

from tradingdev.domain.data.requirements import DataRequirement, DataSourceSpec


def test_data_source_spec_parses_metadata() -> None:
    source = DataSourceSpec(
        name="binance_funding",
        kind="derivatives",
        metadata={"endpoint": "fundingRate", "symbol": "BTCUSDT"},
    )

    assert source.name == "binance_funding"
    assert source.kind == "derivatives"
    assert source.metadata["endpoint"] == "fundingRate"


def test_data_source_spec_defaults_metadata() -> None:
    source = DataSourceSpec(name="deribit_dvol", kind="volatility_index")

    assert source.metadata == {}


def test_data_requirements_parse_features() -> None:
    requirement = DataRequirement(
        market={
            "source": "binance_api",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        },
        features=[
            {
                "type": "dvol",
                "source": "deribit",
                "column": "dvol",
                "path": "workspace/data/processed/dvol.parquet",
            }
        ],
    )

    assert requirement.market.symbol == "BTC/USDT"
    assert requirement.features[0].type == "dvol"
    assert requirement.features[0].column == "dvol"
