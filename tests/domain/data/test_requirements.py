"""Data requirement schema tests."""

from tradingdev.domain.data.requirements import DataRequirement


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
