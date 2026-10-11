"""MCP DTO tests."""

from tradingdev.mcp.schemas import SaveStrategyInput


def test_save_strategy_input_defaults() -> None:
    dto = SaveStrategyInput(
        strategy_id="fixture",
        code="class X: ...",
        yaml_config="strategy: {}",
    )

    assert dto.request_summary == ""
