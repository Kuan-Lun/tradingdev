"""MCP DTO tests."""

from tradingdev.mcp.schemas import SaveStrategyInput, ToolResult


def test_save_strategy_input_defaults() -> None:
    dto = SaveStrategyInput(
        strategy_id="fixture",
        code="class X: ...",
        yaml_config="strategy: {}",
    )

    assert dto.request_summary == ""


def test_tool_result_defaults() -> None:
    result = ToolResult(success=False, error="bad request")

    assert result.message == ""
    assert result.error == "bad request"
