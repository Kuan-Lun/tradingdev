"""The host independently checks worker response identity and signal semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd
import pytest

from tradingdev.adapters.execution.frame_codec import encode_frame
from tradingdev.adapters.execution.generated_strategy import (
    IsolatedStrategy,
    StrategyChannel,
)
from tradingdev.adapters.execution.strategy_protocol import (
    StrategyRequest,
    StrategyResponse,
)
from tradingdev.domain.runtime import IsolatedRuntimeError, RuntimeSpec
from tradingdev.domain.strategies.execution import GeneratedProgram
from tradingdev.domain.strategies.signals import SignalContractError

if TYPE_CHECKING:
    from collections.abc import Callable

    from tradingdev.adapters.execution.container_runtime import ContainerSession


def _channel(change: Callable[[StrategyResponse], None]) -> StrategyChannel:
    class ResponseSession:
        def exchange(self, data: bytes) -> bytes:
            request = StrategyRequest.model_validate_json(data)
            response = StrategyResponse(
                request_id=request.request_id,
                operation=request.operation,
                source_hash=request.source_hash,
                class_name=request.class_name,
            )
            change(response)
            return response.model_dump_json().encode()

    return StrategyChannel(
        cast("ContainerSession", ResponseSession()),
        GeneratedProgram(
            source_hash="a" * 64,
            class_name="Generated",
            runtime=RuntimeSpec(image_id="sha256:" + "b" * 64),
        ),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_id", "b" * 32),
        ("operation", "fit"),
        ("source_hash", "b" * 64),
        ("class_name", "Another"),
    ],
)
def test_channel_rejects_a_response_from_another_operation_or_program(
    field: str, value: str
) -> None:
    channel = _channel(lambda response: setattr(response, field, value))
    with pytest.raises(IsolatedRuntimeError) as caught:
        channel.request("parameters")
    assert caught.value.code == "invalid_runtime_response"


@pytest.mark.parametrize(
    "response_kind", ["missing", "changed_market", "invalid_signal"]
)
def test_proxy_rechecks_signals_returned_by_the_worker(response_kind: str) -> None:
    frame = pd.DataFrame({"close": [1.0]})

    def respond(response: StrategyResponse) -> None:
        if response_kind != "missing":
            result = frame.assign(signal=2 if response_kind == "invalid_signal" else 1)
            if response_kind == "changed_market":
                result["close"] = 9.0
            response.frame = encode_frame(result)

    strategy = IsolatedStrategy(_channel(respond))
    if response_kind == "missing":
        with pytest.raises(IsolatedRuntimeError) as caught:
            strategy.generate_signals(frame)
        assert caught.value.code == "invalid_runtime_response"
    else:
        with pytest.raises(SignalContractError) as failure:
            strategy.generate_signals(frame)
        code = (
            "invalid_signal_values"
            if response_kind == "invalid_signal"
            else "signal_market_data_changed"
        )
        assert code in {item.code for item in failure.value.diagnostics}
    assert frame["close"].tolist() == [1.0]
