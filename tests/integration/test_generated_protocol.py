"""Generated imports and stateful calls use the real data-only container protocol."""

from __future__ import annotations

import base64
import hashlib
import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from tradingdev.adapters.execution.container_runtime import DockerRuntime
from tradingdev.adapters.execution.generated_strategy import (
    inspect_parameters,
    open_channel,
    open_strategy,
)
from tradingdev.domain.runtime import IsolatedRuntimeError, RuntimeLimits
from tradingdev.domain.strategies.execution import GeneratedProgram
from tradingdev.domain.strategies.signals import SignalContractError

if TYPE_CHECKING:
    from collections.abc import Iterator


_SOURCE = """
import os
from tradingdev.domain.randomness import get_random, get_seed
from tradingdev.domain.strategies.base import BaseStrategy
class Generated(BaseStrategy):
    def __init__(self, direction=1):
        print('constructor diagnostic')
        self.direction = direction
        self.trained = 0
        self.draws = [get_random().random()]
    def fit(self, df):
        self.trained += len(df)
        self.draws.append(get_random().random())
    def generate_signals(self, df):
        print('signal diagnostic')
        result = df.copy()
        result['signal'] = self.direction
        return result
    def get_parameters(self):
        return {'direction': self.direction, 'trained': self.trained,
                'uid': os.getuid(), 'seed': get_seed()}
    def rolling_trace(self):
        return {'rows': self.trained, 'draws': self.draws}
"""


@pytest.fixture
def runtime() -> Iterator[DockerRuntime]:
    root = Path(mkdtemp(prefix="tradingdev-generated-protocol-test-"))
    try:
        yield DockerRuntime(root / "controls")
    finally:
        if list((root / "controls").glob("*.json")):
            raise AssertionError(f"Unverified generated protocol cleanup: {root}")
        shutil.rmtree(root)


def _program(
    runtime: DockerRuntime,
    source: str = _SOURCE,
    *,
    limits: RuntimeLimits | None = None,
) -> GeneratedProgram:
    return GeneratedProgram(
        source_hash=hashlib.sha256(source.encode()).hexdigest(),
        class_name="Generated",
        runtime=runtime.resolve(limits=limits),
    )


def _assert_clean(runtime: DockerRuntime) -> None:
    assert not list(runtime.control_root.iterdir())


def test_direct_calls_keep_training_rng_and_exact_frame_types(
    runtime: DockerRuntime,
) -> None:
    import random

    program = _program(runtime)
    frame = pd.DataFrame(
        {
            "close": [np.nextafter(1.0, 2.0), 1.02, 1.03],
            "volume": pd.Series([2**62, None, 7], dtype="Int64"),
            "feature": pd.Series(["中文", None, ""], dtype="string"),
            "timestamp": pd.date_range("2024-01-01", periods=3, tz="Asia/Taipei"),
        }
    )
    frame.index = pd.RangeIndex(5, 8, name="bars")
    frame.attrs = {"source": "fixture"}
    expected_rng = random.Random(42)
    with open_strategy(
        runtime, program, _SOURCE.encode(), {"direction": -1}, 42
    ) as strategy:
        strategy.fit(frame)
        strategy.fit(frame.iloc[:1])
        result = strategy.generate_signals(frame)
        pd.testing.assert_frame_equal(result[frame.columns], frame, check_exact=True)
        assert result.index.identical(frame.index)
        assert result.attrs == frame.attrs
        assert result["signal"].tolist() == [-1, -1, -1]
        assert strategy.get_parameters() == {
            "direction": -1,
            "trained": 4,
            "uid": 65534,
            "seed": 42,
        }
        assert strategy.rolling_trace() == {
            "rows": 4,
            "draws": [expected_rng.random() for _ in range(3)],
        }
    _assert_clean(runtime)


def test_parameter_inspection_captures_and_validates_without_construction(
    runtime: DockerRuntime,
) -> None:
    source = _SOURCE.replace("self.direction = direction", "raise SystemExit(7)")
    program = _program(runtime, source)
    assert inspect_parameters(runtime, program, source.encode(), {}, capture=True) == {
        "direction": 1
    }
    assert inspect_parameters(
        runtime, program, source.encode(), {"direction": 0}, capture=False
    ) == {"direction": 0}
    with pytest.raises(IsolatedRuntimeError, match="missing explicit constructor"):
        inspect_parameters(runtime, program, source.encode(), {}, capture=False)
    _assert_clean(runtime)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (_SOURCE.replace("direction=1", "direction=object()"), "finite JSON values"),
        (
            _SOURCE.replace("direction=1", "direction=float('nan')"),
            "finite JSON values",
        ),
        (
            _SOURCE.replace("direction=1", "direction=1, backtest_engine=None"),
            "application service injections",
        ),
        (
            _SOURCE.replace("class Generated(BaseStrategy)", "class Generated"),
            "inherit from BaseStrategy",
        ),
    ],
    ids=[
        "python-default",
        "nonfinite-default",
        "service-injection",
        "wrong-base-class",
    ],
)
def test_parameter_inspection_rejects_non_data_constructor_contracts(
    runtime: DockerRuntime, source: str, message: str
) -> None:
    with pytest.raises(IsolatedRuntimeError, match=message) as caught:
        inspect_parameters(
            runtime, _program(runtime, source), source.encode(), {}, capture=True
        )
    assert caught.value.cleanup_verified
    _assert_clean(runtime)


@pytest.mark.parametrize("fault", ["hash", "identity", "source", "reconstruct"])
def test_worker_rejects_source_identity_changes_and_reconstruction(
    runtime: DockerRuntime, fault: str
) -> None:
    program = _program(runtime)
    with (
        pytest.raises(IsolatedRuntimeError) as caught,
        open_channel(runtime, program) as channel,
    ):
        source = (
            (_SOURCE + "\n# changed").encode() if fault == "hash" else _SOURCE.encode()
        )
        channel.request(
            "construct",
            source=base64.b64encode(source).decode("ascii"),
            kwargs={"direction": 1},
        )
        if fault == "identity":
            channel.program = program.model_copy(update={"class_name": "Another"})
            channel.request("parameters")
        elif fault == "source":
            channel.request(
                "parameters", source=base64.b64encode(source).decode("ascii")
            )
        else:
            channel.request("construct", kwargs={"direction": 1})
    assert caught.value.code == "strategy_execution_failed"
    assert caught.value.cleanup_verified
    assert {
        "hash": "fixed identity",
        "identity": "cannot change its strategy",
        "source": "cannot change its strategy",
        "reconstruct": "only one strategy",
    }[fault] in str(caught.value)
    _assert_clean(runtime)


def test_input_mutation_preserves_diagnostics_and_available_analysis(
    runtime: DockerRuntime,
) -> None:
    source = _SOURCE.replace(
        "result = df.copy()", "df['close'] = 0\n        result = df.copy()"
    )
    frame = pd.DataFrame({"close": [1.0]})
    with (
        pytest.raises(SignalContractError) as caught,
        open_strategy(
            runtime, _program(runtime, source), source.encode(), {"direction": 1}, None
        ) as strategy,
    ):
        strategy.generate_signals(frame)
    assert "input_mutated" in {item.code for item in caught.value.diagnostics}
    assert caught.value.analysis["rows"] == 1
    assert caught.value.analysis["signal_distribution"] == {"1": 1}
    assert frame["close"].tolist() == [1.0]
    _assert_clean(runtime)


@pytest.mark.parametrize("fault", ["import_exit", "constructor_exit", "signals_loop"])
def test_failed_or_timed_out_calls_clean_up_and_allow_the_next_session(
    runtime: DockerRuntime, fault: str
) -> None:
    if fault == "import_exit":
        source = "raise SystemExit(7)\n" + _SOURCE
    elif fault == "constructor_exit":
        source = _SOURCE.replace("self.direction = direction", "raise SystemExit(7)")
    else:
        source = _SOURCE.replace(
            "result = df.copy()", "while True: pass\n        result = df.copy()"
        )
    program = _program(runtime, source, limits=RuntimeLimits(max_seconds=3))
    with (
        pytest.raises(IsolatedRuntimeError) as caught,
        open_strategy(
            runtime, program, source.encode(), {"direction": -1}, None
        ) as strategy,
    ):
        strategy.generate_signals(pd.DataFrame({"close": [1.0]}))
    if fault == "signals_loop":
        assert caught.value.code in {"runtime_timeout", "runtime_exit"}
    else:
        assert caught.value.code == "strategy_execution_failed"
        assert "SystemExit: 7" in str(caught.value)
    assert caught.value.cleanup_verified
    _assert_clean(runtime)
    with open_strategy(
        runtime, _program(runtime), _SOURCE.encode(), {"direction": -1}, None
    ) as strategy:
        assert strategy.generate_signals(pd.DataFrame({"close": [1.0]}))[
            "signal"
        ].tolist() == [-1]
    _assert_clean(runtime)


@pytest.mark.parametrize("method", ["parameters", "trace"])
def test_proxy_rejects_non_object_parameter_and_trace_values(
    runtime: DockerRuntime, method: str
) -> None:
    source = _SOURCE.replace(
        "return {'direction': self.direction, 'trained': self.trained,\n"
        "                'uid': os.getuid(), 'seed': get_seed()}"
        if method == "parameters"
        else "return {'rows': self.trained, 'draws': self.draws}",
        "return [1, 2]",
    )
    with (
        pytest.raises(ValueError, match="JSON object"),
        open_strategy(
            runtime, _program(runtime, source), source.encode(), {"direction": 1}, None
        ) as strategy,
    ):
        if method == "parameters":
            strategy.get_parameters()
        else:
            strategy.rolling_trace()
    _assert_clean(runtime)
