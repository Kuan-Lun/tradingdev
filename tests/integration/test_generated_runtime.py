"""Real generated imports, instances and frames cross the Linux isolation boundary."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from tradingdev.adapters.execution.container_runtime import DockerRuntime
from tradingdev.adapters.execution.strategy_factory import StrategyFactory
from tradingdev.domain.randomness import execution_randomness
from tradingdev.domain.runtime import IsolatedRuntimeError, RuntimeLimits
from tradingdev.domain.strategies.signals import SignalContractError

if TYPE_CHECKING:
    from collections.abc import Iterator


_SOURCE = """
import os
from tradingdev.domain.randomness import get_seed
from tradingdev.domain.strategies.base import BaseStrategy
class Generated(BaseStrategy):
    def __init__(self, direction=1):
        self.direction = direction
        self.trained = 0
    def fit(self, df):
        self.trained += len(df)
    def generate_signals(self, df):
        result = df.copy()
        result['signal'] = self.direction
        return result
    def get_parameters(self):
        return {'direction': self.direction, 'trained': self.trained,
                'uid': os.getuid(), 'seed': get_seed()}
    def rolling_trace(self):
        return {'rows': self.trained}
"""


@pytest.fixture
def generated_workspace() -> Iterator[Path]:
    root = Path(mkdtemp(prefix="tradingdev-generated-runtime-test-"))
    try:
        yield root
    finally:
        controls = root / ".strategy-runtime"
        if list(controls.glob("*.json")):
            DockerRuntime(controls).recover_abandoned()
        if list(controls.glob("*.json")):
            raise AssertionError(f"Unverified generated runtime cleanup: {root}")
        shutil.rmtree(root)


def _config(root: Path, source: str = _SOURCE) -> dict[str, Any]:
    path = root / "generated_strategies/generated.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(source)
    return {
        "id": "generated",
        "class_name": "Generated",
        "source_path": str(path),
        "source_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parameters": {"direction": -1},
    }


def test_fixed_runtime_keeps_training_state_and_rng_across_data_only_calls(
    generated_workspace: Path,
) -> None:
    factory = StrategyFactory(workspace_root=generated_workspace)
    config = _config(generated_workspace)
    execution = factory.resolve_execution(config)
    assert execution.program is not None
    assert execution.program.source_hash == config["source_hash"]
    assert execution.constructor_kwargs == {"direction": -1}
    frame = pd.DataFrame({"close": [1.01, 1.02, 1.03]}, index=pd.RangeIndex(5, 8))
    with (
        execution_randomness(42),
        factory.open_from_execution(config, execution, None) as strategy,
    ):
        strategy.fit(frame)
        strategy.fit(frame.iloc[:1])
        result = strategy.generate_signals(frame)
        assert result.index.identical(frame.index)
        pd.testing.assert_series_equal(
            result["close"], frame["close"], check_exact=True
        )
        assert result["signal"].tolist() == [-1, -1, -1]
        assert strategy.get_parameters() == {
            "direction": -1,
            "trained": 4,
            "uid": 65534,
            "seed": 42,
        }
        assert strategy.rolling_trace() == {"rows": 4}
    assert not list((generated_workspace / ".strategy-runtime").iterdir())


def test_source_drift_is_rejected_before_a_new_instance_is_opened(
    generated_workspace: Path,
) -> None:
    factory = StrategyFactory(workspace_root=generated_workspace)
    config = _config(generated_workspace)
    execution = factory.resolve_execution(config)
    Path(config["source_path"]).write_text(
        _SOURCE.replace("direction=1", "direction=0")
    )
    with (
        pytest.raises(ValueError, match="source has changed"),
        factory.open_from_execution(config, execution, None),
    ):
        pytest.fail("Modified source cannot enter a fixed execution")
    assert not list((generated_workspace / ".strategy-runtime").iterdir())


def test_lazy_parameter_grid_reuses_one_unit_without_constructing_a_strategy(
    generated_workspace: Path,
) -> None:
    factory = StrategyFactory(workspace_root=generated_workspace)
    config = _config(
        generated_workspace,
        _SOURCE.replace("self.direction = direction", "raise SystemExit(7)"),
    )
    execution = factory.resolve_execution(config)
    owners: list[str] = []

    def combinations() -> Iterator[dict[str, Any]]:
        for parameters in ({"direction": 0}, {"direction": 1}, {"unknown": 1}):
            records = list((generated_workspace / ".strategy-runtime").glob("*.json"))
            assert len(records) == 1
            owners.append(json.loads(records[0].read_text())["owner"])
            yield parameters

    with pytest.raises(ValueError, match="Unknown strategy parameter: unknown"):
        factory.validate_parameter_grid(config, execution, combinations())
    assert len(owners) == 3 and len(set(owners)) == 1
    assert not list((generated_workspace / ".strategy-runtime").iterdir())


def test_input_mutation_is_detected_inside_the_worker(
    generated_workspace: Path,
) -> None:
    factory = StrategyFactory(workspace_root=generated_workspace)
    config = _config(
        generated_workspace,
        _SOURCE.replace(
            "result = df.copy()", "df['close'] = 0\n        result = df.copy()"
        ),
    )
    execution = factory.resolve_execution(config)
    frame = pd.DataFrame({"close": [1.0]})
    with (
        pytest.raises(SignalContractError) as failure,
        factory.open_from_execution(config, execution, None) as strategy,
    ):
        strategy.generate_signals(frame)
    assert "input_mutated" in {item.code for item in failure.value.diagnostics}
    assert frame["close"].tolist() == [1.0]
    assert not list((generated_workspace / ".strategy-runtime").iterdir())


@pytest.mark.parametrize("fault", ["import_exit", "constructor_exit", "signals_loop"])
def test_abnormal_generated_code_is_bounded_and_the_next_execution_still_works(
    generated_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    monkeypatch.setenv(
        "TRADINGDEV_RUNTIME_LIMITS", RuntimeLimits(max_seconds=3).model_dump_json()
    )
    source = _SOURCE
    if fault == "import_exit":
        source = "raise SystemExit(7)\n" + source
    elif fault == "constructor_exit":
        source = source.replace("self.direction = direction", "raise SystemExit(7)")
    else:
        source = source.replace(
            "result = df.copy()", "while True: pass\n        result = df.copy()"
        )
    factory = StrategyFactory(workspace_root=generated_workspace)
    config = _config(generated_workspace, source)
    with pytest.raises(IsolatedRuntimeError) as failure:
        execution = factory.resolve_execution(config)
        with factory.open_from_execution(config, execution, None) as strategy:
            strategy.generate_signals(pd.DataFrame({"close": [1.0]}))
    assert failure.value.cleanup_verified
    assert not list((generated_workspace / ".strategy-runtime").iterdir())
    config = _config(generated_workspace)
    execution = factory.resolve_execution(config)
    with factory.open_from_execution(config, execution, None) as strategy:
        assert strategy.generate_signals(pd.DataFrame({"close": [1.0]}))[
            "signal"
        ].tolist() == [-1]
    assert not list((generated_workspace / ".strategy-runtime").iterdir())
