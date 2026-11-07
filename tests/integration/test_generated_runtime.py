"""The factory binds saved executions and owns generated strategy lifetimes."""

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


def test_factory_binds_saved_program_and_settings_to_a_managed_instance(
    generated_workspace: Path,
) -> None:
    factory = StrategyFactory(workspace_root=generated_workspace)
    config = _config(generated_workspace)
    execution = factory.resolve_execution(config)
    assert execution.program is not None
    assert execution.program.source_hash == config["source_hash"]
    assert execution.program.class_name == config["class_name"]
    assert execution.constructor_kwargs == {"direction": -1}
    config["parameters"] = {"direction": 1}
    frame = pd.DataFrame({"close": [1.0]})
    with (
        execution_randomness(42),
        factory.open_from_execution(config, execution, None) as strategy,
    ):
        records = list((generated_workspace / ".strategy-runtime").glob("*.json"))
        assert len(records) == 1
        assert json.loads(records[0].read_text())["spec"] == (
            execution.program.runtime.model_dump(mode="json")
        )
        strategy.fit(frame)
        assert strategy.generate_signals(frame)["signal"].tolist() == [-1]
        assert strategy.get_parameters() == {
            "direction": -1,
            "trained": 1,
            "uid": 65534,
            "seed": 42,
        }
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
