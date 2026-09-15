"""Strategy loader schema contract tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.strategies.contract import SignalContractChecker
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.strategies.schemas import StrategyMetadata

_PARAMETERIZED_CODE = """\
from typing import Any

import pandas as pd

from tradingdev.domain.strategies.base import BaseStrategy


class ParameterStrategy(BaseStrategy):
    def __init__(
        self,
        threshold: float,
        backtest_engine: object | None = None,
        direction: int = 1,
    ) -> None:
        self.threshold = threshold
        self.direction = direction
        self.engine = backtest_engine

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        result.loc[result["close"] > self.threshold, "signal"] = self.direction
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "direction": self.direction}
"""


@pytest.fixture
def generated_config(tmp_path: Path) -> dict[str, Any]:
    source = tmp_path / "workspace/generated_strategies/parameter_strategy.py"
    source.parent.mkdir(parents=True)
    source.write_text(_PARAMETERIZED_CODE, encoding="utf-8")
    return {
        "strategy": {
            "id": "parameter_strategy",
            "class_name": "ParameterStrategy",
            "source_path": str(source),
            "parameters": {"threshold": 101.0, "direction": -1},
        }
    }


def _contract_metadata(tmp_path: Path, raw: dict[str, Any]) -> StrategyMetadata:
    config_path = tmp_path / "parameter_strategy.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    strategy_cfg = raw["strategy"]
    return StrategyMetadata(
        strategy_id=strategy_cfg["id"],
        class_name=strategy_cfg["class_name"],
        status="draft",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        source_path=strategy_cfg["source_path"],
        config_path=str(config_path),
        source_hash="fixture",
        config_hash="fixture",
    )


def test_loader_rejects_deprecated_strategy_name_field(tmp_path: Path) -> None:
    loader = StrategyLoader(workspace_root=tmp_path / "workspace")

    with pytest.raises(ValueError, match="strategy.id is required"):
        loader.load_class(
            {
                "name": "legacy_strategy",
                "class_name": "LegacyStrategy",
                "source_path": "workspace/generated_strategies/legacy.py",
            }
        )


def test_loader_rejects_deprecated_strategy_class_and_file_fields(
    tmp_path: Path,
) -> None:
    loader = StrategyLoader(workspace_root=tmp_path / "workspace")

    with pytest.raises(ValueError, match="strategy.class_name is required"):
        loader.load_class(
            {
                "id": "legacy_strategy",
                "class": "LegacyStrategy",
                "file": "workspace/generated_strategies/legacy.py",
            }
        )


def test_loader_creates_bundled_strategy_without_hard_coded_aliases() -> None:
    config_path = Path(
        "src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml"
    ).resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    engine = SignalBacktestEngine(init_cash=10_000.0)

    strategy = StrategyLoader().create_from_config(raw_config, engine)

    assert strategy.__class__.__name__ == "KDStrategy"


def test_loader_rejects_legacy_bundled_strategy_id() -> None:
    config_path = Path(
        "src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml"
    ).resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["strategy"]["id"] = "kd_strategy"
    engine = SignalBacktestEngine(init_cash=10_000.0)

    with pytest.raises(ValueError, match="Bundled strategy id 'kd_strategy'"):
        StrategyLoader().create_from_config(raw_config, engine)


def test_generated_parameters_match_contract_and_execution(
    tmp_path: Path,
    generated_config: dict[str, Any],
) -> None:
    loader = StrategyLoader(workspace_root=tmp_path / "workspace")
    engine = SignalBacktestEngine(init_cash=10_000.0)
    strategy = loader.create_from_config(generated_config, engine)

    assert strategy.get_parameters() == {"threshold": 101.0, "direction": -1}
    assert vars(strategy)["engine"] is engine
    metadata = _contract_metadata(tmp_path, generated_config)
    for fixture_rows in (80, 240):
        checked = SignalContractChecker(loader).check(
            metadata, fixture_rows=fixture_rows
        )
        assert checked["diagnostics"] == []
        assert checked["signal_analysis"]["active_signal_ratio"] > 0
        assert "-1" in checked["signal_analysis"]["signal_distribution"]
        assert "1" not in checked["signal_analysis"]["signal_distribution"]


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"direction": 1}, "missing a required argument: 'threshold'"),
        ({"threshold": 101, "thresholdd": 99}, "unexpected keyword argument"),
        ({"threshold": 101, "backtest_engine": None}, "cannot override"),
        ([], "strategy.parameters must be a mapping"),
    ],
)
def test_bad_parameters_fail_contract_and_execution(
    tmp_path: Path,
    generated_config: dict[str, Any],
    parameters: object,
    message: str,
) -> None:
    generated_config["strategy"]["parameters"] = parameters
    loader = StrategyLoader(workspace_root=tmp_path / "workspace")
    with pytest.raises((TypeError, ValueError), match=message):
        loader.create_from_config(generated_config, engine=None)

    checked = SignalContractChecker(loader).check(
        _contract_metadata(tmp_path, generated_config), fixture_rows=80
    )
    assert len(checked["diagnostics"]) == 1
    assert checked["diagnostics"][0].code == "contract_execution_error"
    assert message in checked["diagnostics"][0].message


def test_generated_constructor_can_omit_engine(
    tmp_path: Path,
    generated_config: dict[str, Any],
) -> None:
    source = Path(generated_config["strategy"]["source_path"])
    source.write_text(
        _PARAMETERIZED_CODE.replace(
            "        backtest_engine: object | None = None,\n", ""
        ).replace("        self.engine = backtest_engine\n", ""),
        encoding="utf-8",
    )
    loader = StrategyLoader(workspace_root=tmp_path / "workspace")

    strategy = loader.create_from_config(generated_config, engine=None)

    assert strategy.get_parameters() == {"threshold": 101.0, "direction": -1}


def test_loader_honors_workspace_environment_and_explicit_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generated_config: dict[str, Any],
) -> None:
    monkeypatch.setenv("TRADINGDEV_WORKSPACE", str(tmp_path / "workspace"))

    strategy = StrategyLoader().create_from_config(generated_config, engine=None)

    assert strategy.get_parameters()["threshold"] == 101.0
    loader = StrategyLoader(workspace_root=tmp_path / "other_workspace")
    with pytest.raises(ValueError, match="Generated strategy must live under"):
        loader.create_from_config(generated_config, engine=None)


@pytest.mark.parametrize("field", ["id", "class_name", "source_path"])
def test_contract_rejects_config_pointing_at_another_strategy(
    tmp_path: Path,
    generated_config: dict[str, Any],
    field: str,
) -> None:
    metadata = _contract_metadata(tmp_path, generated_config)
    generated_config["strategy"][field] = "unvalidated_strategy"
    Path(metadata.config_path).write_text(
        yaml.safe_dump(generated_config), encoding="utf-8"
    )

    checked = SignalContractChecker(
        StrategyLoader(workspace_root=tmp_path / "workspace")
    ).check(metadata, fixture_rows=80)

    assert len(checked["diagnostics"]) == 1
    assert checked["diagnostics"][0].message == (
        f"strategy.{field} does not match saved strategy metadata"
    )


def test_loader_reads_latest_source_after_same_size_same_timestamp_rewrite(
    tmp_path: Path,
    generated_config: dict[str, Any],
) -> None:
    loader = StrategyLoader(workspace_root=tmp_path / "workspace")
    source = Path(generated_config["strategy"]["source_path"])
    original_stat = source.stat()
    first = loader.create_from_config(generated_config, engine=None)
    assert first.get_parameters()["threshold"] == 101.0

    revised = _PARAMETERIZED_CODE.replace(
        "self.threshold = threshold", "self.threshold = 999999999"
    )
    source.write_text(revised, encoding="utf-8")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert source.stat().st_size == original_stat.st_size

    second = loader.create_from_config(generated_config, engine=None)

    assert second.get_parameters()["threshold"] == 999999999
    assert not (source.parent / "__pycache__").exists()
