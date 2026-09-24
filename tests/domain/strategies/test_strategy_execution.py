"""Captured constructor settings must outlive changes to strategy defaults."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel, create_model

from tradingdev.domain.backtest.schemas import ParallelConfig
from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.strategies.bundled.kd_strategy.config import KDStrategyConfig
from tradingdev.domain.strategies.bundled.kd_strategy.strategy import KDStrategy
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.domain.strategies.loader import StrategyLoader

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.domain.strategies.base import BaseStrategy

_GENERATED = """\
from tradingdev.domain.strategies.base import BaseStrategy

class CapturedStrategy(BaseStrategy):
    def __init__(self, period=14, optional=None, backtest_engine=None):
        self.period = period
        self.optional = optional
        self.engine = backtest_engine

    def generate_signals(self, df):
        return df.copy()

    def get_parameters(self):
        return {"period": self.period, "optional": self.optional}
"""


@pytest.fixture
def generated(tmp_path: Path) -> tuple[StrategyLoader, dict[str, Any], Path]:
    source = tmp_path / "generated_strategies" / "captured.py"
    source.parent.mkdir()
    source.write_text(_GENERATED)
    loader = StrategyLoader(workspace_root=tmp_path)
    cfg = {
        "id": "captured",
        "class_name": "CapturedStrategy",
        "source_path": str(source),
        "parameters": {},
    }
    return loader, cfg, source


def test_bundled_capture_expands_parameters_and_fit_without_changing_declaration() -> (
    None
):
    loader = StrategyLoader()
    cfg: dict[str, Any] = {
        "id": "kd_crossover",
        "parameters": {"d_period": 5},
        "fit": {"k_period_range": [9, 14]},
    }
    original = deepcopy(cfg)

    execution = loader.resolve_execution(cfg)

    assert cfg == original
    assert execution.kind == "bundled"
    assert execution.constructor_kwargs["config"] == {
        "k_period": 14,
        "d_period": 5,
        "smooth_k": 3,
        "overbought": 80.0,
        "oversold": 20.0,
    }
    fit = execution.constructor_kwargs["fit_config"]
    assert isinstance(fit, dict)
    assert fit["k_period_range"] == [9, 14]
    assert fit["d_period_range"] == [3, 5]
    assert fit["target_metric"] == "sharpe_ratio"
    cfg["fit"]["k_period_range"].append(21)
    assert fit["k_period_range"] == [9, 14]


def test_bundled_omitted_fit_is_explicit_none_and_injections_are_separate() -> None:
    loader = StrategyLoader()
    cfg = {"id": "kd_crossover"}
    execution = loader.resolve_execution(cfg)
    engine = SignalBacktestEngine(init_cash=10_000.0)
    parallel = ParallelConfig(reserve_cores=1)

    strategy = loader.create_from_execution(cfg, execution, engine, parallel)

    assert execution.constructor_kwargs["fit_config"] is None
    assert "backtest_engine" not in execution.constructor_kwargs
    assert "parallel_config" not in execution.constructor_kwargs
    assert vars(strategy)["_fit_config"] is None
    assert vars(strategy)["_backtest_engine"] is engine
    assert vars(strategy)["_parallel_config"] is parallel


def test_capture_never_calls_strategy_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UninstantiableKD(KDStrategy):
        def __init__(self, config: KDStrategyConfig) -> None:
            raise AssertionError("Capture must not execute the strategy constructor")

    loader = StrategyLoader()
    monkeypatch.setattr(loader, "load_class", lambda _cfg: UninstantiableKD)
    monkeypatch.setattr(loader, "_bundled_config_model", lambda _cls: KDStrategyConfig)

    execution = loader.resolve_execution({"id": "kd_crossover"})

    assert execution.constructor_kwargs["config"] == KDStrategyConfig().model_dump()


def test_changed_bundled_default_cannot_change_captured_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = StrategyLoader()
    cfg = {"id": "kd_crossover"}
    execution = loader.resolve_execution(cfg)
    changed = create_model(
        "ChangedKDConfig", k_period=(int, 21), __base__=KDStrategyConfig
    )
    monkeypatch.setattr(loader, "_bundled_config_model", lambda _cls: changed)

    strategy = loader.create_from_execution(cfg, execution, engine=None)

    assert strategy.get_parameters()["k_period"] == 14


def test_new_bundled_field_cannot_silently_supply_a_worker_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = StrategyLoader()
    cfg = {"id": "kd_crossover"}
    execution = loader.resolve_execution(cfg)
    changed = create_model(
        "ExpandedKDConfig", new_field=(int, 1), __base__=KDStrategyConfig
    )
    monkeypatch.setattr(loader, "_bundled_config_model", lambda _cls: changed)

    with pytest.raises(ValueError, match="no longer match"):
        loader.create_from_execution(cfg, execution, engine=None)


def test_nested_model_defaults_are_captured_and_grid_overlays_preserve_them() -> None:
    loader = StrategyLoader()
    cfg = {"id": "xgboost_direction", "parameters": {"model": {"random_state": 7}}}
    execution = loader.resolve_execution(cfg)
    config = execution.constructor_kwargs["config"]
    assert isinstance(config, dict)
    model = config["model"]
    assert isinstance(model, dict)
    assert model["n_jobs"] is None
    assert model["n_estimators"] == 100

    strategy = loader.create_from_execution(
        cfg, execution, None, parameter_overrides={"model": {"max_depth": 4}}
    )

    actual = vars(strategy)["_config"].model.model_dump()
    assert actual == {**model, "max_depth": 4}
    assert model["max_depth"] == 6


@pytest.mark.parametrize(
    "overrides",
    [
        {"unknown": 1},
        {"model": {"unknown": 1}},
        {"model": {"max_depth": "4"}},
        {"model": {"max_depth": float("inf")}},
    ],
)
def test_grid_rejects_unknown_or_implicitly_converted_values(
    overrides: dict[str, Any],
) -> None:
    loader = StrategyLoader()
    cfg = {"id": "xgboost_direction"}
    execution = loader.resolve_execution(cfg)

    with pytest.raises(ValueError):
        loader.validate_parameter_overrides(cfg, execution, overrides)


@pytest.mark.parametrize("overbought", [70, 80, 90])
def test_integer_grid_candidates_are_valid_for_float_model_fields(
    overbought: int,
) -> None:
    loader = StrategyLoader()
    cfg = {"id": "kd_crossover"}
    execution = loader.resolve_execution(cfg)

    loader.validate_parameter_overrides(cfg, execution, {"overbought": overbought})
    strategy = loader.create_from_execution(
        cfg, execution, None, parameter_overrides={"overbought": overbought}
    )

    assert strategy.get_parameters()["overbought"] == float(overbought)


@pytest.mark.parametrize("value", [True, "80"])
def test_float_grid_rejects_boolean_and_string_coercion(value: object) -> None:
    loader = StrategyLoader()
    cfg = {"id": "kd_crossover"}
    execution = loader.resolve_execution(cfg)

    with pytest.raises(ValueError, match="no longer match"):
        loader.validate_parameter_overrides(cfg, execution, {"overbought": value})


def test_nested_float_grid_rejects_lossy_integer_conversion() -> None:
    loader = StrategyLoader()
    cfg = {"id": "xgboost_direction"}
    execution = loader.resolve_execution(cfg)

    with pytest.raises(ValueError, match="no longer match"):
        loader.validate_parameter_overrides(
            cfg, execution, {"model": {"learning_rate": 2**53 + 1}}
        )


def test_generated_defaults_and_none_are_explicit_and_survive_default_changes(
    generated: tuple[StrategyLoader, dict[str, Any], Path],
) -> None:
    loader, cfg, source = generated
    execution = loader.resolve_execution(cfg)
    source.write_text(_GENERATED.replace("period=14", "period=21"))

    strategy = loader.create_from_execution(cfg, execution, engine=None)

    assert execution.constructor_kwargs == {"period": 14, "optional": None}
    assert strategy.get_parameters() == {"period": 14, "optional": None}
    assert cfg["parameters"] == {}


def test_generated_added_constructor_default_requires_resubmission(
    generated: tuple[StrategyLoader, dict[str, Any], Path],
) -> None:
    loader, cfg, source = generated
    execution = loader.resolve_execution(cfg)
    source.write_text(_GENERATED.replace("optional=None", "optional=None, new_field=2"))

    with pytest.raises(ValueError, match="explicit constructor setting 'new_field'"):
        loader.create_from_execution(cfg, execution, engine=None)


@pytest.mark.parametrize("default", ["object()", "float('nan')", "(1, 2)"])
@pytest.mark.parametrize("validation_entry", [False, True])
def test_generated_non_json_defaults_fail_before_submission(
    generated: tuple[StrategyLoader, dict[str, Any], Path],
    default: str,
    validation_entry: bool,
) -> None:
    loader, cfg, source = generated
    source.write_text(_GENERATED.replace("optional=None", f"optional={default}"))

    with pytest.raises(ValueError, match="finite JSON"):
        if validation_entry:
            loader.create_from_config({"strategy": cfg}, None)
        else:
            loader.resolve_execution(cfg)


def test_generated_validation_and_execution_share_parallel_injection(
    generated: tuple[StrategyLoader, dict[str, Any], Path],
) -> None:
    loader, cfg, source = generated
    source.write_text(
        _GENERATED.replace(
            "backtest_engine=None):", "backtest_engine=None, parallel_config=None):"
        ).replace(
            "self.engine = backtest_engine",
            "self.engine = backtest_engine\n        self.parallel = parallel_config",
        )
    )
    raw: dict[str, Any] = {"strategy": cfg, "parallel": {"reserve_cores": 1}}
    engine = SignalBacktestEngine(init_cash=10_000.0)

    checked = loader.create_from_config(raw, engine)
    captured = loader.resolve_execution(cfg)
    executed = loader.create_from_execution(
        cfg, captured, engine, ParallelConfig.model_validate(raw["parallel"])
    )

    assert vars(checked)["parallel"] == vars(executed)["parallel"]
    assert vars(checked)["parallel"].reserve_cores == 1
    assert vars(checked)["engine"] is engine
    assert vars(executed)["engine"] is engine
    assert "parallel_config" not in captured.constructor_kwargs


def test_generated_kwargs_keep_explicit_values(
    generated: tuple[StrategyLoader, dict[str, Any], Path],
) -> None:
    loader, cfg, source = generated
    source.write_text(
        _GENERATED.replace("backtest_engine=None):", "backtest_engine=None, **extras):")
    )
    cfg["parameters"] = {"extra_value": {"limit": 3}}

    execution = loader.resolve_execution(cfg)
    strategy = loader.create_from_execution(cfg, execution, None)

    assert execution.constructor_kwargs["extra_value"] == {"limit": 3}
    assert strategy.get_parameters()["period"] == 14


def test_grid_validation_resolves_class_and_models_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = StrategyLoader()
    cfg = {"id": "kd_crossover"}
    execution = loader.resolve_execution(cfg)
    loaded: list[dict[str, Any]] = []
    models: list[type[BaseStrategy]] = []

    def load_class(value: dict[str, Any]) -> type[BaseStrategy]:
        loaded.append(value)
        return KDStrategy

    def model_type(cls: type[BaseStrategy]) -> type[BaseModel]:
        models.append(cls)
        return KDStrategyConfig

    monkeypatch.setattr(loader, "load_class", load_class)
    monkeypatch.setattr(loader, "_bundled_config_model", model_type)

    loader.validate_parameter_grid(
        cfg, execution, ({"k_period": value} for value in (9, 14, 21))
    )

    assert loaded == [cfg]
    assert models == [KDStrategy]


def test_execution_kind_mismatch_is_rejected(
    generated: tuple[StrategyLoader, dict[str, Any], Path],
) -> None:
    loader, cfg, _ = generated
    mismatched = StrategyExecution(kind="bundled", constructor_kwargs={})

    with pytest.raises(ValueError, match="kind does not match"):
        loader.create_from_execution(cfg, mismatched, None)
