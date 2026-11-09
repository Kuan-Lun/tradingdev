"""Static model/config pairing contracts; negative calls must remain errors."""

from __future__ import annotations

from typing import assert_type

from pydantic import BaseModel as ConfigModel

from tradingdev.domain.ml.base import BaseModel
from tradingdev.domain.ml.models import LightGBMDirectionModel, XGBoostDirectionModel
from tradingdev.domain.ml.models.registry import ModelKind, register
from tradingdev.domain.ml.schemas import LightGBMModelConfig, XGBoostModelConfig


def model_pairs() -> None:
    kind = ModelKind(
        name="xgboost_direction",
        description="typed constructor",
        config_cls=XGBoostModelConfig,
        model_cls=XGBoostDirectionModel,
        framework="xgboost",
    )
    assert_type(kind, ModelKind[XGBoostModelConfig])
    assert_type(kind.create(n_estimators=10), BaseModel)
    ModelKind(
        name="mismatched",
        description="wrong configuration class",
        config_cls=LightGBMModelConfig,
        model_cls=XGBoostDirectionModel,  # type: ignore[arg-type]
        framework="xgboost",
    )
    ModelKind(  # type: ignore[type-var]
        name="not_a_schema",
        description="model class used as configuration",
        config_cls=LightGBMDirectionModel,
        model_cls=LightGBMDirectionModel,  # type: ignore[arg-type]
        framework="lightgbm",
    )


class CustomConfig(ConfigModel):
    trees: int = 10


def _custom_model(config: CustomConfig) -> BaseModel:
    return XGBoostDirectionModel(XGBoostModelConfig(n_estimators=config.trees))


def custom_registration() -> None:
    kind = ModelKind(
        name="custom",
        description="independently typed configuration",
        config_cls=CustomConfig,
        model_cls=_custom_model,
        framework="custom",
    )
    assert_type(kind, ModelKind[CustomConfig])
    register(kind)
